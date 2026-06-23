"""ROS 2 node shared by the MPC and MPCC executables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rclpy
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry, VehicleRatesSetpoint
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String

from drone_cnt.cfg import declare_controller_config
from drone_cnt.cnt import Ctbr, CtbrControllerBase, MpcCtbrController, MpccCtbrController
from drone_cnt.utils import ned_to_enu, yaw_ned_to_enu


ControllerMode = Literal["mpc", "mpcc"]


@dataclass(frozen=True)
class VehicleState:
    """Controller state expressed in ENU coordinates."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class DroneCntNode(Node):
    """Convert PX4 state and a path reference into internal CTBR commands."""

    def __init__(self, mode: ControllerMode) -> None:
        """Create a controller node for ``mpc`` or ``mpcc``."""
        if mode not in ("mpc", "mpcc"):
            raise ValueError(f"Unsupported controller mode: {mode}")
        super().__init__(f"drone_cnt_{mode}")
        self._mode = mode
        self.config = declare_controller_config(self, include_mpcc=mode == "mpcc")
        controller_type = MpcCtbrController if mode == "mpc" else MpccCtbrController
        self.controller: CtbrControllerBase = controller_type(self.config)

        self._declare_node_parameters(mode)
        self._active_state = str(self.get_parameter("active_state").value)
        self._require_odometry = bool(self.get_parameter("require_odometry").value)
        self._state_timeout = float(self.get_parameter("state_timeout").value)
        self._publish_output = bool(self.get_parameter("publish_output").value)
        self._log_solver = bool(self.get_parameter("log_solver").value)

        self._vehicle_state: VehicleState | None = None
        self._local_position_rx_ns: int | None = None
        self._odometry_rx_ns: int | None = None
        self._fsm_state: str | None = None
        self._yaw_command: float | None = None
        self._reference: np.ndarray | None = None
        self._was_active = False
        self._waiting_reason: str | None = None

        rates_topic = self._parameter("rates_setpoint_topic")
        self._rates_publisher = self.create_publisher(VehicleRatesSetpoint, rates_topic, 10)
        self.create_subscription(
            VehicleLocalPosition,
            self._parameter("local_position_topic"),
            self._on_local_position,
            _sensor_qos(),
        )
        if self._require_odometry:
            self.create_subscription(
                VehicleOdometry,
                self._parameter("odometry_topic"),
                self._on_odometry,
                _sensor_qos(),
            )
        self.create_subscription(
            String,
            self._parameter("fsm_state_topic"),
            self._on_fsm_state,
            _latched_qos(),
        )
        self.create_subscription(
            Float32,
            self._parameter("yaw_command_topic"),
            self._on_yaw_command,
            10,
        )
        reference_qos = 10 if mode == "mpc" else _latched_qos()
        self.create_subscription(
            NavPath,
            self._parameter("reference_topic"),
            self._on_reference,
            reference_qos,
        )
        self.create_timer(self.config.control_dt, self._on_control_timer)
        self.get_logger().info(
            f"{mode.upper()} started: horizon={self.config.horizon}, "
            f"control_dt={self.config.control_dt:.3f}, output={rates_topic}"
        )

    def _declare_node_parameters(self, mode: ControllerMode) -> None:
        defaults = {
            "local_position_topic": "/fmu/out/vehicle_local_position",
            "odometry_topic": "/fmu/out/vehicle_odometry",
            "fsm_state_topic": "/fsm/state",
            "yaw_command_topic": "/planning/yaw_cmd_enu",
            "rates_setpoint_topic": "/drone_cnt/vehicle_rates_setpoint",
            "reference_topic": (
                "/tracking/ref_traj_path" if mode == "mpc" else "/tracking/path"
            ),
            "active_state": "tracking",
            "require_odometry": False,
            "state_timeout": 0.25,
            "publish_output": True,
            "log_solver": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        validity_fields = ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid")
        if not all(bool(getattr(message, field)) for field in validity_fields):
            return
        values = np.array(
            [
                message.x,
                message.y,
                message.z,
                message.vx,
                message.vy,
                message.vz,
                message.ax,
                message.ay,
                message.az,
                message.heading,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return
        self._vehicle_state = VehicleState(
            position=ned_to_enu(values[0:3]),
            velocity=ned_to_enu(values[3:6]),
            acceleration=ned_to_enu(values[6:9]),
            yaw=yaw_ned_to_enu(values[9]),
        )
        self._local_position_rx_ns = self.get_clock().now().nanoseconds

    def _on_odometry(self, _: VehicleOdometry) -> None:
        self._odometry_rx_ns = self.get_clock().now().nanoseconds

    def _on_yaw_command(self, message: Float32) -> None:
        if np.isfinite(message.data):
            self._yaw_command = float(message.data)

    def _on_reference(self, message: NavPath) -> None:
        points = np.asarray(
            [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in message.poses
            ],
            dtype=float,
        ).reshape(-1, 3)
        required = self.config.horizon + 1 if self._mode == "mpc" else 2
        if points.shape[0] < required or not np.all(np.isfinite(points)):
            self.get_logger().warning(
                f"Rejected {self._mode.upper()} reference: need {required} finite points"
            )
            return
        self._reference = points[:required] if self._mode == "mpc" else points

    def _on_fsm_state(self, message: String) -> None:
        state = message.data.strip()
        if not state or state == self._fsm_state:
            return
        previous = self._fsm_state
        self._fsm_state = state
        active = state == self._active_state
        if active != self._was_active:
            self.controller.reset()
            self._waiting_reason = None
        self._was_active = active
        self.get_logger().info(f"FSM state: {previous} -> {state}")

    def _on_control_timer(self) -> None:
        if self._fsm_state != self._active_state:
            return
        reason = self._not_ready_reason()
        if reason:
            self._warn_once(reason)
            return
        assert self._vehicle_state is not None and self._reference is not None
        yaw_command = (
            self._vehicle_state.yaw
            if self._yaw_command is None
            else self._yaw_command
        )
        try:
            command = self._compute_command(self._vehicle_state, yaw_command)
        except Exception as error:  # Keep solver faults inside the ROS boundary.
            self._warn_once(f"Controller step failed: {error}")
            self.controller.reset()
            return
        if not np.all(np.isfinite(command)):
            self._warn_once("Controller returned non-finite CTBR")
            self.controller.reset()
            return
        self._waiting_reason = None
        if self._publish_output:
            self._publish_command(command)

    def _compute_command(self, state: VehicleState, yaw_command: float) -> Ctbr:
        reference = (
            {"ref_traj_enu": self._reference.T}
            if self._mode == "mpc"
            else {"path_points_enu": self._reference}
        )
        return self.controller.step(
            state.position,
            state.velocity,
            state.acceleration,
            state.yaw,
            yaw_cmd_enu=yaw_command,
            log_solver=self._log_solver,
            **reference,
        )

    def _not_ready_reason(self) -> str | None:
        now = self.get_clock().now().nanoseconds
        timeout = int(self._state_timeout * 1.0e9)
        if self._local_position_rx_ns is None:
            return "Waiting for valid VehicleLocalPosition"
        if now - self._local_position_rx_ns > timeout:
            return "VehicleLocalPosition is stale"
        if self._require_odometry:
            if self._odometry_rx_ns is None:
                return "Waiting for VehicleOdometry"
            if now - self._odometry_rx_ns > timeout:
                return "VehicleOdometry is stale"
        if self._reference is None:
            return "Waiting for controller reference"
        return None

    def _publish_command(self, command: Ctbr) -> None:
        roll, pitch, yaw, thrust = command
        message = VehicleRatesSetpoint()
        message.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        message.roll = roll
        message.pitch = pitch
        message.yaw = yaw
        message.thrust_body = [0.0, 0.0, -thrust]
        message.reset_integral = False
        self._rates_publisher.publish(message)

    def _warn_once(self, reason: str) -> None:
        if reason != self._waiting_reason:
            self._waiting_reason = reason
            self.get_logger().warning(reason)


def _run(mode: ControllerMode) -> None:
    rclpy.init()
    node = None
    try:
        node = DroneCntNode(mode)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_mpc() -> None:
    """Run the trajectory-tracking MPC node."""
    _run("mpc")


def main_mpcc() -> None:
    """Run the contour-following MPCC node."""
    _run("mpcc")
