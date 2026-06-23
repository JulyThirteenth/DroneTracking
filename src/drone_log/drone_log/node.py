"""ROS node that records MPC and MPCC controller flights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition, VehicleRatesSetpoint
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Float32, String

from drone_log.recorder import FlightLogger, TickData


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


@dataclass(frozen=True)
class _VehicleState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    yaw: float | None


def _enu(values: tuple[float, float, float]) -> tuple[float, float, float]:
    north, east, down = values
    return east, north, -down


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _default_reference_topic(mode: str) -> str:
    if mode == "mpc":
        return "/tracking/ref_traj_path"
    if mode == "mpcc":
        return "/tracking/path"
    raise ValueError("controller_mode must be 'mpc' or 'mpcc'")


class FlightLogNode(Node):
    """Records final CTBR commands with their latest observed inputs."""

    def __init__(self) -> None:
        super().__init__("drone_log")
        mode = str(self.declare_parameter("controller_mode", "mpc").value)
        self._mode = mode
        reference_default = _default_reference_topic(mode)
        self._reference_topic = str(
            self.declare_parameter("reference_topic", reference_default).value
        )
        self._state: _VehicleState | None = None
        self._state_name = "unknown"
        self._yaw_command: float | None = None
        self._reference_position: tuple[float, float, float] | None = None
        log_directory = Path(
            str(self.declare_parameter("log_directory", "logs").value)
        ).expanduser()
        position_topic = str(
            self.declare_parameter(
                "local_position_topic", "/fmu/out/vehicle_local_position"
            ).value
        )
        output_topic = str(
            self.declare_parameter(
                "rates_setpoint_topic", "/drone_cnt/vehicle_rates_setpoint"
            ).value
        )
        self._log = FlightLogger(
            log_directory,
            run_name=str(self.declare_parameter("run_name", "").value),
            flush_interval=int(
                self.declare_parameter("flush_interval", 20).value
            ),
            metadata={
                "controller_mode": mode,
                "reference_topic": self._reference_topic,
                "position_topic": position_topic,
                "output_topic": output_topic,
            },
        )
        self.create_subscription(
            VehicleLocalPosition,
            position_topic,
            self._on_position,
            _sensor_qos(),
        )
        self.create_subscription(
            String,
            str(self.declare_parameter("fsm_state_topic", "/fsm/state").value),
            self._on_state,
            _latched_qos(),
        )
        self.create_subscription(
            Float32,
            str(
                self.declare_parameter(
                    "yaw_command_topic", "/planning/yaw_cmd_enu"
                ).value
            ),
            self._on_yaw_command,
            10,
        )
        self.create_subscription(
            NavPath, self._reference_topic, self._on_reference, 10
        )
        self.create_subscription(
            VehicleRatesSetpoint,
            output_topic,
            self._on_command,
            _sensor_qos(),
        )
        self.get_logger().info(
            f"Logging {mode.upper()} flight to {self._log.run_directory}"
        )

    def _timestamp_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _on_position(self, message: VehicleLocalPosition) -> None:
        values = (
            message.x,
            message.y,
            message.z,
            message.vx,
            message.vy,
            message.vz,
        )
        if not _finite(values):
            return
        acceleration_ned = (message.ax, message.ay, message.az)
        acceleration = (
            _enu(acceleration_ned)
            if _finite(acceleration_ned)
            else (math.nan,) * 3
        )
        yaw = (
            math.atan2(
                math.cos(math.pi / 2.0 - message.heading),
                math.sin(math.pi / 2.0 - message.heading),
            )
            if math.isfinite(message.heading)
            else None
        )
        self._state = _VehicleState(
            position=_enu((message.x, message.y, message.z)),
            velocity=_enu((message.vx, message.vy, message.vz)),
            acceleration=acceleration,
            yaw=yaw,
        )

    def _on_state(self, message: String) -> None:
        state = message.data.strip() or "unknown"
        if state != self._state_name:
            self._state_name = state
            self._log.log_event(self._timestamp_ns(), state, "state_changed")

    def _on_yaw_command(self, message: Float32) -> None:
        if math.isfinite(message.data):
            self._yaw_command = float(message.data)

    def _on_reference(self, message: NavPath) -> None:
        points = [
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            )
            for pose in message.poses
        ]
        if not points or not all(_finite(point) for point in points):
            return
        self._log.log_reference(self._timestamp_ns(), points)
        self._reference_position = points[0] if self._mode == "mpc" else None

    def _on_command(self, message: VehicleRatesSetpoint) -> None:
        state = self._state
        thrust = (
            -float(message.thrust_body[2])
            if len(message.thrust_body) >= 3
            else None
        )
        self._log.log_tick(
            self._timestamp_ns(),
            TickData(
                state=self._state_name,
                position=None if state is None else state.position,
                velocity=None if state is None else state.velocity,
                acceleration=None if state is None else state.acceleration,
                yaw=None if state is None else state.yaw,
                reference_position=self._reference_position,
                yaw_command=self._yaw_command,
                roll_rate_command=float(message.roll),
                pitch_rate_command=float(message.pitch),
                yaw_rate_command=float(message.yaw),
                thrust_command=thrust,
            ),
        )

    def destroy_node(self) -> bool:
        self._log.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Run the flight-log node."""
    rclpy.init(args=args)
    node = FlightLogNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
