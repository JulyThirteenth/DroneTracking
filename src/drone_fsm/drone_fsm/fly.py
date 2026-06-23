"""Translate mission states into one unambiguous PX4 output stream."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from px4_msgs.msg import (
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from drone_fsm.model import (
    STATES,
    STATE_HOVER,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_READY,
    STATE_RETURN_HOVER,
    STATE_TRACKING,
)
from drone_fsm.px4 import Px4Bridge
from drone_fsm.qos import best_effort_qos, latched_qos


def rates_setpoint_is_finite(message: VehicleRatesSetpoint) -> bool:
    """Return whether every CTBR field required by PX4 is finite."""
    try:
        return all(
            math.isfinite(float(value))
            for value in (
                message.roll,
                message.pitch,
                message.yaw,
                message.thrust_body[0],
                message.thrust_body[1],
                message.thrust_body[2],
            )
        )
    except (IndexError, TypeError, ValueError):
        return False


def timestamp_is_fresh(
    received_ns: int | None,
    now_ns: int,
    timeout_s: float,
) -> bool:
    """Return whether a monotonic receive timestamp is inside its timeout."""
    if received_ns is None:
        return False
    age_ns = int(now_ns) - int(received_ns)
    return 0 <= age_ns <= int(float(timeout_s) * 1.0e9)


def local_position_invalid_fields(
    message: VehicleLocalPosition,
) -> tuple[str, ...]:
    """Return failed PX4 local-position checks for precise diagnostics."""
    invalid = [
        name
        for name in (
            "xy_valid",
            "z_valid",
            "v_xy_valid",
            "v_z_valid",
        )
        if not bool(getattr(message, name))
    ]
    values = (
        message.x,
        message.y,
        message.z,
        message.vx,
        message.vy,
        message.vz,
    )
    if not all(math.isfinite(float(value)) for value in values):
        invalid.append("nonfinite_state")
    return tuple(invalid)


class DroneFlyNode(Node):
    """Own the single PX4 writer for takeoff, hold, and CTBR forwarding."""

    def __init__(self) -> None:
        """Initialize state inputs and the PX4 output bridge."""
        super().__init__("drone_fly")

        self._control_rate = float(self._param("control_rate", 50.0))
        self._takeoff_height = float(self._param("takeoff_height", 1.0))
        self._prestream_time = float(self._param("offboard_prestream_time", 1.0))
        self._command_retry_period = float(self._param("command_retry_period", 1.0))
        self._controller_timeout = float(self._param("controller_timeout", 0.30))
        self._vehicle_status_timeout = float(
            self._param("vehicle_status_timeout", 1.50)
        )
        self._local_position_timeout = float(
            self._param("local_position_timeout", 0.50)
        )
        self._send_arm_command = bool(self._param("send_arm_command", True))
        self._send_offboard_command = bool(self._param("send_offboard_command", True))
        self._target_system = int(self._param("target_system", 1))
        self._state_topic = str(self._param("state_topic", "/fsm/state"))
        self._local_position_topic = str(
            self._param(
                "local_position_topic",
                "/fmu/out/vehicle_local_position",
            )
        )
        self._vehicle_status_topic = str(
            self._param("vehicle_status_topic", "/fmu/out/vehicle_status_v1")
        )
        self._command_ack_topic = str(
            self._param(
                "command_ack_topic",
                "/fmu/out/vehicle_command_ack",
            )
        )
        self._controller_topic = str(
            self._param(
                "controller_topic",
                "/drone_cnt/vehicle_rates_setpoint",
            )
        )
        self._validate_parameters()

        self._fsm_state = STATE_PREFLIGHT
        self._position_ned: np.ndarray | None = None
        self._yaw_ned: float | None = None
        self._local_position_invalid_fields: tuple[str, ...] = ()
        self._local_position_rx_ns: int | None = None
        self._vehicle_status: VehicleStatus | None = None
        self._vehicle_status_rx_ns: int | None = None
        self._ref_alt_amsl: float | None = None

        self._hold_target_ned: np.ndarray | None = None
        self._hold_yaw_ned: float | None = None
        self._return_target_ned: np.ndarray | None = None
        self._return_yaw_ned: float | None = None

        self._controller_setpoint: VehicleRatesSetpoint | None = None
        self._controller_rx_ns: int | None = None
        self._rate_control_active = False
        self._tracking_prestream_start_ns: int | None = None

        self._takeoff_accepted = False
        self._takeoff_failed = False
        self._last_takeoff_command_ns: int | None = None
        self._last_arm_command_ns: int | None = None
        self._last_offboard_command_ns: int | None = None
        self._landing_active = False
        self._last_land_command_ns: int | None = None
        self._last_warning: str | None = None

        self._px4 = Px4Bridge(self, target_system=self._target_system)
        self.create_subscription(
            String,
            self._state_topic,
            self._on_state,
            latched_qos(),
        )
        self.create_subscription(
            VehicleLocalPosition,
            self._local_position_topic,
            self._on_local_position,
            best_effort_qos(depth=5),
        )
        self.create_subscription(
            VehicleStatus,
            self._vehicle_status_topic,
            self._on_vehicle_status,
            best_effort_qos(depth=5),
        )
        self.create_subscription(
            VehicleCommandAck,
            self._command_ack_topic,
            self._on_command_ack,
            best_effort_qos(depth=10),
        )
        self.create_subscription(
            VehicleRatesSetpoint,
            self._controller_topic,
            self._on_controller_setpoint,
            10,
        )
        self.create_timer(1.0 / self._control_rate, self._on_timer)

        self.get_logger().info(
            "Fly started: state=%s, controller=%s"
            % (
                self._state_topic,
                self._controller_topic,
            )
        )

    def _param(self, name: str, default):
        return self.declare_parameter(name, default).value

    def _validate_parameters(self) -> None:
        positive = {
            "control_rate": self._control_rate,
            "takeoff_height": self._takeoff_height,
            "command_retry_period": self._command_retry_period,
            "controller_timeout": self._controller_timeout,
            "vehicle_status_timeout": self._vehicle_status_timeout,
            "local_position_timeout": self._local_position_timeout,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self._prestream_time < 0.0:
            raise ValueError("offboard_prestream_time cannot be negative")

    # ROS inputs

    def _on_local_position(self, message: VehicleLocalPosition) -> None:
        self._local_position_rx_ns = self._now_ns()
        self._local_position_invalid_fields = local_position_invalid_fields(message)
        if self._local_position_invalid_fields:
            return

        self._position_ned = np.array(
            [message.x, message.y, message.z],
            dtype=float,
        )
        self._yaw_ned = (
            float(message.heading) if message.heading_good_for_control else None
        )
        self._ref_alt_amsl = (
            float(message.ref_alt)
            if message.z_global and math.isfinite(float(message.ref_alt))
            else None
        )

    def _on_vehicle_status(self, message: VehicleStatus) -> None:
        self._vehicle_status = message
        self._vehicle_status_rx_ns = self._now_ns()

    def _on_command_ack(self, message: VehicleCommandAck) -> None:
        if self._fsm_state != STATE_HOVER_START:
            return

        command = int(message.command)
        result = int(message.result)
        accepted = result in {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS,
        }
        failed = result in {
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED,
        }

        if command == VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF:
            if accepted:
                self._takeoff_accepted = True
                self.get_logger().info("PX4 accepted NAV_TAKEOFF")
            elif failed:
                self._takeoff_failed = True
                self.get_logger().error(f"PX4 rejected NAV_TAKEOFF: result={result}")
        elif command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and failed:
            self._takeoff_failed = True
            self.get_logger().error(f"PX4 rejected Arm: result={result}")

    def _on_controller_setpoint(self, message: VehicleRatesSetpoint) -> None:
        if not rates_setpoint_is_finite(message):
            self._controller_setpoint = None
            self._controller_rx_ns = None
            self._warn_once("Rejected non-finite CTBR")
            return
        self._controller_setpoint = message
        self._controller_rx_ns = self._now_ns()

    def _on_state(self, message: String) -> None:
        new_state = message.data.strip()
        if not new_state or new_state == self._fsm_state:
            return
        if new_state not in STATES:
            self._warn_once(f"Rejected unknown FSM state: {new_state}")
            return

        old_state = self._fsm_state
        self._fsm_state = new_state
        self._last_warning = None
        self.get_logger().info(f"FSM state: {old_state} -> {new_state}")

        if old_state == STATE_TRACKING:
            self._rate_control_active = False
        if new_state == STATE_READY:
            self._begin_ready()
        elif new_state == STATE_HOVER_START:
            self._begin_takeoff()
        elif new_state == STATE_HOVER:
            self._capture_hold()
        elif new_state == STATE_RETURN_HOVER:
            self._begin_return()
        elif new_state == STATE_TRACKING:
            self._begin_tracking()
        elif new_state == STATE_PREFLIGHT:
            self._begin_landing()

    # State output dispatch

    def _on_timer(self) -> None:
        if self._landing_active:
            self._handle_landing()
        elif self._fsm_state == STATE_READY:
            self._handle_ready()
        elif self._fsm_state == STATE_HOVER_START:
            self._handle_takeoff()
        elif self._fsm_state == STATE_HOVER:
            self._publish_hold()
        elif self._fsm_state == STATE_TRACKING:
            self._handle_tracking()
        elif self._fsm_state == STATE_RETURN_HOVER:
            self._handle_return()

    def _handle_ready(self) -> None:
        failure = self._takeoff_failure()
        if failure:
            self._warn_once(failure)
            return
        if self._hold_target_ned is None and not self._capture_hold():
            return
        self._publish_hold()

    def _handle_takeoff(self) -> None:
        if self._takeoff_failed:
            self._warn_once("Takeoff rejected; send prepare and retry after fixing PX4")
            return

        failure = self._takeoff_failure()
        if failure:
            self._warn_once(failure)
            return
        if self._hold_target_ned is None and not self._capture_hold():
            return

        if not self._is_armed():
            if not self._send_arm_command:
                self._warn_once("Waiting for external Arm command")
            elif self._retry_due(self._last_arm_command_ns):
                self._px4.send_arm()
                self._last_arm_command_ns = self._now_ns()
                self.get_logger().info("PX4 Arm command published")
            return

        if not self._takeoff_accepted and self._retry_due(
            self._last_takeoff_command_ns
        ):
            self._px4.send_takeoff(
                altitude_amsl=self._requested_takeoff_altitude_amsl()
            )
            self._last_takeoff_command_ns = self._now_ns()
            self.get_logger().info("PX4 NAV_TAKEOFF published")
            return

        self._last_warning = None

    def _handle_tracking(self) -> None:
        if not self._is_armed():
            self._warn_once("Tracking blocked while vehicle is disarmed")
            return
        if self._is_native_takeoff_active():
            self._warn_once("Waiting for PX4 native takeoff to finish")
            return

        command = self._fresh_controller_setpoint()
        if command is None:
            self._tracking_prestream_start_ns = None
            if self._rate_control_active:
                self._capture_hold()
                self.get_logger().error(
                    "CTBR unavailable; switched to current-position hold"
                )
            self._rate_control_active = False
            self._warn_once("Waiting for a fresh CTBR command; holding position")
            self._publish_hold()
            return

        self._px4.publish_body_rate_mode()
        self._px4.publish_rates_setpoint(command)

        if self._is_offboard():
            if not self._rate_control_active:
                self.get_logger().info("PX4 Offboard confirmed; CTBR active")
            self._rate_control_active = True
            self._last_warning = None
            return

        if self._tracking_prestream_start_ns is None:
            self._tracking_prestream_start_ns = self._now_ns()
            self.get_logger().info("CTBR Offboard prestream started")
            return
        if self._elapsed(self._tracking_prestream_start_ns) < self._prestream_time:
            return
        if not self._send_offboard_command:
            self._warn_once("Waiting for external Offboard command")
            return
        if self._retry_due(self._last_offboard_command_ns):
            self._px4.send_offboard()
            self._last_offboard_command_ns = self._now_ns()
            self.get_logger().info("PX4 Offboard command published")

    def _handle_return(self) -> None:
        if self._hold_target_ned is None:
            self._warn_once("Return target is unavailable")
            return
        self._publish_hold()

    # State entry actions

    def _begin_ready(self) -> None:
        if self._is_disarmed():
            self._hold_target_ned = None
            self._hold_yaw_ned = None
            self._return_target_ned = None
            self._return_yaw_ned = None
        self._reset_takeoff()
        self.get_logger().info("READY: waiting for PX4 health and local position")

    def _begin_takeoff(self) -> None:
        self._reset_takeoff()
        self.get_logger().info("Native PX4 takeoff requested")

    def _reset_takeoff(self) -> None:
        self._takeoff_accepted = False
        self._takeoff_failed = False
        self._last_takeoff_command_ns = None
        self._last_arm_command_ns = None

    def _begin_tracking(self) -> None:
        self._capture_hold()
        if self._hold_target_ned is not None:
            self._return_target_ned = self._hold_target_ned.copy()
            self._return_yaw_ned = self._hold_yaw_ned
        self._controller_setpoint = None
        self._controller_rx_ns = None
        self._rate_control_active = False
        self._tracking_prestream_start_ns = None
        self._last_offboard_command_ns = None

    def _begin_return(self) -> None:
        self._hold_target_ned = (
            None if self._return_target_ned is None else self._return_target_ned.copy()
        )
        self._hold_yaw_ned = self._return_yaw_ned

    def _begin_landing(self) -> None:
        self._rate_control_active = False
        self._landing_active = True
        self._last_land_command_ns = None

    # PX4 outputs and captured targets

    def _handle_landing(self) -> None:
        if self._is_disarmed():
            self._landing_active = False
            self.get_logger().info("Landing completed")
            return
        if self._retry_due(self._last_land_command_ns):
            self._px4.send_land()
            self._last_land_command_ns = self._now_ns()
            self.get_logger().info("PX4 NAV_LAND published")

    def _publish_hold(self) -> None:
        if self._hold_target_ned is None and not self._capture_hold():
            return
        assert self._hold_target_ned is not None
        self._publish_position(self._hold_target_ned)

    def _publish_position(self, target: np.ndarray) -> None:
        yaw = math.nan if self._hold_yaw_ned is None else float(self._hold_yaw_ned)
        self._px4.publish_position_mode()
        self._px4.publish_position_setpoint(
            tuple(float(value) for value in target),
            yaw,
        )

    def _capture_hold(self) -> bool:
        if self._position_ned is None:
            return False
        self._hold_target_ned = self._position_ned.copy()
        self._hold_yaw_ned = self._yaw_ned
        return True

    # Small state predicates

    def _takeoff_failure(self) -> str | None:
        if not timestamp_is_fresh(
            self._vehicle_status_rx_ns,
            self._now_ns(),
            self._vehicle_status_timeout,
        ):
            return "Waiting for fresh VehicleStatus"
        assert self._vehicle_status is not None
        if not self._vehicle_status.pre_flight_checks_pass:
            return "PX4 preflight checks have not passed"
        if self._vehicle_status.failsafe:
            return "PX4 failsafe is active"
        if self._vehicle_status.failure_detector_status != VehicleStatus.FAILURE_NONE:
            return "PX4 failure detector is active"
        if not timestamp_is_fresh(
            self._local_position_rx_ns,
            self._now_ns(),
            self._local_position_timeout,
        ):
            return "Waiting for fresh VehicleLocalPosition"
        if self._local_position_invalid_fields:
            fields = ", ".join(self._local_position_invalid_fields)
            return f"VehicleLocalPosition invalid: {fields}"
        if self._position_ned is None:
            return "VehicleLocalPosition has no valid position"
        return None

    def _requested_takeoff_altitude_amsl(self) -> float | None:
        if self._position_ned is None or self._ref_alt_amsl is None:
            return None
        return float(self._ref_alt_amsl - self._position_ned[2] + self._takeoff_height)

    def _fresh_controller_setpoint(self) -> VehicleRatesSetpoint | None:
        if self._controller_setpoint is None:
            return None
        if not timestamp_is_fresh(
            self._controller_rx_ns,
            self._now_ns(),
            self._controller_timeout,
        ):
            return None
        return self._controller_setpoint

    def _is_armed(self) -> bool:
        return bool(
            self._vehicle_status is not None
            and self._vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )

    def _is_disarmed(self) -> bool:
        # ARMING_STATE_DISARMED added in PX4 1.16; 1.14 uses ARMING_STATE_STANDBY (=1).
        disarmed = getattr(
            VehicleStatus,
            "ARMING_STATE_DISARMED",
            VehicleStatus.ARMING_STATE_STANDBY,
        )
        return bool(
            self._vehicle_status is not None
            and self._vehicle_status.arming_state == disarmed
        )

    def _is_offboard(self) -> bool:
        return bool(
            self._vehicle_status is not None
            and self._vehicle_status.nav_state
            == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

    def _is_native_takeoff_active(self) -> bool:
        return bool(
            self._vehicle_status is not None
            and self._vehicle_status.nav_state
            == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF
        )

    def _retry_due(self, previous_ns: int | None) -> bool:
        if previous_ns is None:
            return True
        return self._now_ns() - previous_ns >= int(self._command_retry_period * 1.0e9)

    def _elapsed(self, start_ns: int | None) -> float:
        if start_ns is None:
            return 0.0
        return (self._now_ns() - start_ns) / 1.0e9

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _warn_once(self, message: str) -> None:
        if message != self._last_warning:
            self._last_warning = message
            self.get_logger().warning(message)


def main(args=None) -> None:
    """Run the flight output node."""
    rclpy.init(args=args)
    node = DroneFlyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
