"""PX4 input-message publisher used by the flight executive."""

from __future__ import annotations

import math

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleRatesSetpoint,
)
from rclpy.node import Node

from drone_fsm.qos import best_effort_qos


class Px4Bridge:
    """
    PX4 bridge owned by drone_fly.

    Responsibilities:
    - position-mode OffboardControlMode;
    - body-rate OffboardControlMode heartbeat;
    - position setpoints;
    - vehicle commands.

    This bridge is the only publisher of PX4 control inputs.
    """

    def __init__(
        self,
        node: Node,
        *,
        target_system: int = 1,
    ) -> None:
        self._node = node
        self._target_system = int(target_system)
        qos = best_effort_qos()

        self._offboard_publisher = node.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            qos,
        )

        self._trajectory_publisher = node.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            qos,
        )

        self._rates_publisher = node.create_publisher(
            VehicleRatesSetpoint,
            "/fmu/in/vehicle_rates_setpoint",
            qos,
        )

        self._command_publisher = node.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            qos,
        )

    def publish_position_mode(self) -> None:
        """Publish a PX4 position-control Offboard heartbeat."""
        self._publish_control_mode(position=True)

    def publish_body_rate_mode(self) -> None:
        """Publish a PX4 body-rate Offboard heartbeat."""
        self._publish_control_mode(body_rate=True)

    def _publish_control_mode(
        self,
        *,
        position: bool = False,
        body_rate: bool = False,
    ) -> None:
        """Publish one explicitly selected PX4 Offboard control mode."""
        message = OffboardControlMode()
        message.timestamp = self._timestamp_us()
        message.position = bool(position)
        message.velocity = False
        message.acceleration = False
        message.attitude = False
        message.body_rate = bool(body_rate)
        # PX4 <=1.14 uses 'actuator'; >=1.15 split into 'thrust_and_torque' and
        # 'direct_actuator'. Set whichever fields the installed px4_msgs exposes.
        if hasattr(message, "thrust_and_torque"):
            message.thrust_and_torque = False
            message.direct_actuator = False
        else:
            message.actuator = False

        self._offboard_publisher.publish(message)

    def publish_position_setpoint(
        self,
        position_ned: tuple[float, float, float],
        yaw_ned: float,
    ) -> None:
        """Publish one PX4 NED position setpoint."""
        nan = float("nan")

        message = TrajectorySetpoint()
        message.timestamp = self._timestamp_us()

        message.position = [
            float(position_ned[0]),
            float(position_ned[1]),
            float(position_ned[2]),
        ]

        message.velocity = [nan, nan, nan]
        message.acceleration = [nan, nan, nan]
        message.jerk = [nan, nan, nan]

        message.yaw = float(yaw_ned)
        message.yawspeed = nan

        self._trajectory_publisher.publish(message)

    def publish_rates_setpoint(
        self,
        source: VehicleRatesSetpoint,
    ) -> None:
        """Forward the PX4-FRD CTBR command without another transform."""
        message = VehicleRatesSetpoint()
        message.timestamp = self._timestamp_us()
        message.roll = float(source.roll)
        message.pitch = float(source.pitch)
        message.yaw = float(source.yaw)
        message.thrust_body = [float(value) for value in source.thrust_body]
        message.reset_integral = bool(source.reset_integral)
        self._rates_publisher.publish(message)

    def send_offboard(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

    def send_arm(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

    def send_takeoff(self, *, altitude_amsl: float | None = None) -> None:
        """
        Request PX4's native takeoff mode.

        PX4 accepts ``NaN`` for the global target fields and then uses its
        configured takeoff altitude.  When a valid local-to-global altitude
        reference is available, ``altitude_amsl`` makes the requested height
        explicit while latitude and longitude remain at the current position.
        """
        altitude = (
            float(altitude_amsl)
            if altitude_amsl is not None and math.isfinite(altitude_amsl)
            else math.nan
        )
        self._send_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param4=math.nan,
            param5=math.nan,
            param6=math.nan,
            param7=altitude,
        )

    def send_land(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            param5=math.nan,
            param6=math.nan,
            param7=math.nan,
        )

    def _send_command(
        self,
        command: int,
        *,
        param1: float = math.nan,
        param2: float = math.nan,
        param3: float = math.nan,
        param4: float = math.nan,
        param5: float = math.nan,
        param6: float = math.nan,
        param7: float = math.nan,
    ) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp_us()

        message.param1 = float(param1)
        message.param2 = float(param2)
        message.param3 = float(param3)
        message.param4 = float(param4)
        message.param5 = float(param5)
        message.param6 = float(param6)
        message.param7 = float(param7)

        message.command = int(command)

        message.target_system = self._target_system
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True

        self._command_publisher.publish(message)

    def _timestamp_us(self) -> int:
        return int(self._node.get_clock().now().nanoseconds // 1000)
