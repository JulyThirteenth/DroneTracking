"""Message-boundary tests for the PX4 flight bridge."""

from __future__ import annotations

import math

from px4_msgs.msg import VehicleCommand

from drone_fsm.px4 import Px4Bridge


class RecordingPublisher:
    """Capture messages passed through one publisher boundary."""

    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


def bridge_with_command_publisher() -> tuple[Px4Bridge, RecordingPublisher]:
    """Build a bridge without constructing a ROS node."""
    publisher = RecordingPublisher()
    bridge = Px4Bridge.__new__(Px4Bridge)
    bridge._target_system = 1
    bridge._command_publisher = publisher
    bridge._timestamp_us = lambda: 123456
    return bridge, publisher


def test_native_takeoff_command_uses_current_location_and_amsl() -> None:
    """Encode NAV_TAKEOFF without accidental zero-valued global fields."""
    bridge, publisher = bridge_with_command_publisher()
    bridge.send_takeoff(altitude_amsl=101.2)

    message = publisher.messages[-1]
    assert message.command == VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF
    assert message.timestamp == 123456
    assert math.isnan(message.param4)
    assert math.isnan(message.param5)
    assert math.isnan(message.param6)
    assert math.isclose(message.param7, 101.2, abs_tol=1.0e-5)


def test_native_takeoff_without_global_reference_lets_px4_choose_height() -> None:
    """Use PX4's configured takeoff altitude if AMSL is unavailable."""
    bridge, publisher = bridge_with_command_publisher()
    bridge.send_takeoff()
    assert math.isnan(publisher.messages[-1].param7)
