"""Behavior tests for the flight executive's control arbitration."""

from __future__ import annotations

import numpy as np
import pytest
import rclpy
from px4_msgs.msg import (
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from std_msgs.msg import String

import drone_fsm.fly as fly


class FakePx4Bridge:
    """Record PX4 output requests without creating PX4 publishers."""

    def __init__(self, node, *, target_system: int) -> None:
        del node, target_system
        self.calls: list[tuple] = []

    def publish_position_mode(self) -> None:
        self.calls.append(("position_mode",))

    def publish_position_setpoint(self, position, yaw) -> None:
        self.calls.append(("position", tuple(position), float(yaw)))

    def publish_body_rate_mode(self) -> None:
        self.calls.append(("body_rate_mode",))

    def publish_rates_setpoint(self, message) -> None:
        self.calls.append(("rates", message))

    def send_offboard(self) -> None:
        self.calls.append(("offboard",))

    def send_arm(self) -> None:
        self.calls.append(("arm",))

    def send_takeoff(self, *, altitude_amsl=None) -> None:
        self.calls.append(("takeoff", altitude_amsl))

    def send_land(self) -> None:
        self.calls.append(("land",))


def local_position(x: float, y: float, z: float) -> VehicleLocalPosition:
    """Construct one finite, estimator-valid local-position sample."""
    message = VehicleLocalPosition()
    message.x = x
    message.y = y
    message.z = z
    message.vx = 0.0
    message.vy = 0.0
    message.vz = 0.0
    message.heading = 0.25
    message.xy_valid = True
    message.z_valid = True
    message.v_xy_valid = True
    message.v_z_valid = True
    message.heading_good_for_control = True
    message.z_global = True
    message.ref_alt = 100.0
    return message


def rates(roll: float = 0.1) -> VehicleRatesSetpoint:
    """Construct one finite CTBR command."""
    message = VehicleRatesSetpoint()
    message.roll = roll
    message.pitch = -0.2
    message.yaw = 0.3
    message.thrust_body = [0.0, 0.0, -0.58]
    return message


def vehicle_status(
    *,
    armed: bool,
    nav_state: int = VehicleStatus.NAVIGATION_STATE_MANUAL,
    healthy: bool = True,
) -> VehicleStatus:
    """Construct one PX4 status sample."""
    message = VehicleStatus()
    message.arming_state = (
        VehicleStatus.ARMING_STATE_ARMED
        if armed
        else VehicleStatus.ARMING_STATE_DISARMED
    )
    message.nav_state = int(nav_state)
    message.pre_flight_checks_pass = bool(healthy)
    message.failsafe = False
    message.failure_detector_status = VehicleStatus.FAILURE_NONE
    return message


def command_ack(command: int, result: int) -> VehicleCommandAck:
    """Construct a PX4 vehicle-command acknowledgement."""
    message = VehicleCommandAck()
    message.command = int(command)
    message.result = int(result)
    return message


def make_preflight_ready(node, *, x=0.0, y=0.0, z=0.0) -> None:
    """Supply fresh, healthy PX4 state."""
    node._on_vehicle_status(vehicle_status(armed=False, healthy=True))
    node._on_local_position(local_position(x, y, z))


@pytest.fixture
def fly_node(monkeypatch):
    """Create a flight node with a recording PX4 boundary."""
    monkeypatch.setattr(fly, "Px4Bridge", FakePx4Bridge)
    if not rclpy.ok():
        rclpy.init()
    node = fly.DroneFlyNode()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_ctbr_validation_and_freshness_boundaries() -> None:
    """Validate CTBR, estimator state, and time boundaries."""
    assert fly.rates_setpoint_is_finite(rates())
    assert fly.timestamp_is_fresh(100, 200, 1.0e-7)
    assert not fly.timestamp_is_fresh(100, 201, 1.0e-7)
    assert not fly.timestamp_is_fresh(None, 200, 1.0)
    assert not fly.timestamp_is_fresh(201, 200, 1.0)

    invalid = rates(float("nan"))
    assert not fly.rates_setpoint_is_finite(invalid)

    position = local_position(0.0, 0.0, 0.0)
    assert not fly.local_position_invalid_fields(position)
    position.heading_good_for_control = False
    position.heading = float("nan")
    assert not fly.local_position_invalid_fields(position)


def test_tracking_prestreams_before_offboard_and_recovers(fly_node) -> None:
    """Enter Body Rate only after takeoff, CTBR prestream, and PX4 confirmation."""
    node = fly_node
    bridge = node._px4

    node._on_local_position(local_position(1.0, 2.0, -1.0))
    node._on_vehicle_status(vehicle_status(armed=True))
    node._on_state(String(data=fly.STATE_TRACKING))

    node._on_timer()
    assert bridge.calls[-2:] == [
        ("position_mode",),
        ("position", (1.0, 2.0, -1.0), 0.25),
    ]

    command = rates()
    node._on_controller_setpoint(command)
    node._on_timer()
    assert bridge.calls[-2:] == [("body_rate_mode",), ("rates", command)]
    assert ("offboard",) not in bridge.calls
    assert not node._rate_control_active

    node._tracking_prestream_start_ns = node._now_ns() - int(
        (node._prestream_time + 0.01) * 1.0e9
    )
    node._on_timer()
    assert bridge.calls[-1] == ("offboard",)
    assert not node._rate_control_active

    node._on_vehicle_status(
        vehicle_status(
            armed=True,
            nav_state=VehicleStatus.NAVIGATION_STATE_OFFBOARD,
        )
    )
    node._on_timer()
    assert node._rate_control_active

    node._on_local_position(local_position(4.0, 5.0, -2.0))
    node._controller_rx_ns = node._now_ns() - int(
        (node._controller_timeout + 0.01) * 1.0e9
    )
    node._on_timer()
    assert bridge.calls[-2:] == [
        ("position_mode",),
        ("position", (4.0, 5.0, -2.0), 0.25),
    ]

    recovered = rates(roll=-0.1)
    node._on_controller_setpoint(recovered)
    node._on_timer()
    assert bridge.calls[-2:] == [("body_rate_mode",), ("rates", recovered)]
    assert node._rate_control_active

    invalid = rates(float("inf"))
    node._on_controller_setpoint(invalid)
    node._on_timer()
    assert bridge.calls[-2:] == [
        ("position_mode",),
        ("position", (4.0, 5.0, -2.0), 0.25),
    ]
    assert node._controller_setpoint is None


def test_tracking_has_no_sample_counter_or_failure_latch(fly_node) -> None:
    """Keep tracking arbitration free of the removed hidden state."""
    removed = (
        "_controller_valid_samples",
        "_controller_previous_rx_ns",
        "_tracking_required_samples",
        "_tracking_engaged",
        "_tracking_failed",
        "_local_position_valid",
        "_local_state_is_fresh",
    )
    assert not any(hasattr(fly_node, name) for name in removed)
    assert np.isclose(fly_node._controller_timeout, 0.30)
    assert np.isclose(fly_node._vehicle_status_timeout, 1.50)
    assert np.isclose(fly_node._local_position_timeout, 0.50)
    assert fly_node._vehicle_status_topic == "/fmu/out/vehicle_status_v1"


def test_vehicle_status_timeout_exceeds_px4_publish_period(fly_node) -> None:
    """Allow scheduling margin around PX4's fixed 2 Hz status publication."""
    node = fly_node
    make_preflight_ready(node)
    now_ns = 10_000_000_000
    node._now_ns = lambda: now_ns
    node._vehicle_status_rx_ns = now_ns - 750_000_000
    node._local_position_rx_ns = now_ns
    assert node._takeoff_failure() is None

    node._vehicle_status_rx_ns = now_ns - 1_500_000_001
    assert node._takeoff_failure() == "Waiting for fresh VehicleStatus"


def test_unaligned_heading_allows_native_takeoff_without_yaw_lock(fly_node) -> None:
    """Match PX4 Auto behavior by leaving yaw unset until alignment completes."""
    node = fly_node
    position = local_position(1.0, 2.0, -0.1)
    position.heading_good_for_control = False
    node._on_vehicle_status(vehicle_status(armed=False, healthy=True))
    node._on_local_position(position)
    node._on_state(String(data=fly.STATE_READY))
    node._on_timer()
    assert node._hold_target_ned is not None
    assert not hasattr(node, "_home_ned")
    assert node._px4.calls[-1][0:2] == ("position", (1.0, 2.0, -0.1))
    assert np.isnan(node._px4.calls[-1][2])


def test_ready_takeoff_return_and_land_are_timer_driven(fly_node) -> None:
    """Use native Takeoff, then cover return and landing output modes."""
    node = fly_node
    bridge = node._px4
    make_preflight_ready(node, x=2.0, y=-3.0, z=-0.2)

    node._on_state(String(data=fly.STATE_READY))
    assert bridge.calls == []
    node._on_timer()
    assert bridge.calls[-2:] == [
        ("position_mode",),
        ("position", (2.0, -3.0, -0.2), 0.25),
    ]
    assert ("offboard",) not in bridge.calls

    node._on_timer()
    assert ("offboard",) not in bridge.calls

    calls_before_state_change = len(bridge.calls)
    node._on_state(String(data=fly.STATE_HOVER_START))
    assert len(bridge.calls) == calls_before_state_change
    node._on_timer()
    assert bridge.calls[-1] == ("arm",)
    assert not any(call[0] == "takeoff" for call in bridge.calls)
    assert ("offboard",) not in bridge.calls

    node._on_command_ack(
        command_ack(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
        )
    )
    node._on_vehicle_status(vehicle_status(armed=True))
    node._on_timer()
    assert bridge.calls[-1] == ("takeoff", pytest.approx(101.2))

    node._on_command_ack(
        command_ack(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
        )
    )
    node._on_vehicle_status(
        vehicle_status(
            armed=True,
            nav_state=VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF,
        )
    )
    node._on_local_position(local_position(2.0, -3.0, -1.2))
    node._on_timer()
    assert node._is_native_takeoff_active()

    node._on_state(String(data=fly.STATE_TRACKING))
    node._on_local_position(local_position(7.0, 8.0, -2.0))
    node._on_state(String(data=fly.STATE_RETURN_HOVER))
    node._on_timer()
    assert bridge.calls[-2:] == [
        ("position_mode",),
        ("position", (2.0, -3.0, -1.2), 0.25),
    ]

    node._on_state(String(data=fly.STATE_PREFLIGHT))
    assert bridge.calls[-1] != ("land",)
    node._on_timer()
    assert bridge.calls[-1] == ("land",)

    node._on_vehicle_status(vehicle_status(armed=False, healthy=True))
    land_count = bridge.calls.count(("land",))
    node._on_timer()
    assert bridge.calls.count(("land",)) == land_count
    assert not node._landing_active


def test_invalid_estimator_or_px4_health_blocks_takeoff(fly_node) -> None:
    """Do not capture hold or command Takeoff from invalid PX4 state."""
    node = fly_node
    invalid_position = local_position(0.0, 0.0, 0.5)
    invalid_position.xy_valid = False
    node._on_vehicle_status(vehicle_status(armed=False, healthy=True))
    node._on_local_position(invalid_position)
    node._on_state(String(data=fly.STATE_READY))
    node._on_timer()
    assert node._hold_target_ned is None
    assert node._px4.calls == []

    node._on_local_position(local_position(0.0, 0.0, 0.5))
    node._on_vehicle_status(vehicle_status(armed=False, healthy=False))
    node._on_timer()
    assert node._hold_target_ned is None
    assert node._px4.calls == []


def test_takeoff_rejection_never_arms_or_switches_offboard(fly_node) -> None:
    """Latch a denied native Takeoff and require a new user attempt."""
    node = fly_node
    make_preflight_ready(node)
    node._on_state(String(data=fly.STATE_READY))
    node._on_timer()
    node._on_state(String(data=fly.STATE_HOVER_START))
    node._on_timer()
    assert node._px4.calls[-1] == ("arm",)
    node._on_vehicle_status(vehicle_status(armed=True))
    node._on_timer()
    assert node._px4.calls[-1][0] == "takeoff"

    node._on_command_ack(
        command_ack(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED,
        )
    )
    call_count = len(node._px4.calls)
    node._on_timer()
    assert len(node._px4.calls) == call_count
    assert node._takeoff_failed
    assert ("offboard",) not in node._px4.calls


def test_execute_before_takeoff_completion_has_no_px4_output(fly_node) -> None:
    """Keep PX4 native control when the external FSM advances too early."""
    node = fly_node
    node._on_local_position(local_position(0.0, 0.0, -0.2))
    node._on_state(String(data=fly.STATE_TRACKING))
    node._on_controller_setpoint(rates())
    node._on_timer()
    assert node._px4.calls == []


def test_missing_local_position_produces_no_px4_output(fly_node) -> None:
    """Never invent a position target before PX4 supplies local position."""
    node = fly_node
    node._on_state(String(data=fly.STATE_READY))
    node._on_timer()
    assert node._px4.calls == []


def test_unknown_state_is_rejected_without_changing_behavior(fly_node) -> None:
    """Keep the last valid state when an invalid state message arrives."""
    node = fly_node
    node._on_state(String(data=fly.STATE_READY))
    node._on_state(String(data="typo"))
    assert node._fsm_state == fly.STATE_READY
