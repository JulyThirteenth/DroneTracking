"""Tests for exogenous waypoint-to-reference generation."""

from __future__ import annotations

import numpy as np
import torch

from drone_ccm.reference import tracking_reference_from_targets
from drone_ccm.waypoint import (
    WaypointTrajectory,
    bounded_heading_from_velocity,
    heading_from_velocity,
    load_waypoints_ned,
)


def test_heading_aligns_with_velocity_and_holds_at_rest() -> None:
    yaw = heading_from_velocity(np.array((0.0, 1.0, 0.0)), 0.0)
    assert np.isclose(yaw, np.pi / 2.0)
    assert heading_from_velocity(np.zeros(3), yaw) == yaw


def test_heading_is_continuous_across_angle_wrap() -> None:
    previous = np.pi - 0.01
    yaw = heading_from_velocity(np.array((-1.0, -0.01, 0.0)), previous)
    assert 0.0 < yaw - previous < 0.03


def test_heading_step_is_rate_bounded() -> None:
    yaw = bounded_heading_from_velocity(
        np.array((0.0, -1.0, 0.0)),
        current=0.0,
        maximum_delta=0.01,
    )
    assert np.isclose(yaw, -0.01)


def test_repository_waypoint_format_and_ned_to_enu(tmp_path) -> None:
    waypoint_file = tmp_path / "waypoints.txt"
    waypoint_file.write_text(
        "# x y z (NED)\n"
        "1.0, 2.0, -3.0,\n"
        "2.0, 4.0, -3.0,\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        load_waypoints_ned(waypoint_file),
        np.array(((1.0, 2.0, -3.0), (2.0, 4.0, -3.0))),
    )
    trajectory = WaypointTrajectory.from_ned_file(waypoint_file, 1.0)
    np.testing.assert_allclose(
        trajectory.sample(0.0).position,
        np.array((2.0, 1.0, 3.0)),
    )
    np.testing.assert_allclose(
        trajectory.sample(trajectory.duration).position,
        np.array((4.0, 2.0, 3.0)),
    )


def test_waypoint_clock_is_rest_to_rest_and_speed_bounded() -> None:
    maximum_speed = 1.2
    points = np.array(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 1.0, 0.4),
            (5.0, 1.5, 0.4),
        )
    )
    trajectory = WaypointTrajectory(points, maximum_speed)
    start = trajectory.sample(0.0)
    end = trajectory.sample(trajectory.duration)
    np.testing.assert_allclose(start.velocity, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(start.acceleration, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(end.velocity, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(end.acceleration, np.zeros(3), atol=1.0e-12)

    times = np.linspace(0.0, trajectory.duration, 4001)
    speeds = np.array(
        [np.linalg.norm(trajectory.sample(time).velocity) for time in times]
    )
    assert float(speeds.max()) <= maximum_speed * (1.0 + 1.0e-3)


def test_waypoint_clock_is_slowed_to_checkpoint_reference_domain() -> None:
    trajectory = WaypointTrajectory(
        np.array(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (1.0, 0.5, 0.0),
            )
        ),
        2.0,
    )
    original_duration = trajectory.duration
    trajectory.enforce_reference_limits(
        gravity=9.81,
        thrust_minimum=6.81,
        thrust_maximum=12.81,
        tilt_maximum=1.0,
        body_rate_maximum=0.3,
    )
    thrust, tilt, body_rate = trajectory._reference_extrema(9.81)
    assert trajectory.duration >= original_duration
    assert thrust[0] >= 6.81
    assert thrust[1] <= 12.81
    assert tilt <= 0.85
    assert body_rate <= 0.85 * 0.3


def test_waypoint_targets_produce_feedforward_without_position() -> None:
    trajectory = WaypointTrajectory(
        np.array(((0.0, 0.0, 0.0), (2.0, 0.5, 0.2), (4.0, 1.0, 0.0))),
        0.8,
    )
    time = 0.4 * trajectory.duration
    time_step = 0.02
    target = trajectory.sample(time)
    next_target = trajectory.sample(time + time_step)
    initial_velocity = torch.zeros(1, 3, dtype=torch.float64)
    initial_yaw = torch.zeros(1, dtype=torch.float64)

    velocity, rotation, control = tracking_reference_from_targets(
        time,
        torch.as_tensor(target.velocity, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(target.acceleration, dtype=torch.float64).reshape(1, 3),
        initial_yaw,
        next_target_velocity=torch.as_tensor(
            next_target.velocity,
            dtype=torch.float64,
        ).reshape(1, 3),
        next_target_acceleration=torch.as_tensor(
            next_target.acceleration,
            dtype=torch.float64,
        ).reshape(1, 3),
        next_target_yaw=initial_yaw,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        transition_duration=1.0,
        time_step=time_step,
        gravity=9.81,
    )
    assert velocity.shape == (1, 3)
    assert rotation.shape == (1, 3, 3)
    assert control.shape == (1, 4)
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
