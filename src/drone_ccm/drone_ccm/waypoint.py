"""Smooth time-parameterized waypoint trajectories for CCM references."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


_MINIMUM_SEGMENT_LENGTH = 1.0e-6
_MINIMUM_HEADING_SPEED = 1.0e-6
_PEAK_QUINTIC_RATE = 1.875


@dataclass(frozen=True)
class WaypointSample:
    """One ENU trajectory sample and its time derivatives."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def heading_from_velocity(velocity: np.ndarray, fallback: float) -> float:
    """Returns continuous ENU yaw aligned with horizontal velocity."""
    horizontal = np.asarray(velocity, dtype=float)[:2]
    if np.linalg.norm(horizontal) <= _MINIMUM_HEADING_SPEED:
        return float(fallback)
    wrapped = float(np.arctan2(horizontal[1], horizontal[0]))
    delta = np.arctan2(np.sin(wrapped - fallback), np.cos(wrapped - fallback))
    return float(fallback + delta)


def bounded_heading_from_velocity(
    velocity: np.ndarray,
    current: float,
    maximum_delta: float,
) -> float:
    """Moves yaw toward horizontal velocity without exceeding one-step motion."""
    target = heading_from_velocity(velocity, current)
    delta = np.arctan2(np.sin(target - current), np.cos(target - current))
    return float(current + np.clip(delta, -maximum_delta, maximum_delta))


def load_waypoints_ned(path: str | Path) -> np.ndarray:
    """Loads finite NED waypoints from a comma- or whitespace-separated file."""
    waypoint_path = Path(path).expanduser().resolve()
    if not waypoint_path.is_file():
        raise FileNotFoundError(f"Waypoint file not found: {waypoint_path}")

    rows: list[str] = []
    for raw_line in waypoint_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            rows.append(line.replace(",", " "))
    if not rows:
        raise ValueError(f"Waypoint file is empty: {waypoint_path}")

    values = np.loadtxt(StringIO("\n".join(rows)), dtype=float, ndmin=2)
    if values.shape[1] < 3:
        raise ValueError("Waypoint file must contain at least x, y, and z columns")
    points = np.asarray(values[:, :3], dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("Waypoint file contains non-finite coordinates")
    return points


def _ned_to_enu(points_ned: np.ndarray) -> np.ndarray:
    """Converts vectors from PX4 NED coordinates to ROS ENU coordinates."""
    points = np.asarray(points_ned, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected waypoint array shaped (N, 3), got {points.shape}")
    return points[:, (1, 0, 2)] * np.array((1.0, 1.0, -1.0))


class WaypointTrajectory:
    """Natural cubic waypoint spline with a rest-to-rest quintic clock.

    The trajectory is exogenous: it advances using elapsed time and never
    projects measured vehicle position onto the path.
    """

    def __init__(
        self,
        points_enu: np.ndarray,
        maximum_speed: float,
    ) -> None:
        if not np.isfinite(maximum_speed) or maximum_speed <= 0.0:
            raise ValueError("maximum_speed must be finite and positive")
        points = self._remove_consecutive_duplicates(points_enu)
        if points.shape[0] < 2:
            raise ValueError("Waypoint trajectory requires two distinct points")

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc_length = np.concatenate(((0.0,), np.cumsum(segment_lengths)))
        self._length = float(arc_length[-1])
        self._spline = CubicSpline(
            arc_length,
            points,
            axis=0,
            bc_type="natural",
        )

        sample_count = max(256, 64 * (points.shape[0] - 1))
        sample_arc = np.linspace(0.0, self._length, sample_count)
        maximum_arc_derivative = float(
            np.linalg.norm(self._spline(sample_arc, 1), axis=1).max()
        )
        self._duration = (
            _PEAK_QUINTIC_RATE
            * self._length
            * maximum_arc_derivative
            / float(maximum_speed)
        )

    @classmethod
    def from_ned_file(
        cls,
        path: str | Path,
        maximum_speed: float,
    ) -> "WaypointTrajectory":
        """Builds a trajectory from a repository-format NED waypoint file."""
        return cls(
            _ned_to_enu(load_waypoints_ned(path)),
            maximum_speed,
        )

    @property
    def duration(self) -> float:
        """Returns the complete rest-to-rest traversal time in seconds."""
        return self._duration

    def sample(self, time: float) -> WaypointSample:
        """Samples position, velocity, and acceleration."""
        time_value = float(np.clip(time, 0.0, self._duration))
        ratio = time_value / self._duration
        envelope = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5
        envelope_rate = (
            30.0 * ratio**2 - 60.0 * ratio**3 + 30.0 * ratio**4
        ) / self._duration
        envelope_acceleration = (
            60.0 * ratio - 180.0 * ratio**2 + 120.0 * ratio**3
        ) / self._duration**2

        arc = self._length * envelope
        arc_rate = self._length * envelope_rate
        arc_acceleration = self._length * envelope_acceleration
        position = np.asarray(self._spline(arc), dtype=float)
        path_tangent = np.asarray(self._spline(arc, 1), dtype=float)
        path_curvature = np.asarray(self._spline(arc, 2), dtype=float)
        velocity = path_tangent * arc_rate
        acceleration = (
            path_curvature * arc_rate**2
            + path_tangent * arc_acceleration
        )
        return WaypointSample(position, velocity, acceleration)

    def enforce_reference_limits(
        self,
        *,
        gravity: float,
        thrust_minimum: float,
        thrust_maximum: float,
        tilt_maximum: float,
        body_rate_maximum: float,
        safety_factor: float = 0.85,
    ) -> None:
        """Slows the clock until the generated physical reference is in-domain.

        Only the exogenous clock is changed. The waypoint geometry is unchanged,
        and no measured vehicle position is used.
        """
        limits = np.asarray(
            (
                gravity,
                thrust_minimum,
                thrust_maximum,
                tilt_maximum,
                body_rate_maximum,
                safety_factor,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(limits)):
            raise ValueError("Reference limits must be finite")
        if gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if thrust_minimum <= 0.0 or thrust_maximum <= thrust_minimum:
            raise ValueError("Invalid thrust limits")
        if tilt_maximum <= 0.0 or body_rate_maximum <= 0.0:
            raise ValueError("Attitude and body-rate limits must be positive")
        if not 0.0 < safety_factor < 1.0:
            raise ValueError("safety_factor must be in (0, 1)")

        for _ in range(24):
            thrust, tilt, body_rate = self._reference_extrema(gravity)
            if (
                thrust[0] >= thrust_minimum
                and thrust[1] <= thrust_maximum
                and tilt <= safety_factor * tilt_maximum
                and body_rate <= safety_factor * body_rate_maximum
            ):
                return
            self._duration *= 1.25
        raise ValueError("Waypoint geometry cannot satisfy checkpoint reference limits")

    def _reference_extrema(
        self,
        gravity: float,
    ) -> tuple[tuple[float, float], float, float]:
        sample_count = 4097
        times = np.linspace(0.0, self._duration, sample_count)
        samples = [self.sample(time) for time in times]
        accelerations = np.stack([sample.acceleration for sample in samples])
        velocities = np.stack([sample.velocity for sample in samples])
        forces = accelerations + np.array((0.0, 0.0, gravity))
        thrust = np.linalg.norm(forces, axis=1)
        if float(thrust.min()) < 1.0e-9:
            return (0.0, float(thrust.max())), float("inf"), float("inf")
        body_z = forces / thrust[:, np.newaxis]
        moving = np.flatnonzero(
            np.linalg.norm(velocities[:, :2], axis=1) > _MINIMUM_HEADING_SPEED
        )
        yaw = (
            float(np.arctan2(velocities[moving[0], 1], velocities[moving[0], 0]))
            if moving.size
            else 0.0
        )
        yaw_values = []
        for velocity in velocities:
            yaw = heading_from_velocity(velocity, yaw)
            yaw_values.append(yaw)
        heading = np.stack(
            (np.cos(yaw_values), np.sin(yaw_values), np.zeros(len(yaw_values))),
            axis=1,
        )
        body_y = np.cross(body_z, heading)
        body_y_norm = np.linalg.norm(body_y, axis=1, keepdims=True)
        if float(body_y_norm.min()) < 1.0e-9:
            return (
                (float(thrust.min()), float(thrust.max())),
                float("inf"),
                float("inf"),
            )
        body_y /= body_y_norm
        body_x = np.cross(body_y, body_z)
        rotations = np.stack((body_x, body_y, body_z), axis=-1)
        relative = np.swapaxes(rotations[:-1], 1, 2) @ rotations[1:]
        time_step = self._duration / (sample_count - 1)
        body_rates = Rotation.from_matrix(relative).as_rotvec() / time_step
        tilt = np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0))
        return (
            (float(thrust.min()), float(thrust.max())),
            float(tilt.max()),
            float(np.linalg.norm(body_rates, axis=1).max()),
        )

    @staticmethod
    def _remove_consecutive_duplicates(points_enu: np.ndarray) -> np.ndarray:
        points = np.asarray(points_enu, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected waypoint array shaped (N, 3), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("Waypoints must be finite")
        if points.shape[0] <= 1:
            return points.copy()
        keep = np.concatenate(
            (
                np.array((True,)),
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > _MINIMUM_SEGMENT_LENGTH,
            )
        )
        return points[keep]
