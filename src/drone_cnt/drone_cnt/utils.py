"""Frame conversion and path geometry used by the controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Vec3 = np.ndarray


def as_vec3(value) -> Vec3:
    """Return a finite three-vector."""
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("Expected three finite values")
    return vector


def ned_to_enu(value) -> Vec3:
    """Convert a vector from NED to ENU."""
    north, east, down = as_vec3(value)
    return np.array([east, north, -down], dtype=float)


def enu_to_ned(value) -> Vec3:
    """Convert a vector from ENU to NED."""
    east, north, up = as_vec3(value)
    return np.array([north, east, -up], dtype=float)


def wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def yaw_ned_to_enu(yaw_ned: float) -> float:
    """Convert a NED heading to an ENU yaw."""
    return wrap_pi(math.pi / 2.0 - float(yaw_ned))


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """Convert an ENU yaw to a NED heading."""
    return wrap_pi(math.pi / 2.0 - float(yaw_enu))


def yaw_rate_enu_to_ned(yaw_rate_enu: float) -> float:
    """Convert an ENU yaw rate to NED."""
    return -float(yaw_rate_enu)


@dataclass(frozen=True)
class Polyline3D:
    """Three-dimensional polyline parameterized by arc length."""

    points: np.ndarray
    arc_lengths: np.ndarray
    length: float

    @classmethod
    def from_points(cls, points) -> "Polyline3D":
        """Validate points, remove adjacent duplicates and build arc lengths."""
        array = np.asarray(points, dtype=float)
        if array.ndim != 2:
            raise ValueError("Path must have shape (M, 3) or (3, M)")
        if array.shape[0] == 3 and array.shape[1] != 3:
            array = array.T
        if array.shape[1:] != (3,) or array.shape[0] < 2:
            raise ValueError("Path must contain at least two 3D points")
        if not np.all(np.isfinite(array)):
            raise ValueError("Path contains non-finite values")

        keep = np.r_[True, np.linalg.norm(np.diff(array, axis=0), axis=1) > 1.0e-9]
        points = array[keep]
        if points.shape[0] < 2:
            raise ValueError("Path must contain at least two distinct points")

        arc_lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        return cls(points=points, arc_lengths=arc_lengths, length=float(arc_lengths[-1]))

    def closest_arc_length(self, position) -> float:
        """Return the arc length of the closest point on the polyline."""
        query = as_vec3(position)
        starts = self.points[:-1]
        segments = np.diff(self.points, axis=0)
        squared_lengths = np.sum(segments * segments, axis=1)
        ratios = np.sum((query - starts) * segments, axis=1) / squared_lengths
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts + ratios[:, None] * segments
        index = int(np.argmin(np.linalg.norm(query - projections, axis=1)))
        segment_length = math.sqrt(float(squared_lengths[index]))
        return float(self.arc_lengths[index] + ratios[index] * segment_length)

    def sample_with_tangent(self, arc_length: float) -> tuple[Vec3, Vec3]:
        """Interpolate a point and unit tangent at an arc length."""
        distance = float(np.clip(arc_length, 0.0, self.length))
        index = int(np.searchsorted(self.arc_lengths, distance, side="right") - 1)
        index = min(index, self.points.shape[0] - 2)
        segment = self.points[index + 1] - self.points[index]
        segment_length = float(np.linalg.norm(segment))
        tangent = segment / segment_length
        offset = distance - float(self.arc_lengths[index])
        return self.points[index] + offset * tangent, tangent


def flatness_to_ctbr(
    acceleration_ned,
    jerk_ned,
    yaw_ned: float,
    yaw_rate_ned: float = 0.0,
    *,
    hover_thrust: float = 0.55,
    gravity: float = 9.81,
    thrust_min: float = 0.10,
    thrust_max: float = 0.90,
) -> tuple[float, float, float, float]:
    """Convert flatness commands to body rates and normalized thrust."""
    acceleration = as_vec3(acceleration_ned)
    jerk = as_vec3(jerk_ned)
    yaw = float(yaw_ned)
    yaw_rate = float(yaw_rate_ned)

    force = gravity * np.array([0.0, 0.0, 1.0]) - acceleration
    force_norm = float(np.linalg.norm(force)) + 1.0e-6
    body_z = force / force_norm
    thrust = float(np.clip(hover_thrust * force_norm / gravity, thrust_min, thrust_max))

    heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cross = np.cross(body_z, heading)
    cross_norm = float(np.linalg.norm(cross)) + 1.0e-6
    body_y = cross / cross_norm
    body_x = np.cross(body_y, body_z)

    projection = np.eye(3) - np.outer(body_z, body_z)
    body_z_dot = projection @ (-jerk) / force_norm
    heading_dot = yaw_rate * np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    cross_dot = np.cross(body_z_dot, heading) + np.cross(body_z, heading_dot)
    body_y_dot = (np.eye(3) - np.outer(body_y, body_y)) @ cross_dot / cross_norm
    body_x_dot = np.cross(body_y_dot, body_z) + np.cross(body_y, body_z_dot)

    rotation = np.column_stack((body_x, body_y, body_z))
    rotation_dot = np.column_stack((body_x_dot, body_y_dot, body_z_dot))
    angular_velocity = rotation.T @ rotation_dot
    return (
        float(angular_velocity[2, 1]),
        float(angular_velocity[0, 2]),
        float(angular_velocity[1, 0]),
        thrust,
    )
