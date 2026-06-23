"""Single authoritative PX4 NED/FRD to controller ENU/FLU conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ENU_FROM_NED = np.array(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
)
FRD_FROM_FLU = np.diag((1.0, -1.0, -1.0))


@dataclass(frozen=True)
class VehicleState:
    """Controller state expressed as ENU velocity and FLU attitude."""

    velocity: np.ndarray
    rotation: np.ndarray


def collective_to_normalized_thrust(
    collective_acceleration: float,
    hover_thrust: float,
    gravity: float,
) -> float:
    """Converts collective acceleration to PX4 normalized thrust."""
    return hover_thrust * collective_acceleration / gravity


def quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Converts an active scalar-first unit quaternion to a rotation."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Quaternion must contain four finite values")
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = values / norm
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)),
        )
    )


def odometry_to_enu_flu(
    quaternion_wxyz: np.ndarray,
    velocity: np.ndarray,
    *,
    velocity_is_body_frd: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts one coherent PX4 odometry sample to ENU/FLU."""
    rotation_ned_frd = quaternion_wxyz_to_rotation(quaternion_wxyz)
    velocity_values = np.asarray(velocity, dtype=float)
    if velocity_values.shape != (3,) or not np.all(np.isfinite(velocity_values)):
        raise ValueError("Velocity must contain three finite values")
    velocity_ned = (
        rotation_ned_frd @ velocity_values
        if velocity_is_body_frd
        else velocity_values
    )
    return (
        ENU_FROM_NED @ velocity_ned,
        ENU_FROM_NED @ rotation_ned_frd @ FRD_FROM_FLU,
    )
