"""Frame conversion and waypoint-loading utilities."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np

Vec3 = np.ndarray
_NED_TO_ENU_SCALE = np.array([1.0, 1.0, -1.0], dtype=float)


def as_vec3(value: Iterable[float]) -> Vec3:
    """Convert an array-like value to a finite vector shaped ``(3,)``."""
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected a finite 3-D vector, got {vector}")
    return vector


def ned_to_enu(value: Iterable[float] | np.ndarray) -> np.ndarray:
    """Transform vector(s) shaped ``(..., 3)`` from NED to ENU."""
    vectors = np.asarray(value, dtype=float)
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError(
            f"NED vector array must end with dimension 3, got {vectors.shape}"
        )
    return vectors[..., [1, 0, 2]] * _NED_TO_ENU_SCALE


def wrap_pi(angle_rad: float) -> float:
    """Wrap a finite angle to ``[-pi, pi]``."""
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError(f"Angle must be finite, got {angle}")
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_ned_to_enu(yaw_ned: float) -> float:
    """Convert PX4/NED heading to ROS ENU yaw."""
    return wrap_pi(0.5 * np.pi - wrap_pi(yaw_ned))


def quat_from_yaw_enu(
    yaw_enu: float,
) -> tuple[float, float, float, float]:
    """Return an ENU quaternion in ROS ``(x, y, z, w)`` order."""
    half_yaw = 0.5 * wrap_pi(yaw_enu)
    return 0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw))


def load_waypoints_ned(
    path: str | Path,
    *,
    origin_ned: Iterable[float] | None = (0.0, 0.0, 0.0),
    origin_mode: str = "fixed",
) -> np.ndarray:
    """Load finite NED waypoints and subtract the selected NED origin."""
    waypoint_path = Path(path).expanduser()
    if not waypoint_path.is_file():
        return np.empty((0, 3), dtype=float)

    rows = []
    for raw_line in waypoint_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            rows.append(line.replace(",", " "))

    if not rows:
        return np.empty((0, 3), dtype=float)

    data = np.loadtxt(
        StringIO("\n".join(rows)),
        dtype=float,
        ndmin=2,
    )
    if data.shape[1] < 3:
        raise ValueError(
            f"Waypoint file must contain at least 3 columns: {waypoint_path}"
        )

    points = np.asarray(data[:, :3], dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError(
            f"Waypoint file contains non-finite values: {waypoint_path}"
        )

    mode = str(origin_mode).strip().casefold()
    if mode == "fixed":
        if origin_ned is None:
            raise ValueError("origin_ned is required when origin_mode='fixed'")
        origin = as_vec3(origin_ned)
    elif mode == "first":
        origin = points[0]
    elif mode == "first_xy":
        origin_z = 0.0 if origin_ned is None else as_vec3(origin_ned)[2]
        origin = np.array([points[0, 0], points[0, 1], origin_z])
    else:
        raise ValueError(
            f"Unsupported origin_mode={origin_mode!r}; "
            "expected fixed, first or first_xy"
        )

    return points - origin
