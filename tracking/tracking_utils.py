from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


Vec3 = np.ndarray


def quat_from_yaw_enu(yaw_enu: float) -> tuple[float, float, float, float]:
    yaw = float(wrap_pi(float(yaw_enu)))
    half = 0.5 * yaw
    return (0.0, 0.0, float(math.sin(half)), float(math.cos(half)))

def as_vec3(value) -> Vec3:
    return np.asarray(value, dtype=float).reshape(3)


def coerce_path_points_3d(path_like) -> np.ndarray | None:
    """
    Coerce a polyline-like value into an array of shape (M,3).

    Returns None if the input looks like a single 3D point (shape (3,),
    (1,3), or (3,1)).
    Accepts (M,3) or (3,M) with M>=2.
    """
    arr = np.asarray(path_like, dtype=float)
    if arr.ndim == 1:
        return None
    if arr.ndim != 2:
        raise ValueError(f"path must be 2D array-like; got shape={arr.shape}")
    # Treat a single 3D point provided as a 2D row/column as "not a path".
    if (arr.shape[0] == 1 and arr.shape[1] == 3) or (arr.shape[0] == 3 and arr.shape[1] == 1):
        return None
    if arr.shape[0] == 3 and arr.shape[1] >= 2:
        return arr.T.copy()
    if arr.shape[1] == 3 and arr.shape[0] >= 2:
        return arr.copy()
    raise ValueError(f"path must have shape (M,3) or (3,M) with M>=2; got {arr.shape}")


def is_finite_vec3(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def ned_to_enu(v) -> Vec3:
    v = as_vec3(v)
    return np.array([v[1], v[0], -v[2]], dtype=float)


def enu_to_ned(v) -> Vec3:
    v = as_vec3(v)
    return np.array([v[1], v[0], -v[2]], dtype=float)


def yaw_ned_to_enu(yaw_ned: float) -> float:
    return wrap_pi(math.pi / 2.0 - float(yaw_ned))


def yaw_enu_to_ned(yaw_enu: float) -> float:
    return wrap_pi(math.pi / 2.0 - float(yaw_enu))


def yaw_rate_enu_to_ned(yaw_rate_enu: float) -> float:
    return -float(yaw_rate_enu)


def wrap_pi(angle_rad: float) -> float:
    a = float(angle_rad)
    return math.atan2(math.sin(a), math.cos(a))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def load_waypoints_ned(
    path: Path,
    *,
    origin_ned: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    origin_mode: str = "fixed",
) -> list[Vec3]:
    """
    Load a list of NED waypoints from a text file.

    Each non-empty line may be "x y z" or "x, y, z". Lines starting with # are ignored.
    `origin_ned` is subtracted from each point.

    origin selection:
    - origin_mode="fixed": subtract `origin_ned` (default PRE behavior)
    - origin_mode="first": subtract the first waypoint (first becomes [0,0,0])
    - origin_mode="first_xy": subtract [x0, y0, origin_ned.z] using first waypoint x/y
    """
    waypoints_abs: list[Vec3] = []
    if not path.exists():
        return []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 3:
            continue
        wp = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
        waypoints_abs.append(wp)

    if not waypoints_abs:
        return []

    mode = str(origin_mode).lower().strip()
    if mode == "fixed":
        if origin_ned is None:
            raise ValueError("origin_ned cannot be None when origin_mode='fixed'")
        origin = np.asarray(origin_ned, dtype=float).reshape(3)
    elif mode == "first":
        origin = waypoints_abs[0]
    elif mode == "first_xy":
        if origin_ned is None:
            z0 = 0.0
        else:
            z0 = float(origin_ned[2])
        origin = np.array(
            [float(waypoints_abs[0][0]), float(waypoints_abs[0][1]), z0], dtype=float
        )
    else:
        raise ValueError(f"Unknown origin_mode={origin_mode!r}")

    return [wp - origin for wp in waypoints_abs]


@dataclass(frozen=True)
class Polyline3D:
    """
    A 3D polyline parameterized by arc-length s (meters).
    """

    points: np.ndarray  # (M,3)
    s: np.ndarray  # (M,)
    length: float

    @classmethod
    def from_points(cls, points, *, dedupe: bool = True) -> "Polyline3D":
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"points must be 2D; got shape={arr.shape}")
        if arr.shape[0] == 3 and arr.shape[1] >= 2:
            arr = arr.T
        if arr.shape[1] != 3 or arr.shape[0] < 2:
            raise ValueError(
                f"points must have shape (M,3) or (3,M) with M>=2; got {arr.shape}"
            )
        pts = arr.reshape(-1, 3)
        if dedupe:
            pts = dedupe_consecutive_points(pts)
        if pts.shape[0] < 2:
            raise ValueError("polyline must contain at least 2 distinct points")

        s = polyline_cumlen(pts)
        L = float(s[-1])
        if not np.isfinite(L) or L <= 1e-9:
            raise ValueError("polyline length must be finite and > 0")
        return cls(points=pts, s=s, length=L)

    def closest_s(self, p) -> float:
        return closest_s_on_polyline(self.points, self.s, p)

    def sample(self, s: float) -> Vec3:
        return sample_polyline(self.points, self.s, s)

    def sample_with_tangent(self, s: float) -> tuple[Vec3, Vec3]:
        return sample_polyline_with_tangent(self.points, self.s, s)


def dedupe_consecutive_points(points: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) <= eps:
            continue
        out.append(p)
    return np.asarray(out, dtype=float)


def polyline_cumlen(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def closest_s_on_polyline(points: np.ndarray, cumlen: np.ndarray, p) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cumlen = np.asarray(cumlen, dtype=float).reshape(-1)
    p = as_vec3(p)
    if pts.shape[0] < 2:
        return 0.0

    best_dist = float("inf")
    best_s = 0.0
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        v = b - a
        vv = float(np.dot(v, v))
        if vv < 1e-12:
            t = 0.0
            proj = a
            seg_len = 0.0
        else:
            t = float(np.dot(p - a, v) / vv)
            t = max(0.0, min(1.0, t))
            proj = a + t * v
            seg_len = float(np.linalg.norm(v))
        d = float(np.linalg.norm(p - proj))
        if d < best_dist:
            best_dist = d
            best_s = float(cumlen[i] + t * seg_len)
    return best_s


def sample_polyline(points: np.ndarray, cumlen: np.ndarray, s: float) -> Vec3:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cumlen = np.asarray(cumlen, dtype=float).reshape(-1)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=float)
    if pts.shape[0] == 1:
        return pts[0].copy()

    L = float(cumlen[-1])
    s = float(s)
    if s <= 0.0:
        return pts[0].copy()
    if s >= L:
        return pts[-1].copy()

    i = int(np.searchsorted(cumlen, s, side="right") - 1)
    i = max(0, min(i, int(pts.shape[0] - 2)))
    a = pts[i]
    b = pts[i + 1]
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-12:
        return a.copy()
    ds = s - float(cumlen[i])
    return a + (ds / seg_len) * seg


def sample_polyline_with_tangent(
    points: np.ndarray, cumlen: np.ndarray, s: float
) -> tuple[Vec3, Vec3]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cumlen = np.asarray(cumlen, dtype=float).reshape(-1)
    if pts.shape[0] < 2:
        return (pts[0].copy(), np.array([1.0, 0.0, 0.0], dtype=float))

    L = float(cumlen[-1])
    s = max(0.0, min(float(s), L))
    i = int(np.searchsorted(cumlen, s, side="right") - 1)
    i = max(0, min(i, int(pts.shape[0] - 2)))
    a = pts[i]
    b = pts[i + 1]
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-12:
        return (a.copy(), np.array([1.0, 0.0, 0.0], dtype=float))
    t = seg / seg_len
    ds = s - float(cumlen[i])
    pr = a + (ds / seg_len) * seg
    return (pr, t)


def flatness_to_ctbr(
    a_ref_ned,
    j_ref_ned,
    yaw_ned: float,
    yaw_rate_ned: float = 0.0,
    *,
    hover_thrust: float = 0.55,
    g: float = 9.81,
    thrust_min: float = 0.10,
    thrust_max: float = 0.90,
    eps: float = 1e-6,
) -> tuple[float, float, float, float]:
    """
    Flatness (a, j, yaw) -> CTBR.

    Returns: (p, q, r, thrust_norm)
    """
    a = as_vec3(a_ref_ned)
    j = as_vec3(j_ref_ned)
    yaw = float(yaw_ned)
    yaw_rate = float(yaw_rate_ned)

    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    F = g * e3 - a
    F_norm = float(np.linalg.norm(F) + eps)
    b3 = F / F_norm  # z_B

    thrust_norm = hover_thrust * (F_norm / g)
    thrust_norm = clamp(thrust_norm, thrust_min, thrust_max)

    b1d = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    c = np.cross(b3, b1d)
    c_norm = float(np.linalg.norm(c) + eps)

    b2 = c / c_norm  # y_B
    b1 = np.cross(b2, b3)  # x_B

    I = np.eye(3, dtype=float)
    b3dot = (I - np.outer(b3, b3)) @ (-j) / F_norm  # h_w

    b1d_dot = yaw_rate * np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    c_dot = np.cross(b3dot, b1d) + np.cross(b3, b1d_dot)
    b2dot = (I - np.outer(b2, b2)) @ c_dot / c_norm
    b1dot = np.cross(b2dot, b3) + np.cross(b2, b3dot)

    Omega = np.column_stack((b1, b2, b3)).T @ np.column_stack((b1dot, b2dot, b3dot))
    return (
        float(Omega[2, 1]),
        float(Omega[0, 2]),
        float(Omega[1, 0]),
        float(thrust_norm),
    )


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
