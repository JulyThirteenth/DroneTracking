import math
import numpy as np
from scipy.spatial.transform import Rotation


def vec3_or_none(value):
    return np.asarray(value, dtype=float).reshape(3) if value is not None else None


def is_finite_vec3(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


# ----------------------------
# math helpers
# ----------------------------
def quat_to_euler_zyx(w: float, x: float, y: float, z: float):
    """Quaternion -> Euler ZYX (roll, pitch, yaw) in radians."""
    yaw, pitch, roll = Rotation.from_quat([x, y, z, w]).as_euler("ZYX", degrees=False)
    return roll, pitch, yaw


def rot_from_quat(w: float, x: float, y: float, z: float):
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ----------------------------
# Enu <->Ned helpers
# ----------------------------


def ned_to_enu(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]])


def enu_to_ned(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]])


def yaw_ned_to_enu(yaw_ned: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw_ned)


def yaw_enu_to_ned(yaw_enu: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw_enu)


def yaw_rate_enu_to_ned(yaw_rate_enu: float) -> float:
    return -yaw_rate_enu


# ----------------------------
# Three-order interger helpers
# ----------------------------


def uav_jerk_discrete_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete triple-integrator for x=[p,v,a], u=jerk (all 3D), held constant on [k,k+1):
      p+ = p + dt v + 0.5 dt^2 a + (1/6) dt^3 u
      v+ = v + dt a + 0.5 dt^2 u
      a+ = a + dt u
    """
    dt = float(dt)
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    A = np.block(
        [
            [I3, dt * I3, 0.5 * dt * dt * I3],
            [Z3, I3, dt * I3],
            [Z3, Z3, I3],
        ]
    )
    B = np.block(
        [
            [(dt**3) / 6.0 * I3],
            [0.5 * dt * dt * I3],
            [dt * I3],
        ]
    )
    return A, B
