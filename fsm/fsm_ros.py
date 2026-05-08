"""ROS message conversion helpers for the FSM node."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from tracking.tracking_utils import (
    is_finite_vec3,
    ned_to_enu,
    wrap_pi,
    yaw_ned_to_enu,
)


@dataclass
class VehicleState:
    """Vehicle state in ENU coordinates."""

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    accel_enu: np.ndarray
    yaw_enu: float


def latched_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
    )


def derive_info_topic(state_topic: str) -> str:
    topic = str(state_topic).strip()
    if topic.endswith("/state"):
        return f"{topic[:-6]}/info"
    return f"{topic}/info"


def points_enu(data) -> np.ndarray:
    return np.asarray(data, dtype=float).reshape(-1, 3)


def path_msg_points_enu(msg: NavPath) -> np.ndarray:
    poses = getattr(msg, "poses", None) or []
    return points_enu(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z] for ps in poses]
    )


def vehicle_state_from_local_position(
    msg: VehicleLocalPosition,
) -> VehicleState | None:
    heading = getattr(msg, "heading", None)
    yaw_ned = float(heading) if heading is not None else 0.0
    if not (heading is not None and np.isfinite(yaw_ned)):
        yaw_ned = 0.0
    yaw_ned = wrap_pi(yaw_ned)

    pos_ned = (
        getattr(msg, "x", float("nan")),
        getattr(msg, "y", float("nan")),
        getattr(msg, "z", float("nan")),
    )
    vel_ned = (
        getattr(msg, "vx", float("nan")),
        getattr(msg, "vy", float("nan")),
        getattr(msg, "vz", float("nan")),
    )
    acc_ned = (
        getattr(msg, "ax", float("nan")),
        getattr(msg, "ay", float("nan")),
        getattr(msg, "az", float("nan")),
    )
    if not (
        is_finite_vec3(pos_ned) and is_finite_vec3(vel_ned) and is_finite_vec3(acc_ned)
    ):
        return None

    return VehicleState(
        position_enu=ned_to_enu(pos_ned),
        velocity_enu=ned_to_enu(vel_ned),
        accel_enu=ned_to_enu(acc_ned),
        yaw_enu=yaw_ned_to_enu(yaw_ned),
    )


def scan_msg_points_enu(
    msg: LaserScan,
    state: VehicleState,
    *,
    camera_xyz_body: np.ndarray,
    min_radius_m: float,
) -> np.ndarray | None:
    ranges = np.asarray(msg.ranges, dtype=float)
    if ranges.size == 0:
        return None

    angles = float(msg.angle_min) + np.arange(ranges.size) * float(msg.angle_increment)
    valid = np.isfinite(ranges) & (ranges > float(msg.range_min))
    if np.isfinite(float(msg.range_max)):
        valid &= ranges < float(msg.range_max)
    if not np.any(valid):
        return None

    ranges = ranges[valid]
    angles = angles[valid]
    cam = np.asarray(camera_xyz_body, dtype=float).reshape(3)
    x_body = cam[0] + ranges * np.cos(angles)
    y_body = cam[1] + ranges * np.sin(angles)
    z_body = np.full_like(x_body, cam[2])

    radius = np.sqrt(x_body * x_body + y_body * y_body + z_body * z_body)
    keep = radius > float(min_radius_m)
    if not np.any(keep):
        return None

    x_body = x_body[keep]
    y_body = y_body[keep]
    z_body = z_body[keep]
    pos = np.asarray(state.position_enu, dtype=float).reshape(3)
    yaw = float(state.yaw_enu)
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.column_stack(
        (
            pos[0] + c * x_body - s * y_body,
            pos[1] + s * x_body + c * y_body,
            pos[2] + z_body,
        )
    )
