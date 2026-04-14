#!/usr/bin/env python3
"""plan2track: bridge planner outputs into tracker-friendly ROS topics.

This node produces:

- `ref_path_topic` (`nav_msgs/Path`): MPC reference trajectory, `N+1` points (ENU).
- `out_path_topic` (`nav_msgs/Path`): MPCC polyline path (ENU). When a vehicle
  state is available, the path is trimmed to start from the vehicle's current
  progress along it.
- `vehicle_pose_topic` (`geometry_msgs/PoseStamped`): current vehicle pose (ENU).

Inputs:
  - Planner `nav_msgs/Path` (ENU assumed).
  - Optional waypoint file with NED waypoints: one `x y z` per line.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import Path as RosPath
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import Float32

_ROOT = Path(__file__).resolve().parent.parent


def _add_project_to_sys_path() -> None:
    project_paths = (_ROOT, _ROOT / "tracking")
    for project_path in project_paths:
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))


_add_project_to_sys_path()

from tracking_cfg import DEFAULT_CONFIG, TrackingConfig
from tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from tracking_utils import (
    as_vec3,
    closest_s_on_polyline,
    is_finite_vec3,
    load_waypoints_ned,
    ned_to_enu,
    polyline_cumlen,
    sample_polyline,
    wrap_pi,
    yaw_ned_to_enu,
    quat_from_yaw_enu,
)
from yamls.config import get_cfg

_CFG = get_cfg()

_CLOSE_LOOP_EPS_M = 1e-6
_LOOP_WRAP_HIGH = 0.85
_LOOP_WRAP_LOW = 0.15


@dataclass
class PathCache:
    """Cached polyline and its cumulative arc-length.

    Attributes:
      points_enu: Polyline points as an array of shape (M, 3).
      cumlen: Cumulative arc-length along the polyline, shape (M,).
    """

    points_enu: np.ndarray
    cumlen: np.ndarray

    @classmethod
    def from_points(cls, points_enu: np.ndarray) -> "PathCache":
        pts = np.asarray(points_enu, dtype=float).reshape(-1, 3)
        return cls(points_enu=pts, cumlen=polyline_cumlen(pts))


def _path_msg_points_enu(msg: RosPath) -> np.ndarray:
    """Extract ENU points from a `nav_msgs/Path` message.

    Args:
      msg: ROS Path message.

    Returns:
      Array of shape (M, 3). May be empty.
    """
    poses = getattr(msg, "poses", None) or []
    return np.array(
        [
            [
                float(ps.pose.position.x),
                float(ps.pose.position.y),
                float(ps.pose.position.z),
            ]
            for ps in poses
        ],
        dtype=float,
    )


def _close_loop_points(points_enu: np.ndarray) -> np.ndarray:
    """Append the first point as the last point if the path is not closed."""
    pts = np.asarray(points_enu, dtype=float).reshape(-1, 3)
    if int(pts.shape[0]) < 2:
        return pts
    if float(np.linalg.norm(pts[0] - pts[-1])) <= _CLOSE_LOOP_EPS_M:
        return pts
    return np.vstack([pts, pts[0:1]])


def _yaw_enu_from_local_position(msg: VehicleLocalPosition) -> float:
    """Extract yaw in ENU from PX4 VehicleLocalPosition."""
    heading = getattr(msg, "heading", None)
    yaw_ned = float(heading) if heading is not None else 0.0
    if not (heading is not None and np.isfinite(yaw_ned)):
        yaw_ned = 0.0
    return float(yaw_ned_to_enu(wrap_pi(yaw_ned)))


class Plan2TrackNode(Node):
    """Publishes MPC reference and MPCC path from a planner path / waypoint file."""

    def __init__(
        self,
        *,
        cfg: TrackingConfig = DEFAULT_CONFIG,
        path_file: str | Path | None = None,
    ):
        super().__init__("plan2track")
        io_cfg = _CFG.plan2track
        self.cfg = cfg
        self._frame_id = str(io_cfg.frame_id).strip()

        self._origin_mode = str(io_cfg.origin_mode).lower().strip()
        self._loop = bool(io_cfg.loop)
        self._fixed_yaw = bool(io_cfg.fixed_yaw)
        self._init_yaw = float(io_cfg.init_yaw)
        self._path_cache: PathCache | None = None
        self._s_last: float | None = None

        self._have_state = False
        self._pos_enu = np.zeros(3, dtype=float)
        self._yaw_enu = 0.0
        self._yaw_cmd_enu_last = float(self._init_yaw)
        self._yaw_sample_ds_m = max(
            float(self.cfg.mpc.v_ref) * float(self.cfg.control.dt),
            1e-6,
        )

        default_file = self.cfg.tasks.waypoint_path(root=_ROOT)
        cfg_path_file = str(io_cfg.path_file).strip()
        if path_file is not None:
            self._path_file = Path(path_file)
        elif cfg_path_file:
            self._path_file = Path(cfg_path_file)
        else:
            self._path_file = default_file

        self._pub_ref_path = self.create_publisher(RosPath, str(io_cfg.ref_path_topic), 10)
        self._pub_path = self.create_publisher(RosPath, str(io_cfg.out_path_topic), 10)
        self._pub_vehicle_pose = self.create_publisher(
            PoseStamped, str(io_cfg.vehicle_pose_topic), 10
        )
        self._pub_yaw_cmd = self.create_publisher(Float32, str(io_cfg.yaw_cmd_topic), 10)

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )

        self.create_subscription(RosPath, str(io_cfg.path_topic), self._on_path, 10)

        self._load_path_from_file(log=True)
        self.get_logger().info(f"config file: {_CFG.config_path}")
        log_msg = (
            "path_file=%s publish ref_path=%s path=%s frame_id=%s"
            % (
                str(self._path_file),
                str(io_cfg.ref_path_topic),
                str(io_cfg.out_path_topic),
                str(self._frame_id),
            )
        )
        self.get_logger().info(log_msg)
        self.get_logger().info(f"Publishing vehicle_pose: {io_cfg.vehicle_pose_topic}")
        self.get_logger().info(f"Publishing yaw_cmd: {io_cfg.yaw_cmd_topic}")
        self.get_logger().info(
            f"fixed_yaw={self._fixed_yaw} init_yaw={self._init_yaw:.4f} rad"
        )
        self.get_logger().info(f"yaw sample ds: {self._yaw_sample_ds_m:.4f} m")

        self.create_timer(float(self.cfg.control.dt), self._tick)

    def _set_path(self, points_enu: np.ndarray, *, log: bool) -> None:
        """Set the current polyline path in ENU coordinates."""
        pts = _close_loop_points(points_enu) if self._loop else np.asarray(
            points_enu, dtype=float
        ).reshape(-1, 3)
        self._path_cache = PathCache.from_points(pts)
        self._s_last = None
        if log:
            self.get_logger().info(
                f"Loaded path with {self._path_cache.points_enu.shape[0]} points."
            )
        self._pub_path.publish(self._to_ros_path(self._path_cache.points_enu))

    def _progress_s(self, s_closest: float, length: float) -> float:
        """Return a monotonic progress coordinate with optional wrap-around."""
        s_closest = float(s_closest)
        if self._s_last is None:
            return s_closest

        s_prev = float(self._s_last)
        if (
            self._loop
            and s_prev > _LOOP_WRAP_HIGH * float(length)
            and s_closest < _LOOP_WRAP_LOW * float(length)
        ):
            return s_closest
        return max(s_prev, s_closest)

    def _load_path_from_file(self, *, log: bool) -> None:
        """Load an initial path from the waypoint file, if present."""
        pts_ned = load_waypoints_ned(self._path_file, origin_mode=self._origin_mode)
        if len(pts_ned) < 2:
            self._path_cache = None
            self.get_logger().warning(
                "No/short path in %s (len=%d); reference will hold position.",
                str(self._path_file),
                int(len(pts_ned)),
            )
            return
        pts_enu = np.asarray([ned_to_enu(p) for p in pts_ned], dtype=float)
        self._set_path(pts_enu, log=log)

    def _on_path(self, msg: RosPath) -> None:
        pts = _path_msg_points_enu(msg)
        if int(pts.shape[0]) < 2:
            return
        self._set_path(pts, log=False)

    def _to_ros_path(self, pts_enu: np.ndarray) -> RosPath:
        """Convert (M, 3) ENU points to `nav_msgs/Path`."""
        pts = np.asarray(pts_enu, dtype=float).reshape(-1, 3)
        msg = RosPath()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = []
        for x, y, z in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _update_progress(self) -> tuple[float, float] | None:
        """Update and return (s0, length) for the current path."""
        if self._path_cache is None:
            return None

        pts = self._path_cache.points_enu
        cum = self._path_cache.cumlen
        length = float(cum[-1])
        if not np.isfinite(length) or length <= 1e-9:
            self._s_last = 0.0
            return (0.0, length)

        s_closest = float(closest_s_on_polyline(pts, cum, self._pos_enu))
        s0 = float(self._progress_s(s_closest, length))
        self._s_last = float(s0)
        return (s0, length)

    def _build_mpcc_path(self, *, s0: float, length: float) -> np.ndarray:
        """Build an MPCC polyline starting from current progress."""
        if self._path_cache is None:
            return np.zeros((0, 3), dtype=float)

        pts = self._path_cache.points_enu
        cum = self._path_cache.cumlen
        if not np.isfinite(length) or length <= 1e-9:
            return pts[-1:].copy()

        if (length - s0) <= 1e-3 and not self._loop:
            return pts[-1:].copy()

        seg_i = int(np.searchsorted(cum, s0, side="right") - 1)
        seg_i = max(0, min(seg_i, int(pts.shape[0] - 2)))
        p0 = sample_polyline(pts, cum, s0).reshape(1, 3)
        tail = pts[seg_i + 1 :].reshape(-1, 3)
        out = np.vstack([p0, tail])

        if self._loop:
            head = pts[1 : seg_i + 1].reshape(-1, 3)
            if int(head.shape[0]) > 0:
                out = np.vstack([out, head])
        return out

    def _publish_vehicle_pose(self) -> None:
        qx, qy, qz, qw = quat_from_yaw_enu(self._yaw_enu)
        msg = PoseStamped()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self._pos_enu[0])
        msg.pose.position.y = float(self._pos_enu[1])
        msg.pose.position.z = float(self._pos_enu[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self._pub_vehicle_pose.publish(msg)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        pos = (
            getattr(msg, "x", float("nan")),
            getattr(msg, "y", float("nan")),
            getattr(msg, "z", float("nan")),
        )
        if not is_finite_vec3(pos):
            return
        self._pos_enu = ned_to_enu(np.array(pos, dtype=float))
        self._yaw_enu = _yaw_enu_from_local_position(msg)
        self._have_state = True
        self._publish_vehicle_pose()

    def _build_ref_traj(self, *, s0: float, length: float) -> np.ndarray:
        """Build a per-step MPC reference trajectory in ENU coordinates.

        Returns:
          Reference trajectory shaped (3, N+1).
        """
        N = int(self.cfg.mpc.horizon)
        ds = max(0.0, float(self.cfg.mpc.v_ref) * float(self.cfg.mpc.dt))

        if self._path_cache is None or int(self._path_cache.points_enu.shape[0]) < 2:
            p = as_vec3(self._pos_enu)
            return np.repeat(p.reshape(3, 1), N + 1, axis=1)

        pts = self._path_cache.points_enu
        cum = self._path_cache.cumlen
        length = float(cum[-1])

        ref = np.zeros((3, N + 1), dtype=float)
        for k in range(N + 1):
            s_k = float(s0 + ds * k)
            if self._loop and np.isfinite(length) and length > 1e-9:
                s_k = s_k % length
            ref[:, k] = sample_polyline(pts, cum, s_k)
        return ref

    def _compute_yaw_cmd_enu(self, *, s0: float, length: float) -> float:
        if self._fixed_yaw:
            return float(self._init_yaw)

        if self._path_cache is None or int(self._path_cache.points_enu.shape[0]) < 2:
            return float(self._yaw_cmd_enu_last)

        if not np.isfinite(length) or length <= 1e-9:
            return float(self._yaw_cmd_enu_last)

        pts = self._path_cache.points_enu
        cum = self._path_cache.cumlen
        s1 = float(s0 + self._yaw_sample_ds_m)
        if self._loop:
            s1 = s1 % length
        else:
            s1 = min(s1, length)

        p0 = sample_polyline(pts, cum, float(s0))
        p1 = sample_polyline(pts, cum, s1)
        delta = np.asarray(p1 - p0, dtype=float).reshape(3)
        if float(np.linalg.norm(delta[:2])) < 1e-6:
            return float(self._yaw_cmd_enu_last)
        return float(np.arctan2(delta[1], delta[0]))

    def _tick(self) -> None:
        if not self._have_state:
            self.get_logger().warning(
                "No vehicle state received yet; cannot build reference trajectory."
            )
            return

        progress = self._update_progress()
        if progress is None:
            s0 = 0.0
            length = float("nan")
        else:
            s0, length = progress

        yaw_cmd_enu = self._compute_yaw_cmd_enu(s0=s0, length=length)
        self._yaw_cmd_enu_last = float(yaw_cmd_enu)
        yaw_msg = Float32()
        yaw_msg.data = float(yaw_cmd_enu)
        self._pub_yaw_cmd.publish(yaw_msg)

        ref = self._build_ref_traj(s0=s0, length=length)
        self._pub_ref_path.publish(self._to_ros_path(ref.T))

        if self._path_cache is None:
            mpcc_pts = np.zeros((0, 3), dtype=float)
        else:
            mpcc_pts = self._build_mpcc_path(s0=s0, length=length)
        self._pub_path.publish(self._to_ros_path(mpcc_pts))


def main() -> None:
    """Entrypoint."""
    rclpy.init()
    node = Plan2TrackNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
