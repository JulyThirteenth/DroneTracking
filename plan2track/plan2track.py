#!/usr/bin/env python3
"""Bridge planner / waypoint paths into tracker-friendly ROS topics.

Published topics:
  - `ref_path_topic` (`nav_msgs/Path`): MPC reference trajectory, `N+1` ENU points.
  - `out_path_topic` (`nav_msgs/Path`): MPCC polyline path in ENU.
  - `vehicle_pose_topic` (`geometry_msgs/PoseStamped`): current vehicle pose in ENU.
  - `yaw_cmd_topic` (`std_msgs/Float32`): yaw command in ENU radians.

Inputs:
  - Planner `nav_msgs/Path`, assumed ENU.
  - Optional waypoint file with NED waypoints, one `x y z` per line.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _add_project_to_sys_path() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


_add_project_to_sys_path()

from tracking.tracking_cfg import DEFAULT_CONFIG, TrackingConfig
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from tracking.tracking_utils import (
    as_vec3,
    closest_s_on_polyline,
    is_finite_vec3,
    load_waypoints_ned,
    ned_to_enu,
    polyline_cumlen,
    quat_from_yaw_enu,
    sample_polyline,
    wrap_pi,
    yaw_ned_to_enu,
)
from yamls.config import Plan2TrackConfig, get_cfg

_PROJECT_CFG = get_cfg()

_CLOSE_LOOP_EPS_M = 1e-6
_END_EPS_M = 1e-3
_LOOP_WRAP_HIGH = 0.85
_LOOP_WRAP_LOW = 0.15
_MIN_PATH_LENGTH_M = 1e-6
_MIN_YAW_DELTA_M = 1e-6


@dataclass(frozen=True)
class PathProgress:
    """Current progress on a polyline."""

    s0: float
    length: float


@dataclass(frozen=True)
class PathCache:
    """Cached ENU polyline and cumulative arc-length."""

    points_enu: np.ndarray
    cumlen: np.ndarray

    @classmethod
    def from_points(cls, points_enu: np.ndarray) -> "PathCache":
        points = _as_points_enu(points_enu)
        return cls(points_enu=points, cumlen=polyline_cumlen(points))

    @property
    def length(self) -> float:
        if int(self.cumlen.shape[0]) < 1:
            return 0.0
        return float(self.cumlen[-1])

    @property
    def point_count(self) -> int:
        return int(self.points_enu.shape[0])

    @property
    def has_segments(self) -> bool:
        return self.point_count >= 2 and np.isfinite(self.length) and self.length > 0.0


class PathReferenceBuilder:
    """Owns path cache, progress tracking, reference generation and yaw sampling."""

    def __init__(self, *, loop: bool):
        self._loop = bool(loop)
        self._cache: PathCache | None = None
        self._last_s: float | None = None

    def clear(self) -> None:
        self._cache = None
        self._last_s = None

    def set_path(self, points_enu: np.ndarray) -> PathCache:
        if self._loop:
            points = _close_loop_points(points_enu)
        else:
            points = _as_points_enu(points_enu)
        self._cache = PathCache.from_points(points)
        self._last_s = None
        return self._cache

    def update_progress(self, position_enu: np.ndarray) -> PathProgress | None:
        if self._cache is None:
            return None

        length = self._cache.length
        if not np.isfinite(length) or length <= _MIN_PATH_LENGTH_M:
            self._last_s = 0.0
            return PathProgress(s0=0.0, length=length)

        s_closest = float(
            closest_s_on_polyline(
                self._cache.points_enu,
                self._cache.cumlen,
                as_vec3(position_enu),
            )
        )
        s0 = self._monotonic_progress(s_closest, length)
        self._last_s = float(s0)
        return PathProgress(s0=s0, length=length)

    def build_ref_traj(
        self,
        *,
        progress: PathProgress | None,
        position_enu: np.ndarray,
        horizon: int,
        ds: float,
    ) -> np.ndarray:
        """Build an MPC reference trajectory shaped `(3, N+1)`."""
        sample_count = max(1, int(horizon) + 1)
        if self._cache is None or not self._cache.has_segments or progress is None:
            point = as_vec3(position_enu)
            return np.repeat(point.reshape(3, 1), sample_count, axis=1)

        ref = np.zeros((3, sample_count), dtype=float)
        for k in range(sample_count):
            s_k = float(progress.s0 + max(0.0, float(ds)) * k)
            ref[:, k] = self._sample(s_k, progress.length)
        return ref

    def build_mpcc_path(self, progress: PathProgress | None) -> np.ndarray:
        """Build an MPCC polyline starting from current progress."""
        if self._cache is None:
            return np.zeros((0, 3), dtype=float)
        if progress is None:
            return self._cache.points_enu.copy()
        if not self._cache.has_segments:
            return self._cache.points_enu[-1:].copy()
        if not self._loop and (progress.length - progress.s0) <= _END_EPS_M:
            return self._cache.points_enu[-1:].copy()

        points = self._cache.points_enu
        cumlen = self._cache.cumlen
        seg_idx = int(np.searchsorted(cumlen, progress.s0, side="right") - 1)
        seg_idx = max(0, min(seg_idx, points.shape[0] - 2))

        current_point = sample_polyline(points, cumlen, progress.s0).reshape(1, 3)
        tail = points[seg_idx + 1 :].reshape(-1, 3)
        output = np.vstack([current_point, tail])

        if self._loop:
            head = points[1 : seg_idx + 1].reshape(-1, 3)
            if int(head.shape[0]) > 0:
                output = np.vstack([output, head])
        return output

    def yaw_cmd_enu(
        self,
        *,
        progress: PathProgress | None,
        sample_ds_m: float,
        fixed_yaw: bool,
        init_yaw: float,
        last_yaw: float,
    ) -> float:
        if fixed_yaw:
            return float(init_yaw)
        if self._cache is None or not self._cache.has_segments or progress is None:
            return float(last_yaw)
        if not np.isfinite(progress.length) or progress.length <= _MIN_PATH_LENGTH_M:
            return float(last_yaw)

        p0 = self._sample(progress.s0, progress.length)
        p1 = self._sample(progress.s0 + max(float(sample_ds_m), 0.0), progress.length)
        delta = np.asarray(p1 - p0, dtype=float).reshape(3)
        if float(np.linalg.norm(delta[:2])) < _MIN_YAW_DELTA_M:
            return float(last_yaw)
        yaw = float(np.arctan2(delta[1], delta[0]))
        return float(last_yaw + wrap_pi(yaw - float(last_yaw)))

    def _monotonic_progress(self, s_closest: float, length: float) -> float:
        if self._last_s is None:
            return float(s_closest)

        last_s = float(self._last_s)
        wrapped_to_start = (
            self._loop
            and last_s > _LOOP_WRAP_HIGH * float(length)
            and s_closest < _LOOP_WRAP_LOW * float(length)
        )
        if wrapped_to_start:
            return float(s_closest)
        return float(max(last_s, s_closest))

    def _sample(self, s: float, length: float) -> np.ndarray:
        assert self._cache is not None
        sample_s = float(s)
        if self._loop and np.isfinite(length) and length > _MIN_PATH_LENGTH_M:
            sample_s = sample_s % length
        else:
            sample_s = min(sample_s, length)
        return sample_polyline(self._cache.points_enu, self._cache.cumlen, sample_s)


def _as_points_enu(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=float).reshape(-1, 3)


def _path_msg_points_enu(msg: NavPath) -> np.ndarray:
    """Extract ENU points from a `nav_msgs/Path` message."""
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
    points = _as_points_enu(points_enu)
    if int(points.shape[0]) < 2:
        return points
    if float(np.linalg.norm(points[0] - points[-1])) <= _CLOSE_LOOP_EPS_M:
        return points
    return np.vstack([points, points[0:1]])


def _yaw_enu_from_local_position(msg: VehicleLocalPosition) -> float:
    """Extract yaw in ENU from PX4 VehicleLocalPosition."""
    heading = getattr(msg, "heading", None)
    yaw_ned = float(heading) if heading is not None else 0.0
    if not (heading is not None and np.isfinite(yaw_ned)):
        yaw_ned = 0.0
    return float(yaw_ned_to_enu(wrap_pi(yaw_ned)))


def _resolve_path_file(
    *,
    io_cfg: Plan2TrackConfig,
    tracking_cfg: TrackingConfig,
    override: str | Path | None,
) -> Path:
    if override is not None:
        return Path(override)

    cfg_path_file = str(io_cfg.path_file).strip()
    if cfg_path_file:
        return Path(cfg_path_file)

    return tracking_cfg.tasks.waypoint_path(root=_PROJECT_ROOT)


class Plan2TrackNode(Node):
    """Publishes MPC reference and MPCC path from a planner path / waypoint file."""

    def __init__(
        self,
        *,
        cfg: TrackingConfig = DEFAULT_CONFIG,
        path_file: str | Path | None = None,
    ):
        super().__init__("plan2track")

        self._tracking_cfg = cfg
        self._io_cfg = _PROJECT_CFG.plan2track
        self._frame_id = str(self._io_cfg.frame_id).strip()
        self._origin_mode = str(self._io_cfg.origin_mode).lower().strip()
        self._path_file = _resolve_path_file(
            io_cfg=self._io_cfg,
            tracking_cfg=self._tracking_cfg,
            override=path_file,
        )

        self._path_builder = PathReferenceBuilder(loop=bool(self._io_cfg.loop))
        self._have_state = False
        self._position_enu = np.zeros(3, dtype=float)
        self._yaw_enu = 0.0
        self._yaw_cmd_enu_last = float(self._io_cfg.init_yaw)
        self._yaw_sample_ds_m = max(
            float(self._tracking_cfg.mpc.v_ref) * float(self._tracking_cfg.mpc.dt),
            1e-6,
        )

        self._create_publishers()
        self._create_subscribers()

        self._load_path_from_file(log=True)
        self._log_startup()
        self.create_timer(float(self._tracking_cfg.control.dt), self._tick)

    def _create_publishers(self) -> None:
        self._pub_ref_path = self.create_publisher(
            NavPath,
            str(self._io_cfg.ref_path_topic),
            10,
        )
        self._pub_path = self.create_publisher(
            NavPath,
            str(self._io_cfg.out_path_topic),
            10,
        )
        self._pub_vehicle_pose = self.create_publisher(
            PoseStamped,
            str(self._io_cfg.vehicle_pose_topic),
            10,
        )
        self._pub_yaw_cmd = self.create_publisher(
            Float32,
            str(self._io_cfg.yaw_cmd_topic),
            10,
        )

    def _create_subscribers(self) -> None:
        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )
        self.create_subscription(
            NavPath,
            str(self._io_cfg.path_topic),
            self._on_path,
            10,
        )

    def _log_startup(self) -> None:
        self.get_logger().info(f"config file: {_PROJECT_CFG.config_path}")
        self.get_logger().info(
            "path_file=%s publish ref_path=%s path=%s frame_id=%s"
            % (
                str(self._path_file),
                str(self._io_cfg.ref_path_topic),
                str(self._io_cfg.out_path_topic),
                str(self._frame_id),
            )
        )
        self.get_logger().info(
            f"Publishing vehicle_pose: {self._io_cfg.vehicle_pose_topic}"
        )
        self.get_logger().info(f"Publishing yaw_cmd: {self._io_cfg.yaw_cmd_topic}")
        self.get_logger().info(
            "fixed_yaw=%s init_yaw=%.4f rad"
            % (bool(self._io_cfg.fixed_yaw), float(self._io_cfg.init_yaw))
        )
        self.get_logger().info(f"yaw sample ds: {self._yaw_sample_ds_m:.4f} m")

    def _load_path_from_file(self, *, log: bool) -> None:
        """Load an initial path from the waypoint file, if present."""
        points_ned = load_waypoints_ned(self._path_file, origin_mode=self._origin_mode)
        if len(points_ned) < 2:
            self._path_builder.clear()
            self.get_logger().warning(
                "No/short path in %s (len=%d); reference will hold position.",
                str(self._path_file),
                int(len(points_ned)),
            )
            return

        points_enu = np.asarray(
            [ned_to_enu(point) for point in points_ned],
            dtype=float,
        )
        self._set_path(points_enu, log=log)

    def _set_path(self, points_enu: np.ndarray, *, log: bool) -> None:
        """Set the current polyline path in ENU coordinates."""
        path_cache = self._path_builder.set_path(points_enu)
        if log:
            self.get_logger().info(
                f"Loaded path with {path_cache.points_enu.shape[0]} points."
            )
        self._pub_path.publish(self._to_ros_path(path_cache.points_enu))

    def _on_path(self, msg: NavPath) -> None:
        points_enu = _path_msg_points_enu(msg)
        if int(points_enu.shape[0]) < 2:
            return
        self._set_path(points_enu, log=False)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        position_ned = (
            getattr(msg, "x", float("nan")),
            getattr(msg, "y", float("nan")),
            getattr(msg, "z", float("nan")),
        )
        if not is_finite_vec3(position_ned):
            return

        self._position_enu = ned_to_enu(np.array(position_ned, dtype=float))
        self._yaw_enu = _yaw_enu_from_local_position(msg)
        self._have_state = True
        self._publish_vehicle_pose()

    def _tick(self) -> None:
        if not self._have_state:
            self.get_logger().warning(
                "No vehicle state received yet; cannot build reference trajectory."
            )
            return

        progress = self._path_builder.update_progress(self._position_enu)
        self._publish_yaw_cmd(progress)
        self._publish_ref_path(progress)
        self._publish_mpcc_path(progress)

    def _publish_yaw_cmd(self, progress: PathProgress | None) -> None:
        yaw_cmd_enu = self._path_builder.yaw_cmd_enu(
            progress=progress,
            sample_ds_m=self._yaw_sample_ds_m,
            fixed_yaw=bool(self._io_cfg.fixed_yaw),
            init_yaw=float(self._io_cfg.init_yaw),
            last_yaw=float(self._yaw_cmd_enu_last),
        )
        self._yaw_cmd_enu_last = float(yaw_cmd_enu)

        msg = Float32()
        msg.data = float(yaw_cmd_enu)
        self._pub_yaw_cmd.publish(msg)

    def _publish_ref_path(self, progress: PathProgress | None) -> None:
        ref_traj_enu = self._path_builder.build_ref_traj(
            progress=progress,
            position_enu=self._position_enu,
            horizon=int(self._tracking_cfg.mpc.horizon),
            ds=float(self._tracking_cfg.mpc.v_ref) * float(self._tracking_cfg.mpc.dt),
        )
        self._pub_ref_path.publish(self._to_ros_path(ref_traj_enu.T))

    def _publish_mpcc_path(self, progress: PathProgress | None) -> None:
        path_points_enu = self._path_builder.build_mpcc_path(progress)
        self._pub_path.publish(self._to_ros_path(path_points_enu))

    def _publish_vehicle_pose(self) -> None:
        qx, qy, qz, qw = quat_from_yaw_enu(self._yaw_enu)
        msg = PoseStamped()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self._position_enu[0])
        msg.pose.position.y = float(self._position_enu[1])
        msg.pose.position.z = float(self._position_enu[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self._pub_vehicle_pose.publish(msg)

    def _to_ros_path(self, points_enu: np.ndarray) -> NavPath:
        """Convert `(M, 3)` ENU points to `nav_msgs/Path`."""
        points = _as_points_enu(points_enu)
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = []

        for x, y, z in points:
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = float(z)
            pose_stamped.pose.orientation.w = 1.0
            msg.poses.append(pose_stamped)

        return msg


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
