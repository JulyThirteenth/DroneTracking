"""Shared ROS path, vehicle-state, pose, and yaw processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import (
    get_package_share_directory,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Float32

from drone_ref.core import PathProgress, ReferencePath
from drone_ref.util import (
    as_vec3,
    load_waypoints_ned,
    ned_to_enu,
    quat_from_yaw_enu,
    wrap_pi,
    yaw_ned_to_enu,
)

PX4_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def run_node(node_factory: Callable[[], Node], args=None) -> None:
    """Initialize ROS, spin one node, and shut it down cleanly."""
    rclpy.init(args=args)
    node = node_factory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class DroneRefBase(Node, ABC):
    """Common path loading, vehicle state and yaw processing."""

    def __init__(self, *, node_name: str) -> None:
        super().__init__(node_name)

        self._declare_parameters()

        self._reference_path = ReferencePath(loop=self._loop)
        self._path_revision = 0
        self._position_enu: np.ndarray | None = None
        self._yaw_enu = self._initial_yaw
        self._yaw_command_enu = self._initial_yaw
        self._warned_no_state = False

        self._yaw_publisher = self.create_publisher(
            Float32,
            self._yaw_topic,
            10,
        )
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self._vehicle_pose_topic,
            10,
        )

        self._vehicle_subscription = self.create_subscription(
            VehicleLocalPosition,
            self._vehicle_position_topic,
            self._on_vehicle_position,
            PX4_SENSOR_QOS,
        )
        self._path_subscription = self.create_subscription(
            NavPath,
            self._planning_path_topic,
            self._on_planning_path,
            10,
        )

        self._load_path_file()

        self._timer = self.create_timer(
            self._publish_period,
            self._on_timer,
        )

    @property
    def path_revision(self) -> int:
        """Monotonically increasing reference-path version."""
        return self._path_revision

    @property
    def reference_path(self) -> ReferencePath:
        return self._reference_path

    @property
    def position_enu(self) -> np.ndarray:
        if self._position_enu is None:
            raise RuntimeError("Vehicle position has not been received")

        return self._position_enu.copy()

    @abstractmethod
    def _publish_reference(
        self,
        progress: PathProgress | None,
    ) -> None:
        """Publish the controller-specific reference."""
        raise NotImplementedError

    def _publish_without_vehicle_state(self) -> None:
        """Publish outputs that do not require vehicle state, if any."""

    def _param(self, name: str, default):
        return self.declare_parameter(name, default).value

    def _declare_parameters(self) -> None:
        self._frame_id = str(self._param("frame_id", "map"))
        self._path_file = str(self._param("path_file", ""))
        self._origin_mode = str(self._param("origin_mode", "first_xy"))
        self._loop = bool(self._param("loop", False))
        self._publish_period = float(self._param("publish_period", 0.02))
        self._fixed_yaw = bool(self._param("fixed_yaw", False))
        self._initial_yaw = float(self._param("initial_yaw", 0.0))
        self._yaw_lookahead = float(self._param("yaw_lookahead", 0.2))
        self._vehicle_position_topic = str(
            self._param(
                "vehicle_position_topic",
                "/fmu/out/vehicle_local_position",
            )
        )
        self._planning_path_topic = str(
            self._param("planning_path_topic", "/planning/path")
        )
        self._vehicle_pose_topic = str(
            self._param("vehicle_pose_topic", "/tracking/vehicle_pose")
        )
        self._yaw_topic = str(self._param("yaw_topic", "/planning/yaw_cmd_enu"))

        if self._publish_period <= 0.0:
            raise ValueError("publish_period must be positive")

    def _load_path_file(self) -> None:
        if not self._path_file.strip():
            self.get_logger().warning(
                "No path_file configured; waiting for " f"{self._planning_path_topic}"
            )
            return

        path = Path(self._path_file).expanduser()

        if not path.is_absolute():
            package_share = Path(get_package_share_directory("drone_ref"))
            path = package_share / path

        points_ned = load_waypoints_ned(
            path,
            origin_mode=self._origin_mode,
        )

        if len(points_ned) < 2:
            self.get_logger().warning(
                f"Path file contains fewer than two points: {path}"
            )
            return

        points_enu = ned_to_enu(points_ned)

        self._reference_path.set_path(points_enu)
        self._path_revision += 1

        self.get_logger().info(f"Loaded {points_enu.shape[0]} path points from {path}")

    def _on_planning_path(self, message: NavPath) -> None:
        points = np.array(
            [
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ]
                for pose in message.poses
            ],
            dtype=float,
        )

        if points.shape[0] < 2:
            self.get_logger().warning(
                "Rejected planning path with fewer than two points"
            )
            return

        try:
            self._reference_path.set_path(points)
            self._path_revision += 1
            self.get_logger().info(
                "Accepted planning path with "
                f"{points.shape[0]} points, revision {self._path_revision}"
            )
        except ValueError as error:
            self.get_logger().error(f"Rejected planning path: {error}")

    def _on_vehicle_position(
        self,
        message: VehicleLocalPosition,
    ) -> None:
        try:
            position_ned = as_vec3((message.x, message.y, message.z))
        except ValueError:
            return

        self._position_enu = np.asarray(
            ned_to_enu(position_ned),
            dtype=float,
        ).reshape(3)

        if np.isfinite(message.heading):
            self._yaw_enu = yaw_ned_to_enu(wrap_pi(float(message.heading)))

        self._warned_no_state = False
        self._publish_vehicle_pose()

    def _on_timer(self) -> None:
        if self._position_enu is None:
            self._publish_without_vehicle_state()
            if not self._warned_no_state:
                self.get_logger().warning("Waiting for VehicleLocalPosition")
                self._warned_no_state = True
            return

        progress = self._reference_path.update_progress(self._position_enu)

        self._publish_yaw(progress)
        self._publish_reference(progress)

    def _publish_yaw(
        self,
        progress: PathProgress | None,
    ) -> None:
        if self._fixed_yaw:
            self._yaw_command_enu = wrap_pi(self._initial_yaw)
        else:
            self._yaw_command_enu = self._reference_path.yaw_reference(
                progress=progress,
                lookahead_distance=self._yaw_lookahead,
                previous_yaw=self._yaw_command_enu,
            )

        self._yaw_publisher.publish(Float32(data=float(self._yaw_command_enu)))

    def _publish_vehicle_pose(self) -> None:
        if self._position_enu is None:
            return

        qx, qy, qz, qw = quat_from_yaw_enu(self._yaw_enu)

        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id

        message.pose.position.x = float(self._position_enu[0])
        message.pose.position.y = float(self._position_enu[1])
        message.pose.position.z = float(self._position_enu[2])

        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw

        self._pose_publisher.publish(message)

    def _make_path_message(
        self,
        points_enu: np.ndarray,
    ) -> NavPath:
        points = np.asarray(
            points_enu,
            dtype=float,
        ).reshape(-1, 3)

        message = NavPath()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id

        for point in points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)

        return message
