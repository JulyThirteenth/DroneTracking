#!/usr/bin/env python3
"""Convert ROS2 depth images into 2D obstacle scan points.

Subscribed topics:
  - `depth_topic` (`sensor_msgs/Image`): single-channel depth image.

Published topics:
  - `scan_topic` (`sensor_msgs/LaserScan`): pseudo 2D laser scan.
  - `points_topic` (`sensor_msgs/PointCloud2`): scan endpoints as XYZ points.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

_PERCEPTION_ROOT = Path(__file__).resolve().parent
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

from depth_transform.depth_cfg import (
    Config,
    LaserScanConfig,
    SensorConfig,
    TransformConfig,
)
from depth_transform.depth_ops import depth_layer_scan


def _image_to_depth_array(msg: Image) -> np.ndarray:
    """Convert common ROS depth image encodings to a 2D numpy array."""
    encoding = str(msg.encoding).lower()
    dtype_by_encoding = {
        "32fc1": np.float32,
        "16uc1": np.uint16,
        "mono16": np.uint16,
        "32sc1": np.int32,
    }
    dtype = dtype_by_encoding.get(encoding)
    if dtype is None:
        raise ValueError(
            "Unsupported depth encoding "
            f"{msg.encoding!r}; expected one of {sorted(dtype_by_encoding)}"
        )

    height = int(msg.height)
    width = int(msg.width)
    channels = 1
    itemsize = np.dtype(dtype).itemsize
    row_stride = int(msg.step)
    expected_stride = width * channels * itemsize
    raw = np.frombuffer(msg.data, dtype=dtype)

    if row_stride == expected_stride:
        depth = raw.reshape(height, width)
    else:
        row_items = row_stride // itemsize
        depth = raw.reshape(height, row_items)[:, :width]

    if bool(msg.is_bigendian) != (sys.byteorder == "big"):
        depth = depth.byteswap().newbyteorder()
    return depth.astype(np.float32, copy=False)


class Depth2ScanNode(Node):
    """ROS2 node that publishes pseudo laser-scan obstacle points from depth."""

    def __init__(self) -> None:
        super().__init__("depth2scan")

        self.declare_parameter("depth_topic", "/depth/image")
        self.declare_parameter("scan_topic", "/depth2scan/scan")
        self.declare_parameter("points_topic", "/depth2scan/points")
        self.declare_parameter("config_path", "")
        self.declare_parameter("frame_id", "/map")
        self.declare_parameter("fov_x_deg", 90.0)
        self.declare_parameter("fov_y_deg", 90.0)
        self.declare_parameter("dist_scale", 1.0)
        self.declare_parameter("coordinate_system", "opengl")
        self.declare_parameter("height", float("nan"))
        self.declare_parameter("aggregation", "min")
        self.declare_parameter("n_intervals", 90)
        self.declare_parameter("default_value", 3.0)
        self.declare_parameter("queue_size", 10)

        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._points_topic = str(self.get_parameter("points_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        queue_size = int(self.get_parameter("queue_size").value)
        self._depth_count = 0
        self._scan_count = 0
        self._last_status_depth_count = 0
        self._last_status_scan_count = 0

        self._cfg = self._load_config_from_params()
        height = float(self.get_parameter("height").value)
        self._height = None if not math.isfinite(height) else height

        self._pub_scan = self.create_publisher(LaserScan, self._scan_topic, queue_size)
        self._pub_points = self.create_publisher(
            PointCloud2, self._points_topic, queue_size
        )
        self._sub_depth = self.create_subscription(
            Image,
            self._depth_topic,
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0, self._log_status)

        self.get_logger().info(
            "depth2scan subscribed to "
            f"{self._depth_topic}, publishing {self._scan_topic} and {self._points_topic}"
        )

    def _log_status(self) -> None:
        depth_publishers = self.count_publishers(self._depth_topic)
        scan_subscribers = self.count_subscribers(self._scan_topic)
        depth_stalled = self._depth_count == self._last_status_depth_count
        scan_stalled = self._scan_count == self._last_status_scan_count
        if depth_publishers == 0 or depth_stalled or scan_stalled:
            self.get_logger().warn(
                "depth2scan status: "
                f"received={self._depth_count}, published={self._scan_count}, "
                f"depth_publishers={depth_publishers}, "
                f"scan_subscribers={scan_subscribers}"
            )
        self._last_status_depth_count = self._depth_count
        self._last_status_scan_count = self._scan_count

    def _load_config_from_params(self) -> Config:
        config_path = str(self.get_parameter("config_path").value).strip()
        if config_path:
            cfg = Config.from_yaml(config_path)
            self.get_logger().info(f"Loaded depth config from {config_path}")
            return cfg

        return Config(
            coordinate_system=str(
                self.get_parameter("coordinate_system").value
            ).lower(),
            sensor=SensorConfig(
                fov_deg=(
                    float(self.get_parameter("fov_x_deg").value),
                    float(self.get_parameter("fov_y_deg").value),
                ),
                dist_scale=float(self.get_parameter("dist_scale").value),
            ),
            transform=TransformConfig(
                rotate_points=[],
                filter_points=[],
            ),
            laserscan=LaserScanConfig(
                aggregation=str(self.get_parameter("aggregation").value),
                n_intervals=int(self.get_parameter("n_intervals").value),
                default_value=float(self.get_parameter("default_value").value),
            ),
        )

    def _on_depth(self, msg: Image) -> None:
        self._depth_count += 1
        try:
            depth = _image_to_depth_array(msg)
            x_coord, y_coord, angles, dist = depth_layer_scan(
                depth,
                height=self._height,
                cfg=self._cfg,
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        frame_id = self._frame_id or msg.header.frame_id
        if angles is None or dist is None:
            angles = self._scan_angles()
            dist = np.full_like(
                angles, self._cfg.laserscan.default_value, dtype=np.float32
            )
        else:
            angles = self._scan_angles()
        range_max = float(self._cfg.laserscan.default_value)
        valid = np.isfinite(dist) & (dist > 0.0) & (dist < range_max)
        x_coord = dist[valid] * np.cos(angles[valid])
        y_coord = dist[valid] * np.sin(angles[valid])

        stamp = self.get_clock().now().to_msg()
        self._pub_scan.publish(self._to_laser_scan(stamp, frame_id, angles, dist))
        self._pub_points.publish(
            self._to_point_cloud(stamp, frame_id, x_coord, y_coord)
        )
        self._scan_count += 1

    def _scan_angles(self) -> np.ndarray:
        fov_x = float(self._cfg.sensor.fov_deg[0])
        return np.linspace(
            math.radians(-fov_x / 2.0),
            math.radians(fov_x / 2.0),
            int(self._cfg.laserscan.n_intervals),
            dtype=np.float32,
        )

    def _to_laser_scan(
        self,
        stamp,
        frame_id: str,
        angles: np.ndarray,
        dist: np.ndarray,
    ) -> LaserScan:
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = frame_id
        scan.range_min = 0.05
        scan.range_max = float(self._cfg.laserscan.default_value)
        ranges = np.asarray(dist, dtype=np.float32)
        angles = self._scan_angles()
        if ranges.shape[0] != angles.shape[0]:
            fixed = np.full(angles.shape, np.inf, dtype=np.float32)
            count = min(fixed.shape[0], ranges.shape[0])
            fixed[:count] = ranges[:count]
            ranges = fixed
        ranges = np.where(ranges >= scan.range_max, np.inf, ranges)
        scan.ranges = ranges.tolist()

        if len(angles) >= 2:
            scan.angle_min = float(angles[0])
            scan.angle_max = float(angles[-1])
            scan.angle_increment = float(angles[1] - angles[0])
        elif len(angles) == 1:
            scan.angle_min = float(angles[0])
            scan.angle_max = float(angles[0])
            scan.angle_increment = 0.0
        return scan

    def _to_point_cloud(
        self,
        stamp,
        frame_id: str,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
    ) -> PointCloud2:
        points = np.column_stack(
            [
                np.asarray(x_coord, dtype=np.float32),
                np.asarray(y_coord, dtype=np.float32),
                np.zeros_like(x_coord, dtype=np.float32),
            ]
        )
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return point_cloud2.create_cloud_xyz32(header, points.tolist())


def main() -> None:
    rclpy.init()
    node = Depth2ScanNode()
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
