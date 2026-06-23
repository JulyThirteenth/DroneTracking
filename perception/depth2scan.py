from __future__ import annotations
import math
import sys
from pathlib import Path
from copy import deepcopy
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
from depth_transform.depth_ops import depth_layer_scan, depth_to_filtered_pointcloud


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
        self.declare_parameter("depth_downsample_factor", 8)
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
        self._depth_downsample_factor = max(
            int(self.get_parameter("depth_downsample_factor").value), 1
        )

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
            depth = self._downsample_depth_min(depth)
            _, _, angles, dist = depth_layer_scan(
                depth,
                height=self._height,
                cfg=self._cfg,
            )
            pts_cfg = deepcopy(self._cfg)
            pts_cfg.transform.filter_points = []
            pts, _ = depth_to_filtered_pointcloud(
                depth,
                height=self._height,
                cfg=pts_cfg,
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

        stamp = msg.header.stamp
        self._pub_scan.publish(self._to_laser_scan(stamp, frame_id, angles, dist))
        self._pub_points.publish(self._to_point_cloud(stamp, frame_id, pts))
        self._scan_count += 1

    def _downsample_depth_min(self, depth: np.ndarray) -> np.ndarray:
        factor = int(self._depth_downsample_factor)
        if factor <= 1:
            return depth

        h, w = depth.shape
        h2 = (h // factor) * factor
        w2 = (w // factor) * factor
        if h2 == 0 or w2 == 0:
            return depth

        depth_crop = depth[:h2, :w2].astype(np.float32, copy=False)
        valid = np.isfinite(depth_crop) & (depth_crop > 0.0)
        safe = np.where(valid, depth_crop, np.inf)
        pooled = safe.reshape(h2 // factor, factor, w2 // factor, factor).min(
            axis=(1, 3)
        )
        pooled[~np.isfinite(pooled)] = 0.0
        return pooled.astype(np.float32, copy=False)

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

    def _opengl_points_to_body(self, pts_cam):
        pts_cam = np.asarray(pts_cam, dtype=np.float32).reshape(-1, 3)
        x_cam = pts_cam[:, 0]
        y_cam = pts_cam[:, 1]
        z_cam = pts_cam[:, 2]

        x_body = -z_cam
        y_body = -x_cam
        z_body = y_cam
        return np.column_stack((x_body, y_body, z_body)).astype(np.float32)

    def _to_point_cloud(
        self,
        stamp,
        frame_id: str,
        points: np.ndarray,
    ) -> PointCloud2:
        points = self._opengl_points_to_body(points)
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return point_cloud2.create_cloud_xyz32(
            header, points.astype(np.float32).tolist()
        )


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
