#!/usr/bin/env python3
"""Publish the DroneTracking TF tree for RViz visualization.

Frames:
  map -> base_link -> drone_fpv_camera

The dynamic map->base_link transform is derived from PX4 VehicleLocalPosition.
PX4 local position is NED; this node publishes ENU/map transforms.
"""

from __future__ import annotations

import math
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

DEFAULT_LOCAL_POSITION_TOPIC = "/fmu/out/vehicle_local_position"
DEFAULT_WORLD_FRAME = "map"


def as_vec3(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def ned_to_enu(value) -> np.ndarray:
    value = as_vec3(value)
    return np.array([value[1], value[0], -value[2]], dtype=float)


def wrap_pi(angle_rad: float) -> float:
    angle = float(angle_rad)
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_ned_to_enu(yaw_ned: float) -> float:
    return wrap_pi(math.pi / 2.0 - float(yaw_ned))


def quat_from_yaw_enu(yaw_enu: float) -> tuple[float, float, float, float]:
    half = 0.5 * wrap_pi(yaw_enu)
    return (0.0, 0.0, float(math.sin(half)), float(math.cos(half)))


class DroneTfTreeNode(Node):
    """Broadcast map->body and body->camera transforms."""

    def __init__(self) -> None:
        super().__init__("drone_tf_tree")

        self.declare_parameter(
            "vehicle_local_position_topic",
            DEFAULT_LOCAL_POSITION_TOPIC,
        )
        self.declare_parameter("world_frame", DEFAULT_WORLD_FRAME)
        self.declare_parameter("body_frame", "base_link")
        self.declare_parameter("camera_frame", "drone_fpv_camera")
        self.declare_parameter("camera_xyz", [0.1, 0.0, 0.02])

        self._local_position_topic = str(
            self.get_parameter("vehicle_local_position_topic").value
        )
        self._world_frame = str(self.get_parameter("world_frame").value).strip("/")
        self._body_frame = str(self.get_parameter("body_frame").value).strip("/")
        self._camera_frame = str(self.get_parameter("camera_frame").value).strip("/")
        self._camera_xyz = np.asarray(
            self.get_parameter("camera_xyz").value,
            dtype=float,
        ).reshape(3)

        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._sub_local_position = self.create_subscription(
            VehicleLocalPosition,
            self._local_position_topic,
            self._on_local_position,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        self._publish_camera_static_tf()
        self.get_logger().info(
            "Publishing TF tree: "
            f"{self._world_frame} -> {self._body_frame} -> {self._camera_frame}, "
            f"source={self._local_position_topic}"
        )

    def _publish_camera_static_tf(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._body_frame
        msg.child_frame_id = self._camera_frame
        msg.transform.translation.x = float(self._camera_xyz[0])
        msg.transform.translation.y = float(self._camera_xyz[1])
        msg.transform.translation.z = float(self._camera_xyz[2])
        msg.transform.rotation.w = 1.0
        self._static_tf.sendTransform(msg)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        pos_ned = (
            getattr(msg, "x", float("nan")),
            getattr(msg, "y", float("nan")),
            getattr(msg, "z", float("nan")),
        )
        if not all(math.isfinite(float(v)) for v in pos_ned):
            return

        heading = getattr(msg, "heading", None)
        yaw_ned = float(heading) if heading is not None else 0.0
        if not math.isfinite(yaw_ned):
            yaw_ned = 0.0

        pos_enu = ned_to_enu(pos_ned)
        quat_xyzw = quat_from_yaw_enu(yaw_ned_to_enu(wrap_pi(yaw_ned)))

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self._world_frame
        tf_msg.child_frame_id = self._body_frame
        tf_msg.transform.translation.x = float(pos_enu[0])
        tf_msg.transform.translation.y = float(pos_enu[1])
        tf_msg.transform.translation.z = float(pos_enu[2])
        tf_msg.transform.rotation.x = float(quat_xyzw[0])
        tf_msg.transform.rotation.y = float(quat_xyzw[1])
        tf_msg.transform.rotation.z = float(quat_xyzw[2])
        tf_msg.transform.rotation.w = float(quat_xyzw[3])
        self._tf.sendTransform(tf_msg)


def main() -> None:
    rclpy.init()
    node = DroneTfTreeNode()
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
