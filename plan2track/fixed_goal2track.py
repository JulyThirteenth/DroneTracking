#!/usr/bin/env python3
"""Publish one fixed point-goal for RL point-to-point validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float32, String

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm.fsm_ros import latched_qos
from tracking.tracking_cfg import DEFAULT_CONFIG
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from tracking.tracking_utils import ned_to_enu
from yamls.config import get_cfg

STATE_TRACKING = "tracking"


def _parse_vec3_env(name: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return np.asarray(default, dtype=float)
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain 3 comma-separated values, got: {raw!r}")
    return np.asarray([float(part) for part in parts], dtype=float)


class FixedGoal2TrackNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("fixed_goal2track")
        self._cfg = get_cfg()
        self._tracking_cfg = DEFAULT_CONFIG

        self._position_enu: np.ndarray | None = None
        self._fsm_state = ""
        self._goal_enu: np.ndarray | None = None
        self._last_tracking = False

        self._offset_enu = _parse_vec3_env("FIXED_GOAL_OFFSET_ENU", (1.0, 0.0, 0.0))
        self._absolute_goal = os.environ.get("FIXED_GOAL_ENU", "").strip()
        self._target_z = float(os.environ.get("FIXED_GOAL_Z", self._cfg.fsm.takeoff_height))
        self._publish_dt = float(os.environ.get("FIXED_GOAL_PUBLISH_DT", 0.05))
        self._frame_id = str(self._cfg.plan2track.frame_id)
        self._horizon = int(self._tracking_cfg.mpc.horizon)

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_vehicle_state,
            qos_px4_out,
        )
        self.create_subscription(
            String,
            str(self._cfg.fsm.state_topic),
            self._on_fsm_state,
            latched_qos(1),
        )
        self._pub_ref = self.create_publisher(
            NavPath, str(self._cfg.plan2track.ref_path_topic), 10
        )
        self._pub_yaw = self.create_publisher(
            Float32, str(self._cfg.plan2track.yaw_cmd_topic), 10
        )
        self.create_timer(self._publish_dt, self._tick)

        self.get_logger().info(
            "fixed_goal2track: offset_enu=%s target_z=%.3f absolute_goal=%s"
            % (self._offset_enu.tolist(), self._target_z, self._absolute_goal or "none")
        )

    def _on_vehicle_state(self, msg: VehicleLocalPosition) -> None:
        self._position_enu = ned_to_enu([msg.x, msg.y, msg.z])

    def _on_fsm_state(self, msg: String) -> None:
        self._fsm_state = str(msg.data).strip()

    def _tick(self) -> None:
        tracking = self._fsm_state == STATE_TRACKING
        if not tracking:
            self._last_tracking = False
            self._goal_enu = None
            return
        if self._position_enu is None:
            return

        if self._goal_enu is None or not self._last_tracking:
            self._goal_enu = self._build_goal()
            self.get_logger().info("fixed goal ENU: %s" % self._goal_enu.tolist())
        self._last_tracking = True

        self._pub_ref.publish(self._to_path(self._goal_enu))
        self._pub_yaw.publish(Float32(data=float(self._cfg.plan2track.init_yaw)))

    def _build_goal(self) -> np.ndarray:
        if self._absolute_goal:
            goal = _parse_vec3_env("FIXED_GOAL_ENU", (0.0, 0.0, self._target_z))
        else:
            goal = np.asarray(self._position_enu, dtype=float).reshape(3) + self._offset_enu
        goal = np.asarray(goal, dtype=float).reshape(3)
        goal[2] = self._target_z
        return goal

    def _to_path(self, goal_enu: np.ndarray) -> NavPath:
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        points = np.repeat(np.asarray(goal_enu, dtype=float).reshape(1, 3), self._horizon + 1, axis=0)
        for x, y, z in points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg


def main() -> None:
    rclpy.init()
    node = FixedGoal2TrackNode()
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
