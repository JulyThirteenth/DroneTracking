#!/usr/bin/env python3
"""ROS2 FSM node for the drone racing example.

This node coordinates:
  - High-level commands (`/fsm/cmd`, std_msgs/String) -> FSM transitions.
  - Vehicle telemetry (PX4 `VehicleLocalPosition`) -> current `VehicleState`.
  - Path inputs (nav_msgs/Path) -> tracker targets.
  - Publishes current FSM state on `/fsm/state` (std_msgs/String).

The FSM drives `DroneBehaviors`, which owns the tracker and publishes PX4 offboard
setpoints.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rclpy
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behaviors import DroneBehaviors
from fsm_core import Event, FiniteStateMachine
from fsm_spec import (
    CMD_TO_EVENT,
    EVENT_LAND,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_TRACKING,
    build_transitions,
)
from fsm_ros import (
    VehicleState,
    derive_info_topic,
    latched_qos,
    path_msg_points_enu,
    scan_msg_points_enu,
    vehicle_state_from_local_position,
)
from tracking.tracking_cfg import DEFAULT_CONFIG
from tracking.tracking_ros import (
    TOPIC_VEHICLE_LOCAL_POSITION,
    qos_px4_out,
)
from yamls.config import get_cfg
_CFG = get_cfg()


@dataclass(frozen=True)
class AutoLandConfig:
    enabled: bool
    distance: float
    velocity: float
    hold_cycles: int


class DroneFSMNode(Node):
    """FSM coordinator node."""

    def __init__(
        self,
        *,
        controller: str,
        solver: str,
        log_dir: str | None = None,
        log_enabled: bool = True,
        log_flush_every: int = 1,
    ):
        super().__init__("drone_fsm")
        fsm_cfg = _CFG.fsm

        self._tracking_cfg = DEFAULT_CONFIG
        self._dt = float(self._tracking_cfg.control.dt)
        self._controller = str(controller).lower().strip()
        self._solver = str(solver).lower().strip()
        self._use_depth_obstacles = self._controller == "mpc" and bool(
            self._tracking_cfg.hocbf.enabled
        )
        self._vehicle_state: VehicleState | None = None
        self._latest_scan: LaserScan | None = None
        self._path_points_enu: np.ndarray | None = None
        self._ref_points_enu: np.ndarray | None = None
        self._auto_land = AutoLandConfig(
            enabled=bool(fsm_cfg.auto_land),
            distance=float(fsm_cfg.auto_land_distance),
            velocity=float(fsm_cfg.auto_land_velocity),
            hold_cycles=max(int(fsm_cfg.auto_land_hold_cycles), 1),
        )
        self._auto_land_stable_count = 0

        self._pub_state = self.create_publisher(
            String,
            str(fsm_cfg.state_topic),
            latched_qos(depth=1),
        )
        self._pub_info = self.create_publisher(
            String,
            derive_info_topic(str(fsm_cfg.state_topic)),
            latched_qos(depth=10),
        )
        self.create_subscription(String, str(fsm_cfg.cmd_topic), self._on_cmd, 10)
        self.create_subscription(
            Float32,
            str(_CFG.plan2track.yaw_cmd_topic),
            self._on_yaw_cmd,
            10,
        )

        self.get_logger().info(f"config file: {_CFG.config_path}")

        self._behaviors = DroneBehaviors(
            node=self,
            cfg=self._tracking_cfg,
            controller=self._controller,
            solver=self._solver,
            log_dir=log_dir,
            log_enabled=log_enabled,
            log_flush_every=log_flush_every,
            takeoff_velocity=float(fsm_cfg.takeoff_velocity),
            takeoff_height=float(fsm_cfg.takeoff_height),
            init_yaw_enu=float(_CFG.plan2track.init_yaw),
        )

        # MPC uses a reference trajectory, while MPCC uses a path input.
        if self._controller == "mpc":
            self.create_subscription(
                NavPath, str(fsm_cfg.ref_path_topic), self._on_ref_path, 10
            )
            self.get_logger().info(f"Subscribed ref_path: {fsm_cfg.ref_path_topic}")
        else:
            self.create_subscription(
                NavPath, str(fsm_cfg.path_topic), self._on_path, 10
            )
            self.get_logger().info(f"Subscribed path: {fsm_cfg.path_topic}")
        self.get_logger().info(f"Subscribed yaw_cmd: {_CFG.plan2track.yaw_cmd_topic}")

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )
        if self._use_depth_obstacles:
            hocbf = self._tracking_cfg.hocbf
            scan_topic = str(hocbf.scan_topic)
            self._depth_camera_xyz = np.asarray(
                hocbf.depth_camera_xyz, dtype=float
            ).reshape(3)
            self._obstacle_min_radius_m = float(hocbf.obstacle_min_radius_m)
            self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
            self.get_logger().info(f"Subscribed scan: {scan_topic}")

        self._fsm = FiniteStateMachine(
            initial=STATE_PREFLIGHT,
            transitions=build_transitions(),
            on_enter=self._on_enter_state,
        )
        self._publish_state(self._fsm.state)

        self.create_timer(self._dt, self._loop)
        self.get_logger().info(
            f"Started. controller={self._controller} solver={self._solver} dt={self._dt}"
        )

    def destroy_node(self) -> bool:
        try:
            if getattr(self, "_behaviors", None) is not None:
                self._behaviors.close()
        except Exception:
            pass
        return super().destroy_node()

    def _on_enter_state(self, new_state: str, event: Event) -> None:
        self._behaviors.on_enter(new_state, str(event.name))
        self._publish_state(new_state)
        self.get_logger().info(f"state={new_state} via {event.name}")

    def _publish_state(self, state: str) -> None:
        msg = String()
        msg.data = str(state)
        self._pub_state.publish(msg)

    def _publish_info(self, text: str) -> None:
        msg = String()
        msg.data = str(text)
        self._pub_info.publish(msg)

    def _on_cmd(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        key = raw.split()[0].strip().lower()
        event_name = CMD_TO_EVENT.get(key) or CMD_TO_EVENT.get(raw)
        if event_name is None:
            return
        self._fsm.send(Event(event_name, raw))

    def _on_path(self, msg: NavPath) -> None:
        points = path_msg_points_enu(msg)
        self._path_points_enu = points
        self._behaviors.update_path(points)

    def _on_ref_path(self, msg: NavPath) -> None:
        points = path_msg_points_enu(msg)
        self._ref_points_enu = points
        self._behaviors.update_ref_traj(points)

    def _on_yaw_cmd(self, msg: Float32) -> None:
        value = float(getattr(msg, "data", float("nan")))
        if np.isfinite(value):
            if self._fsm.state == STATE_HOVER_START:
                self._behaviors.update_takeoff_yaw_cmd_enu()
            else:
                self._behaviors.update_yaw_cmd_enu(value)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        state = vehicle_state_from_local_position(msg)
        if state is None:
            return
        self._vehicle_state = state
        self._behaviors.update_vehicle_state(state)

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _loop(self) -> None:
        obstacle_points = None
        if (
            self._use_depth_obstacles
            and self._fsm.state == STATE_TRACKING
            and self._latest_scan is not None
            and self._vehicle_state is not None
        ):
            obstacle_points = scan_msg_points_enu(
                self._latest_scan,
                self._vehicle_state,
                camera_xyz_body=getattr(
                    self, "_depth_camera_xyz", np.array([0.1, 0.0, 0.02])
                ),
                min_radius_m=float(getattr(self, "_obstacle_min_radius_m", 0.8)),
            )
        self._behaviors.tick(self._fsm.state, self._dt, obstacle_points)
        self._maybe_auto_land()

    def _reset_auto_land_count(self) -> None:
        self._auto_land_stable_count = 0

    def _tracking_terminal_metrics(self) -> tuple[bool, float, float]:
        if self._vehicle_state is None:
            return False, float("inf"), float("inf")

        if self._controller == "mpcc":
            points = self._path_points_enu
            if points is None or int(points.shape[0]) < 1:
                return False, float("inf"), float("inf")
            target_enu = points[-1]
        else:
            points = self._ref_points_enu
            if points is None or int(points.shape[0]) < 1:
                return False, float("inf"), float("inf")
            target_enu = points[
                min(self._tracking_cfg.mpc.horizon, points.shape[0] - 1)
            ]

        pos_enu = np.asarray(self._vehicle_state.position_enu, dtype=float).reshape(3)
        vel_enu = np.asarray(self._vehicle_state.velocity_enu, dtype=float).reshape(3)
        target_enu = np.asarray(target_enu, dtype=float).reshape(3)
        return (
            True,
            float(np.linalg.norm(target_enu - pos_enu)),
            float(np.linalg.norm(vel_enu)),
        )

    def _maybe_auto_land(self) -> None:
        if not self._auto_land.enabled or self._fsm.state != STATE_TRACKING:
            self._reset_auto_land_count()
            return

        valid, distance_m, speed_mps = self._tracking_terminal_metrics()
        if not valid:
            self._reset_auto_land_count()
            return

        if (
            distance_m < self._auto_land.distance
            and speed_mps < self._auto_land.velocity
        ):
            self._auto_land_stable_count += 1
        else:
            self._reset_auto_land_count()
            return

        if self._auto_land_stable_count < self._auto_land.hold_cycles:
            return

        self._reset_auto_land_count()
        info = (
            "auto land triggered: "
            f"dist={distance_m:.3f} m speed={speed_mps:.3f} m/s "
            f"stable_cycles={self._auto_land.hold_cycles}"
        )
        self.get_logger().info(info)
        self._publish_info(info)
        self._fsm.send(Event(EVENT_LAND, "auto_land"))


def main() -> None:
    """Entrypoint (reads config from YAML)."""
    rclpy.init()
    node = DroneFSMNode(
        controller=str(_CFG.runtime.controller),
        solver=str(_CFG.runtime.solver),
        log_dir=(str(_CFG.fsm.log_dir) or None),
        log_enabled=bool(_CFG.fsm.log_enabled),
        log_flush_every=int(_CFG.fsm.log_flush_every),
    )
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
