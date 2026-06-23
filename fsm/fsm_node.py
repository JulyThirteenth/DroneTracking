from __future__ import annotations

import numpy as np
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from .fsm_core import Event, FiniteStateMachine
from .fsm_main import AutoLandMonitor, behavior_creater
from .fsm_ros import (
    VehicleState,
    derive_info_topic,
    latched_qos,
    path_msg_points_enu,
    scan_msg_points_enu,
    vehicle_state_from_local_position,
    TOPIC_VEHICLE_LOCAL_POSITION,
)
from .fsm_spec import (
    CMD_TO_EVENT,
    EVENT_LAND,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_TRACKING,
    build_transitions,
)
from .fsm_wrap import FSMNodeBase
from tracking.tracking_cfg import DEFAULT_CONFIG
from cfg.config import get_cfg

_CFG = get_cfg()
_MPC = "mpc"
_MPCC = "mpcc"


class DroneFSMNode(FSMNodeBase):
    """ROS2 FSM node that translates topics and FSM state into behavior ticks."""

    def __init__(
        self,
        *,
        controller: str,
        solver: str,
        log_dir: str | None = None,
        log_enabled: bool = True,
        log_flush_every: int = 1,
    ):
        super().__init__(name="drone_fsm")

        self._tracking_cfg = DEFAULT_CONFIG
        self._fsm_cfg = _CFG.fsm
        self._dt = float(self._tracking_cfg.control.dt)
        self._controller = str(controller).lower().strip()
        self._solver = str(solver).lower().strip()
        self._init_yaw_enu = float(_CFG.plan2track.yaw.init)
        self._vehicle_state: VehicleState | None = None
        self._tracking_points_enu: np.ndarray | None = None
        self._latest_scan: LaserScan | None = None
        self._auto_land = AutoLandMonitor.from_fsm_cfg(self._fsm_cfg)

        state_topic = str(_CFG.topics.fsm.state)
        self._pub_state = self.create_publisher(String, state_topic, latched_qos(1))
        self._pub_info = self.create_publisher(
            String, derive_info_topic(state_topic), latched_qos(10)
        )
        self.create_subscription(String, str(_CFG.topics.fsm.cmd), self._on_cmd, 10)
        self.create_subscription(
            Float32, str(_CFG.topics.planning.yaw_cmd_enu), self._on_yaw_cmd, 10
        )
        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )

        tracking_topic = (
            str(_CFG.topics.tracking.ref_traj_path)
            if self._controller == _MPC
            else str(_CFG.topics.tracking.path)
        )
        self.create_subscription(NavPath, tracking_topic, self._on_tracking_path, 10)
        self.get_logger().info(f"Subscribed tracking path: {tracking_topic}")
        self.get_logger().info(
            f"Subscribed yaw_cmd: {_CFG.topics.planning.yaw_cmd_enu}"
        )

        hocbf = self._tracking_cfg.hocbf
        self._use_depth_obstacles = self._controller == _MPC and bool(hocbf.enabled)
        if self._use_depth_obstacles:
            self._depth_camera_xyz = np.asarray(
                hocbf.depth_camera_xyz, dtype=float
            ).reshape(3)
            self._obstacle_min_radius_m = float(hocbf.obstacle_min_radius_m)
            self.create_subscription(
                LaserScan, str(_CFG.topics.perception.scan), self._on_scan, 10
            )
            self.get_logger().info(f"Subscribed scan: {_CFG.topics.perception.scan}")

        self.get_logger().info(f"config file: {_CFG.config_path}")
        self._behavior = behavior_creater(
            node=self,
            cfg=self._tracking_cfg,
            controller=self._controller,
            solver=self._solver,
            log_dir=log_dir,
            log_enabled=log_enabled,
            log_flush_every=log_flush_every,
            takeoff_velocity=float(self._fsm_cfg.takeoff.velocity),
            takeoff_height=float(self._fsm_cfg.takeoff.height),
            init_yaw_enu=self._init_yaw_enu,
        )

        self._fsm = FiniteStateMachine(
            initial=STATE_PREFLIGHT,
            transitions=build_transitions(),
            on_enter=self._on_enter_state,
        )
        self._publish(self._pub_state, self._fsm.state)
        self.create_timer(self._dt, self._loop)
        self.get_logger().info(
            f"Started. controller={self._controller} solver={self._solver} dt={self._dt}"
        )

    def _on_enter_state(self, new_state: str, event: Event) -> None:
        event_name = str(event.name)
        self._behavior.on_enter(new_state, event_name)
        self._publish(self._pub_state, new_state)
        self.get_logger().info(f"state={new_state} via {event_name}")

    def _on_cmd(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        key = raw.split()[0].strip().lower() if raw else ""
        event_name = CMD_TO_EVENT.get(key) or CMD_TO_EVENT.get(raw)
        if event_name is not None:
            self._fsm.send(Event(event_name, raw))

    def _on_tracking_path(self, msg: NavPath) -> None:
        self._tracking_points_enu = path_msg_points_enu(msg)
        self._behavior.update_ref_cmd_enu(self._tracking_points_enu)

    def _on_yaw_cmd(self, msg: Float32) -> None:
        yaw_cmd = float(getattr(msg, "data", float("nan")))
        if not np.isfinite(yaw_cmd):
            return
        if self._fsm.state == STATE_HOVER_START:
            yaw_cmd = self._init_yaw_enu
        self._behavior.update_yaw_cmd_enu(yaw_cmd)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        state = vehicle_state_from_local_position(msg)
        if state is None:
            return
        self._vehicle_state = state
        self._behavior.update_vehicle_state(state)

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _loop(self) -> None:
        obstacles = None
        if (
            self._use_depth_obstacles
            and self._fsm.state == STATE_TRACKING
            and self._latest_scan is not None
            and self._vehicle_state is not None
        ):
            obstacles = scan_msg_points_enu(
                self._latest_scan,
                self._vehicle_state,
                camera_xyz_body=self._depth_camera_xyz,
                min_radius_m=self._obstacle_min_radius_m,
            )

        self._behavior.tick(self._fsm.state, self._dt, obstacles)
        if self._auto_land.update(
            fsm_state=self._fsm.state,
            tracking_state=STATE_TRACKING,
            vehicle_state=self._vehicle_state,
            target_enu=self._tracking_target(),
        ):
            self.get_logger().info("auto land triggered")
            self._publish(self._pub_info, "auto land triggered")
            self._fsm.send(Event(EVENT_LAND, "auto_land"))

    def _tracking_target(self) -> np.ndarray | None:
        points = self._tracking_points_enu
        if points is None or points.shape[0] < 1:
            return None
        if self._controller == _MPCC:
            return points[-1]
        idx = min(int(self._tracking_cfg.mpc.horizon), points.shape[0] - 1)
        return points[idx]

    @staticmethod
    def _publish(pub, data: str) -> None:
        msg = String()
        msg.data = str(data)
        pub.publish(msg)
