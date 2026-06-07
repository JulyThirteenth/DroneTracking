from __future__ import annotations

import numpy as np
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleAngularVelocity, VehicleAttitude, VehicleLocalPosition
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

from .fsm_core import Event, FiniteStateMachine
from .fsm_main import AutoLandMonitor, behavior_creater
from .fsm_ros import (
    VehicleState,
    TOPIC_VEHICLE_ANGULAR_VELOCITY,
    TOPIC_VEHICLE_ATTITUDE,
    derive_info_topic,
    latched_qos,
    body_rates_from_angular_velocity,
    quat_from_vehicle_attitude,
    path_msg_points_enu,
    scan_msg_points_enu,
    vehicle_state_from_local_position,
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
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from yamls.config import get_cfg

_CFG = get_cfg()
_MPC = "mpc"
_MPCC = "mpcc"
_RL_HOVER = "rl_hover"


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
        self._init_yaw_enu = float(_CFG.plan2track.init_yaw)
        self._vehicle_state: VehicleState | None = None
        self._body_rates: np.ndarray | None = None
        self._quat_wxyz: np.ndarray | None = None
        self._tracking_points_enu: np.ndarray | None = None
        self._latest_scan: LaserScan | None = None
        self._auto_land = AutoLandMonitor.from_fsm_cfg(self._fsm_cfg)

        state_topic = str(self._fsm_cfg.state_topic)
        self._pub_state = self.create_publisher(String, state_topic, latched_qos(1))
        self._pub_info = self.create_publisher(
            String, derive_info_topic(state_topic), latched_qos(10)
        )
        self.create_subscription(String, str(self._fsm_cfg.cmd_topic), self._on_cmd, 10)
        self.create_subscription(
            Float32, str(_CFG.plan2track.yaw_cmd_topic), self._on_yaw_cmd, 10
        )
        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )
        self.create_subscription(
            VehicleAngularVelocity,
            TOPIC_VEHICLE_ANGULAR_VELOCITY,
            self._on_angular_velocity,
            qos_px4_out,
        )
        self.create_subscription(
            VehicleAttitude,
            TOPIC_VEHICLE_ATTITUDE,
            self._on_attitude,
            qos_px4_out,
        )

        tracking_topic = (
            str(self._fsm_cfg.ref_path_topic)
            if self._controller in {_MPC, _RL_HOVER}
            else str(self._fsm_cfg.path_topic)
        )
        self.create_subscription(NavPath, tracking_topic, self._on_tracking_path, 10)
        self.get_logger().info(f"Subscribed tracking path: {tracking_topic}")
        self.get_logger().info(f"Subscribed yaw_cmd: {_CFG.plan2track.yaw_cmd_topic}")

        hocbf = self._tracking_cfg.hocbf
        self._use_depth_obstacles = self._controller in {_MPC, _RL_HOVER} and bool(hocbf.enabled)
        if self._use_depth_obstacles:
            self._depth_camera_xyz = np.asarray(hocbf.depth_camera_xyz, dtype=float).reshape(3)
            self._obstacle_min_radius_m = float(hocbf.obstacle_min_radius_m)
            self.create_subscription(LaserScan, str(hocbf.scan_topic), self._on_scan, 10)
            self.get_logger().info(f"Subscribed scan: {hocbf.scan_topic}")

        self.get_logger().info(f"config file: {_CFG.config_path}")
        self._behavior = behavior_creater(
            node=self,
            cfg=self._tracking_cfg,
            controller=self._controller,
            solver=self._solver,
            log_dir=log_dir,
            log_enabled=log_enabled,
            log_flush_every=log_flush_every,
            takeoff_velocity=float(self._fsm_cfg.takeoff_velocity),
            takeoff_height=float(self._fsm_cfg.takeoff_height),
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
        state = vehicle_state_from_local_position(
            msg,
            body_rates=self._body_rates,
            quat_wxyz=self._quat_wxyz,
        )
        if state is None:
            return
        self._vehicle_state = state
        self._behavior.update_vehicle_state(state)

    def _on_angular_velocity(self, msg: VehicleAngularVelocity) -> None:
        rates = body_rates_from_angular_velocity(msg)
        if rates is not None:
            self._body_rates = rates

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        quat = quat_from_vehicle_attitude(msg)
        if quat is not None:
            self._quat_wxyz = quat

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
