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
from nav_msgs.msg import Path as RosPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tracking"))

from behaviors import DroneBehaviors, VehicleState
from core import Event, FiniteStateMachine, Transition
from fsm_spec import (
    CMD_TO_EVENT,
    EVENT_LAND,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_TRACKING,
    TRANSITION_SPECS,
)
from fsm_log import FsmCsvLogger
from tracking_cfg import DEFAULT_CONFIG
from tracking_cnt import PathTrackerCtbr
from yamls.config import get_cfg
from tracking_ros import (
    Px4Bridge,
    TOPIC_VEHICLE_LOCAL_POSITION,
    qos_px4_out,
)
from tracking_utils import (
    is_finite_vec3,
    ned_to_enu,
    wrap_pi,
    yaw_ned_to_enu,
)

_CFG = get_cfg()


@dataclass(frozen=True)
class AutoLandConfig:
    enabled: bool
    distance_m: float
    speed_mps: float
    hold_cycles: int


def _latched_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
    )


def _string_msg(data: str) -> String:
    msg = String()
    msg.data = str(data)
    return msg


def _path_msg_points_enu(msg: RosPath) -> np.ndarray:
    """Extract ENU points from a `nav_msgs/Path` message."""
    poses = getattr(msg, "poses", None) or []
    return np.array(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z] for ps in poses],
        dtype=float,
    )


def _derive_info_topic(state_topic: str) -> str:
    topic = str(state_topic).strip()
    if topic.endswith("/state"):
        return f"{topic[:-6]}/info"
    return f"{topic}/info"


def _build_transitions() -> list[Transition]:
    """Return the FSM transition table."""
    return [Transition(src, event, dst) for src, event, dst in TRANSITION_SPECS]


def _vehicle_state_from_local_position(
    msg: VehicleLocalPosition,
) -> VehicleState | None:
    heading = getattr(msg, "heading", None)
    yaw_ned = float(heading) if heading is not None else 0.0
    if not (heading is not None and np.isfinite(yaw_ned)):
        yaw_ned = 0.0
    yaw_ned = wrap_pi(yaw_ned)

    pos_ned = (
        getattr(msg, "x", float("nan")),
        getattr(msg, "y", float("nan")),
        getattr(msg, "z", float("nan")),
    )
    vel_ned = (
        getattr(msg, "vx", float("nan")),
        getattr(msg, "vy", float("nan")),
        getattr(msg, "vz", float("nan")),
    )
    acc_ned = (
        getattr(msg, "ax", float("nan")),
        getattr(msg, "ay", float("nan")),
        getattr(msg, "az", float("nan")),
    )
    if not (
        is_finite_vec3(pos_ned) and is_finite_vec3(vel_ned) and is_finite_vec3(acc_ned)
    ):
        return None

    return VehicleState(
        position_enu=ned_to_enu(pos_ned),
        velocity_enu=ned_to_enu(vel_ned),
        accel_enu=ned_to_enu(acc_ned),
        yaw_enu=yaw_ned_to_enu(yaw_ned),
    )


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
        self._auto_land = AutoLandConfig(
            enabled=bool(fsm_cfg.auto_land),
            distance_m=float(fsm_cfg.auto_land_distance_m),
            speed_mps=float(fsm_cfg.auto_land_speed_mps),
            hold_cycles=max(int(fsm_cfg.auto_land_hold_cycles), 1),
        )
        self._auto_land_stable_count = 0

        self._pub_state = self.create_publisher(
            String,
            str(fsm_cfg.state_topic),
            _latched_qos(depth=1),
        )
        self._pub_info = self.create_publisher(
            String,
            _derive_info_topic(str(fsm_cfg.state_topic)),
            _latched_qos(depth=10),
        )
        self.create_subscription(String, str(fsm_cfg.cmd_topic), self._on_cmd, 10)
        self.create_subscription(
            Float32,
            str(_CFG.plan2track.yaw_cmd_topic),
            self._on_yaw_cmd,
            10,
        )

        self._bridge = Px4Bridge(self)
        self._tracker = PathTrackerCtbr(
            None,
            cfg=self._tracking_cfg,
            controller=self._controller,
            solver=self._solver,
        )
        base_dir = (
            Path(log_dir)
            if log_dir is not None and str(log_dir).strip()
            else (Path(__file__).resolve().parent / "log")
        )
        self._fsm_log = FsmCsvLogger(
            node=self,
            base_dir=base_dir,
            enabled=bool(log_enabled),
            flush_every=int(log_flush_every),
            meta={
                "controller": self._controller,
                "solver": self._solver,
                "dt_control_s": float(self._dt),
                "takeoff_speed_mps": float(fsm_cfg.takeoff_speed_mps),
            },
        )
        if self._fsm_log.enabled and self._fsm_log.run_dir is not None:
            self.get_logger().info(f"Logging to: {self._fsm_log.run_dir}")
        self.get_logger().info(f"config file: {_CFG.config_path}")

        self._behaviors = DroneBehaviors(
            bridge=self._bridge,
            tracker=self._tracker,
            logger=self._fsm_log,
            takeoff_speed_mps=float(fsm_cfg.takeoff_speed_mps),
            init_yaw_enu=float(_CFG.plan2track.init_yaw),
        )

        # Subscribe to path topics based on controller type.
        # MPC uses a reference path for tracking, while other controllers use a direct path input.
        if self._controller == "mpc":
            self.create_subscription(
                RosPath, str(fsm_cfg.ref_path_topic), self._on_ref_path, 10
            )
            self.get_logger().info(f"Subscribed ref_path: {fsm_cfg.ref_path_topic}")
        else:
            self.create_subscription(
                RosPath, str(fsm_cfg.path_topic), self._on_path, 10
            )
            self.get_logger().info(f"Subscribed path: {fsm_cfg.path_topic}")
        self.get_logger().info(f"Subscribed yaw_cmd: {_CFG.plan2track.yaw_cmd_topic}")

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )

        self._fsm = FiniteStateMachine(
            initial=STATE_PREFLIGHT,
            transitions=_build_transitions(),
            on_enter=self._on_enter_state,
        )
        self._publish_state(self._fsm.state)

        self.create_timer(self._dt, self._loop)
        self.get_logger().info(
            f"Started. controller={self._controller} solver={self._solver} dt={self._dt}"
        )

    def destroy_node(self) -> bool:
        try:
            if getattr(self, "_fsm_log", None) is not None:
                self._fsm_log.close()
        except Exception:
            pass
        return super().destroy_node()

    def _on_enter_state(self, new_state: str, event: Event) -> None:
        self._behaviors.on_enter(new_state, str(event.name))
        self._publish_state(new_state)
        self.get_logger().info(f"state={new_state} via {event.name}")

    def _publish_state(self, state: str) -> None:
        self._pub_state.publish(_string_msg(state))

    def _publish_info(self, text: str) -> None:
        self._pub_info.publish(_string_msg(text))

    def _on_cmd(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        key = raw.split()[0].strip().lower()
        event_name = CMD_TO_EVENT.get(key) or CMD_TO_EVENT.get(raw)
        if event_name is None:
            return
        self._fsm.send(Event(event_name, raw))

    def _on_path(self, msg: RosPath) -> None:
        self._behaviors.update_path(_path_msg_points_enu(msg))

    def _on_ref_path(self, msg: RosPath) -> None:
        self._behaviors.update_ref_traj(_path_msg_points_enu(msg))

    def _on_yaw_cmd(self, msg: Float32) -> None:
        value = float(getattr(msg, "data", float("nan")))
        if np.isfinite(value):
            if self._fsm.state == STATE_HOVER_START:
                self._behaviors.update_takeoff_yaw_cmd_enu()
            else:
                self._behaviors.update_yaw_cmd_enu(value)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        state = _vehicle_state_from_local_position(msg)
        if state is None:
            return
        self._behaviors.update_vehicle_state(state)

    def _loop(self) -> None:
        self._behaviors.tick(self._fsm.state, self._dt)
        self._maybe_auto_land()

    def _reset_auto_land_count(self) -> None:
        self._auto_land_stable_count = 0

    def _maybe_auto_land(self) -> None:
        if not self._auto_land.enabled or self._fsm.state != STATE_TRACKING:
            self._reset_auto_land_count()
            return

        valid, distance_m, speed_mps = self._behaviors.tracking_terminal_metrics()
        if not valid:
            self._reset_auto_land_count()
            return

        if (
            distance_m < self._auto_land.distance_m
            and speed_mps < self._auto_land.speed_mps
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
