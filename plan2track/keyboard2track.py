#!/usr/bin/env python3
"""Keyboard velocity command to tracker reference path.

Controls:
  Up/Down    : increase/decrease forward speed
  Left/Right : decrease/increase yaw rate
  Space      : stop forward speed and yaw rate
  q          : quit
"""

from __future__ import annotations

import fcntl
import os
import sys
import termios
import tty
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path as NavPath
from px4_msgs.msg import VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracking.tracking_cfg import DEFAULT_CONFIG
from tracking.tracking_utils import ned_to_enu, wrap_pi, yaw_ned_to_enu
from fsm.fsm_ros import latched_qos
from yamls.config import get_cfg

_PROJECT_CFG = get_cfg()
TOPIC_VEHICLE_LOCAL_POSITION = _PROJECT_CFG.topics.px4.vehicle_local_position


STATE_TRACKING = "tracking"
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_RIGHT = "\x1b[C"
KEY_LEFT = "\x1b[D"


class KeyboardTerminal:
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._term = None
        self._flags = None
        if sys.stdin.isatty():
            self._term = termios.tcgetattr(self._fd)
            self._flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
            tty.setcbreak(self._fd)
            fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags | os.O_NONBLOCK)

    @property
    def active(self) -> bool:
        return self._term is not None

    def read(self) -> str:
        if not self.active:
            return ""
        try:
            return os.read(self._fd, 32).decode(errors="ignore")
        except BlockingIOError:
            return ""

    def close(self) -> None:
        if self._term is not None:
            fcntl.fcntl(self._fd, fcntl.F_SETFL, self._flags)
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term)
            self._term = None


class Keyboard2TrackNode(Node):
    def __init__(self):
        super().__init__("keyboard2track")
        self._cfg = get_cfg()
        self._load_params()

        self._position_enu: np.ndarray | None = None
        self._yaw_cmd_enu = float(self._cfg.plan2track.yaw.init)
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._fsm_state = ""
        self._manual_yaw = False
        self._hover_position_enu: np.ndarray | None = None
        self._target_enu: np.ndarray | None = None
        self._terminal = KeyboardTerminal()

        self._create_ros_io()
        self.create_timer(self._publish_dt, self._tick)
        self._log_startup()
        self._print_status()

    def _load_params(self) -> None:
        cfg = self._cfg
        mpc = DEFAULT_CONFIG.mpc
        yaw = DEFAULT_CONFIG.yaw
        control = DEFAULT_CONFIG.control

        self._frame_id = str(cfg.plan2track.path.frame_id)
        self._dt = self._param_float("mpc_dt", mpc.dt)
        self._horizon = self._param_int("horizon", mpc.horizon)
        self._target_z = self._param_float("target_z", cfg.fsm.takeoff.height)
        self._publish_dt = self._param_float("publish_dt", max(float(control.dt), 0.02))
        self._linear_step = self._param_float("linear_step", 0.1)
        self._yaw_rate_step = self._param_float("yaw_rate_step", 0.1)
        self._linear_min = self._param_float("linear_min", 0.0)
        self._linear_max = self._param_float("linear_max", mpc.v_ref)
        self._yaw_rate_limit = self._param_float(
            "yaw_rate_limit",
            1.0 if yaw.yaw_rate_limit is None else yaw.yaw_rate_limit,
        )

    def _param_float(self, name: str, default: float) -> float:
        return float(self.declare_parameter(name, float(default)).value)

    def _param_int(self, name: str, default: int) -> int:
        return int(self.declare_parameter(name, int(default)).value)

    def _create_ros_io(self) -> None:
        cfg = self._cfg
        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_vehicle_state,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )
        self.create_subscription(
            String,
            str(cfg.topics.fsm.state),
            self._on_fsm_state,
            latched_qos(1),
        )
        self._pub_ref = self.create_publisher(
            NavPath, str(cfg.topics.tracking.ref_traj_path), 10
        )
        self._pub_yaw = self.create_publisher(
            Float32, str(cfg.topics.planning.yaw_cmd_enu), 10
        )
        self._pub_vel = self.create_publisher(
            TwistStamped,
            self._param_str("velocity_topic", "/keyboard/velocity_cmd"),
            10,
        )

    def _param_str(self, name: str, default: str) -> str:
        return str(self.declare_parameter(name, str(default)).value)

    def _log_startup(self) -> None:
        if not self._terminal.active:
            self.get_logger().warn("stdin is not a tty; keyboard input is disabled")
        self.get_logger().info(
            "keyboard2track: Up/Down speed, Left/Right yaw_rate, Space stop, q quit"
        )
        self.get_logger().info(
            "publishing "
            f"{self._cfg.topics.tracking.ref_traj_path}, "
            f"{self._cfg.topics.planning.yaw_cmd_enu}"
        )

    def close(self) -> None:
        self._terminal.close()

    def _on_vehicle_state(self, msg: VehicleLocalPosition) -> None:
        self._position_enu = ned_to_enu([msg.x, msg.y, msg.z])
        heading = float(getattr(msg, "heading", np.nan))
        if np.isfinite(heading) and not self._manual_yaw:
            self._yaw_cmd_enu = yaw_ned_to_enu(heading)

    def _on_fsm_state(self, msg: String) -> None:
        self._fsm_state = str(msg.data).strip()

    def _tick(self) -> None:
        self._handle_keys(self._terminal.read())
        self._update_yaw_cmd()

        if not self._should_publish():
            return

        direction = self._direction_enu()
        start_enu = self._reference_start_enu()
        points = self._straight_ref_points(start_enu, direction)
        velocity_enu = direction * self._speed
        self._target_enu = points[-1].copy()
        self._publish_refs(points, velocity_enu)
        self._print_status()

    def _update_yaw_cmd(self) -> None:
        self._yaw_cmd_enu = wrap_pi(
            self._yaw_cmd_enu + self._yaw_rate * self._publish_dt
        )

    def _should_publish(self) -> bool:
        return self._fsm_state == STATE_TRACKING and self._position_enu is not None

    def _direction_enu(self) -> np.ndarray:
        return np.array(
            [np.cos(self._yaw_cmd_enu), np.sin(self._yaw_cmd_enu), 0.0],
            dtype=float,
        )

    def _reference_start_enu(self) -> np.ndarray:
        if self._speed > 0.0:
            self._hover_position_enu = None
            return self._position_enu

        if self._hover_position_enu is None:
            self._hover_position_enu = (
                np.asarray(self._position_enu, dtype=float).reshape(3).copy()
            )
        return self._hover_position_enu

    def _publish_refs(self, points_enu: np.ndarray, velocity_enu: np.ndarray) -> None:
        self._pub_ref.publish(self._to_path(points_enu))
        self._pub_yaw.publish(Float32(data=float(self._yaw_cmd_enu)))
        self._pub_vel.publish(self._to_twist(velocity_enu))

    def _straight_ref_points(
        self,
        start_enu: np.ndarray,
        direction_enu: np.ndarray,
    ) -> np.ndarray:
        start_enu = np.asarray(start_enu, dtype=float).reshape(3).copy()
        start_enu[2] = self._target_z
        horizon_s = self._dt * self._horizon
        end_enu = start_enu + direction_enu * self._speed * horizon_s
        alpha = np.linspace(0.0, 1.0, self._horizon + 1, dtype=float).reshape(-1, 1)
        return start_enu.reshape(1, 3) + alpha * (end_enu - start_enu).reshape(1, 3)

    def _handle_keys(self, keys: str) -> None:
        if not keys:
            return

        if KEY_UP in keys:
            self._set_speed(self._speed + self._linear_step)
        if KEY_DOWN in keys:
            self._set_speed(self._speed - self._linear_step)
        if KEY_RIGHT in keys:
            self._set_yaw_rate(self._yaw_rate + self._yaw_rate_step)
        if KEY_LEFT in keys:
            self._set_yaw_rate(self._yaw_rate - self._yaw_rate_step)
        if " " in keys:
            self._stop()
        if "q" in keys:
            rclpy.shutdown()
            return
        self._print_status()

    def _set_speed(self, speed: float) -> None:
        self._speed = float(np.clip(speed, self._linear_min, self._linear_max))

    def _set_yaw_rate(self, yaw_rate: float) -> None:
        self._manual_yaw = True
        self._yaw_rate = float(
            np.clip(yaw_rate, -self._yaw_rate_limit, self._yaw_rate_limit)
        )

    def _stop(self) -> None:
        self._speed = 0.0
        self._yaw_rate = 0.0

    def _print_status(self) -> None:
        target = self._target_text()
        print(
            f"\rkeyboard2track speed={self._speed:.2f} m/s, "
            f"yaw_rate={self._yaw_rate:.2f} rad/s, "
            f"state={self._fsm_state or 'unknown'}, target={target}",
            end="",
            flush=True,
        )

    def _target_text(self) -> str:
        if self._target_enu is None:
            return "unknown"
        x, y, z = np.asarray(self._target_enu, dtype=float).reshape(3)
        return f"({x:.2f}, {y:.2f}, {z:.2f})"

    def _to_path(self, points_enu: np.ndarray) -> NavPath:
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, z in np.asarray(points_enu, dtype=float).reshape(-1, 3):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _to_twist(self, velocity_enu: np.ndarray) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(velocity_enu[0])
        msg.twist.linear.y = float(velocity_enu[1])
        msg.twist.linear.z = float(velocity_enu[2])
        msg.twist.angular.z = float(self._yaw_rate)
        return msg


def main() -> None:
    rclpy.init()
    node = Keyboard2TrackNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
