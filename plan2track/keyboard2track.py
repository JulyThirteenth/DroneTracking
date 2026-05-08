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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracking.tracking_cfg import DEFAULT_CONFIG
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from tracking.tracking_utils import ned_to_enu, wrap_pi, yaw_ned_to_enu
from fsm.fsm_ros import latched_qos
from yamls.config import get_cfg

STATE_TRACKING = "tracking"


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
        cfg = get_cfg()
        mpc = DEFAULT_CONFIG.mpc
        yaw = DEFAULT_CONFIG.yaw
        control = DEFAULT_CONFIG.control

        self._frame_id = str(cfg.plan2track.frame_id)
        self._dt = float(self.declare_parameter("mpc_dt", float(mpc.dt)).value)
        self._horizon = int(self.declare_parameter("horizon", int(mpc.horizon)).value)
        self._target_z = float(
            self.declare_parameter("target_z", float(cfg.fsm.takeoff_height)).value
        )
        self._publish_dt = float(
            self.declare_parameter("publish_dt", max(float(control.dt), 0.02)).value
        )
        self._linear_step = float(self.declare_parameter("linear_step", 0.2).value)
        self._yaw_rate_step = float(self.declare_parameter("yaw_rate_step", 0.1).value)
        self._linear_min = float(self.declare_parameter("linear_min", 0.0).value)
        self._linear_max = float(
            self.declare_parameter("linear_max", float(mpc.v_ref)).value
        )
        yaw_rate_default = (
            1.0 if yaw.yaw_rate_limit is None else float(yaw.yaw_rate_limit)
        )
        self._yaw_rate_limit = float(
            self.declare_parameter("yaw_rate_limit", yaw_rate_default).value
        )

        self._position_enu: np.ndarray | None = None
        self._yaw_cmd_enu = float(cfg.plan2track.init_yaw)
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._fsm_state = ""
        self._manual_yaw = False
        self._terminal = KeyboardTerminal()

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_vehicle_state,
            qos_px4_out,
        )
        self.create_subscription(
            String,
            str(cfg.fsm.state_topic),
            self._on_fsm_state,
            latched_qos(depth=1),
        )
        self._pub_ref = self.create_publisher(
            NavPath, cfg.plan2track.ref_path_topic, 10
        )
        self._pub_yaw = self.create_publisher(Float32, cfg.plan2track.yaw_cmd_topic, 10)
        self._pub_vel = self.create_publisher(
            TwistStamped,
            str(
                self.declare_parameter("velocity_topic", "/keyboard/velocity_cmd").value
            ),
            10,
        )
        self.create_timer(self._publish_dt, self._tick)

        if not self._terminal.active:
            self.get_logger().warn("stdin is not a tty; keyboard input is disabled")
        self.get_logger().info(
            "keyboard2track: Up/Down speed, Left/Right yaw_rate, Space stop, q quit"
        )
        self.get_logger().info(
            f"publishing {cfg.plan2track.ref_path_topic}, {cfg.plan2track.yaw_cmd_topic}"
        )
        self._print_status()

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
        self._yaw_cmd_enu = wrap_pi(
            self._yaw_cmd_enu + self._yaw_rate * self._publish_dt
        )

        if self._fsm_state != STATE_TRACKING or self._position_enu is None:
            return

        direction = np.array(
            [np.cos(self._yaw_cmd_enu), np.sin(self._yaw_cmd_enu), 0.0], dtype=float
        )
        points = self._straight_ref_points(self._position_enu, direction)
        self._pub_ref.publish(self._to_path(points))
        self._pub_yaw.publish(Float32(data=float(self._yaw_cmd_enu)))
        self._pub_vel.publish(self._to_twist(direction * self._speed))

    def _straight_ref_points(self, start_enu: np.ndarray, direction_enu: np.ndarray) -> np.ndarray:
        start_enu = np.asarray(start_enu, dtype=float).reshape(3).copy()
        start_enu[2] = self._target_z
        end_enu = start_enu + direction_enu * self._speed * self._dt * self._horizon
        alpha = np.linspace(0.0, 1.0, self._horizon + 1, dtype=float).reshape(-1, 1)
        return start_enu.reshape(1, 3) + alpha * (end_enu - start_enu).reshape(1, 3)

    def _handle_keys(self, keys: str) -> None:
        if not keys:
            return
        if "\x1b[A" in keys:
            self._speed = min(self._linear_max, self._speed + self._linear_step)
        if "\x1b[B" in keys:
            self._speed = max(self._linear_min, self._speed - self._linear_step)
        if "\x1b[C" in keys:
            self._manual_yaw = True
            self._yaw_rate = min(
                self._yaw_rate_limit, self._yaw_rate + self._yaw_rate_step
            )
        if "\x1b[D" in keys:
            self._manual_yaw = True
            self._yaw_rate = max(
                -self._yaw_rate_limit, self._yaw_rate - self._yaw_rate_step
            )
        if " " in keys:
            self._speed = 0.0
            self._yaw_rate = 0.0
        if "q" in keys:
            rclpy.shutdown()
            return
        self._print_status()

    def _print_status(self) -> None:
        print(
            f"\rkeyboard2track speed={self._speed:.2f} m/s, "
            f"yaw_rate={self._yaw_rate:.2f} rad/s, state={self._fsm_state or 'unknown'}",
            end="",
            flush=True,
        )

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
