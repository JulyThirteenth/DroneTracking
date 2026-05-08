from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleRatesSetpoint,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yamls.config import get_cfg

_CFG = get_cfg()

TOPIC_OFFBOARD_CONTROL_MODE = _CFG.topics.px4.offboard_control_mode
TOPIC_VEHICLE_RATES_SETPOINT = _CFG.topics.px4.vehicle_rates_setpoint
TOPIC_VEHICLE_COMMAND = _CFG.topics.px4.vehicle_command
TOPIC_VEHICLE_LOCAL_POSITION = _CFG.topics.px4.vehicle_local_position
TARGET_SYSTEM = int(_CFG.vehicle.target_system)
PUB_OFFBOARD = bool(_CFG.vehicle.pub_offboard)


@dataclass
class VehicleState:
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    accel_enu: np.ndarray
    yaw_enu: float


def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def derive_info_topic(state_topic: str) -> str:
    topic = str(state_topic).rstrip("/")
    if topic.endswith("/state"):
        return f"{topic[:-len('/state')]}/info"
    return f"{topic}/info"


def vehicle_state_from_local_position(
    msg: VehicleLocalPosition,
) -> VehicleState | None:
    pos_ned = np.array([msg.x, msg.y, msg.z], dtype=float)
    vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
    acc_ned = np.array([msg.ax, msg.ay, msg.az], dtype=float)
    if not (
        np.all(np.isfinite(pos_ned))
        and np.all(np.isfinite(vel_ned))
        and np.all(np.isfinite(acc_ned))
        and np.isfinite(float(msg.heading))
    ):
        return None

    return VehicleState(
        position_enu=np.array([pos_ned[1], pos_ned[0], -pos_ned[2]], dtype=float),
        velocity_enu=np.array([vel_ned[1], vel_ned[0], -vel_ned[2]], dtype=float),
        accel_enu=np.array([acc_ned[1], acc_ned[0], -acc_ned[2]], dtype=float),
        yaw_enu=float(np.pi / 2.0 - float(msg.heading)),
    )


def path_msg_points_enu(msg: NavPath) -> np.ndarray:
    points = [
        _point_to_xyz(pose_stamped.pose.position)
        for pose_stamped in getattr(msg, "poses", [])
    ]
    return np.asarray(points, dtype=float).reshape((-1, 3))


def scan_msg_points_enu(
    msg: LaserScan | None,
    vehicle_state: VehicleState | None,
    *,
    camera_xyz_body: np.ndarray,
    min_radius_m: float,
) -> np.ndarray | None:
    if msg is None or vehicle_state is None:
        return None

    ranges = np.asarray(msg.ranges, dtype=float)
    if ranges.size == 0:
        return None

    angles = float(msg.angle_min) + np.arange(ranges.size) * float(msg.angle_increment)
    valid = np.isfinite(ranges)
    if np.isfinite(float(msg.range_min)):
        valid &= ranges >= float(msg.range_min)
    if np.isfinite(float(msg.range_max)):
        valid &= ranges <= float(msg.range_max)
    valid &= ranges >= float(min_radius_m)
    if not np.any(valid):
        return None

    points_body = np.column_stack(
        (
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid]),
            np.zeros(int(np.count_nonzero(valid)), dtype=float),
        )
    )
    points_body += np.asarray(camera_xyz_body, dtype=float).reshape(3)

    yaw = float(vehicle_state.yaw_enu)
    rot_body_to_enu = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return points_body @ rot_body_to_enu.T + np.asarray(
        vehicle_state.position_enu, dtype=float
    ).reshape(3)


def _point_to_xyz(point: Point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


class Px4Bridge:
    """
    Minimal PX4 offboard bridge used by PRE path tracking examples.

    Publishes:
    - OffboardControlMode (body_rate enabled)
    - VehicleRatesSetpoint (p,q,r + collective thrust)
    - VehicleCommand (arm/offboard)
    """

    def __init__(self, node: Node):
        self._node = node
        self._target_system = int(TARGET_SYSTEM)
        self._pub_mode = node.create_publisher(
            OffboardControlMode, TOPIC_OFFBOARD_CONTROL_MODE, 10
        )
        self._pub_rates = node.create_publisher(
            VehicleRatesSetpoint, TOPIC_VEHICLE_RATES_SETPOINT, 10
        )
        self._pub_cmd = node.create_publisher(VehicleCommand, TOPIC_VEHICLE_COMMAND, 10)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self._pub_mode.publish(msg)

    def publish_rates_setpoint(self, p: float, q: float, r: float, thrust_norm: float):
        sp = VehicleRatesSetpoint()
        sp.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        sp.roll = float(p)
        sp.pitch = float(q)
        sp.yaw = float(r)
        sp.thrust_body = [0.0, 0.0, float(-thrust_norm)]
        sp.reset_integral = False
        self._pub_rates.publish(sp)

    def send_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0
    ):
        if command == 176 and not PUB_OFFBOARD:  # OFFBOARD
            self._node.get_logger().info(
                "Not publishing OFFBOARD command, using RC publishing instead."
            )
            return
        msg = VehicleCommand()
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = int(self._target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_cmd.publish(msg)
        return
