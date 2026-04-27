from __future__ import annotations

import sys
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yamls.config import get_cfg

_CFG = get_cfg()

TOPIC_OFFBOARD_CONTROL_MODE = _CFG.tracking_ros.offboard_control_mode_topic
TOPIC_VEHICLE_RATES_SETPOINT = _CFG.tracking_ros.vehicle_rates_setpoint_topic
TOPIC_VEHICLE_COMMAND = _CFG.tracking_ros.vehicle_command_topic
TOPIC_VEHICLE_LOCAL_POSITION = _CFG.tracking_ros.vehicle_local_position_topic
TARGET_SYSTEM = int(_CFG.tracking_ros.target_system)
PUB_OFFBOARD = bool(_CFG.tracking_ros.pub_offboard)


qos_px4_out = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def now_us(node: Node) -> int:
    return int(node.get_clock().now().nanoseconds / 1000)


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
        msg.timestamp = now_us(self._node)
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
        sp.timestamp = now_us(self._node)
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
        msg.timestamp = now_us(self._node)
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
