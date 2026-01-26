import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand


TOPIC_OFFBOARD_CONTROL_MODE = "/fmu/in/offboard_control_mode"
TOPIC_VEHICLE_RATES_SETPOINT = "/fmu/in/vehicle_rates_setpoint"
TOPIC_VEHICLE_COMMAND = "/fmu/in/vehicle_command"
TOPIC_VEHICLE_ATTITUDE = "/fmu/out/vehicle_attitude"
TOPIC_VEHICLE_ODOMETRY = "/fmu/out/vehicle_odometry"

qos_px4_out = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class Px4Bridge:
    def __init__(self, node: Node):
        self._node = node
        self._pub_mode = node.create_publisher(
            OffboardControlMode, TOPIC_OFFBOARD_CONTROL_MODE, 10
        )
        self._pub_rates = node.create_publisher(
            VehicleRatesSetpoint, TOPIC_VEHICLE_RATES_SETPOINT, 10
        )
        self._pub_cmd = node.create_publisher(VehicleCommand, TOPIC_VEHICLE_COMMAND, 10)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
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
        sp.timestamp = self._now_us()
        sp.roll = float(p)
        sp.pitch = float(q)
        sp.yaw = float(r)
        sp.thrust_body = [0.0, 0.0, float(-thrust_norm)]
        sp.reset_integral = False
        self._pub_rates.publish(sp)

    def send_vehicle_command(
        self, command: int, param1: float = 0.0, param2: float = 0.0
    ):
        msg = VehicleCommand()
        msg.timestamp = self._now_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_cmd.publish(msg)

    def _now_us(self) -> int:
        return int(self._node.get_clock().now().nanoseconds / 1000)
