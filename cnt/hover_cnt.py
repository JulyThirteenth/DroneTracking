from dataclasses import dataclass
from utils.utils import quat_to_euler_zyx, wrap_pi, clamp
from utils.ros_utils import (
    Px4Bridge,
    qos_px4_out,
    TOPIC_VEHICLE_ATTITUDE,
    TOPIC_VEHICLE_ODOMETRY,
)

from px4_msgs.msg import VehicleAttitude, VehicleOdometry
import rclpy
from rclpy.node import Node


@dataclass
class RateConfig:
    k_roll: float = 4.0
    k_pitch: float = 6.0
    k_yaw: float = 2.0
    max_p: float = 2.0
    max_q: float = 2.0
    max_r: float = 2.0


@dataclass
class AltitudeConfig:
    target_height_m: float = 2.0
    hover_thrust: float = 0.55  # Typical hover ~0.55-0.60
    kp_z: float = 0.10
    kd_z: float = 0.12
    thrust_min: float = 0.10
    thrust_max: float = 0.90
    thrust_rate_limit: float = 0.8


class HoverCtbrController:
    """
    CTBR = body rates + collective thrust
    - Attitude leveling: roll/pitch -> 0 (P)
    - Yaw hold: yaw -> yaw0 (P)
    - Altitude hold: z -> z0 - 2.0 (PD using VehicleOdometry)
      Assume odometry in local NED (z down+). (Your msg frame=1 strongly suggests this.)
    """

    def __init__(self, rate_cfg=None, alt_cfg=None):
        self.rate = rate_cfg or RateConfig()
        self.alt = alt_cfg or AltitudeConfig()

        self.have_att = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.yaw0 = None

        self.have_odom = False
        self.z = 0.0
        self.vz = 0.0
        self.z0 = None
        self.z_ref = None

        self.thrust_cmd = self.alt.hover_thrust

    def update_attitude(self, q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.roll, self.pitch, self.yaw = quat_to_euler_zyx(w, x, y, z)
        self.have_att = True
        if self.yaw0 is None:
            self.yaw0 = self.yaw

    def update_odometry(self, z, vz):
        self.z = float(z)
        self.vz = float(vz)
        self.have_odom = True

        if self.z0 is None:
            self.z0 = self.z
            self.z_ref = self.z0 - self.alt.target_height_m
            return True
        return False

    def step(self, dt: float):
        p_cmd, q_cmd, r_cmd = self._leveling_rates()
        thrust_target = self._altitude_thrust_target()
        self.thrust_cmd = self._slew(
            self.thrust_cmd, thrust_target, self.alt.thrust_rate_limit, dt
        )
        return p_cmd, q_cmd, r_cmd, self.thrust_cmd

    def _leveling_rates(self):
        if not self.have_att:
            return 0.0, 0.0, 0.0

        p_cmd = -self.rate.k_roll * self.roll
        q_cmd = -self.rate.k_pitch * self.pitch

        yaw_err = 0.0
        if self.yaw0 is not None:
            yaw_err = wrap_pi(self.yaw - self.yaw0)
        r_cmd = -self.rate.k_yaw * yaw_err

        return (
            clamp(p_cmd, -self.rate.max_p, self.rate.max_p),
            clamp(q_cmd, -self.rate.max_q, self.rate.max_q),
            clamp(r_cmd, -self.rate.max_r, self.rate.max_r),
        )

    def _altitude_thrust_target(self):
        if (not self.have_odom) or (self.z_ref is None):
            return self.alt.hover_thrust

        raw = (
            self.alt.hover_thrust
            + self.alt.kp_z * (self.z - self.z_ref)
            + self.alt.kd_z * (self.vz - 0.0)
        )
        return clamp(raw, self.alt.thrust_min, self.alt.thrust_max)

    def _slew(self, current: float, target: float, rate_limit: float, dt: float):
        max_step = rate_limit * dt
        step = clamp(target - current, -max_step, max_step)
        return current + step


class HoverCtbrNode(Node):
    def __init__(self):
        super().__init__("hover_ctbr_odom")

        self.bridge = Px4Bridge(self)
        self.controller = HoverCtbrController()

        self.sub_att = self.create_subscription(
            VehicleAttitude, TOPIC_VEHICLE_ATTITUDE, self.on_attitude, qos_px4_out
        )
        self.sub_odom = self.create_subscription(
            VehicleOdometry, TOPIC_VEHICLE_ODOMETRY, self.on_odometry, qos_px4_out
        )

        self.t0 = self.get_clock().now()
        self.sent_offboard = False
        self.sent_arm = False

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.loop)

        self._last_log = 0.0
        self.get_logger().info("CTBR altitude hold (2m) using ODOMETRY started.")

    def on_attitude(self, msg: VehicleAttitude):
        self.controller.update_attitude(msg.q)

    def on_odometry(self, msg: VehicleOdometry):
        just_set = self.controller.update_odometry(msg.position[2], msg.velocity[2])
        if just_set:
            self.get_logger().info(
                f"z0={self.controller.z0:.3f}, z_ref={self.controller.z_ref:.3f} (assume NED z down+)"
            )

    def loop(self):
        self.bridge.publish_offboard_mode()

        p_cmd, q_cmd, r_cmd, thrust = self.controller.step(self.dt)
        self.bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self._maybe_switch_mode(elapsed)
        self._log_status()

    def _maybe_switch_mode(self, elapsed: float):
        if elapsed > 1.0 and not self.sent_offboard:
            self.bridge.send_vehicle_command(176, 1.0, 6.0)  # OFFBOARD
            self.sent_offboard = True
            self.get_logger().info("Sent OFFBOARD mode command.")

        if elapsed > 1.2 and not self.sent_arm:
            self.bridge.send_vehicle_command(400, 1.0, 0.0)  # ARM
            self.sent_arm = True
            self.get_logger().info("Sent ARM command.")

    def _log_status(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_log <= 0.5:
            return

        if self.controller.have_odom and (self.controller.z_ref is not None):
            self.get_logger().info(
                f"pitch={self.controller.pitch:.3f}, z={self.controller.z:.2f}, "
                f"vz={self.controller.vz:.2f}, z_ref={self.controller.z_ref:.2f}, "
                f"thrust={self.controller.thrust_cmd:.3f}"
            )
        else:
            self.get_logger().info(
                f"pitch={self.controller.pitch:.3f}, thrust={self.controller.thrust_cmd:.3f} (no odom yet)"
            )
        self._last_log = now_s


def main():
    rclpy.init()
    node = HoverCtbrNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
