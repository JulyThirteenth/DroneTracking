import threading
import time
import math
import cv2
from PIL import Image
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from px4_msgs.msg import VehicleLocalPosition
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from .base_control import BaseControl, NavTarget, NavResult, AgentState
import atexit

from tracking.tracking_utils import ned_to_enu, yaw_ned_to_enu, wrap_pi, is_finite_vec3
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from yamls.config import get_cfg
_CFG = get_cfg()

STALE_THRESHOLD = 0.5

class ROS2Control(BaseControl):
    def __init__(self, spf_geometry):
        self.spf_geometry = spf_geometry
        self.bridge = CvBridge()
        self._latest_cv_frame = None
        self._last_frame_time = 0.0
        self._last_position_time = 0.0

        self._is_closed = False

        if not rclpy.ok():
            rclpy.init()

        self.node = Node('agent_ros2_control')
        
        # 图像监听
        custom_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self._sub_image = self.node.create_subscription(
            ROSImage,
            '/rgb',
            self._image_callback,
            qos_profile=custom_qos
        )
        
        self.destroy_event = threading.Event()
        self.spin_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.spin_thread.start()

        atexit.register(self.close)
        print("[ROS2Control] 后台图像监听节点启动成功。")

        # 状态监听
        self._state_lock = threading.Lock()
        self._position_enu: np.ndarray | None = None
        self._velocity_enu: np.ndarray | None = None
        self._yaw_enu: float = 0.0
        self._sub_local_pos = self.node.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_local_position,
            qos_px4_out,
        )

        # 发布导航指令
        self._pub_ref_path = self.node.create_publisher(
            NavPath,
            str(_CFG.plan2track.ref_path_topic),
            10,
        )
        self._pub_yaw_cmd = self.node.create_publisher(
            Float32,
            str(_CFG.plan2track.yaw_cmd_topic),
            10,
        )
    
    def _ros_spin(self):
        """后台轮询线程"""
        while rclpy.ok() and not self.destroy_event.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.03)

    def _image_callback(self, msg):
        try:
            self._latest_cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._last_frame_time = time.time()
        except Exception:
            pass

    def _on_local_position(self, msg: VehicleLocalPosition):
        """缓存 PX4 VehicleLocalPosition（NED → ENU 转换）"""
        pos_ned = (
            getattr(msg, "x", float("nan")),
            getattr(msg, "y", float("nan")),
            getattr(msg, "z", float("nan")),
        )
        if not is_finite_vec3(pos_ned):
            return

        heading = getattr(msg, "heading", None)
        yaw_ned = float(heading) if heading is not None else 0.0
        if not (heading is not None and np.isfinite(yaw_ned)):
            yaw_ned = 0.0

        with self._state_lock:
            self._position_enu = ned_to_enu(np.array(pos_ned, dtype=float))
            self._yaw_enu = yaw_ned_to_enu(wrap_pi(yaw_ned))

            # 缓存速度
            vel_ned = (
                getattr(msg, "vx", float("nan")),
                getattr(msg, "vy", float("nan")),
                getattr(msg, "vz", float("nan")),
            )
            if is_finite_vec3(vel_ned):
                self._velocity_enu = ned_to_enu(np.array(vel_ned, dtype=float))
            self._last_position_time = time.time()

    def get_current_view(self) -> Image.Image:
        """从内存中获取最新的 OpenCV 帧，并翻译为 PIL Image 返回"""
        start_time = time.time()

        while self._latest_cv_frame is None or (time.time() - self._last_frame_time > STALE_THRESHOLD):
            if time.time() - start_time > 5.0:
                raise TimeoutError("[ROS2Control] 获取相机画面超时")
            time.sleep(0.1)

        rgb_frame = cv2.cvtColor(self._latest_cv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        return pil_image

    def get_agent_state(self) -> AgentState:
        """
        从 PX4 VehicleLocalPosition 缓存中获取最新状态。
        返回 ENU 坐标系下的位置和 yaw（度）。
        """
        start_time = time.time()
        
        while self._position_enu is None or (time.time() - self._last_position_time > STALE_THRESHOLD):
            if time.time() - start_time > 5.0:
                raise TimeoutError(
                    "[ROS2Control] 获取无人机位置状态超时"
                )
            time.sleep(0.1)

        with self._state_lock:
            pos = self._position_enu
            yaw = self._yaw_enu

        yaw_deg = math.degrees(yaw) % 360
        return {
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "rotation": {"x": 0.0, "y": yaw_deg, "z": 0.0}
        }

    def navigate_to_point(self, target: NavTarget) -> NavResult:
        """
        发布 MPC 参考轨迹，由 FSM 在 TRACKING 状态下消费执行。
        """
        state = self.get_agent_state()
        current_enu = np.array([
            state["position"]["x"],
            state["position"]["y"],
            state["position"]["z"],
        ], dtype=float)
        current_yaw_deg = state["rotation"]["y"]

        target_x = float(target["x"])
        target_y = float(target["y"])
        target_z = float(target["z"])
        target_enu = np.array([target_x, target_y, target_z], dtype=float)

        target_yaw_deg = float(target["yaw"])
        target_yaw_rad = math.radians(target_yaw_deg)

        horizon = int(_CFG.tracking["mpc"]["horizon"])
        pos_threshold = 0.2
        yaw_threshold_deg = 5.0
        timeout = 30.0

        # ===== STAGE 1: 移动 =====
        progress = np.linspace(0.0, 1.0, horizon + 1, dtype=float)
        delta = (target_enu - current_enu).reshape(3, 1)
        ref_traj_enu = current_enu.reshape(3, 1) + delta * progress.reshape(1, -1)

        current_yaw_rad = math.radians(current_yaw_deg)
        yaw_msg = Float32()
        yaw_msg.data = float(current_yaw_rad)

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                final_state = self.get_agent_state()
                final_pos = final_state["position"]
                return {
                    "success": False,
                    "position": {"x": final_pos["x"], "y": final_pos["y"], "z": final_pos["z"]},
                }

            self._pub_ref_path.publish(self._to_ros_path(ref_traj_enu.T))
            self._pub_yaw_cmd.publish(yaw_msg)

            current_state = self.get_agent_state()
            current_pos = np.array([
                current_state["position"]["x"],
                current_state["position"]["y"],
                current_state["position"]["z"],
            ], dtype=float)
            dist = float(np.linalg.norm(current_pos - target_enu))

            if dist < pos_threshold:
                break

            time.sleep(0.1)

        # ===== STAGE 2: 旋转 =====
        yaw_rate = 0.5 # rad / s
        start_time = time.time()
        dt = 0.03

        current_yaw_rad = math.radians(current_yaw_deg)
        yaw_msg = Float32()
        yaw_msg.data = float(current_yaw_rad)

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                final_state = self.get_agent_state()
                final_pos = final_state["position"]
                return {
                    "success": False,
                    "position": {"x": final_pos["x"], "y": final_pos["y"], "z": final_pos["z"]},
                }
            yaw_error_rad = wrap_pi(target_yaw_rad - current_yaw_rad)
            step = math.copysign(min(abs(yaw_error_rad), yaw_rate * dt), yaw_error_rad)
            current_yaw_rad = wrap_pi(current_yaw_rad + step)
            yaw_msg.data = float(current_yaw_rad)

            current_state = self.get_agent_state()
            hold_enu = np.array([
                current_state["position"]["x"],
                current_state["position"]["y"],
                current_state["position"]["z"],
            ], dtype=float)
            hold_ref = np.repeat(hold_enu.reshape(3, 1), horizon + 1, axis=1)

            self._pub_ref_path.publish(self._to_ros_path(hold_ref.T))
            self._pub_yaw_cmd.publish(yaw_msg)

            current_yaw = current_state["rotation"]["y"]
            yaw_error = abs((current_yaw - target_yaw_deg + 180) % 360 - 180)

            if yaw_error < yaw_threshold_deg:
                return {
                    "success": True,
                    "position": {
                        "x": float(hold_enu[0]),
                        "y": float(hold_enu[1]),
                        "z": float(hold_enu[2]),
                    },
                }

            time.sleep(dt)

    def _to_ros_path(self, points_enu: np.ndarray) -> NavPath:
        """将 (M, 3) ENU 点转换为 nav_msgs/Path"""
        msg = NavPath()
        msg.header.frame_id = str(_CFG.plan2track.frame_id)
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.poses = []

        for x, y, z in points_enu:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg


    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def __del__(self):
        self.close()
    def close(self):
        """释放资源"""
        if self._is_closed: return
        print("正在关闭 ROS2Control 控制器并注销节点...")
        self._is_closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        self.destroy_event.set()
        if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2Control 控制器资源释放完毕")
