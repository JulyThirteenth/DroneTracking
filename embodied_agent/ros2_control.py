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
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from .base_control import BaseControl, NavTarget, NavResult, AgentState
import atexit

from tracking.tracking_utils import ned_to_enu, yaw_ned_to_enu, wrap_pi, is_finite_vec3
from tracking.tracking_ros import TOPIC_VEHICLE_LOCAL_POSITION, qos_px4_out
from yamls.config import get_cfg
_CFG = get_cfg()

STALE_THRESHOLD = 0.5
NAV_GOAL_TOPIC = "/agent/nav/goal" # 发布
NAV_RESULT_TOPIC = "/agent/nav/result" # 接收
NAV_TIMEOUT = 30.0

def yaw_to_quaternion(yaw_deg: float) -> dict:
    yaw_rad = math.radians(yaw_deg)
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw_rad / 2.0),
        "w": math.cos(yaw_rad / 2.0),
    }

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z)
    cosy_cosp = 1.0 - 2.0 * (q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

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
        self._pub_nav_goal = self.node.create_publisher(
            PoseStamped, NAV_GOAL_TOPIC, 10
        )
        self._sub_nav_result = self.node.create_subscription(
            PoseStamped, NAV_RESULT_TOPIC,
            self._on_navigation_result, 10
        )
        self._nav_lock = threading.Lock()
        self._nav_event = threading.Event()
        self._nav_result: NavResult | None = None
    
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

            # 缓存速度（暂时没用）
            vel_ned = (
                getattr(msg, "vx", float("nan")),
                getattr(msg, "vy", float("nan")),
                getattr(msg, "vz", float("nan")),
            )
            if is_finite_vec3(vel_ned):
                self._velocity_enu = ned_to_enu(np.array(vel_ned, dtype=float))
            self._last_position_time = time.time()

    def navigate_to_point(self, target: NavTarget) -> NavResult:
        goal = PoseStamped()
        goal.header.frame_id = str(_CFG.plan2track.frame_id)
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.position.x = float(target["x"])
        goal.pose.position.y = float(target["y"])
        goal.pose.position.z = float(target["z"])

        q = yaw_to_quaternion(float(target["yaw"]))
        goal.pose.orientation.x = q["x"]
        goal.pose.orientation.y = q["y"]
        goal.pose.orientation.z = q["z"]
        goal.pose.orientation.w = q["w"]

        with self._nav_lock:
            self._nav_event.clear()
            self._nav_result = None
        
        self._pub_nav_goal.publish(goal)
        print(f"[ROS2Control] 导航指令已发布 → ({target['x']:.2f}, {target['y']:.2f}, {target['z']:.2f}), yaw={target['yaw']:.1f}°")
        
        if self._nav_event.wait(timeout=NAV_TIMEOUT):
            with self._nav_lock:
                assert self._nav_result is not None
                return self._nav_result

        print(f"[ROS2Control] 导航超时（{NAV_TIMEOUT}s）")
        return {"success": False, "position": None}

    def _on_navigation_result(self, msg: PoseStamped):
        with self._nav_lock:
            pos = msg.pose.position
            q = msg.pose.orientation

            # 检查是否失败
            # 位置(0,0,0) && w = 1 为失败
            is_failure = (
                abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6
                and abs(q.x) < 1e-6 and abs(q.y) < 1e-6
                and abs(q.z) < 1e-6 and abs(q.w - 1.0) < 1e-6
            )

            if is_failure:
                self._nav_result = {"success": False, "position": None}
                print("[ROS2Control] 收到导航结果：失败")
            else:
                self._nav_result = {
                    "success": True,
                    "position": {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z)},
                }
                print(f"[ROS2Control] 收到导航结果：成功 → ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

            self._nav_event.set()

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
