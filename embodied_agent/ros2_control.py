import threading
import time
import cv2
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from .base_control import BaseControl, NavTarget, NavResult, AgentState
import atexit

class ROS2Control(BaseControl):
    def __init__(self, spf_geometry):
        self.spf_geometry = spf_geometry
        self.bridge = CvBridge()
        self._latest_cv_frame = None
        self._last_frame_time = 0.0

        self._is_closed = False

        if not rclpy.ok():
            rclpy.init()

        self.node = Node('agent_ros2_control')
        
        custom_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.sub = self.node.create_subscription(
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

    def _image_callback(self, msg):
        try:
            self._latest_cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._last_frame_time = time.time()
        except Exception:
            pass

    def _ros_spin(self):
        """后台轮询线程"""
        while rclpy.ok() and not self.destroy_event.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.03)

    def get_current_view(self) -> Image.Image:
        """从内存中获取最新的 OpenCV 帧，并翻译为 PIL Image 返回"""
        start_time = time.time()
        STALE_THRESHOLD = 0.5

        while self._latest_cv_frame is None or (time.time() - self._last_frame_time > STALE_THRESHOLD):
            if time.time() - start_time > 5.0:
                raise TimeoutError("[ROS2Control] 无法获取相机画面，相机流已中断或通信断联！")
            time.sleep(0.1)

        rgb_frame = cv2.cvtColor(self._latest_cv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        return pil_image

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
    # =======================================================
    # 暂时未实现
    # =======================================================
    def navigate_to_point(self, target: NavTarget) -> NavResult:
        return {"success": False, "position": None}

    def get_agent_state(self) -> AgentState:
        return {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}
        }

