import threading
import time
import math
from PIL import Image
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image as ROSImage
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Float32, String

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from .base_control import BaseControl, NavTarget, NavResult, AgentState
import atexit

from tracking.tracking_utils import ned_to_enu, yaw_ned_to_enu, wrap_pi, is_finite_vec3
from fsm.fsm_ros import TOPIC_VEHICLE_LOCAL_POSITION, latched_qos
from fsm.fsm_ros import latched_qos
from yamls.config import get_cfg
_CFG = get_cfg()

NAV_PUBLISH_DT = 0.02
TICK_LOG_INTERVAL = 1.0   # 日志打印间隔

STATE_TRACKING = "tracking"

# ---------- 阈值 ----------
ARRIVAL_XY = 0.02
ARRIVAL_Z = 0.02
ARRIVAL_YAW = 0.05
MIN_DIST_IN_NAV = 0.02 # 导航时每秒移动的最小距离，小于时认为被阻塞

# ---------- 导航 ----------
NAV_SPEED = 1.0       # 导航速度（m/s）

# ---------- 超时 ----------
EXPIRE_TIME = 1.0       # 缓存的画面或位置的过期时间
GET_VIEW_TIMEOUT = 5.0    # 获取相机画面超时（秒）
GET_STATE_TIMEOUT = 5.0   # 获取位置状态超时（秒）
NAV_TIMEOUT = 60.0 # 导航超时
ROTATE_TIMEOUT = 60.0 # 旋转超时
BLOCKED_TIMEOUT = 5.0 # 导航时，被障碍物阻挡导致停滞的超时（秒）


qos_px4_out = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

class InfoNode(Node):
    def __init__(self):
        super().__init__("info_node")

        # 图像
        self._latest_cv_frame: np.ndarray | None = None
        self._last_frame_time: float = 0.0

        # 位置
        self._position_enu: np.ndarray | None = None
        self._last_position_time: float = 0.0

        # 姿态
        self._yaw_enu: float = 0.0
        self._roll_enu: float = 0.0
        self._pitch_enu: float = 0.0

        custom_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            ROSImage,
            "/rgb",
            self._image_callback,
            qos_profile=custom_qos,
        )

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_vehicle_state,
            qos_px4_out,
        )
        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self._on_attitude,
            qos_px4_out,
        )
        self.get_logger().info("启动完成")

    def _image_callback(self, msg: ROSImage):
        try:
            # rgb8 → numpy array
            self._latest_cv_frame = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape(msg.height, msg.width, 3)
            self._last_frame_time = time.time()
        except Exception as e:
            self.get_logger().error("_image_callback 异常: {e}")

    def _on_vehicle_state(self, msg: VehicleLocalPosition):
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

        self._position_enu = ned_to_enu(np.array(pos_ned, dtype=float))
        self._last_position_time = time.time()

    def _on_attitude(self, msg: VehicleAttitude):
        try:
            q = msg.q  # [w, x, y, z]
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

            # 四元数 → Euler 角（NED 坐标系）
            # roll
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll_ned = math.atan2(sinr_cosp, cosr_cosp)

            # pitch
            sinp = 2.0 * (w * y - z * x)
            if abs(sinp) >= 1.0:
                pitch_ned = math.copysign(math.pi / 2.0, sinp)
            else:
                pitch_ned = math.asin(sinp)

            # yaw
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw_ned = math.atan2(siny_cosp, cosy_cosp)

            self._roll_enu = roll_ned
            self._pitch_enu = -pitch_ned
            self._yaw_enu = yaw_ned_to_enu(wrap_pi(yaw_ned))
        except Exception as e:
            self.get_logger().error("_on_attitude 异常: {e}")

    def get_cv_frame(self) -> np.ndarray | None:
        if self._latest_cv_frame is None:
            return None
        if time.time() - self._last_frame_time > EXPIRE_TIME:
            return None
        return self._latest_cv_frame

    def get_position_enu(self) -> np.ndarray | None:
        if time.time() - self._last_position_time > EXPIRE_TIME:
            return None
        return self._position_enu

    def get_yaw_enu(self) -> float:
        return self._yaw_enu

    def get_roll_enu(self) -> float:
        return self._roll_enu

    def get_pitch_enu(self) -> float:
        return self._pitch_enu

class NavigationNode(Node):
    HOLD = "hold"
    MOVING = "moving"
    ROTATING = "rotating"

    def __init__(self):
        super().__init__("navigation_node")

        self._frame_id = str(_CFG.plan2track.path.frame_id)
        self._target_z = float(_CFG.fsm.takeoff.height)
        self._dt = float(_CFG.tracking.get("mpc", {}).get("dt", 0.1))
        self._horizon = int(_CFG.tracking.get("mpc", {}).get("horizon", 15))
        self._nav_speed = NAV_SPEED
        self.get_logger().info(
            f"初始化: horizon={self._horizon}, "
            f"nav_speed={self._nav_speed}, dt={self._dt}, target_z={self._target_z}"
        )
        self._arrival_xy = ARRIVAL_XY
        self._arrival_z = ARRIVAL_Z
        self._arrival_yaw = ARRIVAL_YAW

        self._internal_state = self.HOLD
        self._position_enu: np.ndarray | None = None
        self._yaw_enu: float = 0.0
        self._fsm_state: str = ""
        self._last_position_time: float = 0.0

        self._target_pos: np.ndarray | None = None  # (3,) ENU
        self._target_yaw: float = 0.0  # radians
        self._hold_pos: np.ndarray | None = None
        self._hold_yaw: float = 0.0

        self._nav_event = threading.Event()

        self._min_dist = float("inf")       # 本次移动中达到的最小距离
        self._last_progress_time = 0.0       # 上次距离有明显减小的时刻
        self._nav_fail_reason = ""

        self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self._on_vehicle_state,
            qos_px4_out,
        )
        self.create_subscription(
            String,
            str(_CFG.topics.fsm.state),
            self._on_fsm_state,
            latched_qos(1),
        )
        self._pub_ref = self.create_publisher(
            NavPath, str(_CFG.topics.tracking.ref_traj_path), 10
        )
        self._pub_yaw = self.create_publisher(
            Float32, str(_CFG.topics.planning.yaw_cmd_enu), 10
        )
        self.create_timer(NAV_PUBLISH_DT, self._tick_run)
        self.create_timer(TICK_LOG_INTERVAL, self._tick_log)
        self.get_logger().info("启动完成")

    def _on_vehicle_state(self, msg: VehicleLocalPosition):
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

        self._position_enu = ned_to_enu(np.array(pos_ned, dtype=float))
        self._yaw_enu = yaw_ned_to_enu(wrap_pi(yaw_ned))
        self._last_position_time = time.time()

    def _on_fsm_state(self, msg: String):
        prev_state = self._fsm_state
        self._fsm_state = str(msg.data).strip()
        if prev_state != self._fsm_state:
            self.get_logger().info(f"FSM 状态变化: '{prev_state}' -> '{self._fsm_state}'")

    def _tick_log(self):
        if self._internal_state == NavigationNode.HOLD:
            return
        if self._position_enu is None:
            return

        info = (
            f"{self._internal_state}: "
            f"fsm='{self._fsm_state}', "
            f"pos=({self._position_enu[0]:.3f},{self._position_enu[1]:.3f},{self._position_enu[2]:.3f})"
        )

        if self._internal_state == NavigationNode.MOVING and self._target_pos is not None:
            current_pos = self._position_enu
            dist_xy = np.linalg.norm(self._target_pos[:2] - current_pos[:2])
            self.get_logger().info(
                f"{info}, "
                f"target=({self._target_pos[0]:.3f},{self._target_pos[1]:.3f}), "
                f"dist_xy={dist_xy:.3f}, "
                f"yaw_cmd={self._target_yaw:.3f}"
            )
        elif self._internal_state == NavigationNode.ROTATING:
            yaw_diff = abs(wrap_pi(self._yaw_enu - self._target_yaw))
            self.get_logger().info(
                f"{info}, "
                f"yaw_diff={yaw_diff:.4f}, "
                f"target_yaw={self._target_yaw:.3f}, "
                f"current_yaw={self._yaw_enu:.3f}"
            )
        else:
            self.get_logger().info(info)

    def _tick_run(self):
        if self._position_enu is None:
            return
        if time.time() - self._last_position_time > EXPIRE_TIME:
            self._position_enu = None
            return
        if self._internal_state == self.HOLD:
            self._publish_hold()
        elif self._internal_state == self.MOVING:
            self._publish_move()
        elif self._internal_state == self.ROTATING:
            self._publish_rotate()

    def _publish_hold(self):
        if self._position_enu is None:
            return
        hold_pos = self._hold_pos if self._hold_pos is not None else self._position_enu
        hold_yaw = self._hold_yaw if self._hold_pos is not None else self._yaw_enu

        start = hold_pos.copy().reshape(3)
        start[2] = self._target_z
        points = np.tile(start, (self._horizon + 1, 1))

        self._publish_refs(points)
        self._publish_yaw(hold_yaw)

    def _publish_move(self):
        if self._target_pos is None or self._position_enu is None:
            return

        current_pos = self._position_enu
        direction = self._target_pos - current_pos
        dist_xy = np.linalg.norm(direction[:2])
        dz = abs(self._target_pos[2] - current_pos[2])

        if dist_xy < 0.001:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / dist_xy
            direction = np.array([direction[0], direction[1], 0.0])

        start = current_pos.copy().reshape(3)
        start[2] = self._target_z

        step = direction * self._nav_speed * self._dt

        max_steps = dist_xy / (self._nav_speed * self._dt) if dist_xy > 0 else 0.0
        n_steps = min(self._horizon, max_steps)
        alpha = np.linspace(0, n_steps, self._horizon + 1, dtype=float).reshape(-1, 1)
        points = start.reshape(1, 3) + alpha * step.reshape(1, 3)

        self._publish_refs(points)
        self._publish_yaw(self._target_yaw)

        now = time.time()
        if dist_xy < self._min_dist - MIN_DIST_IN_NAV:
            self._min_dist = dist_xy
            self._last_progress_time = now
        if (now - self._last_progress_time > BLOCKED_TIMEOUT
                and self._min_dist > self._arrival_xy):
            self._nav_fail_reason = "blocked_by_obstacle"
            self.get_logger().warn(
                f"检测到阻塞！停滞{now - self._last_progress_time:.1f}s"
            )
            self._nav_event.set()
            return

        if dist_xy < self._arrival_xy and dz < self._arrival_z:
            self.get_logger().info("已到达目标点")
            self._nav_event.set()

    def _publish_rotate(self):
        if self._position_enu is None:
            return
        hold_pos = self._hold_pos if self._hold_pos is not None else self._position_enu

        start = hold_pos.copy().reshape(3)
        start[2] = self._target_z
        points = np.tile(start, (self._horizon + 1, 1))

        self._publish_refs(points)
        self._publish_yaw(self._target_yaw)

        yaw_diff = abs(wrap_pi(self._yaw_enu - self._target_yaw))
        if yaw_diff < self._arrival_yaw:
            self.get_logger().info("已到达目标偏航")
            self._nav_event.set()

    def _publish_refs(self, points_enu: np.ndarray):
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        poses_list = []
        for x, y, z in np.asarray(points_enu, dtype=float).reshape(-1, 3):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            poses_list.append(ps)
        msg.poses = poses_list
        self._pub_ref.publish(msg)

    def _publish_yaw(self, yaw_rad: float):
        self._pub_yaw.publish(Float32(data=float(yaw_rad)))

    def _make_nav_result(self, success: bool) -> NavResult:
        p = self._position_enu
        yaw_deg = math.degrees(self._yaw_enu) % 360
        reason = self._nav_fail_reason
        self._nav_fail_reason = ""
        return {
            "success": success,
            "position": None if p is None else {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "yaw": yaw_deg,
            "reason": reason,
        }

    def _rotate_to_yaw(self, target_yaw: float) -> bool:
        assert self._position_enu is not None
        yaw_diff = abs(wrap_pi(target_yaw - self._yaw_enu))
        self.get_logger().info(
            f"开始旋转, target_yaw={target_yaw:.3f}, "
            f"current_yaw={self._yaw_enu:.3f}, yaw_diff={yaw_diff:.4f}"
        )
        if yaw_diff <= self._arrival_yaw:
            return True
        self._hold_pos = self._position_enu.copy()
        self._hold_yaw = target_yaw

        self._internal_state = self.ROTATING
        self._target_yaw = target_yaw
        self._nav_event.clear()

        arrived = self._nav_event.wait(timeout=ROTATE_TIMEOUT)
        if not arrived:
            self.get_logger().error(f"偏航旋转超时")
            self._nav_fail_reason = "timeout"
            self._go_hold()
            return False
        return True

    def _move_to_point(self, target_pos: np.ndarray, move_yaw: float) -> bool:
        assert self._position_enu is not None
        current_pos = self._position_enu
        dist_xy = np.linalg.norm(target_pos[:2] - current_pos[:2])
        dist_z = abs(target_pos[2] - current_pos[2])
        self.get_logger().info(
            f"开始导航, "
            f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}), "
            f"current=({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f}), "
            f"dist_xy={dist_xy:.3f}, dist_z={dist_z:.3f}, "
            f"move_yaw={move_yaw:.3f}, nav_speed={self._nav_speed}"
        )
        if dist_xy <= self._arrival_xy and dist_z <= self._arrival_z:
            return True

        self._internal_state = self.MOVING
        self._target_pos = target_pos
        self._target_yaw = move_yaw
        self._nav_fail_reason = ""
        self._min_dist = dist_xy
        self._last_progress_time = time.time()
        self._nav_event.clear()

        arrived = self._nav_event.wait(timeout=NAV_TIMEOUT)
        if self._nav_fail_reason:
            self.get_logger().error(f"导航失败: {self._nav_fail_reason}")
            self._go_hold()
            return False
        if not arrived:
            self._nav_fail_reason = "timeout"
            self.get_logger().error(f"导航超时（{NAV_TIMEOUT}s）")
            self._go_hold()
            return False
        return True

    def navigate_to_point(self, target: NavTarget) -> NavResult:
        """两阶段导航：先旋转面向目标，再直线移动到目标点。"""
        if self._fsm_state != STATE_TRACKING:
            self.get_logger().error(
                f"FSM 状态为 '{self._fsm_state}'，"
                f"需要 'tracking' 状态才能执行导航。请先执行 'execute' 命令。"
            )
            self._nav_fail_reason = "fsm_not_tracking"
            return self._make_nav_result(False)

        if self._position_enu is None:
            self.get_logger().error("尚未获取到无人机位置")
            return {"success": False, "position": None, "yaw": 0.0, "reason": "drone_location_is_missing"}

        target_pos = np.array(
            [float(target["x"]), float(target["y"]), float(target["z"])]
        )

        dx = target_pos[0] - self._position_enu[0]
        dy = target_pos[1] - self._position_enu[1]
        face_yaw = math.atan2(dy, dx)

        if not self._rotate_to_yaw(face_yaw):
            self.get_logger().error("阶段一（旋转面向）失败")
            return self._make_nav_result(False)

        if not self._move_to_point(target_pos, face_yaw):
            self.get_logger().error("阶段二（直线移动）失败")
            return self._make_nav_result(False)

        self._go_hold()
        self.get_logger().info("导航完成")
        return self._make_nav_result(True)

    def rotate(self, yaw: float) -> NavResult:
        """原地旋转到目标偏航角（度），不改变位置。"""
        target_yaw_rad = math.radians(yaw)
        if self._fsm_state != STATE_TRACKING:
            self.get_logger().error(
                f"FSM 状态为 '{self._fsm_state}'，"
                f"需要 'tracking' 状态才能执行旋转。请先执行 'execute' 命令。"
            )
            self._nav_fail_reason = "fsm_not_tracking"
            return self._make_nav_result(False)

        if self._position_enu is None:
            self.get_logger().error("尚未获取到无人机位置")
            return {"success": False, "position": None, "yaw": 0.0, "reason": "drone_location_is_missing"}

        if not self._rotate_to_yaw(target_yaw_rad):
            self.get_logger().error("旋转失败")
            return self._make_nav_result(False)

        self._go_hold()
        self.get_logger().info("旋转完成")
        return self._make_nav_result(True)

    def go_hold(self):
        self._go_hold()

    def _go_hold(self):
        """切换为保持状态，记录悬停位置。"""
        self.get_logger().info("切换到 HOLD 状态")
        self._internal_state = self.HOLD
        if self._position_enu is not None:
            self._hold_pos = self._position_enu.copy()
            self._hold_yaw = self._yaw_enu
        self._target_pos = None
        self._target_yaw = 0.0


class ROS2Control(BaseControl):
    def __init__(self, spf_geometry):
        self.spf_geometry = spf_geometry
        self._is_closed = False

        if not rclpy.ok():
            rclpy.init()

        self._info_node = InfoNode()
        self._nav_node = NavigationNode()

        self.destroy_event = threading.Event()
        self.spin_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.spin_thread.start()

        atexit.register(self.close)

    def _ros_spin(self):
        while rclpy.ok() and not self.destroy_event.is_set():
            rclpy.spin_once(self._info_node, timeout_sec=0.1)
            rclpy.spin_once(self._nav_node, timeout_sec=0.05)

    def navigate_to_point(self, target: NavTarget) -> NavResult:
        return self._nav_node.navigate_to_point(target)

    def rotate(self, yaw: float) -> NavResult:
        return self._nav_node.rotate(yaw)

    def get_current_view(self) -> Image.Image:
        start_time = time.time()

        while True:
            frame = self._info_node.get_cv_frame()
            if frame is not None:
                break
            if time.time() - start_time > GET_VIEW_TIMEOUT:
                raise TimeoutError("[ROS2Control] 获取相机画面超时")
            time.sleep(0.1)
        pil_image = Image.fromarray(frame)
        return pil_image

    def get_agent_state(self) -> AgentState:
        start_time = time.time()

        pos = None
        while pos is None:
            if time.time() - start_time > GET_STATE_TIMEOUT:
                raise TimeoutError("[ROS2Control] 获取无人机位置状态超时")
            pos = self._info_node.get_position_enu()
            if pos is None:
                time.sleep(0.1)

        roll = self._info_node.get_roll_enu()
        pitch = self._info_node.get_pitch_enu()
        yaw = self._info_node.get_yaw_enu()

        return {
            "position": {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            },
            "rotation": {
                "roll": math.degrees(roll) % 360,
                "pitch": math.degrees(pitch) % 360,
                "yaw": math.degrees(yaw) % 360,
            },
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """释放资源"""
        if self._is_closed:
            return
        print("正在关闭 ROS2Control 控制器并注销节点...")
        self._is_closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        self.destroy_event.set()
        if hasattr(self, "spin_thread") and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        if hasattr(self, "_info_node"):
            self._info_node.destroy_node()
        if hasattr(self, "_nav_node"):
            self._nav_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS2Control 控制器资源释放完毕")
