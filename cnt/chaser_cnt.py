#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition

from opt.casadi_opt import Dynamics, ChaserParams, Chaser, TrackerParams, Tracker
from opt.mpc import ChaserOSQP
from opt.mpcc import TrackerOSQP
from utils.ros_utils import Px4Bridge, qos_px4_out
from utils.utils import (
    wrap_pi,
    clamp,
    vec3_or_none,
    is_finite_vec3,
    yaw_enu_to_ned,
    yaw_ned_to_enu,
    enu_to_ned,
    ned_to_enu,
    yaw_rate_enu_to_ned,
)

TOPIC_VEHICLE_LOCAL_POSITION = "/fmu/out/vehicle_local_position"


def resolve_task_path(task_path: str | Path | None = None) -> Path:
    """
    Resolve a waypoint task file path.

    Default location after refactor:
      examples/drone_racing/tasks/task.txt

    Backward-compatible fallback:
      examples/drone_racing/cnt/task.txt
    """

    drone_racing_dir = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []

    if task_path:
        p = Path(task_path).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(drone_racing_dir / p)
            candidates.append(Path.cwd() / p)

    candidates.append(drone_racing_dir / "tasks" / "task.txt")
    candidates.append(Path(__file__).with_name("task.txt"))

    for p in candidates:
        if p.exists():
            return p

    # Nothing exists; return the refactor-default path for clearer logs.
    return drone_racing_dir / "tasks" / "task.txt"


def load_waypoints(path: Path):
    waypoints = []
    if not path.exists():
        return waypoints

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 3:
            continue
        waypoints.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
    return waypoints


def flatness_to_ctbr(
    a_ref,
    j_ref,
    yaw: float,
    yaw_rate: float = 0.0,
    hover_thrust: float = 0.55,
    g: float = 9.81,
    thrust_min: float = 0.10,
    thrust_max: float = 0.90,
    eps: float = 1e-6,
):
    """Flatness (a, j, yaw) -> CTBR in NED: returns (p, q, r, thrust_norm)."""
    a = np.asarray(a_ref, dtype=float).reshape(3)
    j = np.asarray(j_ref, dtype=float).reshape(3)
    yaw = float(yaw)
    yaw_rate = float(yaw_rate)

    e3 = np.array([0.0, 0.0, 1.0])
    F = g * e3 - a
    F_norm = float(np.linalg.norm(F) + eps)
    b3 = F / F_norm  # z_B

    thrust_norm = hover_thrust * (F_norm / g)
    thrust_norm = clamp(float(thrust_norm), thrust_min, thrust_max)

    b1d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    c = np.cross(b3, b1d)
    c_norm = float(np.linalg.norm(c) + eps)

    b2 = c / c_norm  # y_B
    b1 = np.cross(b2, b3)  # x_B

    I = np.eye(3)
    b3dot = (I - np.outer(b3, b3)) @ (-j) / F_norm  # h_w

    b1d_dot = yaw_rate * np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    c_dot = np.cross(b3dot, b1d) + np.cross(b3, b1d_dot)
    b2dot = (I - np.outer(b2, b2)) @ c_dot / c_norm
    b1dot = np.cross(b2dot, b3) + np.cross(b2, b3dot)

    Omega = np.column_stack((b1, b2, b3)).T @ np.column_stack((b1dot, b2dot, b3dot))
    return (
        float(Omega[2, 1]),
        float(Omega[0, 2]),
        float(Omega[1, 0]),
        float(thrust_norm),
    )


class WaypointTrackerCtbr:
    """
    Combine Chaser (MPC jerk optimization) with CTBR mapping for waypoint tracking.
    """

    def __init__(
        self,
        target,
        dt: float = 0.05,
        horizon: int = 20,
        controller: str = "tracker",
        solver: str = "ipopt",
        hover_thrust: float = 0.58,
        g: float = 9.81,
        thrust_min: float = 0.10,
        thrust_max: float = 0.90,
        u_min=[-50.0, -50.0, -50.0],
        u_max=[50.0, 50.0, 50.0],
        du_min=[-50.0, -50.0, -50.0],
        du_max=[50.0, 50.0, 50.0],
        v_min=[-8.0, -8.0, -8.0],
        v_max=[8.0, 8.0, 8.0],
        a_min=[-20.0, -20.0, -10.0],
        a_max=[20.0, 20.0, 20.0],
        terminal=2.0,
        Q=np.diag([10.0, 10.0, 10.0]),
        R=np.diag([0.01, 0.01, 0.01]),
        Rd=np.diag([0.2, 0.2, 0.2]),
    ):
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.controller = str(controller).lower().strip()
        self.solver = str(solver).lower().strip()
        self.target = vec3_or_none(target)
        self.u_min = vec3_or_none(u_min)
        self.u_max = vec3_or_none(u_max)
        self.v_min = vec3_or_none(v_min)
        self.v_max = vec3_or_none(v_max)
        self.a_min = vec3_or_none(a_min)
        self.a_max = vec3_or_none(a_max)

        self._is_tracker = self.controller in {"tracker", "mpcc"}
        if self.controller not in {"chaser", "tracker", "mpcc"}:
            raise ValueError("controller must be 'chaser' or 'tracker'.")

        use_qp = self.solver in {"osqp"}

        if self.controller == "chaser":
            params = ChaserParams(
                nx=9,
                nu=3,
                horizon=self.horizon,
                dyn_func=Dynamics("uav"),
                dt=self.dt,
                u_min=self.u_min,
                u_max=self.u_max,
                du_min=vec3_or_none(du_min),
                du_max=vec3_or_none(du_max),
                Q=Q,
                R=R,
                terminal=terminal,
                Rd=Rd,
                track_idx=np.array([0, 1, 2]),
                v_min=self.v_min,
                v_max=self.v_max,
                a_min=self.a_min,
                a_max=self.a_max,
                v_slack_weight=4000.0,
                a_slack_weight=4000.0,
                terminal_v_weight=200.0,
                terminal_a_weight=200.0,
            )
            if use_qp:
                self._opt = ChaserOSQP(params)
            else:
                self._opt = Chaser(params)
        else:
            params = TrackerParams(
                nx=9,
                nu=3,
                horizon=self.horizon,
                dyn_func=Dynamics("uav"),
                dt=self.dt,
                u_min=self.u_min,
                u_max=self.u_max,
                du_min=vec3_or_none(du_min),
                du_max=vec3_or_none(du_max),
                Q=Q,
                R=R,
                terminal=terminal,
                Rd=Rd,
                track_idx=np.array([0, 1, 2]),
                v_min=self.v_min,
                v_max=self.v_max,
                a_min=self.a_min,
                a_max=self.a_max,
                v_slack_weight=4000.0,
                a_slack_weight=4000.0,
                terminal_v_weight=200.0,
                terminal_a_weight=200.0,
                q_contour=2.0,
                q_lag=1.0,
                q_progress=1.0,
                q_terminal_s=3.0,
                vs_max=6.0,
                vs_min=0.0,
            )
            if use_qp:
                self._opt = TrackerOSQP(params)
            else:
                self._opt = Tracker(params)

        self._opt.setup()
        self._ctbr_kwargs = {
            "hover_thrust": float(hover_thrust),
            "g": float(g),
            "thrust_min": float(thrust_min),
            "thrust_max": float(thrust_max),
        }

        self._warm_started = False
        self._x_ws = np.zeros((params.nx, params.horizon + 1))
        self._u_ws = np.zeros((params.nu, params.horizon))
        self._s_ws = np.zeros((1, params.horizon + 1))
        self._vs_ws = np.zeros((1, params.horizon))
        self._u_last = np.zeros(3)
        self._a_est = np.zeros(3)
        self._imu_alpha = 0.02
        self._last_yaw_enu = None
        self._last_vel_enu = None

    def set_target(self, target):
        self.target = vec3_or_none(target)

    def reset_warmstart(self):
        self._warm_started = False
        self._u_last = np.zeros(3)
        self._a_est = np.zeros(3)
        self._s_ws.fill(0.0)
        self._vs_ws.fill(0.0)

    def step(
        self,
        position_enu,
        velocity_enu,
        acceleration_enu,
        yaw_enu: float,
        step_dt: float,
    ):
        p = vec3_or_none(position_enu)
        v = vec3_or_none(velocity_enu)
        a_imu = vec3_or_none(acceleration_enu)
        if self._warm_started:
            self._a_est = self._a_est + self._u_last * step_dt
            # self._a_est = (velocity_enu - self._last_vel_enu) / step_dt
        else:
            self._a_est = a_imu
        self._a_est = (1.0 - self._imu_alpha) * self._a_est + self._imu_alpha * a_imu
        x0 = np.hstack((p, v, self._a_est))
        u0 = self._u_last
        if not self._warm_started:
            self._prime_warmstart(x0, u0)

        if self._is_tracker:
            x_sol, u_sol, s_sol, vs_sol = self._opt.solve(
                x0,
                u0,
                self.target,
                self._x_ws,
                self._u_ws,
                self._s_ws,
                self._vs_ws,
                log=False,
            )
            self._s_ws = s_sol
            self._vs_ws = vs_sol
        else:
            x_sol, u_sol = self._opt.solve(
                x0, u0, self.target, self._x_ws, self._u_ws, log=False
            )

        self._x_ws = x_sol
        self._u_ws = u_sol

        a_enu = x_sol[6:9, 0]
        jerk_enu = u_sol[:, 0]

        yaw_cmd_enu = 0.0
        yaw_rate_enu = wrap_pi(yaw_cmd_enu - yaw_enu)

        a_ned = enu_to_ned(a_enu)
        jerk_ned = enu_to_ned(jerk_enu)
        yaw_cmd_ned = yaw_enu_to_ned(yaw_cmd_enu)
        yaw_rate_ned = yaw_rate_enu_to_ned(yaw_rate_enu)

        # print(f"acc_enu:{a_enu}, acc_ned:{a_ned}")
        # print(f"jerk_enu:{jerk_enu}, jerk_ned:{jerk_ned}")

        p_cmd, q_cmd, r_cmd, thrust = flatness_to_ctbr(
            a_ned, jerk_ned, yaw_cmd_ned, yaw_rate_ned, **self._ctbr_kwargs
        )

        self._u_last = jerk_enu
        self._last_yaw_enu = yaw_cmd_enu
        self._last_vel_enu = velocity_enu
        return p_cmd, q_cmd, r_cmd, thrust

    def _prime_warmstart(self, x0, u0):
        self._x_ws = np.repeat(x0.reshape(-1, 1), self.horizon + 1, axis=1)
        self._u_ws = np.repeat(u0.reshape(-1, 1), self.horizon, axis=1)
        self._s_ws.fill(0.0)
        self._vs_ws.fill(0.0)
        self._warm_started = True


class WaypointTrackerCtbrNode(Node):
    def __init__(
        self,
        controller: str = "tracker",
        solver: str = "ipopt",
        task_path: str | None = None,
    ):
        super().__init__("waypoint_tracker_ctbr")

        self.bridge = Px4Bridge(self)
        self.controller = WaypointTrackerCtbr(
            target=[0.0, 0.0, 0.07],
            dt=0.1,
            horizon=15,
            controller=controller,
            solver=solver,
        )

        self.sub_local_pos = self.create_subscription(
            VehicleLocalPosition,
            TOPIC_VEHICLE_LOCAL_POSITION,
            self.on_local_position,
            qos_px4_out,
        )

        self.have_local_pos = False
        self.have_target = False

        # Extracted from /fmu/out/vehicle_local_position (NED)
        self.local_position = None
        self.local_velocity = None
        self.local_accel = None
        self.local_yaw = None  # heading/yaw in NED (rad)

        self.t0 = self.get_clock().now()
        self.sent_offboard = False
        self.sent_arm = False

        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.loop)

        task_path_resolved = resolve_task_path(task_path)
        self.waypoints_ned = load_waypoints(task_path_resolved)
        self.waypoints_enu = [ned_to_enu(wp) for wp in self.waypoints_ned]
        self.waypoint_index = 0
        self.waypoint_reach_threshold = 0.3
        if self.waypoints_ned:
            self.controller.set_target(self.waypoints_enu[0])
            self.controller.reset_warmstart()
            self.have_target = True
            self.get_logger().info(
                f"Loaded {len(self.waypoints_ned)} waypoints from {task_path_resolved}"
            )
        else:
            self.get_logger().warning(
                f"No waypoints found at {task_path_resolved}; will hold current position."
            )

        self._last_log = 0.0
        self._trace_step = 0
        self._trace_path = Path(__file__).with_name("waypoint_tracker_trace.jsonl")
        self._trace_flush_period_s = 0.5
        self._trace_last_flush_s = 0.0
        try:
            self._trace_fp = self._trace_path.open("w", encoding="utf-8")
            self.get_logger().info(f"Trace logging to {self._trace_path}")
        except OSError:
            self._trace_fp = None
        self.get_logger().info("Waypoint Tracker CTBR node started.")

    def destroy_node(self):
        if getattr(self, "_trace_fp", None) is not None:
            try:
                self._trace_fp.flush()
                self._trace_fp.close()
            except OSError:
                pass
        super().destroy_node()

    def on_local_position(self, msg: VehicleLocalPosition):
        # LocalPosition is NED; acceleration (if present) is typically dv/dt (no gravity compensation).

        heading = getattr(msg, "heading", None)
        if heading is not None and math.isfinite(float(heading)):
            yaw = wrap_pi(float(heading))
            self.local_yaw = yaw

        pos = (
            getattr(msg, "x", float("nan")),
            getattr(msg, "y", float("nan")),
            getattr(msg, "z", float("nan")),
        )
        vel = (
            getattr(msg, "vx", float("nan")),
            getattr(msg, "vy", float("nan")),
            getattr(msg, "vz", float("nan")),
        )
        acc = (
            getattr(msg, "ax", float("nan")),
            getattr(msg, "ay", float("nan")),
            getattr(msg, "az", float("nan")),
        )
        if not (is_finite_vec3(pos) and is_finite_vec3(vel) and is_finite_vec3(acc)):
            return

        self.local_position = np.array(pos, dtype=float)
        self.local_velocity = np.array(vel, dtype=float)
        self.local_accel = np.array(acc, dtype=float)

        self.have_local_pos = True

    def loop(self):
        self.bridge.publish_offboard_mode()

        if not (self.have_local_pos and self.have_target):
            return

        position_enu = ned_to_enu(self.local_position)
        velocity_enu = ned_to_enu(self.local_velocity)
        accel_enu = ned_to_enu(self.local_accel)
        yaw_enu = yaw_ned_to_enu(self.local_yaw)
        p_cmd, q_cmd, r_cmd, thrust = self.controller.step(
            position_enu, velocity_enu, accel_enu, yaw_enu, self.dt
        )
        self._append_trace(
            position_enu, velocity_enu, accel_enu, yaw_enu, self.controller._a_est
        )
        self.bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        elapsed = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self._maybe_switch_mode(elapsed)
        self._maybe_advance_waypoint()
        # self._log_status(p_cmd, q_cmd, r_cmd, thrust)

    def _append_trace(
        self, position_enu, velocity_enu, acceleration_enu, yaw_enu: float, a_est_enu
    ):
        if self._trace_fp is None:
            return
        record = {
            "step": int(self._trace_step),
            "t_s": float(self.get_clock().now().nanoseconds * 1e-9),
            "position_enu": np.asarray(position_enu, dtype=float).reshape(3).tolist(),
            "velocity_enu": np.asarray(velocity_enu, dtype=float).reshape(3).tolist(),
            "acceleration_enu": np.asarray(acceleration_enu, dtype=float)
            .reshape(3)
            .tolist(),
            "a_est_enu": np.asarray(a_est_enu, dtype=float).reshape(3).tolist(),
            "yaw_enu": float(yaw_enu),
        }
        self._trace_step += 1
        try:
            self._trace_fp.write(json.dumps(record, ensure_ascii=True) + "\n")
            now_s = float(self.get_clock().now().nanoseconds * 1e-9)
            if (now_s - self._trace_last_flush_s) >= self._trace_flush_period_s:
                self._trace_fp.flush()
                self._trace_last_flush_s = now_s
        except OSError:
            pass

    def _maybe_switch_mode(self, elapsed: float):
        if elapsed > 1.0 and not self.sent_offboard:
            self.bridge.send_vehicle_command(176, 1.0, 6.0)  # OFFBOARD
            self.sent_offboard = True
            self.get_logger().info("Sent OFFBOARD mode command.")

        if elapsed > 1.2 and not self.sent_arm:
            self.bridge.send_vehicle_command(400, 1.0, 0.0)  # ARM
            self.sent_arm = True
            self.get_logger().info("Sent ARM command.")

    def _log_status(self, p_cmd: float, q_cmd: float, r_cmd: float, thrust: float):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_log <= 0.5:
            return
        # self.get_logger().info(
        #     f"pqr=({p_cmd:.2f},{q_cmd:.2f},{r_cmd:.2f}) "
        #     f"thrust={thrust:.3f} "
        #     f"pos=({self.local_position[0]:.2f},{self.local_position[1]:.2f},{self.local_position[2]:.2f}) "
        #     f"acc_enu=({self.local_accel[0]:.2f},{self.local_accel[1]:.2f},{self.local_accel[2]:.2f}) "
        #     f"|a|={float(np.linalg.norm(self.local_accel)):.2f} "
        #     f"yaw={float(self.local_yaw):.2f} "
        #     f"target=({self.waypoints_ned[self.waypoint_index % len(self.waypoints_ned)]})"
        # )
        self._last_log = now_s

    def _maybe_advance_waypoint(self):
        if not self.waypoints_ned:
            return
        self.waypoint_index = self.waypoint_index % len(self.waypoints_ned)
        target = self.waypoints_ned[self.waypoint_index]
        if np.linalg.norm(self.local_position - target) > self.waypoint_reach_threshold:
            return

        self.waypoint_index += 1
        if self.waypoint_index >= len(self.waypoints_ned):
            self.get_logger().info("Reached final waypoint.")
            return

        next_target_ned = self.waypoints_ned[self.waypoint_index]
        next_target_enu = self.waypoints_enu[self.waypoint_index]
        self.controller.set_target(next_target_enu)
        # self.controller.reset_warmstart()
        self.get_logger().info(
            f"Advance waypoint {self.waypoint_index + 1}/{len(self.waypoints_ned)}: "
            f"{next_target_ned[0]:.2f}, {next_target_ned[1]:.2f}, {next_target_ned[2]:.2f}"
        )
