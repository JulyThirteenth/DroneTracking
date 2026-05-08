#!/usr/bin/env python3
"""
CTBR tracking controller using MPC/MPCC:
- `tracking_opt.MPC` (per-step reference tracking)
- `tracking_opt.MPCC` / `tracking_osqp.MPCCOSQP` (MPCC)
- `tracking_osqp.MPCOSQP` with optional HOCBF obstacle constraints

This module intentionally contains no ROS node; the drone racing example uses the
FSM node (`examples/DroneTracking/fsm/fsm_node.py`) for orchestration and PX4
offboard publishing.
"""

from __future__ import annotations

import numpy as np

from tracking.tracking_cfg import (
    DEFAULT_CONFIG,
    TrackingConfig,
    make_mpc_params,
    make_mpcc_params,
)
from tracking.tracking_utils import (
    as_vec3,
    enu_to_ned,
    flatness_to_ctbr,
    wrap_pi,
    yaw_enu_to_ned,
    yaw_rate_enu_to_ned,
    obstacle_points_to_planes,
)


class PathTrackerCtbr:
    """
    Path tracking controller (jerk-control) + flatness-to-CTBR mapping.

    - MPC: tracks a provided per-step reference trajectory (3, N+1)
    - MPCC: follows a provided polyline path
    """

    def __init__(
        self,
        polyline_enu: list[np.ndarray] | np.ndarray | None,
        *,
        cfg: TrackingConfig = DEFAULT_CONFIG,
        controller: str | None = None,
        solver: str | None = None,
    ):
        self.cfg = cfg
        self.dt = float(cfg.mpc.dt)
        self.horizon = int(cfg.mpc.horizon)
        self.controller = str(controller or cfg.default_controller).lower().strip()
        self.solver = str(solver or cfg.default_solver).lower().strip()

        if polyline_enu is None:
            self._path_points = np.zeros((0, 3), dtype=float)
        else:
            pts = np.asarray(polyline_enu, dtype=float)
            self._path_points = pts.reshape(-1, 3)

        with_dynamics = self.solver not in {"osqp"}
        params = make_mpc_params(cfg, with_dynamics=with_dynamics)
        params_mpcc = make_mpcc_params(cfg, with_dynamics=with_dynamics)

        if self.controller not in {"mpc", "mpcc"}:
            raise ValueError("controller must be 'mpc' or 'mpcc'.")

        if self.controller == "mpc":
            if self.solver in {"osqp"}:
                from tracking.tracking_osqp import HOCBFConfig, MPCOSQP

                cbf = None
                if bool(cfg.hocbf.enabled):
                    cbf = HOCBFConfig(
                        max_obstacles=int(cfg.hocbf.max_obstacles),
                        safe_distance=float(cfg.hocbf.safe_distance),
                        lambda_gain=float(cfg.hocbf.lambda_gain),
                        slack_weight=float(cfg.hocbf.slack_weight),
                    )
                self._opt = MPCOSQP(params, cbf=cbf)
            else:
                if bool(cfg.hocbf.enabled):
                    raise ValueError("HOCBF obstacle avoidance requires solver 'osqp'.")
                from tracking.tracking_opt import MPC

                self._opt = MPC(params)
        else:
            if self.solver in {"osqp"}:
                from tracking.tracking_osqp import MPCCOSQP

                self._opt = MPCCOSQP(params_mpcc)
            else:
                from tracking.tracking_opt import MPCC

                self._opt = MPCC(params_mpcc)
        self._opt.setup()

        self._ctbr_kwargs = cfg.ctbr.as_kwargs()

        self._warm_started = False
        self._x_ws = np.zeros((params.nx, params.horizon + 1))
        self._u_ws = np.zeros((params.nu, params.horizon))
        self._s_ws = np.zeros((1, params.horizon + 1))
        self._vs_ws = np.zeros((1, params.horizon))
        self._u_last = np.zeros(3)
        self._a_est = np.zeros(3)
        self._imu_alpha = float(cfg.accel_fusion.imu_alpha)
        self._yaw_kp = float(cfg.yaw.kp)
        if cfg.yaw.yaw_rate_limit is None:
            self._yaw_rate_limit = np.inf
        else:
            self._yaw_rate_limit = float(cfg.yaw.yaw_rate_limit)
        self._yaw_cmd_prev_enu: float | None = None
        self.last_debug: dict[str, object] = {}

    def reset_warmstart(self):
        self._warm_started = False
        self._u_last = np.zeros(3)
        self._a_est = np.zeros(3)
        self._s_ws.fill(0.0)
        self._vs_ws.fill(0.0)
        self._yaw_cmd_prev_enu = None

    def step(
        self,
        position_enu,
        velocity_enu,
        acceleration_enu,
        yaw_enu: float,
        step_dt: float,
        *,
        yaw_cmd_enu: float = np.pi / 2.0,
        ref_traj_enu: np.ndarray | None = None,
        path_points_enu: np.ndarray | None = None,
        obstacle_points_enu: np.ndarray | None = None,
        log_solver: bool = False,
    ):
        p = as_vec3(position_enu)
        v = as_vec3(velocity_enu)
        a_enu = as_vec3(acceleration_enu)

        if self._warm_started:
            self._a_est = self._a_est + self._u_last * float(step_dt)
        else:
            self._a_est = a_enu
        self._a_est = (1.0 - self._imu_alpha) * self._a_est + self._imu_alpha * a_enu
        # self._a_est = a_enu


        x0 = np.hstack((p, v, self._a_est))
        u0 = self._u_last
        if not self._warm_started:
            self._prime_warmstart(x0, u0)

        if self.controller == "mpc":
            if ref_traj_enu is None:
                ref_traj = np.repeat(p.reshape(3, 1), self.horizon + 1, axis=1)
            else:
                ref_traj = np.asarray(ref_traj_enu, dtype=float)
            if getattr(self._opt, "cbf", None) is not None:
                obstacle_normals = None
                obstacle_points = None
                if obstacle_points_enu is not None:
                    

                    obstacle_points = np.asarray(obstacle_points_enu, dtype=float)
                    obstacle_normals, obstacle_points = obstacle_points_to_planes(
                        p,
                        obstacle_points,
                        max_obstacles=int(self._opt.cbf.max_obstacles),
                    )
                x_sol, u_sol = self._opt.solve(
                    x0,
                    u0,
                    ref_traj,
                    self._x_ws,
                    self._u_ws,
                    obstacle_normals=obstacle_normals,
                    obstacle_points=obstacle_points,
                    log=log_solver,
                )
            else:
                x_sol, u_sol = self._opt.solve(
                    x0, u0, ref_traj, self._x_ws, self._u_ws, log=log_solver
                )
            self._x_ws = x_sol
            self._u_ws = u_sol
        else:
            if path_points_enu is not None:
                path_points = np.asarray(path_points_enu, dtype=float).reshape(-1, 3)
            else:
                path_points = self._path_points

            if path_points.shape[0] < 2:
                target = path_points[0]
            else:
                target = path_points

            x_sol, u_sol, s_sol, vs_sol = self._opt.solve(
                x0,
                u0,
                target,
                self._x_ws,
                self._u_ws,
                self._s_ws,
                self._vs_ws,
                log=log_solver,
            )
            self._x_ws = x_sol
            self._u_ws = u_sol
            self._s_ws = s_sol
            self._vs_ws = vs_sol
            ref_traj = x_sol[0:3, :]

        a_enu = x_sol[6:9, 1]
        jerk_enu = u_sol[:, 0]

        if self._yaw_cmd_prev_enu is None:
            yaw_rate_ff_enu = 0.0
        else:
            yaw_rate_ff_enu = wrap_pi(yaw_cmd_enu - self._yaw_cmd_prev_enu) / step_dt
        yaw_err_enu = wrap_pi(yaw_cmd_enu - yaw_enu) 
        yaw_rate_fb_enu = self._yaw_kp * yaw_err_enu / step_dt
        yaw_rate_enu = yaw_rate_ff_enu + yaw_rate_fb_enu
        if np.isfinite(self._yaw_rate_limit):
            yaw_rate_enu = float(
                np.clip(yaw_rate_enu, -self._yaw_rate_limit, self._yaw_rate_limit)
            )

        a_ned = enu_to_ned(a_enu)
        jerk_ned = enu_to_ned(jerk_enu)
        yaw_ned = yaw_enu_to_ned(yaw_enu)
        yaw_cmd_ned = yaw_enu_to_ned(yaw_cmd_enu)
        yaw_rate_ned = yaw_rate_enu_to_ned(yaw_rate_enu)

        self.last_debug = {
            "x0": x0,
            "u0": u0,
            "a_est_enu": self._a_est.copy(),
            "acc_cmd_enu": a_enu.copy(),
            "jerk_cmd_enu": jerk_enu.copy(),
            "acc_cmd_ned": a_ned.copy(),
            "jerk_cmd_ned": jerk_ned.copy(),
            "yaw_enu": float(yaw_enu),
            "yaw_cmd_enu": float(yaw_cmd_enu),
            "yaw_err_enu": float(yaw_err_enu),
            "yaw_rate_ff_enu": float(yaw_rate_ff_enu),
            "yaw_rate_fb_enu": float(yaw_rate_fb_enu),
            "yaw_rate_cmd_enu": float(yaw_rate_enu),
            "yaw_cmd_ned": float(yaw_cmd_ned),
            "yaw_rate_cmd_ned": float(yaw_rate_ned),
            "controller": str(self.controller),
            "solver": str(self.solver),
        }

        p_cmd, q_cmd, r_cmd, thrust = flatness_to_ctbr(
            a_ned, jerk_ned, yaw_ned, yaw_rate_ned, **self._ctbr_kwargs
        )

        self._u_last = jerk_enu
        self._yaw_cmd_prev_enu = yaw_cmd_enu
        return p_cmd, q_cmd, r_cmd, thrust, ref_traj

    def _prime_warmstart(self, x0, u0):
        self._x_ws = np.repeat(x0.reshape(-1, 1), self.horizon + 1, axis=1)
        self._u_ws = np.repeat(u0.reshape(-1, 1), self.horizon, axis=1)
        self._s_ws.fill(0.0)
        self._vs_ws.fill(0.0)
        self._warm_started = True
