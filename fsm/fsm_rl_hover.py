from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fsm_log import TickData
from .fsm_mpc import MPCBehaviorBase
from .fsm_spec import (
    EVENT_LAND,
    EVENT_TAKEOFF,
    STATE_HOVER,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_READY,
    STATE_RETURN_HOVER,
    STATE_TRACKING,
)


@dataclass(frozen=True)
class RLHoverConfig:
    checkpoint: str
    device: str = "cpu"
    thrust_ratio_min: float = 0.4
    thrust_ratio_max: float = 1.6
    body_rate_limit: tuple[float, float, float] = (1.5, 1.5, 1.0)
    hover_thrust: float = 0.58
    thrust_min: float = 0.1
    thrust_max: float = 0.9
    max_position_error: float = 4.0
    fallback_to_mpc_tracking: bool = True


class _HoverPolicyNet:
    """Small wrapper around the skrl MLP policy used by Xsim_new hover."""

    def __init__(self, checkpoint: Path, device: str):
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._model = torch.nn.Sequential(
            torch.nn.Linear(17, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 4),
        ).to(self._device)
        payload = torch.load(str(checkpoint), map_location=self._device)
        state = payload.get("policy", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported hover checkpoint payload: {type(payload)!r}")
        state = {
            key.replace("net_container.", ""): value
            for key, value in state.items()
            if key.startswith("net_container.")
        }
        self._model.load_state_dict(state, strict=True)
        self._model.eval()

    def act(self, observation: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs = torch.as_tensor(
            observation.reshape(1, -1), dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            action = self._model(obs).clamp(-1.0, 1.0)
        return action.cpu().numpy().reshape(4)


class RLHoverBehavior(MPCBehaviorBase):
    """FSM behavior that replaces hover-state MPC setpoints with an RL policy."""

    def __init__(
        self,
        *,
        node,
        logger,
        tracker,
        cfg: RLHoverConfig,
        takeoff_height: float = 1.0,
        takeoff_velocity: float = 0.5,
    ):
        super().__init__(node=node, logger=logger, tracker=tracker)
        self._cfg = cfg
        self._takeoff_height = float(takeoff_height)
        self._takeoff_velocity = float(takeoff_velocity)
        self._tracking_dt = float(getattr(tracker, "dt", 0.1)) if tracker is not None else 0.1
        self._tracking_horizon = int(getattr(tracker, "horizon", 15)) if tracker is not None else 15
        self._policy = _HoverPolicyNet(Path(cfg.checkpoint).expanduser(), cfg.device)
        self._last_action = np.zeros(4, dtype=np.float32)

    def on_enter(self, state: str, event_name: str) -> None:
        if self._tracker is not None:
            self._tracker.reset_warmstart()
        self.clear_hover_target()
        self._last_action.fill(0.0)
        self._logger.log_event(state=state, event=event_name)

        if state == STATE_READY:
            self.offboard()
        elif state == STATE_HOVER_START and event_name == EVENT_TAKEOFF:
            self.offboard()
            self.arm()
        elif state == STATE_PREFLIGHT and event_name == EVENT_LAND:
            self.land()
            self._start_point_enu = None
            self._ref_cmd_enu = None

    def update_ref_cmd_enu(self, ref_cmd_enu: np.ndarray):
        if ref_cmd_enu is None or int(ref_cmd_enu.shape[0]) < 1:
            return
        ref = np.asarray(ref_cmd_enu, dtype=float).reshape(-1, 3)
        if self._start_point_enu is None:
            self._start_point_enu = ref[0].copy()
        self._ref_cmd_enu = ref[: self._tracking_horizon + 1].T.copy()

    def tick(
        self,
        fsm_state: str,
        dt: float,
        obstacle_points: np.ndarray | None = None,
    ) -> None:
        if (
            self._vehicle_state is None
            or self._disengaged
            or fsm_state == STATE_PREFLIGHT
        ):
            return

        self._px4_bridge.publish_offboard_mode()

        if (
            fsm_state == STATE_TRACKING
            and self._cfg.fallback_to_mpc_tracking
            and self._tracker is not None
        ):
            self._tick_tracking_mpc(fsm_state, dt, obstacle_points)
            return
        if fsm_state in (STATE_READY, STATE_HOVER_START) and self._tracker is not None:
            self._tick_hover_mpc(fsm_state, dt, obstacle_points)
            return

        hover_target = self._rl_target_enu(fsm_state)

        action = self._policy.act(self._build_observation(hover_target))
        self._last_action = action.astype(np.float32)
        p_cmd, q_cmd, r_cmd, thrust = self._action_to_ctbr(action)
        self._px4_bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        self._logger.log_tick(
            TickData(
                fsm_state=str(fsm_state),
                pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                ref_enu=np.asarray(hover_target, dtype=float).reshape(3),
                vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                yaw_enu=float(self._vehicle_state.yaw_enu),
                p_cmd=float(p_cmd),
                q_cmd=float(q_cmd),
                r_cmd=float(r_cmd),
                thrust_cmd=float(thrust),
                yaw_cmd_enu=float(self._yaw_cmd_enu),
            )
        )

    def _rl_target_enu(self, fsm_state: str) -> np.ndarray:
        if fsm_state == STATE_TRACKING and self._ref_cmd_enu is not None:
            return np.asarray(self._ref_cmd_enu[:, -1], dtype=float).reshape(3)

        hover_target = self.get_hover_target(fsm_state)
        if hover_target is None:
            return np.asarray(self._vehicle_state.position_enu, dtype=float).reshape(3)
        return np.asarray(hover_target, dtype=float).reshape(3)

    def _build_observation(self, hover_target_enu: np.ndarray) -> np.ndarray:
        # Xsim's policy was trained in an NED-like world frame: x=north,
        # y=east, z=down. DroneTracking stores vehicle state in ENU, so convert
        # the goal-relative state before passing it to the policy.
        pos_enu = np.asarray(self._vehicle_state.position_enu, dtype=float).reshape(3)
        target_enu = np.asarray(hover_target_enu, dtype=float).reshape(3)
        pos_err_enu = pos_enu - target_enu
        pos_err = np.array(
            [pos_err_enu[1], pos_err_enu[0], -pos_err_enu[2]],
            dtype=np.float32,
        )
        pos_err = np.clip(
            pos_err, -float(self._cfg.max_position_error), float(self._cfg.max_position_error)
        )
        vel_enu = np.asarray(self._vehicle_state.velocity_enu, dtype=float).reshape(3)
        vel = np.array([vel_enu[1], vel_enu[0], -vel_enu[2]], dtype=np.float32)
        quat_wxyz = np.asarray(
            getattr(self._vehicle_state, "quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(4)
        quat_norm = float(np.linalg.norm(quat_wxyz))
        if quat_norm > 1.0e-6:
            quat_wxyz = quat_wxyz / quat_norm
        else:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega_body = np.asarray(
            getattr(self._vehicle_state, "body_rates", np.zeros(3, dtype=float)),
            dtype=np.float32,
        ).reshape(3)
        return np.concatenate(
            [
                pos_err.astype(np.float32),
                vel.astype(np.float32),
                quat_wxyz.astype(np.float32),
                omega_body,
                self._last_action.astype(np.float32),
            ]
        )

    def _action_to_ctbr(self, action: np.ndarray) -> tuple[float, float, float, float]:
        action = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        thrust_center = 0.5 * (self._cfg.thrust_ratio_min + self._cfg.thrust_ratio_max)
        thrust_scale = 0.5 * (self._cfg.thrust_ratio_max - self._cfg.thrust_ratio_min)
        thrust = float(self._cfg.hover_thrust) * float(thrust_center + thrust_scale * action[0])
        thrust = float(np.clip(thrust, self._cfg.thrust_min, self._cfg.thrust_max))
        rates = action[1:4] * np.asarray(self._cfg.body_rate_limit, dtype=float)
        return float(rates[0]), float(rates[1]), float(rates[2]), thrust

    def _build_hover_reference(self, fsm_state: str) -> np.ndarray | None:
        hover_target = self.get_hover_target(fsm_state)
        if hover_target is None:
            return None

        current_position = np.asarray(self._vehicle_state.position_enu, dtype=float)
        hover_target = np.asarray(hover_target, dtype=float).reshape(3)
        dist2target = float(np.linalg.norm(hover_target - current_position))
        if dist2target < 0.01:
            return np.repeat(
                hover_target.reshape(3, 1), self._tracking_horizon + 1, axis=1
            )

        if fsm_state == STATE_HOVER_START:
            step = self._takeoff_velocity * self._tracking_dt
            samples = step * np.arange(self._tracking_horizon + 1, dtype=float)
            progress = np.clip(samples / dist2target, 0.0, 1.0).reshape(1, -1)
        else:
            progress = np.linspace(
                0.0, 1.0, self._tracking_horizon + 1, dtype=float
            ).reshape(1, -1)

        delta = (hover_target - current_position).reshape(3, 1)
        return current_position.reshape(3, 1) + delta * progress

    def _tick_hover_mpc(
        self,
        fsm_state: str,
        dt: float,
        obstacle_points: np.ndarray | None,
    ) -> None:
        ref_cmd_enu = self._build_hover_reference(fsm_state)
        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._vehicle_state.position_enu,
            self._vehicle_state.velocity_enu,
            self._vehicle_state.accel_enu,
            self._vehicle_state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=ref_cmd_enu,
            path_points_enu=None,
            obstacle_points_enu=obstacle_points,
            log_solver=False,
        )
        self._px4_bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)
        debug = getattr(self._tracker, "last_debug", {}) or {}
        self._logger.log_tick(
            TickData(
                fsm_state=str(fsm_state),
                pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                ref_enu=(
                    np.asarray(ref_cmd_enu[:, 0].reshape(3).copy())
                    if ref_cmd_enu is not None
                    else None
                ),
                vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                yaw_enu=float(self._vehicle_state.yaw_enu),
                p_cmd=float(p_cmd),
                q_cmd=float(q_cmd),
                r_cmd=float(r_cmd),
                thrust_cmd=float(thrust),
                acc_est_enu=debug.get("a_est_enu"),
                jerk_cmd_enu=debug.get("jerk_cmd_enu"),
                acc_cmd_enu=debug.get("acc_cmd_enu"),
                yaw_cmd_enu=debug.get("yaw_cmd_enu"),
                yaw_rate_cmd_enu=debug.get("yaw_rate_cmd_enu"),
            )
        )

    def _tick_tracking_mpc(
        self,
        fsm_state: str,
        dt: float,
        obstacle_points: np.ndarray | None,
    ) -> None:
        ref_cmd_enu = self._ref_cmd_enu
        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._vehicle_state.position_enu,
            self._vehicle_state.velocity_enu,
            self._vehicle_state.accel_enu,
            self._vehicle_state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=ref_cmd_enu,
            path_points_enu=None,
            obstacle_points_enu=obstacle_points,
            log_solver=False,
        )
        self._px4_bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)
        self._logger.log_tick(
            TickData(
                fsm_state=str(fsm_state),
                pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                ref_enu=(
                    np.asarray(ref_cmd_enu[:, 0].reshape(3).copy())
                    if ref_cmd_enu is not None
                    else None
                ),
                vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                yaw_enu=float(self._vehicle_state.yaw_enu),
                p_cmd=float(p_cmd),
                q_cmd=float(q_cmd),
                r_cmd=float(r_cmd),
                thrust_cmd=float(thrust),
                yaw_cmd_enu=float(self._yaw_cmd_enu),
            )
        )
