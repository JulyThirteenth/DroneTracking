"""ROS-independent MPC and MPCC CTBR controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drone_cnt.cfg import ControllerConfig
from drone_cnt.osqp import MPCCOSQP, MPCOSQP
from drone_cnt.utils import (
    Polyline3D,
    as_vec3,
    enu_to_ned,
    flatness_to_ctbr,
    wrap_pi,
    yaw_enu_to_ned,
    yaw_rate_enu_to_ned,
)


Ctbr = tuple[float, float, float, float]


class CtbrControllerBase(ABC):
    """Share state estimation, warm start and flatness conversion."""

    def __init__(self, config: ControllerConfig) -> None:
        """Initialize estimator and optimizer warm-start storage."""
        self.config = config
        self._warm_started = False
        self._last_jerk = np.zeros(3)
        self._acceleration = np.zeros(3)
        self._previous_yaw_command: float | None = None
        self._state_warmstart = np.zeros((9, config.horizon + 1))
        self._jerk_warmstart = np.zeros((3, config.horizon))

    def reset(self) -> None:
        """Clear estimator and optimizer warm-start state."""
        self._warm_started = False
        self._last_jerk.fill(0.0)
        self._acceleration.fill(0.0)
        self._previous_yaw_command = None
        self._state_warmstart.fill(0.0)
        self._jerk_warmstart.fill(0.0)
        self._reset_extra_warmstart()

    def step(
        self,
        position_enu,
        velocity_enu,
        acceleration_enu,
        yaw_enu: float,
        *,
        yaw_cmd_enu: float,
        log_solver: bool = False,
        **reference,
    ) -> Ctbr:
        """Compute one collective-thrust/body-rate command."""
        state = self._update_state(position_enu, velocity_enu, acceleration_enu)
        if not self._warm_started:
            self._prime_warmstart(state)

        state_solution, jerk_solution = self._solve(
            state,
            self._last_jerk,
            log_solver=log_solver,
            **reference,
        )
        acceleration_command = state_solution[6:9, 1]
        jerk_command = jerk_solution[:, 0]
        yaw_rate = self._yaw_rate(float(yaw_enu), float(yaw_cmd_enu))

        command = flatness_to_ctbr(
            enu_to_ned(acceleration_command),
            enu_to_ned(jerk_command),
            yaw_enu_to_ned(yaw_enu),
            yaw_rate_enu_to_ned(yaw_rate),
            hover_thrust=self.config.hover_thrust,
            gravity=self.config.gravity,
            thrust_min=self.config.thrust_min,
            thrust_max=self.config.thrust_max,
        )
        limit = self.config.body_rate_limit
        roll, pitch, yaw_r, thrust = command
        command = (
            float(np.clip(roll, -limit, limit)),
            float(np.clip(pitch, -limit, limit)),
            float(np.clip(yaw_r, -limit, limit)),
            thrust,
        )
        self._last_jerk = jerk_command.copy()
        self._previous_yaw_command = float(yaw_cmd_enu)
        return command

    def _update_state(self, position, velocity, measured_acceleration) -> np.ndarray:
        measured = as_vec3(measured_acceleration)
        if self._warm_started:
            predicted = self._acceleration + self._last_jerk * self.config.control_dt
        else:
            predicted = measured
        alpha = self.config.acceleration_alpha
        self._acceleration = (1.0 - alpha) * predicted + alpha * measured
        return np.hstack((as_vec3(position), as_vec3(velocity), self._acceleration))

    def _prime_warmstart(self, state: np.ndarray) -> None:
        self._state_warmstart[:] = state[:, None]
        self._jerk_warmstart.fill(0.0)
        self._warm_started = True

    def _yaw_rate(self, yaw_enu: float, yaw_command_enu: float) -> float:
        feedforward = 0.0
        if self._previous_yaw_command is not None:
            feedforward = wrap_pi(
                yaw_command_enu - self._previous_yaw_command
            ) / self.config.control_dt
        error = wrap_pi(yaw_command_enu - yaw_enu)
        yaw_rate = feedforward + self.config.yaw_kp * error / self.config.control_dt
        return float(
            np.clip(
                yaw_rate,
                -self.config.yaw_rate_limit,
                self.config.yaw_rate_limit,
            )
        )

    def _reset_extra_warmstart(self) -> None:
        pass

    @abstractmethod
    def _solve(
        self,
        state: np.ndarray,
        previous_jerk: np.ndarray,
        *,
        log_solver: bool,
        **reference,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one optimizer step."""


class MpcCtbrController(CtbrControllerBase):
    """OSQP trajectory-tracking MPC."""

    def __init__(self, config: ControllerConfig) -> None:
        """Build and configure the MPC optimizer."""
        super().__init__(config)
        self._optimizer = MPCOSQP(config.mpc_params())
        self._optimizer.setup()

    def _solve(
        self,
        state: np.ndarray,
        previous_jerk: np.ndarray,
        *,
        log_solver: bool,
        ref_traj_enu=None,
        **_,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference = ref_traj_enu
        if reference is None:
            reference = np.repeat(state[:3, None], self.config.horizon + 1, axis=1)
        solution = self._optimizer.solve(
            state,
            previous_jerk,
            reference,
            self._state_warmstart,
            self._jerk_warmstart,
            log=log_solver,
        )
        self._state_warmstart, self._jerk_warmstart = solution
        return solution


class MpccCtbrController(CtbrControllerBase):
    """OSQP contour-following MPCC."""

    def __init__(self, config: ControllerConfig) -> None:
        """Build and configure the MPCC optimizer."""
        super().__init__(config)
        self._optimizer = MPCCOSQP(config.mpcc_params())
        self._optimizer.setup()
        self._progress_warmstart = np.zeros((1, config.horizon + 1))
        self._progress_rate_warmstart = np.zeros((1, config.horizon))

    def _reset_extra_warmstart(self) -> None:
        self._progress_warmstart.fill(0.0)
        self._progress_rate_warmstart.fill(0.0)

    def _solve(
        self,
        state: np.ndarray,
        previous_jerk: np.ndarray,
        *,
        log_solver: bool,
        path_points_enu=None,
        **_,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = Polyline3D.from_points(path_points_enu)
        state_solution, jerk_solution, progress, progress_rate = (
            self._optimizer.solve(
                state,
                previous_jerk,
                path.points,
                self._state_warmstart,
                self._jerk_warmstart,
                self._progress_warmstart,
                self._progress_rate_warmstart,
                log=log_solver,
            )
        )
        self._state_warmstart = state_solution
        self._jerk_warmstart = jerk_solution
        self._progress_warmstart = progress
        self._progress_rate_warmstart = progress_rate
        return state_solution, jerk_solution
