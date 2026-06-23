"""ROS parameter configuration for MPC and MPCC."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter


_INTEGER_PARAMETERS = {"horizon"}
_VECTOR_PARAMETERS = {
    "jerk_min",
    "jerk_max",
    "jerk_delta_min",
    "jerk_delta_max",
    "velocity_min",
    "velocity_max",
    "acceleration_min",
    "acceleration_max",
    "position_weight",
    "jerk_weight",
    "jerk_delta_weight",
}
_COMMON_PARAMETER_NAMES = (
    "horizon",
    "mpc_dt",
    "control_dt",
    "yaw_kp",
    "yaw_rate_limit",
    "body_rate_limit",
    "hover_thrust",
    "gravity",
    "thrust_min",
    "thrust_max",
    "acceleration_alpha",
    "jerk_min",
    "jerk_max",
    "jerk_delta_min",
    "jerk_delta_max",
    "velocity_min",
    "velocity_max",
    "acceleration_min",
    "acceleration_max",
    "velocity_slack_weight",
    "acceleration_slack_weight",
    "position_weight",
    "jerk_weight",
    "jerk_delta_weight",
    "terminal_weight",
)
_MPCC_PARAMETER_NAMES = (
    "contour_weight",
    "lag_weight",
    "progress_weight",
    "terminal_progress_weight",
    "progress_rate_min",
    "progress_rate_max",
)


@dataclass(frozen=True)
class ControllerConfig:
    """Validated controller and optimizer parameters."""

    horizon: int
    mpc_dt: float
    control_dt: float
    yaw_kp: float
    yaw_rate_limit: float
    body_rate_limit: float
    hover_thrust: float
    gravity: float
    thrust_min: float
    thrust_max: float
    acceleration_alpha: float
    jerk_min: np.ndarray
    jerk_max: np.ndarray
    jerk_delta_min: np.ndarray
    jerk_delta_max: np.ndarray
    velocity_min: np.ndarray
    velocity_max: np.ndarray
    acceleration_min: np.ndarray
    acceleration_max: np.ndarray
    velocity_slack_weight: float
    acceleration_slack_weight: float
    position_weight: np.ndarray
    jerk_weight: np.ndarray
    jerk_delta_weight: np.ndarray
    terminal_weight: float
    contour_weight: float | None = None
    lag_weight: float | None = None
    progress_weight: float | None = None
    terminal_progress_weight: float | None = None
    progress_rate_min: float | None = None
    progress_rate_max: float | None = None

    def mpc_params(self) -> SimpleNamespace:
        """Return the parameter names consumed by the MPC solver."""
        return SimpleNamespace(
            nx=9,
            nu=3,
            horizon=self.horizon,
            dt=self.mpc_dt,
            u_min=self.jerk_min,
            u_max=self.jerk_max,
            du_min=self.jerk_delta_min,
            du_max=self.jerk_delta_max,
            Q=np.diag(self.position_weight),
            R=np.diag(self.jerk_weight),
            Rd=np.diag(self.jerk_delta_weight),
            terminal=self.terminal_weight,
            track_idx=np.arange(3),
            v_min=self.velocity_min,
            v_max=self.velocity_max,
            a_min=self.acceleration_min,
            a_max=self.acceleration_max,
            v_slack_weight=self.velocity_slack_weight,
            a_slack_weight=self.acceleration_slack_weight,
        )

    def mpcc_params(self) -> SimpleNamespace:
        """Extend the common optimizer parameters with MPCC weights."""
        mpcc_values = (
            self.contour_weight,
            self.lag_weight,
            self.progress_weight,
            self.terminal_progress_weight,
            self.progress_rate_min,
            self.progress_rate_max,
        )
        if any(value is None for value in mpcc_values):
            raise ValueError("MPCC configuration parameters are missing")
        params = vars(self.mpc_params()).copy()
        params.update(
            q_contour=self.contour_weight,
            q_lag=self.lag_weight,
            q_progress=self.progress_weight,
            q_terminal_s=self.terminal_progress_weight,
            vs_min=self.progress_rate_min,
            vs_max=self.progress_rate_max,
        )
        return SimpleNamespace(**params)


def declare_controller_config(node: Node, *, include_mpcc: bool) -> ControllerConfig:
    """Read required controller parameters supplied by a ROS YAML file."""
    raw_values = {}
    names = _COMMON_PARAMETER_NAMES
    if include_mpcc:
        names += _MPCC_PARAMETER_NAMES
    for name in names:
        if name in _VECTOR_PARAMETERS:
            parameter_type = Parameter.Type.DOUBLE_ARRAY
        elif name in _INTEGER_PARAMETERS:
            parameter_type = Parameter.Type.INTEGER
        else:
            parameter_type = Parameter.Type.DOUBLE
        value = node.declare_parameter(name, parameter_type).value
        if value is None:
            raise ValueError(
                f"Missing required parameter '{name}'; start with the drone_cnt launch file"
            )
        raw_values[name] = value
    return controller_config_from_mapping(raw_values)


def controller_config_from_mapping(values: Mapping[str, object]) -> ControllerConfig:
    """Build a validated controller configuration from YAML-like values."""
    names = _COMMON_PARAMETER_NAMES
    if any(name in values for name in _MPCC_PARAMETER_NAMES):
        names += _MPCC_PARAMETER_NAMES
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Missing controller parameters: {', '.join(missing)}")
    return _build_config({name: values[name] for name in names})


def _build_config(raw_values) -> ControllerConfig:
    values = {}
    for name, value in raw_values.items():
        if name in _VECTOR_PARAMETERS:
            values[name] = _vector(value, name)
        elif name in _INTEGER_PARAMETERS:
            values[name] = int(value)
        else:
            values[name] = float(value)

    config = ControllerConfig(**values)
    _validate(config)
    return config


def _vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Parameter '{name}' must contain three finite numbers")
    return vector


def _validate(config: ControllerConfig) -> None:
    if config.horizon < 1:
        raise ValueError("Parameter 'horizon' must be positive")
    if config.mpc_dt <= 0.0 or config.control_dt <= 0.0:
        raise ValueError("Controller time steps must be positive")
    if not 0.0 <= config.acceleration_alpha <= 1.0:
        raise ValueError("Parameter 'acceleration_alpha' must be in [0, 1]")
    if not config.thrust_min <= config.hover_thrust <= config.thrust_max:
        raise ValueError("Expected thrust_min <= hover_thrust <= thrust_max")
