"""Feasible exogenous velocity-attitude references without position feedback."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as functional

from drone_ccm.geometry import log as so3_log


def _attitude_from_acceleration_yaw(
    acceleration: Tensor,
    yaw: Tensor,
    *,
    gravity: float,
) -> tuple[Tensor, Tensor]:
    """Lifts inertial acceleration and yaw to attitude and specific thrust."""
    if yaw.ndim == acceleration.ndim:
        yaw = yaw.squeeze(-1)
    e3 = torch.tensor(
        [0.0, 0.0, 1.0],
        device=acceleration.device,
        dtype=acceleration.dtype,
    ).expand_as(acceleration)
    force = acceleration + gravity * e3
    thrust = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    if bool((thrust < 1.0e-6).any()):
        raise ValueError("Acceleration plus gravity must be nonzero")
    body_z = force / thrust
    zeros = torch.zeros_like(yaw)
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw), zeros), dim=-1)
    body_y_raw = torch.linalg.cross(body_z, heading, dim=-1)
    if bool((torch.linalg.vector_norm(body_y_raw, dim=-1) < 1.0e-6).any()):
        raise ValueError("Thrust direction is singular with the yaw heading")
    body_y = functional.normalize(body_y_raw, dim=-1)
    body_x = torch.linalg.cross(body_y, body_z, dim=-1)
    return torch.stack((body_x, body_y, body_z), dim=-1), thrust


def _feasible_reference(
    acceleration: Tensor,
    yaw: Tensor,
    *,
    gravity: float,
    next_acceleration: Tensor,
    next_yaw: Tensor,
    time_step: float,
) -> tuple[Tensor, Tensor]:
    """Builds a dynamically feasible attitude and physical CTBR reference."""
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    rotation, thrust = _attitude_from_acceleration_yaw(
        acceleration,
        yaw,
        gravity=gravity,
    )
    next_rotation, _ = _attitude_from_acceleration_yaw(
        next_acceleration,
        next_yaw,
        gravity=gravity,
    )
    body_rate = so3_log(rotation.transpose(-1, -2) @ next_rotation) / time_step
    return rotation, torch.cat((thrust, body_rate), dim=-1)


def _quintic_envelope(time: float, duration: float) -> tuple[float, float]:
    if duration <= 0.0:
        return 1.0, 0.0
    ratio = min(max(time / duration, 0.0), 1.0)
    value = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5
    derivative = (
        30.0 * ratio**2 - 60.0 * ratio**3 + 30.0 * ratio**4
    ) / duration
    return value, derivative


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _blend_target(
    time: float,
    target_velocity: Tensor,
    target_acceleration: Tensor,
    target_yaw: Tensor,
    *,
    initial_velocity: Tensor,
    initial_yaw: Tensor,
    transition_duration: float,
    blend_yaw: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Blends an engagement state into an exogenous reference target."""
    blend, blend_rate = _quintic_envelope(time, transition_duration)
    velocity_delta = target_velocity - initial_velocity
    velocity = initial_velocity + blend * velocity_delta
    acceleration = blend_rate * velocity_delta + blend * target_acceleration
    if blend_yaw:
        yaw_delta = _wrap_angle(target_yaw - initial_yaw)
        yaw = initial_yaw + blend * yaw_delta
    else:
        yaw = target_yaw
    return velocity, acceleration, yaw


def hover_reference(
    time: float,
    template: Tensor,
    *,
    initial_velocity: Tensor,
    initial_yaw: Tensor,
    transition_duration: float,
    time_step: float,
    gravity: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns a smooth transition from the engagement state to hover."""
    zero = torch.zeros_like(template)
    return tracking_reference_from_targets(
        time,
        zero,
        zero,
        initial_yaw,
        next_target_velocity=zero,
        next_target_acceleration=zero,
        next_target_yaw=initial_yaw,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        transition_duration=transition_duration,
        time_step=time_step,
        gravity=gravity,
    )


def tracking_reference_from_targets(
    time: float,
    target_velocity: Tensor,
    target_acceleration: Tensor,
    target_yaw: Tensor,
    *,
    next_target_velocity: Tensor,
    next_target_acceleration: Tensor,
    next_target_yaw: Tensor,
    initial_velocity: Tensor,
    initial_yaw: Tensor,
    transition_duration: float,
    time_step: float,
    gravity: float,
    blend_yaw: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lifts externally generated velocity targets to a feasible reference."""
    velocity, acceleration, yaw = _blend_target(
        time,
        target_velocity,
        target_acceleration,
        target_yaw,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        transition_duration=transition_duration,
        blend_yaw=blend_yaw,
    )
    _, next_acceleration, next_yaw = _blend_target(
        time + time_step,
        next_target_velocity,
        next_target_acceleration,
        next_target_yaw,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        transition_duration=transition_duration,
        blend_yaw=blend_yaw,
    )
    rotation, control = _feasible_reference(
        acceleration,
        yaw,
        gravity=gravity,
        next_acceleration=next_acceleration,
        next_yaw=next_yaw,
        time_step=time_step,
    )
    return velocity, rotation, control
