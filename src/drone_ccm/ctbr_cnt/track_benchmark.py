"""Aligned Neural CCM/SO(3)-CTBR velocity-tracking benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SO3, UnitQuaternion
from spatialmath.base import vex
import torch
import yaml

from uav_ccm import (
    PEGASUS_CONTROL_LOWER,
    PEGASUS_CONTROL_UPPER,
    ctbr_numpy,
    load_models,
)
from uav_ego_ccm import load_models as load_ego_models
from uav_sim import CtbrCnt, Quadrotor, QuadrotorParams, q_rot
from uav_so3 import SO3Controller, attitude_from_force_acceleration

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROLLERS = ("ccm", "ego-ccm", "so3-ctbr", "so3-full")


@dataclass(frozen=True)
class TrackCase:
    name: str
    slug: str
    duration: float
    trajectory: callable
    speed_scale: float = 1.0


@dataclass
class TrackResult:
    case: TrackCase
    controller_type: str
    times: np.ndarray
    states: np.ndarray
    position_ref: np.ndarray
    velocity_ref: np.ndarray
    yaw_ref: np.ndarray
    nominal_attitude_ref: np.ndarray
    internal_attitude_ref: np.ndarray
    ctbr_ref: np.ndarray
    raw_ctbr: np.ndarray
    applied_ctbr: np.ndarray
    applied_wrench: np.ndarray
    normalized_thrust: np.ndarray
    position_error: np.ndarray
    velocity_error: np.ndarray
    yaw_error: np.ndarray
    ctbr_clipped: np.ndarray
    normalized_ctbr_violation: np.ndarray
    allocation_saturated: np.ndarray

    def metrics(self, settling_time=3.0):
        use = self.times >= min(settling_time, 0.2 * self.times[-1])
        velocity_bias = np.mean(self.velocity_error[use], axis=0)

        def rotation_distance(actual, reference):
            relative = reference.T @ actual
            cosine = np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0)
            return np.arccos(cosine)

        rotations = [q_rot(q) for q in self.states[:, 6:10]]
        nominal_attitude_error = np.asarray(
            [
                rotation_distance(R, R_ref)
                for R, R_ref in zip(rotations, self.nominal_attitude_ref)
            ]
        )
        internal_attitude_error = np.asarray(
            [
                rotation_distance(R, R_ref)
                for R, R_ref in zip(rotations, self.internal_attitude_ref)
            ]
        )
        attitude_reference_offset = np.asarray(
            [
                rotation_distance(R_internal, R_nominal)
                for R_internal, R_nominal in zip(
                    self.internal_attitude_ref, self.nominal_attitude_ref
                )
            ]
        )
        def angular_rmse_deg(values):
            return float(np.rad2deg(np.sqrt(np.mean(values[use] ** 2))))

        finite_thrust = self.normalized_thrust[
            np.isfinite(self.normalized_thrust)
        ]
        raw_clip_fraction = (
            np.nan
            if self.controller_type == "so3-full"
            else float(np.mean(self.ctbr_clipped))
        )
        max_ctbr_violation = (
            np.nan
            if self.controller_type == "so3-full"
            else float(np.max(self.normalized_ctbr_violation))
        )
        torque_norm = np.linalg.vector_norm(
            self.applied_wrench[use, 1:4], axis=1
        )
        return {
            "position_drift_rmse_m": float(
                np.sqrt(np.mean(self.position_error[use] ** 2))
            ),
            "velocity_rmse_mps": float(np.sqrt(np.mean(self.velocity_error[use] ** 2))),
            "velocity_bias_norm_mps": float(np.linalg.norm(velocity_bias)),
            "vertical_velocity_bias_mps": float(velocity_bias[2]),
            "final_position_error_m": float(np.linalg.norm(self.position_error[-1])),
            "nominal_attitude_rmse_deg": angular_rmse_deg(nominal_attitude_error),
            "internal_attitude_rmse_deg": angular_rmse_deg(internal_attitude_error),
            "attitude_reference_offset_rmse_deg": angular_rmse_deg(
                attitude_reference_offset
            ),
            "yaw_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(self.yaw_error[use] ** 2)))
            ),
            "max_position_drift_m": float(
                np.max(np.linalg.norm(self.position_error, axis=1))
            ),
            "max_velocity_error_mps": float(
                np.max(np.linalg.norm(self.velocity_error, axis=1))
            ),
            "raw_ctbr_violation_fraction": raw_clip_fraction,
            "max_normalized_ctbr_violation": max_ctbr_violation,
            "normalized_thrust_min": (
                float(np.min(finite_thrust)) if finite_thrust.size else np.nan
            ),
            "normalized_thrust_max": (
                float(np.max(finite_thrust)) if finite_thrust.size else np.nan
            ),
            "allocation_saturation_fraction": float(np.mean(self.allocation_saturated)),
            "torque_rms_nm": float(np.sqrt(np.mean(torque_norm**2))),
            "max_torque_norm_nm": float(np.max(torque_norm)),
            "finite": bool(np.all(np.isfinite(self.states))),
        }


def make_track_cases(speed_scale=1.0):
    """Create fixed geometric paths with common time scaling."""
    if speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive")

    def circle(t):
        radius, w = 1.6, 0.42
        p = np.array([radius * np.cos(w * t), radius * np.sin(w * t), -1.0])
        v = np.array([-radius * w * np.sin(w * t), radius * w * np.cos(w * t), 0.0])
        a = np.array(
            [-radius * w**2 * np.cos(w * t), -radius * w**2 * np.sin(w * t), 0.0]
        )
        j = np.array(
            [radius * w**3 * np.sin(w * t), -radius * w**3 * np.cos(w * t), 0.0]
        )
        return p, v, a, j

    def figure8(t):
        w, north, east = 0.42, 1.8, 1.15
        p = np.array([north * np.sin(w * t), east * np.sin(2 * w * t), -1.0])
        v = np.array([north * w * np.cos(w * t), 2 * east * w * np.cos(2 * w * t), 0.0])
        a = np.array(
            [-north * w**2 * np.sin(w * t), -4 * east * w**2 * np.sin(2 * w * t), 0.0]
        )
        j = np.array(
            [-north * w**3 * np.cos(w * t), -8 * east * w**3 * np.cos(2 * w * t), 0.0]
        )
        return p, v, a, j

    def helix(t):
        radius, w = 1.45, 0.48
        vertical_w = 0.5 * w
        p = np.array(
            [
                radius * np.cos(w * t),
                radius * np.sin(w * t),
                -1.0 - 0.35 * np.sin(vertical_w * t),
            ]
        )
        v = np.array(
            [
                -radius * w * np.sin(w * t),
                radius * w * np.cos(w * t),
                -0.35 * vertical_w * np.cos(vertical_w * t),
            ]
        )
        a = np.array(
            [
                -radius * w**2 * np.cos(w * t),
                -radius * w**2 * np.sin(w * t),
                0.35 * vertical_w**2 * np.sin(vertical_w * t),
            ]
        )
        j = np.array(
            [
                radius * w**3 * np.sin(w * t),
                -radius * w**3 * np.cos(w * t),
                0.35 * vertical_w**3 * np.cos(vertical_w * t),
            ]
        )
        return p, v, a, j

    def multi_sine(t):
        wx, wy, wz, phase = 0.52, 0.83, 0.37, 0.6
        p = np.array(
            [
                1.5 * np.sin(wx * t),
                np.sin(wy * t + phase),
                -1.0 - 0.3 * np.sin(wz * t),
            ]
        )
        v = np.array(
            [
                1.5 * wx * np.cos(wx * t),
                wy * np.cos(wy * t + phase),
                -0.3 * wz * np.cos(wz * t),
            ]
        )
        a = np.array(
            [
                -1.5 * wx**2 * np.sin(wx * t),
                -(wy**2) * np.sin(wy * t + phase),
                0.3 * wz**2 * np.sin(wz * t),
            ]
        )
        j = np.array(
            [
                -1.5 * wx**3 * np.cos(wx * t),
                -(wy**3) * np.cos(wy * t + phase),
                0.3 * wz**3 * np.cos(wz * t),
            ]
        )
        return p, v, a, j

    base_cases = (
        TrackCase("Circle", "circle", 2 * np.pi / 0.42, circle),
        TrackCase("Figure-8", "figure_8", 2 * np.pi / 0.42, figure8),
        TrackCase("3D helix", "3d_helix", 2 * np.pi / 0.48, helix),
        TrackCase("Multi-sine", "multi_sine", 18.0, multi_sine),
    )
    if speed_scale == 1.0:
        return base_cases

    scaled_cases = []
    for base_case in base_cases:

        def scaled_trajectory(t, trajectory=base_case.trajectory):
            p, v, a, jerk = trajectory(speed_scale * t)
            return p, speed_scale * v, speed_scale**2 * a, speed_scale**3 * jerk

        scaled_cases.append(
            TrackCase(
                base_case.name,
                base_case.slug,
                base_case.duration / speed_scale,
                scaled_trajectory,
                speed_scale,
            )
        )
    return tuple(scaled_cases)


def attitude_from_acceleration(acceleration, yaw, gravity):
    e3 = np.array([0.0, 0.0, 1.0])
    force_acceleration = gravity * e3 - acceleration
    collective = np.linalg.norm(force_acceleration)
    return attitude_from_force_acceleration(force_acceleration, yaw), collective


def trajectory_yaw(velocity, acceleration, fixed_yaw, yaw_mode):
    """Return reference yaw and yaw rate from the selected policy."""
    if yaw_mode == "fixed":
        return fixed_yaw, 0.0
    if yaw_mode != "velocity":
        raise ValueError("yaw_mode must be 'fixed' or 'velocity'")
    speed_squared = velocity[0] ** 2 + velocity[1] ** 2
    if speed_squared <= 1.0e-8:
        raise ValueError("velocity yaw requires nonzero horizontal speed")
    yaw = np.arctan2(velocity[1], velocity[0])
    yaw_rate = (
        velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
    ) / speed_squared
    return yaw, yaw_rate


def flatness_reference(
    case,
    time,
    gravity,
    fixed_yaw=0.0,
    yaw_mode="fixed",
    yaw_override=None,
    yaw_rate_override=None,
):
    """Return feasible ``(p*, v*, a*, yaw*, R*, [c*, omega*])``."""
    p_ref, v_ref, a_ref, jerk_ref = case.trajectory(time)
    if (yaw_override is None) != (yaw_rate_override is None):
        raise ValueError("yaw and yaw-rate overrides must be provided together")
    if yaw_override is None:
        yaw_ref, yaw_rate_ref = trajectory_yaw(
            v_ref, a_ref, fixed_yaw, yaw_mode
        )
    else:
        yaw_ref, yaw_rate_ref = yaw_override, yaw_rate_override
    R_ref, collective = attitude_from_acceleration(a_ref, yaw_ref, gravity)
    epsilon = 1e-4
    R_plus, _ = attitude_from_acceleration(
        a_ref + epsilon * jerk_ref,
        yaw_ref + epsilon * yaw_rate_ref,
        gravity,
    )
    R_minus, _ = attitude_from_acceleration(
        a_ref - epsilon * jerk_ref,
        yaw_ref - epsilon * yaw_rate_ref,
        gravity,
    )
    R_dot = (R_plus - R_minus) / (2.0 * epsilon)
    body_rate_ref = vex(0.5 * (R_ref.T @ R_dot - R_dot.T @ R_ref))
    return p_ref, v_ref, a_ref, yaw_ref, R_ref, np.r_[collective, body_rate_ref]


def reference_profile(
    case,
    times,
    gravity,
    fixed_yaw,
    yaw_mode,
    yaw_rate_limit,
):
    """Build one shared, rate-limited trajectory reference."""
    if yaw_rate_limit <= 0.0:
        raise ValueError("yaw_rate_limit must be positive")
    references = []
    previous_yaw = None
    for index, time in enumerate(times):
        _, velocity, acceleration, _ = case.trajectory(time)
        target_yaw, target_rate = trajectory_yaw(
            velocity, acceleration, fixed_yaw, yaw_mode
        )
        if yaw_mode == "fixed":
            yaw_ref, yaw_rate_ref = target_yaw, target_rate
        elif previous_yaw is None:
            yaw_ref = target_yaw
            yaw_rate_ref = np.clip(target_rate, -yaw_rate_limit, yaw_rate_limit)
        else:
            dt = times[index] - times[index - 1]
            yaw_error = np.arctan2(
                np.sin(target_yaw - previous_yaw),
                np.cos(target_yaw - previous_yaw),
            )
            yaw_rate_ref = np.clip(yaw_error / dt, -yaw_rate_limit, yaw_rate_limit)
            yaw_ref = previous_yaw + dt * yaw_rate_ref
        references.append(
            flatness_reference(
                case,
                time,
                gravity,
                fixed_yaw,
                yaw_mode,
                yaw_ref,
                yaw_rate_ref,
            )
        )
        previous_yaw = yaw_ref
    return references


def check_reference_feasibility(
    cases,
    gravity,
    frequency,
    fixed_yaw,
    yaw_mode,
    yaw_rate_limit,
    lower,
    upper,
):
    """Densely check every nominal CTBR against the fixed training domain."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    reports = []
    for case in cases:
        times = np.arange(0.0, case.duration + 1.0 / frequency, 1.0 / frequency)
        references = reference_profile(
            case, times, gravity, fixed_yaw, yaw_mode, yaw_rate_limit
        )
        controls = np.array([reference[-1] for reference in references])
        feasible = bool(np.all(controls >= lower) and np.all(controls <= upper))
        report = {
            "case": case.name,
            "speed_scale": case.speed_scale,
            "feasible": feasible,
            "ctbr_min": controls.min(axis=0).tolist(),
            "ctbr_max": controls.max(axis=0).tolist(),
        }
        reports.append(report)
        if not feasible:
            violation = np.maximum(lower - controls, 0.0) + np.maximum(
                controls - upper, 0.0
            )
            raise ValueError(
                f"infeasible flatness reference: {case.name} at {case.speed_scale:g}x; "
                f"max violation={violation.max(axis=0)}"
            )
    return reports


def bound_feedback(reference, feedback, lower, upper):
    """Apply the controller's smooth physical-bound map with NumPy."""
    span = upper - lower
    q = np.clip((reference - lower) / span, 1e-6, 1.0 - 1e-6)
    margin = np.maximum((reference - lower) * (upper - reference) / span, 1e-12)
    latent = np.log(q / (1.0 - q)) + feedback / margin
    return lower + span / (1.0 + np.exp(-np.clip(latent, -60.0, 60.0)))


def ego_ctbr_numpy(
    controller,
    velocity,
    rotation,
    velocity_ref,
    rotation_ref,
    acceleration_ref,
    control_ref,
):
    """Evaluate the ego-centric controller from a common trajectory reference."""
    e3 = np.array([0.0, 0.0, 1.0])
    gamma = rotation.T @ e3
    gamma_ref = rotation_ref.T @ e3
    velocity_error = rotation.T @ (velocity - velocity_ref)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    yaw_ref = np.arctan2(rotation_ref[1, 0], rotation_ref[0, 0])
    yaw_error = np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref))
    state = np.r_[gamma, velocity_error, yaw_error]
    cosine, sine = np.cos(yaw_ref), np.sin(yaw_ref)
    heading_alignment = np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    acceleration_aligned = heading_alignment @ acceleration_ref
    denominator = gamma_ref[1] ** 2 + gamma_ref[2] ** 2
    yaw_rate_ref = (
        control_ref[2] * gamma_ref[1] + control_ref[3] * gamma_ref[2]
    ) / max(denominator, 1.0e-6)
    parameter = next(controller.parameters())
    as_tensor = lambda value: torch.as_tensor(
        value, dtype=parameter.dtype, device=parameter.device
    )
    with torch.no_grad():
        control = controller(
            as_tensor(state),
            as_tensor(acceleration_aligned),
            as_tensor(yaw_rate_ref),
            as_tensor(control_ref),
        )
    return control.cpu().numpy()


def simulate_track(
    controller,
    base_params,
    case,
    seed=0,
    frequency=400.0,
    outer_frequency=100.0,
    initial_perturbation=False,
    controller_type="ccm",
    fixed_yaw=0.0,
    yaw_mode="fixed",
    yaw_rate_limit=0.5,
    control_lower=PEGASUS_CONTROL_LOWER,
    control_upper=PEGASUS_CONTROL_UPPER,
    hover_thrust=None,
    normalized_thrust_lower=None,
    normalized_thrust_upper=None,
    attitude_disturbance_torque=(0.0, 0.0, 0.0),
    initial_velocity_error_mps=0.35,
    initial_attitude_error_rad=np.deg2rad(15.0),
):
    if controller_type not in CONTROLLERS:
        raise ValueError(f"controller_type must be one of {CONTROLLERS}")
    if frequency <= 0.0 or outer_frequency <= 0.0 or outer_frequency > frequency:
        raise ValueError("frequencies must satisfy 0 < outer <= simulation")
    if initial_velocity_error_mps < 0.0 or initial_attitude_error_rad < 0.0:
        raise ValueError("initial error bounds must be nonnegative")
    thrust_mapping = (
        hover_thrust,
        normalized_thrust_lower,
        normalized_thrust_upper,
    )
    if any(value is not None for value in thrust_mapping) and not all(
        value is not None for value in thrust_mapping
    ):
        raise ValueError("PX4 thrust mapping requires hover, lower, and upper")
    outer_interval = round(frequency / outer_frequency)
    if not np.isclose(outer_interval * outer_frequency, frequency):
        raise ValueError("frequency must be an integer multiple of outer_frequency")

    dt = 1.0 / frequency
    times = np.arange(0.0, case.duration + dt, dt)
    references = reference_profile(
        case, times, base_params.g, fixed_yaw, yaw_mode, yaw_rate_limit
    )
    params = copy.deepcopy(base_params)
    uav = Quadrotor(params)
    ctbr_lower = np.asarray(control_lower, dtype=np.float64)
    ctbr_upper = np.asarray(control_upper, dtype=np.float64)
    if ctbr_lower.shape != (4,) or ctbr_upper.shape != (4,):
        raise ValueError("control_lower and control_upper must contain four values")
    if np.any(ctbr_lower >= ctbr_upper):
        raise ValueError("each control lower bound must be below its upper bound")
    ctbr_scales = np.maximum(ctbr_upper - ctbr_lower, 1e-9)

    geometric = (
        SO3Controller(params) if controller_type.startswith("so3-") else None
    )
    inner = (
        None
        if controller_type == "so3-full"
        else CtbrCnt(
            params.mass,
            thrust_accel_limits=(ctbr_lower[0], ctbr_upper[0]),
        )
    )

    states = np.zeros((len(times), 17))
    position_ref = np.zeros((len(times), 3))
    velocity_ref = np.zeros((len(times), 3))
    yaw_ref = np.zeros(len(times))
    nominal_attitude_ref = np.zeros((len(times), 3, 3))
    internal_attitude_ref = np.zeros((len(times), 3, 3))
    ctbr_ref_log = np.zeros((len(times), 4))
    raw_ctbr = np.full((len(times), 4), np.nan)
    applied_ctbr = np.full((len(times), 4), np.nan)
    applied_wrench = np.full((len(times), 4), np.nan)
    normalized_thrust = np.full(len(times), np.nan)
    ctbr_clipped = np.zeros(len(times), dtype=bool)
    normalized_ctbr_violation = np.zeros(len(times))
    allocation_saturated = np.zeros(len(times), dtype=bool)

    rng = np.random.default_rng(seed)
    p0, v0, _, _, R0, _ = references[0]
    if initial_perturbation:
        dp = np.zeros(3)  # position is intentionally absent from controller input
        dv = rng.uniform(
            -initial_velocity_error_mps, initial_velocity_error_mps, 3
        )
        dtheta = rng.uniform(
            -initial_attitude_error_rad, initial_attitude_error_rad, 3
        )
    else:
        dp, dv, dtheta = np.zeros(3), np.zeros(3), np.zeros(3)
    states[0, 0:3] = p0 + dp
    states[0, 3:6] = v0 + dv
    states[0, 6:10] = UnitQuaternion(R0 @ SO3.Exp(dtheta).R).vec
    states[0, 13:17] = np.sqrt(params.mass * params.g / (4 * params.thrust_coeff))

    S = np.diag([1.0, 1.0, -1.0])
    sat_positive = np.zeros(3, dtype=bool)
    sat_negative = np.zeros(3, dtype=bool)
    command = np.zeros(4)
    command_reference = np.zeros(4)
    controller_R_ref = R0.copy()
    disturbance_amplitude = np.asarray(
        attitude_disturbance_torque, dtype=np.float64
    )
    if disturbance_amplitude.shape != (3,) or np.any(disturbance_amplitude < 0.0):
        raise ValueError(
            "attitude_disturbance_torque must contain three nonnegative values"
        )
    disturbance_frequency = 2.0 * np.pi * np.array([0.7, 0.9, 0.5])
    disturbance_phase = np.array([0.0, 1.1, 2.0])
    np.random.seed(seed)

    for i, time in enumerate(times):
        reference = references[i]
        p_ref, v_ref, a_ref, yaw_ref_i, R_ref, ctbr_ref = reference
        position_ref[i], velocity_ref[i] = p_ref, v_ref
        yaw_ref[i] = yaw_ref_i
        nominal_attitude_ref[i] = R_ref
        ctbr_ref_log[i] = ctbr_ref
        R = q_rot(states[i, 6:10])
        external_torque = disturbance_amplitude * np.sin(
            disturbance_frequency * time + disturbance_phase
        )

        if i % outer_interval == 0:
            disturbance_force_world = np.zeros(3)
            command_reference = ctbr_ref.copy()
            if controller_type == "ccm":
                command_ccm = ctbr_numpy(
                    controller,
                    S @ states[i, 3:6],
                    S @ R @ S,
                    S @ v_ref,
                    S @ R_ref @ S,
                    np.r_[ctbr_ref[0], -S @ ctbr_ref[1:4]],
                )
                command = np.r_[command_ccm[0], -S @ command_ccm[1:4]]
                controller_R_ref = R_ref.copy()
            elif controller_type == "ego-ccm":
                acceleration_command = (
                    params.g * np.array([0.0, 0.0, 1.0])
                    - ctbr_ref[0] * R_ref[:, 2]
                )
                command_ego = ego_ctbr_numpy(
                    controller,
                    S @ states[i, 3:6],
                    S @ R @ S,
                    S @ v_ref,
                    S @ R_ref @ S,
                    S @ acceleration_command,
                    np.r_[ctbr_ref[0], -S @ ctbr_ref[1:4]],
                )
                command = np.r_[command_ego[0], -S @ command_ego[1:4]]
                controller_R_ref = R_ref.copy()
            else:
                geometric.update_reference(
                    states[i, 3:6],
                    v_ref,
                    a_ref,
                    yaw_ref_i,
                    outer_interval * dt,
                    disturbance_force_world,
                )
                controller_R_ref = geometric.R_ref.copy()
                if controller_type == "so3-ctbr":
                    command = geometric.control_ctbr(R)
                    command = bound_feedback(
                        command_reference,
                        command - command_reference,
                        ctbr_lower,
                        ctbr_upper,
                    )

        internal_attitude_ref[i] = controller_R_ref
        if controller_type == "so3-full":
            wrench = geometric.control(R, states[i, 10:13])
            collective_acceleration = wrench[0] / params.mass
        else:
            raw_ctbr[i] = command
            violation = np.maximum(ctbr_lower - command, 0.0) + np.maximum(
                command - ctbr_upper, 0.0
            )
            normalized_ctbr_violation[i] = np.max(violation / ctbr_scales)
            ctbr_clipped[i] = bool(np.any(violation > 1e-9))
            if ctbr_clipped[i]:
                raise RuntimeError(
                    "controller produced a CTBR command outside its bounds"
                )
            wrench = inner(
                states[i, 10:13], command, dt, sat_positive, sat_negative
            )
            applied_ctbr[i] = command
            collective_acceleration = command[0]
        applied_wrench[i] = wrench
        if hover_thrust is not None:
            normalized_thrust[i] = hover_thrust * collective_acceleration / params.g
            if not (
                normalized_thrust_lower - 1e-9
                <= normalized_thrust[i]
                <= normalized_thrust_upper + 1e-9
            ):
                raise RuntimeError("collective command violates PX4 thrust bounds")

        rotor_speed = uav.wrench_to_rotor_speed(wrench)
        achieved_wrench = uav.rotor_speed_to_wrench(rotor_speed)
        unallocated_wrench = wrench - achieved_wrench
        unallocated_torque = unallocated_wrench[1:4]
        sat_positive = unallocated_torque > 1e-6
        sat_negative = unallocated_torque < -1e-6
        allocation_saturated[i] = np.any(np.abs(unallocated_wrench) > 1e-6)
        if i + 1 < len(times):
            states[i + 1] = uav.step(
                states[i], rotor_speed, dt, external_torque=external_torque
            )

    actual_yaw = np.array(
        [np.arctan2(q_rot(q)[1, 0], q_rot(q)[0, 0]) for q in states[:, 6:10]]
    )
    yaw_error = np.arctan2(np.sin(actual_yaw - yaw_ref), np.cos(actual_yaw - yaw_ref))
    return TrackResult(
        case=case,
        controller_type=controller_type,
        times=times,
        states=states,
        position_ref=position_ref,
        velocity_ref=velocity_ref,
        yaw_ref=yaw_ref,
        nominal_attitude_ref=nominal_attitude_ref,
        internal_attitude_ref=internal_attitude_ref,
        ctbr_ref=ctbr_ref_log,
        raw_ctbr=raw_ctbr,
        applied_ctbr=applied_ctbr,
        applied_wrench=applied_wrench,
        normalized_thrust=normalized_thrust,
        position_error=states[:, 0:3] - position_ref,
        velocity_error=states[:, 3:6] - velocity_ref,
        yaw_error=yaw_error,
        ctbr_clipped=ctbr_clipped,
        normalized_ctbr_violation=normalized_ctbr_violation,
        allocation_saturated=allocation_saturated,
    )


def _plot_reference_velocity(ax, case, samples=36):
    points, vectors = [], []
    for time in np.linspace(0.0, case.duration, samples, endpoint=False):
        point, velocity, _, _ = case.trajectory(time)
        points.append(point)
        vectors.append(velocity)
    points, vectors = np.asarray(points), np.asarray(vectors)
    ax.quiver(
        points[:, 0],
        points[:, 1],
        -points[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        -vectors[:, 2],
        length=0.18,
        normalize=True,
        color="0.5",
        alpha=0.5,
    )


def plot_track_result(result, output_dir="fig"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case, times = result.case, result.times
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)

    ax_track = figure.add_subplot(2, 2, 1, projection="3d")
    _plot_reference_velocity(ax_track, case)
    reference = result.position_ref.copy()
    reference[:, 2] *= -1
    actual = result.states[:, 0:3].copy()
    actual[:, 2] *= -1
    ax_track.plot(
        *reference.T, "--", color="tab:orange", linewidth=2, label="reference"
    )
    ax_track.plot(*actual.T, color="tab:blue", linewidth=1.6, label="UAV")
    ax_track.scatter(*actual[0], color="tab:green", s=35, label="start")
    points = np.vstack((reference, actual))
    ranges = np.ptp(points, axis=0)
    z_center = 0.5 * (points[:, 2].min() + points[:, 2].max())
    z_span = max(ranges[2], 0.8)
    ax_track.set_zlim(z_center - 0.5 * z_span, z_center + 0.5 * z_span)
    ax_track.set_box_aspect((max(ranges[0], 1.0), max(ranges[1], 1.0), z_span))
    ax_track.set(
        xlabel="North [m]",
        ylabel="East [m]",
        zlabel="Altitude [m]",
        title=f"{case.name}: fixed-reference velocity tracking",
    )
    ax_track.legend()

    ax_position = figure.add_subplot(2, 2, 2)
    for axis, label in enumerate(("x", "y", "z")):
        ax_position.plot(times, result.position_error[:, axis], label=f"e_{label}")
    ax_position.set(
        xlabel="Time [s]",
        ylabel="Position drift [m]",
        title="Integrated position drift",
    )
    ax_position.legend(ncol=3)

    ax_velocity = figure.add_subplot(2, 2, 3)
    for axis, label in enumerate(("vx", "vy", "vz")):
        ax_velocity.plot(times, result.velocity_error[:, axis], label=f"e_{label}")
    ax_velocity.set(
        xlabel="Time [s]",
        ylabel="Velocity error [m/s]",
        title="Velocity tracking error",
    )
    ax_velocity.legend(ncol=3)

    ax_yaw = figure.add_subplot(2, 2, 4)
    ax_yaw.plot(times, np.rad2deg(result.yaw_error), color="tab:purple")
    ax_yaw.set(
        xlabel="Time [s]", ylabel="Yaw error [deg]", title="Yaw tracking error"
    )
    for axis in (ax_track, ax_position, ax_velocity, ax_yaw):
        axis.grid(alpha=0.3)
    path = output_dir / f"track_{case.slug}.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure, path


def _controller_name(controller_type):
    names = {
        "ccm": "CCM",
        "ego-ccm": "Ego-CCM",
        "so3-ctbr": "SO3-CTBR",
        "so3-full": "SO3-Full",
    }
    return names[controller_type]


def run_track_benchmark(
    checkpoint,
    vehicle_config,
    frequency,
    outer_frequency,
    robustness_seeds,
    output_dir,
    speed_scale,
    controller_type,
    fixed_yaw,
    yaw_mode,
    yaw_rate_limit,
    control_lower,
    control_upper,
    hover_thrust=None,
    normalized_thrust_lower=None,
    normalized_thrust_upper=None,
    attitude_disturbance_torque=(0.0, 0.0, 0.0),
    initial_velocity_error_mps=0.35,
    initial_attitude_error_rad=np.deg2rad(15.0),
):
    controller = None
    if controller_type in ("ccm", "ego-ccm"):
        loader = load_models if controller_type == "ccm" else load_ego_models
        controller, _, checkpoint_config = loader(checkpoint)
        if not np.array_equal(checkpoint_config["control_lower"], control_lower):
            raise ValueError("benchmark and checkpoint control_lower differ")
        if not np.array_equal(checkpoint_config["control_upper"], control_upper):
            raise ValueError("benchmark and checkpoint control_upper differ")
    controller_name = _controller_name(controller_type)
    params = QuadrotorParams.from_yaml(vehicle_config)
    cases = make_track_cases(speed_scale=speed_scale)

    display_results = {
        case.slug: simulate_track(
            controller,
            params,
            case,
            frequency=frequency,
            outer_frequency=outer_frequency,
            controller_type=controller_type,
            fixed_yaw=fixed_yaw,
            yaw_mode=yaw_mode,
            yaw_rate_limit=yaw_rate_limit,
            control_lower=control_lower,
            control_upper=control_upper,
            hover_thrust=hover_thrust,
            normalized_thrust_lower=normalized_thrust_lower,
            normalized_thrust_upper=normalized_thrust_upper,
            attitude_disturbance_torque=attitude_disturbance_torque,
        )
        for case in cases
    }
    robust_results = {
        case.slug: [
            simulate_track(
                controller,
                params,
                case,
                seed=seed,
                frequency=frequency,
                outer_frequency=outer_frequency,
                initial_perturbation=True,
                controller_type=controller_type,
                fixed_yaw=fixed_yaw,
                yaw_mode=yaw_mode,
                yaw_rate_limit=yaw_rate_limit,
                control_lower=control_lower,
                control_upper=control_upper,
                hover_thrust=hover_thrust,
                normalized_thrust_lower=normalized_thrust_lower,
                normalized_thrust_upper=normalized_thrust_upper,
                attitude_disturbance_torque=attitude_disturbance_torque,
                initial_velocity_error_mps=initial_velocity_error_mps,
                initial_attitude_error_rad=initial_attitude_error_rad,
            )
            for seed in robustness_seeds
        ]
        for case in cases
    }
    figures = {
        slug: plot_track_result(result, output_dir=output_dir)
        for slug, result in display_results.items()
    }

    output_dir = Path(output_dir)
    metrics_path = output_dir / "track_benchmark_metrics.csv"
    rows = []
    print(
        f"controller: {controller_name}; inner/sim: {frequency:.0f} Hz; "
        f"outer: {outer_frequency:.0f} Hz; speed: {speed_scale:g}x; "
        f"yaw: {yaw_mode}"
    )
    print("case        stable   vel RMSE   yaw RMSE   raw CTBR clip   allocation")
    for case in cases:
        metrics = [result.metrics() for result in robust_results[case.slug]]
        def mean(key):
            values = np.asarray([item[key] for item in metrics], dtype=np.float64)
            return np.nan if np.all(np.isnan(values)) else float(np.nanmean(values))

        row = {
            "case": case.name,
            "controller": controller_name,
            "speed_scale": speed_scale,
            "inner_frequency_hz": frequency,
            "outer_frequency_hz": outer_frequency,
            "initial_velocity_error_bound_mps": initial_velocity_error_mps,
            "initial_attitude_error_bound_rad": initial_attitude_error_rad,
            "motor_time_constant_s": params.motor_time_constant,
            "motor_noise_std_radps": params.motor_noise_std,
            "stable_runs": sum(item["finite"] for item in metrics),
            "runs": len(metrics),
            "position_drift_rmse_m": mean("position_drift_rmse_m"),
            "velocity_rmse_mps": mean("velocity_rmse_mps"),
            "velocity_bias_norm_mps": mean("velocity_bias_norm_mps"),
            "vertical_velocity_bias_mps": mean("vertical_velocity_bias_mps"),
            "final_position_error_m": mean("final_position_error_m"),
            "nominal_attitude_rmse_deg": mean("nominal_attitude_rmse_deg"),
            "internal_attitude_rmse_deg": mean("internal_attitude_rmse_deg"),
            "attitude_reference_offset_rmse_deg": mean(
                "attitude_reference_offset_rmse_deg"
            ),
            "yaw_rmse_deg": mean("yaw_rmse_deg"),
            "raw_ctbr_violation_fraction": mean("raw_ctbr_violation_fraction"),
            "max_normalized_ctbr_violation": mean("max_normalized_ctbr_violation"),
            "normalized_thrust_min": mean("normalized_thrust_min"),
            "normalized_thrust_max": mean("normalized_thrust_max"),
            "allocation_saturation_fraction": mean("allocation_saturation_fraction"),
            "torque_rms_nm": mean("torque_rms_nm"),
            "max_torque_norm_nm": mean("max_torque_norm_nm"),
        }
        rows.append(row)
        raw_clip = row["raw_ctbr_violation_fraction"]
        raw_text = "n/a" if np.isnan(raw_clip) else f"{raw_clip:.2%}"
        print(
            f"{case.name:10s} {row['stable_runs']}/{row['runs']}"
            f"      {row['velocity_rmse_mps']:.4f}"
            f"      {row['yaw_rmse_deg']:.3f}"
            f"      {raw_text:>7s}"
            f"          {row['allocation_saturation_fraction']:.2%}"
        )
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows, figures, metrics_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("checkpoint", "ego_checkpoint", "vehicle_config"):
        if key not in config:
            continue
        candidate = Path(config[key])
        config[key] = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    config["config_path"] = path
    return config


def validate_thrust_mapping(config, gravity):
    """Validate c <-> normalized PX4 thrust for practical configurations."""
    keys = (
        "hover_thrust",
        "normalized_thrust_lower",
        "normalized_thrust_upper",
    )
    present = [key in config for key in keys]
    if not any(present):
        return
    if not all(present):
        raise ValueError(f"practical benchmark requires {keys}")
    hover = float(config["hover_thrust"])
    lower = float(config["normalized_thrust_lower"])
    upper = float(config["normalized_thrust_upper"])
    if not 0.0 < hover < 1.0 or not 0.0 <= lower < upper <= 1.0:
        raise ValueError("invalid hover or normalized thrust bounds")
    expected = gravity * np.asarray((lower, upper)) / hover
    actual = np.asarray((config["control_lower"][0], config["control_upper"][0]))
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError("collective bounds do not match hover-thrust mapping")


def _write_summary(metric_paths, summary_path):
    rows = []
    for path in metric_paths:
        with Path(path).open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    with Path(summary_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark_suite(
    config_path,
    output_dir,
    controllers=None,
    speed_scales=None,
    checkpoint=None,
    ego_checkpoint=None,
):
    config = load_benchmark_config(config_path)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        config["checkpoint"] = (
            checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
        )
    if ego_checkpoint is not None:
        ego_checkpoint = Path(ego_checkpoint)
        config["ego_checkpoint"] = (
            ego_checkpoint
            if ego_checkpoint.is_absolute()
            else PROJECT_ROOT / ego_checkpoint
        )
    default_controllers = tuple(
        config.get(
            "controllers",
            CONTROLLERS if "ego_checkpoint" in config else ("ccm", "so3-ctbr"),
        )
    )
    controllers = tuple(controllers or default_controllers)
    if "ego-ccm" in controllers and "ego_checkpoint" not in config:
        raise ValueError("ego-ccm requires ego_checkpoint")
    speed_scales = tuple(speed_scales or config["speed_scales"])
    params = QuadrotorParams.from_yaml(config["vehicle_config"])
    validate_thrust_mapping(config, params.g)
    all_cases = [case for scale in speed_scales for case in make_track_cases(scale)]
    preflight = check_reference_feasibility(
        all_cases,
        params.g,
        config["frequency_hz"],
        config["fixed_yaw_rad"],
        config.get("yaw_mode", "fixed"),
        config.get("yaw_rate_limit_radps", 0.5),
        config["reference_lower"],
        config["reference_upper"],
    )
    print(f"preflight: {len(preflight)}/{len(preflight)} references feasible")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = []
    for controller_type in controllers:
        for speed_scale in speed_scales:
            run_dir = output_dir / f"{controller_type}_speed{speed_scale:g}"
            _, figures, metrics_path = run_track_benchmark(
                checkpoint=(
                    config["ego_checkpoint"]
                    if controller_type == "ego-ccm"
                    else config["checkpoint"]
                ),
                vehicle_config=config["vehicle_config"],
                frequency=config["frequency_hz"],
                outer_frequency=config["outer_frequency_hz"],
                robustness_seeds=tuple(config["robustness_seeds"]),
                output_dir=run_dir,
                speed_scale=speed_scale,
                controller_type=controller_type,
                fixed_yaw=config["fixed_yaw_rad"],
                yaw_mode=config.get("yaw_mode", "fixed"),
                yaw_rate_limit=config.get("yaw_rate_limit_radps", 0.5),
                control_lower=config["control_lower"],
                control_upper=config["control_upper"],
                hover_thrust=config.get("hover_thrust"),
                normalized_thrust_lower=config.get("normalized_thrust_lower"),
                normalized_thrust_upper=config.get("normalized_thrust_upper"),
                attitude_disturbance_torque=config.get(
                    "attitude_disturbance_torque_nm", (0.0, 0.0, 0.0)
                ),
                initial_velocity_error_mps=config.get(
                    "initial_velocity_error_bound_mps", 0.35
                ),
                initial_attitude_error_rad=config.get(
                    "initial_attitude_error_bound_rad", np.deg2rad(15.0)
                ),
            )
            metric_paths.append(metrics_path)
            for figure, _ in figures.values():
                plt.close(figure)

    summary_path = output_dir / "benchmark_summary.csv"
    _write_summary(metric_paths, summary_path)
    source_paths = {
        name: PROJECT_ROOT / name
        for name in (
            "track_benchmark.py",
            "uav_ccm.py",
            "uav_ego_ccm.py",
            "uav_so3.py",
            "uav_sim.py",
        )
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": config["name"],
        "config": str(config["config_path"]),
        "config_sha256": _sha256(config["config_path"]),
        "checkpoint": str(config["checkpoint"]),
        "checkpoint_sha256": _sha256(config["checkpoint"]),
        "ego_checkpoint": (
            str(config["ego_checkpoint"]) if "ego_checkpoint" in config else None
        ),
        "ego_checkpoint_sha256": (
            _sha256(config["ego_checkpoint"])
            if "ego_checkpoint" in config
            else None
        ),
        "vehicle_config": str(config["vehicle_config"]),
        "vehicle_config_sha256": _sha256(config["vehicle_config"]),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "controllers": list(controllers),
        "yaw_mode": config.get("yaw_mode", "fixed"),
        "yaw_rate_limit_radps": config.get("yaw_rate_limit_radps", 0.5),
        "speed_scales": list(speed_scales),
        "hover_thrust": config.get("hover_thrust"),
        "normalized_thrust_lower": config.get("normalized_thrust_lower"),
        "normalized_thrust_upper": config.get("normalized_thrust_upper"),
        "control_lower": config["control_lower"],
        "control_upper": config["control_upper"],
        "motor_time_constant_s": params.motor_time_constant,
        "motor_noise_std_radps": params.motor_noise_std,
        "drag_force_coeff": params.drag_force_coeff.tolist(),
        "drag_torque_coeff": params.drag_torque_coeff.tolist(),
        "attitude_disturbance_torque_nm": list(
            config.get("attitude_disturbance_torque_nm", (0.0, 0.0, 0.0))
        ),
        "initial_velocity_error_bound_mps": config.get(
            "initial_velocity_error_bound_mps", 0.35
        ),
        "initial_attitude_error_bound_rad": config.get(
            "initial_attitude_error_bound_rad", np.deg2rad(15.0)
        ),
        "preflight": preflight,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return summary_path, output_dir / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cfg/benchmark.yaml")
    parser.add_argument("--output-dir", default="fig/baseline")
    parser.add_argument("--controller", choices=("all", *CONTROLLERS), default="all")
    parser.add_argument("--speed-scale", type=float, nargs="+")
    parser.add_argument("--checkpoint", help="override the checkpoint in the config")
    parser.add_argument(
        "--ego-checkpoint", help="override the ego-centric checkpoint"
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    controllers = None if args.controller == "all" else (args.controller,)
    run_benchmark_suite(
        args.config,
        args.output_dir,
        controllers=controllers,
        speed_scales=args.speed_scale,
        checkpoint=args.checkpoint,
        ego_checkpoint=args.ego_checkpoint,
    )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
