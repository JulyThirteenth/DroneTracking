"""Core tests for the inference-only ctbr_cnt ROS adapter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_ccm.frame import collective_to_normalized_thrust
from drone_ccm.geometry import exp as so3_exp
from drone_ccm.geometry import log as so3_log
from drone_ccm.reference import hover_reference
from drone_ccm.runtime import EgoCcmDomain, load_domain, load_runtime


CHECKPOINT = Path(__file__).parents[1] / "ctbr_cnt" / "neu_ccm_practical.pt"
EGO_CHECKPOINT = Path(__file__).parents[1] / "ctbr_cnt" / "neu_ego_ccm_active.pt"


def _runtime():
    return load_runtime(
        CHECKPOINT,
        device_name="cpu",
        dtype_name="float32",
    )


def test_so3_round_trip_near_pi_reference_attitude() -> None:
    vectors = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (0.2, -0.1, 0.05),
            (math.radians(170.0), 0.0, 0.0),
            (0.0, math.radians(-175.0), 0.0),
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        so3_log(so3_exp(vectors)),
        vectors,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_ctbr_cnt_checkpoint_domain() -> None:
    domain = load_domain(CHECKPOINT)
    assert domain.velocity_reference_max == 2.5
    assert domain.velocity_error_max == 1.5
    assert domain.rotation_error_angle_max == 0.7
    assert domain.reference_body_rate_max == (1.0, 1.0, 0.5)
    assert domain.control_lower[0] == pytest.approx(1.6913793103448278)
    assert domain.control_upper[0] == pytest.approx(15.22241379310345)


def test_runtime_matches_ctbr_cnt_regression_output() -> None:
    command = _runtime().command(
        np.array((0.2, -0.1, 0.05)),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        np.array((9.81, 0.1, -0.1, 0.05)),
    )
    np.testing.assert_allclose(
        command,
        np.array((9.709871, 0.04571271, -0.2039938, 0.05334437)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_runtime_preserves_zero_error_feedforward() -> None:
    reference_control = np.array((9.81, 0.1, -0.1, 0.05))
    command = _runtime().command(
        np.zeros(3),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        reference_control,
    )
    np.testing.assert_allclose(command, reference_control, atol=1.0e-6)


def test_ego_checkpoint_selects_ego_runtime_and_domain() -> None:
    runtime = load_runtime(
        EGO_CHECKPOINT,
        device_name="cpu",
        dtype_name="float32",
    )
    assert isinstance(runtime.domain, EgoCcmDomain)
    assert runtime.domain.velocity_reference_max is None
    assert runtime.domain.velocity_error_max == 1.5
    assert runtime.domain.actual_tilt_angle_max == 1.3
    assert runtime.domain.actual_pitch_angle_max == 1.0


def test_ego_runtime_matches_training_implementation() -> None:
    runtime = load_runtime(
        EGO_CHECKPOINT,
        device_name="cpu",
        dtype_name="float32",
    )
    rotation = so3_exp(torch.tensor((0.05, -0.04, 0.10))).numpy()
    command = runtime.command(
        np.array((0.2, -0.1, 0.05)),
        rotation,
        np.array((0.1, 0.0, 0.0)),
        np.eye(3),
        np.array((9.81, 0.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(
        command,
        np.array((9.4812994, -0.23391271, 0.04592276, -0.16906714)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_ego_runtime_preserves_feasible_feedforward() -> None:
    runtime = load_runtime(
        EGO_CHECKPOINT,
        device_name="cpu",
        dtype_name="float32",
    )
    rotation = so3_exp(torch.tensor((0.10, -0.08, 0.20))).numpy()
    reference_control = np.array((9.9, 0.1, -0.1, 0.05))
    command = runtime.command(
        np.array((5.0, -3.0, 0.5)),
        rotation,
        np.array((5.0, -3.0, 0.5)),
        rotation,
        reference_control,
    )
    np.testing.assert_allclose(command, reference_control, atol=1.0e-6)


def test_ego_runtime_rejects_out_of_domain_state() -> None:
    runtime = load_runtime(
        EGO_CHECKPOINT,
        device_name="cpu",
        dtype_name="float32",
    )
    hover = np.array((9.81, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="velocity-error domain"):
        runtime.command(
            np.array((1.51, 0.0, 0.0)),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            hover,
        )
    tilted = so3_exp(torch.tensor((1.31, 0.0, 0.0))).numpy()
    with pytest.raises(RuntimeError, match="attitude domain"):
        runtime.command(np.zeros(3), tilted, np.zeros(3), tilted, hover)


def test_pegasus_hover_thrust_mapping() -> None:
    assert collective_to_normalized_thrust(9.81, 0.5812, 9.81) == pytest.approx(
        0.5812
    )
    assert collective_to_normalized_thrust(9.81, 0.40, 9.81) == pytest.approx(0.40)


def test_runtime_rejects_state_outside_training_domain() -> None:
    with pytest.raises(RuntimeError, match="tracking-error domain"):
        _runtime().command(
            np.array((1.51, 0.0, 0.0)),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array((9.81, 0.0, 0.0, 0.0)),
        )


def test_hover_engagement_uses_velocity_and_yaw_only() -> None:
    template = torch.zeros(1, 3, dtype=torch.float64)
    initial_velocity = torch.tensor(((0.12, -0.07, 0.03),), dtype=torch.float64)
    initial_yaw = torch.tensor((0.8,), dtype=torch.float64)
    velocity, rotation, control = hover_reference(
        0.0,
        template,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        transition_duration=3.0,
        time_step=0.02,
        gravity=9.81,
    )
    torch.testing.assert_close(velocity, initial_velocity)
    yaw = torch.atan2(rotation[0, 1, 0], rotation[0, 0, 0])
    assert abs(float(yaw) - 0.8) < 1.0e-10
    assert abs(float(control[0, 0]) - 9.81) < 1.0e-9
