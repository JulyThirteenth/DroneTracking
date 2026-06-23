"""Regression checks for the frozen benchmark definitions."""

import dataclasses
from pathlib import Path
import sys

import numpy as np
from spatialmath import SO3
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track_benchmark import (  # noqa: E402
    CONTROLLERS,
    TrackResult,
    check_reference_feasibility,
    flatness_reference,
    ego_ctbr_numpy,
    load_benchmark_config,
    make_track_cases,
    reference_profile,
    trajectory_yaw,
    validate_thrust_mapping,
)
from uav_sim import Quadrotor, QuadrotorParams  # noqa: E402
from uav_so3 import attitude_from_force_acceleration  # noqa: E402
from uav_ccm import (  # noqa: E402
    CTBRController,
    ContractionMetric,
    ccm_loss,
    metric_normalize,
    sample_batch,
    strong_ccm_terms,
)


def main():
    config = load_benchmark_config(
        PROJECT_ROOT / "cfg/benchmark.yaml"
    )
    np.testing.assert_allclose(config["reference_lower"][1:], [-1.0, -1.0, -0.5])
    np.testing.assert_allclose(config["reference_upper"][1:], [1.0, 1.0, 0.5])
    params = QuadrotorParams.from_yaml(config["vehicle_config"])
    assert params.motor_time_constant == 0.02
    assert params.motor_noise_std == 2.0
    np.testing.assert_allclose(params.drag_force_coeff, (0.50, 0.30, 0.0))
    np.testing.assert_allclose(params.drag_torque_coeff, (0.01, 0.01, 0.02))
    validate_thrust_mapping(config, params.g)
    np.testing.assert_allclose(
        config["hover_thrust"]
        * np.asarray((config["control_lower"][0], config["control_upper"][0]))
        / params.g,
        (config["normalized_thrust_lower"], config["normalized_thrust_upper"]),
        atol=1e-12,
    )
    invalid_mapping = dict(config)
    invalid_mapping.pop("normalized_thrust_upper")
    try:
        validate_thrust_mapping(invalid_mapping, params.g)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete PX4 thrust mapping was accepted")
    deterministic_params = dataclasses.replace(
        params,
        motor_noise_std=0.0,
    )
    quadrotor = Quadrotor(deterministic_params)
    state = np.zeros(17)
    state[6] = 1.0
    rotor_speed = np.zeros(4)
    commanded_speed = np.full(4, 700.0)
    stepped = quadrotor.step(state, commanded_speed, 1.0 / 400.0)
    expected_speed = commanded_speed * (
        1.0 - np.exp(-1.0 / 400.0 / params.motor_time_constant)
    )
    np.testing.assert_allclose(stepped[13:17], expected_speed, atol=2e-3)

    torque = np.array([0.06, -0.04, 0.02])
    nominal_derivative = quadrotor.x_dot(state, rotor_speed)
    disturbed_derivative = quadrotor.x_dot(
        state, rotor_speed, external_torque=torque
    )
    np.testing.assert_allclose(
        disturbed_derivative[10:13] - nominal_derivative[10:13],
        np.linalg.solve(params.inertia, torque),
    )
    force_acceleration = np.array([1.2, -0.8, 9.5])
    yaw = 0.4
    force_attitude = attitude_from_force_acceleration(force_acceleration, yaw)
    np.testing.assert_allclose(
        force_attitude[:, 2],
        force_acceleration / np.linalg.norm(force_acceleration),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.arctan2(force_attitude[1, 0], force_attitude[0, 0]), yaw, atol=1e-12
    )
    cases = [
        case
        for scale in config["speed_scales"]
        for case in make_track_cases(scale)
    ]
    reports = check_reference_feasibility(
        cases,
        params.g,
        config["frequency_hz"],
        config["fixed_yaw_rad"],
        config.get("yaw_mode", "fixed"),
        config.get("yaw_rate_limit_radps", 0.5),
        config["reference_lower"],
        config["reference_upper"],
    )
    assert all(report["feasible"] for report in reports)
    assert CONTROLLERS == ("ccm", "ego-ccm", "so3-ctbr", "so3-full")
    reference_a = flatness_reference(
        cases[0],
        0.7,
        params.g,
        config["fixed_yaw_rad"],
        config.get("yaw_mode", "fixed"),
    )
    reference_b = flatness_reference(
        cases[0],
        0.7,
        params.g,
        config["fixed_yaw_rad"],
        config.get("yaw_mode", "fixed"),
    )
    for first, second in zip(reference_a, reference_b):
        np.testing.assert_allclose(first, second)
    dynamic_reference = flatness_reference(
        cases[0], 0.7, params.g, config["fixed_yaw_rad"], "velocity"
    )
    _, velocity, acceleration, yaw, rotation, control = dynamic_reference
    expected_yaw, expected_yaw_rate = trajectory_yaw(
        velocity, acceleration, config["fixed_yaw_rad"], "velocity"
    )
    gamma_ref = rotation.T @ np.array([0.0, 0.0, 1.0])
    actual_yaw_rate = (
        control[2] * gamma_ref[1] + control[3] * gamma_ref[2]
    ) / (gamma_ref[1] ** 2 + gamma_ref[2] ** 2)
    np.testing.assert_allclose(yaw, expected_yaw)
    np.testing.assert_allclose(actual_yaw_rate, expected_yaw_rate, atol=1e-7)
    profile_times = np.arange(0.0, 2.0, 1.0 / config["frequency_hz"])
    profile = reference_profile(
        cases[0], profile_times, params.g, 0.0, "velocity", 0.5
    )
    profile_yaw = np.unwrap([reference[3] for reference in profile])
    assert np.max(np.abs(np.diff(profile_yaw) * config["frequency_hz"])) <= 0.500001

    from uav_ego_ccm import EgoController

    ego_controller = EgoController(hidden=8)
    p_ref, v_ref, a_ref, _, rotation_ref, control_ref = reference_a
    del p_ref
    frame_flip = np.diag([1.0, 1.0, -1.0])
    control_ref_flu = np.r_[control_ref[0], -frame_flip @ control_ref[1:4]]
    torch.testing.assert_close(
        torch.as_tensor(
            ego_ctbr_numpy(
                ego_controller,
                frame_flip @ v_ref,
                frame_flip @ rotation_ref @ frame_flip,
                frame_flip @ v_ref,
                frame_flip @ rotation_ref @ frame_flip,
                frame_flip @ a_ref,
                control_ref_flu,
            )
        ),
        torch.as_tensor(control_ref_flu, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    times = np.array([0.0, 1.0])
    identity_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    states = np.zeros((2, 17))
    states[:, 6:10] = identity_quaternion
    nominal = np.repeat(np.eye(3)[None], 2, axis=0)
    internal = nominal.copy()
    internal[1] = SO3.Rx(0.2).R
    result = TrackResult(
        case=cases[0],
        controller_type="so3-ctbr",
        times=times,
        states=states,
        position_ref=np.zeros((2, 3)),
        velocity_ref=np.zeros((2, 3)),
        yaw_ref=np.zeros(2),
        nominal_attitude_ref=nominal,
        internal_attitude_ref=internal,
        ctbr_ref=np.zeros((2, 4)),
        raw_ctbr=np.zeros((2, 4)),
        applied_ctbr=np.zeros((2, 4)),
        applied_wrench=np.zeros((2, 4)),
        normalized_thrust=np.full(2, np.nan),
        position_error=np.zeros((2, 3)),
        velocity_error=np.zeros((2, 3)),
        yaw_error=np.zeros(2),
        ctbr_clipped=np.zeros(2, dtype=bool),
        normalized_ctbr_violation=np.zeros(2),
        allocation_saturated=np.zeros(2, dtype=bool),
    )
    metrics = result.metrics(settling_time=0.0)
    assert metrics["nominal_attitude_rmse_deg"] == 0.0
    assert metrics["internal_attitude_rmse_deg"] > 0.0
    assert metrics["attitude_reference_offset_rmse_deg"] > 0.0

    controller = CTBRController()
    linear_velocity_error = torch.tensor([[0.2, -0.3, 0.4]])
    linear_reference = torch.tensor([[9.81, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        controller.raw_feedback(
            linear_velocity_error,
            torch.eye(3)[None],
            torch.zeros_like(linear_velocity_error),
            torch.eye(3)[None],
            linear_reference,
        ),
        torch.tensor([[-0.8, -0.45, -0.3, 0.0]]),
    )
    controller.zero_grad(set_to_none=True)
    controller.raw_feedback(
        linear_velocity_error,
        torch.eye(3)[None],
        torch.zeros_like(linear_velocity_error),
        torch.eye(3)[None],
        linear_reference,
    ).sum().backward()
    assert controller.linear_gain.grad is not None

    reference = torch.tensor(
        [[6.81, -1.0, -1.0, -0.5], [12.81, 1.0, 1.0, 0.5]]
    )
    torch.testing.assert_close(
        controller.apply_feedback(reference, torch.zeros_like(reference)), reference
    )
    identity = torch.eye(3).expand(2, 3, 3)
    torch.testing.assert_close(
        controller.raw_feedback(
            torch.zeros(2, 3),
            identity,
            torch.zeros(2, 3),
            identity,
            reference,
        ),
        torch.zeros_like(reference),
    )
    commands = controller.apply_feedback(
        reference,
        torch.tensor([[-1e6, -1e6, -1e6, -1e6], [1e6, 1e6, 1e6, 1e6]]),
    )
    assert torch.all(commands >= controller.control_lower)
    assert torch.all(commands <= controller.control_upper)

    feedback = torch.tensor([[7.0, -6.0, 5.0, -4.0], [-3.0, 2.0, -1.0, 8.0]])
    zero = torch.zeros_like(reference, requires_grad=True)
    mapped = controller.apply_feedback(reference, zero)
    torch.testing.assert_close(mapped, reference)
    torch.testing.assert_close(
        torch.autograd.grad(mapped.sum(), zero)[0], torch.ones_like(zero)
    )
    extreme = controller.apply_feedback(reference, feedback * 1e3)
    assert torch.all(extreme >= controller.control_lower)
    assert torch.all(extreme <= controller.control_upper)

    torch.manual_seed(29)
    _, sampled_rotation, _, sampled_reference, _ = sample_batch(4096)
    relative = sampled_reference.transpose(-1, -2) @ sampled_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    rotation_angle = torch.acos(cosine)
    assert float(rotation_angle.max()) <= 0.70001
    assert float(rotation_angle[2048:].min()) >= 0.8 * 0.7 - 1e-5
    reference_tilt = torch.acos(sampled_reference[:, 2, 2].clamp(-1.0, 1.0))
    assert float(reference_tilt.max()) <= 1.00001

    metric = ContractionMetric(hidden=16)
    velocity = torch.zeros(3)
    rotation = torch.eye(3)
    primal = metric.primal(velocity, rotation)
    dual = metric.dual(velocity, rotation)
    torch.testing.assert_close(metric(velocity, rotation), primal)
    torch.testing.assert_close(primal @ dual, torch.eye(6), atol=1e-5, rtol=1e-5)
    matrix = torch.tensor([[2.0, 0.4], [0.4, -1.0]])
    scale_metric = torch.tensor([[3.0, 0.2], [0.2, 2.0]])
    torch.testing.assert_close(
        metric_normalize(5.0 * matrix, 5.0 * scale_metric),
        metric_normalize(matrix, scale_metric),
    )
    c1, c2, dual = strong_ccm_terms(metric, velocity, rotation, 0.5)
    assert c1.shape == (2, 2)
    assert c2.shape == (4, 2, 2)
    assert dual.shape == (6, 6)
    loss, statistics = ccm_loss(
        controller,
        metric,
        sample_batch(4),
        rate=0.5,
    )
    loss.backward()
    assert statistics["c1_loss"] >= 0.0 and statistics["c2_loss"] >= 0.0
    assert any(parameter.grad is not None for parameter in controller.parameters())
    assert any(parameter.grad is not None for parameter in metric.parameters())
    print(
        f"PASS: {len(reports)} feasible references and unified CTBR bounds"
    )


if __name__ == "__main__":
    main()
