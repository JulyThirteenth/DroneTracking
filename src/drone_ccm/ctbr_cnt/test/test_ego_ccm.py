"""Deterministic checks for the ego-centric CCM implementation."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_ego_ccm import (  # noqa: E402
    EgoController,
    EgoMetric,
    GRAVITY,
    body_reference_acceleration,
    ccm_loss,
    ego_dynamics,
    gamma_from_state,
    input_matrix,
    rotation_from_gamma_heading,
    sample_batch,
    tangent_basis,
)


def main() -> None:
    state = torch.tensor((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    control = torch.tensor((GRAVITY, 0.0, 0.0, 0.0))
    torch.testing.assert_close(
        ego_dynamics(state, control, torch.zeros(3), torch.tensor(0.0)),
        torch.zeros(7),
    )

    state = torch.tensor((0.2, -0.3, 0.9327379, 0.4, -0.2, 0.1, 0.25))
    gamma = gamma_from_state(state)
    rotation = rotation_from_gamma_heading(gamma, state[-1])
    torch.testing.assert_close(rotation[2], gamma, atol=1e-6, rtol=1e-6)
    acceleration = torch.tensor((0.3, -0.2, 0.1))
    torch.testing.assert_close(
        body_reference_acceleration(state, acceleration),
        rotation.T @ acceleration,
    )
    derivative = ego_dynamics(
        state,
        torch.tensor((10.0, 0.2, -0.3, 0.4)),
        acceleration,
        torch.tensor(0.1),
    )
    assert abs(float(gamma @ derivative[:3])) < 1e-6
    basis = tangent_basis(state)
    torch.testing.assert_close(basis.T @ basis, torch.eye(6), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        gamma @ basis[:3, :2], torch.zeros(2), atol=1e-6, rtol=1e-6
    )
    numerical_input = torch.func.jacrev(
        lambda value: ego_dynamics(state, value, acceleration, torch.tensor(0.1))
    )(torch.zeros(4))
    torch.testing.assert_close(input_matrix(state), numerical_input)

    gamma_ref = torch.tensor((0.2, -0.1, 0.9746794))
    reference_rotation = rotation_from_gamma_heading(gamma_ref, torch.tensor(0.0))
    reference_control = torch.tensor((10.2, 0.1, -0.2, 0.15))
    reference_acceleration = (
        -GRAVITY * torch.tensor((0.0, 0.0, 1.0))
        + reference_control[0] * reference_rotation[:, 2]
    )
    zero_error_state = torch.cat((gamma_ref, torch.zeros(3), torch.zeros(1)))
    controller = EgoController(hidden=8)
    yaw_rate_ref = (
        reference_control[2] * gamma_ref[1]
        + reference_control[3] * gamma_ref[2]
    ) / (gamma_ref[1].square() + gamma_ref[2].square())
    torch.testing.assert_close(
        controller(
            zero_error_state,
            reference_acceleration,
            yaw_rate_ref,
            reference_control,
        ),
        reference_control,
        atol=1e-6,
        rtol=1e-6,
    )

    torch.manual_seed(7)
    batch = sample_batch(16)
    sampled_gamma = gamma_from_state(batch[0])
    torch.testing.assert_close(
        sampled_gamma.square().sum(-1), torch.ones(16), atol=1e-6, rtol=1e-6
    )
    assert float(torch.acos(sampled_gamma[:, 2]).max()) <= 1.30001
    assert float(torch.asin(sampled_gamma[:, 0]).abs().max()) <= 1.00001
    metric = EgoMetric(hidden=8)
    loss, stats = ccm_loss(controller, metric, batch, rate=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    parameters = list(controller.parameters()) + list(metric.parameters())
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert set(stats) == {
        "loss",
        "contraction_loss",
        "contracting_fraction",
        "max_C_eig",
        "c1_fraction",
        "c1_loss",
        "c2_loss",
        "metric_upper_loss",
    }
    print("PASS: ego dynamics, tangent projection, controller, certificate gradients")


if __name__ == "__main__":
    main()
