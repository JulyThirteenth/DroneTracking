"""Finite-difference audit of the R^3 x SO(3) contraction implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_ccm import (  # noqa: E402
    contraction_terms,
    hat,
    load_models,
    retract,
    sample_batch,
    so3_exp,
    torch_dynamics,
    vee,
)


def so3_log(R):
    cosine = ((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    skew_vector = vee(R - R.T)
    if theta.abs() < 1e-7:
        return 0.5 * skew_vector
    return theta * skew_vector / (2.0 * torch.sin(theta))


def relative_error(actual, expected):
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).detach())


def max_error(actual, expected):
    return float((actual - expected).abs().max().detach())


def closed_loop_velocity(controller, v, R, v_ref, R_ref, u_ref):
    return torch_dynamics(v, R, controller(v, R, v_ref, R_ref, u_ref))


def raw_jacobian_finite_difference(controller, v, R, v_ref, R_ref, u_ref, epsilon):
    columns = []
    eye = torch.eye(6, dtype=v.dtype, device=v.device)
    for direction in eye:
        vp, Rp = retract(v, R, epsilon * direction)
        vm, Rm = retract(v, R, -epsilon * direction)
        fp = closed_loop_velocity(controller, vp, Rp, v_ref, R_ref, u_ref)
        fm = closed_loop_velocity(controller, vm, Rm, v_ref, R_ref, u_ref)
        columns.append((fp - fm) / (2.0 * epsilon))
    return torch.stack(columns, dim=1)


def metric_dot_finite_difference(metric, v, R, f, step):
    vp, Rp = retract(v, R, step * f)
    vm, Rm = retract(v, R, -step * f)
    return (metric(vp, Rp) - metric(vm, Rm)) / (2.0 * step)


def flow_step(controller, v, R, v_ref, R_ref, u_ref, step):
    f = closed_loop_velocity(controller, v, R, v_ref, R_ref, u_ref)
    return v + step * f[:3], R @ so3_exp(step * f[3:])


def flow_jacobian_finite_difference(
    controller, v, R, v_ref, R_ref, u_ref, epsilon, step
):
    v_next, R_next = flow_step(controller, v, R, v_ref, R_ref, u_ref, step)
    columns = []
    eye = torch.eye(6, dtype=v.dtype, device=v.device)
    for direction in eye:
        vp, Rp = retract(v, R, epsilon * direction)
        vm, Rm = retract(v, R, -epsilon * direction)
        vp_next, Rp_next = flow_step(controller, vp, Rp, v_ref, R_ref, u_ref, step)
        vm_next, Rm_next = flow_step(controller, vm, Rm, v_ref, R_ref, u_ref, step)
        qp = torch.cat((vp_next - v_next, so3_log(R_next.T @ Rp_next)))
        qm = torch.cat((vm_next - v_next, so3_log(R_next.T @ Rm_next)))
        columns.append(((qp - qm) / (2.0 * epsilon) - direction) / step)
    return torch.stack(columns, dim=1)


def sample_from_checkpoint(config, size, seed, dtype):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    batch = sample_batch(size, dtype=dtype, **config["sampling_ranges"])
    torch.random.set_rng_state(rng_state)
    return batch


def verify(args):
    dtype = torch.float64
    controller, metric, config = load_models(args.checkpoint)
    controller, metric = controller.to(dtype=dtype), metric.to(dtype=dtype)
    batch = sample_from_checkpoint(config, args.samples, args.seed, dtype)
    rate = float(config.get("rate", 2.0))
    names = ("raw_A", "correction", "flow_A", "no_correction", "wrong_sign", "M_dot", "C")
    errors = {name: [] for name in names}
    absolute = {name: [] for name in names}
    for v, R, v_ref, R_ref, u_ref in zip(*batch):
        C, M, _, f, A, M_dot = contraction_terms(
            controller, metric, v, R, v_ref, R_ref, u_ref, rate
        )
        correction = torch.zeros(6, 6, dtype=dtype)
        correction[3:, 3:] = -hat(f[3:])
        raw_A = A - correction
        raw_A_fd = raw_jacobian_finite_difference(
            controller, v, R, v_ref, R_ref, u_ref, args.epsilon
        )
        A_flow_fd = flow_jacobian_finite_difference(
            controller, v, R, v_ref, R_ref, u_ref, args.epsilon, args.flow_step
        )
        M_dot_fd = metric_dot_finite_difference(metric, v, R, f, args.flow_step)
        C_fd = M_dot_fd + A_flow_fd.T @ M + M @ A_flow_fd + 2.0 * rate * M
        C_fd = 0.5 * (C_fd + C_fd.T)
        comparisons = {
            "raw_A": (raw_A, raw_A_fd),
            "correction": (A - raw_A_fd, correction),
            "flow_A": (A, A_flow_fd),
            "no_correction": (raw_A, A_flow_fd),
            "wrong_sign": (raw_A - correction, A_flow_fd),
            "M_dot": (M_dot, M_dot_fd),
            "C": (C, C_fd),
        }
        for name, (actual, expected) in comparisons.items():
            errors[name].append(relative_error(actual, expected))
            absolute[name].append(max_error(actual, expected))

    print(
        f"checkpoint={args.checkpoint}; samples={args.samples}; "
        f"epsilon={args.epsilon:g}; flow_step={args.flow_step:g}"
    )
    for name in names:
        print(
            f"{name:10s}: max relative={max(errors[name]):.3e}, "
            f"max absolute={max(absolute[name]):.3e}"
        )
    passed = (
        max(errors["raw_A"]) < args.derivative_tolerance
        and max(errors["flow_A"]) < args.flow_tolerance
        and max(errors["M_dot"]) < args.derivative_tolerance
        and max(errors["C"]) < args.flow_tolerance
    )
    print("PASS" if passed else "FAIL")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "neu_ccm_linear.pt"),
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--flow-step", type=float, default=1e-4)
    parser.add_argument("--derivative-tolerance", type=float, default=2e-4)
    parser.add_argument("--flow-tolerance", type=float, default=5e-3)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args) else 1)


if __name__ == "__main__":
    main()
