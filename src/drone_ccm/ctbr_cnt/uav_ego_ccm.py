"""CCM learning for ego-centric CTBR dynamics on S2 x R3 x S1."""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from uav_ccm import (
    MLP,
    PEGASUS_CONTROL_LOWER,
    PEGASUS_CONTROL_UPPER,
    metric_normalize,
    so3_exp,
)

GRAVITY = 9.81
AMBIENT_DIM = 7
TANGENT_DIM = 6
CONTROL_DIM = 4


def gamma_from_state(state: Tensor) -> Tensor:
    """Return the embedded unit gravity direction."""
    return state[..., :3]


def tangent_basis(state: Tensor) -> Tensor:
    """Return a smooth orthonormal basis of T_gamma S2 x R3 x S1."""
    gx, gy, gz = gamma_from_state(state).unbind(-1)
    scale = 1.0 / (1.0 + gz).clamp_min(1.0e-6)
    first = torch.stack((1.0 - gx.square() * scale, -gx * gy * scale, -gx), dim=-1)
    second = torch.stack((-gx * gy * scale, 1.0 - gy.square() * scale, -gy), dim=-1)
    sphere = torch.stack((first, second), dim=-1)
    top = torch.cat(
        (
            sphere,
            torch.zeros(
                state.shape[:-1] + (3, 4),
                dtype=state.dtype,
                device=state.device,
            ),
        ),
        dim=-1,
    )
    middle = torch.cat(
        (
            torch.zeros(
                state.shape[:-1] + (3, 2),
                dtype=state.dtype,
                device=state.device,
            ),
            torch.eye(3, dtype=state.dtype, device=state.device).expand(
                state.shape[:-1] + (3, 3)
            ),
            torch.zeros(
                state.shape[:-1] + (3, 1),
                dtype=state.dtype,
                device=state.device,
            ),
        ),
        dim=-1,
    )
    bottom = torch.cat(
        (
            torch.zeros(
                state.shape[:-1] + (1, 5),
                dtype=state.dtype,
                device=state.device,
            ),
            torch.ones(
                state.shape[:-1] + (1, 1),
                dtype=state.dtype,
                device=state.device,
            ),
        ),
        dim=-1,
    )
    return torch.cat((top, middle, bottom), dim=-2)


def rotation_from_gamma_heading(gamma: Tensor, heading: Tensor) -> Tensor:
    """Construct ZYX attitude from body gravity direction and heading."""
    pitch = -torch.asin(gamma[..., 0].clamp(-1.0, 1.0))
    roll = torch.atan2(gamma[..., 1], gamma[..., 2])
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(heading), torch.sin(heading)
    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    ).reshape(gamma.shape[:-1] + (3, 3))


def body_reference_acceleration(state: Tensor, acceleration_world: Tensor) -> Tensor:
    """Return a_B_ref = R.T @ a_W_ref in the reference-heading frame."""
    gamma = gamma_from_state(state)
    rotation = rotation_from_gamma_heading(gamma, state[..., 6])
    return (rotation.transpose(-1, -2) @ acceleration_world[..., None]).squeeze(-1)


def ego_dynamics(
    state: Tensor,
    control: Tensor,
    acceleration_world: Tensor,
    yaw_rate_ref: Tensor,
) -> Tensor:
    """Evaluate the embedded ego-centric dynamics."""
    gamma = gamma_from_state(state)
    velocity_error = state[..., 3:6]
    collective = control[..., 0]
    p, q, r = control[..., 1:].unbind(-1)
    gamma_rate = torch.stack(
        (
            r * gamma[..., 1] - q * gamma[..., 2],
            p * gamma[..., 2] - r * gamma[..., 0],
            q * gamma[..., 0] - p * gamma[..., 1],
        ),
        dim=-1,
    )
    omega = control[..., 1:]
    e3 = torch.tensor((0.0, 0.0, 1.0), dtype=state.dtype, device=state.device)
    velocity_rate = (
        torch.linalg.cross(velocity_error, omega, dim=-1)
        - GRAVITY * gamma
        + collective[..., None] * e3
    )
    velocity_rate = velocity_rate - body_reference_acceleration(
        state, acceleration_world
    )
    denominator = gamma[..., 1].square() + gamma[..., 2].square()
    heading_rate = (q * gamma[..., 1] + r * gamma[..., 2]) / denominator.clamp_min(
        1.0e-6
    ) - yaw_rate_ref
    return torch.cat((gamma_rate, velocity_rate, heading_rate[..., None]), dim=-1)


def ego_features(state: Tensor) -> Tensor:
    """Return continuous embedded features for S2 x R3 x S1."""
    return torch.cat(
        (
            gamma_from_state(state),
            state[..., 3:6],
            torch.sin(state[..., 6:7]),
            torch.cos(state[..., 6:7]),
        ),
        dim=-1,
    )


class EgoMetric(nn.Module):
    """Directly parameterized dual metric W on the six-dimensional tangent."""

    def __init__(self, hidden: int = 64, minimum_eigenvalue: float = 0.1):
        super().__init__()
        self.factor = MLP(8, TANGENT_DIM * TANGENT_DIM, hidden)
        self.minimum_eigenvalue = minimum_eigenvalue

    def dual(self, state: Tensor) -> Tensor:
        single = state.ndim == 1
        values = state[None] if single else state
        raw = self.factor(ego_features(values)).reshape(-1, TANGENT_DIM, TANGENT_DIM)
        identity = torch.eye(TANGENT_DIM, dtype=state.dtype, device=state.device)
        theta = identity + 0.1 * raw
        metric = theta.transpose(-1, -2) @ theta + self.minimum_eigenvalue * identity
        return metric[0] if single else metric

    def primal(self, state: Tensor) -> Tensor:
        return torch.linalg.inv(self.dual(state))


class EgoController(nn.Module):
    """Bounded CTBR feedforward plus learned ego-centric residual."""

    FACTOR_DIM = 18
    CONTEXT_DIM = 16

    def __init__(
        self,
        hidden: int = 64,
        control_lower=PEGASUS_CONTROL_LOWER,
        control_upper=PEGASUS_CONTROL_UPPER,
        initialize_linear_gain: bool = True,
    ) -> None:
        super().__init__()
        self.factor_input = MLP(
            self.CONTEXT_DIM,
            self.FACTOR_DIM * TANGENT_DIM,
            hidden,
        )
        self.factor_output = MLP(
            self.CONTEXT_DIM,
            CONTROL_DIM * self.FACTOR_DIM,
            hidden,
        )
        linear_gain = torch.zeros(CONTROL_DIM, TANGENT_DIM)
        if initialize_linear_gain:
            linear_gain[0, 2] = -2.0
            linear_gain[1, 1], linear_gain[1, 3] = 1.5, -3.0
            linear_gain[2, 0], linear_gain[2, 4] = -1.5, -3.0
            linear_gain[3, 5] = -2.0
            nn.init.zeros_(self.factor_output.net[-1].weight)
            nn.init.zeros_(self.factor_output.net[-1].bias)
        self.linear_gain = nn.Parameter(linear_gain)
        lower = torch.as_tensor(control_lower, dtype=torch.float32)
        upper = torch.as_tensor(control_upper, dtype=torch.float32)
        if lower.shape != (CONTROL_DIM,) or upper.shape != (CONTROL_DIM,):
            raise ValueError("control bounds must contain four values")
        if bool((lower >= upper).any()):
            raise ValueError("control lower bounds must be below upper bounds")
        self.register_buffer("control_lower", lower, persistent=False)
        self.register_buffer("control_upper", upper, persistent=False)

    def _error_context(
        self,
        state: Tensor,
        acceleration_world: Tensor,
        yaw_rate_ref: Tensor,
        control_ref: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gamma = gamma_from_state(state)
        acceleration_body = body_reference_acceleration(state, acceleration_world)
        desired_body_z = functional.normalize(
            acceleration_body + GRAVITY * gamma,
            dim=-1,
        )
        tilt_error = torch.stack(
            (desired_body_z[..., 1], -desired_body_z[..., 0]), dim=-1
        )
        error = torch.cat((state[..., 3:6], tilt_error, state[..., 6:7]), dim=-1)
        context = torch.cat(
            (
                ego_features(state),
                acceleration_body,
                yaw_rate_ref[..., None],
                control_ref,
            ),
            dim=-1,
        )
        return error, context

    def forward(
        self,
        state: Tensor,
        acceleration_world: Tensor,
        yaw_rate_ref: Tensor,
        control_ref: Tensor,
    ) -> Tensor:
        single = state.ndim == 1
        if single:
            state = state[None]
            acceleration_world = acceleration_world[None]
            yaw_rate_ref = yaw_rate_ref[None]
            control_ref = control_ref[None]
        error, context = self._error_context(
            state, acceleration_world, yaw_rate_ref, control_ref
        )
        left = self.factor_input(context).reshape(-1, self.FACTOR_DIM, TANGENT_DIM)
        right = self.factor_output(context).reshape(-1, CONTROL_DIM, self.FACTOR_DIM)
        hidden = torch.tanh(torch.einsum("bij,bj->bi", left, error))
        feedback = torch.einsum("ij,bj->bi", self.linear_gain, error)
        feedback = feedback + torch.einsum("bij,bj->bi", right, hidden)
        lower = self.control_lower.to(feedback)
        upper = self.control_upper.to(feedback)
        span = upper - lower
        ratio = ((control_ref - lower) / span).clamp(1.0e-6, 1.0 - 1.0e-6)
        margin = ((control_ref - lower) * (upper - control_ref) / span).clamp_min(
            torch.finfo(feedback.dtype).eps
        )
        output = lower + span * torch.sigmoid(torch.logit(ratio) + feedback / margin)
        return output[0] if single else output


def input_matrix(state: Tensor) -> Tensor:
    """Return the analytic ambient control matrix B(x)."""
    gamma = gamma_from_state(state)
    gx, gy, gz = gamma.unbind(-1)
    vx, vy, vz = state[..., 3:6].unbind(-1)
    zero = torch.zeros_like(gx)
    one = torch.ones_like(gx)
    denominator = gy.square() + gz.square()
    return torch.stack(
        (
            zero,
            zero,
            -gz,
            gy,
            zero,
            gz,
            zero,
            -gx,
            zero,
            -gy,
            gx,
            zero,
            zero,
            zero,
            -vz,
            vy,
            zero,
            vz,
            zero,
            -vx,
            one,
            -vy,
            vx,
            zero,
            zero,
            zero,
            gy / denominator,
            gz / denominator,
        ),
        dim=-1,
    ).reshape(state.shape[:-1] + (AMBIENT_DIM, CONTROL_DIM))


def _ambient_metric_rate(metric_fn, state: Tensor, direction: Tensor):
    basis, basis_rate = torch.func.jvp(tangent_basis, (state,), (direction,))
    local_metric, local_rate = torch.func.jvp(metric_fn, (state,), (direction,))
    basis = basis.to(dtype=local_metric.dtype)
    basis_rate = basis_rate.to(dtype=local_metric.dtype)
    ambient = basis @ local_metric @ basis.T
    ambient_rate = (
        basis_rate @ local_metric @ basis.T
        + basis @ local_rate @ basis.T
        + basis @ local_metric @ basis_rate.T
    )
    return ambient, ambient_rate


def contraction_terms(
    controller: EgoController,
    metric: EgoMetric,
    state: Tensor,
    acceleration_world: Tensor,
    yaw_rate_ref: Tensor,
    control_ref: Tensor,
    rate: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return closed-loop contraction matrix, primal metric and CTBR."""
    closed_loop = lambda value: ego_dynamics(
        value,
        controller(value, acceleration_world, yaw_rate_ref, control_ref),
        acceleration_world,
        yaw_rate_ref,
    )
    flow = closed_loop(state)
    jacobian = torch.func.jacrev(closed_loop)(state)
    basis = tangent_basis(state).to(dtype=state.dtype)
    ambient_primal, metric_rate = _ambient_metric_rate(metric.primal, state, flow)
    ambient_matrix = (
        metric_rate
        + jacobian.T @ ambient_primal
        + ambient_primal @ jacobian
        + 2.0 * rate * ambient_primal
    )
    matrix = basis.T @ ambient_matrix @ basis
    return (
        0.5 * (matrix + matrix.T),
        metric.primal(state),
        controller(state, acceleration_world, yaw_rate_ref, control_ref),
    )


def strong_ccm_terms(
    metric: EgoMetric,
    state: Tensor,
    acceleration_world: Tensor,
    yaw_rate_ref: Tensor,
    rate: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return tangent-projected dual C1/C2 constraints."""
    zero_control = torch.zeros(CONTROL_DIM, dtype=state.dtype, device=state.device)
    drift_fn = lambda value: ego_dynamics(
        value, zero_control, acceleration_world, yaw_rate_ref
    )
    drift = drift_fn(state)
    drift_jacobian = torch.func.jacrev(drift_fn)(state)
    basis = tangent_basis(state).to(dtype=state.dtype)
    ambient_metric, dual_rate = _ambient_metric_rate(metric.dual, state, drift)
    dual = metric.dual(state)
    control_matrix = input_matrix(state)
    tangent_control = basis.T @ control_matrix
    local_annihilator = (
        torch.linalg.svd(tangent_control, full_matrices=True)
        .U[:, CONTROL_DIM:]
        .detach()
    )
    annihilator = basis @ local_annihilator
    c1_inner = (
        -dual_rate
        + drift_jacobian @ ambient_metric
        + ambient_metric @ drift_jacobian.T
        + 2.0 * rate * ambient_metric
    )
    c1 = annihilator.T @ c1_inner @ annihilator

    c2 = []
    for index in range(CONTROL_DIM):
        field_fn = lambda value: input_matrix(value)[:, index]
        field = field_fn(state)
        field_jacobian = torch.func.jacrev(field_fn)(state)
        _, directional_dual = _ambient_metric_rate(metric.dual, state, field)
        inner = (
            directional_dual
            - field_jacobian @ ambient_metric
            - ambient_metric @ field_jacobian.T
        )
        c2.append(annihilator.T @ inner @ annihilator)
    projected_dual = local_annihilator.T @ dual @ local_annihilator
    return c1, torch.stack(c2), projected_dual, dual


def sample_batch(
    size: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    velocity_error_max: float = 1.5,
    reference_tilt_max: float = 1.0,
    rotation_error_max: float = 0.7,
    actual_tilt_max: float = 1.3,
    actual_pitch_max: float = 1.0,
    collective_delta: float = 3.0,
    body_rate_max=(1.0, 1.0, 0.5),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample physically consistent ego states and reference signals."""
    accepted = []
    while sum(item[0].shape[0] for item in accepted) < size:
        count = max(32, 2 * (size - sum(item[0].shape[0] for item in accepted)))
        random = lambda *shape: 2.0 * torch.rand(*shape, dtype=dtype) - 1.0
        tilt = reference_tilt_max * torch.rand(count, dtype=dtype)
        direction = torch.pi * random(count)
        gamma_ref = torch.stack(
            (
                torch.sin(tilt) * torch.cos(direction),
                torch.sin(tilt) * torch.sin(direction),
                torch.cos(tilt),
            ),
            dim=-1,
        )
        rotation_ref = rotation_from_gamma_heading(gamma_ref, torch.zeros_like(tilt))
        axes = torch.randn(count, 3, dtype=dtype)
        axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-9)
        unit_angle = torch.rand(count, 1, dtype=dtype)
        unit_angle[count // 2 :] = 0.8 + 0.2 * unit_angle[count // 2 :]
        rotation = rotation_ref @ so3_exp(axes * rotation_error_max * unit_angle)
        gamma = rotation[..., 2, :]
        actual_tilt = torch.acos(rotation[..., 2, 2].clamp(-1.0, 1.0))
        actual_pitch = -torch.asin(gamma[:, 0].clamp(-1.0, 1.0))
        valid = (actual_tilt <= actual_tilt_max) & (
            actual_pitch.abs() <= actual_pitch_max
        )
        if not bool(valid.any()):
            continue
        rotation = rotation[valid]
        rotation_ref = rotation_ref[valid]
        gamma = gamma[valid]
        gamma_ref = gamma_ref[valid]
        count = rotation.shape[0]
        heading = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        velocity_error = velocity_error_max * random(count, 3)
        state = torch.cat((gamma, velocity_error, heading[:, None]), dim=-1)
        collective_ref = GRAVITY + collective_delta * random(count, 1)
        rate_limit = torch.tensor(body_rate_max, dtype=dtype)
        body_rate_ref = rate_limit * random(count, 3)
        control_ref = torch.cat((collective_ref, body_rate_ref), dim=-1)
        e3 = torch.tensor((0.0, 0.0, 1.0), dtype=dtype)
        acceleration_world = -GRAVITY * e3 + collective_ref * rotation_ref[..., :, 2]
        yaw_rate_ref = (
            body_rate_ref[:, 1] * gamma_ref[:, 1]
            + body_rate_ref[:, 2] * gamma_ref[:, 2]
        ) / (gamma_ref[:, 1].square() + gamma_ref[:, 2].square())
        accepted.append((state, acceleration_world, yaw_rate_ref, control_ref))
    return tuple(
        torch.cat([item[index] for item in accepted], dim=0)[:size].to(device)
        for index in range(4)
    )


def fixed_sample_batch(size: int, seed: int, **kwargs) -> tuple[Tensor, ...]:
    """Generate a reproducible sampled set without changing caller RNG state."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    batch = sample_batch(size, **kwargs)
    torch.random.set_rng_state(rng_state)
    return batch


def ccm_loss(
    controller: EgoController,
    metric: EgoMetric,
    batch: tuple[Tensor, ...],
    rate: float,
    c1_weight: float = 1.0,
    c2_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute normalized closed-loop, C1, C2 and metric-bound losses."""
    state, acceleration, yaw_rate, control_ref = batch
    closed, primal, _ = torch.vmap(
        lambda x, a, y, u: contraction_terms(controller, metric, x, a, y, u, rate)
    )(state, acceleration, yaw_rate, control_ref)
    c1, c2, projected_dual, dual = torch.vmap(
        lambda x, a, y: strong_ccm_terms(metric, x, a, y, rate)
    )(state, acceleration, yaw_rate)
    normalized_closed = metric_normalize(closed, primal)
    normalized_c1 = metric_normalize(c1, projected_dual)
    normalized_c2 = metric_normalize(
        c2,
        projected_dual[:, None].expand_as(c2),
    )
    closed_eigenvalues = torch.linalg.eigvalsh(normalized_closed)
    c1_eigenvalues = torch.linalg.eigvalsh(normalized_c1)

    def active_spectral_hinge(values: Tensor) -> Tensor:
        violations = functional.relu(values[..., -1])
        active_count = (violations > 0.0).sum().clamp_min(1)
        return violations.sum() / active_count

    margin = 0.1 * rate
    contraction = active_spectral_hinge(closed_eigenvalues + margin)
    c1_loss = active_spectral_hinge(c1_eigenvalues + margin)
    c2_loss = normalized_c2.square().mean()
    upper = functional.relu(torch.linalg.eigvalsh(dual)[..., -1] - 10.0).mean()
    loss = contraction + c1_weight * c1_loss + c2_weight * c2_loss + upper
    return loss, {
        "loss": float(loss.detach()),
        "contraction_loss": float(contraction.detach()),
        "contracting_fraction": float(
            (closed_eigenvalues[:, -1] < 0.0).float().mean().detach()
        ),
        "max_C_eig": float(closed_eigenvalues[:, -1].max().detach()),
        "c1_fraction": float((c1_eigenvalues[:, -1] < 0.0).float().mean().detach()),
        "c1_loss": float(c1_loss.detach()),
        "c2_loss": float(c2_loss.detach()),
        "metric_upper_loss": float(upper.detach()),
    }


def evaluate(
    controller: EgoController,
    metric: EgoMetric,
    batch: tuple[Tensor, ...],
    rate: float,
    chunk_size: int,
) -> dict[str, float]:
    """Evaluate one fixed sample set."""
    totals = []
    for start in range(0, batch[0].shape[0], chunk_size):
        _, stats = ccm_loss(
            controller,
            metric,
            tuple(value[start : start + chunk_size] for value in batch),
            rate,
        )
        count = min(chunk_size, batch[0].shape[0] - start)
        totals.append((count, stats))
    count = sum(item[0] for item in totals)
    result = {
        key: sum(n * stats[key] for n, stats in totals) / count for key in totals[0][1]
    }
    result["max_C_eig"] = max(stats["max_C_eig"] for _, stats in totals)
    return result


def train(
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    training_size: int = 131072,
    validation_size: int = 32768,
    hidden: int = 64,
    rate: float = 0.5,
    learning_rate: float = 1.0e-3,
    seed: int = 0,
    device: str | None = None,
    checkpoint: str | Path = "neu_ego_ccm.pt",
) -> tuple[EgoController, EgoMetric]:
    """Jointly train an ego-centric controller and dual metric."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    controller = EgoController(hidden).to(device)
    metric = EgoMetric(hidden).to(device)
    parameters = list(controller.parameters()) + list(metric.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.0e-5
    )
    training = fixed_sample_batch(training_size, 20260902, device=device)
    validation = fixed_sample_batch(validation_size, 20260903, device=device)
    generator = torch.Generator().manual_seed(20260904)
    best_score = None
    best = None
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(training_size, generator=generator)
        for start in range(0, training_size, batch_size):
            indices = permutation[start : start + batch_size].to(device)
            batch = tuple(value[indices] for value in training)
            loss, _ = ccm_loss(controller, metric, batch, rate)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
        train_stats = evaluate(controller, metric, training, rate, batch_size)
        validation_stats = evaluate(controller, metric, validation, rate, batch_size)
        score = (
            validation_stats["contracting_fraction"],
            -validation_stats["max_C_eig"],
            validation_stats["c1_fraction"],
            -validation_stats["c2_loss"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best = {
                "epoch": epoch,
                "controller": copy.deepcopy(controller.state_dict()),
                "metric": copy.deepcopy(metric.state_dict()),
                "training": train_stats,
                "validation": validation_stats,
            }
        print(
            f"epoch {epoch:03d}: train={train_stats['contracting_fraction']:.3%}, "
            f"val={validation_stats['contracting_fraction']:.3%}, "
            f"max_C={validation_stats['max_C_eig']:.4g}, "
            f"C1={validation_stats['c1_fraction']:.3%}, "
            f"C2={validation_stats['c2_loss']:.4g}"
        )
        scheduler.step()
    assert best is not None
    controller.load_state_dict(best["controller"])
    metric.load_state_dict(best["metric"])
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best,
            "config": {
                "architecture": "ego_s2_r3_s1_c3m_v4",
                "hidden": hidden,
                "rate": rate,
                "seed": seed,
                "contraction_loss": "active_max_eigenvalue_hinge",
                "lr_schedule": "cosine_annealing_to_1e-5",
                "control_lower": PEGASUS_CONTROL_LOWER,
                "control_upper": PEGASUS_CONTROL_UPPER,
                "velocity_error_max": 1.5,
                "reference_tilt_max": 1.0,
                "rotation_error_max": 0.7,
                "actual_tilt_max": 1.3,
                "actual_pitch_max": 1.0,
                "collective_delta": 3.0,
                "body_rate_max": (1.0, 1.0, 0.5),
                "training_size": training_size,
                "validation_size": validation_size,
            },
        },
        checkpoint,
    )
    return controller, metric


def load_models(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[EgoController, EgoMetric, dict[str, object]]:
    """Load one ego-centric checkpoint with architecture validation."""
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    config = data["config"]
    if config.get("architecture") != "ego_s2_r3_s1_c3m_v4":
        raise ValueError("checkpoint is not an ego-centric CCM model")
    controller = EgoController(
        int(config["hidden"]),
        config["control_lower"],
        config["control_upper"],
    ).to(device)
    metric = EgoMetric(int(config["hidden"])).to(device)
    controller.load_state_dict(data["controller"])
    metric.load_state_dict(data["metric"])
    return controller.eval(), metric.eval(), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--training-size", type=int, default=131072)
    parser.add_argument("--validation-size", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", default="neu_ego_ccm.pt")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_size=args.training_size,
        validation_size=args.validation_size,
        hidden=args.hidden,
        rate=args.rate,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
