"""Neural contraction metric and CTBR controller on R^3 x SO(3).

The physical state is ``(v, R)`` and control is
``[collective_thrust, body_rate_x, body_rate_y, body_rate_z]``. Contraction is
checked in the 6-D right-trivialized tangent space ``[delta_v, delta_theta]``.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from spatialmath import SO3
import torch
from torch import Tensor, nn
import torch.nn.functional as F

PEGASUS_CONTROL_LOWER = (
    1.6913793103448278,
    -3.839724354387525,
    -3.839724354387525,
    -1.57,
)
PEGASUS_CONTROL_UPPER = (
    15.22241379310345,
    3.839724354387525,
    3.839724354387525,
    1.57,
)

# Differentiable SO(3) operations ------------------------------------------


def hat(x: Tensor) -> Tensor:
    """Batched hat map, (..., 3) -> (..., 3, 3)."""
    z = torch.zeros_like(x[..., 0])
    x0, x1, x2 = x.unbind(dim=-1)
    return torch.stack((z, -x2, x1, x2, z, -x0, -x1, x0, z), dim=-1).reshape(
        x.shape[:-1] + (3, 3)
    )


def vee(X: Tensor) -> Tensor:
    return torch.stack((X[..., 2, 1], X[..., 0, 2], X[..., 1, 0]), dim=-1)


def so3_exp(phi: Tensor) -> Tensor:
    """Differentiable exponential map, including first/second derivatives at zero."""
    theta2 = (phi * phi).sum(dim=-1, keepdim=True)
    # Clamp only protects the inactive exact-form branch at theta=0. The
    # selected Taylor branch remains the true local expansion and differentiates
    # through theta^2 without the undefined derivative of sqrt(0).
    theta = torch.sqrt(theta2.clamp_min(torch.finfo(phi.dtype).eps))
    small = theta2 < 1e-6
    A_taylor = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    B_taylor = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    A = torch.where(small, A_taylor, torch.sin(theta) / theta)
    B = torch.where(small, B_taylor, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-6))
    K = hat(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I + A[..., None] * K + B[..., None] * (K @ K)


def attitude_error(R: Tensor, R_ref: Tensor) -> Tensor:
    """Geometric SO(3) tracking error."""
    E = R_ref.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_ref
    return 0.5 * vee(E)


def retract(v: Tensor, R: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Right retraction for tangent coordinate q=[dv,dtheta]."""
    return v + q[..., :3], R @ so3_exp(q[..., 3:])


def torch_dynamics(v: Tensor, R: Tensor, u: Tensor, g: float = 9.81) -> Tensor:
    """Intrinsic state velocity [v_dot, omega]."""
    e3 = torch.tensor([0.0, 0.0, 1.0], dtype=v.dtype, device=v.device)
    acceleration = -g * e3 + u[..., :1] * (R @ e3)
    return torch.cat((acceleration, u[..., 1:]), dim=-1)


# Networks ------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CTBRController(nn.Module):
    """Factorized residual controller that outputs CTBR commands.

    Inputs
    ------
    ``(v, R)`` is the current state, ``(v_ref, R_ref)`` is the reference state,
    and ``ctbr_ref`` is its feedforward CTBR command.  The six-dimensional
    geometric tracking error is

    ``e = [v - v_ref, attitude_error(R, R_ref)]``.

    The feedback is ``Phi_2(context) @ tanh(Phi_1(context) @ error)`` and the
    final output is ``ctbr_ref + feedback``. Therefore the controller exactly
    returns the feedforward CTBR at zero tracking error.
    """

    CTBR_DIM = 4
    ERROR_DIM = 6
    FULL_CONTEXT_DIM = 28
    FACTOR_DIM = 3 * ERROR_DIM

    def __init__(
        self,
        hidden=128,
        control_lower=PEGASUS_CONTROL_LOWER,
        control_upper=PEGASUS_CONTROL_UPPER,
        init_seed=None,
        initialize_linear_gain=True,
    ):
        super().__init__()
        self.factor_dim = self.FACTOR_DIM

        # Factorized residual: Phi_2(context) @ tanh(Phi_1(context) @ error).
        self.factor_input = MLP(
            self.FULL_CONTEXT_DIM,
            self.factor_dim * self.ERROR_DIM,
            hidden,
        )
        self.factor_output = MLP(
            self.FULL_CONTEXT_DIM,
            self.CTBR_DIM * self.factor_dim,
            hidden,
        )

        if init_seed is not None:
            self._reset_mlp(self.factor_input, init_seed + 2)
            self._reset_mlp(self.factor_output, init_seed + 3)
        nn.init.zeros_(self.factor_output.net[-1].weight)
        nn.init.zeros_(self.factor_output.net[-1].bias)
        linear_gain = torch.zeros(self.CTBR_DIM, self.ERROR_DIM)
        if initialize_linear_gain:
            linear_gain[0, 2] = -2.0
            linear_gain[1, 1], linear_gain[1, 3] = 1.5, -3.0
            linear_gain[2, 0], linear_gain[2, 4] = -1.5, -3.0
            linear_gain[3, 5] = -2.0
        self.linear_gain = nn.Parameter(linear_gain)
        lower = torch.as_tensor(control_lower, dtype=torch.float32)
        upper = torch.as_tensor(control_upper, dtype=torch.float32)
        if lower.shape != (self.CTBR_DIM,) or upper.shape != (self.CTBR_DIM,):
            raise ValueError("control bounds must contain four values")
        if torch.any(lower >= upper):
            raise ValueError("control lower bounds must be below upper bounds")
        self.register_buffer(
            "control_lower",
            lower,
            persistent=False,
        )
        self.register_buffer(
            "control_upper",
            upper,
            persistent=False,
        )

    @staticmethod
    def _reset_mlp(module, seed):
        """Give shared branches identical initialization across ablations."""
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        torch.random.set_rng_state(rng_state)

    def apply_feedback(self, ctbr_ref, raw_feedback):
        """Map residual feedback to the configured physical command set."""
        lower = self.control_lower.to(raw_feedback)
        upper = self.control_upper.to(raw_feedback)
        span = upper - lower
        q = ((ctbr_ref - lower) / span).clamp(1e-6, 1.0 - 1e-6)
        margin = (ctbr_ref - lower) * (upper - ctbr_ref) / span
        margin = margin.clamp_min(torch.finfo(raw_feedback.dtype).eps)
        latent = torch.logit(q) + raw_feedback / margin
        return lower + span * torch.sigmoid(latent)

    def error_and_context(self, v, R, v_ref, R_ref, ctbr_ref):
        """Return feedback error and the corresponding gain-scheduling context."""
        error = torch.cat((v - v_ref, attitude_error(R, R_ref)), dim=-1)
        state_reference = (v, R.flatten(-2), v_ref, R_ref.flatten(-2))
        return error, torch.cat((*state_reference, ctbr_ref), dim=-1)

    def raw_feedback(self, v, R, v_ref, R_ref, ctbr_ref):
        """Return the residual before the final command map."""
        single = v.ndim == 1
        if single:
            v, R = v[None], R[None]
            v_ref, R_ref, ctbr_ref = v_ref[None], R_ref[None], ctbr_ref[None]

        error, context = self.error_and_context(v, R, v_ref, R_ref, ctbr_ref)
        velocity_error_body = torch.einsum(
            "bij,bj->bi", R.transpose(-1, -2), error[:, :3]
        )
        linear_error = torch.cat((velocity_error_body, error[:, 3:]), dim=-1)
        phi_1 = self.factor_input(context).reshape(-1, self.factor_dim, self.ERROR_DIM)
        phi_2 = self.factor_output(context).reshape(-1, self.CTBR_DIM, self.factor_dim)
        latent = torch.tanh(torch.einsum("bij,bj->bi", phi_1, error))
        raw_feedback = torch.einsum("ij,bj->bi", self.linear_gain, linear_error)
        raw_feedback = raw_feedback + torch.einsum("bij,bj->bi", phi_2, latent)
        return raw_feedback[0] if single else raw_feedback

    def forward(self, v, R, v_ref, R_ref, ctbr_ref):
        """Return ``[collective_acceleration, p_cmd, q_cmd, r_cmd]``."""
        raw_feedback = self.raw_feedback(v, R, v_ref, R_ref, ctbr_ref)
        ctbr = self.apply_feedback(ctbr_ref, raw_feedback)
        return ctbr


class ContractionMetric(nn.Module):
    """Smooth SPD dual metric W; ``forward`` returns M=W^-1."""

    def __init__(self, hidden=128, min_eigenvalue=0.1, representation="dual"):
        super().__init__()
        if representation not in {"dual", "primal"}:
            raise ValueError("metric representation must be 'dual' or 'primal'")
        self.factor = MLP(12, 36, hidden)
        self.min_eigenvalue = min_eigenvalue
        self.representation = representation

    def _network_metric(self, v, R):
        single = v.ndim == 1
        if single:
            v, R = v.unsqueeze(0), R.unsqueeze(0)
        raw = self.factor(torch.cat((v, R.flatten(-2)), dim=-1)).reshape(-1, 6, 6)
        I = torch.eye(6, dtype=v.dtype, device=v.device).expand_as(raw)
        theta = I + 0.1 * raw
        matrix = theta.transpose(-1, -2) @ theta + self.min_eigenvalue * I
        return matrix[0] if single else matrix

    def dual(self, v, R):
        """Return W directly for new models and by inversion for legacy models."""
        matrix = self._network_metric(v, R)
        return matrix if self.representation == "dual" else torch.linalg.inv(matrix)

    def primal(self, v, R):
        """Return M, the inverse of the directly learned W."""
        matrix = self._network_metric(v, R)
        return torch.linalg.inv(matrix) if self.representation == "dual" else matrix

    def forward(self, v, R):
        return self.primal(v, R)


def input_annihilator(R):
    """Return B_perp for CTBR inputs in [delta_v, delta_theta] coordinates."""
    zeros = torch.zeros(3, 2, dtype=R.dtype, device=R.device)
    return torch.cat((R[:, :2], zeros), dim=0)


def metric_normalize(matrix, metric):
    """Return L^-1 matrix L^-T for metric=L L^T."""
    factor = torch.linalg.cholesky(metric)
    left = torch.linalg.solve_triangular(factor, matrix, upper=False)
    normalized = torch.linalg.solve_triangular(
        factor,
        left.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)
    return 0.5 * (normalized + normalized.transpose(-1, -2))


def strong_ccm_terms(metric, v, R, rate=1.0):
    """Return C1/C2 matrices for the directly learned dual metric W."""
    zero = torch.zeros(6, dtype=v.dtype, device=v.device)

    def local_dual(q):
        vq, Rq = retract(v, R, q)
        return metric.dual(vq, Rq)

    W = local_dual(zero)
    e3 = torch.tensor((0.0, 0.0, 1.0), dtype=v.dtype, device=v.device)
    zero3 = torch.zeros(3, dtype=v.dtype, device=v.device)
    drift = torch.cat((-9.81 * e3, zero3))
    _, drift_derivative = torch.func.jvp(local_dual, (zero,), (drift,))
    annihilator = input_annihilator(R)
    c1_inner = -drift_derivative + 2.0 * rate * W
    c1 = annihilator.T @ c1_inner @ annihilator

    zeros = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
    axes = torch.eye(3, dtype=v.dtype, device=v.device)
    fields = [torch.cat((R @ e3, zero3))]
    fields.extend(torch.cat((zero3, axis)) for axis in axes)
    field_jacobians = [
        torch.cat(
            (
                torch.cat((zeros, -R @ hat(e3)), dim=1),
                torch.cat((zeros, zeros), dim=1),
            ),
            dim=0,
        )
    ]
    for axis in axes:
        field_jacobians.append(
            torch.cat(
                (
                    torch.cat((zeros, zeros), dim=1),
                    torch.cat((zeros, -hat(axis)), dim=1),
                ),
                dim=0,
            )
        )
    c2 = []
    for field, field_jacobian in zip(fields, field_jacobians):
        _, derivative = torch.func.jvp(local_dual, (zero,), (field,))
        inner = derivative - field_jacobian @ W - W @ field_jacobian.T
        c2.append(annihilator.T @ inner @ annihilator)
    return c1, torch.stack(c2), W


def sampling_ranges(domain_scale=1.0, body_rate_range=(1.0, 1.0, 0.5)):
    """Return the design domain with a feasible nominal CTBR-rate margin."""
    if not 0.0 <= domain_scale <= 1.0:
        raise ValueError("domain_scale must be in [0, 1]")

    # The 2x benchmark peaks at 1.932 m/s on one axis and 2.453 m/s in
    # Euclidean norm.  Sampling each component up to 2.5 m/s retains margin
    # without spending most training capacity on unreachable cube corners.
    if len(body_rate_range) != 3 or any(limit <= 0.0 for limit in body_rate_range):
        raise ValueError("body_rate_range must contain three positive values")
    narrow = (1.5, 1.0, 0.6, 2.0)
    wide = (2.5, 1.5, 0.7, 3.0)
    lerp = lambda lo, hi: lo + domain_scale * (hi - lo)
    return {
        "velocity_range": lerp(narrow[0], wide[0]),
        "velocity_error": lerp(narrow[1], wide[1]),
        "reference_tilt_angle_max": 1.0,
        "reference_yaw_angle_max": torch.pi,
        "rotation_error_angle_max": lerp(narrow[2], wide[2]),
        "collective_delta": lerp(narrow[3], wide[3]),
        # The fixed 2x benchmark peaks at approximately (0.556, 0.172, 0.078)
        # rad/s.  Keep nominal rates strictly inside the +/-4 rad/s command
        # bounds so the learned feedback has authority in both directions.
        "body_rate_range": tuple(float(limit) for limit in body_rate_range),
    }


def sample_batch(
    batch_size,
    device="cpu",
    dtype=None,
    velocity_range=2.5,
    velocity_error=1.5,
    reference_tilt_angle_max=1.0,
    reference_yaw_angle_max=torch.pi,
    rotation_error_angle_max=0.7,
    collective_delta=3.0,
    body_rate_range=(4.0, 4.0, 3.0),
):
    """Sample local tracking pairs and feasible instantaneous nominal inputs."""

    dtype = dtype or torch.float32
    rand = lambda *shape: 2.0 * torch.rand(*shape, device=device, dtype=dtype) - 1.0
    v_ref = velocity_range * rand(batch_size, 3)
    v = v_ref + velocity_error * rand(batch_size, 3)
    yaw = reference_yaw_angle_max * rand(batch_size)
    tilt_direction = torch.pi * rand(batch_size)
    tilt_angle = reference_tilt_angle_max * torch.rand(
        batch_size, device=device, dtype=dtype
    )
    zeros = torch.zeros_like(yaw)
    tilt = torch.stack(
        (
            tilt_angle * torch.cos(tilt_direction),
            tilt_angle * torch.sin(tilt_direction),
            zeros,
        ),
        dim=-1,
    )
    R_ref = so3_exp(torch.stack((zeros, zeros, yaw), dim=-1)) @ so3_exp(tilt)

    axes = torch.randn(batch_size, 3, device=device, dtype=dtype)
    axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-12)
    unit_angle = torch.rand(batch_size, 1, device=device, dtype=dtype)
    half = batch_size // 2
    unit_angle[half:] = 0.8 + 0.2 * unit_angle[half:]
    rotation_error = axes * rotation_error_angle_max * unit_angle
    R = R_ref @ so3_exp(rotation_error)

    collective_ref = 9.81 + collective_delta * rand(batch_size, 1)
    rate_limit = torch.tensor(
        body_rate_range,
        device=device,
        dtype=dtype,
    )
    body_rate_ref = rate_limit * rand(batch_size, 3)

    u_ref = torch.cat((collective_ref, body_rate_ref), dim=-1)
    return v, R, v_ref, R_ref, u_ref


def fixed_sample_batch(
    size,
    seed,
    device="cpu",
    domain_scale=1.0,
    body_rate_range=(1.0, 1.0, 0.5),
):
    """Generate one reproducible sampled design set."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    batch = sample_batch(
        size,
        device="cpu",
        **sampling_ranges(domain_scale, body_rate_range),
    )
    torch.random.set_rng_state(rng_state)
    return tuple(x.to(device) for x in batch)


def contraction_terms(controller, metric, v, R, v_ref, R_ref, u_ref, rate=1.0):
    """Return the terms of the right-trivialized contraction matrix."""

    zero = torch.zeros(6, dtype=v.dtype, device=v.device)

    def closed_loop(q):
        vq, Rq = retract(v, R, q)
        uq = controller(vq, Rq, v_ref, R_ref, u_ref)
        return torch_dynamics(vq, Rq, uq)

    f = closed_loop(zero)
    A = torch.func.jacrev(closed_loop)(zero)
    # Right-trivialized attitude variation: dtheta_dot=-hat(omega)dtheta+domega.
    attitude_correction = torch.cat(
        (
            torch.zeros(3, 6, dtype=v.dtype, device=v.device),
            torch.cat(
                (torch.zeros(3, 3, dtype=v.dtype, device=v.device), -hat(f[3:])), 1
            ),
        ),
        0,
    )
    A = A + attitude_correction

    def local_primal_metric(q):
        vq, Rq = retract(v, R, q)
        return metric.primal(vq, Rq)

    M, M_dot = torch.func.jvp(local_primal_metric, (zero,), (f,))
    C = M_dot + A.T @ M + M @ A + 2.0 * rate * M
    C = 0.5 * (C + C.T)
    return C, M, controller(v, R, v_ref, R_ref, u_ref), f, A, M_dot


def contraction_matrix(controller, metric, v, R, v_ref, R_ref, u_ref, rate=1.0):
    """Compute Mdot + Acl.T M + M Acl + 2*rate*M on R^3 x SO(3)."""
    C, M, u, _, _, _ = contraction_terms(
        controller, metric, v, R, v_ref, R_ref, u_ref, rate
    )
    return C, M, u


def ccm_loss(
    controller,
    metric,
    batch,
    rate=1.0,
    max_dual_eigenvalue=10.0,
    c1_weight=1.0,
    c2_weight=1.0,
):
    """Scale-invariant C3M loss for a directly parameterized dual metric."""
    v, R, v_ref, R_ref, u_ref = batch

    def one_sample(vi, Ri, vri, Rri, uri):
        return contraction_matrix(controller, metric, vi, Ri, vri, Rri, uri, rate)

    C, M, _ = torch.vmap(one_sample)(v, R, v_ref, R_ref, u_ref)
    C1, C2, W = torch.vmap(lambda vi, Ri: strong_ccm_terms(metric, vi, Ri, rate))(v, R)
    annihilator = torch.vmap(input_annihilator)(R)
    dual_perp = annihilator.transpose(-1, -2) @ W @ annihilator
    normalized_c = metric_normalize(C, M)
    normalized_c1 = metric_normalize(C1, dual_perp)
    expanded_dual = dual_perp[:, None].expand_as(C2)
    normalized_c2 = metric_normalize(C2, expanded_dual)
    c_eigs = torch.linalg.eigvalsh(normalized_c)
    c1_eigs = torch.linalg.eigvalsh(normalized_c1)
    m_eigs = torch.linalg.eigvalsh(M)
    w_eigs = torch.linalg.eigvalsh(W)

    # Penalize every violated eigendirection, not only the current maximum.
    # Averaging over active violations prevents already-contracting samples from
    # diluting the few hard samples left near the end of training.  A linear
    # hinge keeps a useful gradient at the constraint boundary.
    margin = 0.1 * rate

    def active_hinge(values):
        violations = F.relu(values)
        active_count = (values > 0.0).sum().clamp_min(1)
        return violations.sum() / active_count

    contraction = active_hinge(c_eigs + margin)
    c1 = active_hinge(c1_eigs + margin)
    c2 = normalized_c2.square().mean()
    upper = active_hinge(w_eigs - max_dual_eigenvalue)
    loss = contraction + c1_weight * c1 + c2_weight * c2 + upper
    stats = {
        "loss": float(loss.detach()),
        "max_C_eig": float(c_eigs[..., -1].max().detach()),
        "mean_C_eig": float(c_eigs[..., -1].mean().detach()),
        "contracting_fraction": float((c_eigs[..., -1] < 0.0).float().mean().detach()),
        "contraction_loss": float(contraction.detach()),
        "c1_loss": float(c1.detach()),
        "c2_loss": float(c2.detach()),
        "c1_fraction": float((c1_eigs[..., -1] < 0.0).float().mean().detach()),
        "max_C1_eig": float(c1_eigs[..., -1].max().detach()),
        "metric_upper_loss": float(upper.detach()),
        "min_M_eig": float(m_eigs[..., 0].min().detach()),
        "max_M_eig": float(m_eigs[..., -1].max().detach()),
        "min_W_eig": float(w_eigs[..., 0].min().detach()),
        "max_W_eig": float(w_eigs[..., -1].max().detach()),
    }
    return loss, stats


def evaluate_ccm(
    controller,
    metric,
    batch,
    rate=1.0,
    chunk_size=64,
    control_lower=PEGASUS_CONTROL_LOWER,
    control_upper=PEGASUS_CONTROL_UPPER,
):
    """Evaluate contraction on a fixed held-out batch without parameter gradients."""
    v, R, v_ref, R_ref, u_ref = batch
    parameters = list(controller.parameters()) + list(metric.parameters())
    grad_flags = [parameter.requires_grad for parameter in parameters]
    was_training = controller.training, metric.training
    controller.eval()
    metric.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)

    max_c_eigs = []
    max_c1_eigs, c2_losses = [], []
    min_m_eigs, max_m_eigs = [], []
    min_w_eigs, max_w_eigs = [], []
    controls, control_violations = [], []
    try:
        for start in range(0, v.shape[0], chunk_size):
            stop = start + chunk_size
            chunk = tuple(x[start:stop] for x in batch)

            def one_sample(vi, Ri, vri, Rri, uri):
                return contraction_matrix(
                    controller, metric, vi, Ri, vri, Rri, uri, rate
                )

            C, M, u = torch.vmap(one_sample)(*chunk)
            C1, C2, W = torch.vmap(
                lambda vi, Ri: strong_ccm_terms(metric, vi, Ri, rate)
            )(chunk[0], chunk[1])
            c_eigs = torch.linalg.eigvalsh(C).detach()
            c1_eigs = torch.linalg.eigvalsh(C1).detach()
            m_eigs = torch.linalg.eigvalsh(M).detach()
            max_c_eigs.append(c_eigs[:, -1])
            max_c1_eigs.append(c1_eigs[:, -1])
            annihilator = torch.vmap(input_annihilator)(chunk[1])
            dual_perp = annihilator.transpose(-1, -2) @ W @ annihilator
            normalized_c2 = metric_normalize(
                C2,
                dual_perp[:, None].expand_as(C2),
            )
            w_eigs = torch.linalg.eigvalsh(W).detach()
            c2_losses.append(normalized_c2.detach().square().mean(dim=(-2, -1)))
            min_m_eigs.append(m_eigs[:, 0])
            max_m_eigs.append(m_eigs[:, -1])
            min_w_eigs.append(w_eigs[:, 0])
            max_w_eigs.append(w_eigs[:, -1])
            lower = torch.as_tensor(control_lower, dtype=u.dtype, device=u.device)
            upper = torch.as_tensor(control_upper, dtype=u.dtype, device=u.device)
            span = upper - lower
            violation = F.relu(lower - u) + F.relu(u - upper)
            controls.append(u.detach())
            control_violations.append((violation / span).detach())
    finally:
        for parameter, requires_grad in zip(parameters, grad_flags):
            parameter.requires_grad_(requires_grad)
        controller.train(was_training[0])
        metric.train(was_training[1])

    c_max = torch.cat(max_c_eigs)
    c1_max = torch.cat(max_c1_eigs)
    c2_values = torch.cat(c2_losses)
    m_min = torch.cat(min_m_eigs)
    m_max = torch.cat(max_m_eigs)
    w_min = torch.cat(min_w_eigs)
    w_max = torch.cat(max_w_eigs)
    control_violation = torch.cat(control_violations)
    controls = torch.cat(controls)
    quantiles = torch.quantile(
        c_max, torch.tensor([0.5, 0.9, 0.95, 0.99], device=c_max.device)
    )
    return {
        "samples": int(c_max.numel()),
        "contracting_fraction": float((c_max < 0.0).float().mean()),
        "mean_C_eig": float(c_max.mean()),
        "median_C_eig": float(quantiles[0]),
        "p90_C_eig": float(quantiles[1]),
        "p95_C_eig": float(quantiles[2]),
        "p99_C_eig": float(quantiles[3]),
        "max_C_eig": float(c_max.max()),
        "c1_fraction": float((c1_max < 0.0).float().mean()),
        "max_C1_eig": float(c1_max.max()),
        "c2_loss": float(c2_values.mean()),
        "min_control_by_channel": controls.min(dim=0).values.tolist(),
        "max_control_by_channel": controls.max(dim=0).values.tolist(),
        "control_bound_fraction": float(
            (control_violation > 0.0).any(dim=-1).float().mean()
        ),
        "max_normalized_control_violation": float(control_violation.max()),
        "min_M_eig": float(m_min.min()),
        "max_M_eig": float(m_max.max()),
        "max_condition_bound": float((m_max / m_min).max()),
        "min_W_eig": float(w_min.min()),
        "max_W_eig": float(w_max.max()),
    }


def train_joint_ccm(
    epochs=15,
    batch_size=1024,
    learning_rate=1e-3,
    lr_step=5,
    lr_gamma=0.3,
    rate=1.0,
    hidden=128,
    seed=0,
    device=None,
    checkpoint="neu_ccm_linear.pt",
    controller_hidden=128,
    training_size=131072,
    training_seed=20260828,
    validation_size=32768,
    validation_seed=20260829,
    control_lower=PEGASUS_CONTROL_LOWER,
    control_upper=PEGASUS_CONTROL_UPPER,
    domain_scale=1.0,
    body_rate_range=(1.0, 1.0, 0.5),
    c1_weight=1.0,
    c2_weight=1.0,
    resume_checkpoint=None,
):
    """Train on one fixed design set and save the best validation epoch."""

    if len(control_lower) != 4 or len(control_upper) != 4:
        raise ValueError("control_lower and control_upper must contain four values")
    if any(lo >= hi for lo, hi in zip(control_lower, control_upper)):
        raise ValueError("each control lower bound must be below its upper bound")
    if c1_weight < 0.0 or c2_weight < 0.0:
        raise ValueError("C1 and C2 weights must be nonnegative")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if training_size < batch_size:
        raise ValueError("training_size must be at least batch_size")
    torch.manual_seed(seed + 10_000)
    if resume_checkpoint is None:
        metric = ContractionMetric(hidden=hidden, representation="dual").to(device)
        controller = CTBRController(
            hidden=controller_hidden,
            control_lower=control_lower,
            control_upper=control_upper,
            init_seed=seed,
        ).to(device)
    else:
        controller, metric, resume_config = load_models(
            resume_checkpoint,
            device=device,
        )
        hidden = int(resume_config["hidden"])
        controller_hidden = int(resume_config.get("controller_hidden", hidden))
        if resume_config.get("metric_representation") != "dual":
            raise ValueError("C3M training can only resume a direct-W checkpoint")
        controller.train()
        metric.train()
    params = list(controller.parameters()) + list(metric.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma
    )
    ranges = sampling_ranges(domain_scale, body_rate_range)
    reference_lower = (
        9.81 - ranges["collective_delta"],
        *(-limit for limit in ranges["body_rate_range"]),
    )
    reference_upper = (
        9.81 + ranges["collective_delta"],
        *ranges["body_rate_range"],
    )
    training_batch = fixed_sample_batch(
        training_size,
        training_seed,
        device=device,
        domain_scale=domain_scale,
        body_rate_range=body_rate_range,
    )
    validation_batch = fixed_sample_batch(
        validation_size,
        validation_seed,
        device=device,
        domain_scale=domain_scale,
        body_rate_range=body_rate_range,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(seed + 20_000)
    print(
        f"fixed training samples={training_size}, training seed={training_seed}, "
        f"validation samples={validation_size}, domain scale={domain_scale:g}, "
        f"ranges={ranges}"
    )
    best_score = None
    best_epoch = 0
    best_state = None
    last_state = None
    for epoch in range(epochs):
        epoch_stats = []
        permutation = torch.randperm(training_size, generator=shuffle_generator)
        for start in range(0, training_size, batch_size):
            indices = permutation[start : start + batch_size].to(device)
            mini_batch = tuple(x[indices] for x in training_batch)
            loss, stats = ccm_loss(
                controller,
                metric,
                mini_batch,
                rate=rate,
                c1_weight=c1_weight,
                c2_weight=c2_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            epoch_stats.append(stats)
        training = evaluate_ccm(
            controller,
            metric,
            training_batch,
            rate=rate,
            chunk_size=batch_size,
            control_lower=control_lower,
            control_upper=control_upper,
        )
        validation = evaluate_ccm(
            controller,
            metric,
            validation_batch,
            rate=rate,
            chunk_size=batch_size,
            control_lower=control_lower,
            control_upper=control_upper,
        )
        mean_loss = sum(x["loss"] for x in epoch_stats) / len(epoch_stats)
        mean_c1 = sum(x["c1_loss"] for x in epoch_stats) / len(epoch_stats)
        mean_c2 = sum(x["c2_loss"] for x in epoch_stats) / len(epoch_stats)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:03d}/{epochs}: lr={current_lr:.2g}, "
            f"train loss={mean_loss:.4g}, "
            f"train max eig(C)={training['max_C_eig']:.4g}, "
            f"train contracting={training['contracting_fraction']:.1%}, "
            f"val max eig(C)={validation['max_C_eig']:.4g}, "
            f"val contracting={validation['contracting_fraction']:.5%}, "
            f"C1={validation['c1_fraction']:.5%}/{mean_c1:.3g}, "
            f"C2={mean_c2:.3g}, "
            f"eig(M)=[{validation['min_M_eig']:.3g}, {validation['max_M_eig']:.3g}]"
        )
        score = (
            validation["contracting_fraction"],
            -validation["max_C_eig"],
            validation["c1_fraction"],
            -validation["c2_loss"],
            -validation["p99_C_eig"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                "controller": copy.deepcopy(controller.state_dict()),
                "metric": copy.deepcopy(metric.state_dict()),
                "training": training,
                "validation": validation,
            }
        last_state = {
            "epoch": epoch + 1,
            "controller": copy.deepcopy(controller.state_dict()),
            "metric": copy.deepcopy(metric.state_dict()),
            "training": training,
            "validation": validation,
        }
        scheduler.step()
    controller.load_state_dict(best_state["controller"])
    metric.load_state_dict(best_state["metric"])
    config = {
        "hidden": hidden,
        "controller_hidden": controller_hidden,
        "rate": rate,
        "learning_rate": learning_rate,
        "lr_step": lr_step,
        "lr_gamma": lr_gamma,
        "contraction_loss": "metric_normalized_active_eigenvalue_hinge",
        "contraction_margin_factor": 0.1,
        "auxiliary_loss_normalization": "metric_congruence",
        "c1_weight": c1_weight,
        "c2_weight": c2_weight,
        "metric_representation": "dual",
        "control_lower": tuple(control_lower),
        "control_upper": tuple(control_upper),
        "reference_lower": tuple(reference_lower),
        "reference_upper": tuple(reference_upper),
        "seed": seed,
        "controller_arch": "factorized_control_bounds_c3m_v10_linear_gain",
        "rotation_sampling": "yaw_tilt_axis_angle_boundary_v1",
        "controller_type": "residual",
        "factor_dim": controller.factor_dim,
        "selection": "closed_fraction_then_worst_C_then_C1_C2_p99",
        "best_epoch": best_epoch,
        "training_seed": training_seed,
        "training_size": training_size,
        "validation_seed": validation_seed,
        "validation_size": validation_size,
        "domain_scale": domain_scale,
        "sampling_ranges": ranges,
        "resume_checkpoint": (
            str(Path(resume_checkpoint).resolve())
            if resume_checkpoint is not None
            else None
        ),
    }
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "controller": best_state["controller"],
                "metric": best_state["metric"],
                "config": config,
                "training": best_state["training"],
                "validation": best_state["validation"],
                "last": last_state,
            },
            Path(checkpoint),
        )
    print(
        f"best epoch={best_epoch}, "
        f"val contracting={best_state['validation']['contracting_fraction']:.5%}, "
        f"C1={best_state['validation']['c1_fraction']:.5%}, "
        f"C2={best_state['validation']['c2_loss']:.4g}, "
        f"val p99/max eig(C)=[{best_state['validation']['p99_C_eig']:.4g}, "
        f"{best_state['validation']['max_C_eig']:.4g}], "
        f"checkpoint={checkpoint}"
    )
    return controller, metric


def load_models(checkpoint, device="cpu"):
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    architecture = data.get("config", {}).get("controller_arch")
    controller_type = data.get("config", {}).get("controller_type")
    architectures = {
        "factorized_control_bounds_c3m_v9_axis_angle",
        "factorized_control_bounds_c3m_v10_linear_gain",
    }
    if architecture not in architectures or controller_type != "residual":
        raise ValueError("checkpoint does not use the maintained bounded controller")
    hidden = data["config"]["hidden"]
    controller_hidden = data["config"].get("controller_hidden", hidden)
    controller = CTBRController(
        hidden=controller_hidden,
        control_lower=data["config"]["control_lower"],
        control_upper=data["config"]["control_upper"],
        initialize_linear_gain=architecture.endswith("linear_gain"),
    ).to(device)
    metric = ContractionMetric(
        hidden=hidden,
        representation=data["config"].get("metric_representation", "primal"),
    ).to(device)
    controller.load_state_dict(
        data["controller"],
        strict=architecture.endswith("linear_gain"),
    )
    metric.load_state_dict(data["metric"])
    return controller.eval(), metric.eval(), data["config"]


def ctbr_numpy(controller, v, R, v_ref, R_ref, u_ref):
    """Adapter from numpy/spatialmath states to a CTBR ndarray."""

    p = next(controller.parameters())
    as_tensor = lambda x: torch.as_tensor(x, dtype=p.dtype, device=p.device)
    with torch.no_grad():
        u = controller(
            as_tensor(v),
            as_tensor(R.R if isinstance(R, SO3) else R),
            as_tensor(v_ref),
            as_tensor(R_ref.R if isinstance(R_ref, SO3) else R_ref),
            as_tensor(u_ref),
        )
    return u.cpu().numpy()


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step", type=int, default=5)
    parser.add_argument("--lr-gamma", type=float, default=0.3)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--controller-hidden",
        type=int,
        default=128,
        help="controller hidden width",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="neu_ccm_linear.pt")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--training-size",
        type=int,
        default=131072,
        help="fixed design-set size",
    )
    parser.add_argument("--training-seed", type=int, default=20260828)
    parser.add_argument("--validation-size", type=int, default=32768)
    parser.add_argument("--validation-seed", type=int, default=20260829)
    parser.add_argument("--c1-weight", type=float, default=1.0)
    parser.add_argument("--c2-weight", type=float, default=1.0)
    parser.add_argument(
        "--control-lower",
        type=float,
        nargs=4,
        default=PEGASUS_CONTROL_LOWER,
        metavar=("T_MIN", "P_MIN", "Q_MIN", "R_MIN"),
    )
    parser.add_argument(
        "--control-upper",
        type=float,
        nargs=4,
        default=PEGASUS_CONTROL_UPPER,
        metavar=("T_MAX", "P_MAX", "Q_MAX", "R_MAX"),
    )
    parser.add_argument(
        "--domain-scale",
        type=float,
        default=1.0,
        help="sampling domain: 0=validated narrow, 1=wide",
    )
    parser.add_argument(
        "--reference-body-rate-range",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 0.5),
        metavar=("P_REF_MAX", "Q_REF_MAX", "R_REF_MAX"),
        help="symmetric nominal body-rate sampling limits in rad/s",
    )
    args = parser.parse_args()
    train_joint_ccm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        rate=args.rate,
        hidden=args.hidden,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        controller_hidden=args.controller_hidden,
        training_size=args.training_size,
        training_seed=args.training_seed,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        control_lower=args.control_lower,
        control_upper=args.control_upper,
        domain_scale=args.domain_scale,
        body_rate_range=args.reference_body_rate_range,
        c1_weight=args.c1_weight,
        c2_weight=args.c2_weight,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    _main()
