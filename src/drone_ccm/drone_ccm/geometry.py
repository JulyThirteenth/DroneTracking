"""Differentiable SO(3) operations used by training and deployment."""

from __future__ import annotations

import torch
from torch import Tensor


def hat(vector: Tensor) -> Tensor:
    """Returns the skew matrix whose action is a cross product."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def vee(matrix: Tensor) -> Tensor:
    """Returns the inverse of :func:`hat` for a skew matrix."""
    return torch.stack(
        (matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]),
        dim=-1,
    )


def exp(vector: Tensor) -> Tensor:
    """Computes the SO(3) exponential with a stable small-angle expansion."""
    angle_squared = (vector * vector).sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle_squared.clamp_min(1.0e-16))
    small = angle_squared < 1.0e-8
    coefficient_a = torch.where(
        small,
        1.0 - angle_squared / 6.0 + angle_squared.square() / 120.0,
        torch.sin(angle) / angle,
    )
    coefficient_b = torch.where(
        small,
        0.5 - angle_squared / 24.0 + angle_squared.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle_squared,
    )
    skew = hat(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    identity = identity.expand(vector.shape[:-1] + (3, 3))
    return (
        identity
        + coefficient_a.unsqueeze(-1) * skew
        + coefficient_b.unsqueeze(-1) * (skew @ skew)
    )


def log(rotation: Tensor) -> Tensor:
    """Computes the principal SO(3) logarithm for angles below pi."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew_vector = 0.5 * vee(rotation - rotation.transpose(-1, -2))
    sine = torch.sin(angle)
    scale = torch.where(
        angle.abs() < 1.0e-5,
        1.0 + angle.square() / 6.0,
        angle / sine.clamp_min(1.0e-8),
    )
    return scale.unsqueeze(-1) * skew_vector
