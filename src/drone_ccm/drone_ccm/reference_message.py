"""Atomic encoding of velocity, attitude, and CTBR reference values."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
from std_msgs.msg import Float64MultiArray

from drone_ccm.runtime import CcmDomain

REFERENCE_SIZE = 17


@dataclasses.dataclass(frozen=True)
class CcmReference:
    """Atomic controller reference in ENU/FLU coordinates."""

    velocity: np.ndarray
    rotation: np.ndarray
    control: np.ndarray
    domain_signature: int


def domain_signature(domain: CcmDomain) -> int:
    """Returns an exactly float64-representable signature of one domain."""
    serialized = json.dumps(
        dataclasses.asdict(domain),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 53) - 1)


def encode_reference(reference: CcmReference) -> Float64MultiArray:
    """Encodes one coherent ENU/FLU reference message."""
    data = np.concatenate(
        (
            np.asarray(reference.velocity, dtype=float).reshape(3),
            np.asarray(reference.rotation, dtype=float).reshape(9),
            np.asarray(reference.control, dtype=float).reshape(4),
            np.asarray((reference.domain_signature,), dtype=float),
        )
    )
    if not np.all(np.isfinite(data)):
        raise ValueError("Reference contains a non-finite value")
    message = Float64MultiArray()
    message.data = data.tolist()
    return message


def decode_reference(message: Float64MultiArray) -> CcmReference:
    """Decodes and validates one coherent ENU/FLU reference message."""
    data = np.asarray(message.data, dtype=float)
    if data.shape != (REFERENCE_SIZE,) or not np.all(np.isfinite(data)):
        raise ValueError("Reference must contain exactly 17 finite values")
    rotation = data[3:12].reshape(3, 3).copy()
    orthogonality_error = np.linalg.norm(rotation.T @ rotation - np.eye(3))
    if orthogonality_error > 1.0e-6 or np.linalg.det(rotation) < 0.0:
        raise ValueError("Reference attitude is not a valid SO(3) matrix")
    signature_value = data[16]
    signature = int(signature_value)
    if signature < 0 or float(signature) != signature_value:
        raise ValueError("Reference domain signature is invalid")
    return CcmReference(
        velocity=data[0:3].copy(),
        rotation=rotation,
        control=data[12:16].copy(),
        domain_signature=signature,
    )
