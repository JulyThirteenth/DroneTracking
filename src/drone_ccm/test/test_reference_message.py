"""Tests for the atomic CCM reference transport."""

import numpy as np
import pytest
from std_msgs.msg import Float64MultiArray

from drone_ccm.reference_message import (
    CcmReference,
    decode_reference,
    encode_reference,
)


def test_reference_message_round_trip() -> None:
    reference = CcmReference(
        velocity=np.array((1.0, -2.0, 0.5)),
        rotation=np.eye(3),
        control=np.array((9.81, 0.1, -0.2, 0.3)),
        domain_signature=12345,
    )
    decoded = decode_reference(encode_reference(reference))
    np.testing.assert_allclose(decoded.velocity, reference.velocity)
    np.testing.assert_allclose(decoded.rotation, reference.rotation)
    np.testing.assert_allclose(decoded.control, reference.control)
    assert decoded.domain_signature == reference.domain_signature


def test_reference_message_rejects_invalid_size() -> None:
    message = Float64MultiArray()
    message.data = [0.0] * 16
    with pytest.raises(ValueError, match="exactly 17"):
        decode_reference(message)
