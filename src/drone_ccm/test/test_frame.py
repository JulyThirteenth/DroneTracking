"""Tests for the single PX4/controller frame conversion."""

from __future__ import annotations

import numpy as np
import pytest
from px4_msgs.msg import VehicleOdometry

from drone_ccm.frame import ENU_FROM_NED, VehicleState, odometry_to_enu_flu
from drone_ccm import ros_utils
from drone_ccm.ros_utils import resolve_checkpoint, vehicle_state_from_odometry


class _Parameters:
    """Minimal node stub for checkpoint path resolution."""

    def __init__(self, checkpoint: str) -> None:
        self._checkpoint = checkpoint

    def declare_parameter(self, name: str, default: str) -> object:
        assert name == "checkpoint"
        return type("Parameter", (), {"value": self._checkpoint or default})()


def test_identity_px4_pose() -> None:
    """An identity NED/FRD pose must map to the declared ENU/FLU frame."""
    velocity, rotation = odometry_to_enu_flu(
        np.array((1.0, 0.0, 0.0, 0.0)),
        np.array((1.0, 2.0, 3.0)),
        velocity_is_body_frd=False,
    )
    np.testing.assert_allclose(velocity, np.array((2.0, 1.0, -3.0)))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) > 0.0


def test_body_velocity_uses_same_odometry_attitude() -> None:
    """BODY_FRD velocity must rotate through the same atomic pose sample."""
    half_yaw = 0.25 * np.pi
    quaternion = np.array((np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)))
    body_velocity = np.array((1.0, 0.0, 0.0))
    velocity, _ = odometry_to_enu_flu(
        quaternion,
        body_velocity,
        velocity_is_body_frd=True,
    )
    expected_ned = np.array((0.0, 1.0, 0.0))
    np.testing.assert_allclose(
        velocity,
        ENU_FROM_NED @ expected_ned,
        atol=1.0e-12,
    )


def test_px4_odometry_has_one_validated_conversion_path() -> None:
    message = VehicleOdometry()
    message.pose_frame = VehicleOdometry.POSE_FRAME_NED
    message.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
    message.q = [1.0, 0.0, 0.0, 0.0]
    message.velocity = [1.0, 2.0, 3.0]
    state = vehicle_state_from_odometry(message)
    assert isinstance(state, VehicleState)
    np.testing.assert_allclose(state.velocity, np.array((2.0, 1.0, -3.0)))


def test_px4_odometry_rejects_unsupported_frames() -> None:
    message = VehicleOdometry()
    message.pose_frame = VehicleOdometry.POSE_FRAME_UNKNOWN
    with pytest.raises(ValueError, match="pose frame must be NED"):
        vehicle_state_from_odometry(message)


def test_bare_checkpoint_name_uses_installed_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    package_share = tmp_path / "drone_ccm"
    monkeypatch.setattr(
        ros_utils, "get_package_share_directory", lambda _: str(package_share)
    )
    checkpoint = resolve_checkpoint(_Parameters("neu_ego_ccm_active.pt"))
    assert checkpoint == package_share / "models" / "neu_ego_ccm_active.pt"
