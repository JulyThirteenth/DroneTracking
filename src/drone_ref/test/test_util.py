"""Tests for frame conversion and waypoint loading."""

import numpy as np
import pytest

from drone_ref.util import (
    as_vec3,
    load_waypoints_ned,
    ned_to_enu,
    quat_from_yaw_enu,
    wrap_pi,
    yaw_ned_to_enu,
)


def test_ned_to_enu_supports_vectors_and_batches() -> None:
    """Apply the documented north/east swap and down/up sign change."""
    assert np.allclose(ned_to_enu([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])
    batch = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert np.allclose(
        ned_to_enu(batch),
        [[2.0, 1.0, -3.0], [5.0, -4.0, 6.0]],
    )


@pytest.mark.parametrize("value", [1.0, [1.0, 2.0], [[1.0, 2.0]]])
def test_ned_to_enu_rejects_invalid_shapes(value) -> None:
    """Reject inputs whose final dimension is not three."""
    with pytest.raises(ValueError, match="dimension 3"):
        ned_to_enu(value)


def test_yaw_and_quaternion_conversions() -> None:
    """Convert cardinal NED headings to ENU yaw and quaternion."""
    assert np.isclose(yaw_ned_to_enu(0.0), np.pi / 2.0)
    assert np.isclose(yaw_ned_to_enu(np.pi / 2.0), 0.0)
    assert np.allclose(
        quat_from_yaw_enu(np.pi / 2.0),
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
    )
    assert -np.pi <= wrap_pi(9.0) <= np.pi


@pytest.mark.parametrize("value", [[1.0, 2.0], [1.0, np.nan, 3.0]])
def test_as_vec3_rejects_bad_vectors(value) -> None:
    """Reject malformed and non-finite vectors."""
    with pytest.raises(ValueError):
        as_vec3(value)


def test_load_waypoints_handles_comments_commas_and_origins(tmp_path) -> None:
    """Parse waypoint text and apply all supported origin modes."""
    path = tmp_path / "path.txt"
    path.write_text(
        "# N E D extra\n10, 20, -2, 9\n12 23 -4 8 # sample\n",
        encoding="utf-8",
    )

    assert np.allclose(
        load_waypoints_ned(path, origin_ned=(1.0, 2.0, 3.0)),
        [[9.0, 18.0, -5.0], [11.0, 21.0, -7.0]],
    )
    assert np.allclose(
        load_waypoints_ned(path, origin_mode="first"),
        [[0.0, 0.0, 0.0], [2.0, 3.0, -2.0]],
    )
    assert np.allclose(
        load_waypoints_ned(path, origin_mode="first_xy"),
        [[0.0, 0.0, -2.0], [2.0, 3.0, -4.0]],
    )


def test_load_waypoints_returns_empty_for_missing_file(tmp_path) -> None:
    """Represent a missing optional path as an empty M-by-3 array."""
    points = load_waypoints_ned(tmp_path / "missing.txt")
    assert points.shape == (0, 3)


def test_load_waypoints_rejects_unknown_origin(tmp_path) -> None:
    """Reject an origin policy typo instead of silently shifting a path."""
    path = tmp_path / "path.txt"
    path.write_text("0 0 0\n1 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported origin_mode"):
        load_waypoints_ned(path, origin_mode="typo")
