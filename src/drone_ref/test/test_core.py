"""Tests for controller-independent reference-path geometry."""

import numpy as np
import pytest

from drone_ref.core import ReferencePath


def line_path(*, loop: bool = False) -> ReferencePath:
    """Construct a two-metre path on the ENU x axis."""
    path = ReferencePath(loop=loop)
    path.set_path(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
    )
    return path


def test_path_removes_duplicates_and_samples_arc_length() -> None:
    """Remove zero-length segments and interpolate with endpoint clamping."""
    path = ReferencePath()
    path.set_path(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
    )
    assert path.points.shape == (2, 3)
    assert np.isclose(path.length, 2.0)
    assert np.allclose(path.sample(-1.0), [0.0, 0.0, 0.0])
    assert np.allclose(path.sample(0.5), [0.5, 0.0, 0.0])
    assert np.allclose(path.sample(9.0), [2.0, 0.0, 0.0])


def test_progress_is_monotonic_on_non_loop_path() -> None:
    """Prevent noisy localization from moving progress backward."""
    path = line_path()
    first = path.update_progress(np.array([1.5, 0.2, 0.0]))
    second = path.update_progress(np.array([0.5, 0.1, 0.0]))
    assert first is not None
    assert second is not None
    assert np.isclose(first.distance, 1.5)
    assert np.isclose(second.distance, 1.5)


def test_mpc_reference_uses_horizon_and_fallback_hold() -> None:
    """Generate N+1 path samples or repeat the current position."""
    path = line_path()
    progress = path.update_progress(np.array([0.5, 0.0, 0.0]))
    points = path.mpc_reference(
        progress=progress,
        current_position_enu=np.zeros(3),
        horizon=3,
        sample_distance=0.5,
    )
    assert points.shape == (4, 3)
    assert np.allclose(points[:, 0], [0.5, 1.0, 1.5, 2.0])

    empty = ReferencePath()
    hold = empty.mpc_reference(
        progress=None,
        current_position_enu=np.array([1.0, 2.0, 3.0]),
        horizon=2,
        sample_distance=0.5,
    )
    assert np.allclose(hold, [[1.0, 2.0, 3.0]] * 3)


def test_yaw_follows_path_tangent() -> None:
    """Point yaw along the ENU path tangent."""
    path = line_path()
    progress = path.update_progress(np.array([0.5, 0.0, 0.0]))
    yaw = path.yaw_reference(
        progress=progress,
        lookahead_distance=0.2,
        previous_yaw=1.0,
    )
    assert np.isclose(yaw, 0.0)


def test_loop_path_is_closed_and_wraps_sampling() -> None:
    """Close a loop and wrap samples beyond its total length."""
    path = line_path(loop=True)
    assert np.allclose(path.points[0], path.points[-1])
    assert np.allclose(path.sample(path.length + 0.5), path.sample(0.5))


@pytest.mark.parametrize(
    "points",
    [
        np.array([0.0, 1.0, 2.0]),
        np.ones((3, 2)),
        np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]),
        np.zeros((2, 3)),
    ],
)
def test_invalid_paths_are_rejected(points) -> None:
    """Reject wrong shapes, non-finite points, and zero-length paths."""
    with pytest.raises(ValueError):
        ReferencePath().set_path(points)
