"""Core regression tests for drone_cnt."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from drone_cnt.cfg import controller_config_from_mapping
from drone_cnt.cnt import MpcCtbrController, MpccCtbrController
from drone_cnt.utils import Polyline3D, flatness_to_ctbr, ned_to_enu


def _load_config(name: str):
    path = Path(__file__).parents[1] / "config" / f"{name}.yaml"
    parameters = yaml.safe_load(path.read_text(encoding="utf-8"))
    return controller_config_from_mapping(parameters[f"drone_cnt_{name}"]["ros__parameters"])


def test_frame_and_hover_conversion() -> None:
    """Preserve frame conventions and the hover CTBR operating point."""
    assert np.allclose(ned_to_enu([1.0, 2.0, 3.0]), [2.0, 1.0, -3.0])
    assert np.allclose(
        flatness_to_ctbr([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        [0.0, 0.0, 0.0, 0.55],
    )


def test_controller_config_requires_complete_yaml() -> None:
    """Reject a partial YAML mapping instead of using hidden defaults."""
    with pytest.raises(ValueError, match="Missing controller parameters"):
        controller_config_from_mapping({"horizon": 15})


def test_polyline_geometry_and_validation() -> None:
    """Deduplicate a path and project positions onto its arc length."""
    path = Polyline3D.from_points([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    point, tangent = path.sample_with_tangent(1.5)
    assert np.allclose(point, [1.0, 0.5, 0.0])
    assert np.allclose(tangent, [0.0, 1.0, 0.0])
    assert path.closest_arc_length([1.2, 0.4, 0.0]) == pytest.approx(1.4)

    with pytest.raises(ValueError, match="distinct"):
        Polyline3D.from_points([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="non-finite"):
        Polyline3D.from_points([[0, 0, 0], [np.nan, 0, 0]])


def test_mpc_and_mpcc_commands_are_finite_and_bounded() -> None:
    """Solve representative MPC and MPCC steps through the public API."""
    config = _load_config("mpc")
    common_state = ([0, 0, 1], [0, 0, 0], [0, 0, 0], 0.0)
    trajectory = np.column_stack(
        (
            np.linspace(0.0, 2.0, config.horizon + 1),
            np.zeros(config.horizon + 1),
            np.ones(config.horizon + 1),
        )
    )
    path = np.array([[0, 0, 1], [2, 0, 1], [4, 1, 1]], dtype=float)

    mpc_command = MpcCtbrController(config).step(
        *common_state,
        yaw_cmd_enu=0.0,
        ref_traj_enu=trajectory,
    )
    mpcc_command = MpccCtbrController(_load_config("mpcc")).step(
        *common_state,
        yaw_cmd_enu=0.0,
        path_points_enu=path,
    )

    for command in (mpc_command, mpcc_command):
        assert np.all(np.isfinite(command))
        assert config.thrust_min <= command[3] <= config.thrust_max
        assert all(abs(command[i]) <= config.body_rate_limit for i in range(3))


def test_mpcc_rejects_a_single_point() -> None:
    """Reject an invalid path before invoking OSQP."""
    controller = MpccCtbrController(_load_config("mpcc"))
    with pytest.raises(ValueError, match="at least two"):
        controller.step(
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            0.0,
            yaw_cmd_enu=0.0,
            path_points_enu=[[0, 0, 1]],
        )
