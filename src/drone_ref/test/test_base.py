"""Behavior tests for vehicle-dependent and independent timer outputs."""

from types import SimpleNamespace

import numpy as np

from drone_ref.base import DroneRefBase
from drone_ref.mpcc import DroneRefMpcc


def test_timer_calls_no_state_hook_before_returning() -> None:
    """Allow MPCC path publication before vehicle state is available."""
    calls = []
    node = SimpleNamespace(
        _position_enu=None,
        _warned_no_state=True,
        _publish_without_vehicle_state=lambda: calls.append("no_state"),
    )
    DroneRefBase._on_timer(node)
    assert calls == ["no_state"]


def test_timer_publishes_yaw_and_reference_with_vehicle_state() -> None:
    """Preserve mandatory pose/yaw reference behavior after state arrives."""
    progress = object()
    calls = []
    path = SimpleNamespace(update_progress=lambda position: progress)
    node = SimpleNamespace(
        _position_enu=np.array([1.0, 2.0, 3.0]),
        _reference_path=path,
        _publish_yaw=lambda value: calls.append(("yaw", value)),
        _publish_reference=lambda value: calls.append(("reference", value)),
    )
    DroneRefBase._on_timer(node)
    assert calls == [("yaw", progress), ("reference", progress)]


def test_mpcc_no_state_hook_publishes_changed_path() -> None:
    """Route the base no-state hook to MPCC's revision publisher."""
    calls = []
    node = SimpleNamespace(
        _publish_path_if_changed=lambda: calls.append("path")
    )
    DroneRefMpcc._publish_without_vehicle_state(node)
    assert calls == ["path"]
