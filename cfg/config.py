from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILE = "cfg_0.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _section_update(
    config: dict[str, Any], name: str, defaults: dict[str, Any]
) -> dict[str, Any]:
    section_data = config.get(name, {}) or {}
    if not isinstance(section_data, dict):
        section_data = {}
    return _deep_update(defaults, section_data)


def load_config() -> tuple[dict[str, Any], Path]:
    raw = (os.environ.get("DRONE_TRACKING_CONFIG", "") or DEFAULT_CONFIG_FILE).strip()
    path = (
        Path(raw) if Path(raw).is_absolute() else Path(__file__).resolve().parent / raw
    )
    if not path.exists() and path.name.startswith("config_"):
        renamed = path.with_name("cfg_" + path.name.removeprefix("config_"))
        if renamed.exists():
            path = renamed
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data, path


@dataclass(frozen=True)
class RuntimeConfig:
    controller: str
    solver: str


@dataclass(frozen=True)
class FsmTopicsConfig:
    cmd: str
    state: str


@dataclass(frozen=True)
class PlanningTopicsConfig:
    path: str
    yaw_cmd_enu: str


@dataclass(frozen=True)
class TrackingTopicsConfig:
    path: str
    ref_traj_path: str
    vehicle_pose: str


@dataclass(frozen=True)
class Px4TopicsConfig:
    offboard_control_mode: str
    vehicle_rates_setpoint: str
    vehicle_command: str
    vehicle_local_position: str


@dataclass(frozen=True)
class PerceptionTopicsConfig:
    scan: str


@dataclass(frozen=True)
class TopicsConfig:
    fsm: FsmTopicsConfig
    planning: PlanningTopicsConfig
    tracking: TrackingTopicsConfig
    px4: Px4TopicsConfig
    perception: PerceptionTopicsConfig


@dataclass(frozen=True)
class VehicleConfig:
    target_system: int
    pub_offboard: bool


@dataclass(frozen=True)
class FsmLogConfig:
    dir: str
    enabled: bool
    flush_every: int


@dataclass(frozen=True)
class FsmTakeoffConfig:
    velocity: float
    height: float


@dataclass(frozen=True)
class FsmAutoLandConfig:
    enabled: bool
    distance: float
    velocity: float
    hold_cycles: int


@dataclass(frozen=True)
class FsmConfig:
    log: FsmLogConfig
    takeoff: FsmTakeoffConfig
    auto_land: FsmAutoLandConfig


@dataclass(frozen=True)
class Plan2TrackPathConfig:
    file: str
    origin_mode: str
    frame_id: str
    loop: bool


@dataclass(frozen=True)
class Plan2TrackYawConfig:
    fixed: bool
    init: float


@dataclass(frozen=True)
class Plan2TrackConfig:
    path: Plan2TrackPathConfig
    yaw: Plan2TrackYawConfig


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    runtime: RuntimeConfig
    topics: TopicsConfig
    vehicle: VehicleConfig
    fsm: FsmConfig
    plan2track: Plan2TrackConfig
    tracking: dict[str, Any]


_RUNTIME_DEFAULTS = {
    "controller": "mpcc",
    "solver": "osqp",
}
_TOPICS_DEFAULTS = {
    "fsm": {
        "cmd": "/fsm/cmd",
        "state": "/fsm/state",
    },
    "planning": {
        "path": "/planning/path",
        "yaw_cmd_enu": "/planning/yaw_cmd_enu",
    },
    "tracking": {
        "path": "/tracking/path",
        "ref_traj_path": "/tracking/ref_traj_path",
        "vehicle_pose": "/tracking/vehicle_pose",
    },
    "px4": {
        "offboard_control_mode": "/fmu/in/offboard_control_mode",
        "vehicle_rates_setpoint": "/fmu/in/vehicle_rates_setpoint",
        "vehicle_command": "/fmu/in/vehicle_command",
        "vehicle_local_position": "/fmu/out/vehicle_local_position",
    },
    "perception": {
        "scan": "/depth2scan/scan",
    },
}
_VEHICLE_DEFAULTS = {
    "target_system": 1,
    "pub_offboard": True,
}
_FSM_DEFAULTS = {
    "log": {
        "dir": "",
        "enabled": True,
        "flush_every": 1,
    },
    "takeoff": {
        "velocity": 0.67,
        "height": 1.0,
    },
    "auto_land": {
        "enabled": False,
        "distance": 0.1,
        "velocity": 0.1,
        "hold_cycles": 20,
    },
}
_PLAN2TRACK_DEFAULTS = {
    "path": {
        "file": "",
        "origin_mode": "first_xy",
        "frame_id": "map",
        "loop": False,
    },
    "yaw": {
        "fixed": False,
        "init": 0.0,
    },
}


def _topics(data: dict[str, Any]) -> TopicsConfig:
    topics = _section_update(data, "topics", _TOPICS_DEFAULTS)
    return TopicsConfig(
        fsm=FsmTopicsConfig(**topics["fsm"]),
        planning=PlanningTopicsConfig(**topics["planning"]),
        tracking=TrackingTopicsConfig(**topics["tracking"]),
        px4=Px4TopicsConfig(**topics["px4"]),
        perception=PerceptionTopicsConfig(**topics["perception"]),
    )


def _fsm(data: dict[str, Any]) -> FsmConfig:
    fsm = _section_update(data, "fsm", _FSM_DEFAULTS)
    return FsmConfig(
        log=FsmLogConfig(**fsm["log"]),
        takeoff=FsmTakeoffConfig(**fsm["takeoff"]),
        auto_land=FsmAutoLandConfig(**fsm["auto_land"]),
    )


def _plan2track(data: dict[str, Any]) -> Plan2TrackConfig:
    plan2track = _section_update(data, "plan2track", _PLAN2TRACK_DEFAULTS)
    return Plan2TrackConfig(
        path=Plan2TrackPathConfig(**plan2track["path"]),
        yaw=Plan2TrackYawConfig(**plan2track["yaw"]),
    )


def _tracking(data: dict[str, Any]) -> dict[str, Any]:
    tracking = deepcopy(data.get("tracking", {}) or {})
    if "control_loop" in tracking:
        tracking["control"] = tracking.pop("control_loop")
    if "mpc" in tracking and isinstance(tracking["mpc"], dict):
        mpc_cost = tracking["mpc"].pop("cost", None)
        if mpc_cost is not None:
            tracking["mpc_cost"] = mpc_cost
    if "mpcc" in tracking and isinstance(tracking["mpcc"], dict):
        mpcc = tracking.pop("mpcc")
        if isinstance(mpcc.get("cost"), dict):
            tracking["mpcc_cost"] = mpcc["cost"]
        if isinstance(mpcc.get("progress"), dict):
            tracking.setdefault("mpcc_cost", {}).update(mpcc["progress"])
    return tracking


@lru_cache(maxsize=1)
def get_cfg() -> ProjectConfig:
    data, path = load_config()

    return ProjectConfig(
        config_path=path,
        runtime=RuntimeConfig(**_section_update(data, "runtime", _RUNTIME_DEFAULTS)),
        topics=_topics(data),
        vehicle=VehicleConfig(**_section_update(data, "vehicle", _VEHICLE_DEFAULTS)),
        fsm=_fsm(data),
        plan2track=_plan2track(data),
        tracking=_tracking(data),
    )
