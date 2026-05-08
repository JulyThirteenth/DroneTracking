from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


YAMLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = "config_0.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config_path() -> Path:
    raw = (
        os.environ.get("DRONE_RACING_CONFIG", "")
        or os.environ.get("DRONE_RACING_TOPICS_CONFIG", "")
        or DEFAULT_CONFIG_FILE
    ).strip()
    if not raw:
        raw = DEFAULT_CONFIG_FILE

    path = Path(raw)
    if not path.is_absolute():
        path = YAMLS_DIR / path
    return path


def load_config() -> tuple[dict[str, Any], Path]:
    path = resolve_config_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data, path


@dataclass(frozen=True)
class RuntimeConfig:
    controller: str
    solver: str


@dataclass(frozen=True)
class FsmConfig:
    cmd_topic: str
    state_topic: str
    path_topic: str
    ref_path_topic: str
    log_dir: str
    log_enabled: bool
    log_flush_every: int
    takeoff_velocity: float
    auto_land: bool
    auto_land_distance: float
    auto_land_velocity: float
    auto_land_hold_cycles: int


@dataclass(frozen=True)
class Plan2TrackConfig:
    path_topic: str
    ref_path_topic: str
    out_path_topic: str
    vehicle_pose_topic: str
    yaw_cmd_topic: str
    fixed_yaw: bool
    init_yaw: float
    path_file: str
    origin_mode: str
    frame_id: str
    loop: bool


@dataclass(frozen=True)
class TrackingRosConfig:
    offboard_control_mode_topic: str
    vehicle_rates_setpoint_topic: str
    vehicle_command_topic: str
    vehicle_local_position_topic: str
    target_system: int
    pub_offboard: bool


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    runtime: RuntimeConfig
    fsm: FsmConfig
    plan2track: Plan2TrackConfig
    tracking_ros: TrackingRosConfig
    tracking: dict[str, Any]


_RUNTIME_DEFAULTS = {
    "controller": "mpcc",
    "solver": "osqp",
}
_FSM_DEFAULTS = {
    "cmd_topic": "/fsm/cmd",
    "state_topic": "/fsm/state",
    "path_topic": "/tracking/path",
    "ref_path_topic": "/tracking/ref_traj_path",
    "log_dir": "",
    "log_enabled": True,
    "log_flush_every": 1,
    "takeoff_velocity": 0.67,
    "auto_land": False,
    "auto_land_distance": 0.1,
    "auto_land_velocity": 0.1,
    "auto_land_hold_cycles": 20,
}
_PLAN2TRACK_DEFAULTS = {
    "path_topic": "/planning/path",
    "ref_path_topic": "/tracking/ref_traj_path",
    "out_path_topic": "/tracking/path",
    "vehicle_pose_topic": "/tracking/vehicle_pose",
    "yaw_cmd_topic": "/planning/yaw_cmd_enu",
    "fixed_yaw": False,
    "init_yaw": 0.0,
    "path_file": "",
    "origin_mode": "first_xy",
    "frame_id": "map",
    "loop": False,
}
_TRACKING_ROS_DEFAULTS = {
    "offboard_control_mode_topic": "/fmu/in/offboard_control_mode",
    "vehicle_rates_setpoint_topic": "/fmu/in/vehicle_rates_setpoint",
    "vehicle_command_topic": "/fmu/in/vehicle_command",
    "vehicle_local_position_topic": "/fmu/out/vehicle_local_position",
    "target_system": 1,
    "pub_offboard": True,
}


def _section(config: dict[str, Any], name: str, defaults: dict[str, Any]) -> dict[str, Any]:
    section_data = config.get(name, {}) or {}
    if not isinstance(section_data, dict):
        section_data = {}
    return _deep_update(defaults, section_data)


@lru_cache(maxsize=1)
def get_cfg() -> ProjectConfig:
    data, path = load_config()

    runtime = RuntimeConfig(**_section(data, "runtime", _RUNTIME_DEFAULTS))
    fsm = FsmConfig(**_section(data, "fsm", _FSM_DEFAULTS))
    plan2track = Plan2TrackConfig(**_section(data, "plan2track", _PLAN2TRACK_DEFAULTS))
    tracking_ros = TrackingRosConfig(**_section(data, "tracking_ros", _TRACKING_ROS_DEFAULTS))
    tracking = _section(data, "tracking", {})

    return ProjectConfig(
        config_path=path,
        runtime=runtime,
        fsm=fsm,
        plan2track=plan2track,
        tracking_ros=tracking_ros,
        tracking=tracking,
    )
