from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from .fsm_log import FSMLogger
from .fsm_mpc import MPCBehavior, MPCCBehavior
from .fsm_rl_hover import RLHoverBehavior, RLHoverConfig
from tracking.tracking_cnt import PathTrackerCtbr

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "fsm" / "log"


@dataclass(frozen=True)
class AutoLandConfig:
    enabled: bool
    distance: float
    velocity: float
    hold_cycles: int


class AutoLandMonitor:
    def __init__(self, cfg: AutoLandConfig):
        self._cfg = cfg
        self._stable_count = 0

    @classmethod
    def from_fsm_cfg(cls, fsm_cfg: Any) -> "AutoLandMonitor":
        return cls(
            AutoLandConfig(
                enabled=bool(fsm_cfg.auto_land),
                distance=float(fsm_cfg.auto_land_distance),
                velocity=float(fsm_cfg.auto_land_velocity),
                hold_cycles=max(int(fsm_cfg.auto_land_hold_cycles), 1),
            )
        )

    def update(
        self,
        *,
        fsm_state: str,
        tracking_state: str,
        vehicle_state: Any,
        target_enu: np.ndarray | None,
    ) -> bool:
        if not self._cfg.enabled or fsm_state != tracking_state:
            self._stable_count = 0
            return False

        valid, distance_m, speed_mps = self._metrics(vehicle_state, target_enu)
        if (
            not valid
            or distance_m >= self._cfg.distance
            or speed_mps >= self._cfg.velocity
        ):
            self._stable_count = 0
            return False

        self._stable_count += 1
        if self._stable_count < self._cfg.hold_cycles:
            return False

        self._stable_count = 0
        return True

    @staticmethod
    def _metrics(
        vehicle_state: Any, target_enu: np.ndarray | None
    ) -> tuple[bool, float, float]:
        if vehicle_state is None or target_enu is None:
            return False, float("inf"), float("inf")
        pos_enu = np.asarray(vehicle_state.position_enu, dtype=float).reshape(3)
        vel_enu = np.asarray(vehicle_state.velocity_enu, dtype=float).reshape(3)
        target_enu = np.asarray(target_enu, dtype=float).reshape(3)
        return (
            True,
            float(np.linalg.norm(target_enu - pos_enu)),
            float(np.linalg.norm(vel_enu)),
        )


def logger_creater(
    *,
    node: Any,
    controller: str,
    solver: str,
    dt_control_s: float,
    log_dir: str | None,
    log_enabled: bool,
    log_flush_every: int,
    takeoff_velocity: float,
    takeoff_height: float,
    behavior_name: str,
) -> FSMLogger:
    logger = FSMLogger(
        node=node,
        log_dir=(
            Path(log_dir)
            if log_dir is not None and str(log_dir).strip()
            else _DEFAULT_LOG_DIR
        ),
        enable=bool(log_enabled),
        flush_interval=int(log_flush_every),
        meta={
            "controller": str(controller),
            "solver": str(solver),
            "dt_control_s": float(dt_control_s),
            "takeoff_velocity": float(takeoff_velocity),
            "takeoff_height": float(takeoff_height),
            "behavior": str(behavior_name),
        },
    )
    if logger.enabled and logger.run_dir is not None:
        node.get_logger().info(f"Logging to: {logger.run_dir}")
    return logger


def behavior_creater(
    *,
    node: Any,
    cfg: Any,
    controller: str,
    solver: str,
    log_dir: str | None,
    log_enabled: bool,
    log_flush_every: int,
    takeoff_velocity: float,
    takeoff_height: float,
    init_yaw_enu: float,
):
    controller_name = str(controller).lower().strip()
    behavior_cls = (
        RLHoverBehavior
        if controller_name == "rl_hover"
        else MPCCBehavior
        if controller_name == "mpcc"
        else MPCBehavior
    )
    logger = logger_creater(
        node=node,
        controller=controller,
        solver=solver,
        dt_control_s=float(cfg.control.dt),
        log_dir=log_dir,
        log_enabled=log_enabled,
        log_flush_every=log_flush_every,
        takeoff_velocity=takeoff_velocity,
        takeoff_height=takeoff_height,
        behavior_name=f"{behavior_cls.__name__}",
    )
    tracker_controller = "mpc" if controller_name == "rl_hover" else str(controller)
    tracker = PathTrackerCtbr(None, cfg=cfg, controller=tracker_controller, solver=str(solver))
    if behavior_cls is RLHoverBehavior:
        project_cfg = __import__("yamls.config", fromlist=["get_cfg"]).get_cfg()
        checkpoint = str(project_cfg.rl_hover.checkpoint).strip()
        if not checkpoint:
            raise ValueError(
                "runtime.controller=rl_hover requires rl_hover.checkpoint in YAML."
            )
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            candidates = [
                project_cfg.config_path.parent / checkpoint_path,
                project_cfg.config_path.parent.parent / checkpoint_path,
                Path.cwd() / checkpoint_path,
            ]
            checkpoint_path = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[1],
            )
        rl_cfg = RLHoverConfig(
            checkpoint=str(checkpoint_path),
            device=str(project_cfg.rl_hover.device),
            thrust_ratio_min=float(project_cfg.rl_hover.thrust_ratio_min),
            thrust_ratio_max=float(project_cfg.rl_hover.thrust_ratio_max),
            body_rate_limit=tuple(project_cfg.rl_hover.body_rate_limit),
            hover_thrust=float(project_cfg.rl_hover.hover_thrust),
            thrust_min=float(project_cfg.rl_hover.thrust_min),
            thrust_max=float(project_cfg.rl_hover.thrust_max),
            max_position_error=float(project_cfg.rl_hover.max_position_error),
            fallback_to_mpc_tracking=bool(project_cfg.rl_hover.fallback_to_mpc_tracking),
        )
        behavior = behavior_cls(
            node=node,
            logger=logger,
            tracker=tracker,
            cfg=rl_cfg,
            takeoff_height=float(takeoff_height),
            takeoff_velocity=float(takeoff_velocity),
        )
        behavior.update_yaw_cmd_enu(float(init_yaw_enu))
        return behavior

    kwargs = {
        "node": node,
        "logger": logger,
        "tracker": tracker,
        "takeoff_height": float(takeoff_height),
    }
    if behavior_cls is MPCBehavior:
        kwargs["takeoff_velocity"] = float(takeoff_velocity)
    behavior = behavior_cls(**kwargs)
    behavior.update_yaw_cmd_enu(float(init_yaw_enu))
    return behavior


def fsm_creater(*, node_cls: type, cfg: Any) -> None:
    rclpy.init()
    node = node_cls(
        controller=str(cfg.runtime.controller),
        solver=str(cfg.runtime.solver),
        log_dir=(str(cfg.fsm.log_dir) or None),
        log_enabled=bool(cfg.fsm.log_enabled),
        log_flush_every=int(cfg.fsm.log_flush_every),
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    from yamls.config import get_cfg

    from .fsm_node import DroneFSMNode

    fsm_creater(node_cls=DroneFSMNode, cfg=get_cfg())


if __name__ == "__main__":
    main()
