from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
import sys
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from yamls.config import get_cfg


@dataclass(frozen=True)
class MpcConfig:
    dt: float = 0.1
    horizon: int = 15
    v_ref: float = 1.0


@dataclass(frozen=True)
class ControlLoopConfig:
    dt: float = 0.01


@dataclass(frozen=True)
class YawControlConfig:
    kp: float = 1.0
    yaw_rate_limit: float | None = None


@dataclass(frozen=True)
class CtbrConfig:
    hover_thrust: float = 0.56
    g: float = 9.81
    thrust_min: float = 0.10
    thrust_max: float = 0.90

    def as_kwargs(self) -> dict:
        return {
            "hover_thrust": float(self.hover_thrust),
            "g": float(self.g),
            "thrust_min": float(self.thrust_min),
            "thrust_max": float(self.thrust_max),
        }


@dataclass(frozen=True)
class AccelFusionConfig:
    imu_alpha: float = 0.02


@dataclass(frozen=True)
class ConstraintsConfig:
    u_min: tuple[float, float, float] | None = (-25.0, -25.0, -25.0)
    u_max: tuple[float, float, float] | None = (25.0, 25.0, 25.0)
    du_min: tuple[float, float, float] | None = (-50.0, -50.0, -50.0)
    du_max: tuple[float, float, float] | None = (50.0, 50.0, 50.0)

    v_min: tuple[float, float, float] | None = (-8.0, -8.0, -8.0)
    v_max: tuple[float, float, float] | None = (8.0, 8.0, 8.0)
    a_min: tuple[float, float, float] | None = (-20.0, -20.0, -10.0)
    a_max: tuple[float, float, float] | None = (20.0, 20.0, 20.0)

    v_slack_weight: float = 10000.0
    a_slack_weight: float = 10000.0


@dataclass(frozen=True)
class MpcCostConfig:
    Q_diag: tuple[float, float, float] = (2000.0, 2000.0, 2000.0)
    R_diag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    Rd_diag: tuple[float, float, float] = (0.2, 0.2, 0.2)
    terminal: float = 1.0


@dataclass(frozen=True)
class MpccCostConfig:
    q_contour: float = 500.0
    q_lag: float = 2.0
    q_progress: float = 1.0
    q_terminal_s: float = 10.0
    vs_max: float | None = 1.0
    vs_min: float | None = 0.0


@dataclass(frozen=True)
class HocbfConfig:
    enabled: bool = False
    scan_topic: str = "/depth2scan/scan"
    depth_camera_xyz: tuple[float, float, float] = (0.1, 0.0, 0.02)
    obstacle_min_radius_m: float = 0.8
    max_obstacles: int = 90
    safe_distance: float = 0.9
    lambda_gain: float = 0.8
    slack_weight: float = 1.0e6


@dataclass(frozen=True)
class TrackingConfig:
    default_controller: str = "mpc"
    default_solver: str = "ipopt"

    mpc: MpcConfig = field(default_factory=MpcConfig)
    control: ControlLoopConfig = field(default_factory=ControlLoopConfig)
    yaw: YawControlConfig = field(default_factory=YawControlConfig)
    ctbr: CtbrConfig = field(default_factory=CtbrConfig)
    accel_fusion: AccelFusionConfig = field(default_factory=AccelFusionConfig)

    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    mpc_cost: MpcCostConfig = field(default_factory=MpcCostConfig)
    mpcc_cost: MpccCostConfig = field(default_factory=MpccCostConfig)
    hocbf: HocbfConfig = field(default_factory=HocbfConfig)


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_default_config() -> TrackingConfig:
    base = asdict(TrackingConfig())
    cfg = get_cfg()
    merged = _deep_update(base, cfg.tracking)
    merged["default_controller"] = str(cfg.runtime.controller)
    merged["default_solver"] = str(cfg.runtime.solver)

    return TrackingConfig(
        default_controller=str(merged["default_controller"]),
        default_solver=str(merged["default_solver"]),
        mpc=MpcConfig(**merged["mpc"]),
        control=ControlLoopConfig(**merged["control"]),
        yaw=YawControlConfig(**merged["yaw"]),
        ctbr=CtbrConfig(**merged["ctbr"]),
        accel_fusion=AccelFusionConfig(**merged["accel_fusion"]),
        constraints=ConstraintsConfig(**merged["constraints"]),
        mpc_cost=MpcCostConfig(**merged["mpc_cost"]),
        mpcc_cost=MpccCostConfig(**merged["mpcc_cost"]),
        hocbf=HocbfConfig(**merged["hocbf"]),
    )


DEFAULT_CONFIG = _load_default_config()


@dataclass
class MPCParamsCfg:
    nx: int
    nu: int
    horizon: int
    dyn_func: Callable | None
    dt: float
    u_min: np.ndarray | None
    u_max: np.ndarray | None
    du_min: np.ndarray | None
    du_max: np.ndarray | None
    Q: np.ndarray
    terminal: float
    R: np.ndarray
    Rd: np.ndarray
    track_idx: np.ndarray
    v_max: np.ndarray | None
    v_min: np.ndarray | None
    a_max: np.ndarray | None
    a_min: np.ndarray | None
    v_slack_weight: float
    a_slack_weight: float


@dataclass
class MPCCParamsCfg(MPCParamsCfg):
    q_contour: float
    q_lag: float
    q_progress: float
    q_terminal_s: float
    vs_max: float | None
    vs_min: float | None


def make_mpc_params(cfg: TrackingConfig, *, with_dynamics: bool) -> MPCParamsCfg:
    c = cfg.constraints
    k = cfg.mpc_cost

    dyn = None
    if with_dynamics:
        # Imported lazily so `--solver osqp` does not require casadi.
        from tracking.tracking_opt import Dynamics

        dyn = Dynamics("uav")

    return MPCParamsCfg(
        nx=9,
        nu=3,
        horizon=int(cfg.mpc.horizon),
        dyn_func=dyn,
        dt=float(cfg.mpc.dt),
        u_min=None if c.u_min is None else np.asarray(c.u_min, dtype=float).reshape(3),
        u_max=None if c.u_max is None else np.asarray(c.u_max, dtype=float).reshape(3),
        du_min=(
            None if c.du_min is None else np.asarray(c.du_min, dtype=float).reshape(3)
        ),
        du_max=(
            None if c.du_max is None else np.asarray(c.du_max, dtype=float).reshape(3)
        ),
        Q=np.diag(np.asarray(k.Q_diag, dtype=float).reshape(3)),
        R=np.diag(np.asarray(k.R_diag, dtype=float).reshape(3)),
        terminal=float(k.terminal),
        Rd=np.diag(np.asarray(k.Rd_diag, dtype=float).reshape(3)),
        track_idx=np.array([0, 1, 2], dtype=int),
        v_min=None if c.v_min is None else np.asarray(c.v_min, dtype=float).reshape(3),
        v_max=None if c.v_max is None else np.asarray(c.v_max, dtype=float).reshape(3),
        a_min=None if c.a_min is None else np.asarray(c.a_min, dtype=float).reshape(3),
        a_max=None if c.a_max is None else np.asarray(c.a_max, dtype=float).reshape(3),
        v_slack_weight=float(c.v_slack_weight),
        a_slack_weight=float(c.a_slack_weight),
    )


def make_mpcc_params(cfg: TrackingConfig, *, with_dynamics: bool) -> MPCCParamsCfg:
    base = make_mpc_params(cfg, with_dynamics=with_dynamics)
    mpcc = cfg.mpcc_cost
    return MPCCParamsCfg(
        **base.__dict__,
        q_contour=float(mpcc.q_contour),
        q_lag=float(mpcc.q_lag),
        q_progress=float(mpcc.q_progress),
        q_terminal_s=float(mpcc.q_terminal_s),
        vs_max=None if mpcc.vs_max is None else float(mpcc.vs_max),
        vs_min=None if mpcc.vs_min is None else float(mpcc.vs_min),
    )
