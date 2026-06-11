import csv
import time
import json
from typing import Dict, Any
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from .fsm_wrap import FSMLoggerBase


@dataclass(frozen=True)
class TickData:
    """One log sample for a control loop tick."""

    fsm_state: str
    pos_enu: np.ndarray
    vel_enu: np.ndarray
    acc_enu: np.ndarray
    yaw_enu: float

    ref_enu: np.ndarray | None = None

    acc_est_enu: np.ndarray | None = None
    acc_cmd_enu: np.ndarray | None = None
    jerk_cmd_enu: np.ndarray | None = None
    yaw_cmd_enu: float | None = None
    yaw_rate_cmd_enu: float | None = None
    p_cmd: float | None = None
    q_cmd: float | None = None
    r_cmd: float | None = None
    thrust_cmd: float | None = None

    def __iter__(self):
        """Allow `dict(tick_data)` with the same payload as `to_dict()`."""
        return iter(self.to_dict().items())

    def to_dict(self) -> Dict[str, Any]:
        p_cmd = self._optional_float(self.p_cmd)
        q_cmd = self._optional_float(self.q_cmd)
        r_cmd = self._optional_float(self.r_cmd)
        thrust_cmd = self._optional_float(self.thrust_cmd)
        return {
            "state": self.fsm_state,
            "pos_x": float(self.pos_enu[0]),
            "pos_y": float(self.pos_enu[1]),
            "pos_z": float(self.pos_enu[2]),
            "yaw": float(self.yaw_enu),
            "vel_x": float(self.vel_enu[0]),
            "vel_y": float(self.vel_enu[1]),
            "vel_z": float(self.vel_enu[2]),
            "acc_x": float(self.acc_enu[0]),
            "acc_y": float(self.acc_enu[1]),
            "acc_z": float(self.acc_enu[2]),
            "ref_x": float(self.ref_enu[0]) if self.ref_enu is not None else None,
            "ref_y": float(self.ref_enu[1]) if self.ref_enu is not None else None,
            "ref_z": float(self.ref_enu[2]) if self.ref_enu is not None else None,
            "ref_err": (
                float(np.linalg.norm(self.ref_enu - self.pos_enu))
                if self.ref_enu is not None
                else None
            ),
            "acc_x_est": (
                float(self.acc_est_enu[0]) if self.acc_est_enu is not None else None
            ),
            "acc_y_est": (
                float(self.acc_est_enu[1]) if self.acc_est_enu is not None else None
            ),
            "acc_z_est": (
                float(self.acc_est_enu[2]) if self.acc_est_enu is not None else None
            ),
            "acc_x_cmd": (
                float(self.acc_cmd_enu[0]) if self.acc_cmd_enu is not None else None
            ),
            "acc_y_cmd": (
                float(self.acc_cmd_enu[1]) if self.acc_cmd_enu is not None else None
            ),
            "acc_z_cmd": (
                float(self.acc_cmd_enu[2]) if self.acc_cmd_enu is not None else None
            ),
            "jerk_x_cmd": (
                float(self.jerk_cmd_enu[0]) if self.jerk_cmd_enu is not None else None
            ),
            "jerk_y_cmd": (
                float(self.jerk_cmd_enu[1]) if self.jerk_cmd_enu is not None else None
            ),
            "jerk_z_cmd": (
                float(self.jerk_cmd_enu[2]) if self.jerk_cmd_enu is not None else None
            ),
            "yaw_cmd": (
                float(self.yaw_cmd_enu) if self.yaw_cmd_enu is not None else None
            ),
            "yaw_rate_cmd": (
                float(self.yaw_rate_cmd_enu)
                if self.yaw_rate_cmd_enu is not None
                else None
            ),
            "p_cmd": p_cmd,
            "q_cmd": q_cmd,
            "r_cmd": r_cmd,
            "thrust_cmd": thrust_cmd,
        }

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)


class FSMLogger(FSMLoggerBase):

    _TICK_FIELDS = [
        "t_wall_s",
        "t_ros_ns",
        "state",
        "pos_x",
        "pos_y",
        "pos_z",
        "yaw",
        "vel_x",
        "vel_y",
        "vel_z",
        "acc_x",
        "acc_y",
        "acc_z",
        "ref_x",
        "ref_y",
        "ref_z",
        "ref_err",
        "acc_x_est",
        "acc_y_est",
        "acc_z_est",
        "acc_x_cmd",
        "acc_y_cmd",
        "acc_z_cmd",
        "jerk_x_cmd",
        "jerk_y_cmd",
        "jerk_z_cmd",
        "yaw_cmd",
        "yaw_rate_cmd",
        "p_cmd",
        "q_cmd",
        "r_cmd",
        "thrust_cmd",
    ]
    _EVENT_FIELDS = ["t_wall_s", "t_ros_ns", "state", "event"]

    def __init__(
        self,
        log_dir: Path = None,
        node: Any = None,
        meta: Dict[str, Any] = None,
        enable: bool = True,
        flush_interval: int = 1,
        run_name: str | None = None,
    ):
        self.log_dir = log_dir or Path(__file__).resolve().parent / "log"
        self.node = node
        self._enabled = bool(enable)
        self.flush_interval = max(1, flush_interval)
        self.run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
        self._tick_count = 0
        self._run_dir: Path | None = None
        self._ticks_path: Path | None = None
        self._events_path: Path | None = None
        self._ticks_file = None
        self._events_file = None
        self._ticks_writer = None
        self._events_writer = None

        if not self._enabled:
            return

        self._run_dir = self.log_dir / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._ticks_path = self._run_dir / "ticks.csv"
        self._events_path = self._run_dir / "events.csv"
        self._ticks_file = self._ticks_path.open("w", newline="", encoding="utf-8")
        self._events_file = self._events_path.open("w", newline="", encoding="utf-8")

        self._ticks_writer = csv.DictWriter(
            self._ticks_file, fieldnames=self._TICK_FIELDS, extrasaction="ignore"
        )
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=self._EVENT_FIELDS, extrasaction="ignore"
        )
        self._ticks_writer.writeheader()
        self._events_writer.writeheader()
        self._ticks_file.flush()
        self._events_file.flush()

        meta_payload = dict(meta or {})
        meta_payload.setdefault("run_name", self.run_id)
        meta_payload.setdefault("t_wall_start_s", time.time())
        meta_payload.setdefault("base_dir", str(self.log_dir))
        (self._run_dir / "meta.json").write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def timestamp(self) -> tuple[float, int]:
        t_wall_s = time.time()
        if self.node is None:
            return t_wall_s, 0
        try:
            t_ros_ns = int(self.node.get_clock().now().nanoseconds)
        except Exception:
            t_ros_ns = 0
        return t_wall_s, t_ros_ns

    def log_event(self, *, state: str, event: str) -> None:
        if not self.enabled or not self._events_writer:
            return
        t_wall_s, t_ros_ns = self.timestamp()
        self._events_writer.writerow(
            {
                "t_wall_s": float(t_wall_s),
                "t_ros_ns": int(t_ros_ns),
                "state": str(state),
                "event": str(event),
            }
        )
        self._events_file.flush()

    def log_tick(self, tick_data: Any) -> None:
        if not self.enabled or not self._ticks_writer:
            return
        t_wall_s, t_ros_ns = self.timestamp()
        payload = self._tick_payload(tick_data)
        self._ticks_writer.writerow(
            {"t_wall_s": t_wall_s, "t_ros_ns": int(t_ros_ns), **payload}
        )
        self._tick_count += 1
        if self._tick_count % self.flush_interval == 0:
            self._ticks_file.flush()

    @staticmethod
    def _tick_payload(tick_data: Any) -> Dict[str, Any]:
        try:
            return dict(tick_data)
        except Exception as exc:
            raise TypeError("tick_data must be convertible to dict.") from exc

    def close(self) -> None:
        for handle in (self._ticks_file, self._events_file):
            try:
                if handle is not None:
                    handle.flush()
                    handle.close()
            except Exception:
                pass
        self._ticks_file = None
        self._events_file = None
        self._ticks_writer = None
        self._events_writer = None
