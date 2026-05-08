"""FSM logging and visualization for the drone racing example.

This file intentionally stays dependency-light:
  - Logging uses only stdlib + NumPy (optional ROS clock via `node.get_clock()`).
  - Visualization additionally requires `matplotlib`.

Each run is written into a directory:
  `<base_dir>/run_YYYYmmdd_HHMMSS/`

Files:
  - `ticks.csv`: per-control-loop samples (telemetry + controller outputs).
  - `events.csv`: FSM transitions (state + event + timestamp).
  - `meta.json`: metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_TICKS_CSV = "ticks.csv"
_EVENTS_CSV = "events.csv"
_META_JSON = "meta.json"
_PLOT_PNG = "plot.png"
_DEFAULT_LOG_SUBDIR = "log"


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parent / _DEFAULT_LOG_SUBDIR


def _to_float(value: Any) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def _to_vec3(value: Any) -> np.ndarray:
    """Convert an input into a (3,) float vector; missing values become NaN."""
    if value is None:
        return np.full(3, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if int(arr.shape[0]) >= 3:
        return arr[:3].astype(float, copy=False)
    out = np.full(3, np.nan, dtype=float)
    out[: int(arr.shape[0])] = arr
    return out


def _maybe_node_time_ns(node: Any | None) -> int:
    if node is None:
        return 0
    try:
        return int(node.get_clock().now().nanoseconds)
    except Exception:
        return 0


@dataclass(frozen=True)
class TickData:
    """One log sample for a control loop tick."""

    fsm_state: str
    pos_enu: np.ndarray
    ref_enu: np.ndarray | None
    vel_enu: np.ndarray
    acc_enu: np.ndarray
    yaw_enu: float
    p_cmd: float
    q_cmd: float
    r_cmd: float
    thrust_cmd: float
    acc_est_enu: np.ndarray | None = None
    jerk_cmd_enu: np.ndarray | None = None
    acc_cmd_enu: np.ndarray | None = None
    jerk_cmd_ned: np.ndarray | None = None
    acc_cmd_ned: np.ndarray | None = None
    yaw_cmd_enu: float | None = None
    yaw_rate_cmd_enu: float | None = None
    yaw_cmd_ned: float | None = None
    yaw_rate_cmd_ned: float | None = None


class FsmCsvLogger:
    """Write FSM ticks and events to CSV.

    This logger keeps file handles open for fast per-tick logging.
    """

    _TICK_FIELDS = [
        "t_wall_s",
        "t_ros_ns",
        "fsm_state",
        "px",
        "py",
        "pz",
        "ref_x",
        "ref_y",
        "ref_z",
        "ref_err",
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
        "ax_est",
        "ay_est",
        "az_est",
        "yaw",
        "ax_cmd",
        "ay_cmd",
        "az_cmd",
        "jx_cmd",
        "jy_cmd",
        "jz_cmd",
        "ax_cmd_ned",
        "ay_cmd_ned",
        "az_cmd_ned",
        "jx_cmd_ned",
        "jy_cmd_ned",
        "jz_cmd_ned",
        "yaw_cmd",
        "yaw_rate_cmd",
        "yaw_cmd_ned",
        "yaw_rate_cmd_ned",
        "p_cmd",
        "q_cmd",
        "r_cmd",
        "thrust_cmd",
    ]
    _EVENT_FIELDS = ["t_wall_s", "t_ros_ns", "state", "event"]

    def __init__(
        self,
        *,
        node: Any | None,
        base_dir: str | Path,
        run_name: str | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool = True,
        flush_every: int = 1,
    ):
        self._node = node
        self._enabled = bool(enabled)
        self._flush_every = max(1, int(flush_every))
        self._tick_count = 0

        self.run_dir: Path | None = None
        self.ticks_path: Path | None = None
        self.events_path: Path | None = None

        self._ticks_file: Any | None = None
        self._events_file: Any | None = None
        self._ticks_writer: Any | None = None
        self._events_writer: Any | None = None

        if not self._enabled:
            return

        base_path = Path(base_dir)
        run_id = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = base_path / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.ticks_path = self.run_dir / _TICKS_CSV
        self.events_path = self.run_dir / _EVENTS_CSV

        self._ticks_file = self.ticks_path.open("w", newline="", encoding="utf-8")
        self._events_file = self.events_path.open("w", newline="", encoding="utf-8")
        self._ticks_writer = csv.DictWriter(
            self._ticks_file, fieldnames=self._TICK_FIELDS
        )
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=self._EVENT_FIELDS
        )
        self._ticks_writer.writeheader()
        self._events_writer.writeheader()

        meta_payload = dict(meta or {})
        meta_payload.setdefault("run_name", run_id)
        meta_payload.setdefault("t_wall_start_s", time.time())
        meta_payload.setdefault("base_dir", str(base_path))
        (self.run_dir / _META_JSON).write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        """Flush and close output files."""
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

    def _timestamps(self) -> tuple[float, int]:
        return float(time.time()), _maybe_node_time_ns(self._node)

    def log_event(self, *, state: str, event: str) -> None:
        """Append a state transition entry to `events.csv`."""
        if not self._enabled or self._events_writer is None:
            return
        t_wall_s, t_ros_ns = self._timestamps()
        self._events_writer.writerow(
            {
                "t_wall_s": t_wall_s,
                "t_ros_ns": int(t_ros_ns),
                "state": str(state),
                "event": str(event),
            }
        )
        try:
            self._events_file.flush()
        except Exception:
            pass

    def log_tick(self, tick: TickData) -> None:
        """Append a control-loop sample to `ticks.csv`."""
        if not self._enabled or self._ticks_writer is None:
            return

        t_wall_s, t_ros_ns = self._timestamps()
        pos = _to_vec3(tick.pos_enu)
        ref = _to_vec3(tick.ref_enu)
        vel = _to_vec3(tick.vel_enu)
        acc = _to_vec3(tick.acc_enu)
        ref_err = float(np.linalg.norm(ref - pos)) if np.all(np.isfinite(ref)) else float("nan")

        acc_est = _to_vec3(tick.acc_est_enu)
        acc_cmd = _to_vec3(tick.acc_cmd_enu)
        jerk_cmd = _to_vec3(tick.jerk_cmd_enu)
        acc_cmd_ned = _to_vec3(tick.acc_cmd_ned)
        jerk_cmd_ned = _to_vec3(tick.jerk_cmd_ned)

        row = {
            "t_wall_s": t_wall_s,
            "t_ros_ns": int(t_ros_ns),
            "fsm_state": str(tick.fsm_state),
            "px": float(pos[0]),
            "py": float(pos[1]),
            "pz": float(pos[2]),
            "ref_x": float(ref[0]),
            "ref_y": float(ref[1]),
            "ref_z": float(ref[2]),
            "ref_err": ref_err,
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "vz": float(vel[2]),
            "ax": float(acc[0]),
            "ay": float(acc[1]),
            "az": float(acc[2]),
            "ax_est": float(acc_est[0]),
            "ay_est": float(acc_est[1]),
            "az_est": float(acc_est[2]),
            "yaw": _to_float(tick.yaw_enu),
            "ax_cmd": float(acc_cmd[0]),
            "ay_cmd": float(acc_cmd[1]),
            "az_cmd": float(acc_cmd[2]),
            "jx_cmd": float(jerk_cmd[0]),
            "jy_cmd": float(jerk_cmd[1]),
            "jz_cmd": float(jerk_cmd[2]),
            "ax_cmd_ned": float(acc_cmd_ned[0]),
            "ay_cmd_ned": float(acc_cmd_ned[1]),
            "az_cmd_ned": float(acc_cmd_ned[2]),
            "jx_cmd_ned": float(jerk_cmd_ned[0]),
            "jy_cmd_ned": float(jerk_cmd_ned[1]),
            "jz_cmd_ned": float(jerk_cmd_ned[2]),
            "yaw_cmd": _to_float(tick.yaw_cmd_enu),
            "yaw_rate_cmd": _to_float(tick.yaw_rate_cmd_enu),
            "yaw_cmd_ned": _to_float(tick.yaw_cmd_ned),
            "yaw_rate_cmd_ned": _to_float(tick.yaw_rate_cmd_ned),
            "p_cmd": _to_float(tick.p_cmd),
            "q_cmd": _to_float(tick.q_cmd),
            "r_cmd": _to_float(tick.r_cmd),
            "thrust_cmd": _to_float(tick.thrust_cmd),
        }
        self._ticks_writer.writerow(row)

        self._tick_count += 1
        if (self._tick_count % self._flush_every) == 0:
            try:
                self._ticks_file.flush()
            except Exception:
                pass


def _read_numeric_csv_columns(csv_path: Path) -> dict[str, np.ndarray]:
    """Read a CSV and return numeric columns as float arrays.

    Non-numeric values are stored as NaN.
    """
    columns: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in (row or {}).items():
                if key is None:
                    continue
                columns.setdefault(key, [])
                columns[key].append(_to_float(value))
    return {k: np.asarray(v, dtype=float) for k, v in columns.items()}


def _read_events_csv(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            events.append(
                {
                    "t_wall_s": _to_float(row.get("t_wall_s")),
                    "t_ros_ns": _to_float(row.get("t_ros_ns")),
                    "state": str(row.get("state", "")),
                    "event": str(row.get("event", "")),
                }
            )
    return events


def _tick_time_seconds(ticks: dict[str, np.ndarray]) -> np.ndarray:
    """Return the best available tick timestamp array in seconds."""
    t_ros_ns = ticks.get("t_ros_ns", np.zeros(0, dtype=float))
    if t_ros_ns.size > 0 and np.nanmax(t_ros_ns) > 0:
        return 1e-9 * t_ros_ns
    return ticks.get("t_wall_s", np.zeros_like(t_ros_ns))


def _wrap_pi_array(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles element-wise to [-pi, pi]."""
    arr = np.asarray(angle_rad, dtype=float)
    return np.arctan2(np.sin(arr), np.cos(arr))


def _plot_xyz(
    axes: Any,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    ylabel: str,
    labels: tuple[str, str, str],
    styles: tuple[str, str, str] = ("-", "-", "-"),
    alpha: float = 1.0,
    scatter: bool = True,
    scatter_size: float = 10.0,
) -> None:
    axes.plot(t, x, styles[0], label=labels[0], alpha=alpha)
    axes.plot(t, y, styles[1], label=labels[1], alpha=alpha)
    axes.plot(t, z, styles[2], label=labels[2], alpha=alpha)
    if scatter:
        axes.scatter(t, x, s=float(scatter_size), alpha=0.6)
        axes.scatter(t, y, s=float(scatter_size), alpha=0.6)
        axes.scatter(t, z, s=float(scatter_size), alpha=0.6)
    axes.set_ylabel(ylabel)
    axes.legend(loc="upper right", ncol=3, fontsize=8)


def _add_event_markers(
    axes_list: list[Any], *, events: list[dict[str, Any]], t0: float, use_ros_time: bool
) -> None:
    if not events:
        return

    for event in events:
        t_abs = (
            (1e-9 * float(event["t_ros_ns"]))
            if use_ros_time
            else float(event["t_wall_s"])
        )
        if not np.isfinite(t_abs):
            continue
        x = t_abs - float(t0)
        for ax in axes_list:
            ax.axvline(x, color="k", alpha=0.15, linewidth=1)

        # Annotate state name on the first subplot.
        axes_list[0].text(
            x,
            axes_list[0].get_ylim()[1],
            str(event.get("state", "")),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            alpha=0.7,
        )


def visualize_run(
    *,
    run_dir: Path | None = None,
    ticks_path: Path | None = None,
    events_path: Path | None = None,
    out_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Visualize a logged run and save a PNG.

    Args:
      run_dir: Run directory containing ticks/events CSVs.
      ticks_path: Optional explicit path to `ticks.csv`.
      events_path: Optional explicit path to `events.csv`.
      out_path: Optional output PNG path.
      show: If True, show an interactive window.

    Returns:
      Path to the saved PNG file.
    """
    if run_dir is not None:
        ticks_path = ticks_path or (run_dir / _TICKS_CSV)
        events_path = events_path or (run_dir / _EVENTS_CSV)
        out_path = out_path or (run_dir / _PLOT_PNG)

    if ticks_path is None or not ticks_path.exists():
        raise FileNotFoundError("ticks.csv not found; pass run_dir or ticks_path.")

    ticks = _read_numeric_csv_columns(ticks_path)
    events = (
        _read_events_csv(events_path)
        if (events_path is not None and events_path.exists())
        else []
    )

    t_abs = _tick_time_seconds(ticks)
    if t_abs.size < 1:
        raise ValueError("No tick data.")
    t = t_abs - float(t_abs[0])

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for visualization: {exc}") from exc

    def series(name: str) -> np.ndarray:
        return ticks.get(name, np.full_like(t, np.nan, dtype=float))

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(5, 2)
    ax_err = fig.add_subplot(grid[0, :])
    ax_pos = fig.add_subplot(grid[1, 0], sharex=ax_err)
    ax_vel = fig.add_subplot(grid[1, 1], sharex=ax_err)
    ax_ax = fig.add_subplot(grid[2, 0], sharex=ax_err)
    ax_jerk = fig.add_subplot(grid[2, 1], sharex=ax_err)
    ax_ay = fig.add_subplot(grid[3, 0], sharex=ax_err)
    ax_yaw = fig.add_subplot(grid[3, 1], sharex=ax_err)
    ax_az = fig.add_subplot(grid[4, 0], sharex=ax_err)
    ax_ctbr = fig.add_subplot(grid[4, 1], sharex=ax_err)
    axes_list = [ax_err, ax_pos, ax_vel, ax_ax, ax_jerk, ax_ay, ax_yaw, ax_az, ax_ctbr]

    ref_err_logged = series("ref_err")
    ref_ex = series("px") - series("ref_x")
    ref_ey = series("py") - series("ref_y")
    ref_ez = series("pz") - series("ref_z")
    ref_err_calc = np.sqrt(ref_ex * ref_ex + ref_ey * ref_ey + ref_ez * ref_ez)
    tracking_err = np.where(np.isfinite(ref_err_logged), ref_err_logged, ref_err_calc)
    ax_err.plot(t, tracking_err, label="tracking_error", linewidth=1.5)
    ax_err.plot(t, ref_ex, label="ex", alpha=0.75)
    ax_err.plot(t, ref_ey, label="ey", alpha=0.75)
    ax_err.plot(t, ref_ez, label="ez", alpha=0.75)
    ax_err.axhline(0.0, color="k", alpha=0.15, linewidth=1)
    ax_err.set_ylabel("track err (m)")
    ax_err.legend(loc="upper right", ncol=4, fontsize=8)

    _plot_xyz(
        ax_pos,
        t,
        series("px"),
        series("py"),
        series("pz"),
        ylabel="pos (ENU)",
        labels=("x", "y", "z"),
    )
    _plot_xyz(
        ax_pos,
        t,
        series("ref_x"),
        series("ref_y"),
        series("ref_z"),
        ylabel="pos (ENU)",
        labels=("ref_x", "ref_y", "ref_z"),
        styles=("--", "--", "--"),
        alpha=0.75,
    )
    _plot_xyz(
        ax_vel,
        t,
        series("vx"),
        series("vy"),
        series("vz"),
        ylabel="vel (ENU)",
        labels=("vx", "vy", "vz"),
    )
    ax_ax.plot(t, series("ax"), label="ax")
    ax_ax.plot(t, series("ax_est"), label="ax_est", linestyle=":")
    ax_ax.plot(t, series("ax_cmd"), label="ax_cmd", linestyle="--")
    ax_ax.set_ylabel("ax (ENU)")
    ax_ax.legend(loc="upper right", ncol=3, fontsize=8)

    ax_ay.plot(t, series("ay"), label="ay")
    ax_ay.plot(t, series("ay_est"), label="ay_est", linestyle=":")
    ax_ay.plot(t, series("ay_cmd"), label="ay_cmd", linestyle="--")
    ax_ay.set_ylabel("ay (ENU)")
    ax_ay.legend(loc="upper right", ncol=3, fontsize=8)

    ax_az.plot(t, series("az"), label="az")
    ax_az.plot(t, series("az_est"), label="az_est", linestyle=":")
    ax_az.plot(t, series("az_cmd"), label="az_cmd", linestyle="--")
    ax_az.set_ylabel("az (ENU)")
    ax_az.set_xlabel("t (s)")
    ax_az.legend(loc="upper right", ncol=3, fontsize=8)

    _plot_xyz(
        ax_jerk,
        t,
        series("jx_cmd"),
        series("jy_cmd"),
        series("jz_cmd"),
        ylabel="jerk_cmd (ENU)",
        labels=("jx_cmd", "jy_cmd", "jz_cmd"),
    )

    yaw = _wrap_pi_array(series("yaw"))
    yaw_cmd = _wrap_pi_array(series("yaw_cmd"))
    yaw_err = _wrap_pi_array(yaw_cmd - yaw)
    ax_yaw.plot(t, yaw, label="yaw")
    ax_yaw.plot(t, yaw_cmd, label="yaw_cmd", linestyle="--")
    ax_yaw.plot(t, yaw_err, label="yaw_err", linestyle=":")
    ax_yaw.set_ylabel("yaw (rad)")
    ax_yaw.set_ylim(-float(np.pi), float(np.pi))
    ax_yaw.legend(loc="upper right", ncol=3, fontsize=8)

    ax_ctbr.plot(t, series("p_cmd"), label="p_cmd")
    ax_ctbr.plot(t, series("q_cmd"), label="q_cmd")
    ax_ctbr.plot(t, series("r_cmd"), label="r_cmd")
    ax_ctbr.plot(t, series("thrust_cmd"), label="thrust_cmd")
    ax_ctbr.set_ylabel("CTBR out")
    ax_ctbr.set_xlabel("t (s)")
    ax_ctbr.legend(loc="upper right", ncol=4, fontsize=8)

    use_ros_time = bool(np.nanmax(ticks.get("t_ros_ns", np.array([0.0]))) > 0)
    _add_event_markers(
        axes_list,
        events=events,
        t0=float(t_abs[0]),
        use_ros_time=use_ros_time,
    )

    if out_path is None:
        out_path = ticks_path.parent / _PLOT_PNG
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(str(ticks_path.parent))
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    return out_path


def discover_runs(log_dir: Path) -> list[Path]:
    """List run directories under `log_dir` that contain `ticks.csv`."""
    base = Path(log_dir)
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir() and (p / _TICKS_CSV).exists()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _print_runs(runs: list[Path], *, log_dir: Path) -> None:
    print(f"Log dir: {log_dir}")
    for i, run_dir in enumerate(runs):
        print(f"[{i}] {run_dir.name}")


def _select_run_dir(log_dir: Path, index: int | None) -> Path:
    runs = discover_runs(log_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found under: {log_dir}")

    if index is None:
        _print_runs(runs, log_dir=log_dir)
        raw = input("Select run index: ").strip()
        index = int(raw)

    if index < 0 or index >= len(runs):
        raise ValueError(f"index out of range: {index} (0..{len(runs) - 1})")
    return runs[index]


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for visualizing recorded logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory containing ticks.csv/events.csv.",
    )
    parser.add_argument("--ticks", default="", help="Path to ticks.csv.")
    parser.add_argument("--events", default="", help="Path to events.csv (optional).")
    parser.add_argument(
        "--out",
        default="",
        help="Output image path (default: <run-dir>/plot.png).",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=f"Select a run by index from the default log directory ({_DEFAULT_LOG_SUBDIR}/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=f"List available runs under the default log directory ({_DEFAULT_LOG_SUBDIR}/) and exit.",
    )
    args = parser.parse_args(argv)

    ticks_path = Path(args.ticks).resolve() if args.ticks else None
    events_path = Path(args.events).resolve() if args.events else None
    out_path = Path(args.out).resolve() if args.out else None

    run_dir: Path | None = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None and ticks_path is None:
        log_dir = _default_log_dir()
        if args.list:
            _print_runs(discover_runs(log_dir), log_dir=log_dir)
            return
        run_dir = _select_run_dir(log_dir, args.index)

    saved_path = visualize_run(
        run_dir=run_dir,
        ticks_path=ticks_path,
        events_path=events_path,
        out_path=out_path,
        show=bool(args.show),
    )
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
