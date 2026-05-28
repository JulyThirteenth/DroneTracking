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
_AXES_FIELDS = ("x", "y", "z")


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


def _flush(handle: Any | None) -> None:
    try:
        if handle is not None:
            handle.flush()
    except Exception:
        pass


def _close(handle: Any | None) -> None:
    try:
        if handle is not None:
            handle.flush()
            handle.close()
    except Exception:
        pass


def _vec_row(prefix: str, vec: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{axis}": float(vec[i]) for i, axis in enumerate(_AXES_FIELDS)}


def _axis_row(stem: str, suffix: str, vec: np.ndarray) -> dict[str, float]:
    return {
        f"{stem}{axis}{suffix}": float(vec[i])
        for i, axis in enumerate(_AXES_FIELDS)
    }


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

        self._open_run(Path(base_dir), run_name, meta)

    def _open_run(
        self,
        base_path: Path,
        run_name: str | None,
        meta: dict[str, Any] | None,
    ) -> None:
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
        self._write_meta(base_path, run_id, meta)

    def _write_meta(
        self,
        base_path: Path,
        run_id: str,
        meta: dict[str, Any] | None,
    ) -> None:
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
            _close(handle)
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
        self._events_writer.writerow({
            "t_wall_s": t_wall_s,
            "t_ros_ns": int(t_ros_ns),
            "state": str(state),
            "event": str(event),
        })
        _flush(self._events_file)

    def log_tick(self, tick: TickData) -> None:
        """Append a control-loop sample to `ticks.csv`."""
        if not self._enabled or self._ticks_writer is None:
            return

        self._ticks_writer.writerow(self._tick_row(tick))
        self._tick_count += 1
        if (self._tick_count % self._flush_every) == 0:
            _flush(self._ticks_file)

    def _tick_row(self, tick: TickData) -> dict[str, Any]:
        t_wall_s, t_ros_ns = self._timestamps()
        pos = _to_vec3(tick.pos_enu)
        ref = _to_vec3(tick.ref_enu)
        vel = _to_vec3(tick.vel_enu)
        acc = _to_vec3(tick.acc_enu)

        acc_est = _to_vec3(tick.acc_est_enu)
        acc_cmd = _to_vec3(tick.acc_cmd_enu)
        jerk_cmd = _to_vec3(tick.jerk_cmd_enu)
        acc_cmd_ned = _to_vec3(tick.acc_cmd_ned)
        jerk_cmd_ned = _to_vec3(tick.jerk_cmd_ned)

        return {
            "t_wall_s": t_wall_s,
            "t_ros_ns": int(t_ros_ns),
            "fsm_state": str(tick.fsm_state),
            **_vec_row("p", pos),
            **_vec_row("ref_", ref),
            "ref_err": self._ref_error(pos, ref),
            **_vec_row("v", vel),
            **_vec_row("a", acc),
            **_axis_row("a", "_est", acc_est),
            "yaw": _to_float(tick.yaw_enu),
            **_axis_row("a", "_cmd", acc_cmd),
            **_axis_row("j", "_cmd", jerk_cmd),
            **_axis_row("a", "_cmd_ned", acc_cmd_ned),
            **_axis_row("j", "_cmd_ned", jerk_cmd_ned),
            "yaw_cmd": _to_float(tick.yaw_cmd_enu),
            "yaw_rate_cmd": _to_float(tick.yaw_rate_cmd_enu),
            "yaw_cmd_ned": _to_float(tick.yaw_cmd_ned),
            "yaw_rate_cmd_ned": _to_float(tick.yaw_rate_cmd_ned),
            "p_cmd": _to_float(tick.p_cmd),
            "q_cmd": _to_float(tick.q_cmd),
            "r_cmd": _to_float(tick.r_cmd),
            "thrust_cmd": _to_float(tick.thrust_cmd),
        }

    @staticmethod
    def _ref_error(pos: np.ndarray, ref: np.ndarray) -> float:
        if not np.all(np.isfinite(ref)):
            return float("nan")
        return float(np.linalg.norm(ref - pos))


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


def _resolve_run_paths(
    run_dir: Path | None,
    ticks_path: Path | None,
    events_path: Path | None,
    out_path: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    if run_dir is None:
        return ticks_path, events_path, out_path
    return (
        ticks_path or (run_dir / _TICKS_CSV),
        events_path or (run_dir / _EVENTS_CSV),
        out_path or (run_dir / _PLOT_PNG),
    )


def _make_plot_axes(fig: Any) -> dict[str, Any]:
    grid = fig.add_gridspec(5, 2)
    axes = {
        "err": fig.add_subplot(grid[0, :]),
    }
    axes["pos"] = fig.add_subplot(grid[1, 0], sharex=axes["err"])
    axes["vel"] = fig.add_subplot(grid[1, 1], sharex=axes["err"])
    axes["ax"] = fig.add_subplot(grid[2, 0], sharex=axes["err"])
    axes["jerk"] = fig.add_subplot(grid[2, 1], sharex=axes["err"])
    axes["ay"] = fig.add_subplot(grid[3, 0], sharex=axes["err"])
    axes["yaw"] = fig.add_subplot(grid[3, 1], sharex=axes["err"])
    axes["az"] = fig.add_subplot(grid[4, 0], sharex=axes["err"])
    axes["ctbr"] = fig.add_subplot(grid[4, 1], sharex=axes["err"])
    return axes


def _plot_tracking_error(ax: Any, t: np.ndarray, series: Any) -> None:
    ref_err = series("ref_err")
    ex = series("px") - series("ref_x")
    ey = series("py") - series("ref_y")
    ez = series("pz") - series("ref_z")
    err_calc = np.sqrt(ex * ex + ey * ey + ez * ez)
    err = np.where(np.isfinite(ref_err), ref_err, err_calc)

    ax.plot(t, err, label="tracking_error", linewidth=1.5)
    ax.plot(t, ex, label="ex", alpha=0.75)
    ax.plot(t, ey, label="ey", alpha=0.75)
    ax.plot(t, ez, label="ez", alpha=0.75)
    ax.axhline(0.0, color="k", alpha=0.15, linewidth=1)
    ax.set_ylabel("track err (m)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)


def _plot_axis_compare(ax: Any, t: np.ndarray, series: Any, axis: str) -> None:
    ax.plot(t, series(axis), label=axis)
    ax.plot(t, series(f"{axis}_est"), label=f"{axis}_est", linestyle=":")
    ax.plot(t, series(f"{axis}_cmd"), label=f"{axis}_cmd", linestyle="--")
    ax.set_ylabel(f"{axis} (ENU)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)


def _plot_yaw(ax: Any, t: np.ndarray, series: Any) -> None:
    yaw = _wrap_pi_array(series("yaw"))
    yaw_cmd = _wrap_pi_array(series("yaw_cmd"))
    ax.plot(t, yaw, label="yaw")
    ax.plot(t, yaw_cmd, label="yaw_cmd", linestyle="--")
    ax.plot(t, _wrap_pi_array(yaw_cmd - yaw), label="yaw_err", linestyle=":")
    ax.set_ylabel("yaw (rad)")
    ax.set_ylim(-float(np.pi), float(np.pi))
    ax.legend(loc="upper right", ncol=3, fontsize=8)


def _plot_ctbr(ax: Any, t: np.ndarray, series: Any) -> None:
    for name in ("p_cmd", "q_cmd", "r_cmd", "thrust_cmd"):
        ax.plot(t, series(name), label=name)
    ax.set_ylabel("CTBR out")
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)


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
    ticks_path, events_path, out_path = _resolve_run_paths(
        run_dir, ticks_path, events_path, out_path
    )

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
    axes = _make_plot_axes(fig)
    axes_list = list(axes.values())

    _plot_tracking_error(axes["err"], t, series)

    _plot_xyz(
        axes["pos"],
        t,
        series("px"),
        series("py"),
        series("pz"),
        ylabel="pos (ENU)",
        labels=("x", "y", "z"),
    )
    _plot_xyz(
        axes["pos"],
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
        axes["vel"],
        t,
        series("vx"),
        series("vy"),
        series("vz"),
        ylabel="vel (ENU)",
        labels=("vx", "vy", "vz"),
    )
    _plot_axis_compare(axes["ax"], t, series, "ax")
    _plot_axis_compare(axes["ay"], t, series, "ay")
    _plot_axis_compare(axes["az"], t, series, "az")
    axes["az"].set_xlabel("t (s)")

    _plot_xyz(
        axes["jerk"],
        t,
        series("jx_cmd"),
        series("jy_cmd"),
        series("jz_cmd"),
        ylabel="jerk_cmd (ENU)",
        labels=("jx_cmd", "jy_cmd", "jz_cmd"),
    )

    _plot_yaw(axes["yaw"], t, series)
    _plot_ctbr(axes["ctbr"], t, series)

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
