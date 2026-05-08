"""Plot FSM logs produced by `fsm.fsm_log`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

TICKS_CSV = "ticks.csv"
EVENTS_CSV = "events.csv"
PLOT_PNG = "plot.png"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "log"


class FsmLogPlotter:
    def __init__(self, ticks_path: Path, events_path: Path | None = None):
        self.ticks_path = Path(ticks_path)
        self.events_path = Path(events_path) if events_path is not None else None
        self.ticks = self._read_numeric_csv(self.ticks_path)
        self.events = self._read_events_csv(self.events_path)
        self.t_abs = self._time_seconds()
        if self.t_abs.size < 1:
            raise ValueError("No tick data.")
        self.t = self.t_abs - float(self.t_abs[0])

    def save(self, out_path: Path, *, show: bool = False) -> Path:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                f"matplotlib is required for visualization: {exc}"
            ) from exc

        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        grid = fig.add_gridspec(5, 2)
        axes = {
            "err": fig.add_subplot(grid[0, :]),
            "pos": fig.add_subplot(grid[1, 0]),
            "vel": fig.add_subplot(grid[1, 1]),
            "acc": fig.add_subplot(grid[2, 0]),
            "acc_cmd": fig.add_subplot(grid[2, 1]),
            "jerk": fig.add_subplot(grid[3, 0]),
            "yaw": fig.add_subplot(grid[3, 1]),
            "ctbr": fig.add_subplot(grid[4, :]),
        }

        self._plot_error(axes["err"])
        self._plot_xyz(
            axes["pos"],
            ("pos_x", "px"),
            ("pos_y", "py"),
            ("pos_z", "pz"),
            ylabel="pos (ENU)",
            labels=("x", "y", "z"),
        )
        self._plot_xyz(
            axes["pos"],
            ("ref_x",),
            ("ref_y",),
            ("ref_z",),
            ylabel="pos (ENU)",
            labels=("ref_x", "ref_y", "ref_z"),
            style="--",
            alpha=0.75,
        )
        self._plot_xyz(
            axes["vel"],
            ("vel_x", "vx"),
            ("vel_y", "vy"),
            ("vel_z", "vz"),
            ylabel="vel (ENU)",
            labels=("vx", "vy", "vz"),
        )
        self._plot_xyz(
            axes["acc"],
            ("acc_x", "ax"),
            ("acc_y", "ay"),
            ("acc_z", "az"),
            ylabel="acc (ENU)",
            labels=("ax", "ay", "az"),
        )
        self._plot_xyz(
            axes["acc_cmd"],
            ("acc_x_cmd", "ax_cmd"),
            ("acc_y_cmd", "ay_cmd"),
            ("acc_z_cmd", "az_cmd"),
            ylabel="acc_cmd",
            labels=("ax_cmd", "ay_cmd", "az_cmd"),
        )
        self._plot_xyz(
            axes["jerk"],
            ("jerk_x_cmd", "jx_cmd"),
            ("jerk_y_cmd", "jy_cmd"),
            ("jerk_z_cmd", "jz_cmd"),
            ylabel="jerk_cmd",
            labels=("jx_cmd", "jy_cmd", "jz_cmd"),
        )
        self._plot_yaw(axes["yaw"])
        self._plot_ctbr(axes["ctbr"])
        self._add_event_markers(list(axes.values()))

        fig.suptitle(str(self.ticks_path.parent))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        if show:
            plt.show()
        return out_path

    def series(self, *names: str) -> np.ndarray:
        for name in names:
            values = self.ticks.get(name)
            if values is not None:
                return values
        return np.full(self._num_ticks(), np.nan, dtype=float)

    def _num_ticks(self) -> int:
        return max((len(values) for values in self.ticks.values()), default=0)

    def _plot_error(self, ax: Any) -> None:
        ax.plot(self.t, self.series("ref_err"), label="tracking_error", linewidth=1.5)
        ax.axhline(0.0, color="k", alpha=0.15, linewidth=1)
        ax.set_ylabel("track err (m)")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_xyz(
        self,
        ax: Any,
        x_names: tuple[str, ...],
        y_names: tuple[str, ...],
        z_names: tuple[str, ...],
        *,
        ylabel: str,
        labels: tuple[str, str, str],
        style: str = "-",
        alpha: float = 1.0,
    ) -> None:
        ax.plot(self.t, self.series(*x_names), style, label=labels[0], alpha=alpha)
        ax.plot(self.t, self.series(*y_names), style, label=labels[1], alpha=alpha)
        ax.plot(self.t, self.series(*z_names), style, label=labels[2], alpha=alpha)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    def _plot_yaw(self, ax: Any) -> None:
        yaw = self._wrap_pi(self.series("yaw"))
        yaw_cmd = self._wrap_pi(self.series("yaw_cmd"))
        ax.plot(self.t, yaw, label="yaw")
        ax.plot(self.t, yaw_cmd, "--", label="yaw_cmd")
        ax.plot(self.t, self._wrap_pi(yaw_cmd - yaw), ":", label="yaw_err")
        ax.set_ylabel("yaw (rad)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    def _plot_ctbr(self, ax: Any) -> None:
        for name in ("p_cmd", "q_cmd", "r_cmd", "thrust_cmd"):
            ax.plot(self.t, self.series(name), label=name)
        ax.set_ylabel("CTBR out")
        ax.set_xlabel("t (s)")
        ax.legend(loc="upper right", ncol=4, fontsize=8)

    def _add_event_markers(self, axes: list[Any]) -> None:
        for event in self.events:
            t_abs = (
                event["t_ros_ns"] * 1e-9 if event["t_ros_ns"] > 0 else event["t_wall_s"]
            )
            if not np.isfinite(t_abs):
                continue
            x = float(t_abs) - float(self.t_abs[0])
            for ax in axes:
                ax.axvline(x, color="k", alpha=0.15, linewidth=1)
            axes[0].text(
                x,
                axes[0].get_ylim()[1],
                str(event.get("state", "")),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                alpha=0.7,
            )

    def _time_seconds(self) -> np.ndarray:
        t_ros_ns = self.series("t_ros_ns")
        if t_ros_ns.size > 0 and np.nanmax(t_ros_ns) > 0:
            return t_ros_ns * 1e-9
        return self.series("t_wall_s")

    @staticmethod
    def _wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
        arr = np.asarray(angle_rad, dtype=float)
        return np.arctan2(np.sin(arr), np.cos(arr))

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    @classmethod
    def _read_numeric_csv(cls, csv_path: Path) -> dict[str, np.ndarray]:
        columns: dict[str, list[float]] = {}
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key, value in (row or {}).items():
                    if key is not None:
                        columns.setdefault(key, []).append(cls._to_float(value))
        return {key: np.asarray(values, dtype=float) for key, values in columns.items()}

    @classmethod
    def _read_events_csv(cls, events_path: Path | None) -> list[dict[str, Any]]:
        if events_path is None or not Path(events_path).exists():
            return []
        events: list[dict[str, Any]] = []
        with Path(events_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row:
                    events.append(
                        {
                            "t_wall_s": cls._to_float(row.get("t_wall_s")),
                            "t_ros_ns": cls._to_float(row.get("t_ros_ns")),
                            "state": str(row.get("state", "")),
                            "event": str(row.get("event", "")),
                        }
                    )
        return events


def visualize_run(
    *,
    run_dir: Path | None = None,
    ticks_path: Path | None = None,
    events_path: Path | None = None,
    out_path: Path | None = None,
    show: bool = False,
) -> Path:
    if run_dir is not None:
        run_path = Path(run_dir)
        if run_path.is_file():
            ticks_path = ticks_path or run_path
            events_path = events_path or run_path.parent / EVENTS_CSV
            out_path = out_path or run_path.parent / PLOT_PNG
        else:
            run_path = select_run_dir(run_path)
            ticks_path = ticks_path or run_path / TICKS_CSV
            events_path = events_path or run_path / EVENTS_CSV
            out_path = out_path or run_path / PLOT_PNG
    if ticks_path is None or not Path(ticks_path).exists():
        raise FileNotFoundError("ticks.csv not found; pass run_dir or ticks_path.")

    ticks_path = Path(ticks_path)
    out_path = Path(out_path) if out_path is not None else ticks_path.parent / PLOT_PNG
    return FsmLogPlotter(ticks_path, events_path).save(out_path, show=show)


def select_run_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run path not found: {run_dir}")
    if (run_dir / TICKS_CSV).exists():
        return run_dir

    runs = sorted(
        (
            path
            for path in run_dir.iterdir()
            if path.is_dir() and (path / TICKS_CSV).exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No {TICKS_CSV} found in {run_dir}.")
    if len(runs) == 1:
        print(f"Using run: {runs[0]}")
        return runs[0]

    print(f"Found {len(runs)} runs in {run_dir}:")
    for idx, path in enumerate(runs, start=1):
        print(f"{idx:>2}. {path.name} ({tick_rows(path)} ticks)")

    while True:
        choice = input(f"Select run [1-{len(runs)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(runs):
            selected = runs[int(choice) - 1]
            if tick_rows(selected) > 0:
                return selected
            print(f"{selected.name} has no tick data.")
            continue
        print("Invalid selection.")


def tick_rows(run_dir: Path) -> int:
    ticks_path = Path(run_dir) / TICKS_CSV
    with ticks_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--ticks", default="")
    parser.add_argument("--events", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    saved_path = visualize_run(
        run_dir=Path(args.run_dir).resolve() if args.run_dir else DEFAULT_LOG_DIR,
        ticks_path=Path(args.ticks).resolve() if args.ticks else None,
        events_path=Path(args.events).resolve() if args.events else None,
        out_path=Path(args.out).resolve() if args.out else None,
        show=bool(args.show),
    )
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
