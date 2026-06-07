#!/usr/bin/env python3
"""Summarize the latest FSM log for fixed-goal RL validation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value) if value not in ("", "None", None) else math.nan
    except ValueError:
        return math.nan


def _latest_run(log_dir: Path) -> Path:
    runs = [path for path in log_dir.glob("*") if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No log runs under {log_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _tracking_rows(ticks_path: Path) -> list[dict[str, str]]:
    with ticks_path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("state") == "tracking"]


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return fmean(finite) if finite else math.nan


def _max_abs(values: list[float]) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _sat_ratio(rows: list[dict[str, str]], key: str, limit: float) -> float:
    values = [abs(_float(row, key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(value >= limit for value in finite) / len(finite)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("fsm/log"))
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--good-error", type=float, default=0.30)
    parser.add_argument("--good-speed", type=float, default=0.25)
    args = parser.parse_args()

    run_dir = args.run if args.run is not None else _latest_run(args.log_dir)
    ticks_path = run_dir / "ticks.csv"
    rows = _tracking_rows(ticks_path)
    if not rows:
        raise RuntimeError(f"No tracking rows in {ticks_path}")

    window = max(1, min(int(args.window), len(rows)))
    tail = rows[-window:]
    final = rows[-1]

    final_err = _float(final, "ref_err")
    final_speed = math.sqrt(
        sum(_float(final, key) ** 2 for key in ("vel_x", "vel_y", "vel_z"))
    )
    tail_err = _mean([_float(row, "ref_err") for row in tail])
    tail_speed = _mean(
        [
            math.sqrt(sum(_float(row, key) ** 2 for key in ("vel_x", "vel_y", "vel_z")))
            for row in tail
        ]
    )
    z_values = [_float(row, "pos_z") for row in rows]
    p_sat = _sat_ratio(rows, "p_cmd", 0.399)
    q_sat = _sat_ratio(rows, "q_cmd", 0.399)
    r_sat = _sat_ratio(rows, "r_cmd", 0.299)

    print(f"run={run_dir}")
    print(f"tracking_rows={len(rows)} tail_window={window}")
    print(
        "final_pos=(%.3f, %.3f, %.3f) final_ref=(%.3f, %.3f, %.3f)"
        % tuple(_float(final, key) for key in ("pos_x", "pos_y", "pos_z", "ref_x", "ref_y", "ref_z"))
    )
    print(f"final_err={final_err:.4f} final_speed={final_speed:.4f}")
    print(f"tail_err_mean={tail_err:.4f} tail_speed_mean={tail_speed:.4f}")
    print(f"z_min={min(z_values):.4f} z_max={max(z_values):.4f} z_mean={_mean(z_values):.4f}")
    print(f"cmd_sat_ratio p={p_sat:.3f} q={q_sat:.3f} r={r_sat:.3f}")

    ok = (
        math.isfinite(tail_err)
        and math.isfinite(tail_speed)
        and tail_err <= float(args.good_error)
        and tail_speed <= float(args.good_speed)
    )
    print("verdict=%s" % ("PASS" if ok else "CHECK"))


if __name__ == "__main__":
    main()
