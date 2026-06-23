"""CSV and JSONL flight-log writer."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TICK_FIELDS = (
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
    "yaw_cmd",
    "p_cmd",
    "q_cmd",
    "r_cmd",
    "thrust_cmd",
)
EVENT_FIELDS = ("t_wall_s", "t_ros_ns", "state", "event")


@dataclass(frozen=True)
class TickData:
    """One controller-output sample in ENU/FLU coordinates."""

    state: str
    position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    acceleration: tuple[float, float, float] | None = None
    yaw: float | None = None
    reference_position: tuple[float, float, float] | None = None
    yaw_command: float | None = None
    roll_rate_command: float | None = None
    pitch_rate_command: float | None = None
    yaw_rate_command: float | None = None
    thrust_command: float | None = None

    def row(self) -> dict[str, float | str | None]:
        """Returns the stable CSV representation."""
        row: dict[str, float | str | None] = {"state": self.state}
        row.update(_vector_row("pos", self.position))
        row.update(_vector_row("vel", self.velocity))
        row.update(_vector_row("acc", self.acceleration))
        row.update(_vector_row("ref", self.reference_position))
        row["yaw"] = self.yaw
        row["yaw_cmd"] = self.yaw_command
        row["p_cmd"] = self.roll_rate_command
        row["q_cmd"] = self.pitch_rate_command
        row["r_cmd"] = self.yaw_rate_command
        row["thrust_cmd"] = self.thrust_command
        row["ref_err"] = _distance(self.position, self.reference_position)
        return row


def _vector_row(
    prefix: str, vector: tuple[float, float, float] | None
) -> dict[str, float | None]:
    values = (None, None, None) if vector is None else vector
    return dict(zip((f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"), values))


def _distance(
    first: tuple[float, float, float] | None,
    second: tuple[float, float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    squared_error = sum(
        (left - right) ** 2 for left, right in zip(first, second)
    )
    return squared_error**0.5


class FlightLogger:
    """Writes ticks, events and reference paths for one flight run."""

    def __init__(
        self,
        log_directory: Path,
        *,
        metadata: dict[str, Any],
        run_name: str = "",
        flush_interval: int = 20,
    ) -> None:
        self._tick_count = 0
        self._flush_interval = max(1, flush_interval)
        run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_directory = log_directory / run_id
        if self.run_directory.exists():
            run_id = f"{run_id}_{time.strftime('%H%M%S')}"
            self.run_directory = log_directory / run_id
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._ticks_file = (self.run_directory / "ticks.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._events_file = (self.run_directory / "events.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._references_file = (self.run_directory / "references.jsonl").open(
            "w", encoding="utf-8"
        )
        self._ticks_writer = csv.DictWriter(
            self._ticks_file, fieldnames=TICK_FIELDS
        )
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=EVENT_FIELDS
        )
        self._ticks_writer.writeheader()
        self._events_writer.writeheader()
        payload = dict(metadata)
        payload.update({"run_name": run_id, "t_wall_start_s": time.time()})
        (self.run_directory / "meta.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def log_tick(self, timestamp_ns: int, tick: TickData) -> None:
        """Writes one final-controller-output sample."""
        self._ticks_writer.writerow(
            {"t_wall_s": time.time(), "t_ros_ns": timestamp_ns, **tick.row()}
        )
        self._tick_count += 1
        if self._tick_count % self._flush_interval == 0:
            self._ticks_file.flush()

    def log_event(self, timestamp_ns: int, state: str, event: str) -> None:
        """Writes one state transition event."""
        self._events_writer.writerow(
            {
                "t_wall_s": time.time(),
                "t_ros_ns": timestamp_ns,
                "state": state,
                "event": event,
            }
        )
        self._events_file.flush()

    def log_reference(
        self, timestamp_ns: int, points: list[tuple[float, float, float]]
    ) -> None:
        """Writes one complete reference-path snapshot."""
        self._references_file.write(
            json.dumps({"t_ros_ns": timestamp_ns, "points": points}) + "\n"
        )
        self._references_file.flush()

    def close(self) -> None:
        """Flushes and closes all flight-log files."""
        for handle in (
            self._ticks_file,
            self._events_file,
            self._references_file,
        ):
            handle.flush()
            handle.close()
