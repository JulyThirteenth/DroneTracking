"""Tests for the flight-log file format."""

import csv
import json
from pathlib import Path

import pytest

from drone_log.node import _default_reference_topic
from drone_log.recorder import FlightLogger, TickData


def test_logger_writes_tick_event_reference_and_metadata(tmp_path: Path) -> None:
    logger = FlightLogger(
        tmp_path,
        run_name="run",
        flush_interval=2,
        metadata={"controller_mode": "mpc"},
    )
    logger.log_event(10, "tracking", "state_changed")
    logger.log_reference(11, [(1.0, 2.0, 3.0)])
    logger.log_tick(
        12,
        TickData(
            state="tracking",
            position=(1.0, 2.0, 3.0),
            reference_position=(2.0, 2.0, 3.0),
            thrust_command=0.58,
        ),
    )
    logger.close()

    run = tmp_path / "run"
    with (run / "ticks.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["state"] == "tracking"
    assert float(row["ref_err"]) == 1.0
    assert float(row["thrust_cmd"]) == 0.58
    assert json.loads((run / "references.jsonl").read_text())["points"] == [[1.0, 2.0, 3.0]]
    assert json.loads((run / "meta.json").read_text())["controller_mode"] == "mpc"


def test_tick_without_reference_leaves_error_empty(tmp_path: Path) -> None:
    logger = FlightLogger(tmp_path, run_name="run", metadata={})
    logger.log_tick(1, TickData(state="ready"))
    logger.close()
    with (tmp_path / "run" / "ticks.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["ref_err"] == ""


def test_reference_topic_matches_controller_mode() -> None:
    assert _default_reference_topic("mpc") == "/tracking/ref_traj_path"
    assert _default_reference_topic("mpcc") == "/tracking/path"
    with pytest.raises(ValueError):
        _default_reference_topic("ccm")
