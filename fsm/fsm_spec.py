"""Shared FSM specification (states, events, aliases, transitions)."""

from __future__ import annotations

from fsm_core import Transition

DEFAULT_CMD_TOPIC = "/fsm/cmd"
DEFAULT_STATE_TOPIC = "/fsm/state"

STATE_PREFLIGHT = "preflight"
STATE_READY = "ready"
STATE_HOVER_START = "hover_start"
STATE_TRACKING = "tracking"
STATE_RETURN_HOVER = "return_hover"
STATE_HOVER = "hover"

EVENT_PREPARE = "cmd_prepare"
EVENT_TAKEOFF = "cmd_takeoff"
EVENT_EXECUTE = "cmd_execute"
EVENT_RETURN = "cmd_return"
EVENT_LAND = "cmd_land"
EVENT_ABORT = "cmd_abort"

EVENT_ALIASES: dict[str, tuple[str, str]] = {
    EVENT_PREPARE: ("准备", "prepare"),
    EVENT_TAKEOFF: ("起飞", "takeoff"),
    EVENT_EXECUTE: ("执行", "execute"),
    EVENT_RETURN: ("返航", "return"),
    EVENT_LAND: ("降落", "land"),
    EVENT_ABORT: ("中止", "abort"),
}

CMD_TO_EVENT: dict[str, str] = {}
for _event_name, _aliases in EVENT_ALIASES.items():
    for _alias in _aliases:
        CMD_TO_EVENT[str(_alias)] = str(_event_name)
        CMD_TO_EVENT[str(_alias).lower()] = str(_event_name)

TRANSITION_SPECS: tuple[tuple[str, str, str], ...] = (
    (STATE_PREFLIGHT, EVENT_PREPARE, STATE_READY),
    (STATE_READY, EVENT_TAKEOFF, STATE_HOVER_START),
    (STATE_HOVER_START, EVENT_EXECUTE, STATE_TRACKING),
    (STATE_TRACKING, EVENT_RETURN, STATE_RETURN_HOVER),
    (STATE_RETURN_HOVER, EVENT_EXECUTE, STATE_TRACKING),
    (STATE_HOVER, EVENT_PREPARE, STATE_READY),
    (STATE_HOVER, EVENT_EXECUTE, STATE_TRACKING),
    (STATE_HOVER_START, EVENT_LAND, STATE_PREFLIGHT),
    (STATE_TRACKING, EVENT_LAND, STATE_PREFLIGHT),
    (STATE_RETURN_HOVER, EVENT_LAND, STATE_PREFLIGHT),
    (STATE_HOVER, EVENT_LAND, STATE_PREFLIGHT),
    (STATE_READY, EVENT_LAND, STATE_PREFLIGHT),
    (STATE_PREFLIGHT, EVENT_ABORT, STATE_PREFLIGHT),
    (STATE_READY, EVENT_ABORT, STATE_HOVER),
    (STATE_HOVER_START, EVENT_ABORT, STATE_HOVER),
    (STATE_TRACKING, EVENT_ABORT, STATE_HOVER),
    (STATE_RETURN_HOVER, EVENT_ABORT, STATE_HOVER),
    (STATE_HOVER, EVENT_ABORT, STATE_HOVER),
)

def build_transitions() -> list[Transition]:
    """Return the FSM transition table."""
    return [Transition(src, event, dst) for src, event, dst in TRANSITION_SPECS]
