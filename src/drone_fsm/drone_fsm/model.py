"""States, commands, and transition logic for the drone FSM."""

from __future__ import annotations

DEFAULT_CMD_TOPIC = "/fsm/cmd"
DEFAULT_STATE_TOPIC = "/fsm/state"
DEFAULT_INFO_TOPIC = "/fsm/info"
DEFAULT_TRANSITION_TOPIC = "/fsm/transition"

STATE_PREFLIGHT = "preflight"
STATE_READY = "ready"
STATE_HOVER_START = "hover_start"
STATE_TRACKING = "tracking"
STATE_RETURN_HOVER = "return_hover"
STATE_HOVER = "hover"

STATES = frozenset(
    {
        STATE_PREFLIGHT,
        STATE_READY,
        STATE_HOVER_START,
        STATE_TRACKING,
        STATE_RETURN_HOVER,
        STATE_HOVER,
    }
)

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

CMD_TO_EVENT = {
    alias.casefold(): event
    for event, aliases in EVENT_ALIASES.items()
    for alias in aliases
}

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
    (STATE_TRACKING, EVENT_ABORT, STATE_HOVER),
    (STATE_RETURN_HOVER, EVENT_ABORT, STATE_HOVER),
    (STATE_HOVER, EVENT_ABORT, STATE_HOVER),
)

TRANSITIONS = {
    (source, event): destination for source, event, destination in TRANSITION_SPECS
}

EVENTS_BY_STATE: dict[str, tuple[str, ...]] = {
    state: tuple(
        event for source, event, _destination in TRANSITION_SPECS if source == state
    )
    for state in STATES
}


def parse_command(raw_command: str) -> str | None:
    """Return the event represented by the first command token."""
    command = str(raw_command or "").strip()
    if not command:
        return None
    return CMD_TO_EVENT.get(command.split(maxsplit=1)[0].casefold())


def allowed_events(state: str) -> tuple[str, ...]:
    """Return the events accepted from a state."""
    return EVENTS_BY_STATE.get(str(state), ())


class FiniteStateMachine:
    """Own the current state and apply the fixed drone transition table."""

    def __init__(self, initial: str = STATE_PREFLIGHT) -> None:
        initial = str(initial)
        if initial not in STATES:
            raise ValueError(f"Unknown initial_state: {initial}")
        self._state = initial

    @property
    def state(self) -> str:
        """Return the current state."""
        return self._state

    def send(self, event: str) -> bool:
        """Apply an event, returning whether a transition was accepted."""
        destination = TRANSITIONS.get((self._state, str(event)))
        if destination is None:
            return False
        self._state = destination
        return True
