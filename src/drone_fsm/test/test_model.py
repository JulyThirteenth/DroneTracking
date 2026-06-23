"""Tests for the pure drone FSM model."""

import pytest

from drone_fsm.model import (
    EVENT_ABORT,
    EVENT_EXECUTE,
    EVENT_LAND,
    EVENT_PREPARE,
    EVENT_RETURN,
    EVENT_TAKEOFF,
    FiniteStateMachine,
    STATE_HOVER,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_READY,
    STATE_RETURN_HOVER,
    STATE_TRACKING,
    allowed_events,
    parse_command,
)


@pytest.mark.parametrize(
    ("command", "event"),
    [
        ("prepare", EVENT_PREPARE),
        ("TAKEOFF", EVENT_TAKEOFF),
        ("执行", EVENT_EXECUTE),
        ("return ignored-argument", EVENT_RETURN),
        ("降落", EVENT_LAND),
        ("abort", EVENT_ABORT),
    ],
)
def test_parse_command_accepts_documented_aliases(command, event) -> None:
    """Normalize English, Chinese, case, and trailing arguments."""
    assert parse_command(command) == event


@pytest.mark.parametrize("command", ["", "   ", "unknown", "take off"])
def test_parse_command_rejects_unknown_input(command) -> None:
    """Reject empty and undocumented command tokens."""
    assert parse_command(command) is None


def test_nominal_mission_and_return_cycle() -> None:
    """Apply the nominal takeoff, tracking, return, and landing path."""
    machine = FiniteStateMachine()
    expected = (
        (EVENT_PREPARE, STATE_READY),
        (EVENT_TAKEOFF, STATE_HOVER_START),
        (EVENT_EXECUTE, STATE_TRACKING),
        (EVENT_RETURN, STATE_RETURN_HOVER),
        (EVENT_EXECUTE, STATE_TRACKING),
        (EVENT_LAND, STATE_PREFLIGHT),
    )

    for event, state in expected:
        assert machine.send(event)
        assert machine.state == state


def test_rejected_event_preserves_state() -> None:
    """Keep the state unchanged when no transition exists."""
    machine = FiniteStateMachine()
    assert not machine.send(EVENT_TAKEOFF)
    assert machine.state == STATE_PREFLIGHT


def test_abort_transitions_to_hover() -> None:
    """Move active non-preflight states into the hover safety state."""
    machine = FiniteStateMachine(STATE_TRACKING)
    assert machine.send(EVENT_ABORT)
    assert machine.state == STATE_HOVER


def test_preflight_ready_and_hover_start_reject_abort() -> None:
    """Ensure ground and takeoff states reject abort."""
    for initial in (STATE_PREFLIGHT, STATE_READY, STATE_HOVER_START):
        machine = FiniteStateMachine(initial)
        assert not machine.send(EVENT_ABORT)
        assert machine.state == initial


def test_invalid_initial_state_is_rejected() -> None:
    """Never construct an inert state machine from a typo."""
    with pytest.raises(ValueError, match="Unknown initial_state"):
        FiniteStateMachine("typo")


def test_allowed_events_are_derived_from_transition_table() -> None:
    """Expose the exact commands accepted by each state."""
    assert allowed_events(STATE_PREFLIGHT) == (EVENT_PREPARE,)
    assert allowed_events(STATE_READY) == (
        EVENT_TAKEOFF,
        EVENT_LAND,
    )
    assert allowed_events(STATE_HOVER_START) == (
        EVENT_EXECUTE,
        EVENT_LAND,
    )
    assert allowed_events("unknown") == ()
