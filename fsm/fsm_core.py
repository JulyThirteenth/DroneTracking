"""A small, explicit finite state machine implementation.

This is intentionally minimal:
  - One transition per (state, event) pair.
  - Optional guard and action per transition.
  - Optional on-enter / on-exit hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

StateId = str
EventId = str


@dataclass(frozen=True)
class Event:
    """FSM event."""

    name: EventId
    data: Any = None


Guard = Callable[[Event], bool]
Action = Callable[[Event], None]


@dataclass(frozen=True)
class Transition:
    """A state transition."""

    src: StateId
    event: EventId
    dst: StateId
    guard: Guard | None = None
    action: Action | None = None


class FiniteStateMachine:
    """Finite state machine with a fixed transition table."""

    def __init__(
        self,
        *,
        initial: StateId,
        transitions: Iterable[Transition],
        on_enter: Callable[[StateId, Event], None] | None = None,
        on_exit: Callable[[StateId, Event], None] | None = None,
    ):
        self._state: StateId = str(initial)
        self._on_enter = on_enter
        self._on_exit = on_exit

        table: dict[tuple[StateId, EventId], Transition] = {}
        for transition in transitions:
            key = (str(transition.src), str(transition.event))
            if key in table:
                raise ValueError(f"Duplicate transition for {key}.")
            table[key] = Transition(
                src=str(transition.src),
                event=str(transition.event),
                dst=str(transition.dst),
                guard=transition.guard,
                action=transition.action,
            )
        self._table = table

    @property
    def state(self) -> StateId:
        return self._state

    def send(self, event: Event) -> bool:
        """Process an event and update state if a transition matches."""
        transition = self._table.get((self._state, str(event.name)))
        if transition is None:
            return False

        if transition.guard is not None and not bool(transition.guard(event)):
            return False

        prev = self._state
        if self._on_exit is not None:
            self._on_exit(prev, event)
        if transition.action is not None:
            transition.action(event)

        self._state = transition.dst
        if self._on_enter is not None:
            self._on_enter(self._state, event)
        return True

