#!/usr/bin/env python3
"""Interactive terminal app for commanding the drone racing FSM.

This replaces repetitive one-off commands like:
  ros2 topic pub --once /fsm/cmd std_msgs/msg/String "{data: 'prepare'}"

Features:
  - Reads commands continuously from stdin.
  - Subscribes to `/fsm/state` and shows current state.
  - Shows which operations are valid for the current state.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fsm_spec import (
    CMD_TO_EVENT,
    EVENT_ALIASES,
    TRANSITION_SPECS,
)
from fsm_ros import derive_info_topic, latched_qos

from yamls.config import get_cfg
_CFG = get_cfg()

_BUILTIN_COMMANDS: tuple[tuple[str, str], ...] = (
    ("help", "Show available operations for current state."),
    ("state", "Print the latest FSM state."),
    ("quit", "Exit this application."),
)
_EXIT_COMMANDS = {"quit", "exit"}
_HELP_COMMANDS = {"help", "?"}
_STATE_COMMANDS = {"state", "status"}

_events_by_state: dict[str, set[str]] = {}
for _src, _event, _dst in TRANSITION_SPECS:
    _events_by_state.setdefault(str(_src), set()).add(str(_event))
_EVENTS_BY_STATE = {
    state: tuple(sorted(events)) for state, events in _events_by_state.items()
}
del _events_by_state


class FsmTerminalApp(Node):
    """Terminal UI for sending `/fsm/cmd` and observing `/fsm/state`."""

    def __init__(self, *, cmd_topic: str, state_topic: str):
        super().__init__("fsm_terminal_app")
        self._cmd_topic = str(cmd_topic)
        self._state_topic = str(state_topic)
        self._info_topic = derive_info_topic(self._state_topic)

        self._pub_cmd = self.create_publisher(String, self._cmd_topic, 10)
        self.create_subscription(
            String, self._state_topic, self._on_state, latched_qos(depth=1)
        )
        self.create_subscription(
            String, self._info_topic, self._on_info, latched_qos(depth=10)
        )

        self._state_lock = threading.Lock()
        self._fsm_state: str | None = None
        self._state_changed = False
        self._info_lock = threading.Lock()
        self._pending_info: list[str] = []

    def _on_state(self, msg: String) -> None:
        state = (msg.data or "").strip()
        if not state:
            return
        with self._state_lock:
            if state != self._fsm_state:
                self._fsm_state = state
                self._state_changed = True

    def _on_info(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        with self._info_lock:
            self._pending_info.append(text)

    def latest_state(self) -> str | None:
        with self._state_lock:
            return self._fsm_state

    def consume_state_changed(self) -> bool:
        with self._state_lock:
            changed = bool(self._state_changed)
            self._state_changed = False
            return changed

    def consume_info_messages(self) -> tuple[str, ...]:
        with self._info_lock:
            msgs = tuple(self._pending_info)
            self._pending_info.clear()
            return msgs

    def allowed_events(self, state: str) -> tuple[str, ...]:
        return _EVENTS_BY_STATE.get(state, ())

    def allowed_commands(self, state: str) -> tuple[str, ...]:
        return tuple(
            str(aliases[-1])
            for event_name in self.allowed_events(state)
            if (aliases := EVENT_ALIASES.get(event_name))
        )

    def state_banner(self) -> str:
        state = self.latest_state() or "unknown"
        ops = " ".join(self.allowed_commands(state)) or "(no known operations)"
        return f"state={state} ops={ops}"

    def print_help(self) -> None:
        state = self.latest_state() or "unknown"
        self.get_logger().info(self.state_banner())
        self.get_logger().info(f"FSM commands: {self._fsm_help_text(state)}")

        for cmd, desc in _BUILTIN_COMMANDS:
            self.get_logger().info(f"{cmd:<6} - {desc}")

    def maybe_publish_cmd(self, raw: str) -> bool:
        """Publish a valid FSM command. Returns True if published."""
        token, event_name = self._parse_cmd(raw)
        if event_name is None:
            return False

        if not self._is_event_allowed(event_name, token):
            return False

        msg = String()
        msg.data = token
        self._pub_cmd.publish(msg)
        return True

    def _fsm_help_text(self, state: str) -> str:
        pairs = []
        for event_name in self.allowed_events(state):
            aliases = EVENT_ALIASES.get(event_name)
            if aliases:
                pairs.append(f"{aliases[-1]}/{aliases[0]}")
        return " ".join(pairs) or "(none)"

    def _parse_cmd(self, raw: str) -> tuple[str, str | None]:
        key = (raw or "").strip()
        token = key.split()[0].strip().lower() if key else ""
        return token, CMD_TO_EVENT.get(token) or CMD_TO_EVENT.get(key)

    def _is_event_allowed(self, event_name: str, token: str) -> bool:
        state = self.latest_state()
        if state is None or event_name in self.allowed_events(state):
            return True

        self.get_logger().warning(
            f"Command '{token}' not allowed from state '{state}'."
        )
        return False


def _stdin_reader(cmd_queue: queue.Queue[str], stop: threading.Event) -> None:
    """Blocking stdin reader (runs in a daemon thread)."""
    while not stop.is_set():
        line = sys.stdin.readline()
        if not line:
            stop.set()
            return
        cmd_queue.put(line.strip())


def _print_prompt(node: FsmTerminalApp) -> None:
    sys.stdout.write(f"{node.state_banner()}\ncmd> ")
    sys.stdout.flush()


def _redraw_prompt(node: FsmTerminalApp) -> None:
    sys.stdout.write("\r\x1b[2K")
    _print_prompt(node)


def _print_info_messages(node: FsmTerminalApp) -> None:
    messages = node.consume_info_messages()
    for text in messages:
        node.get_logger().info(text)
    if messages:
        _redraw_prompt(node)


def _handle_builtin_cmd(node: FsmTerminalApp, cmd: str, stop: threading.Event) -> bool:
    if cmd in _EXIT_COMMANDS:
        stop.set()
        return True

    if cmd in _HELP_COMMANDS:
        node.print_help()
        _print_prompt(node)
        return True

    if cmd in _STATE_COMMANDS:
        node.get_logger().info(node.state_banner())
        _print_prompt(node)
        return True

    return False


def _handle_user_cmd(node: FsmTerminalApp, cmd: str, stop: threading.Event) -> None:
    if _handle_builtin_cmd(node, cmd, stop):
        return

    if not node.maybe_publish_cmd(cmd):
        node.get_logger().warning(
            f"Unknown command '{cmd}'. Type 'help' to see options."
        )
        _print_prompt(node)


def _drain_user_cmds(
    node: FsmTerminalApp,
    cmd_queue: queue.Queue[str],
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            return
        _handle_user_cmd(node, cmd, stop)


def main() -> None:
    rclpy.init()
    node = FsmTerminalApp(
        cmd_topic=str(_CFG.fsm.cmd_topic),
        state_topic=str(_CFG.fsm.state_topic),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    cmd_queue: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=_stdin_reader, args=(cmd_queue, stop), daemon=True)
    thread.start()

    node.get_logger().info(
        f"Listening. cmd_topic={_CFG.fsm.cmd_topic} state_topic={_CFG.fsm.state_topic}"
    )
    node.get_logger().info(f"info_topic={node._info_topic}")
    node.get_logger().info(f"config file: {_CFG.config_path}")
    node.print_help()
    _print_prompt(node)

    try:
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)

            _print_info_messages(node)
            _drain_user_cmds(node, cmd_queue, stop)

            if node.consume_state_changed():
                _redraw_prompt(node)

            time.sleep(0.01)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop.set()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
