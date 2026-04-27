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
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fsm_spec import (
    CMD_TO_EVENT,
    EVENT_ALIASES,
    TRANSITION_SPECS,
)
from yamls.config import get_cfg

_CFG = get_cfg()


_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _derive_info_topic(state_topic: str) -> str:
    topic = str(state_topic).strip()
    if topic.endswith("/state"):
        return f"{topic[:-6]}/info"
    return f"{topic}/info"


@dataclass(frozen=True)
class _CommandHelp:
    cmd: str
    description: str


_BUILTIN_COMMANDS: tuple[_CommandHelp, ...] = (
    _CommandHelp("help", "Show available operations for current state."),
    _CommandHelp("state", "Print the latest FSM state."),
    _CommandHelp("quit", "Exit this application."),
)


class FsmTerminalApp(Node):
    """Terminal UI for sending `/fsm/cmd` and observing `/fsm/state`."""

    def __init__(self, *, cmd_topic: str, state_topic: str):
        super().__init__("fsm_terminal_app")
        self._cmd_topic = str(cmd_topic)
        self._state_topic = str(state_topic)
        self._info_topic = _derive_info_topic(self._state_topic)

        self._pub_cmd = self.create_publisher(String, self._cmd_topic, 10)
        self.create_subscription(String, self._state_topic, self._on_state, _STATE_QOS)
        self.create_subscription(String, self._info_topic, self._on_info, 10)

        self._transitions = tuple(TRANSITION_SPECS)
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
        events = sorted({event for src, event, _dst in self._transitions if src == state})
        return tuple(events)

    def allowed_commands(self, state: str) -> tuple[str, ...]:
        cmds: list[str] = []
        for event_name in self.allowed_events(state):
            aliases = EVENT_ALIASES.get(event_name)
            if not aliases:
                continue
            # Prefer the English alias for display.
            cmd_en = str(aliases[-1])
            cmds.append(cmd_en)
        return tuple(cmds)

    def state_banner(self) -> str:
        state = self.latest_state() or "unknown"
        cmds = self.allowed_commands(state)
        if cmds:
            ops = " ".join(cmds)
        else:
            ops = "(no known operations)"
        return f"state={state} ops={ops}"

    def print_help(self) -> None:
        state = self.latest_state() or "unknown"
        events = self.allowed_events(state)
        cmds = self.allowed_commands(state)

        self.get_logger().info(self.state_banner())
        if events:
            pairs: list[str] = []
            for event_name in events:
                aliases = EVENT_ALIASES.get(event_name)
                if not aliases:
                    continue
                cn, en = str(aliases[0]), str(aliases[-1])
                pairs.append(f"{en}/{cn}")
            if pairs:
                self.get_logger().info(f"FSM commands: {' '.join(pairs)}")
            else:
                self.get_logger().info("FSM commands: (none)")
        else:
            self.get_logger().info("FSM commands: (none)")

        for item in _BUILTIN_COMMANDS:
            self.get_logger().info(f"{item.cmd:<6} - {item.description}")

    def maybe_publish_cmd(self, raw: str) -> bool:
        """Publish a valid FSM command. Returns True if published."""
        key = (raw or "").strip()
        if not key:
            return False

        token = key.split()[0].strip().lower()
        event_name = CMD_TO_EVENT.get(token) or CMD_TO_EVENT.get(key)
        if event_name is None:
            return False

        state = self.latest_state()
        if state is not None:
            allowed = set(self.allowed_events(state))
            if event_name not in allowed:
                self.get_logger().warning(
                    f"Command '{token}' not allowed from state '{state}'."
                )
                return False

        msg = String()
        msg.data = token
        self._pub_cmd.publish(msg)
        return True


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
    thread = threading.Thread(
        target=_stdin_reader, args=(cmd_queue, stop), daemon=True
    )
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

            info_messages = node.consume_info_messages()
            for text in info_messages:
                node.get_logger().info(text)
            if info_messages:
                _redraw_prompt(node)

            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd in {"quit", "exit"}:
                    stop.set()
                    break

                if cmd in {"help", "?"}:
                    node.print_help()
                    _print_prompt(node)
                    continue

                if cmd in {"state", "status"}:
                    node.get_logger().info(node.state_banner())
                    _print_prompt(node)
                    continue

                published = node.maybe_publish_cmd(cmd)
                if not published:
                    node.get_logger().warning(
                        f"Unknown command '{cmd}'. Type 'help' to see options."
                    )
                    _print_prompt(node)

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
