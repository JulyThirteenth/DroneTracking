#!/usr/bin/env python3
"""Interactive terminal interface for commanding the FSM."""

from __future__ import annotations

import queue
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .fsm_ros import derive_info_topic, latched_qos
from .fsm_spec import CMD_TO_EVENT, EVENT_ALIASES, TRANSITION_SPECS
from cfg.config import get_cfg

_CFG = get_cfg()
_BUILTINS = {
    "help": "help",
    "?": "help",
    "state": "state",
    "status": "state",
    "quit": "quit",
    "exit": "quit",
}


def build_events_by_state() -> dict[str, tuple[str, ...]]:
    events_by_state: dict[str, set[str]] = {}
    for src, event, _dst in TRANSITION_SPECS:
        events_by_state.setdefault(str(src), set()).add(str(event))
    return {state: tuple(sorted(events)) for state, events in events_by_state.items()}


_EVENTS_BY_STATE = build_events_by_state()


class FsmTerminalApp(Node):
    def __init__(self, *, cmd_topic: str, state_topic: str):
        super().__init__("fsm_terminal_app")
        self._cmd_topic = str(cmd_topic)
        self._state_topic = str(state_topic)
        self._info_topic = derive_info_topic(self._state_topic)
        self._fsm_state: str | None = None
        self._state_changed = False
        self._info_msgs: list[str] = []
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()

        self._pub_cmd = self.create_publisher(String, self._cmd_topic, 10)
        self.create_subscription(
            String, self._state_topic, self._on_state, latched_qos(depth=1)
        )
        self.create_subscription(
            String, self._info_topic, self._on_info, latched_qos(depth=10)
        )
        threading.Thread(target=self._read_stdin, daemon=True).start()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _on_state(self, msg: String) -> None:
        state = (msg.data or "").strip()
        if state and state != self._fsm_state:
            self._fsm_state = state
            self._state_changed = True

    def _on_info(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if text:
            self._info_msgs.append(text)

    def _read_stdin(self) -> None:
        while not self._stop.is_set():
            line = sys.stdin.readline()
            if not line:
                self._stop.set()
                return
            self._cmd_queue.put(line.strip())

    def banner(self) -> str:
        state = self._fsm_state or "unknown"
        commands = " ".join(self._allowed_commands(state)) or "(no known operations)"
        return f"state={state} ops={commands}"

    def print_prompt(self, *, redraw: bool = False) -> None:
        if redraw:
            sys.stdout.write("\r\x1b[2K")
        sys.stdout.write(f"{self.banner()}\ncmd> ")
        sys.stdout.flush()

    def print_help(self) -> None:
        state = self._fsm_state or "unknown"
        self.get_logger().info(self.banner())
        self.get_logger().info(f"FSM commands: {self._fsm_help_text(state)}")
        self.get_logger().info("help   - Show available operations.")
        self.get_logger().info("state  - Print current state.")
        self.get_logger().info("quit   - Exit this application.")

    def process_terminal(self) -> None:
        for text in self._info_msgs:
            self.get_logger().info(text)
        if self._info_msgs:
            self._info_msgs.clear()
            self.print_prompt(redraw=True)

        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_command(cmd)

        if self._state_changed:
            self._state_changed = False
            self.print_prompt(redraw=True)

    def close(self) -> None:
        self._stop.set()

    def _handle_command(self, raw: str) -> None:
        cmd = (raw or "").strip()
        action = _BUILTINS.get(cmd)
        if action == "quit":
            self._stop.set()
            return
        if action == "help":
            self.print_help()
            self.print_prompt()
            return
        if action == "state":
            self.get_logger().info(self.banner())
            self.print_prompt()
            return
        if not self._publish_fsm_command(cmd):
            self.get_logger().warning(
                f"Unknown command '{cmd}'. Type 'help' to see options."
            )
            self.print_prompt()

    def _publish_fsm_command(self, raw: str) -> bool:
        token = raw.split()[0].strip().lower() if raw else ""
        event_name = CMD_TO_EVENT.get(token) or CMD_TO_EVENT.get(raw)
        if event_name is None:
            return False

        state = self._fsm_state
        if state is not None and event_name not in _EVENTS_BY_STATE.get(state, ()):
            self.get_logger().warning(
                f"Command '{token}' not allowed from state '{state}'."
            )
            return False

        msg = String()
        msg.data = token
        self._pub_cmd.publish(msg)
        return True

    def _allowed_commands(self, state: str) -> tuple[str, ...]:
        return tuple(
            str(aliases[-1])
            for event_name in _EVENTS_BY_STATE.get(state, ())
            if (aliases := EVENT_ALIASES.get(event_name))
        )

    def _fsm_help_text(self, state: str) -> str:
        pairs = []
        for event_name in _EVENTS_BY_STATE.get(state, ()):
            aliases = EVENT_ALIASES.get(event_name)
            if aliases:
                pairs.append(f"{aliases[-1]}/{aliases[0]}")
        return " ".join(pairs) or "(none)"


def main() -> None:
    rclpy.init()
    node = FsmTerminalApp(
        cmd_topic=str(_CFG.topics.fsm.cmd),
        state_topic=str(_CFG.topics.fsm.state),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.get_logger().info(
        f"Listening. cmd_topic={_CFG.topics.fsm.cmd} state_topic={_CFG.topics.fsm.state}"
    )
    node.get_logger().info(f"info_topic={node._info_topic}")
    node.get_logger().info(f"config file: {_CFG.config_path}")
    node.print_help()
    node.print_prompt()

    try:
        while rclpy.ok() and not node.stopped:
            executor.spin_once(timeout_sec=0.1)
            node.process_terminal()
            time.sleep(0.01)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
