#!/usr/bin/env python3
"""Interactive command-line interface for the drone FSM."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import TextIO

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from drone_fsm.model import (
    DEFAULT_CMD_TOPIC,
    DEFAULT_INFO_TOPIC,
    DEFAULT_STATE_TOPIC,
    EVENT_ALIASES,
    allowed_events,
    parse_command,
)
from drone_fsm.qos import latched_qos

_BUILTIN_COMMANDS = {
    "help": "help",
    "?": "help",
    "state": "state",
    "status": "state",
    "quit": "quit",
    "exit": "quit",
}


class DroneFsmCli(Node):
    """Publish interactive commands and display FSM feedback."""

    def __init__(self) -> None:
        """Initialize ROS interfaces and terminal input."""
        super().__init__("drone_cli")
        self._cmd_topic = self._param("cmd_topic", DEFAULT_CMD_TOPIC)
        self._state_topic = self._param("state_topic", DEFAULT_STATE_TOPIC)
        self._info_topic = self._param("info_topic", DEFAULT_INFO_TOPIC)

        self._fsm_state: str | None = None
        self._state_changed = False
        self._info_messages: list[str] = []
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._input = self._open_input()
        self._output = self._open_output()
        self._prompt_prefix = os.environ.get("DRONE_CLI_PROMPT_PREFIX", "")

        self._command_publisher = self.create_publisher(
            String,
            self._cmd_topic,
            10,
        )
        self.create_subscription(
            String,
            self._state_topic,
            self._on_state,
            latched_qos(),
        )
        self.create_subscription(
            String,
            self._info_topic,
            self._on_info,
            10,
        )

        threading.Thread(target=self._read_stdin, daemon=True).start()
        self.get_logger().info(
            "FSM CLI started: "
            f"cmd={self._cmd_topic}, state={self._state_topic}, "
            f"info={self._info_topic}"
        )

    def _param(self, name: str, default: str) -> str:
        return str(self.declare_parameter(name, default).value)

    @staticmethod
    def _open_input() -> TextIO:
        """Open the launch terminal for interactive input."""
        terminal = os.environ.get("DRONE_CLI_TTY")
        if terminal is None:
            return sys.stdin
        try:
            return open(terminal, encoding="utf-8")
        except OSError:
            return sys.stdin

    @staticmethod
    def _open_output() -> TextIO:
        """Open the caller's terminal when ROS launch captures stdout."""
        terminal = os.environ.get("DRONE_CLI_TTY")
        if terminal is None:
            return sys.stdout
        try:
            return open(terminal, "w", encoding="utf-8")
        except OSError:
            return sys.stdout

    @property
    def stopped(self) -> bool:
        """Return whether terminal processing should stop."""
        return self._stop_event.is_set()

    def close(self) -> None:
        """Request terminal processing shutdown."""
        self._stop_event.set()

    def _on_state(self, message: String) -> None:
        state = message.data.strip()
        if state:
            self._fsm_state = state
            self._state_changed = True

    def _on_info(self, message: String) -> None:
        text = message.data.strip()
        if text:
            self._info_messages.append(text)

    def _read_stdin(self) -> None:
        while not self.stopped:
            try:
                line = self._input.readline()
            except (OSError, ValueError):
                return
            if not line:
                self.close()
                return
            self._command_queue.put(line.strip())

    def _event_aliases(self) -> tuple[tuple[str, str], ...]:
        state = self._fsm_state or ""
        return tuple(
            EVENT_ALIASES[event]
            for event in allowed_events(state)
            if event in EVENT_ALIASES
        )

    def banner(self) -> str:
        """Return the current state and allowed English commands."""
        state = self._fsm_state or "unknown"
        commands = " ".join(aliases[-1] for aliases in self._event_aliases())
        return f"state={state} ops={commands or '(no known operations)'}"

    def print_line(self, text: str, *, redraw: bool = False) -> None:
        """Write one prefixed line directly to the interactive terminal."""
        if redraw:
            self._output.write("\r\x1b[2K")
        self._output.write(f"{self._prompt_prefix}{text}\n")
        self._output.flush()

    def print_prompt(self, *, redraw: bool = False) -> None:
        """Write the command prompt directly to the interactive terminal."""
        if redraw:
            self._output.write("\r\x1b[2K")
        self._output.write(f"{self._prompt_prefix}cmd> ")
        self._output.flush()

    def print_help(self) -> None:
        """Print state-specific FSM commands and terminal commands."""
        commands = " ".join(
            f"{english}/{chinese}" for chinese, english in self._event_aliases()
        )
        self.print_line(self.banner())
        self.print_line(f"FSM commands: {commands or '(none)'}")
        self.print_line("help - Show commands; state - Show state; quit - Exit")

    def publish_fsm_command(self, raw_command: str) -> bool:
        """Validate and publish one FSM command token."""
        raw = str(raw_command or "").strip()
        event = parse_command(raw)
        if event is None:
            return False

        if self._fsm_state and event not in allowed_events(self._fsm_state):
            self.print_line(
                f"Command '{raw}' is not allowed from state " f"'{self._fsm_state}'."
            )
            return False

        token = raw.split(maxsplit=1)[0].casefold()
        self._command_publisher.publish(String(data=token))
        self.print_line(f"Published FSM command: {token}")
        return True

    def handle_command(self, raw_command: str) -> None:
        """Handle one terminal command."""
        command = str(raw_command or "").strip()
        action = _BUILTIN_COMMANDS.get(command.casefold())
        if action == "quit":
            self.close()
            return
        if action == "help":
            self.print_help()
        elif action == "state":
            self.print_line(self.banner())
        elif command and self.publish_fsm_command(command):
            return
        elif command:
            self.print_line(
                f"Unknown or invalid command '{command}'. Type 'help'."
            )
        self.print_prompt()

    def process_terminal(self) -> None:
        """Process queued feedback and commands without blocking ROS."""
        if self._info_messages:
            for message in self._info_messages:
                self.print_line(message, redraw=True)
            self._info_messages.clear()
            self.print_prompt()

        while not self.stopped:
            try:
                self.handle_command(self._command_queue.get_nowait())
            except queue.Empty:
                break

        if self._state_changed:
            self._state_changed = False
            self.print_line(self.banner(), redraw=True)
            self.print_prompt()


def main(args=None) -> None:
    """Run the interactive CLI node."""
    rclpy.init(args=args)
    node = DroneFsmCli()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + 0.5
    while node._fsm_state is None and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    node._state_changed = False
    node.print_help()
    node.print_prompt()

    try:
        while rclpy.ok() and not node.stopped:
            executor.spin_once(timeout_sec=0.1)
            node.process_terminal()
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
