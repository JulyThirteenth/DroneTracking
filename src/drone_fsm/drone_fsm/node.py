#!/usr/bin/env python3

from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from drone_fsm.model import (
    DEFAULT_CMD_TOPIC,
    DEFAULT_INFO_TOPIC,
    DEFAULT_STATE_TOPIC,
    DEFAULT_TRANSITION_TOPIC,
    FiniteStateMachine,
    STATE_PREFLIGHT,
    parse_command,
)
from drone_fsm.qos import latched_qos


def string_msg(data: str) -> String:
    msg = String()
    msg.data = str(data)
    return msg


class DroneFsmNode(Node):
    """Pure ROS 2 finite-state-machine node."""

    def __init__(self) -> None:
        super().__init__("drone_fsm")

        initial_state = self._param("initial_state", STATE_PREFLIGHT)
        cmd_topic = self._param("cmd_topic", DEFAULT_CMD_TOPIC)
        state_topic = self._param("state_topic", DEFAULT_STATE_TOPIC)
        transition_topic = self._param(
            "transition_topic",
            DEFAULT_TRANSITION_TOPIC,
        )
        info_topic = self._param("info_topic", DEFAULT_INFO_TOPIC)

        self._state_pub = self.create_publisher(
            String,
            state_topic,
            latched_qos(),
        )
        self._transition_pub = self.create_publisher(
            String,
            transition_topic,
            10,
        )
        self._info_pub = self.create_publisher(
            String,
            info_topic,
            10,
        )

        self.create_subscription(
            String,
            cmd_topic,
            self._on_command,
            10,
        )

        self._fsm = FiniteStateMachine(initial_state)

        self._publish_state()

        self.get_logger().info(f"FSM started: initial_state={self._fsm.state}")
        self.get_logger().info(
            f"input={cmd_topic}, state={state_topic}, " f"transition={transition_topic}"
        )

    def _param(self, name: str, default: str) -> str:
        return str(self.declare_parameter(name, default).value)

    def _publish_state(self) -> None:
        self._state_pub.publish(string_msg(self._fsm.state))

    def _publish_info(self, text: str) -> None:
        self._info_pub.publish(string_msg(text))

    def _publish_transition(
        self,
        *,
        accepted: bool,
        previous_state: str,
        event_name: str,
        current_state: str,
        raw_command: str,
    ) -> None:
        record = {
            "accepted": bool(accepted),
            "from": str(previous_state),
            "event": str(event_name),
            "to": str(current_state),
            "command": str(raw_command),
        }

        self._transition_pub.publish(string_msg(json.dumps(record)))

    def _on_command(self, msg: String) -> None:
        raw_command = (msg.data or "").strip()

        if not raw_command:
            return

        event_name = parse_command(raw_command)

        if event_name is None:
            text = f"Unknown FSM command: {raw_command}"
            self.get_logger().warning(text)
            self._publish_info(text)
            return

        previous_state = self._fsm.state

        accepted = self._fsm.send(event_name)

        current_state = self._fsm.state

        self._publish_transition(
            accepted=accepted,
            previous_state=previous_state,
            event_name=event_name,
            current_state=current_state,
            raw_command=raw_command,
        )

        if not accepted:
            text = "Rejected transition: " f"state={previous_state}, event={event_name}"
            self.get_logger().warning(text)
            self._publish_info(text)
            return

        self._publish_state()

        self.get_logger().info(f"{previous_state} --{event_name}--> {current_state}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DroneFsmNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
