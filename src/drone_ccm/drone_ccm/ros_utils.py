"""Shared ROS 2 parameter and QoS helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from px4_msgs.msg import VehicleOdometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from drone_ccm.frame import VehicleState, odometry_to_enu_flu


_ParameterT = TypeVar("_ParameterT")
_DEFAULT_CHECKPOINT = "neu_ccm_practical.pt"


def parameter(node: Node, name: str, default: _ParameterT) -> _ParameterT:
    """Declares a parameter and returns its value with the default's type."""
    return cast(_ParameterT, node.declare_parameter(name, default).value)


def resolve_checkpoint(node: Node) -> Path:
    """Returns a checkpoint from a path or the installed models directory."""
    configured = parameter(node, "checkpoint", "").strip()
    package_share = Path(get_package_share_directory("drone_ccm"))
    if not configured:
        return package_share / "models" / _DEFAULT_CHECKPOINT

    checkpoint = Path(configured).expanduser()
    if checkpoint.is_absolute() or checkpoint.parent != Path(".") or checkpoint.exists():
        return checkpoint
    return package_share / "models" / checkpoint.name


def vehicle_state_from_odometry(message: VehicleOdometry) -> VehicleState:
    """Validates one PX4 odometry sample and converts it to ENU/FLU."""
    if message.pose_frame != VehicleOdometry.POSE_FRAME_NED:
        raise ValueError("VehicleOdometry pose frame must be NED")
    if message.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
        velocity_is_body = False
    elif message.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
        velocity_is_body = True
    else:
        raise ValueError("Unsupported VehicleOdometry velocity frame")
    velocity, rotation = odometry_to_enu_flu(
        np.asarray(message.q, dtype=float),
        np.asarray(message.velocity, dtype=float),
        velocity_is_body_frd=velocity_is_body,
    )
    return VehicleState(velocity=velocity, rotation=rotation)


def sensor_qos() -> QoSProfile:
    """Returns PX4-compatible best-effort sensor QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def latched_qos() -> QoSProfile:
    """Returns reliable transient-local QoS for the FSM state."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def run_node(factory: Callable[[], Node], args: list[str] | None = None) -> None:
    """Runs one ROS node and tolerates repeated launch shutdown signals."""
    rclpy.init(args=args)
    node = factory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
