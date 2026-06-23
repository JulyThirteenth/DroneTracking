"""Shared ROS 2 QoS profiles for the drone FSM nodes."""

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def latched_qos(depth: int = 1) -> QoSProfile:
    """Return a reliable transient-local profile for retained state."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def best_effort_qos(depth: int = 1) -> QoSProfile:
    """Return a volatile best-effort profile for PX4 transport topics."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
