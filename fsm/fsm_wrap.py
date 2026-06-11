from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from dataclasses import dataclass
import numpy as np
from rclpy.node import Node
from .fsm_ros import Px4Bridge
from tracking.tracking_cnt import TrackerCtbrBase


@dataclass
class VehicleState:
    """Vehicle state in ENU coordinates."""

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    accel_enu: np.ndarray
    yaw_enu: float


class FSMLoggerBase(ABC):
    """Base interface for FSM event/tick logging."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether logging is active."""
        raise NotImplementedError

    @property
    @abstractmethod
    def run_dir(self) -> Path | None:
        """Directory or sink identifier for the current logging run."""
        raise NotImplementedError

    @abstractmethod
    def log_event(self, *, state: str, event: str) -> None:
        """Log an FSM transition or state-entry event."""
        raise NotImplementedError

    @abstractmethod
    def log_tick(self, tick: Any) -> None:
        """Log one control-loop sample."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Flush and release logger resources."""
        raise NotImplementedError


class FSMBehaviorBase(ABC):
    """Base interface for FSM-driven vehicle behavior execution."""

    def __init__(
        self,
        *,
        node: Node,
        logger: FSMLoggerBase,
        tracker: TrackerCtbrBase,
    ):
        self._node = node
        self._logger = logger
        self._tracker = tracker
        self._px4_bridge = Px4Bridge(node)
        # After AUTO.LAND, stop publishing offboard mode and rates setpoints.
        self._disengaged: bool = False
        self._vehicle_state: VehicleState | None = None

    @property
    def logger(self) -> FSMLoggerBase:
        """Logger owned by this behavior."""
        return self._logger

    @abstractmethod
    def on_enter(self, state: str, event_name: str) -> None:
        """Handle FSM state-entry actions."""
        raise NotImplementedError

    @abstractmethod
    def tick(self, fsm_state: str, dt: float, *args, **kwargs) -> None:
        """Run one behavior/control update for the current FSM state."""
        raise NotImplementedError

    def update_vehicle_state(self, state: VehicleState) -> None:
        """Update latest vehicle telemetry/state in ENU coordinates."""
        self._vehicle_state = state

    def arm(self) -> None:
        """Arm the vehicle and prepare for takeoff."""
        self._px4_bridge.send_vehicle_command(400, 1.0, 0.0)

    def offboard(self) -> None:
        """Switch the vehicle to offboard control mode."""
        self._disengaged = False
        self._px4_bridge.send_vehicle_command(176, 1.0, 6.0)

    def land(self) -> None:
        """Command the vehicle to land and stop offboard control."""
        self._disengaged = True
        self._px4_bridge.send_vehicle_command(21, 0.0, 0.0)

    def close(self) -> None:
        """Release resources owned by the behavior implementation."""
        self._logger.close()


class FSMNodeBase(Node):
    def __init__(self, *, name: str):
        super().__init__(node_name=name)
        self._behavior: FSMBehaviorBase | None = None

    def destroy_node(self) -> bool:
        try:
            if self._behavior is not None:
                self._behavior.close()
        except Exception:
            pass
        return super().destroy_node()
