from abc import ABC, abstractmethod
from PIL import Image
from typing import TypedDict

from .spf_geometry import SPFGeometry


class Point3D(TypedDict):
    x: float
    y: float
    z: float


class NavTarget(TypedDict):
    x: float # EAST
    y: float # NORTH
    z: float # UP


class Rotation(TypedDict):
    roll: float
    pitch: float
    yaw: float


class NavResult(TypedDict):
    success: bool
    position: Point3D | None
    yaw: float
    reason: str


class AgentState(TypedDict):
    position: Point3D
    rotation: Rotation


class BaseControl(ABC):
    """Abstract controller for embodied agent environments.

    Subclasses must implement all abstract methods to provide the actual
    environment interaction logic.
    """

    spf_geometry: SPFGeometry

    @abstractmethod
    def get_current_view(self) -> Image.Image:
        """Get the current camera frame as a PIL Image."""
        ...

    @abstractmethod
    def navigate_to_point(self, target: NavTarget) -> NavResult:
        """Navigate to a specified 3D coordinate point."""
        ...

    @abstractmethod
    def rotate(self, yaw: float) -> NavResult:
        """Rotate the agent to a target yaw angle (in degrees)."""
        ...

    @abstractmethod
    def get_agent_state(self) -> AgentState:
        """Get the agent's current state (position, rotation)."""
        ...

    @abstractmethod
    def close(self):
        """Release any resources held by the controller."""
        ...
