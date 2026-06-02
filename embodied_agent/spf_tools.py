"""LangChain tools for SPF-based embodied navigation.

These tools operate against the abstract BaseControl interface rather than
any concrete environment, making them portable across backends.
"""

import math
import base64
from io import BytesIO
from typing import cast

from PIL import Image
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .base_control import BaseControl, NavTarget
from .spf_navigation_prompts import create_spf_prompt, SPFActionSchema

_control: BaseControl | None = None
_sub_llm: BaseChatModel | None = None


def init_env(control: BaseControl, sub_llm: BaseChatModel):
    """Initialize the global environment controller and sub-LLM.

    Must be called before any tool is used.
    """
    global _control, _sub_llm
    _control = control
    _sub_llm = sub_llm


def _encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@tool
def get_current_view():
    """
    Get the robot's current camera frame.
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")

    image = _control.get_current_view()
    b64_str = _encode_image(image)

    content = [
        {
            "type": "text",
            "text": "The following image shows the robot's current perspective."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_str}"
            }
        }
    ]
    return content


@tool
def get_current_position_and_rotation():
    """
    Get the robot's current position and rotation.
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")
    state = _control.get_agent_state()
    pos = state["position"]
    rot = state["rotation"]
    return (
        f"Current state: position(x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}), "
        f"rotation(roll={rot['roll']:.3f}, pitch={rot['pitch']:.3f}, yaw={rot['yaw']:.3f})"
    )

class GetTargetInput(BaseModel):
    instruction: str = Field(description="Description of the target object to locate.")


@tool(args_schema=GetTargetInput)
def get_target_object(instruction: str):
    """
    Locate a target object in the current environment and return its 3D world coordinates.
    This tool AUTOMATICALLY captures the current view; do NOT call get_current_view before this.
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")
    if _sub_llm is None:
        raise Exception("Error: Chat Model for this task is not initialized.")

    img = _control.get_current_view()
    img_url = f"data:image/jpeg;base64,{_encode_image(img)}"

    structured_llm = _sub_llm.with_structured_output(SPFActionSchema)
    prompt_value = create_spf_prompt().invoke({
        "instruction": instruction,
        "image_url": img_url
    })
    response = cast(SPFActionSchema, structured_llm.invoke(prompt_value))

    if not response.success:
        return "Target not found. Try to rotate perspective, analyze the camera view directly or analyze previous actions."

    spf_geometry = _control.spf_geometry
    norm_y, norm_x = response.point
    pixel_x = (norm_x / 1000.0) * spf_geometry.width
    pixel_y = (norm_y / 1000.0) * spf_geometry.height

    if response.depth <= 2:
        return f"Target is very close, no need to move."

    d_adj = _control.spf_geometry.calculate_adjusted_depth(vlm_depth=response.depth, s=6)
    sx, sy, sz = _control.spf_geometry.reverse_project_point(pixel_x, pixel_y, d_adj)

    state = _control.get_agent_state()
    pos = state["position"]
    rot = state["rotation"]
    yaw_rad = math.radians(rot["yaw"])

    world_x = pos["x"] + (sx * math.cos(yaw_rad) + sy * math.sin(yaw_rad))
    world_y = pos["y"] + (-sx * math.sin(yaw_rad) + sy * math.cos(yaw_rad))
    world_z = pos["z"] + sz  

    return (
        f"Target '{instruction}' found at world coordinates: "
        f"x={world_x:.2f}, y={world_y:.2f}, z={world_z:.2f}. "
        f"You can now use navigate_to_point with these coordinates."
    )


class NavigateInput(BaseModel):
    x: float = Field(description="Target East coordinate (ENU x).")
    y: float = Field(description="Target North coordinate (ENU y).")
    z: float = Field(description="Target Up coordinate (ENU z, altitude).")


@tool(args_schema=NavigateInput)
def navigate_to_point(x: float, y: float, z: float) -> str:
    """
    Move the agent to (x, y, z).
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")

    start_state = _control.get_agent_state()
    start_x = start_state["position"]["x"]
    start_y = start_state["position"]["y"]
    start_z = start_state["position"]["z"]
    start_yaw = start_state["rotation"]["yaw"]

    dx = x - start_x
    dy = y - start_y
    if math.sqrt(dx**2 + dy**2) > 0.1:
        target_yaw = math.degrees(math.atan2(dx, dy)) % 360
    else:
        target_yaw = start_yaw

    target_yaw = round(target_yaw / 30) * 30 % 360

    target: NavTarget = {"x": x, "z": z,"y": y, "yaw": target_yaw}
    result = _control.navigate_to_point(target)

    end_state = _control.get_agent_state()
    end_x = end_state["position"]["x"]
    end_y = end_state["position"]["y"]
    end_z = end_state["position"]["z"]
    end_yaw = end_state["rotation"]["yaw"]

    if result["success"]:
        return (
            f"Navigation successful. Perspective shifted: "
            f"From A(x={start_x:.2f}, y={start_y:.2f}, z={start_z:.2f}, yaw={start_yaw:.2f}°) "
            f"To B(x={end_x:.2f}, y={end_y:.2f}, z={end_z:.2f}, yaw={end_yaw:.2f}°)."
        )
    else:
        return (
            f"Navigation failed. Current position: "
            f"(x={end_x:.2f}, y={end_y:.2f}, z={end_z:.2f}, yaw={end_yaw:.2f}°)."
        )


class RotateInput(BaseModel):
    yaw: float = Field(description="The target yaw angle (in degrees) for the agent.")


@tool(args_schema=RotateInput)
def rotate(yaw: float) -> str:
    """
    Set agent orientation (yaw). Only multiples of 30 degrees are valid and effective.
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")

    target_yaw = round(yaw / 30) * 30 % 360

    start_state = _control.get_agent_state()
    start_yaw = start_state["rotation"]["yaw"]

    result = _control.rotate(target_yaw)

    end_state = _control.get_agent_state()
    end_yaw = end_state["rotation"]["yaw"]

    if result["success"]:
        return (
            f"Rotation successful. Perspective shifted: "
            f"From yaw={start_yaw:.2f}° To yaw={end_yaw:.2f}°."
        )
    else:
        return f"Rotation failed. Current yaw: {end_yaw:.2f}°."


TOOLS_LIST = [
    get_current_view,
    get_target_object,
    navigate_to_point,
    rotate,
    get_current_position_and_rotation,
]
