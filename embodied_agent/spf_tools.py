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
from .spf_navigation_prompts import create_spf_prompt, create_image_match_prompt, SPFActionSchema

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
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
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

    yaw_rad = math.radians(rot["yaw"] % 360)
    fx = math.cos(yaw_rad)
    fy = math.sin(yaw_rad)
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    d = dirs[round(rot["yaw"] % 360 / 45) % 8]

    return (
        f"Position: x={pos['x']:.3f}(E) y={pos['y']:.3f}(N) z={pos['z']:.3f}(Up). "
        f"Yaw: {rot['yaw']:.1f}° ≈ {d}. "
        f"Forward vector: ({fx:.2f}, {fy:.2f}) — "
        f"right is ({fy:.2f}, {-fx:.2f}). "
        f"(↑yaw=left/CCW, ↓yaw=right/CW)"
    )

class GetTargetInput(BaseModel):
    instruction: str = Field(description="Description of the target object to locate.")
    image_path: str = Field(default="", description=
                            "Optional path to a target image on disk. When provided, the VLM will "
                            "match this image against the current camera view instead of using the "
                            "text instruction alone.")


@tool(args_schema=GetTargetInput)
def get_target_object(instruction: str, image_path: str=""):
    """
    Get 3D world coordinates of a target that is ALREADY VISIBLE in the current camera view.
    
    PREREQUISITE: You MUST first call get_current_view and visually confirm the target
    is in the frame. NEVER call this tool blindly — it will waste time and fail if the
    target is not visible.

    Parameters:
    - instruction: Text description of the target.
    - image_path:   (OPTIONAL) Path to a target image file. When provided, the VLM
                    compares this image against the current camera view to locate the
                    target. When empty, the VLM uses the text instruction only.

    Use this ONLY when you are ready to navigate to a confirmed-visible target.
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")
    if _sub_llm is None:
        raise Exception("Error: Chat Model for this task is not initialized.")

    img = _control.get_current_view()
    img_url = f"data:image/jpeg;base64,{_encode_image(img)}"

    structured_llm = _sub_llm.with_structured_output(SPFActionSchema)

    if image_path and image_path.strip():
        try:
            target_img = Image.open(image_path.strip())
        except FileNotFoundError:
            return f"Error: Target image not found at path '{image_path}'."
        target_url = f"data:image/jpeg;base64,{_encode_image(target_img)}"
        prompt_value = create_image_match_prompt().invoke({
            "instruction": instruction,
            "target_url": target_url,
            "image_url": img_url,
        })
    else:
        prompt_value = create_spf_prompt().invoke({
            "instruction": instruction,
            "image_url": img_url,
        })

    response = cast(SPFActionSchema, structured_llm.invoke(prompt_value))

    if not response.success:
        return "Target not found. Try to rotate perspective, analyze the camera view directly or analyze previous actions."

    spf_geometry = _control.spf_geometry
    norm_x, norm_y = response.point
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

    world_x = pos["x"] + (sx * math.sin(yaw_rad) + sy * math.cos(yaw_rad))
    world_y = pos["y"] + (-sx * math.cos(yaw_rad) + sy * math.sin(yaw_rad))
    # world_z = pos["z"] + sz  
    world_z = pos["z"]

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
    Move the agent to a specific world coordinate (ENU: x=East, y=North, z=Up).

    CRITICAL RULES:
    - NEVER call navigate_to_point with arbitrary coordinates — ONLY use exact values from get_target_object, OR small adjustments (<0.5m from current position) to fine-tune after a failed navigation.
    - NEVER invent or guess coordinates for exploration.
    - NEVER use navigate for "getting a better view" — use rotate instead.
    - z should always match current altitude (use get_current_position_and_rotation).
    """
    if _control is None:
        raise Exception("Error: Environment controller is not initialized. Call init_env() first.")

    start_state = _control.get_agent_state()
    start_x = start_state["position"]["x"]
    start_y = start_state["position"]["y"]
    start_z = start_state["position"]["z"]
    start_yaw = start_state["rotation"]["yaw"]

    target: NavTarget = {"x": x, "z": z,"y": y}
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
        reason = result.get("reason")
        if reason == "fsm_not_tracking":
            hint = "FSM not in tracking state. Tell user to run 'execute' command in fsm_interface terminal."
        elif reason == "blocked_by_obstacle":
            hint = "Blocked by obstacle. Check if target is already reached via get_current_view and describe how close it is. If NOT reached, you MUST call rotate to scan, then try to nav again."
        elif reason == "timeout":
            hint = "Navigation timed out. Check if target is already reached via get_current_view and describe how close it is. If NOT reached, you MUST call rotate to scan, then try to nav again."
        else:
            hint = reason
        return (
            f"Navigation failed: {hint}. Current position: "
            f"(x={end_x:.2f}, y={end_y:.2f}, z={end_z:.2f}, yaw={end_yaw:.2f}°)."
        )


class RotateInput(BaseModel):
    yaw: float = Field(description="The target yaw angle (in degrees) for the agent.")


@tool(args_schema=RotateInput)
def rotate(yaw: float) -> str:
    """
    Set agent absolute yaw orientation in degrees.
    
    Yaw convention (ENU):
    - 0°=East, 90°=North, 180°=West, 270°=South
    - Increase yaw = turn LEFT (CCW):  90→180→270
    - Decrease yaw = turn RIGHT (CW): 180→90→0
    
    To look RIGHT of current heading: use a SMALLER yaw value
    To look LEFT of current heading:  use a LARGER yaw value
    
    Only multiples of 30° are accepted (auto-rounded).
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
        reason = result.get("reason")
        if reason == "fsm_not_tracking":
            hint = "FSM not in tracking state. Tell user to run 'execute' command in fsm_interface terminal."
        elif reason == "timeout":
            hint = "Rotation timed out. Try again."
        else:
            hint = reason
        return f"Rotation FAILED: {hint} Current yaw: {end_yaw:.2f}°."

class LoadImageInput(BaseModel):
    image_path: str = Field(description="Path to the image file on disk.")

@tool(args_schema=LoadImageInput)
def load_image(image_path: str):
    """
    Load and display an image from disk. Use this to view a target image
    provided by the user so you know what to look for during scanning.
    """
    try:
        img = Image.open(image_path.strip())
    except FileNotFoundError:
        return f"Error: Image not found at path '{image_path}'."

    b64_str = _encode_image(img)
    return [
        {"type": "text", "text": f"Target image loaded from: {image_path}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
    ]

TOOLS_LIST = [
    get_current_view,
    get_target_object,
    load_image,
    navigate_to_point,
    rotate,
    get_current_position_and_rotation,
]
