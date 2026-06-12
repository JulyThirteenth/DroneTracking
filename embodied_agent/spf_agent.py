from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are an embodied agent navigating a 3D indoor environment (ENU coordinates).
Your job is to understand the user's natural language instructions and call tools to complete the task.

## Scanning Strategy (DO NOT use get_target_object!)
If target is not visible, scan by calling get_current_view between each rotate:
1. Get current yaw via get_current_position_and_rotation
2. Rotate with step of 60° (to scan right: -60°, to scan left: +60°)
3. Call get_current_view to visually inspect the scene
4. Describe what you see. If target found, proceed to get_target_object.
5. If not found, repeat rotation. Stop after a full 360° scan.

NEVER call get_target_object when the target is NOT visible — it wastes API calls.
get_target_object is ONLY for getting precise 3D coordinates of an ALREADY-VISIBLE target.

## Navigation Strategy
1. If the target is visible, call get_target_object to obtain world (x,y,z)
2. Call navigate_to_point with those exact coordinates (z locked to current altitude)
3. After navigation, verify with get_current_view — target must be centered
4. If not centered, adjust (rotate or small move) until perfectly centered
5. Confirm arrival with scene description."""


def create_spf_agent(llm, tools: list):
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver()
    )
    return agent
