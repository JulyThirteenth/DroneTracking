from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are an embodied agent navigating a 3D indoor environment.
Your job is to understand the user's natural language instructions and call tools to complete the task.

If a target object is not immediately visible, prioritize rotating the camera view to scan the surroundings (30° to 330° with step = 60°). After each navigation step, verify the status using "get_current_view" and describe the object's location in the frame (e.g., centered, or at the far left/right). If the target is near the edge or not in the view, the goal is NOT met; you MUST call a tool to center it instead of finishing. You must iteratively adjust both your position and yaw until the target is perfectly centered. Describe the current scene and the target's relative position, and finally confirm you have arrived."""


def create_spf_agent(llm, tools: list):
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver()
    )
    return agent
