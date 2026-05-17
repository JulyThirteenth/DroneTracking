from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class Obstacle(BaseModel):
    bounding_box: list[int] = Field(
        description="[ymin, xmin, ymax, xmax] coordinates for detected obstacles."
    )
    label: str = Field(description="The type of obstacle (e.g., 'traffic cone', 'wall').")


class SPFActionSchema(BaseModel):
    success: bool = Field(
        description="True if the target object is found and reachable, False otherwise."
    )
    point: tuple[int, int] = Field(description="The [y, x] pixel coordinates in a 0-1000 scale. [500, 500] is the center.")
    depth: int = Field(ge=1, le=10, description="Discrete depth label from 1 (very close) to 10 (very far).")
    label: str = Field(description="A natural language description of the specific action being taken.")
    obstacles: list[Obstacle] = Field(
        default_factory=list,
        description="A list of objects to avoid in the current scene."
    )


NAV_SYSTEM_PROMPT = """You are a drone navigation expert analyzing a drone camera view.
Your goal is to guide a drone to reach a target."""

NAV_USER_PROMPT_TEMPLATE = """Task: {instruction}
Navigation-only mode: the output obstables should be an empty list.

Instructions:
1. Analyze the camera view for the target.
2. Identify all objects matching the task description.
3. Select the most relevant target object.
4. Place a single point DIRECTLY ON the object (if found).

Coordinate System:
- x: 0-1000 scale (0=left, 500=center, 1000=right)
- y: 0-1000 scale (0=top/sky, 500=center, 1000=bottom/ground)
- depth: 1-10 scale based on the relative size/distance of the object in the frame.

IMPORTANT:
- Place the point PRECISELY on the center of the target object
- Choose the largest/closest matching object if multiple exist
- Assess the depth based on how much of the frame the object occupies
- Your accuracy in point placement is critical for navigation success"""

NAV_OBSTABLE_USER_PROMPT_TEMPLATE = """Task: {instruction}

Instructions:
1. Analyze the camera view for the target and all potential obstacles.
2. Detect obstacles (cones, walls, etc.) and provide their bounding boxes. 
3. Select a waypoint that leads to the target WITHOUT intersecting any obstacle boxes. 
4. Place the point on the path to the target that ensures safe clearance.

Coordinate System:
- x: 0-1000 scale (0=left, 500=center, 1000=right)
- y: 0-1000 scale (0=top/sky, 500=center, 1000=bottom/ground)
- depth: 1-10 scale based on the relative size/distance of the object in the frame.

IMPORTANT:
- Place the point PRECISELY on the center of the target object
- Choose the largest/closest matching object if multiple exist
- Assess the depth based on how much of the frame the object occupies
- The waypoint must be in FREE SPACE to avoid collisions
- Accuracy in bounding box placement is critical for safety"""


def create_spf_prompt(avoid_obstacles: bool = False):
    user_template = NAV_OBSTABLE_USER_PROMPT_TEMPLATE if avoid_obstacles else NAV_USER_PROMPT_TEMPLATE
    return ChatPromptTemplate.from_messages([
        ("system", NAV_SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": user_template},
            {"type": "image_url", "image_url": "{image_url}"}
        ])
    ])
