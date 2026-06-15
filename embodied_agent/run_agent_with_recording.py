import base64
import json
from io import BytesIO
from typing import Callable, Optional, Literal

import numpy as np
from PIL import Image

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from embodied_agent.task_recorder import TaskRecorder


def run_agent_with_recording(
    agent: CompiledStateGraph,
    user_message: str,
    session_dir: str,
    get_frame: Callable[[], Optional[np.ndarray]],
    fps: int = 20,
    config: Optional[RunnableConfig] = None,
    stream_mode: Optional[Literal["updates", "values"]] = "updates",
) -> TaskRecorder:
    """Run an agent with background frame recording and log collection.

    Args:
        agent:         The compiled LangGraph agent (from create_spf_agent).
        user_message:  The user's instruction string.
        session_dir:   Output directory for frames/ and logs.jsonl.
        get_frame:     Zero-arg callable returning np.ndarray (H,W,3) or None.
        fps:           Background frame capture rate (default 20).
        config:        Optional LangGraph config dict (e.g. {"configurable": {"thread_id": "..."}}).
        stream_mode:   LangGraph stream mode (default "updates").

    Returns:
        The TaskRecorder instance (call TaskRecorder.stitch(session_dir) later to produce MP4).
    """
    if config is None:
        config = {"configurable": {"thread_id": "default"}}

    with TaskRecorder(session_dir, get_frame, fps=fps) as rec:

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            stream_mode=stream_mode,
        ):
            for node_name, update in chunk.items():
                if node_name in ("agent", "model"):
                    messages = update.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"\n[Tool Call] {tc['name']}")
                                print(f"   Args: {tc['args']}")
                                rec.log(
                                    "tool_call",
                                    tool_name=tc["name"],
                                    args=json.dumps(tc["args"], ensure_ascii=False),
                                )
                        text = getattr(msg, "content", "")
                        if isinstance(text, list):
                            text = " ".join(
                                block.get("text", "")
                                for block in text
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )
                        if text:
                            print(f"\n[Agent]: {text[:500]}")
                            rec.log("agent_reply", text=text)

                # ---------- Tool response (may contain images) ----------
                elif node_name == "tools":
                    messages = update.get("messages", [])
                    for msg in messages:
                        content = msg.content

                        if isinstance(content, list):
                            for block in content:
                                # Text response
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    txt = block["text"]
                                    if len(txt) > 300:
                                        txt = txt[:300] + "..."
                                    print(f"   [Tool Reply]: {txt}")
                                    rec.log("tool_reply", text=block["text"])

                                # Image response
                                elif (
                                    isinstance(block, dict)
                                    and block.get("type") == "image_url"
                                ):
                                    b64 = block["image_url"]["url"]
                                    if b64.startswith("data:image/jpeg;base64,"):
                                        b64 = b64[
                                            len("data:image/jpeg;base64,") :
                                        ]
                                    img_data = base64.b64decode(b64)
                                    img = Image.open(BytesIO(img_data))
                                    print(f"   [Tool returned image] ↓")
                                    # Display inline if running in a notebook
                                    try:
                                        from IPython.display import display as ipy_display
                                        ipy_display(img)
                                    except ImportError:
                                        pass
                        else:
                            txt = str(content)
                            if len(txt) > 300:
                                txt = txt[:300] + "..."
                            print(f"   [Tool Reply]: {txt}")
                            rec.log("tool_reply", text=str(content))

    # TaskRecorder.__exit__ has already saved logs.jsonl
    print(f"\nRecording saved to: {session_dir}")
    print(f"   Call TaskRecorder.stitch('{session_dir}') to generate MP4.")
    return rec
