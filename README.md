# DroneTracking — Agent Navigation

This branch provides **agent-based navigation** for drone tracking. The agent interprets visual scenes and
controls drone movement via ROS2, using LLM-powered reasoning (SPF — Semantic Platform Framework).

## Progress

| Module | Status |
|--------|--------|
| **ROS2_CONTROL** — Bridge between agent and ROS2 (navigation / rotation) | ✅ Done |
| **AGENT Debugging** — Prompt engineering (skill & tool tuning) | ⬜ Not started |

Currently tested via [`test_agent.ipynb`](./test_agent.ipynb).

---

## Environment Configuration

LLM settings are loaded from a `.env` file in the project root via [`load_dotenv()`](./embodied_agent/init_agent_env_and_create_agent.py:6). Refer to [`.env.example`](./.env.example) for the available variables and a ready-to-use template.

Copy the template and fill in your API key:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider (`OPENAI` / `GOOGLE`) | `OPENAI` |
| `LLM_API_KEY` | API key | **Required** |
| `LLM_BASE_URL` | Custom API endpoint (optional) | — |
| `LLM_MODEL` | Model name | `gpt-4o` (OPENAI) / `gemini-2.5-pro` (GOOGLE) |
| `LLM_TEMPERATURE` | Generation temperature | `0.7` |

> **Note**: `.env` is listed in `.gitignore` — do not commit it to version control.

---

## Quick Start (Jupyter Kernel)

The notebook requires the `mpcc` conda environment (defined in [`env_ros_humble.yml`](./env_ros_humble.yml)).

### 1. Install the kernel

```bash
conda activate mpcc
python -m ipykernel install --user --name simdrone_env --display-name "Python (SimDrone)"
```

### 2. Edit the kernel launcher config

Edit `~/.local/share/jupyter/kernels/simdrone_env/kernel.json` with the following content:

```json
{
 "argv": [
  "bash",
  "/path/to/DroneTracking/tools/simdrone_env_ipynb_wrapper.sh",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (SimDrone)",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "kernel_protocol_version": "5.5"
}
```

> **Note**: Make sure the wrapper script is executable:
> ```bash
> chmod +x /path/to/DroneTracking/tools/simdrone_env_ipynb_wrapper.sh
> ```

### 3. Open the notebook

```bash
jupyter notebook test_agent.ipynb
```

Or in VS Code, open [`test_agent.ipynb`](./test_agent.ipynb) and select the `Python (SimDrone)` kernel.

---

## Project Structure

```
embodied_agent/
├── __init__.py
├── base_control.py          # Abstract controller interface
├── ros2_control.py          # ROS2 bridge (InfoNode + NavigationNode)
├── spf_agent.py             # LangGraph agent definition
├── spf_geometry.py          # Camera geometry utilities
├── spf_navigation_prompts.py# SPF prompts & schemas
└── spf_tools.py             # LangChain tools (get_view, navigate, rotate)
```

[`ros2_control.py`](./embodied_agent/ros2_control.py) implements two ROS2 nodes:

- **`InfoNode`** — Receives camera feed, position, and attitude; provides the agent's perception.
- **`NavigationNode`** — Publishes reference trajectories and yaw commands to the MPC tracking system.
