# DroneTracking — Agent Navigation

This branch provides **agent-based navigation** for drone tracking. The agent interprets visual scenes and
controls drone movement via ROS2, using LLM-powered reasoning (SPF — Semantic Platform Framework).

## Progress

| Module | Status |
|--------|--------|
| **ROS2_CONTROL** — Bridge between agent and ROS2 (navigation / rotation) | ✅ Done |
| **AGENT Debugging** — Prompt engineering (skill & tool tuning) | ⬜ Not started |

- Isaac Sim 5.1 with Pegasus Simulator
- PX4 1.16 configured in the Pegasus extension
- ROS 2 Humble on Ubuntu 22.04
- ROS 2 overlay with `px4_msgs`, sourced by `bash/simdrone_env.sh`
- Micro XRCE-DDS Agent for forwarding PX4 uORB topics to ROS 2
- Optional: `tmux` for one-command launch scripts
- Optional: QGroundControl, configured by `QGC_PATH` in the launch bash scripts

---

## Environment Configuration

LLM settings are loaded from a `.env` file in the project root via [`load_dotenv()`](./embodied_agent/init_agent_env_and_create_agent.py:6). Refer to [`.env.example`](./.env.example) for the available variables and a ready-to-use template.

Copy the template and fill in your API key:

```bash
cd ${Path2Project}/DroneTracking
source ./bash/simdrone_env.sh
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

<<<<<<< HEAD
## Quick Start

### Prerequisites

The system requires the `mpcc` conda environment (defined in [`env_ros_humble.yml`](./env_ros_humble.yml)).

### 1. Launch the simulator (Isaac Sim)

```bash
isaac_run /path/to/DroneTracking/isaacsim/sim_single.py \
  --ext-folder ~/PegasusSimulator/extensions \
  --enable pegasus.simulator \
  --enable omni.isaac.ros2_bridge \
  --enable omni.replicator.core
```

### 2. Start the ROS2 bridge (MicroXRCE-DDS Agent)

```bash
/path/to/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent udp4 -p 8888
```

### 3. Launch the drone control pipeline

```bash
export DRONE_TRACKING_CONFIG=config_oa.yaml
cd /path/to/DroneTracking/tools
bash run_oa_code_1.sh
```

This starts the FSM and tracking modules in a `tmux` session (see [`run_oa_code_1.sh`](./tools/run_oa_code_1.sh) for details).

### 4. Jupyter notebook development

If you prefer to interact with the agent via a notebook instead of the full pipeline:

#### 4.1 Install the kernel

```bash
conda activate mpcc
python -m ipykernel install --user --name simdrone_env --display-name "Python (SimDrone)"
```

#### 4.2 Edit the kernel launcher config

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

#### 4.3 Open the notebook

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

<<<<<<< HEAD
- **`InfoNode`** — Receives camera feed, position, and attitude; provides the agent's perception.
- **`NavigationNode`** — Publishes reference trajectories and yaw commands to the MPC tracking system.
=======
## Start Controller

Tracking only:

```bash
cd ${Path2Project}/DroneTracking
./bash/run_code.sh
./bash/run_code.sh dronecnt cfg_visual.yaml
```

Obstacle-avoidance stack:

```bash
cd ${Path2Project}/DroneTracking
./bash/run_oa_code.sh
./bash/run_oa_code.sh dronecnt cfg_0.yaml
```

`run_code.sh` starts:

- `python plan2track/plan2track.py`
- `python -m fsm.fsm_main`
- `python -m fsm.fsm_interface`

`run_oa_code.sh` starts:

- `python plan2track/vel2track.py`
- `python -m fsm.fsm_main`
- `python -m fsm.fsm_interface`
- `python perception/depth2scan.py`

Manual startup:

```bash
cd ${Path2Project}/DroneTracking
source ./bash/simdrone_env.sh
export DRONE_TRACKING_CONFIG=cfg_0.yaml

python plan2track/plan2track.py
python -m fsm.fsm_main
python -m fsm.fsm_interface
python perception/depth2scan.py --ros-args \
  -p depth_topic:=/depth \
  -p scan_topic:=/depth2scan/scan \
  -p points_topic:=/depth2scan/points \
  -p frame_id:=drone_fpv_camera \
  -p config_path:=$PWD/perception/yaml/depth_transform.yaml
```

## FSM Commands

Use `python -m fsm.fsm_interface` for interactive commands:

- `prepare`
- `takeoff`
- `execute`
- `return`
- `land`
- `abort`
- `help`
- `state`
- `quit`

## Planning Data

Waypoint files for tracking live under:

```text
plan2track/waypoints/
```

Minimum-snap task/control files live under:

```text
plan2track/tasks/
```

Generate waypoints with:

```bash
python plan2track/generate_waypoints.py
```

The selected waypoint file is configured by:

```yaml
plan2track:
  path:
    file: plan2track/waypoints/line_waypoint.txt
```

## Perception

`perception/depth2scan.py` converts a ROS depth image into:

- `sensor_msgs/LaserScan` on `/depth2scan/scan`
- `sensor_msgs/PointCloud2` on `/depth2scan/points`

The depth conversion config is:

```text
perception/yaml/depth_transform.yaml
```

Current assumptions:

- Isaac Sim depth image is `32FC1` in meters
- horizontal FOV is 90 deg
- scan bins are angular buckets over the camera horizontal FOV
- max-range points are published as `inf` in `LaserScan` and are ignored by the
  obstacle-avoidance controller

## Controllers

The runtime controller is selected by:

```yaml
runtime:
  controller: mpc
  solver: osqp
```

Supported controller families:

- `mpc`: OSQP tracking MPC
- `mpcc`: MPCC tracker

When `tracking.hocbf.enabled` is true and depth scan points are available, the
MPC path tracker adds HOCBF obstacle-avoidance constraints. Without valid depth
scan points, it behaves as the normal tracking MPC.

Relevant YAML sections:

- `runtime`: controller and solver selection
- `topics`: FSM, planning, tracking, PX4, and perception ROS topics
- `vehicle`: PX4 target system and offboard publication switch
- `fsm`: FSM logging, takeoff, and auto-land behavior parameters
- `plan2track`: waypoint file, loading mode, loop mode, fixed yaw, and initial yaw
- `tracking.mpc`: horizon, timestep, and reference speed
- `tracking.mpc.cost`: MPC cost weights
- `tracking.mpcc`: MPCC cost weights and progress limits
- `tracking.control_loop`: controller timer period
- `tracking.yaw`: yaw gain and yaw-rate limit
- `tracking.ctbr`: body-rate/thrust conversion parameters
- `tracking.accel_fusion`: acceleration smoothing
- `tracking.constraints`: MPC state/input bounds
- `tracking.hocbf`: camera offset, safety radius, gains, and slack weight

## Coordinate Notes

- Isaac Sim uses an ENU world frame.
- Waypoint text files are loaded as NED and converted to ENU by planning and
  simulation utilities.
- `origin_mode: fixed` keeps the waypoint-file origin fixed.
- `origin_mode: first_xy` shifts the path so the first waypoint has local
  `x/y = 0/0`.
- `fixed_yaw: true` publishes `init_yaw` as the yaw command.
- `fixed_yaw: false` publishes path-tangent yaw.

## Key Files

- `isaacsim/sim_txt.py`: txt-scene Isaac Sim app
- `isaacsim/sim_behavior1k.py`: Behavior1k USD-scene Isaac Sim app
- `isaacsim/sim_base.py`: shared Pegasus vehicle, camera, and run loop
- `isaacsim/sim_utils.py`: scene, waypoint marker, and behavior1k helpers
- `isaacsim/sim_tf_tree.py`: ROS TF tree publisher
- `plan2track/plan2track.py`: waypoint/path bridge for MPC/MPCC tracking
- `plan2track/generate_waypoints.py`: minimum-snap waypoint generation helper
- `perception/depth2scan.py`: depth image to pseudo LaserScan converter
- `tracking/tracking_cnt.py`: controller step, CTBR conversion, and yaw-rate logic
- `tracking/tracking_osqp.py`: OSQP MPC/HOCBF solver implementation
- `fsm/fsm_main.py`: FSM node startup
- `fsm/fsm_node.py`: FSM ROS node and transition coordinator
- `fsm/fsm_mpc.py`: MPC/MPCC behavior and tracker command publication
- `fsm/fsm_interface.py`: interactive FSM terminal
- `cfg/config.py`: YAML config loader
- `cfg/*.yaml`: runtime configs
>>>>>>> b9b2d4b (dev obstacle avoidance: tidy code)
=======
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
>>>>>>> 2aa21c2 (Update readme and env file)
