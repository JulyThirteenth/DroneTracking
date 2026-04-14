# drone_racing

Current workflow for Pegasus Simulator 5.1, PX4 1.16, ROS 2 Humble, and the `mpcc` conda environment.

<p align="center">
  <img src="./assets/chaser_racing.gif" alt="Chaser racing" width="32%">
  <img src="./assets/mpc.gif" alt="MPC demo" width="32%">
  <img src="./assets/chaser_racing_isaacsim.gif" alt="Chaser racing in Isaac Sim" width="32%">
</p>

## Requirements

- Isaac Sim 5.1 with Pegasus Simulator
- PX4 1.16 configured in the Pegasus extension
- ROS 2 Humble on Ubuntu 22.04
- `px4-ros_ws` overlay built and sourced by `tools/simdrone_env.sh`
- `conda` environment `mpcc`
- ROS 2 package: `px4_msgs`

Load the project environment with:

```bash
cd /home/shaw/DroneSimulator/PegasusSimulator/examples/drone_racing
source ./tools/simdrone_env.sh
```

## Config

Runtime options are loaded from YAML through `DRONE_RACING_CONFIG`.

```bash
export DRONE_RACING_CONFIG=config_0.yaml
export DRONE_RACING_CONFIG=config_1.yaml
export DRONE_RACING_CONFIG=cfg_visual.yaml
export DRONE_RACING_CONFIG=cfg_lidar.yaml
```

Relative config names are resolved under `yamls/`; absolute paths also work. If unset, `config_0.yaml` is used.

Current built-in configs:

- `config_0.yaml`: default single-vehicle config
- `config_1.yaml`: namespaced `/px4_1/*` config
- `cfg_visual.yaml`: visual pipeline config
- `cfg_lidar.yaml`: lidar pipeline config

## Start Simulation

Recommended launcher:

```bash
cd /home/shaw/DroneSimulator/PegasusSimulator/examples/drone_racing
./tools/simdrone_single.sh
```

It starts Isaac Sim, `MicroXRCEAgent`, and QGroundControl.

Manual startup:

```bash
cd /home/shaw/DroneSimulator/PegasusSimulator/examples/drone_racing
isaac_run ./isaacsim/sim_single.py
MicroXRCEAgent udp4 -p 8888
~/DroneSimulator/QGroundControl-x86_64.AppImage
```

Task and scene selection:

```bash
isaac_run ./isaacsim/sim_single.py --list-tasks
isaac_run ./isaacsim/sim_single.py --list-scenes
isaac_run ./isaacsim/sim_single.py --task-index 0 --scene-index 0
```

## Start Controller

Recommended launcher:

```bash
cd /home/shaw/DroneSimulator/PegasusSimulator/examples/drone_racing
./tools/run_code.sh
./tools/run_code.sh mysession cfg_visual.yaml
```

It starts:

- `python plan2track/plan2track.py`
- `python fsm/fsm_node.py`
- `python fsm/fsm_app.py`

Manual startup:

```bash
cd /home/shaw/DroneSimulator/PegasusSimulator/examples/drone_racing
source ./tools/simdrone_env.sh
export DRONE_RACING_CONFIG=cfg_visual.yaml

python plan2track/plan2track.py
python fsm/fsm_node.py
python fsm/fsm_app.py
```

`fsm/fsm_node.py` reads controller and solver settings from the selected YAML.

## FSM Commands

Use `fsm/fsm_app.py` for interactive commands:

- `prepare`
- `takeoff`
- `execute`
- `return`
- `land`
- `abort`
- `help`
- `state`
- `quit`

## YAML Sections

The config loader is `yamls/config.py`.

- `runtime`: controller and solver selection
- `fsm`: FSM topics, logging, takeoff speed, and optional auto-land thresholds
- `plan2track`: path topics, waypoint loading mode, loop mode, fixed yaw, and `init_yaw`
- `tracking_ros`: PX4 ROS topics, target system, and `pub_offboard`
- `tracking.tasks`: waypoint file selection
- `tracking.mpc`: horizon, timestep, and reference speed
- `tracking.control`: controller timer period
- `tracking.yaw`: yaw gain and yaw-rate limit
- `tracking.ctbr`: body-rate/thrust conversion parameters
- `tracking.accel_fusion`: acceleration smoothing
- `tracking.constraints`: MPC state/input bounds
- `tracking.mpc_cost`: MPC cost weights
- `tracking.mpcc_cost`: MPCC cost weights

Notes:

- Isaac Sim uses an ENU world frame. Waypoint text files are loaded as NED and converted to ENU by the planner/simulation utilities.
- `origin_mode: fixed` keeps the waypoint file origin fixed; `origin_mode: first_xy` shifts the path so the first waypoint has local `x/y = 0/0`.
- `fixed_yaw: true` publishes `init_yaw` as the yaw command. `fixed_yaw: false` publishes path-tangent yaw.
- `pub_offboard: false` prevents publishing the OFFBOARD `VehicleCommand`; rate setpoints and offboard mode messages are still controlled by the FSM behavior state.

## Key Files

- `fsm/fsm_node.py`: FSM ROS node and transition coordinator
- `fsm/behaviors.py`: per-state behavior and tracker command publication
- `fsm/fsm_app.py`: interactive FSM terminal
- `plan2track/plan2track.py`: waypoint/path bridge for MPC/MPCC tracking
- `tracking/tracking_cnt.py`: controller step and yaw-rate logic
- `tracking/tracking_ros.py`: PX4 ROS message bridge
- `isaacsim/sim_single.py`: single-vehicle Isaac Sim app
- `yamls/config.py`: YAML config loader
- `yamls/*.yaml`: runtime configs
