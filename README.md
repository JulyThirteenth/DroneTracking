# DroneTracking

DroneTracking is a Pegasus/Isaac Sim based drone tracking and obstacle-avoidance
workspace. The current workflow targets Isaac Sim 5.1, PX4 1.16, ROS 2 Humble,
and the conda environment.

<p align="center">
  <img src="./assets/chaser_racing.gif" alt="Chaser racing" width="32%">
  <img src="./assets/mpc.gif" alt="MPC demo" width="32%">
  <img src="./assets/chaser_racing_isaacsim.gif" alt="Chaser racing in Isaac Sim" width="32%">
</p>

## Requirements

- Isaac Sim 5.1 with Pegasus Simulator
- PX4 1.16 configured in the Pegasus extension
- ROS 2 Humble on Ubuntu 22.04
- ROS 2 overlay with `px4_msgs`, sourced by `bash/simdrone_env.sh`
- Micro XRCE-DDS Agent for forwarding PX4 uORB topics to ROS 2
- Optional: `tmux` for one-command launch scripts
- Optional: QGroundControl, configured by `QGC_PATH` in the launch bash scripts

Load the project environment with:

```bash
cd ${Path2Project}/DroneTracking
source ./bash/simdrone_env.sh
```

## Config

Runtime options are loaded from YAML through `DRONE_TRACKING_CONFIG`.

```bash
export DRONE_TRACKING_CONFIG=cfg_0.yaml
export DRONE_TRACKING_CONFIG=cfg_1.yaml
export DRONE_TRACKING_CONFIG=cfg_oa.yaml
export DRONE_TRACKING_CONFIG=cfg_visual.yaml
export DRONE_TRACKING_CONFIG=cfg_lidar.yaml
```

Relative config names are resolved under `cfg/`; absolute paths also work. If
unset, `cfg/cfg_0.yaml` is used.

Built-in configs:

- `cfg_0.yaml`: default single-vehicle config
- `cfg_1.yaml`: namespaced `/px4_1/*` config
- `cfg_oa.yaml`: MPC obstacle-avoidance config

## Start Simulation

Recommended launcher:

```bash
cd ${Path2Project}/DroneTracking
./bash/sim_scenes_txt.sh
```

It starts a tmux session with:

- `rviz2 -d layout.rviz`
- `python isaacsim/sim_tf_tree.py`
- `isaac_run isaacsim/sim_txt.py`
- `MicroXRCEAgent udp4 -p 8888`
- QGroundControl

Manual startup:

```bash
cd ${Path2Project}/DroneTracking
source ./bash/simdrone_env.sh
isaac_run isaacsim/sim_txt.py
MicroXRCEAgent udp4 -p 8888
./QGroundControl-x86_64.AppImage
```

Txt scene selection:

```bash
isaac_run isaacsim/sim_scenes_txt.py --list-scenes
isaac_run isaacsim/sim_scenes_txt.py --list-waypoints
isaac_run isaacsim/sim_scenes_txt.py --scene isaacsim/scenes/safmc --scene-index 0
isaac_run isaacsim/sim_scenes_txt.py --scene isaacsim/scenes/safmc/two_gate.txt --waypoints plan2track/waypoints/line_waypoint.txt
isaac_run isaacsim/sim_scenes_txt.py --scene-index 0 --waypoint-index 12
```

`sim_scenes_txt.py` loads txt scene files from `isaacsim/scenes/*.txt` or
subdirectories such as `isaacsim/scenes/scenes_txt/`. It loads path points from
`plan2track/waypoints/`, draws them in Isaac Sim, and spawns the drone from the
first waypoint converted from NED to ENU with a 0.07m height offset.

Behavior1k examples:

```bash
isaac_run isaacsim/sim_behavior1k.py --list-scenes
isaac_run isaacsim/sim_behavior1k.py --scene-index 0 --spawn-point 1 2 0.07
```

`bash/sim_scenes_txt.sh` starts `isaacsim/sim_scenes_txt.py` and forwards extra arguments to
it. To launch Behavior1k through the same tmux layout:

```bash
./bash/sim_behavior1k.sh --scene-index 0 --spawn-point 1 2 0.07
```

Behavior1k material paths are configured by `sim_behavior1k.py`.

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
