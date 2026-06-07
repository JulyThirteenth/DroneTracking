# RL Point-Goal Integration Runbook

This repository is the ROS/PX4/Isaac deployment side of the drone experiments.
The RL policy itself is trained and evaluated in the sibling `Xsim_new`
repository. This document records the current working setup for deploying the
trained point-goal policy in `DroneTracking`.

## Repository Roles

### `Xsim_new`

`Xsim_new` is the RL training and offline evaluation repository.

- The current policy is a point-goal / hover-style policy.
- The policy is trained in an empty simulated dynamics environment.
- Its observation is a 17D vector:
  - position relative to goal in an NED-like frame
  - velocity in the same NED-like frame
  - full attitude quaternion
  - body angular velocity
  - previous action
- The current checkpoint used by `DroneTracking` is:

```text
../Xsim_new/runs/codex_hover_near_stabilize_400k/checkpoints/agent_400000.pt
```

### `DroneTracking`

`DroneTracking` is the deployment and validation repository.

- Isaac Sim / Pegasus provides the rendered simulator.
- PX4 SITL runs the flight stack.
- ROS 2 Humble bridges PX4 topics and controller commands.
- `fsm` coordinates prepare, takeoff, execute, return, and land.
- `tracking` contains the MPC / MPCC tracking controllers.
- `plan2track` publishes reference paths or fixed goals.
- `perception` converts depth images to scan-like obstacle points.

## Current Controller Split

There are two separate workflows. Keep them separate.

### MPC / Obstacle-Avoidance Workflow

Use this for normal waypoint tracking, keyboard-driven references, and HOCBF
obstacle avoidance.

```bash
export DRONE_RACING_CONFIG=config_oa.yaml
PLAN2TRACK_MODE=keyboard ./tools/run_oa_code.sh
```

`config_oa.yaml` is kept as an MPC/OA config:

- `runtime.controller: mpc`
- `tracking.hocbf.enabled: true`
- `PLAN2TRACK_MODE=keyboard` is appropriate here

### RL Fixed Point-Goal Workflow

Use this to validate the trained RL point-goal policy in PX4/Isaac.

```bash
./tools/run_fixed_goal_test.sh
```

This script uses `yamls/config_rl_goal.yaml` by default:

- `runtime.controller: rl_hover`
- `rl_hover.fallback_to_mpc_tracking: false`
- `tracking.hocbf.enabled: false`
- `plan2track.fixed_yaw: true`
- `PLAN2TRACK_MODE=fixed_goal`

Do not use keyboard velocity commands to judge this RL policy. Keyboard mode
creates a moving target, which is a different task from fixed point-goal
navigation.

## Start the Simulator

Start the simulator in one terminal:

```bash
cd /root/gpufree-data/devspace/drone/DroneTracking
SIM_SCENE_SOURCE=behavior1k SIM_SCENE_INDEX=0 ./tools/simdrone_single.sh
```

For an empty/simple scene, omit the Behavior1K variables or pass the scene/task
indices supported by `isaacsim/sim_single.py`.

Useful simulator environment variables:

```bash
SIM_SCENE_SOURCE=behavior1k
SIM_SCENE_INDEX=0
SIM_TASK_INDEX=0
SIM_SPAWN_CLEARANCE=1.0
SIM_SPAWN_SEED=123
QGC_APP=/path/to/QGroundControl-x86_64.AppImage
```

## Run the RL Fixed-Goal Test

In a second terminal:

```bash
cd /root/gpufree-data/devspace/drone/DroneTracking
./tools/run_fixed_goal_test.sh
```

The script automatically:

1. Starts the controller tmux session.
2. Publishes `prepare`.
3. Publishes `takeoff`.
4. Publishes `execute`.
5. Tracks a fixed point-goal.
6. Runs `tools/analyze_rl_goal_log.py`.

Default fixed goal:

```text
goal = position_at_execute + ENU(1.0, 0.0, 0.0)
goal_z = 1.0
```

Change the goal:

```bash
FIXED_GOAL_OFFSET_ENU=2.0,0,0 ./tools/run_fixed_goal_test.sh
FIXED_GOAL_OFFSET_ENU=0,2.0,0 ./tools/run_fixed_goal_test.sh
FIXED_GOAL_OFFSET_ENU=2.0,2.0,0 EXECUTE_S=40 ./tools/run_fixed_goal_test.sh
```

Use an absolute ENU goal instead of an offset:

```bash
FIXED_GOAL_ENU=1.0,2.0,1.0 ./tools/run_fixed_goal_test.sh
```

## RL Validation Metrics

`tools/analyze_rl_goal_log.py` reads the latest `fsm/log/*/ticks.csv`.

Important fields:

- `final_err`: final distance to goal
- `final_speed`: final vehicle speed
- `tail_err_mean`: mean distance over the last window
- `tail_speed_mean`: mean speed over the last window
- `cmd_sat_ratio`: body-rate command saturation ratio
- `z_min`, `z_max`, `z_mean`: altitude sanity check

Current PASS thresholds:

```text
tail_err_mean <= 0.30 m
tail_speed_mean <= 0.25 m/s
```

Known good result for a 1m fixed goal:

```text
final_err=0.0570
final_speed=0.0121
tail_err_mean=0.0591
tail_speed_mean=0.0085
verdict=PASS
```

## Important Implementation Notes

The RL policy only became stable after the deployed observation was matched to
the training observation:

- Convert ENU position error to the NED-like frame used by `Xsim_new`.
- Convert ENU velocity to the same NED-like frame.
- Subscribe to `/fmu/out/vehicle_attitude` and feed the full quaternion.
- Subscribe to `/fmu/out/vehicle_angular_velocity` and feed body rates.
- Include the previous policy action.

The relevant files are:

```text
fsm/fsm_rl_hover.py
fsm/fsm_node.py
fsm/fsm_ros.py
plan2track/fixed_goal2track.py
tools/run_fixed_goal_test.sh
tools/analyze_rl_goal_log.py
yamls/config_rl_goal.yaml
```

## Stop Sessions

```bash
tmux kill-session -t dronecnt
tmux kill-session -t dronesim
```

Use the first command for the controller session and the second for the
simulator session.
