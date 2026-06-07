# DroneTracking RL Point-Goal Deployment

This repository is the ROS 2 / PX4 / Isaac Sim deployment side for a drone
point-goal RL policy trained in the sibling `Xsim_new` repository.

The current focus is **deploying a trained RL policy into DroneTracking and
validating fixed point-to-point flight**. The older MPC / MPCC / obstacle
avoidance stack is still present, but it is secondary for the current workflow.

## What Is New

- Added an `rl_hover` FSM behavior that loads a trained `Xsim_new` policy and
  publishes PX4 body-rate + thrust setpoints.
- Matched the deployed RL observation to the training observation:
  - ENU position error is converted to the NED-like frame used by `Xsim_new`.
  - ENU velocity is converted to the same frame.
  - PX4 attitude is read from `/fmu/out/vehicle_attitude`.
  - PX4 body angular velocity is read from `/fmu/out/vehicle_angular_velocity`.
  - Previous policy action is fed back to the policy.
- Added a fixed-goal publisher for clean RL point-goal validation.
- Added an automated fixed-goal test script and log analyzer.
- Split configs so MPC/OA and RL validation do not conflict.

## Repository Roles

`Xsim_new` trains and evaluates the RL policy in an empty dynamics environment.
The checkpoint currently used here is:

```text
../Xsim_new/runs/codex_hover_near_stabilize_400k/checkpoints/agent_400000.pt
```

`DroneTracking` runs the deployed stack:

- Isaac Sim / Pegasus for simulation
- PX4 SITL for the flight stack
- ROS 2 Humble for topics and control
- FSM commands: `prepare`, `takeoff`, `execute`, `land`
- RL or MPC controller output to PX4 body-rate setpoints

## Main Configs

Use `DRONE_TRACKING_CONFIG` or `DRONE_RACING_CONFIG` to select a YAML file under
`yamls/`.

```text
yamls/config_rl_goal.yaml  RL fixed point-goal validation
yamls/config_oa.yaml       MPC obstacle avoidance / keyboard tracking
yamls/config_0.yaml        Legacy MPCC/default tracking config
```

`config_rl_goal.yaml` is the validated RL deployment config:

```yaml
runtime:
  controller: rl_hover

rl_hover:
  checkpoint: ../Xsim_new/runs/codex_hover_near_stabilize_400k/checkpoints/agent_400000.pt
  fallback_to_mpc_tracking: false

tracking:
  hocbf:
    enabled: false
```

`config_oa.yaml` is intentionally kept as MPC/OA:

```yaml
runtime:
  controller: mpc

tracking:
  hocbf:
    enabled: true
```

Do not use keyboard velocity commands to evaluate the current RL policy.
Keyboard mode creates a moving target, while the trained policy should be tested
as a fixed point-goal controller.

## Start the Simulator

On the server:

```bash
cd /root/gpufree-data/devspace/drone/DroneTracking
SIM_SCENE_SOURCE=behavior1k SIM_SCENE_INDEX=0 ./tools/simdrone_single.sh
```

Useful simulator variables:

```bash
SIM_SCENE_SOURCE=behavior1k
SIM_SCENE_INDEX=0
SIM_TASK_INDEX=0
SIM_SPAWN_CLEARANCE=1.0
SIM_SPAWN_SEED=123
```

If Behavior1K is not needed, omit `SIM_SCENE_SOURCE` and `SIM_SCENE_INDEX`.

## Run RL Fixed-Goal Validation

In another terminal:

```bash
cd /root/gpufree-data/devspace/drone/DroneTracking
./tools/run_fixed_goal_test.sh
```

The script:

1. Starts the controller tmux session.
2. Sends `prepare`.
3. Sends `takeoff`.
4. Sends `execute`.
5. Publishes one fixed goal.
6. Waits for tracking.
7. Analyzes the latest FSM log.

Default goal:

```text
goal = position_at_execute + ENU(1.0, 0.0, 0.0)
goal_z = 1.0
```

Try farther goals:

```bash
FIXED_GOAL_OFFSET_ENU=2.0,0,0 ./tools/run_fixed_goal_test.sh
FIXED_GOAL_OFFSET_ENU=3.0,0,0 EXECUTE_S=40 ./tools/run_fixed_goal_test.sh
FIXED_GOAL_OFFSET_ENU=2.0,2.0,0 EXECUTE_S=40 ./tools/run_fixed_goal_test.sh
```

Expected analyzer output for a good 1m run:

```text
final_err=0.0570
final_speed=0.0121
tail_err_mean=0.0591
tail_speed_mean=0.0085
verdict=PASS
```

The analyzer currently marks a run as `PASS` when:

```text
tail_err_mean <= 0.30 m
tail_speed_mean <= 0.25 m/s
```

You can also run the analyzer manually:

```bash
python3 tools/analyze_rl_goal_log.py --window 300
```

## Run Legacy MPC / OA

For the older MPC obstacle-avoidance stack:

```bash
cd /root/gpufree-data/devspace/drone/DroneTracking
export DRONE_RACING_CONFIG=config_oa.yaml
PLAN2TRACK_MODE=keyboard ./tools/run_oa_code.sh
```

This mode is useful for keyboard-generated references and MPC/HOCBF tracking,
not for judging the current RL point-goal policy.

## Important Files

```text
fsm/fsm_rl_hover.py              RL policy loader and PX4 CTBR adapter
fsm/fsm_node.py                  FSM ROS node, PX4 state subscriptions
fsm/fsm_ros.py                   PX4 message conversion helpers
plan2track/fixed_goal2track.py   Fixed-goal publisher for RL validation
tools/run_fixed_goal_test.sh     Automated RL fixed-goal test
tools/analyze_rl_goal_log.py     Log summary and PASS/CHECK verdict
tools/run_oa_code.sh             Controller launcher, keyboard or fixed-goal mode
tools/simdrone_single.sh         Simulator launcher
yamls/config_rl_goal.yaml        RL point-goal config
yamls/config_oa.yaml             MPC/OA config
docs/rl_point_goal_runbook.md    Detailed setup and validation notes
```

## Stop Sessions

```bash
tmux kill-session -t dronecnt
tmux kill-session -t dronesim
```

`dronecnt` is the controller session. `dronesim` is the simulator session.
