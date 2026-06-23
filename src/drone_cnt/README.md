# drone_cnt

`drone_cnt` provides two ROS 2 controllers. Both consume PX4 local state and a
reference from `drone_ref`, then publish collective-thrust/body-rate (CTBR)
commands for `drone_fly`. This package never publishes directly to PX4.

## Controllers

| Executable | Reference | Method |
| --- | --- | --- |
| `drone_cnt_mpc` | `/tracking/ref_traj_path` | Time-indexed trajectory MPC |
| `drone_cnt_mpcc` | `/tracking/path` | Contouring MPC with optimized path progress |

The model is a 3D triple integrator with state `[position, velocity,
acceleration]` and jerk input. Differential-flatness conversion maps optimized
acceleration and jerk to normalized collective thrust and body rates.

## Interfaces

Inputs:

- `/fmu/out/vehicle_local_position` (`px4_msgs/VehicleLocalPosition`): NED PX4
  state, converted internally to ENU.
- `/fsm/state` (`std_msgs/String`): output is enabled only in `tracking`.
- `/planning/yaw_cmd_enu` (`std_msgs/Float32`): ENU yaw command.
- Controller-specific `nav_msgs/Path` reference shown above.
- `/fmu/out/vehicle_odometry` is checked only when `require_odometry:=true`.

All `nav_msgs/Path` positions and yaw commands are ENU.

Output:

- `/drone_cnt/vehicle_rates_setpoint` (`px4_msgs/VehicleRatesSetpoint`): internal
  CTBR command consumed and forwarded by `drone_fly`.

The node emits no command when the FSM is inactive, PX4 state is invalid/stale,
or the reference is unavailable. Solver failures reset the warm start and are
reported once per distinct failure.

## Run

Build and source the workspace:

```bash
pixi run colcon --log-base log_pixi build --symlink-install \
  --build-base build_pixi --install-base install_pixi \
  --packages-select drone_cnt
source install_pixi/setup.bash
```

Start exactly one controller after its matching `drone_ref` node. The launch
file always loads the installed controller configuration:

```bash
ros2 launch drone_cnt mpc.launch.py
# or
ros2 launch drone_cnt mpcc.launch.py
```

For the complete reference, controller and CSV-log stack, use one launch:

```bash
ros2 launch drone_cnt tracking.launch.py \
  controller_mode:=mpc \
  path_file:=waypoints/line_waypoint.txt \
  reference_speed:=3.0 \
  run_name:=mpc_line_01

ros2 launch drone_cnt tracking.launch.py \
  controller_mode:=mpcc \
  path_file:=waypoints/line_waypoint.txt \
  run_name:=mpcc_line_01
```

It starts exactly one `drone_ref`, one `drone_cnt`, and one `drone_log` node.
`drone_fsm` remains a separate launch and owns PX4 mode switching.

Typical complete chains are:

```text
drone_ref_mpc  -> drone_cnt_mpc  -> drone_fly
drone_ref_mpcc -> drone_cnt_mpcc -> drone_fly
```

Use another complete parameter file when needed:

```bash
ros2 launch drone_cnt mpc.launch.py config_file:=/absolute/path/mpc.yaml
```

Direct `ros2 run` requires an explicit parameter file because controller
parameters have no hidden code defaults:

```bash
ros2 run drone_cnt drone_cnt_mpc --ros-args \
  --params-file src/drone_cnt/config/mpc.yaml
```

## Main parameters

| Parameter | Configured value | Meaning |
| --- | ---: | --- |
| `horizon` | `15` | Optimization intervals |
| `mpc_dt` | `0.1` | Prediction interval, seconds |
| `control_dt` | `0.01` | ROS control period, seconds |
| `hover_thrust` | `0.58` | PX4 normalized hover thrust |
| `thrust_min`, `thrust_max` | `0.10`, `0.90` | CTBR thrust bounds |
| `yaw_kp` | `0.04` | Yaw feedback gain |
| `yaw_rate_limit` | `1.57` | Yaw-rate limit, rad/s |
| `state_timeout` | `0.25` | PX4 state freshness limit, seconds |
| `position_weight` | `[2000, 2000, 2000]` | MPC position cost |
| `contour_weight` | `500` | MPCC contour-error cost |
| `lag_weight` | `2` | MPCC lag-error cost |
| `progress_weight` | `1` | MPCC progress reward |

Jerk, jerk-rate, velocity and acceleration bounds are three-element ENU
vectors. Query the complete list with:

```bash
ros2 param list /drone_cnt_mpc
```

## Code layout

- `base.py`: the single ROS node implementation used by both executables.
- `cnt.py`: ROS-independent controller API.
- `cfg.py`: parameter schema, type conversion and validation; no tuned values.
- `config/mpc.yaml`, `config/mpcc.yaml`: controller and topic values.
- `launch/mpc.launch.py`, `launch/mpcc.launch.py`: installed launch entry points.
- `osqp.py`: sparse MPC/MPCC QP formulations.
- `utils.py`: coordinate conversion, path geometry and CTBR conversion.
