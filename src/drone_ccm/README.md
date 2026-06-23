# drone_ccm

ROS 2 deployment package for standard CCM and Ego-CCM. Training and benchmark
code is isolated in [`ctbr_cnt/`](ctbr_cnt/README.md); the ROS package contains
only reference generation, checkpoint inference, frame conversion and CTBR
publication.

## Data flow

```text
PX4 VehicleOdometry ----> controller ----> /drone_cnt/vehicle_rates_setpoint
                              ^
waypoint/hover -> reference --+
                              ^
                         /fsm/state
```

| Topic | Type | Owner |
|---|---|---|
| `/fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | PX4 |
| `/fsm/state` | `std_msgs/String` | `drone_fsm` |
| `/tracking/ccm_reference` | `std_msgs/Float64MultiArray` | reference node |
| `/drone_cnt/vehicle_rates_setpoint` | `px4_msgs/VehicleRatesSetpoint` | controller node |

Only `drone_fly` forwards the final setpoint to PX4. The reference message is an
atomic `(v_ref, R_ref, ctbr_ref, domain_signature)` sample; the signature blocks
controller/reference checkpoint mismatches.

The common interface is:

```text
input:  v, R, v_ref, R_ref, ctbr_ref
output: [collective_acceleration, roll_rate, pitch_rate, yaw_rate]
```

For `neu_ego_ccm_active.pt`, the runtime derives:

```text
gamma        = R.T @ e3
delta_v_body = R.T @ (v - v_ref)
delta_yaw    = wrap(yaw - yaw_ref)
```

It reconstructs reference body acceleration and yaw rate from
`R_ref, ctbr_ref`. The network needs no position, raw IMU acceleration,
measured body rate or additional ROS topic.

The package supports exactly two reference modes:

- `hover`: smoothly changes the engagement velocity to zero while retaining yaw.
- `waypoint`: generates time-indexed velocity, acceleration and yaw from a NED
  waypoint file. It does not use position-error feedback.

Waypoint yaw follows horizontal ENU velocity and holds its last value at rest.

## Parameters

| Launch argument | Default | Meaning |
| --- | ---: | --- |
| `checkpoint` | `neu_ccm_practical.pt` | Installed model basename or checkpoint path |
| `reference_mode` | `hover` | `hover` or `waypoint` |
| `waypoint_file` | empty | Absolute waypoint path, required in waypoint mode |
| `waypoint_speed` | `1.0` | Maximum 3-D path speed in m/s |
| `device` | `cpu` | PyTorch inference device |
| `hover_thrust` | `0.5812` | Normalized PX4 thrust balancing vehicle weight |

Static topic names, rates and timeouts are in `config/ccm.yaml`. Physical
collective acceleration is mapped to PX4 by:

```text
normalized_thrust = hover_thrust * collective_acceleration / 9.81
```

`hover_thrust` changes deployment calibration only; it does not require
retraining the checkpoint.

## Build

```bash
cd ~/devspace/agile_flight/DroneTracking
pixi run colcon --log-base log_pixi build --symlink-install \
  --build-base build_pixi --install-base install_pixi \
  --packages-up-to drone_ccm
source install_pixi/setup.bash
```

Only ROS-compatible `neu_ccm_practical.pt` and `neu_ego_ccm_active.pt` are
installed. Research checkpoint `neu_ccm_linear.pt` remains under `ctbr_cnt/`.

## Pegasus SITL

Start Pegasus/PX4:

```bash
bash sim/sim_scenes_txt.sh \
  --scene sim/scenes/one_cube.txt \
  --waypoints src/drone_ref/waypoints/line_waypoint.txt
```

Start the flight state machine:

```bash
source install_pixi/setup.bash
ros2 launch drone_fsm fsm.launch.py
```

Start Ego-CCM hover:

```bash
ros2 launch drone_ccm ccm.launch.py \
  checkpoint:=neu_ego_ccm_active.pt \
  reference_mode:=hover hover_thrust:=0.5812 device:=cpu
```

Start Ego-CCM waypoint tracking:

```bash
ros2 launch drone_ccm ccm.launch.py \
  checkpoint:=neu_ego_ccm_active.pt \
  reference_mode:=waypoint \
  waypoint_file:=$PWD/src/drone_ref/waypoints/line_waypoint.txt \
  waypoint_speed:=1.0 hover_thrust:=0.5812 device:=cpu
```

Use the FSM CLI sequence `prepare`, `takeoff`, wait for stable hover, then
`execute`. `drone_fly` is the sole PX4 control-input publisher.

## Real vehicle

Pass the measured value explicitly; for a vehicle hovering at `0.40`:

```bash
ros2 launch drone_ccm ccm.launch.py \
  checkpoint:=neu_ego_ccm_active.pt \
  reference_mode:=hover hover_thrust:=0.40 device:=cpu
```

Before flight, align PX4 `MPC_THR_HOVER` with the measured value, configure the
Offboard-loss action, and verify that the vehicle safely accepts these limits:

```text
velocity error:       each axis <= 1.5 m/s
reference tilt:       <= 1.0 rad
attitude error:       <= 0.7 rad
reference CTBR:       [6.81..12.81, +/-1.0, +/-1.0, +/-0.5]
controller CTBR:      [1.691..15.222, +/-3.840, +/-3.840, +/-1.570]
```

The controller latches a fault after stale, invalid or out-of-domain inputs.
`drone_fly` then stops rate control and commands current-position hold.

## Validate

```bash
source install_pixi/setup.bash
PYTHONPATH=$PWD/src/drone_ccm:$PYTHONPATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q src/drone_ccm/test
```
