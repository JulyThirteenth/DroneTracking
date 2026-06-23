# Drone FSM

`drone_fsm` owns the mission state, converts each state into one flight
behavior, and is the only package component that publishes PX4 control inputs.
The package deliberately keeps state decisions, flight behavior, and PX4
message construction separated so each boundary can be tested independently.

## Components

| Component | Responsibility |
| --- | --- |
| `drone_fsm` | Validate commands, apply the transition table, publish state |
| `drone_fly` | Single PX4 writer: native takeoff, hold, and CTBR forwarding |
| `Px4Bridge` | Construct and publish PX4 Offboard messages and commands |
| `drone_cli` | Interactive English/Chinese command terminal |

The pure transition model is in `drone_fsm/model.py`. ROS QoS policy is shared
through `drone_fsm/qos.py`. The FSM state is published with reliable,
transient-local QoS so a newly started `drone_fly` receives the latest state.

## State flow

Nominal mission:

```text
preflight --prepare--> ready --takeoff--> hover_start
hover_start --execute--> tracking --return--> return_hover
return_hover --execute--> tracking
tracking/hover/ready/return_hover --land--> preflight
```

`abort` moves an active flight state (`tracking`, `return_hover`, `hover`)
to `hover`. In `preflight`, only `prepare` is accepted; in `ready`, only
`takeoff` and `land`; in `hover_start`, only `execute` and `land`. Unknown
commands are rejected by the FSM. Unknown state messages are rejected by
`drone_fly`, which keeps its last valid behavior.

State behavior:

| State | Flight behavior |
| --- | --- |
| `preflight` | On entry from a flight state, repeatedly request PX4 landing until disarmed |
| `ready` | Require fresh healthy PX4 state and capture the current hold point |
| `hover_start` | Arm, confirm armed, request native `NAV_TAKEOFF`, then wait for its ACK |
| `tracking` | Wait until native takeoff mode ends, pre-stream finite CTBR, request Offboard, and confirm the mode |
| `return_hover` | Return to the position captured when tracking started |
| `hover` | Hold the position captured when entering the state |

All internal positions and yaws passed to PX4 are NED. Controller body-rate
setpoints are expected in PX4 FRD convention and are forwarded without another
frame transform. A CTBR command older than `controller_timeout` is not used.

Native takeoff uses one fixed sequence: validate PX4 state, request Arm, confirm
armed through `VehicleStatus`, request `VEHICLE_CMD_NAV_TAKEOFF`, then require an
accepted/in-progress ACK. Position/velocity validity, passed preflight checks,
and no failsafe/failure-detector flag remain mandatory. Until PX4 reports a
control-quality heading, yaw remains unset. The tracking start is retained as
the return target; no global PX4 HomePoint is managed. Height/velocity stability
remains PX4's responsibility.

## Configuration

The installed default file is `config/drone_fsm.yaml`.

### `drone_fsm`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `initial_state` | `preflight` | Initial state; unknown values are rejected |
| `cmd_topic` | `/fsm/cmd` | String command input |
| `state_topic` | `/fsm/state` | Latched current state |
| `transition_topic` | `/fsm/transition` | JSON transition result |
| `info_topic` | `/fsm/info` | Human-readable rejection information |

### `drone_fly`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `control_rate` | `100.0` | Flight behavior timer frequency in Hz |
| `takeoff_height` | `1.0` | Positive takeoff height in metres |
| `offboard_prestream_time` | `1.0` | CTBR streaming time after `execute` and before requesting Offboard |
| `command_retry_period` | `1.0` | PX4 vehicle-command retry period in seconds |
| `controller_timeout` | `0.30` | Maximum CTBR receive age in seconds |
| `vehicle_status_timeout` | `1.50` | Maximum age for PX4's 2 Hz VehicleStatus |
| `local_position_timeout` | `0.50` | Maximum local-position receive age |
| `send_arm_command` | `true` | Allow this node to request arming |
| `send_offboard_command` | `true` | Allow this node to request Offboard after CTBR prestream |
| `target_system` | `1` | PX4 MAVLink target system ID |
| `local_position_topic` | `/fmu/out/vehicle_local_position` | PX4 local position and validity input |
| `vehicle_status_topic` | `/fmu/out/vehicle_status_v1` | PX4 status input |
| `command_ack_topic` | `/fmu/out/vehicle_command_ack` | PX4 command acknowledgement input |
| `controller_topic` | `/drone_cnt/vehicle_rates_setpoint` | CTBR controller input |

## Build and start

From the repository root inside the Pixi environment:

```bash
pixi shell
colcon build \
  --build-base build_pixi \
  --install-base install_pixi \
  --symlink-install \
  --packages-select drone_fsm
source install_pixi/setup.bash
```

Start the FSM state owner and flight executive:

```bash
ros2 launch drone_fsm fsm.launch.py
```

Start the interactive CLI in a second terminal:

```bash
source install_pixi/setup.bash
ros2 launch drone_fsm cli.launch.py
```

The usual command sequence is:

```text
prepare
takeoff
wait until PX4 leaves AUTO_TAKEOFF and holds altitude
execute
return
land
```

If `execute` arrives during `AUTO_TAKEOFF`, `drone_fly` waits and starts CTBR
pre-streaming only after PX4 has left native takeoff mode.

Commands can also be published without the CLI:

```bash
ros2 topic pub --once /fsm/cmd std_msgs/msg/String "{data: prepare}"
```

Override the complete configuration at launch time:

```bash
ros2 launch drone_fsm fsm.launch.py \
  config_file:=/absolute/path/to/drone_fsm.yaml
```

Inspect the live state and transition decisions:

```bash
ros2 topic echo /fsm/state
ros2 topic echo /fsm/transition
ros2 topic echo /fsm/info
```

Run static and behavior checks:

```bash
python -m py_compile src/drone_fsm/drone_fsm/*.py
pytest -q src/drone_fsm/test/test_model.py \
  src/drone_fsm/test/test_fly.py src/drone_fsm/test/test_px4.py
```
