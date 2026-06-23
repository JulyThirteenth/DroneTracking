# drone_log

Independent CSV logger for `drone_ref` + `drone_cnt` MPC or MPCC flights. It
never publishes control input. A tick is written when the controller publishes
`/drone_cnt/vehicle_rates_setpoint`.

Each run creates:

```text
logs/<run_name>/
  ticks.csv          state, yaw reference and final CTBR at each output tick
  events.csv         FSM state transitions
  references.jsonl   complete NavPath snapshots
  meta.json          mode and topic configuration
```

`ticks.csv` uses ENU for position, velocity and acceleration. `thrust_cmd` is
positive normalized thrust. MPC logs the first point of its local reference as
`ref_*`; MPCC leaves `ref_*` empty because its complete global path is stored in
`references.jsonl` and must be projected offline.

Build and run:

```bash
pixi run colcon --log-base log_pixi build --symlink-install \
  --build-base build_pixi --install-base install_pixi \
  --packages-select drone_log
source install_pixi/setup.bash

ros2 launch drone_log log.launch.py \
  controller_mode:=mpc log_directory:=logs run_name:=mpc_line_01

ros2 launch drone_log log.launch.py \
  controller_mode:=mpcc log_directory:=logs run_name:=mpcc_line_01
```

Start the logger before `execute`; stop it with Ctrl-C after landing.
