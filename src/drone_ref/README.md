# Drone Reference

`drone_ref` turns a geometric ENU path into the reference messages consumed by
the MPC or MPCC controller. It also owns the controller-required vehicle pose
and yaw-command topics for whichever reference mode is running.

## Nodes and outputs

Only one reference mode should normally run at a time.

| Node | Reference output | Publication policy |
| --- | --- | --- |
| `drone_ref_mpc` | `/tracking/ref_traj_path` | N+1 points every timer tick after vehicle state arrives |
| `drone_ref_mpcc` | `/tracking/path` | Complete path once per path revision, even before vehicle state arrives |

Both nodes publish these mandatory controller interfaces after receiving PX4
vehicle state:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/tracking/vehicle_pose` | `geometry_msgs/PoseStamped` | Current vehicle pose in ENU/map |
| `/planning/yaw_cmd_enu` | `std_msgs/Float32` | Fixed or path-tangent yaw command in ENU radians |

Do not start MPC and MPCC reference nodes simultaneously with the default
configuration. They would both publish `vehicle_pose` and `yaw_cmd_enu`.

## Frames and path sources

PX4 `VehicleLocalPosition` is interpreted as NED and converted as:

```text
x_enu = y_ned
y_enu = x_ned
z_enu = -z_ned
yaw_enu = pi/2 - yaw_ned
```

Waypoint text files are interpreted as NED. A runtime `nav_msgs/Path` received
on `/planning/path` is already expected to be ENU in the configured `frame_id`.

Waypoint rows may use spaces or commas:

```text
# north east down
0.0 0.0 -1.0
1.0, 0.5, -1.0
```

`origin_mode` controls the NED origin subtraction:

- `fixed`: subtract the configured/default `(0, 0, 0)` origin;
- `first`: make the first waypoint the full xyz origin;
- `first_xy`: make the first waypoint the horizontal origin and retain NED z.

Consecutive duplicate points are removed. A valid path must contain at least
two distinct finite points shaped `(M, 3)`.

## Reference behavior

MPC projects the current ENU position onto the path and samples:

```text
N + 1 points
sample spacing = reference_speed * mpc_dt
```

If no valid path exists after vehicle state arrives, MPC publishes an N+1-point
position hold at the current position.

MPCC publishes the complete geometric path with reliable transient-local QoS.
A late subscriber receives the most recent path. Path publication does not wait
for `VehicleLocalPosition`; pose and yaw publication still requires vehicle
state.

Yaw follows the ENU path tangent at `progress + yaw_lookahead`. With
`fixed_yaw: true`, the configured `initial_yaw` is published instead.

## Configuration

Both node sections are stored in `config/drone_ref.yaml`.

Common parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `frame_id` | `map` | ROS output frame |
| `path_file` | `waypoints/line_waypoint.txt` | Absolute or package-share-relative NED path |
| `origin_mode` | `first_xy` | Waypoint origin policy |
| `loop` | `false` | Close and wrap the path |
| `publish_period` | `0.02` | Reference/yaw timer period in seconds |
| `fixed_yaw` | `false` | Use `initial_yaw` instead of path tangent |
| `initial_yaw` | `0.0` | Initial/fixed ENU yaw in radians |
| `yaw_lookahead` | `0.2` | Tangent lookahead distance in metres |
| `vehicle_position_topic` | `/fmu/out/vehicle_local_position` | PX4 NED vehicle input |
| `planning_path_topic` | `/planning/path` | Runtime ENU path input |
| `vehicle_pose_topic` | `/tracking/vehicle_pose` | Mandatory ENU pose output |
| `yaw_topic` | `/planning/yaw_cmd_enu` | Mandatory ENU yaw output |

MPC-only parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `horizon` | `15` | Controller horizon N |
| `mpc_dt` | `0.1` | Controller time step in seconds |
| `reference_speed` | `3.0` | Path sampling speed in m/s |
| `output_topic` | `/tracking/ref_traj_path` | MPC trajectory output |

MPCC uses `output_topic: /tracking/path`.

## Build

From the repository root:

```bash
pixi shell
colcon build \
  --build-base build_pixi \
  --install-base install_pixi \
  --symlink-install \
  --packages-select drone_ref
source install_pixi/setup.bash
```

## Start MPC reference

```bash
ros2 launch drone_ref mpc.launch.py \
  path_file:=waypoints/comp_waypoint_1.txt \
  reference_speed:=1.25
```

Use an absolute waypoint path when it is outside the package share directory.

## Start MPCC reference

```bash
ros2 launch drone_ref mpcc.launch.py \
  path_file:=waypoints/comp_waypoint_1.txt
```

Override the complete parameter file:

```bash
ros2 launch drone_ref mpc.launch.py \
  config_file:=/absolute/path/to/drone_ref.yaml
```

## Inspect outputs

```bash
ros2 topic echo /tracking/vehicle_pose
ros2 topic echo /planning/yaw_cmd_enu
ros2 topic echo /tracking/ref_traj_path   # MPC
ros2 topic echo /tracking/path            # MPCC
```

## Checks

```bash
python -m py_compile src/drone_ref/drone_ref/*.py
cd src/drone_ref
pytest -q
```

The closest-point calculation currently searches the complete polyline. At a
self-intersection such as a figure-eight, geometrically equal branches remain
ambiguous. Physical simulation should explicitly check that progress does not
jump to a later branch.
