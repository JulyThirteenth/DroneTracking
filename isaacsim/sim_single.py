#!/usr/bin/env python
from pathlib import Path
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation

from sim_utils import (
    behavior1k_resource_root,
    behavior1k_scene_name,
    discover_behavior1k_scene_files,
    generate_scene,
    generate_waypoint,
    load_waypoints_ned,
    ned_to_enu,
    sample_behavior1k_spawn,
)

CAMERA_PRIM_PATH = "/World/quadrotor/body/camera_fpv"
CAMERA_RESOLUTION = (640, 480)


class PegasusApp:
    def __init__(
        self,
        *,
        simulation_app,
        waypoints_ned: list[np.ndarray],
        scene_file: Path | None = None,
        environment_path: Path | str | None = None,
        spawn_position: list[float] | None = None,
    ):
        import carb
        import omni.timeline
        import omni.usd
        from omni.isaac.core.world import World
        import omni.isaac.core.utils.prims as prim_utils
        from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
        from pegasus.simulator.logic.backends.px4_mavlink_backend import (
            PX4MavlinkBackend,
            PX4MavlinkBackendConfig,
        )
        from pegasus.simulator.logic.vehicles.multirotor import (
            Multirotor,
            MultirotorConfig,
        )
        from pegasus.simulator.logic.dynamics import LinearDrag
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

        self._simulation_app = simulation_app
        self._carb = carb
        self._prim_utils = prim_utils

        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch either an external USD environment or one of the built-in worlds.
        self.pg.load_environment(
            str(environment_path)
            if environment_path is not None
            else SIMULATION_ENVIRONMENTS["Flat Plane"]
        )

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        config_multirotor.drag = LinearDrag([0.1, 0.1, 0.0])
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,  # CHANGE this line to 'iris' if using PX4 version bellow v1.14
            }
        )
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        spawn_pos_enu = (
            list(spawn_position) if spawn_position is not None else [0.0, 0.0, 0.07]
        )
        if spawn_position is None and waypoints_ned:
            first_wp_enu = ned_to_enu(waypoints_ned[0])
            spawn_pos_enu = (first_wp_enu + np.array([0.0, 0.0, 0.07])).tolist()

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            spawn_pos_enu,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Add the FPV camera to the drone
        self._add_fpv_camera()

        # Publish camera data to ROS2
        self._build_camera_publishers()

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Set scene light intensity (environment loading can be asynchronous)
        light_path = "/World/layout/DistantLight"
        if self._prim_utils.is_prim_path_valid(light_path):
            self._prim_utils.set_prim_attribute_value(
                light_path, "inputs:intensity", 10000
            )
        else:
            self._carb.log_warn(
                f"Light prim not found at {light_path}; cannot set intensity."
            )

        stage = omni.usd.get_context().get_stage()
        if scene_file is not None:
            generate_scene(
                stage,
                env_root="/World/layout/PreEnv",
                scene_file=scene_file,
            )

        # Create transparent green balls for the selected task waypoints.
        if waypoints_ned:
            generate_waypoint(
                stage,
                waypoints_ned=waypoints_ned,
                parent_path="/World/layout/Waypoints",
            )
        else:
            self._carb.log_warn("No waypoints found; skipping waypoint markers.")

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _add_fpv_camera(self):
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        from omni.isaac.sensor import Camera

        desired_fov_degrees = 90.0
        horizontal_aperture = 24.0
        focal_length = horizontal_aperture / (
            2 * math.tan(math.radians(desired_fov_degrees) / 2.0)
        )
        vertical_aperture = horizontal_aperture * (
            CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0]
        )

        print("-" * 50)
        print(f"Camera Settings for {desired_fov_degrees}-degree FOV:")
        print(f"  - Focal Length: {focal_length:.2f} mm")
        print(f"  - Horizontal Aperture: {horizontal_aperture:.2f} mm")
        print(f"  - Vertical Aperture: {vertical_aperture:.2f} mm")
        print("-" * 50)

        prim_utils.create_prim(
            prim_path=CAMERA_PRIM_PATH,
            prim_type="Camera",
            attributes={
                "focalLength": focal_length,
                "horizontalAperture": horizontal_aperture,
                "verticalAperture": vertical_aperture,
                "clippingRange": (0.2, 5.0),
            },
        )

        self.camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            resolution=CAMERA_RESOLUTION,
        )
        self.camera.initialize()

        cam_pos = np.array([0.1, 0.0, 0.0])
        cam_orientation = rot_utils.euler_angles_to_quats(
            np.array([0.0, 0.0, 0.0]), degrees=True
        )
        self.camera.set_local_pose(translation=cam_pos, orientation=cam_orientation)

    def _build_camera_publishers(self):
        """Publish the FPV camera images to ROS2 from a dedicated render product."""
        import omni.replicator.core as rep

        self._camera_render_product = rep.create.render_product(
            CAMERA_PRIM_PATH, CAMERA_RESOLUTION, name="fpv_camera"
        )
        self._camera_writers = []

        rgb_writer = rep.writers.get("LdrColorSDROS2PublishImage")
        rgb_writer.initialize(topicName="rgb", frameId="drone_fpv_camera", queueSize=1)
        rgb_writer.attach([self._camera_render_product])
        self._camera_writers.append(rgb_writer)

        depth_writer = rep.writers.get("DistanceToImagePlaneSDROS2PublishImage")
        depth_writer.initialize(
            topicName="depth", frameId="drone_fpv_camera", queueSize=1
        )
        depth_writer.attach([self._camera_render_product])
        self._camera_writers.append(depth_writer)

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while self._simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        self._carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self._simulation_app.close()


def _discover_task_files(tasks_dir: Path) -> list[Path]:
    return sorted(
        p for p in tasks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
    )


def _discover_scene_files(scenes_dir: Path) -> list[Path]:
    return sorted(
        p for p in scenes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
    )


def _prompt_task_index(task_files: list[Path], *, tasks_dir: Path) -> int:
    print(f"Waypoints dir: {tasks_dir}")
    for i, path in enumerate(task_files):
        print(f"[{i}] {path.name}")
    raw = input("Select task index: ").strip()
    return int(raw)


def _prompt_scene_index(scene_files: list[Path], *, scenes_dir: Path) -> int:
    print(f"Scenes dir: {scenes_dir}")
    for i, path in enumerate(scene_files):
        print(f"[{i}] {path.name}")
    raw = input("Select scene index: ").strip()
    return int(raw)


def _print_indexed_files(files: list[Path], *, title: str) -> None:
    print(title)
    for i, path in enumerate(files):
        label = (
            behavior1k_scene_name(path) if path.suffix.lower() == ".usd" else path.name
        )
        print(f"[{i}] {label}")


def _select_task_file(tasks_dir: Path, task_index: int | None) -> Path | None:
    task_files = _discover_task_files(tasks_dir)
    if task_index is None:
        task_index = _prompt_task_index(task_files, tasks_dir=tasks_dir)

    if task_index < 0 or task_index >= len(task_files):
        raise ValueError(
            f"task_index out of range: {task_index} (0..{len(task_files) - 1})"
        )
    return task_files[task_index]


def _select_scene_file(
    *,
    scenes_dir: Path,
    scene_index: int | None,
) -> Path:
    scene_files = _discover_scene_files(scenes_dir)
    if not scene_files:
        raise FileNotFoundError(f"No scene files found under: {scenes_dir}")

    if scene_index is None:
        scene_index = _prompt_scene_index(scene_files, scenes_dir=scenes_dir)

    if scene_index < 0 or scene_index >= len(scene_files):
        raise ValueError(
            f"scene_index out of range: {scene_index} (0..{len(scene_files) - 1})"
        )
    return scene_files[scene_index]


def _select_behavior1k_scene_file(
    *,
    behavior1k_root: Path,
    scene_index: int | None,
) -> Path:
    scene_files = discover_behavior1k_scene_files(behavior1k_root)
    if not scene_files:
        raise FileNotFoundError(f"No behavior1k scene files found under: {behavior1k_root}")

    if scene_index is None:
        _print_indexed_files(
            scene_files,
            title=f"behavior1k scenes: {behavior1k_root / 'scenes.txt'}",
        )
        scene_index = int(input("Select behavior1k scene index: ").strip())

    if scene_index < 0 or scene_index >= len(scene_files):
        raise ValueError(
            f"behavior1k scene_index out of range: {scene_index} (0..{len(scene_files) - 1})"
        )
    return scene_files[scene_index]


def _select_scene_source(
    *,
    scenes_dir: Path,
    behavior1k_root: Path,
    scene_index: int | None,
) -> tuple[str, Path]:
    scene_files = _discover_scene_files(scenes_dir)

    while True:
        index = scene_index
        if index is None:
            print(f"Scenes dir: {scenes_dir}")
            for i, path in enumerate(scene_files):
                print(f"[{i}] {path.name}")
            print(f"[{len(scene_files)}] behavior1k")
            index = int(input("Select scene index: ").strip())

        if 0 <= index < len(scene_files):
            return "pre", scene_files[index]

        if index == len(scene_files):
            if not behavior1k_root.exists():
                print(f"No behavior1k directory found: {behavior1k_root}")
                if scene_index is None:
                    continue
                raise FileNotFoundError(str(behavior1k_root))
            try:
                return "behavior1k", _select_behavior1k_scene_file(
                    behavior1k_root=behavior1k_root,
                    scene_index=None,
                )
            except FileNotFoundError as exc:
                print(str(exc))
                if scene_index is None:
                    continue
                raise

        raise ValueError(f"scene_index out of range: {index} (0..{len(scene_files)})")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Index of waypoint file under plan2track/waypoints/ (interactive if omitted).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available waypoint files under plan2track/waypoints/ and exit.",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=None,
        help="Index of selected scene source (interactive if omitted).",
    )
    parser.add_argument(
        "--scene-source",
        choices=("pre", "behavior1k"),
        default=None,
        help="Scene source: DroneTracking txt scenes or behavior1k USD scenes.",
    )
    parser.add_argument(
        "--spawn-clearance",
        type=float,
        default=1.0,
        help="Minimum behavior1k occupancy-map clearance for random spawn.",
    )
    parser.add_argument(
        "--spawn-seed",
        type=int,
        default=10,
        help="Random seed for behavior1k occupancy-map spawn.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available scene files under scenes/ and exit.",
    )
    args, _ = parser.parse_known_args()

    pre_root = Path(__file__).resolve().parent.parent
    tasks_dir = pre_root / "plan2track" / "waypoints"
    scenes_dir = pre_root / "scenes"
    behavior1k_root = behavior1k_resource_root()

    if args.list_tasks:
        task_files = _discover_task_files(tasks_dir)
        if not task_files:
            print(f"No task files found under: {tasks_dir}")
        else:
            _print_indexed_files(task_files, title=f"Waypoints dir: {tasks_dir}")

    if args.list_scenes:
        if args.scene_source == "behavior1k":
            scene_files = discover_behavior1k_scene_files(behavior1k_root)
            title = "Scenes source: behavior1k"
        else:
            scene_files = _discover_scene_files(scenes_dir)
            title = f"Scenes dir: {scenes_dir}"
        if not scene_files:
            print(f"No scene files found for source: {args.scene_source or 'pre'}")
        else:
            _print_indexed_files(scene_files, title=title)
            if args.scene_source is None:
                print(f"[{len(scene_files)}] behavior1k")

    if args.list_tasks or args.list_scenes:
        return

    if args.scene_source == "behavior1k":
        scene_source = "behavior1k"
        selected_scene = _select_behavior1k_scene_file(
            behavior1k_root=behavior1k_root, scene_index=args.scene_index
        )
    elif args.scene_source == "pre":
        scene_source = "pre"
        selected_scene = _select_scene_file(
            scenes_dir=scenes_dir, scene_index=args.scene_index
        )
    else:
        scene_source, selected_scene = _select_scene_source(
            scenes_dir=scenes_dir,
            behavior1k_root=behavior1k_root,
            scene_index=args.scene_index,
        )

    if scene_source == "behavior1k":
        os.environ["MDL_USER_PATH"] = str(behavior1k_root / "MTL")
        scene_path = None
        task_path = None
        environment_path = selected_scene
        spawn_position = sample_behavior1k_spawn(
            selected_scene,
            clearance_m=float(args.spawn_clearance),
            seed=int(args.spawn_seed),
        )
        print(f"Selected behavior1k scene: {behavior1k_scene_name(selected_scene)}")
        print(f"Sampled spawn ENU: {spawn_position}")
    else:
        scene_path = selected_scene
        task_path = _select_task_file(tasks_dir, args.task_index)
        environment_path = None
        spawn_position = None

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    waypoints_ned = load_waypoints_ned(task_path) if task_path is not None else []
    pg_app = PegasusApp(
        simulation_app=simulation_app,
        waypoints_ned=waypoints_ned,
        scene_file=scene_path,
        environment_path=environment_path,
        spawn_position=spawn_position,
    )

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
