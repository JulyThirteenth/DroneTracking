#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim_utils import ned_to_enu, load_waypoints_ned, generate_scene, generate_waypoint


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(
        self,
        *,
        simulation_app,
        task_path: Path | None,
        waypoints_ned: list[np.ndarray],
        scene_file: Path | None = None,
    ):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        import carb
        import omni.timeline
        import omni.usd
        from omni.isaac.core.world import World
        import isaacsim.core.utils.prims as prim_utils
        from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
        from pegasus.simulator.logic.backends.px4_mavlink_backend import (
            PX4MavlinkBackend,
            PX4MavlinkBackendConfig,
        )
        from pegasus.simulator.logic.vehicles.multirotor import (
            Multirotor,
            MultirotorConfig,
        )
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

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
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

        spawn_pos_enu = [0.0, 0.0, 0.07]
        if waypoints_ned:
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

        # Add PRE scene props (middle wall + gates) matching `examples/pre/env_utils.py`.
        stage = omni.usd.get_context().get_stage()
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
            self._carb.log_warn(
                f"No waypoints found at {task_path}; skipping waypoint markers."
            )

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

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
    if not tasks_dir.exists():
        return []
    task_files = [
        p for p in tasks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
    ]
    task_files.sort(key=lambda p: p.name)
    return task_files


def _discover_scene_files(scenes_dir: Path) -> list[Path]:
    if not scenes_dir.exists():
        return []
    scene_files = [
        p for p in scenes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
    ]
    scene_files.sort(key=lambda p: p.name)
    return scene_files


def _prompt_task_index(task_files: list[Path], *, tasks_dir: Path) -> int:
    print(f"Tasks dir: {tasks_dir}")
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
        print(f"[{i}] {path.name}")


def _select_task_file(tasks_dir: Path, task_index: int | None) -> Path | None:
    task_files = _discover_task_files(tasks_dir)
    if not task_files:
        print(f"No task files found under: {tasks_dir}")
        return None

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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Index of task file under tasks/ (interactive if omitted).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available task files under tasks/ and exit.",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=None,
        help="Index of scene file under scenes/ (interactive if omitted).",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available scene files under scenes/ and exit.",
    )
    args, _ = parser.parse_known_args()

    pre_root = Path(__file__).resolve().parent.parent
    tasks_dir = pre_root / "tasks"
    scenes_dir = pre_root / "scenes"

    if args.list_tasks:
        task_files = _discover_task_files(tasks_dir)
        if not task_files:
            print(f"No task files found under: {tasks_dir}")
        else:
            _print_indexed_files(task_files, title=f"Tasks dir: {tasks_dir}")

    if args.list_scenes:
        scene_files = _discover_scene_files(scenes_dir)
        if not scene_files:
            print(f"No scene files found under: {scenes_dir}")
        else:
            _print_indexed_files(scene_files, title=f"Scenes dir: {scenes_dir}")

    if args.list_tasks or args.list_scenes:
        return

    scene_path = _select_scene_file(
        scenes_dir=scenes_dir,
        scene_index=args.scene_index,
    )
    task_path = _select_task_file(tasks_dir, args.task_index)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    waypoints_ned = load_waypoints_ned(task_path) if task_path is not None else []
    pg_app = PegasusApp(
        simulation_app=simulation_app,
        task_path=task_path,
        waypoints_ned=waypoints_ned,
        scene_file=scene_path,
    )

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
