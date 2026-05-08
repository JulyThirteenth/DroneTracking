#!/usr/bin/env python
"""
Multi-vehicle Isaac Sim app for drone_racing.

Startup flow (interactive):
1) Select scene file from `scenes/`
2) Select number of vehicles
3) Select one waypoint file for each vehicle from `plan2track/waypoints/`
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim_utils import ned_to_enu, load_waypoints_ned, generate_scene, generate_waypoint


class PegasusApp:
    def __init__(
        self,
        *,
        simulation_app,
        scene_file: Path,
        waypoints_by_vehicle: list[list[np.ndarray]],
        task_paths: list[Path],
    ):
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

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        for vehicle_id, waypoints_ned in enumerate(waypoints_by_vehicle):
            config_multirotor = MultirotorConfig()
            mavlink_config = PX4MavlinkBackendConfig(
                {
                    "vehicle_id": vehicle_id,
                    "px4_autolaunch": True,
                    "px4_dir": self.pg.px4_path,
                    "px4_vehicle_model": self.pg.px4_default_airframe,
                }
            )
            config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

            spawn_pos_enu = [float(vehicle_id), 0.0, 0.07]
            if waypoints_ned:
                first_wp_enu = ned_to_enu(waypoints_ned[0])
                spawn_pos_enu = (first_wp_enu + np.array([0.0, 0.0, 0.07])).tolist()

            Multirotor(
                "/World/quadrotor",
                ROBOTS["Iris"],
                vehicle_id,
                spawn_pos_enu,
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor,
            )

        self.world.reset()

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
        generate_scene(stage, env_root="/World/layout/PreEnv", scene_file=scene_file)

        for vehicle_id, waypoints_ned in enumerate(waypoints_by_vehicle):
            if waypoints_ned:
                generate_waypoint(
                    stage,
                    waypoints_ned=waypoints_ned,
                    parent_path=f"/World/layout/Waypoints/veh_{vehicle_id:02d}",
                )
            else:
                self._carb.log_warn(
                    f"Vehicle {vehicle_id}: no waypoints found at {task_paths[vehicle_id]}"
                )

        self.stop_sim = False

    def run(self):
        self.timeline.play()
        while self._simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        self._carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self._simulation_app.close()


def _discover_scene_files(scenes_dir: Path) -> list[Path]:
    if not scenes_dir.exists():
        return []
    files = [p for p in scenes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    files.sort(key=lambda p: p.name)
    return files


def _discover_task_files(tasks_dir: Path) -> list[Path]:
    if not tasks_dir.exists():
        return []
    files = [p for p in tasks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    files.sort(key=lambda p: p.name)
    return files


def _print_indexed_files(files: list[Path], *, title: str) -> None:
    print(title)
    for i, path in enumerate(files):
        print(f"[{i}] {path.name}")


def _prompt_index(*, count: int, prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please input an integer.")
            continue
        if idx < 0 or idx >= count:
            print(f"Index out of range: {idx} (0..{count - 1})")
            continue
        return idx


def _prompt_vehicle_count() -> int:
    while True:
        raw = input("Select number of vehicles: ").strip()
        try:
            n = int(raw)
        except ValueError:
            print("Please input an integer.")
            continue
        if n <= 0:
            print("Vehicle count must be > 0.")
            continue
        return n


def main():
    pre_root = Path(__file__).resolve().parent.parent
    scenes_dir = pre_root / "scenes"
    tasks_dir = pre_root / "plan2track" / "waypoints"

    scene_files = _discover_scene_files(scenes_dir)
    if not scene_files:
        raise FileNotFoundError(f"No scene files found under: {scenes_dir}")
    _print_indexed_files(scene_files, title=f"Scenes dir: {scenes_dir}")
    scene_index = _prompt_index(
        count=len(scene_files),
        prompt="Select scene index: ",
    )
    scene_path = scene_files[scene_index]

    vehicle_count = _prompt_vehicle_count()

    task_files = _discover_task_files(tasks_dir)
    if not task_files:
        raise FileNotFoundError(f"No task files found under: {tasks_dir}")
    _print_indexed_files(task_files, title=f"Waypoints dir: {tasks_dir}")

    task_paths: list[Path] = []
    for vehicle_id in range(vehicle_count):
        idx = _prompt_index(
            count=len(task_files),
            prompt=f"Select task index for vehicle {vehicle_id}: ",
        )
        task_paths.append(task_files[idx])

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    waypoints_by_vehicle = [load_waypoints_ned(p) for p in task_paths]

    pg_app = PegasusApp(
        simulation_app=simulation_app,
        scene_file=scene_path,
        waypoints_by_vehicle=waypoints_by_vehicle,
        task_paths=task_paths,
    )
    pg_app.run()


if __name__ == "__main__":
    main()
