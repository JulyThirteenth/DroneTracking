#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from sim_base import BasePegasusApp
from sim_utils import (
    generate_scene,
    generate_waypoint,
    load_waypoints_ned,
    ned_to_enu,
    select_txt_scene,
    select_waypoint_file,
)


SPAWN_Z_OFFSET_M = 0.07


class TxtPegasusApp(BasePegasusApp):
    def __init__(
        self,
        *,
        simulation_app,
        scene_path: Path,
        waypoints_ned,
        spawn_position: list[float],
    ):
        self._waypoints_ned = list(waypoints_ned)
        super().__init__(
            simulation_app=simulation_app,
            scene_path=scene_path,
            spawn_position=spawn_position,
        )

    def _load_scene(self, scene_path: Path):
        from pegasus.simulator.params import SIMULATION_ENVIRONMENTS

        _ = scene_path
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Flat Plane"], "/World/layout")

    def _post_reset(self, scene_path: Path):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        generate_scene(stage, env_root="/World/layout/PreEnv", scene_file=scene_path)
        generate_waypoint(stage, waypoints_ned=self._waypoints_ned)


def spawn_from_waypoints(waypoints_ned) -> list[float]:
    waypoints_ned = list(waypoints_ned)
    if not waypoints_ned:
        raise ValueError("waypoint file has no valid waypoint rows")

    first_wp_enu = ned_to_enu(waypoints_ned[0])
    first_wp_enu[2] += SPAWN_Z_OFFSET_M
    return first_wp_enu.tolist()


def main():
    import argparse

    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=script_dir / "scenes" / "scenes_txt",
        help="Txt scene file or directory containing txt scene files.",
    )
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=script_dir.parent / "plan2track" / "waypoints",
        help="Waypoint txt file or directory containing waypoint txt files.",
    )
    parser.add_argument("--waypoint-index", type=int, default=None)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--list-waypoints", action="store_true")
    args, _ = parser.parse_known_args()

    if args.list_scenes:
        select_txt_scene(
            args.scene,
            scene_index=args.scene_index,
            list_scenes=True,
            script_dir=script_dir,
        )
        return

    if args.list_waypoints:
        select_waypoint_file(
            args.waypoints,
            waypoint_index=args.waypoint_index,
            list_waypoints=True,
            script_dir=script_dir,
        )
        return

    scene_path = select_txt_scene(
        args.scene,
        scene_index=args.scene_index,
        list_scenes=False,
        script_dir=script_dir,
    )

    waypoint_path = select_waypoint_file(
        args.waypoints,
        waypoint_index=args.waypoint_index,
        list_waypoints=False,
        script_dir=script_dir,
    )

    waypoints_ned = load_waypoints_ned(waypoint_path)
    spawn_position = spawn_from_waypoints(waypoints_ned)
    print(f"Selected txt scene: {scene_path.name}")
    print(f"Scene path: {scene_path}")
    print(f"Selected waypoints: {waypoint_path.name}")
    print(f"Waypoints path: {waypoint_path}")
    print(f"Spawn ENU: {spawn_position}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    TxtPegasusApp(
        simulation_app=simulation_app,
        scene_path=scene_path,
        waypoints_ned=waypoints_ned,
        spawn_position=spawn_position,
    ).run()


if __name__ == "__main__":
    main()
