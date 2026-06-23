#!/usr/bin/env python3
"""Launch a Pegasus quadrotor in a text-defined Isaac Sim scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_base import BasePegasusApp
from sim_utils import (
    draw_waypoints,
    load_scene,
    load_waypoints_ned,
    ned_to_enu,
    select_text_file,
)

SPAWN_Z_OFFSET_M = 0.07


class TextSceneApp(BasePegasusApp):
    def __init__(self, *, simulation_app, scene_path: Path, waypoints_ned):
        self._scene_path = scene_path
        self._waypoints_ned = list(waypoints_ned)

        spawn = ned_to_enu(self._waypoints_ned[0])
        spawn[2] += SPAWN_Z_OFFSET_M
        super().__init__(simulation_app=simulation_app, spawn_position=spawn.tolist())

    def build_scene(self) -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        load_scene(stage, self._scene_path)
        draw_waypoints(self._waypoints_ned)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=root / "sim" / "scenes",
        help="Scene .txt file or directory (default: sim/scenes).",
    )
    parser.add_argument("--scene-index", type=int)
    parser.add_argument(
        "--waypoints",
        type=Path,
        default=root / "src" / "drone_ref" / "waypoints",
        help="Waypoint .txt file or directory.",
    )
    parser.add_argument("--waypoint-index", type=int)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--list-waypoints", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    if args.list_scenes:
        select_text_file(
            args.scene,
            index=args.scene_index,
            list_only=True,
            label="scene",
        )
        return

    if args.list_waypoints:
        select_text_file(
            args.waypoints,
            index=args.waypoint_index,
            list_only=True,
            label="waypoint",
        )
        return

    scene_path = select_text_file(
        args.scene,
        index=args.scene_index,
        list_only=False,
        label="scene",
    )
    waypoint_path = select_text_file(
        args.waypoints,
        index=args.waypoint_index,
        list_only=False,
        label="waypoint",
    )
    assert scene_path is not None and waypoint_path is not None

    waypoints = load_waypoints_ned(waypoint_path)
    if not waypoints:
        raise ValueError(f"No valid waypoint rows in {waypoint_path}")

    print(f"Scene: {scene_path}")
    print(f"Waypoints: {waypoint_path}")

    # Isaac modules must only be imported after SimulationApp is constructed.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    TextSceneApp(
        simulation_app=simulation_app,
        scene_path=scene_path,
        waypoints_ned=waypoints,
    ).run()


if __name__ == "__main__":
    main()
