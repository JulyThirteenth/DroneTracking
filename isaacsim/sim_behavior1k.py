#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import os

from sim_base import BasePegasusApp
from sim_utils import spawn_position_from_arg, select_behavior1k_scene


class Behavior1kPegasusApp(BasePegasusApp):
    def _load_scene(self, scene_path: Path):
        prim_path = "/World/layout"
        prim = self.world.stage.DefinePrim(prim_path, "Xform")
        success = prim.GetReferences().AddReference(str(scene_path), "/World")
        if not success:
            raise RuntimeError(f"failed to reference scene usd: {scene_path}")


def main():
    import argparse

    script_dir = Path(__file__).resolve().parent
    behavior1k_root = script_dir / "scenes" / "behavior1k"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=behavior1k_root,
        help="Behavior1k root directory containing scenes.txt.",
    )
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument(
        "--spawn",
        "--spawn-point",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Drone spawn point in Isaac ENU frame.",
    )
    parser.add_argument("--list-scenes", action="store_true")
    args, _ = parser.parse_known_args()

    scene_path, mdl_user_path = select_behavior1k_scene(
        args.scene_root,
        scene_index=args.scene_index,
        list_scenes=bool(args.list_scenes),
        script_dir=script_dir,
    )
    if scene_path is None:
        return

    os.environ["MDL_USER_PATH"] = str(mdl_user_path)
    spawn_position = spawn_position_from_arg(args.spawn)
    print(f"Selected behavior1k scene: {scene_path.parent.name}")
    print(f"Scene path: {scene_path}")
    print(f"Spawn ENU: {spawn_position}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    Behavior1kPegasusApp(
        simulation_app=simulation_app,
        scene_path=scene_path,
        spawn_position=spawn_position,
    ).run()


if __name__ == "__main__":
    main()
