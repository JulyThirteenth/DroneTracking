#!/usr/bin/env python
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation

from sim_utils import ned_to_enu, load_waypoints_ned, generate_scene, generate_waypoint

ROS_CAMERA_GRAPH_PATH = "/ROSCameraGraph"
CAMERA_PRIM_PATH = "/World/quadrotor/body/camera_fpv"
CAMERA_RESOLUTION = (640, 480)


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

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

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

        # Add the FPV camera to the drone
        self._add_fpv_camera()

        # Build the ROS graph to publish camera data
        self._build_camera_graph()

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

    def _add_fpv_camera(self):
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        from omni.isaac.sensor import Camera

        """
        Attaches a camera to the drone and sets its intrinsic properties for a specific FOV.
        """
        # ==================== MODIFIED CODE START ====================

        # Define camera intrinsic properties
        # Example: To achieve a 90-degree horizontal FOV, the focal length should be half the horizontal aperture.
        # FOV = 2 * atan(aperture / (2 * focal_length))

        # Let's set a desired horizontal FOV in degrees
        desired_fov_degrees = 90.0

        # We can fix the horizontal aperture (sensor width) to a common value, e.g., 24mm
        horizontal_aperture = 24.0

        # Calculate the required focal length
        focal_length = horizontal_aperture / (
            2 * math.tan(math.radians(desired_fov_degrees) / 2.0)
        )

        # Calculate the vertical aperture based on the image aspect ratio
        vertical_aperture = horizontal_aperture * (
            CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0]
        )

        print("-" * 50)
        print(f"Camera Settings for {desired_fov_degrees}-degree FOV:")
        print(f"  - Focal Length: {focal_length:.2f} mm")
        print(f"  - Horizontal Aperture: {horizontal_aperture:.2f} mm")
        print(f"  - Vertical Aperture: {vertical_aperture:.2f} mm")
        print("-" * 50)

        # Create the camera prim with all its properties (intrinsics) defined at creation.
        # This is a more robust way to set these values than using the high-level Camera class alone.
        camera_prim = prim_utils.create_prim(
            prim_path=CAMERA_PRIM_PATH,
            prim_type="Camera",
            attributes={
                "focalLength": focal_length,
                "horizontalAperture": horizontal_aperture,
                "verticalAperture": vertical_aperture,
                "clippingRange": (0.2, 5.0),  # Near and far clipping planes
            },
        )

        # Now, apply the high-level Isaac Sim Camera API to this prim for easy control
        self.camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            # The resolution of the output image is set here.
            # This is separate from the FOV.
            resolution=CAMERA_RESOLUTION,
        )
        self.camera.initialize()

        # Position and orient the camera on the drone
        cam_pos = np.array([0.1, 0.0, 0.0])
        cam_orientation = rot_utils.euler_angles_to_quats(
            np.array([0.0, 0.0, 0.0]), degrees=True
        )
        self.camera.set_local_pose(translation=cam_pos, orientation=cam_orientation)

        # ===================== MODIFIED CODE END =====================

    def _build_camera_graph(self):
        """
        Creates the OmniGraph for streaming the FPV camera's view via ROS2.
        This remains unchanged, as it correctly uses the camera prim defined above.
        """
        import omni.graph.core as og
        import usdrt.Sdf

        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": ROS_CAMERA_GRAPH_PATH, "evaluator_name": "push"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                    # 2. 设置分辨率（新增节点）
                    (
                        "setViewportResolution",
                        "isaacsim.core.nodes.IsaacSetViewportResolution",
                    ),
                    (
                        "getViewportRenderProduct",
                        "isaacsim.core.nodes.IsaacGetViewportRenderProduct",
                    ),
                    ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                keys.CONNECT: [
                    # 1. 执行流
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    (
                        "createViewport.outputs:execOut",
                        "setViewportResolution.inputs:execIn",
                    ),
                    (
                        "setViewportResolution.outputs:execOut",
                        "getViewportRenderProduct.inputs:execIn",
                    ),
                    (
                        "getViewportRenderProduct.outputs:execOut",
                        "setCamera.inputs:execIn",
                    ),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    # 2. 渲染流
                    (
                        "getViewportRenderProduct.outputs:renderProductPath",
                        "setCamera.inputs:renderProductPath",
                    ),
                    (
                        "getViewportRenderProduct.outputs:renderProductPath",
                        "cameraHelperRgb.inputs:renderProductPath",
                    ),
                    (
                        "getViewportRenderProduct.outputs:renderProductPath",
                        "cameraHelperDepth.inputs:renderProductPath",
                    ),
                ],
                keys.SET_VALUES: [
                    ("createViewport.inputs:name", "fpv_camera_viewport"),
                    (
                        "setViewportResolution.inputs:viewport",
                        "fpv_camera_viewport",
                    ),  # token
                    (
                        "getViewportRenderProduct.inputs:viewport",
                        "fpv_camera_viewport",
                    ),  # token
                    ("setViewportResolution.inputs:width", 640),
                    ("setViewportResolution.inputs:height", 480),
                    ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_PRIM_PATH)]),
                    ("cameraHelperRgb.inputs:frameId", "drone_fpv_camera"),
                    ("cameraHelperRgb.inputs:topicName", "rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperDepth.inputs:frameId", "drone_fpv_camera"),
                    ("cameraHelperDepth.inputs:topicName", "depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ],
            },
        )

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
