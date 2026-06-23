"""Shared Pegasus application setup for DroneTracking simulations."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

CAMERA_PATH = "/World/quadrotor/body/camera_fpv"
CAMERA_RESOLUTION = (640, 480)
SKY_TEXTURE_PATH = "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr"


class BasePegasusApp:
    def __init__(self, *, simulation_app, spawn_position: list[float]) -> None:
        import carb
        import omni.timeline
        from omni.isaac.core.world import World
        from pegasus.simulator.logic.backends.px4_mavlink_backend import (
            PX4MavlinkBackend,
            PX4MavlinkBackendConfig,
        )
        from pegasus.simulator.logic.dynamics import LinearDrag
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
        from pegasus.simulator.logic.vehicles.multirotor import (
            Multirotor,
            MultirotorConfig,
        )
        from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

        self._app = simulation_app
        self._carb = carb
        self._timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_asset(SIMULATION_ENVIRONMENTS["Flat Plane"], "/World/layout")

        px4_dir = Path(os.environ.get("PX4_DIR", self.pg.px4_path)).expanduser()
        px4_binary = px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
        if not px4_binary.is_file():
            raise FileNotFoundError(
                f"PX4 SITL binary not found: {px4_binary}. "
                "Set PX4_DIR or update the Pegasus PX4 path."
            )

        vehicle = MultirotorConfig()
        vehicle.drag = LinearDrag([0.1, 0.1, 0.0])
        backend = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": str(px4_dir),
                "px4_vehicle_model": os.environ.get(
                    "PX4_MODEL", self.pg.px4_default_airframe
                ),
            }
        )
        vehicle.backends = [PX4MavlinkBackend(backend)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            list(spawn_position),
            [0.0, 0.0, 0.0, 1.0],
            config=vehicle,
        )

        self._add_camera()
        self._add_camera_publishers()
        self.world.reset()
        self.build_scene()
        self._set_light()

    def build_scene(self) -> None:
        """Create task-specific scene content after the world reset."""
        raise NotImplementedError

    def _add_camera(self) -> None:
        import omni.isaac.core.utils.numpy.rotations as rotation_utils
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.sensor import Camera

        width, height = CAMERA_RESOLUTION
        aperture = 24.0
        focal_length = aperture / (2.0 * math.tan(math.radians(45.0)))

        prim_utils.create_prim(
            prim_path=CAMERA_PATH,
            prim_type="Camera",
            attributes={
                "focalLength": focal_length,
                "horizontalAperture": aperture,
                "verticalAperture": aperture * height / width,
                "clippingRange": (0.2, 100.0),
            },
        )

        camera = Camera(prim_path=CAMERA_PATH, resolution=CAMERA_RESOLUTION)
        camera.initialize()
        camera.set_local_pose(
            translation=np.array([0.1, 0.0, 0.0]),
            orientation=rotation_utils.euler_angles_to_quats(np.zeros(3), degrees=True),
        )
        self.camera = camera

    def _add_camera_publishers(self) -> None:
        import omni.replicator.core as rep

        product = rep.create.render_product(
            CAMERA_PATH, CAMERA_RESOLUTION, name="fpv_camera"
        )
        self._camera_writers = []

        for writer_name, topic in (
            ("LdrColorSDROS2PublishImage", "rgb"),
            ("DistanceToImagePlaneSDROS2PublishImage", "depth"),
        ):
            writer = rep.writers.get(writer_name)
            writer.initialize(
                topicName=topic,
                frameId="drone_fpv_camera",
                queueSize=1,
            )
            writer.attach([product])
            self._camera_writers.append(writer)

    def _set_light(self) -> None:
        import omni.isaac.core.utils.prims as prim_utils
        from isaacsim.storage.native import get_assets_root_path

        path = "/World/layout/DistantLight"
        if prim_utils.is_prim_path_valid(path):
            prim_utils.set_prim_attribute_value(path, "inputs:intensity", 10000)

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac Sim assets root is unavailable")
        prim_utils.create_prim(
            "/World/Sky",
            "DomeLight",
            attributes={
                "inputs:intensity": 1000.0,
                "inputs:texture:file": assets_root + SKY_TEXTURE_PATH,
            },
        )

    def run(self) -> None:
        self._timeline.play()
        try:
            while self._app.is_running():
                self.world.step(render=True)
        finally:
            self._carb.log_warn("DroneTracking simulation is closing")
            self._timeline.stop()
            self._app.close()
