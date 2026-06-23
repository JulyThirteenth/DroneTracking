#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from scipy.spatial.transform import Rotation


CAMERA_PRIM_PATH = "/World/quadrotor/body/camera_fpv"
CAMERA_RESOLUTION = (640, 480)


class BasePegasusApp:
    def __init__(
        self,
        *,
        simulation_app,
        scene_path: Path,
        spawn_position: list[float],
    ):
        import carb
        import omni.timeline
        from omni.isaac.core.world import World
        import omni.isaac.core.utils.prims as prim_utils
        from pegasus.simulator.params import ROBOTS
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

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self._load_scene(scene_path)

        config_multirotor = MultirotorConfig()
        config_multirotor.drag = LinearDrag([0.1, 0.1, 0.0])
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            list(spawn_position),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self._add_fpv_camera()
        self._build_camera_publishers()
        self.world.reset()
        self._post_reset(scene_path)
        self._set_scene_light()
        self.stop_sim = False

    def _load_scene(self, scene_path: Path):
        raise NotImplementedError

    def _post_reset(self, scene_path: Path):
        pass

    def _set_scene_light(self):
        light_path = "/World/layout/DistantLight"
        if self._prim_utils.is_prim_path_valid(light_path):
            self._prim_utils.set_prim_attribute_value(
                light_path, "inputs:intensity", 10000
            )
        else:
            self._carb.log_warn(
                f"Light prim not found at {light_path}; cannot set intensity."
            )

    def _add_fpv_camera(self):
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        from omni.isaac.sensor import Camera

        horizontal_aperture = 24.0
        focal_length = horizontal_aperture / (2 * math.tan(math.radians(90.0) / 2.0))
        vertical_aperture = horizontal_aperture * (
            CAMERA_RESOLUTION[1] / CAMERA_RESOLUTION[0]
        )

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

        self.camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=CAMERA_RESOLUTION)
        self.camera.initialize()
        self.camera.set_local_pose(
            translation=np.array([0.1, 0.0, 0.0]),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0.0, 0.0, 0.0]), degrees=True
            ),
        )

    def _build_camera_publishers(self):
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
        self.timeline.play()
        while self._simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        self._carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self._simulation_app.close()
