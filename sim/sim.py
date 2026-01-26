#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
import omni.usd
from omni.isaac.core.world import World

# Used for adding extra lights to the environment
import isaacsim.core.utils.prims as prim_utils

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary scipy and numpy modules
import os.path
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_waypoints_ned(path: Path):
    waypoints = []
    if not path.exists():
        return waypoints

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2]) if len(parts) >= 3 else 0.0
        waypoints.append(np.array([x, y, z], dtype=float))
    return waypoints


def ned_to_enu(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([v[1], v[0], -v[2]], dtype=float)


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
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

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Set scene light intensity (environment loading can be asynchronous)
        light_path = "/World/layout/DistantLight"
        if prim_utils.is_prim_path_valid(light_path):
            prim_utils.set_prim_attribute_value(light_path, "inputs:intensity", 10000)
        else:
            carb.log_warn(f"Light prim not found at {light_path}; cannot set intensity.")

        # Read waypoints from task.txt and create transparent green cylinders (no collisions)
        drone_racing_dir = Path(__file__).resolve().parents[1]
        task_path = drone_racing_dir / "tasks" / "task.txt"
        if not task_path.exists():
            # Backward-compatible fallback (pre-refactor layout).
            task_path = Path(__file__).with_name("task.txt")
        waypoints_ned = load_waypoints_ned(task_path)
        if waypoints_ned:
            stage = omni.usd.get_context().get_stage()
            parent_path = "/World/layout/Waypoints"
            if not prim_utils.is_prim_path_valid(parent_path):
                prim_utils.create_prim(parent_path, "Xform")

            from pxr import Gf, UsdGeom

            for i, wp_ned in enumerate(waypoints_ned):
                wp_enu = ned_to_enu(wp_ned)
                prim_path = f"{parent_path}/wp_{i:02d}"
                if prim_utils.is_prim_path_valid(prim_path):
                    continue
                prim_utils.create_prim(
                    prim_path,
                    "Cylinder",
                    position=wp_enu,
                    attributes={"radius": 0.3, "height": 0.1},
                )
                prim = stage.GetPrimAtPath(prim_path)
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
                gprim.CreateDisplayOpacityAttr().Set([0.1])
        else:
            carb.log_warn(f"No waypoints found at {task_path}; skipping waypoint markers.")

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
