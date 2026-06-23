# DroneTracking
* It is developed for Drone Tracking Task: Given the path or trajectory, the drone tracks it smoothly and stablely.
* It is build on ROS2 Humble and PX4 v1.16 with Pegasus Simulation based on IsaacSim v5.1.
* There are two parts in this repo: simulation and control.

## Compile ROS2 Package
```
pixi run colcon --log-base log_pixi build --symlink-install --build-base build_pixi --install-base install_pixi --packages-select <package-names>
```