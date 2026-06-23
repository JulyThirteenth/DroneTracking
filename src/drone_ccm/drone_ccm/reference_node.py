"""ROS 2 publisher for feasible velocity-attitude references without position."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from drone_ccm.frame import VehicleState
from drone_ccm.reference import (
    hover_reference,
    tracking_reference_from_targets,
)
from drone_ccm.reference_message import (
    CcmReference,
    domain_signature,
    encode_reference,
)
from drone_ccm.ros_utils import (
    latched_qos,
    parameter,
    resolve_checkpoint,
    run_node,
    sensor_qos,
    vehicle_state_from_odometry,
)
from drone_ccm.runtime import CcmDomain, load_domain
from drone_ccm.waypoint import WaypointTrajectory, bounded_heading_from_velocity


_REFERENCE_RATE_SAFETY_FACTOR = 0.85


class ReferenceNode(Node):
    """Publishes a feasible exogenous reference using no position measurement."""

    def __init__(self) -> None:
        super().__init__("drone_ccm_reference")
        checkpoint = resolve_checkpoint(self)
        self._domain: CcmDomain = load_domain(checkpoint)
        self._domain_signature = domain_signature(self._domain)
        self._mode = parameter(self, "mode", "hover").strip().lower()
        if self._mode not in {"hover", "waypoint"}:
            raise ValueError("mode must be hover or waypoint")
        self._period = float(parameter(self, "publish_period", 0.01))
        self._transition_duration = float(parameter(self, "transition_duration", 3.0))
        self._state_timeout = float(parameter(self, "state_timeout", 0.25))
        self._active_state = parameter(self, "active_state", "tracking")
        if self._period <= 0.0 or self._transition_duration <= 0.0:
            raise ValueError("Publish period and transition duration must be positive")
        if self._state_timeout <= 0.0:
            raise ValueError("state_timeout must be positive")

        self._waypoint_trajectory: WaypointTrajectory | None = None
        if self._mode == "waypoint":
            waypoint_file = parameter(self, "waypoint_file", "").strip()
            waypoint_speed = float(parameter(self, "waypoint_speed", 1.0))
            if not waypoint_file:
                raise ValueError("waypoint_file is required in waypoint mode")
            if (
                self._domain.velocity_reference_max is not None
                and waypoint_speed > self._domain.velocity_reference_max
            ):
                raise ValueError(
                    "waypoint_speed exceeds checkpoint velocity-reference domain: "
                    f"{waypoint_speed:.3f} > "
                    f"{self._domain.velocity_reference_max:.3f} m/s"
                )
            self._waypoint_trajectory = WaypointTrajectory.from_ned_file(
                waypoint_file,
                waypoint_speed,
            )
            self._waypoint_trajectory.enforce_reference_limits(
                gravity=self._domain.gravity,
                thrust_minimum=self._domain.reference_thrust_min,
                thrust_maximum=self._domain.reference_thrust_max,
                tilt_maximum=self._domain.reference_tilt_angle_max,
                body_rate_maximum=min(self._domain.reference_body_rate_max),
            )
            self.get_logger().info(
                "Loaded exogenous waypoint trajectory: "
                f"file={Path(waypoint_file).expanduser().resolve()}, "
                f"duration={self._waypoint_trajectory.duration:.2f} s, "
                f"speed_limit={waypoint_speed:.2f} m/s"
            )

        self._fsm_state: str | None = None
        self._was_active = False
        self._active_start_ns: int | None = None
        self._measured_state: VehicleState | None = None
        self._state_received_ns: int | None = None
        self._initial_velocity = torch.zeros(1, 3, dtype=torch.float64)
        self._initial_yaw = torch.zeros(1, dtype=torch.float64)
        self._waypoint_yaw = 0.0
        self._template = torch.zeros(1, 3, dtype=torch.float64)
        self._last_status: str | None = None

        output_topic = parameter(
            self,
            "output_topic",
            "/tracking/ccm_reference",
        )
        self._publisher = self.create_publisher(Float64MultiArray, output_topic, 10)
        self.create_subscription(
            VehicleOdometry,
            parameter(self, "odometry_topic", "/fmu/out/vehicle_odometry"),
            self._on_odometry,
            sensor_qos(),
        )
        self.create_subscription(
            String,
            parameter(self, "fsm_state_topic", "/fsm/state"),
            self._on_fsm_state,
            latched_qos(),
        )
        self.create_timer(self._period, self._on_timer)
        self.get_logger().info(
            f"Reference mode={self._mode}, checkpoint domain={checkpoint}"
        )

    def _on_odometry(self, message: VehicleOdometry) -> None:
        try:
            self._measured_state = vehicle_state_from_odometry(message)
        except ValueError as error:
            self._report(str(error))
            return
        self._state_received_ns = self.get_clock().now().nanoseconds
        self._last_status = None

    def _on_fsm_state(self, message: String) -> None:
        self._fsm_state = message.data.strip()

    def _latch_initial_reference(self, now_ns: int) -> bool:
        if (
            self._measured_state is None
            or self._state_received_ns is None
        ):
            self._report("Waiting for VehicleOdometry before reference engagement")
            return False
        age = now_ns - self._state_received_ns
        if age < 0 or age > int(self._state_timeout * 1.0e9):
            self._report("VehicleOdometry is stale before reference engagement")
            return False
        yaw = float(
            np.arctan2(
                self._measured_state.rotation[1, 0],
                self._measured_state.rotation[0, 0],
            )
        )
        self._initial_velocity = torch.as_tensor(
            self._measured_state.velocity,
            dtype=torch.float64,
        ).reshape(1, 3)
        self._initial_yaw = torch.tensor((yaw,), dtype=torch.float64)
        self._waypoint_yaw = yaw
        self._active_start_ns = now_ns
        self.get_logger().info(
            "Latched CCM engagement state: "
            f"velocity={np.array2string(self._measured_state.velocity, precision=3)}, "
            f"yaw={yaw:.3f}"
        )
        return True

    def _waypoint_reference(
        self,
        elapsed: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._waypoint_trajectory is None:
            raise RuntimeError("Waypoint trajectory was not initialized")
        target = self._waypoint_trajectory.sample(elapsed)
        next_target = self._waypoint_trajectory.sample(elapsed + self._period)
        maximum_yaw_step = (
            _REFERENCE_RATE_SAFETY_FACTOR
            * self._domain.reference_body_rate_max[2]
            * self._period
        )
        target_yaw = bounded_heading_from_velocity(
            target.velocity,
            self._waypoint_yaw,
            maximum_yaw_step,
        )
        next_yaw = bounded_heading_from_velocity(
            next_target.velocity,
            target_yaw,
            maximum_yaw_step,
        )
        self._waypoint_yaw = target_yaw
        return tracking_reference_from_targets(
            elapsed,
            self._vector_tensor(target.velocity),
            self._vector_tensor(target.acceleration),
            torch.tensor((target_yaw,), dtype=torch.float64),
            next_target_velocity=self._vector_tensor(next_target.velocity),
            next_target_acceleration=self._vector_tensor(next_target.acceleration),
            next_target_yaw=torch.tensor((next_yaw,), dtype=torch.float64),
            initial_velocity=self._initial_velocity,
            initial_yaw=self._initial_yaw,
            transition_duration=self._transition_duration,
            time_step=self._period,
            gravity=self._domain.gravity,
            blend_yaw=False,
        )

    @staticmethod
    def _vector_tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float64).reshape(1, 3)

    def _on_timer(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        active = self._fsm_state == self._active_state
        if active and not self._was_active:
            if not self._latch_initial_reference(now_ns):
                return
        if not active:
            self._active_start_ns = None
            self._initial_velocity.zero_()
            self._initial_yaw.zero_()
            self._waypoint_yaw = 0.0
        elapsed = (
            0.0
            if self._active_start_ns is None
            else max(0.0, (now_ns - self._active_start_ns) * 1.0e-9)
        )
        with torch.inference_mode():
            if active and self._mode == "waypoint":
                velocity, rotation, control = self._waypoint_reference(elapsed)
            else:
                velocity, rotation, control = hover_reference(
                    elapsed,
                    self._template,
                    initial_velocity=self._initial_velocity,
                    initial_yaw=self._initial_yaw,
                    transition_duration=self._transition_duration,
                    time_step=self._period,
                    gravity=self._domain.gravity,
                )
        self._validate_domain(velocity, rotation, control)
        self._publisher.publish(
            encode_reference(
                CcmReference(
                    velocity=velocity[0].numpy(),
                    rotation=rotation[0].numpy(),
                    control=control[0].numpy(),
                    domain_signature=self._domain_signature,
                )
            )
        )
        self._was_active = active
        self._last_status = None

    def _validate_domain(
        self,
        velocity: torch.Tensor,
        rotation: torch.Tensor,
        control: torch.Tensor,
    ) -> None:
        if (
            self._domain.velocity_reference_max is not None
            and float(velocity.abs().max()) > self._domain.velocity_reference_max
        ):
            raise RuntimeError("Generated velocity leaves checkpoint domain")
        tilt = torch.acos(rotation[..., 2, 2].clamp(-1.0, 1.0))
        if float(tilt.max()) > self._domain.reference_tilt_angle_max:
            raise RuntimeError("Generated attitude leaves checkpoint domain")
        thrust = float(control[0, 0])
        rate_limit = torch.as_tensor(
            self._domain.reference_body_rate_max,
            dtype=control.dtype,
            device=control.device,
        )
        if not (
            self._domain.reference_thrust_min
            <= thrust
            <= self._domain.reference_thrust_max
        ):
            raise RuntimeError("Generated thrust leaves checkpoint domain")
        if bool((control[0, 1:4].abs() > rate_limit).any()):
            raise RuntimeError("Generated body rate leaves checkpoint domain")

    def _report(self, message: str) -> None:
        if message != self._last_status:
            self.get_logger().warning(message)
            self._last_status = message


def main(args: list[str] | None = None) -> None:
    """Runs the reference node."""
    run_node(ReferenceNode, args)


if __name__ == "__main__":
    main()
