"""ROS 2 deployment node for feedforward plus pure learned Lie-CCM."""

from __future__ import annotations

from px4_msgs.msg import VehicleOdometry, VehicleRatesSetpoint
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from drone_ccm.frame import (
    FRD_FROM_FLU,
    VehicleState,
    collective_to_normalized_thrust,
)
from drone_ccm.reference_message import (
    CcmReference,
    decode_reference,
    domain_signature,
)
from drone_ccm.ros_utils import (
    latched_qos,
    parameter,
    resolve_checkpoint,
    run_node,
    sensor_qos,
    vehicle_state_from_odometry,
)
from drone_ccm.runtime import load_runtime


class ControllerNode(Node):
    """Publishes one internally arbitrated PX4 CTBR setpoint stream."""

    def __init__(self) -> None:
        super().__init__("drone_ccm_controller")
        checkpoint = resolve_checkpoint(self)
        self._control_period = float(parameter(self, "control_period", 0.01))
        self._input_timeout = float(parameter(self, "input_timeout", 0.25))
        self._hover_thrust = float(parameter(self, "hover_thrust", 0.5812))
        self._active_state = parameter(self, "active_state", "tracking")
        if self._control_period <= 0.0 or self._input_timeout <= 0.0:
            raise ValueError("Control period and input timeout must be positive")
        if not 0.0 < self._hover_thrust < 1.0:
            raise ValueError("hover_thrust must be in (0, 1)")
        self._runtime = load_runtime(
            checkpoint,
            device_name=parameter(self, "device", "cpu"),
            dtype_name=parameter(self, "dtype", "float32"),
        )
        self._runtime.warmup()
        self._domain_signature = domain_signature(self._runtime.domain)

        self._state: VehicleState | None = None
        self._reference: CcmReference | None = None
        self._state_received_ns: int | None = None
        self._reference_received_ns: int | None = None
        self._fsm_state: str | None = None
        self._activation_ns: int | None = None
        self._fault_latched = False
        self._has_published_active = False
        self._last_status: str | None = None

        output_topic = parameter(
            self,
            "rates_setpoint_topic",
            "/drone_cnt/vehicle_rates_setpoint",
        )
        self._publisher = self.create_publisher(
            VehicleRatesSetpoint,
            output_topic,
            10,
        )
        self.create_subscription(
            VehicleOdometry,
            parameter(self, "odometry_topic", "/fmu/out/vehicle_odometry"),
            self._on_odometry,
            sensor_qos(),
        )
        self.create_subscription(
            Float64MultiArray,
            parameter(self, "reference_topic", "/tracking/ccm_reference"),
            self._on_reference,
            10,
        )
        self.create_subscription(
            String,
            parameter(self, "fsm_state_topic", "/fsm/state"),
            self._on_fsm_state,
            latched_qos(),
        )
        self.create_timer(self._control_period, self._on_timer)
        runtime_name = type(self._runtime).__name__.lstrip("_")
        self.get_logger().info(
            f"Loaded {runtime_name} checkpoint: {checkpoint}"
        )
        self.get_logger().info(
            f"Internal CTBR topic: {output_topic}; "
            f"hover_thrust={self._hover_thrust:.4f}"
        )

    def _on_odometry(self, message: VehicleOdometry) -> None:
        try:
            self._state = vehicle_state_from_odometry(message)
        except ValueError as error:
            self._report(str(error), error=True)
            return
        self._state_received_ns = self.get_clock().now().nanoseconds

    def _on_reference(self, message: Float64MultiArray) -> None:
        try:
            reference = decode_reference(message)
            if reference.domain_signature != self._domain_signature:
                raise ValueError(
                    "Reference domain does not match the controller checkpoint"
                )
            self._reference = reference
        except ValueError as error:
            self._report(str(error), error=True)
            return
        self._reference_received_ns = self.get_clock().now().nanoseconds

    def _on_fsm_state(self, message: String) -> None:
        next_state = message.data.strip()
        if next_state == self._active_state and self._fsm_state != self._active_state:
            self._activation_ns = self.get_clock().now().nanoseconds
        if next_state != self._active_state:
            self._activation_ns = None
            self._fault_latched = False
            self._has_published_active = False
        self._fsm_state = next_state

    def _readiness_failure(self) -> str | None:
        now = self.get_clock().now().nanoseconds
        timeout_ns = int(self._input_timeout * 1.0e9)
        inputs = (
            (self._state, self._state_received_ns, "VehicleOdometry"),
            (self._reference, self._reference_received_ns, "CCM reference"),
        )
        for value, received_ns, label in inputs:
            if value is None or received_ns is None:
                return f"Waiting for {label}"
            age = now - received_ns
            if age < 0 or age > timeout_ns:
                return f"{label} is stale"
            if self._activation_ns is not None and received_ns < self._activation_ns:
                return f"Waiting for post-activation {label}"
        return None

    def _on_timer(self) -> None:
        if self._fsm_state != self._active_state:
            return
        if self._fault_latched:
            self._report(
                "CCM fault is latched; leave tracking before attempting re-entry",
                error=True,
            )
            return
        failure = self._readiness_failure()
        if failure is not None:
            if self._has_published_active:
                self._fault_latched = True
            self._report(failure)
            return
        assert self._state is not None
        assert self._reference is not None
        try:
            control = self._runtime.command(
                self._state.velocity,
                self._state.rotation,
                self._reference.velocity,
                self._reference.rotation,
                self._reference.control,
            )
            normalized_thrust = collective_to_normalized_thrust(
                control[0],
                self._hover_thrust,
                self._runtime.gravity,
            )
            if not 0.0 < normalized_thrust < 1.0:
                raise RuntimeError(
                    "Physical-to-PX4 thrust conversion is outside (0, 1)"
                )
        except (RuntimeError, ValueError) as error:
            self._fault_latched = True
            self._report(str(error), error=True)
            return

        body_rate_frd = FRD_FROM_FLU @ control[1:4]
        output = VehicleRatesSetpoint()
        output.timestamp = self.get_clock().now().nanoseconds // 1000
        output.roll = float(body_rate_frd[0])
        output.pitch = float(body_rate_frd[1])
        output.yaw = float(body_rate_frd[2])
        output.thrust_body = [0.0, 0.0, -float(normalized_thrust)]
        output.reset_integral = False
        self._publisher.publish(output)
        self._has_published_active = True
        self._last_status = None

    def _report(self, message: str, *, error: bool = False) -> None:
        if message == self._last_status:
            return
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)
        self._last_status = message


def main(args: list[str] | None = None) -> None:
    """Runs the controller node."""
    run_node(ControllerNode, args)


if __name__ == "__main__":
    main()
