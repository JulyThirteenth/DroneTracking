"""MPC reference trajectory node."""

from __future__ import annotations

from nav_msgs.msg import Path as NavPath

from drone_ref.base import DroneRefBase, run_node
from drone_ref.core import PathProgress


class DroneRefMpc(DroneRefBase):
    """Generate an N+1-point trajectory for MPC."""

    def __init__(self) -> None:
        super().__init__(node_name="drone_ref_mpc")

        self._horizon = int(self._param("horizon", 15))
        self._mpc_dt = float(self._param("mpc_dt", 0.1))
        self._reference_speed = float(self._param("reference_speed", 3.0))
        self._output_topic = str(self._param("output_topic", "/tracking/ref_traj_path"))

        if self._horizon < 1:
            raise ValueError("horizon must be at least 1")

        if self._mpc_dt <= 0.0:
            raise ValueError("mpc_dt must be positive")

        if self._reference_speed < 0.0:
            raise ValueError("reference_speed cannot be negative")

        self._publisher = self.create_publisher(
            NavPath,
            self._output_topic,
            10,
        )

        self.get_logger().info(
            f"MPC reference: topic={self._output_topic}, "
            f"horizon={self._horizon}, "
            f"dt={self._mpc_dt:.3f}, "
            f"speed={self._reference_speed:.3f}"
        )

    def _publish_reference(
        self,
        progress: PathProgress | None,
    ) -> None:
        points = self.reference_path.mpc_reference(
            progress=progress,
            current_position_enu=self.position_enu,
            horizon=self._horizon,
            sample_distance=(self._reference_speed * self._mpc_dt),
        )

        self._publisher.publish(self._make_path_message(points))


def main(args=None) -> None:
    """Run the MPC reference node."""
    run_node(DroneRefMpc, args)


if __name__ == "__main__":
    main()
