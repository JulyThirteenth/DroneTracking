"""MPCC full-path reference publisher."""

from __future__ import annotations

from nav_msgs.msg import Path as NavPath
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from drone_ref.base import DroneRefBase, run_node
from drone_ref.core import PathProgress

PATH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class DroneRefMpcc(DroneRefBase):
    """Publish one complete path message for each path revision."""

    def __init__(self) -> None:
        super().__init__(node_name="drone_ref_mpcc")

        self._output_topic = str(self._param("output_topic", "/tracking/path"))

        self._publisher = self.create_publisher(
            NavPath,
            self._output_topic,
            PATH_QOS,
        )

        # The path may already have been loaded by DroneRefBase.
        # Keeping this at -1 ensures it is published on the first timer tick.
        self._published_revision = -1

        self.get_logger().info(
            "MPCC full-path publisher started: " f"topic={self._output_topic}"
        )

    def _publish_reference(
        self,
        progress: PathProgress | None,
    ) -> None:
        # Vehicle progress is maintained by the MPCC controller,
        # so it is intentionally unused here.
        del progress

        self._publish_path_if_changed()

    def _publish_without_vehicle_state(self) -> None:
        # A complete geometric path does not depend on vehicle state.
        self._publish_path_if_changed()

    def _publish_path_if_changed(self) -> None:
        if not self.reference_path.valid:
            return

        revision = self.path_revision

        if revision == self._published_revision:
            return

        points = self.reference_path.points
        message = self._make_path_message(points)

        self._publisher.publish(message)
        self._published_revision = revision

        self.get_logger().info(
            "Published complete MPCC path: "
            f"revision={revision}, "
            f"points={points.shape[0]}, "
            f"length={self.reference_path.length:.3f} m"
        )


def main(args=None) -> None:
    """Run the MPCC reference node."""
    run_node(DroneRefMpcc, args)


if __name__ == "__main__":
    main()
