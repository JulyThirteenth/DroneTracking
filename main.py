import rclpy
from cnt.chaser_cnt import WaypointTrackerCtbrNode


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        default="tracker",
        choices=["chaser", "tracker"],
        help="Optimization problem: point tracking (chaser) or MPCC (tracker).",
    )
    parser.add_argument(
        "--solver",
        default="ipopt",
        choices=["ipopt", "osqp"],
        help="Backend: IPOPT (nonlinear) or OSQP (QP form).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Waypoint task file path (default: examples/drone_racing/tasks/task.txt).",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = WaypointTrackerCtbrNode(
        controller=args.controller, solver=args.solver, task_path=args.task
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
