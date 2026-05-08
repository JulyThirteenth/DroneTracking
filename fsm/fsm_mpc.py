import numpy as np
from abc import abstractmethod
from rclpy.node import Node
from .fsm_log import TickData
from .fsm_wrap import FSMBehaviorBase, FSMLoggerBase
from .fsm_spec import (
    STATE_READY,
    STATE_HOVER,
    STATE_PREFLIGHT,
    STATE_HOVER_START,
    STATE_RETURN_HOVER,
    STATE_TRACKING,
    EVENT_TAKEOFF,
    EVENT_LAND,
)
from tracking.tracking_cnt import PathTrackerCtbr


class MPCBehaviorBase(FSMBehaviorBase):
    """MPC-based tracking behavior."""

    def __init__(self, *, node, logger, tracker):
        super().__init__(node=node, logger=logger, tracker=tracker)
        self._start_point_enu: np.ndarray | None = None
        self._hover_point_enu: np.ndarray | None = None
        self._hover_key: str | None = None
        self._yaw_cmd_enu: float = float | None
        self._ref_cmd_enu: np.ndarray | None = None

    def on_enter(self, state: str, event_name: str) -> None:
        """Run on FSM state entry."""
        self._tracker.reset_warmstart()
        self.clear_hover_target()
        self._logger.log_event(state=state, event=event_name)

        if state == STATE_READY:
            self.offboard()
        elif state == STATE_HOVER_START and event_name == EVENT_TAKEOFF:
            self.offboard()
            self.arm()
        elif state == STATE_PREFLIGHT and event_name == EVENT_LAND:
            self.land()
            self._start_point_enu = None
            self._ref_cmd_enu = None

    def update_yaw_cmd_enu(self, yaw_cmd_enu: float) -> None:
        """Update yaw command in ENU coordinates."""
        self._yaw_cmd_enu = float(yaw_cmd_enu)

    @abstractmethod
    def update_ref_cmd_enu(self, ref_cmd_enu: np.ndarray) -> None:
        """Update tracking reference input in ENU coordinates."""
        raise NotImplementedError

    def _hover_target_once(self, key: str, hover_target: np.ndarray) -> None:
        if self._hover_key != key or self._hover_point_enu is None:
            self._hover_point_enu = np.asarray(hover_target, dtype=float).copy()
            self._hover_key = str(key)
        return self._hover_point_enu

    def _hover_current_position(self, key: str) -> np.ndarray | None:
        return self._hover_target_once(
            key=key,
            hover_target=self._vehicle_state.position_enu,
        )

    def clear_hover_target(self) -> None:
        """Clear any currently active hover target."""
        self._hover_point_enu = None
        self._hover_key = None

    def get_hover_target(self, state: str) -> np.ndarray | None:
        """Get the current hover target point in ENU coordinates, if any."""
        if state in (STATE_READY, STATE_HOVER):
            return self._hover_current_position(key=state)

        elif state == STATE_HOVER_START:
            if self._start_point_enu is not None:
                return self._hover_target_once(
                    key="hover_start:takeoff", hover_target=self._start_point_enu
                )

            target_enu = self._vehicle_state.position_enu.copy()
            target_enu[2] += self._takeoff_height
            return self._hover_target_once(
                key=f"hover_start:takeoff_{self._takeoff_height}",
                hover_target=target_enu,
            )

        elif state == STATE_RETURN_HOVER:
            if self._start_point_enu is not None:
                return self._hover_target_once(
                    key="return_hover:start",
                    hover_target=self._start_point_enu,
                )
            return self._hover_current_position(key="return_hover:current")

        elif state == STATE_TRACKING:
            if self._ref_cmd_enu is None:
                return self._hover_current_position(key="tracking:hover_current_no_ref")
            if self._hover_key is not None and self._hover_key.startswith("tracking:"):
                self.clear_hover_target()
            return None

        else:
            return self._hover_current_position(key=f"{state}:hover_current_unknown")


class MPCBehavior(MPCBehaviorBase):
    """MPC-based tracking behavior."""

    def __init__(
        self,
        *,
        node: Node,
        logger: FSMLoggerBase,
        tracker: PathTrackerCtbr,
        takeoff_height: float = 1.0,
        takeoff_velocity: float = 0.5,
    ):
        super().__init__(
            node=node,
            logger=logger,
            tracker=tracker,
        )
        self._takeoff_height = takeoff_height
        self._takeoff_velocity = takeoff_velocity
        self._tracking_dt = self._tracker.dt
        self._tracking_horizon = self._tracker.horizon

    def update_ref_cmd_enu(self, ref_cmd_enu: np.ndarray):
        if ref_cmd_enu is None or int(ref_cmd_enu.shape[0]) < 1:
            return
        ref = np.asarray(ref_cmd_enu, dtype=float).reshape(-1, 3)
        if self._start_point_enu is None:
            self._start_point_enu = ref[0].copy()
        self._ref_cmd_enu = ref[: self._tracking_horizon + 1].T.copy()  # [3, N+1]

    def tick(
        self, fsm_state: str, dt: float, obstacle_points: np.ndarray | None
    ) -> None:
        if (
            self._vehicle_state is None
            or self._disengaged
            or fsm_state == STATE_PREFLIGHT
        ):
            return

        self._px4_bridge.publish_offboard_mode()

        ref_cmd_enu = None
        if fsm_state == STATE_TRACKING and self._ref_cmd_enu is not None:
            ref_cmd_enu = self._ref_cmd_enu
        else:
            current_position = self._vehicle_state.position_enu
            hover_target = self.get_hover_target(fsm_state)
            if hover_target is not None:
                dist2target = np.linalg.norm(hover_target - current_position)
                if dist2target < 0.01:
                    ref_cmd_enu = np.repeat(
                        hover_target.reshape(3, 1), self._tracking_horizon + 1, axis=1
                    )
                else:
                    if fsm_state == STATE_HOVER_START:
                        step = self._takeoff_velocity * self._tracking_dt
                        samples = step * np.arange(
                            self._tracking_horizon + 1, dtype=float
                        )
                        progress = np.clip(samples / dist2target, 0.0, 1.0).reshape(
                            1, -1
                        )
                        delta = (hover_target - current_position).reshape(3, 1)
                        ref_cmd_enu = current_position.reshape(3, 1) + delta * progress
                    else:
                        progress = np.linspace(
                            0.0, 1.0, self._tracking_horizon + 1, dtype=float
                        )
                        delta = (hover_target - current_position).reshape(3, 1)
                        ref_cmd_enu = current_position.reshape(3, 1) + delta * progress

        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._vehicle_state.position_enu,
            self._vehicle_state.velocity_enu,
            self._vehicle_state.accel_enu,
            self._vehicle_state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=ref_cmd_enu,
            path_points_enu=None,
            obstacle_points_enu=obstacle_points,
            log_solver=False,
        )

        self._px4_bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        debug = getattr(self._tracker, "last_debug", {}) or {}
        self._logger.log_tick(
            TickData(
                fsm_state=str(fsm_state),
                pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                ref_enu=(
                    np.asarray(ref_cmd_enu[:, 0].reshape(3).copy())
                    if ref_cmd_enu is not None
                    else None
                ),
                vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                yaw_enu=float(self._vehicle_state.yaw_enu),
                p_cmd=float(p_cmd),
                q_cmd=float(q_cmd),
                r_cmd=float(r_cmd),
                thrust_cmd=float(thrust),
                acc_est_enu=debug.get("a_est_enu"),
                jerk_cmd_enu=debug.get("jerk_cmd_enu"),
                acc_cmd_enu=debug.get("acc_cmd_enu"),
                yaw_cmd_enu=debug.get("yaw_cmd_enu"),
                yaw_rate_cmd_enu=debug.get("yaw_rate_cmd_enu"),
            )
        )


class MPCCBehavior(MPCBehaviorBase):
    def __init__(
        self,
        *,
        node: Node,
        logger: FSMLoggerBase,
        tracker: PathTrackerCtbr,
        takeoff_height: float = 1.0,
    ):
        super().__init__(node=node, logger=logger, tracker=tracker)
        self._takeoff_height = takeoff_height

    def update_ref_cmd_enu(self, ref_cmd_enu):
        if ref_cmd_enu is None or int(ref_cmd_enu.shape[0]) < 1:
            return
        ref = np.asarray(ref_cmd_enu, dtype=float).reshape(-1, 3)
        if self._start_point_enu is None:
            self._start_point_enu = ref[0].copy()
        self._ref_cmd_enu = ref.copy()  # [N, 3]

    def tick(
        self,
        fsm_state: str,
        dt: float,
        obstacle_points: np.ndarray | None = None,
    ) -> None:
        if (
            self._vehicle_state is None
            or self._disengaged
            or fsm_state == STATE_PREFLIGHT
        ):
            return

        self._px4_bridge.publish_offboard_mode()
        ref_cmd_enu = None
        if fsm_state == STATE_TRACKING and self._ref_cmd_enu is not None:
            ref_cmd_enu = self._ref_cmd_enu
        else:
            hover_target = self.get_hover_target(fsm_state)
            if hover_target is not None:
                ref_cmd_enu = hover_target.reshape(1, 3)

        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._vehicle_state.position_enu,
            self._vehicle_state.velocity_enu,
            self._vehicle_state.accel_enu,
            self._vehicle_state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=None,
            path_points_enu=ref_cmd_enu,
            obstacle_points_enu=obstacle_points,
            log_solver=False,
        )

        self._px4_bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        debug = getattr(self._tracker, "last_debug", {}) or {}
        self._logger.log_tick(
            TickData(
                fsm_state=str(fsm_state),
                pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                ref_enu=(
                    np.asarray(ref_cmd_enu[0, :].reshape(3).copy())
                    if ref_cmd_enu is not None
                    else None
                ),
                vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                yaw_enu=float(self._vehicle_state.yaw_enu),
                p_cmd=float(p_cmd),
                q_cmd=float(q_cmd),
                r_cmd=float(r_cmd),
                thrust_cmd=float(thrust),
                acc_est_enu=debug.get("a_est_enu"),
                jerk_cmd_enu=debug.get("jerk_cmd_enu"),
                acc_cmd_enu=debug.get("acc_cmd_enu"),
                yaw_cmd_enu=debug.get("yaw_cmd_enu"),
                yaw_rate_cmd_enu=debug.get("yaw_rate_cmd_enu"),
            )
        )
