"""Behavior layer for the drone racing FSM.

`DroneBehaviors` owns the low-level tracker and PX4 bridge. The FSM node feeds it:
  - vehicle telemetry (`VehicleState`)
  - path / reference trajectory topics
  - state transitions (via `on_enter`)
"""

from __future__ import annotations
<<<<<<< HEAD
from dataclasses import dataclass
from typing import Any
import numpy as np
=======

from dataclasses import dataclass
from typing import Any
import numpy as np
from fsm_log import TickData
from fsm_spec import (
    EVENT_LAND,
    EVENT_TAKEOFF,
    STATE_HOVER,
    STATE_HOVER_START,
    STATE_PREFLIGHT,
    STATE_READY,
    STATE_RETURN_HOVER,
    STATE_TRACKING,
)
_CONTROLLER_MPC = "mpc"
_CONTROLLER_MPCC = "mpcc"
_PX4_CMD_NAV_LAND = 21
_PX4_CMD_DO_SET_MODE = 176
_PX4_CMD_COMPONENT_ARM_DISARM = 400
_PX4_MODE_OFFBOARD = 6.0
_MIN_PROGRESS_DISTANCE_M = 0.01
>>>>>>> 908d818 (real_drone_dev)


@dataclass
class VehicleState:
    """Vehicle state in ENU coordinates."""

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    accel_enu: np.ndarray
    yaw_enu: float


def _points_enu(data: np.ndarray) -> np.ndarray:
    """Normalize path-like input to an `(N, 3)` ENU array."""
    return np.asarray(data, dtype=float).reshape(-1, 3)


def _vec3(data: np.ndarray) -> np.ndarray:
    """Normalize vector-like input to a `(3,)` float array."""
    return np.asarray(data, dtype=float).reshape(3)


class DroneBehaviors:
    """Per-state behaviors that translate FSM state into tracking targets."""

    def __init__(
        self,
        *,
        bridge: Any,
        tracker: Any,
        logger: Any | None = None,
        takeoff_speed_mps: float = 0.67,
        init_yaw_enu: float = 0.0,
    ):
        self._bridge = bridge
        self._tracker = tracker
        self._csv_logger = logger

        self._controller = str(getattr(tracker, "controller", _CONTROLLER_MPC))
        self._controller = self._controller.lower().strip()
        self._horizon_steps = int(getattr(tracker, "horizon", 10))
        self._tracker_dt_s = float(getattr(tracker, "dt", 0.1))
        self._takeoff_speed_mps = max(float(takeoff_speed_mps), 1e-3)
        self._init_yaw_enu = float(init_yaw_enu)

        self._vehicle_state: VehicleState | None = None
        self._path_points_enu: np.ndarray | None = None  # (M, 3) for MPCC
        self._ref_traj_enu: np.ndarray | None = None  # (3, N+1) for MPC
        self._start_point_enu: np.ndarray | None = None

        self._hold_point_enu: np.ndarray | None = None
        self._hold_key: str | None = None
        self._yaw_cmd_enu: float = -np.pi / 2.0

        # After AUTO.LAND, stop publishing offboard mode and rates setpoints.
        self._disengaged: bool = False

    def update_vehicle_state(self, state: VehicleState) -> None:
        self._vehicle_state = state

    def update_yaw_cmd_enu(self, yaw_cmd_enu: float) -> None:
        self._yaw_cmd_enu = float(yaw_cmd_enu)

    def update_takeoff_yaw_cmd_enu(self) -> None:
        self._yaw_cmd_enu = self._init_yaw_enu

    def update_path(self, path_points_enu: np.ndarray) -> None:
        points_enu = _points_enu(path_points_enu)
        if int(points_enu.shape[0]) < 1:
            return

        self._set_start_point_once(points_enu[0], source="path")
        self._path_points_enu = points_enu

    def update_ref_traj(self, ref_traj_enu: np.ndarray) -> None:
        points_enu = _points_enu(ref_traj_enu)
        if int(points_enu.shape[0]) < 1:
            return

        self._set_start_point_once(points_enu[0], source="ref_traj")
        self._ref_traj_enu = points_enu[: self._horizon_steps + 1].T

    def tracking_terminal_metrics(self) -> tuple[bool, float, float]:
        """Return whether the tracking terminal target is valid plus distance/speed."""
        if self._vehicle_state is None:
            return False, float("inf"), float("inf")

        target_enu = self._tracking_terminal_target_enu()
        if target_enu is None:
            return False, float("inf"), float("inf")

        pos_enu = _vec3(self._vehicle_state.position_enu)
        vel_enu = _vec3(self._vehicle_state.velocity_enu)
        distance_m = float(np.linalg.norm(target_enu - pos_enu))
        speed_mps = float(np.linalg.norm(vel_enu))
        return True, distance_m, speed_mps

    def on_enter(self, state: str, event_name: str) -> None:
        """Handle FSM state-entry actions."""
        self._tracker.reset_warmstart()
        self._clear_hold()
        self._log_event(state, event_name)

        if state == STATE_READY:
            self._enable_offboard()
            return

        if state == STATE_HOVER_START and event_name == EVENT_TAKEOFF:
            self._enable_offboard()
            self._arm_vehicle()
            return

        if state == STATE_PREFLIGHT and event_name == EVENT_LAND:
            self._send_land()
            self._reset_tracking_inputs()

    def tick(self, fsm_state: str, dt: float) -> None:
        """Advance the behavior loop for the current FSM state."""
        if (
            self._vehicle_state is None
            or self._disengaged
            or fsm_state == STATE_PREFLIGHT
        ):
            return

        self._bridge.publish_offboard_mode()

        ref_traj_enu = self._reference_for_state(fsm_state)
        p_cmd, q_cmd, r_cmd, thrust = self._run_tracker(ref_traj_enu, dt)
        self._bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)
        self._log_tick(fsm_state, p_cmd, q_cmd, r_cmd, thrust)

    def _set_start_point_once(self, point_enu: np.ndarray, *, source: str) -> None:
        if self._start_point_enu is not None:
            return
        self._start_point_enu = _vec3(point_enu).copy()
        print(f"Update {source}: Set start_pt to enu {self._start_point_enu}")

    def _tracking_terminal_target_enu(self) -> np.ndarray | None:
        if self._controller == _CONTROLLER_MPCC:
            if self._path_points_enu is None or int(self._path_points_enu.shape[0]) < 1:
                return None
            return _vec3(self._path_points_enu[-1])

        if self._ref_traj_enu is None or int(self._ref_traj_enu.shape[1]) < 1:
            return None
        return _vec3(self._ref_traj_enu[:, -1])

    def _enable_offboard(self) -> None:
        self._disengaged = False
        self._bridge.send_vehicle_command(
            _PX4_CMD_DO_SET_MODE,
            1.0,
            _PX4_MODE_OFFBOARD,
        )

    def _arm_vehicle(self) -> None:
        self._bridge.send_vehicle_command(_PX4_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)

    def _send_land(self) -> None:
        self._disengaged = True
        self._bridge.send_vehicle_command(_PX4_CMD_NAV_LAND, 0.0, 0.0)

    def _reset_tracking_inputs(self) -> None:
        self._path_points_enu = None
        self._ref_traj_enu = None
        self._start_point_enu = None

    def _log_event(self, state: str, event_name: str) -> None:
        if self._csv_logger is None:
            return
        try:
            self._csv_logger.log_event(state=str(state), event=str(event_name))
        except Exception:
            pass

    def _reference_for_state(self, fsm_state: str) -> np.ndarray | None:
        hold_point_enu = self._hold_position(fsm_state)

        if self._controller == _CONTROLLER_MPC:
            return self._mpc_reference(fsm_state, hold_point_enu)

        if self._controller == _CONTROLLER_MPCC:
            return self._mpcc_reference(hold_point_enu)

        return None

    def _mpcc_reference(self, hold_point_enu: np.ndarray | None) -> np.ndarray | None:
        """Return an MPCC path or single-point hold target."""
        if hold_point_enu is not None:
            return _vec3(hold_point_enu).reshape(1, 3)

        return self._path_points_enu

    def _run_tracker(
        self, reference_enu: np.ndarray | None, dt: float
    ) -> tuple[float, float, float, float]:
        ref_traj_enu = reference_enu if self._controller == _CONTROLLER_MPC else None
        path_points_enu = reference_enu if self._controller == _CONTROLLER_MPCC else None
        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._vehicle_state.position_enu,
            self._vehicle_state.velocity_enu,
            self._vehicle_state.accel_enu,
            self._vehicle_state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=ref_traj_enu,
            path_points_enu=path_points_enu,
            log_solver=False,
        )
        return float(p_cmd), float(q_cmd), float(r_cmd), float(thrust)

    def _log_tick(
        self,
        fsm_state: str,
        p_cmd: float,
        q_cmd: float,
        r_cmd: float,
        thrust: float,
    ) -> None:
        if self._csv_logger is None:
            return

        try:
            debug = getattr(self._tracker, "last_debug", {}) or {}
            self._csv_logger.log_tick(
                TickData(
                    fsm_state=str(fsm_state),
                    pos_enu=np.asarray(self._vehicle_state.position_enu, dtype=float),
                    vel_enu=np.asarray(self._vehicle_state.velocity_enu, dtype=float),
                    acc_enu=np.asarray(self._vehicle_state.accel_enu, dtype=float),
                    yaw_enu=float(self._vehicle_state.yaw_enu),
                    p_cmd=float(p_cmd),
                    q_cmd=float(q_cmd),
                    r_cmd=float(r_cmd),
                    thrust_cmd=float(thrust),
                    jerk_cmd_enu=debug.get("jerk_cmd_enu", None),
                    acc_cmd_enu=debug.get("acc_cmd_enu", None),
                    jerk_cmd_ned=debug.get("jerk_cmd_ned", None),
                    acc_cmd_ned=debug.get("acc_cmd_ned", None),
                    yaw_cmd_enu=debug.get("yaw_cmd_enu", None),
                    yaw_rate_cmd_enu=debug.get("yaw_rate_cmd_enu", None),
                    yaw_cmd_ned=debug.get("yaw_cmd_ned", None),
                    yaw_rate_cmd_ned=debug.get("yaw_rate_cmd_ned", None),
                )
            )
        except Exception:
            pass

    def _hold_position(self, fsm_state: str) -> np.ndarray | None:
        """Return an ENU point to hold, or None if no hold override is needed."""
        if fsm_state in {STATE_READY, STATE_HOVER}:
            return self._hold_current_position(key=fsm_state)

        if fsm_state == STATE_HOVER_START:
            return self._takeoff_hold_position()

        if fsm_state == STATE_RETURN_HOVER:
            return self._return_hold_position()

        if fsm_state == STATE_TRACKING:
            return self._tracking_hold_position()

        return self._hold_current_position(key="unknown")

    def _takeoff_hold_position(self) -> np.ndarray:
        print("Takeoff hold position")
        if self._start_point_enu is not None:
            return self._hold_target_once(
                key="hover_start:start",
                point_enu=self._start_point_enu,
            )

        target_enu = _vec3(self._vehicle_state.position_enu).copy()
        target_enu[2] += 1.0
        return self._hold_target_once(
            key="hover_start:takeoff_1m",
            point_enu=target_enu,
        )

    def _return_hold_position(self) -> np.ndarray:
        print("Return hold position")
        if self._start_point_enu is not None:
            return self._hold_target_once(
                key="return_hover:start",
                point_enu=self._start_point_enu,
            )
        return self._hold_current_position(key="return_hover:current")

    def _tracking_hold_position(self) -> np.ndarray | None:
        if self._controller == _CONTROLLER_MPCC:
            print("MPCC tracking hold position")
            return self._mpcc_tracking_hold_position()
        else:
            print("MPC tracking hold position")
            return self._mpc_tracking_hold_position()

    def _mpcc_tracking_hold_position(self) -> np.ndarray | None:
        if self._path_points_enu is None or int(self._path_points_enu.shape[0]) < 1:
            return self._hold_current_position(key="tracking:mpcc:hold_current_no_path")

        if int(self._path_points_enu.shape[0]) < 2:
            return self._hold_target_once(
                key="tracking:mpcc:hold_last_singleton",
                point_enu=self._path_points_enu[-1],
            )

        self._clear_tracking_hold()
        return None

    def _mpc_tracking_hold_position(self) -> np.ndarray | None:
        if self._ref_traj_enu is None:
            return self._hold_current_position(key="tracking:mpc:hold_current_no_ref")

        self._clear_tracking_hold()
        return None

    def _hold_target_once(self, *, key: str, point_enu: np.ndarray) -> np.ndarray:
        if self._hold_key != key or self._hold_point_enu is None:
            self._hold_point_enu = _vec3(point_enu).copy()
            self._hold_key = str(key)
        return self._hold_point_enu

    def _hold_current_position(self, *, key: str) -> np.ndarray:
        return self._hold_target_once(
            key=key,
            point_enu=self._vehicle_state.position_enu,
        )

    def _clear_hold(self) -> None:
        self._hold_point_enu = None
        self._hold_key = None

    def _clear_tracking_hold(self) -> None:
        if self._hold_key is not None and self._hold_key.startswith("tracking:"):
            self._clear_hold()

    def _mpc_reference(
        self,
        fsm_state: str,
        hold_point_enu: np.ndarray | None,
    ) -> np.ndarray | None:
        """Return an MPC reference trajectory."""
        if fsm_state == STATE_TRACKING and self._ref_traj_enu is not None:
            return self._ref_traj_enu

        if hold_point_enu is None:
            return None

        target_enu = _vec3(hold_point_enu)
        current_enu = _vec3(self._vehicle_state.position_enu)
        distance_m = float(np.linalg.norm(target_enu - current_enu))
        if not np.isfinite(distance_m) or distance_m <= _MIN_PROGRESS_DISTANCE_M:
            return self._repeat_reference(target_enu)

        if fsm_state == STATE_HOVER_START:
            return self._takeoff_reference(current_enu, target_enu, distance_m)

        progress = np.linspace(0.0, 1.0, self._horizon_steps + 1, dtype=float)
        return self._linear_reference(current_enu, target_enu, progress)

    def _takeoff_reference(
        self,
        current_enu: np.ndarray,
        target_enu: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        step_m = self._takeoff_speed_mps * self._tracker_dt_s
        samples_m = step_m * np.arange(self._horizon_steps + 1, dtype=float)
        progress = np.clip(samples_m / distance_m, 0.0, 1.0)
        return self._linear_reference(current_enu, target_enu, progress)

    def _linear_reference(
        self,
        current_enu: np.ndarray,
        target_enu: np.ndarray,
        progress: np.ndarray,
    ) -> np.ndarray:
        delta_enu = (target_enu - current_enu).reshape(3, 1)
        return current_enu.reshape(3, 1) + delta_enu * progress.reshape(1, -1)

    def _repeat_reference(self, target_enu: np.ndarray) -> np.ndarray:
        return np.repeat(target_enu.reshape(3, 1), self._horizon_steps + 1, axis=1)
