"""Geometric quadrotor controller on SO(3).

This implements the velocity/attitude tracking structure from Lee, Leok and
McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC
2010, adapted to the NED/FRD convention used by :mod:`uav_sim`.
"""

from __future__ import annotations

import numpy as np
from spatialmath import SO3
from spatialmath.base import vex

from uav_sim import QuadrotorParams


def _hat(x):
    x = np.asarray(x, dtype=np.float64)
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def attitude_from_force_acceleration(force_acceleration, yaw):
    """Return an attitude with the requested body-z direction and ZYX yaw."""
    force_acceleration = np.asarray(force_acceleration, dtype=np.float64)
    magnitude = np.linalg.norm(force_acceleration)
    if magnitude < 1e-8:
        raise ValueError("force acceleration must be nonzero")

    b3 = force_acceleration / magnitude
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    b3_yaw = np.array(
        [
            cosine_yaw * b3[0] + sine_yaw * b3[1],
            -sine_yaw * b3[0] + cosine_yaw * b3[1],
            b3[2],
        ]
    )
    roll = -np.arcsin(np.clip(b3_yaw[1], -1.0, 1.0))
    pitch = np.arctan2(b3_yaw[0], b3_yaw[2])

    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    rotation_yaw = np.array(
        [[cosine_yaw, -sine_yaw, 0.0], [sine_yaw, cosine_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_pitch = np.array(
        [
            [cosine_pitch, 0.0, sine_pitch],
            [0.0, 1.0, 0.0],
            [-sine_pitch, 0.0, cosine_pitch],
        ]
    )
    rotation_roll = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine_roll, -sine_roll], [0.0, sine_roll, cosine_roll]]
    )
    return rotation_yaw @ rotation_pitch @ rotation_roll


class SO3Controller:
    """CDC geometric velocity controller followed by SO(3) attitude control."""

    def __init__(
        self,
        params: QuadrotorParams,
        velocity_gain=2.0,
        attitude_frequency=8.0,
        attitude_damping=0.8,
    ):
        self.mass = float(params.mass)
        self.inertia = np.asarray(params.inertia, dtype=np.float64)
        self.gravity = float(params.g)
        self.velocity_gain = float(velocity_gain)
        self.attitude_rate_gain = float(attitude_frequency)

        inertia_diagonal = np.diag(self.inertia)
        self.attitude_gain = inertia_diagonal * attitude_frequency**2
        self.angular_rate_gain = (
            2.0 * attitude_damping * attitude_frequency * inertia_diagonal
        )
        self.reset()

    def reset(self):
        self.force_acceleration = np.array([0.0, 0.0, self.gravity])
        self.R_ref = np.eye(3)
        self.body_rate_ref = np.zeros(3)
        self.body_rate_acceleration_ref = np.zeros(3)
        self._previous_R_ref = None
        self._previous_body_rate_ref = None
        self.last_ctbr = np.r_[self.gravity, np.zeros(3)]
        self.last_wrench = np.zeros(4)

    @staticmethod
    def _desired_attitude(force_acceleration, yaw_ref):
        return attitude_from_force_acceleration(force_acceleration, yaw_ref)

    def update_reference(
        self,
        velocity,
        velocity_ref,
        acceleration_ref,
        yaw_ref,
        dt,
        disturbance_force_world=None,
    ):
        """Update the velocity-loop command at the outer-loop frequency."""
        velocity_error = np.asarray(velocity) - np.asarray(velocity_ref)
        acceleration_command = (
            np.asarray(acceleration_ref) - self.velocity_gain * velocity_error
        )
        e3 = np.array([0.0, 0.0, 1.0])
        disturbance_force_world = (
            np.zeros(3)
            if disturbance_force_world is None
            else np.asarray(disturbance_force_world, dtype=np.float64)
        )
        self.force_acceleration = (
            self.gravity * e3
            - acceleration_command
            + disturbance_force_world / self.mass
        )
        R_ref = self._desired_attitude(self.force_acceleration, yaw_ref)

        if self._previous_R_ref is None:
            body_rate_ref = np.zeros(3)
            body_rate_acceleration_ref = np.zeros(3)
        else:
            body_rate_ref = SO3(self._previous_R_ref.T @ R_ref).log(twist=True) / dt
            body_rate_acceleration_ref = (
                body_rate_ref - self._previous_body_rate_ref
            ) / dt

        self._previous_R_ref = R_ref.copy()
        self._previous_body_rate_ref = body_rate_ref.copy()
        self.R_ref = R_ref
        self.body_rate_ref = body_rate_ref
        self.body_rate_acceleration_ref = body_rate_acceleration_ref

    def control(self, R, body_rate):
        """Return ``[thrust, tau_x, tau_y, tau_z]`` at the inner-loop rate."""
        R = np.asarray(R, dtype=np.float64)
        body_rate = np.asarray(body_rate, dtype=np.float64)
        R_error = 0.5 * vex(self.R_ref.T @ R - R.T @ self.R_ref)
        relative_rotation = R.T @ self.R_ref
        rate_error = body_rate - relative_rotation @ self.body_rate_ref

        feedforward = _hat(body_rate) @ relative_rotation @ self.body_rate_ref
        feedforward -= relative_rotation @ self.body_rate_acceleration_ref
        moment = (
            -self.attitude_gain * R_error
            - self.angular_rate_gain * rate_error
            + np.cross(body_rate, self.inertia @ body_rate)
            - self.inertia @ feedforward
        )

        collective_acceleration = max(0.0, self.force_acceleration @ R[:, 2])
        wrench = np.r_[self.mass * collective_acceleration, moment]
        self.last_ctbr = np.r_[collective_acceleration, self.body_rate_ref]
        self.last_wrench = wrench
        return wrench

    def control_ctbr(self, R):
        """Return ``[collective_acceleration, p_cmd, q_cmd, r_cmd]``.

        This is the cascaded form of the geometric attitude controller. The
        attitude error is converted into a body-rate command, leaving rate
        control and actuator allocation to the downstream controller.
        """
        R = np.asarray(R, dtype=np.float64)
        R_error = 0.5 * vex(self.R_ref.T @ R - R.T @ self.R_ref)
        relative_rotation = R.T @ self.R_ref
        body_rate_command = (
            relative_rotation @ self.body_rate_ref - self.attitude_rate_gain * R_error
        )
        collective_acceleration = max(0.0, self.force_acceleration @ R[:, 2])
        self.last_ctbr = np.r_[collective_acceleration, body_rate_command]
        return self.last_ctbr.copy()

    __call__ = control
