import dataclasses
import numpy as np
import yaml
from spatialmath import UnitQuaternion
from spatialmath.base import qdotb


@dataclasses.dataclass
class QuadrotorParams:
    mass: float
    arm_length: float
    inertia: np.ndarray  # (3, 3)
    thrust_coeff: float
    torque_coeff: float
    motor_time_constant: float
    motor_noise_std: float = 0.0
    rotor_min_speed: float = 0.0
    rotor_max_speed: float = np.inf
    g: float = 9.81
    rotor_direction: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray([1, -1, 1, -1])
    )
    # translational drag in body frame
    drag_force_coeff: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [dx, dy, dz]

    # rotational drag in body frame
    drag_torque_coeff: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [dp, dq, dr]

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in [
            "inertia",
            "rotor_direction",
            "drag_force_coeff",
            "drag_torque_coeff",
        ]:
            if key in data:
                data[key] = np.asarray(data[key], dtype=np.float64)
        return cls(**data)


@dataclasses.dataclass(init=False)
class BatchQuadrotorParams:
    batch_size: int  # N
    mass: np.ndarray  # (N,)
    arm_length: np.ndarray  # (N,)
    inertia: np.ndarray  # (N, 3, 3)
    thrust_coeff: np.ndarray  # (N,)
    torque_coeff: np.ndarray  # (N,)
    motor_time_constant: np.ndarray  # (N,)
    motor_noise_std: np.ndarray  # (N,)
    drag_force_coeff: np.ndarray  # (N, 3)
    drag_torque_coeff: np.ndarray  # (N, 3)
    rotor_min_speed: np.ndarray  # (N,)
    rotor_max_speed: np.ndarray  # (N,)
    g: np.ndarray  # (N,)
    rotor_direction: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray([1, -1, 1, -1])
    )  # (N, 4)

    def __init__(self, params: list[QuadrotorParams]):
        self.batch_size = len(params)
        self.mass = np.asarray([p.mass for p in params], dtype=np.float64)
        self.arm_length = np.asarray([p.arm_length for p in params], dtype=np.float64)
        self.inertia = np.stack(
            [np.asarray(p.inertia, dtype=np.float64) for p in params]
        )
        self.thrust_coeff = np.asarray(
            [p.thrust_coeff for p in params], dtype=np.float64
        )
        self.torque_coeff = np.asarray(
            [p.torque_coeff for p in params], dtype=np.float64
        )
        self.motor_time_constant = np.asarray(
            [p.motor_time_constant for p in params], dtype=np.float64
        )
        self.motor_noise_std = np.asarray(
            [p.motor_noise_std for p in params], dtype=np.float64
        )
        self.drag_force_coeff = np.stack(
            [np.asarray(p.drag_force_coeff, dtype=np.float64) for p in params]
        )
        self.drag_torque_coeff = np.stack(
            [np.asarray(p.drag_torque_coeff, dtype=np.float64) for p in params]
        )
        self.rotor_min_speed = np.asarray(
            [p.rotor_min_speed for p in params], dtype=np.float64
        )
        self.rotor_max_speed = np.asarray(
            [p.rotor_max_speed for p in params], dtype=np.float64
        )
        self.g = np.asarray([p.g for p in params], dtype=np.float64)
        self.rotor_direction = np.stack(
            [np.asarray(p.rotor_direction, dtype=np.float64) for p in params]
        )


def generate_random_params(
    batch_size: int, nominal_params: QuadrotorParams, ranges: dict
):
    batch_params = BatchQuadrotorParams([nominal_params for _ in range(batch_size)])
    for idx in range(batch_size):
        if "mass" in ranges:
            batch_params.mass[idx] = np.random.uniform(
                ranges["mass"][0], ranges["mass"][1]
            )
        if "thrust_coeff" in ranges:
            batch_params.thrust_coeff[idx] = np.random.uniform(
                ranges["thrust_coeff"][0], ranges["thrust_coeff"][1]
            )
        if "torque_coeff" in ranges:
            batch_params.torque_coeff[idx] = np.random.uniform(
                ranges["torque_coeff"][0], ranges["torque_coeff"][1]
            )
        if "motor_time_constant" in ranges:
            batch_params.motor_time_constant[idx] = np.random.uniform(
                ranges["motor_time_constant"][0], ranges["motor_time_constant"][1]
            )
        if "motor_noise_std" in ranges:
            batch_params.motor_noise_std[idx] = np.random.uniform(
                ranges["motor_noise_std"][0], ranges["motor_noise_std"][1]
            )
    return batch_params


def q_dot(q, w):
    """Quaternion derivative for body-frame angular velocity."""
    q = np.asarray(q, dtype=np.float64)
    return qdotb(q / np.linalg.norm(q), w)


def q_rot(q):
    """Convert quaternion to rotation matrix."""
    return UnitQuaternion(q).R


def q_dot_batch(q, w):
    """Convert batched angular velocity to batched quaternion derivative."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return np.stack([qdotb(qi, wi) for qi, wi in zip(q, w)], axis=0)


def q_rot_batch(q):
    """Convert batched quaternions to batched rotation matrices."""
    return UnitQuaternion(q).R


def rk4_step(f, x, u, dt, **kwargs):
    """Runge-Kutta 4th order integration."""
    k1 = f(x, u, **kwargs)
    k2 = f(x + 0.5 * dt * k1, u, **kwargs)
    k3 = f(x + 0.5 * dt * k2, u, **kwargs)
    k4 = f(x + dt * k3, u, **kwargs)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


"""
World Frame: NED-like, with gravity along +z and x forward, y right. 
Body Frame: x forward, y right, z down. Rotor speeds are in the body frame.
State is [p, v, q, w, omega]. Control input is rotor speed: omega.
X convention quadrotor with 4 rotors, rotor 1 is front left, then going clockwise.
"""


class Quadrotor:
    def __init__(self, params: QuadrotorParams):
        if params.motor_time_constant < 0.0:
            raise ValueError("motor_time_constant must be nonnegative")
        self.params = params
        self._mix_mat = Quadrotor.x_quad_mixing_matrix(
            params.arm_length,
            params.thrust_coeff,
            params.torque_coeff,
            params.rotor_direction,
        )
        self._inv_mix_mat = np.linalg.inv(self._mix_mat)
        self._inv_J = np.linalg.inv(params.inertia)

    @staticmethod
    def x_quad_mixing_matrix(l, kf, km, rotor_direction):
        """Mixing matrix for x configuration quadrotor."""
        a = l / np.sqrt(2) * kf
        s1, s2, s3, s4 = rotor_direction
        return np.array(
            [
                [kf, kf, kf, kf],
                [a, -a, -a, a],
                [a, a, -a, -a],
                [s1 * km, s2 * km, s3 * km, s4 * km],
            ],
            dtype=np.float64,
        )

    def rotor_speed_to_wrench(self, omega: np.ndarray):
        assert omega.shape == (4,)
        omega = np.asarray(omega, dtype=np.float64)
        omega = np.clip(omega, self.params.rotor_min_speed, self.params.rotor_max_speed)
        return self._mix_mat @ (omega**2)

    def wrench_to_rotor_speed(self, wrench: np.ndarray):
        assert wrench.shape == (4,)
        wrench = np.asarray(wrench, dtype=np.float64)
        omega = np.sqrt(np.maximum(self._inv_mix_mat @ wrench, 0.0))
        return np.clip(omega, self.params.rotor_min_speed, self.params.rotor_max_speed)

    @staticmethod
    def body_drag_force(vel_body, quad_params):
        return -quad_params.drag_force_coeff * vel_body

    @staticmethod
    def body_drag_torque(w_body, quad_params):
        return -quad_params.drag_torque_coeff * w_body

    def x_dot(self, x: np.ndarray, u: np.ndarray, external_torque=None):
        assert x.shape == (17,)
        assert u.shape == (4,)
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        vel = x[3:6]
        q = x[6:10]
        w = x[10:13]
        eta = x[13:17]
        rot = q_rot(q)

        rotor_speed = u if self.params.motor_time_constant == 0.0 else eta
        wrench = self.rotor_speed_to_wrench(rotor_speed)

        f_thrust = np.array([0, 0, -wrench[0]], dtype=np.float64)
        f_drag = Quadrotor.body_drag_force(rot.T @ vel, self.params)
        f_world = rot @ (f_thrust + f_drag)
        g = np.array([0, 0, self.params.g], dtype=np.float64)
        acc = g + f_world / self.params.mass

        q_deriv = q_dot(q, w)
        J = self.params.inertia
        tau_cmd = wrench[1:4]
        external_torque = (
            np.zeros(3)
            if external_torque is None
            else np.asarray(external_torque, dtype=np.float64)
        )
        if external_torque.shape != (3,):
            raise ValueError("external_torque must have shape (3,)")
        tau_drag = Quadrotor.body_drag_torque(w, self.params)
        w_deriv = self._inv_J @ (
            tau_cmd + external_torque + tau_drag - np.cross(w, J @ w)
        )

        u = np.clip(u, self.params.rotor_min_speed, self.params.rotor_max_speed)
        eta_deriv = (
            np.zeros(4)
            if self.params.motor_time_constant == 0.0
            else (u - eta) / self.params.motor_time_constant
        )

        return np.concatenate([vel, acc, q_deriv, w_deriv, eta_deriv])

    def x_dot_wrench(self, x: np.ndarray, u: np.ndarray):
        return self.x_dot(x, self.wrench_to_rotor_speed(u))

    def step(self, x, u, dt, **kwargs):
        """
        u: [rotor_speed1, rotor_speed2, rotor_speed3, rotor_speed4]
        """
        x_next = rk4_step(self.x_dot, x, u, dt, **kwargs)
        q = x_next[6:10]
        x_next[6:10] = q / np.linalg.norm(q)
        if self.params.motor_time_constant == 0.0:
            x_next[13:17] = np.clip(
                u,
                self.params.rotor_min_speed,
                self.params.rotor_max_speed,
            )
        x_next[13:17] += np.random.normal(
            scale=np.abs(self.params.motor_noise_std), size=(4,)
        )
        x_next[13:17] = np.clip(
            x_next[13:17],
            self.params.rotor_min_speed,
            self.params.rotor_max_speed,
        )
        return x_next

    def step_wrench(self, x, u, dt, **kwargs):
        """
        u: [thrust, tau_x, tau_y, tau_z]
        """
        return self.step(x, self.wrench_to_rotor_speed(u), dt, **kwargs)


class BatchQuadrotor:
    """Vectorized quadrotor dynamics for x: (N, 17), u: (N, 4)."""

    def __init__(self, params: BatchQuadrotorParams):
        self.params = params
        self.batch_size = self.params.batch_size
        self._rotor_min_speed = self.params.rotor_min_speed
        self._rotor_max_speed = self.params.rotor_max_speed
        self._batch_inv_J = np.linalg.inv(self.params.inertia)
        self._batch_mix_mat = self._make_batch_mixing_matrix()
        self._batch_inv_mix_mat = np.linalg.inv(self._batch_mix_mat)
        self._inv_mass = 1.0 / self.params.mass
        if np.any(self.params.motor_time_constant < 0.0):
            raise ValueError("motor_time_constant must be nonnegative")
        self._instantaneous_motor = self.params.motor_time_constant == 0.0
        self._inv_motor_time_constant = np.divide(
            1.0,
            self.params.motor_time_constant,
            out=np.zeros_like(self.params.motor_time_constant),
            where=~self._instantaneous_motor,
        )

    def _make_batch_mixing_matrix(self):
        a = self.params.arm_length / np.sqrt(2) * self.params.thrust_coeff
        mix = np.empty((self.batch_size, 4, 4), dtype=np.float64)
        mix[:, 0, :] = self.params.thrust_coeff[:, None]
        mix[:, 1, 0] = a
        mix[:, 1, 1] = -a
        mix[:, 1, 2] = -a
        mix[:, 1, 3] = a
        mix[:, 2, 0] = a
        mix[:, 2, 1] = a
        mix[:, 2, 2] = -a
        mix[:, 2, 3] = -a
        mix[:, 3, :] = self.params.rotor_direction * self.params.torque_coeff[:, None]
        return mix

    def rotor_speed_to_wrench(self, omega: np.ndarray):
        assert omega.shape == (self.batch_size, 4)
        omega = np.asarray(omega, dtype=np.float64)
        omega = np.clip(
            omega,
            self._rotor_min_speed[:, None],
            self._rotor_max_speed[:, None],
        )
        return np.einsum("bij,bj->bi", self._batch_mix_mat, omega**2)

    def wrench_to_rotor_speed(self, wrench: np.ndarray):
        assert wrench.shape == (self.batch_size, 4)
        wrench = np.asarray(wrench, dtype=np.float64)
        omega_sq = np.einsum("bij,bj->bi", self._batch_inv_mix_mat, wrench)
        omega = np.sqrt(np.maximum(omega_sq, 0.0))
        return np.clip(
            omega,
            self._rotor_min_speed[:, None],
            self._rotor_max_speed[:, None],
        )

    def x_dot(self, x: np.ndarray, u: np.ndarray):
        assert x.shape == (self.params.batch_size, 17)
        assert u.shape == (self.params.batch_size, 4)
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        vel = x[:, 3:6]
        q = x[:, 6:10]
        w = x[:, 10:13]
        eta = x[:, 13:17]
        rot = q_rot_batch(q)

        rotor_speed = np.where(self._instantaneous_motor[:, None], u, eta)
        wrench = self.rotor_speed_to_wrench(rotor_speed)
        vel_body = np.einsum("bji,bj->bi", rot, vel)
        f_body = -self.params.drag_force_coeff * vel_body
        f_body[:, 2] -= wrench[:, 0]
        f_world = np.einsum("bij,bj->bi", rot, f_body)
        acc = f_world * self._inv_mass[:, None]
        acc[:, 2] += self.params.g

        q_deriv = q_dot_batch(q, w)
        J = self.params.inertia
        tau = (
            wrench[:, 1:4]
            - self.params.drag_torque_coeff * w
            - np.cross(w, np.einsum("bij,bj->bi", J, w))
        )
        w_deriv = np.einsum("bij,bj->bi", self._batch_inv_J, tau)

        u = np.clip(
            u,
            self._rotor_min_speed[:, None],
            self._rotor_max_speed[:, None],
        )
        eta_deriv = (u - eta) * self._inv_motor_time_constant[:, None]

        out = np.empty_like(x)
        out[:, 0:3] = vel
        out[:, 3:6] = acc
        out[:, 6:10] = q_deriv
        out[:, 10:13] = w_deriv
        out[:, 13:17] = eta_deriv
        return out

    def x_dot_wrench(self, x: np.ndarray, u: np.ndarray):
        return self.x_dot(x, self.wrench_to_rotor_speed(u))

    def step(self, x, u, dt, **kwargs):
        x_next = rk4_step(self.x_dot, x, u, dt, **kwargs)
        q = x_next[:, 6:10]
        x_next[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
        x_next[self._instantaneous_motor, 13:17] = np.clip(
            u[self._instantaneous_motor],
            self._rotor_min_speed[self._instantaneous_motor, None],
            self._rotor_max_speed[self._instantaneous_motor, None],
        )
        x_next[:, 13:17] += np.random.normal(
            scale=np.abs(self.params.motor_noise_std)[:, None],
            size=(self.batch_size, 4),
        )
        x_next[:, 13:17] = np.clip(
            x_next[:, 13:17],
            self._rotor_min_speed[:, None],
            self._rotor_max_speed[:, None],
        )
        return x_next

    def step_wrench(self, x, u, dt, **kwargs):
        return self.step(x, self.wrench_to_rotor_speed(u), dt, **kwargs)


class CtbrCnt:
    """Body-rate PI controller: CTBR -> wrench.

    CTBR is ``[collective_acceleration, p, q, r]``. Motor allocation and
    vehicle simulation are deliberately outside this controller.
    """

    def __init__(
        self,
        mass,
        rate_kp=(0.5825, 0.5825, 0.55225),
        rate_ki=(0.1165, 0.1165, 0.11045),
        integral_limit=(0.03495, 0.03495, 0.033135),
        thrust_accel_limits=None,
    ):
        self.mass = float(mass)
        self.rate_kp = self._vector3(rate_kp, "rate_kp")
        self.rate_ki = self._vector3(rate_ki, "rate_ki")
        self.integral_limit = self._vector3(integral_limit, "integral_limit")
        self.thrust_accel_limits = np.asarray(
            [0.0, np.inf] if thrust_accel_limits is None else thrust_accel_limits,
            dtype=np.float64,
        )
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.thrust_accel_limits.shape != (2,):
            raise ValueError("thrust_accel_limits must have shape (2,)")

        self.rate_integral = np.zeros(3)
        self.last_wrench = np.zeros(4)
        self.last_rate_error = np.zeros(3)

    @staticmethod
    def _vector3(value, name):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have shape (3,) and be finite")
        return value

    def reset(self):
        self.rate_integral.fill(0.0)

    def control(
        self,
        body_rate,
        ctbr,
        dt,
        saturation_positive=None,
        saturation_negative=None,
        torque_ff=None,
    ):
        """Return ``[thrust, tau_x, tau_y, tau_z]``."""
        body_rate = self._vector3(body_rate, "body_rate")
        ctbr = np.asarray(ctbr, dtype=np.float64)
        if ctbr.shape != (4,) or not np.all(np.isfinite(ctbr)):
            raise ValueError("ctbr must have shape (4,) and be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive")

        sat_pos = (
            np.zeros(3, dtype=bool)
            if saturation_positive is None
            else np.asarray(saturation_positive, dtype=bool)
        )
        sat_neg = (
            np.zeros(3, dtype=bool)
            if saturation_negative is None
            else np.asarray(saturation_negative, dtype=bool)
        )
        torque_ff = (
            np.zeros(3) if torque_ff is None else self._vector3(torque_ff, "torque_ff")
        )
        if sat_pos.shape != (3,) or sat_neg.shape != (3,):
            raise ValueError("saturation flags must have shape (3,)")

        rate_error = ctbr[1:4] - body_rate
        torque = self.rate_kp * rate_error + self.rate_integral + torque_ff

        # Do not integrate farther into an actuator-saturated direction.
        integration_error = rate_error.copy()
        integration_error[sat_pos & (integration_error > 0.0)] = 0.0
        integration_error[sat_neg & (integration_error < 0.0)] = 0.0
        i_factor = np.maximum(0.0, 1.0 - (integration_error / np.deg2rad(400.0)) ** 2)
        self.rate_integral = np.clip(
            self.rate_integral + i_factor * self.rate_ki * integration_error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        acceleration = np.clip(ctbr[0], *self.thrust_accel_limits)
        wrench = np.r_[self.mass * acceleration, torque]
        self.last_rate_error = rate_error
        self.last_wrench = wrench
        return wrench

    __call__ = control
