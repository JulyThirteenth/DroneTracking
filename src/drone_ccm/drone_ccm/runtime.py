"""Inference-only adapter for the maintained ctbr_cnt checkpoint."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from drone_ccm.geometry import log as so3_log

_CONTROLLER_ARCH = "factorized_control_bounds_c3m_v9_axis_angle"
_EGO_ARCH = "ego_s2_r3_s1_c3m_v4"
_ROTATION_SAMPLING = "yaw_tilt_axis_angle_boundary_v1"
_GRAVITY = 9.81


@dataclasses.dataclass(frozen=True)
class CcmDomain:
    """Deployment domain reconstructed from one ctbr_cnt checkpoint."""

    gravity: float
    velocity_reference_max: float | None
    velocity_error_max: float
    reference_tilt_angle_max: float
    rotation_error_angle_max: float
    reference_thrust_min: float
    reference_thrust_max: float
    reference_body_rate_max: tuple[float, float, float]
    control_lower: tuple[float, float, float, float]
    control_upper: tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class EgoCcmDomain(CcmDomain):
    """Deployment bounds specific to the ego-centric checkpoint."""

    actual_tilt_angle_max: float
    actual_pitch_angle_max: float


class _Mlp(nn.Module):
    """Network layout used by ctbr_cnt.CTBRController."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


def _bounded_control(
    reference: Tensor,
    feedback: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Maps residual feedback into the checkpoint's physical control bounds."""
    span = upper - lower
    ratio = ((reference - lower) / span).clamp(1.0e-6, 1.0 - 1.0e-6)
    margin = ((reference - lower) * (upper - reference) / span).clamp_min(
        torch.finfo(feedback.dtype).eps
    )
    return lower + span * torch.sigmoid(torch.logit(ratio) + feedback / margin)


class _CtbrController(nn.Module):
    """Inference-compatible copy of the maintained ctbr_cnt controller."""

    _CONTROL_DIM = 4
    _ERROR_DIM = 6
    _CONTEXT_DIM = 28
    _FACTOR_DIM = 18

    def __init__(
        self,
        hidden_dim: int,
        control_lower: tuple[float, ...],
        control_upper: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.factor_dim = self._FACTOR_DIM
        self.factor_input = _Mlp(
            self._CONTEXT_DIM,
            self._FACTOR_DIM * self._ERROR_DIM,
            hidden_dim,
        )
        self.factor_output = _Mlp(
            self._CONTEXT_DIM,
            self._CONTROL_DIM * self._FACTOR_DIM,
            hidden_dim,
        )
        self.register_buffer(
            "control_lower",
            torch.tensor(control_lower),
            persistent=False,
        )
        self.register_buffer(
            "control_upper",
            torch.tensor(control_upper),
            persistent=False,
        )

    @staticmethod
    def _attitude_error(rotation: Tensor, reference: Tensor) -> Tensor:
        relative_skew = (
            reference.transpose(-1, -2) @ rotation
            - rotation.transpose(-1, -2) @ reference
        )
        return 0.5 * torch.stack(
            (
                relative_skew[..., 2, 1],
                relative_skew[..., 0, 2],
                relative_skew[..., 1, 0],
            ),
            dim=-1,
        )

    def forward(
        self,
        velocity: Tensor,
        rotation: Tensor,
        reference_velocity: Tensor,
        reference_rotation: Tensor,
        reference_control: Tensor,
    ) -> Tensor:
        error = torch.cat(
            (
                velocity - reference_velocity,
                self._attitude_error(rotation, reference_rotation),
            ),
            dim=-1,
        )
        context = torch.cat(
            (
                velocity,
                rotation.flatten(-2),
                reference_velocity,
                reference_rotation.flatten(-2),
                reference_control,
            ),
            dim=-1,
        )
        factor_input = self.factor_input(context).reshape(
            -1,
            self._FACTOR_DIM,
            self._ERROR_DIM,
        )
        factor_output = self.factor_output(context).reshape(
            -1,
            self._CONTROL_DIM,
            self._FACTOR_DIM,
        )
        hidden = torch.tanh(torch.einsum("bij,bj->bi", factor_input, error))
        feedback = torch.einsum("bij,bj->bi", factor_output, hidden)

        return _bounded_control(
            reference_control,
            feedback,
            self.control_lower,
            self.control_upper,
        )


class _EgoCtbrController(nn.Module):
    """Inference-compatible copy of the ego-centric CTBR controller."""

    _CONTROL_DIM = 4
    _ERROR_DIM = 6
    _CONTEXT_DIM = 16
    _FACTOR_DIM = 18

    def __init__(
        self,
        hidden_dim: int,
        control_lower: tuple[float, ...],
        control_upper: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.factor_input = _Mlp(
            self._CONTEXT_DIM,
            self._FACTOR_DIM * self._ERROR_DIM,
            hidden_dim,
        )
        self.factor_output = _Mlp(
            self._CONTEXT_DIM,
            self._CONTROL_DIM * self._FACTOR_DIM,
            hidden_dim,
        )
        self.linear_gain = nn.Parameter(torch.zeros(self._CONTROL_DIM, self._ERROR_DIM))
        self.register_buffer(
            "control_lower", torch.tensor(control_lower), persistent=False
        )
        self.register_buffer(
            "control_upper", torch.tensor(control_upper), persistent=False
        )

    def forward(
        self,
        state: Tensor,
        reference_acceleration_body: Tensor,
        reference_yaw_rate: Tensor,
        reference_control: Tensor,
    ) -> Tensor:
        gamma = state[..., :3]
        desired_body_z = functional.normalize(
            reference_acceleration_body + _GRAVITY * gamma,
            dim=-1,
        )
        tilt_error = torch.stack(
            (desired_body_z[..., 1], -desired_body_z[..., 0]), dim=-1
        )
        error = torch.cat((state[..., 3:6], tilt_error, state[..., 6:7]), dim=-1)
        features = torch.cat(
            (
                gamma,
                state[..., 3:6],
                torch.sin(state[..., 6:7]),
                torch.cos(state[..., 6:7]),
            ),
            dim=-1,
        )
        context = torch.cat(
            (
                features,
                reference_acceleration_body,
                reference_yaw_rate[..., None],
                reference_control,
            ),
            dim=-1,
        )
        left = self.factor_input(context).reshape(-1, self._FACTOR_DIM, self._ERROR_DIM)
        right = self.factor_output(context).reshape(
            -1, self._CONTROL_DIM, self._FACTOR_DIM
        )
        hidden = torch.tanh(torch.einsum("bij,bj->bi", left, error))
        feedback = torch.einsum("ij,bj->bi", self.linear_gain, error)
        feedback += torch.einsum("bij,bj->bi", right, hidden)

        return _bounded_control(
            reference_control,
            feedback,
            self.control_lower,
            self.control_upper,
        )


def _domain_from_config(config: dict[str, object]) -> CcmDomain:
    sampling = config.get("sampling_ranges")
    if not isinstance(sampling, dict):
        raise ValueError("checkpoint is missing sampling_ranges")
    reference_lower = tuple(float(value) for value in config["reference_lower"])
    reference_upper = tuple(float(value) for value in config["reference_upper"])
    control_lower = tuple(float(value) for value in config["control_lower"])
    control_upper = tuple(float(value) for value in config["control_upper"])
    if any(
        len(values) != 4
        for values in (
            reference_lower,
            reference_upper,
            control_lower,
            control_upper,
        )
    ):
        raise ValueError("checkpoint CTBR bounds must contain four values")
    reference_rates = tuple(
        max(abs(lower), abs(upper))
        for lower, upper in zip(reference_lower[1:], reference_upper[1:])
    )
    return CcmDomain(
        gravity=_GRAVITY,
        velocity_reference_max=float(sampling["velocity_range"]),
        velocity_error_max=float(sampling["velocity_error"]),
        reference_tilt_angle_max=float(sampling["reference_tilt_angle_max"]),
        rotation_error_angle_max=float(sampling["rotation_error_angle_max"]),
        reference_thrust_min=reference_lower[0],
        reference_thrust_max=reference_upper[0],
        reference_body_rate_max=reference_rates,
        control_lower=control_lower,
        control_upper=control_upper,
    )


def _ego_domain_from_config(config: dict[str, object]) -> EgoCcmDomain:
    control_lower = tuple(float(value) for value in config["control_lower"])
    control_upper = tuple(float(value) for value in config["control_upper"])
    body_rate_max = tuple(float(value) for value in config["body_rate_max"])
    if len(control_lower) != 4 or len(control_upper) != 4 or len(body_rate_max) != 3:
        raise ValueError("ego checkpoint contains invalid CTBR bounds")
    collective_delta = float(config["collective_delta"])
    return EgoCcmDomain(
        gravity=_GRAVITY,
        velocity_reference_max=None,
        velocity_error_max=float(config["velocity_error_max"]),
        reference_tilt_angle_max=float(config["reference_tilt_max"]),
        rotation_error_angle_max=float(config["rotation_error_max"]),
        reference_thrust_min=_GRAVITY - collective_delta,
        reference_thrust_max=_GRAVITY + collective_delta,
        reference_body_rate_max=body_rate_max,
        control_lower=control_lower,
        control_upper=control_upper,
        actual_tilt_angle_max=float(config["actual_tilt_max"]),
        actual_pitch_angle_max=float(config["actual_pitch_max"]),
    )


def _read_payload(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing config")
    return payload, config


def load_domain(checkpoint_path: Path) -> CcmDomain:
    """Load the authoritative domain without constructing the model."""
    _, config = _read_payload(checkpoint_path, torch.device("cpu"))
    if config.get("architecture") == _EGO_ARCH:
        return _ego_domain_from_config(config)
    _validate_standard_config(config)
    return _domain_from_config(config)


def _validate_standard_config(config: dict[str, object]) -> None:
    if (
        config.get("controller_arch") != _CONTROLLER_ARCH
        or config.get("controller_type") != "residual"
        or config.get("rotation_sampling") != _ROTATION_SAMPLING
    ):
        raise ValueError("checkpoint is not a maintained CCM model")
    if int(config.get("factor_dim", -1)) != _CtbrController._FACTOR_DIM:
        raise ValueError("checkpoint factor_dim is incompatible")


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError("dtype must be float32 or float64") from error


def _as_tensor(
    values: np.ndarray,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    array = np.asarray(values, dtype=float)
    if array.shape != shape[1:] or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected finite array with shape {shape[1:]}")
    return torch.as_tensor(array, device=device, dtype=dtype).reshape(shape)


def _runtime_inputs(
    velocity: np.ndarray,
    rotation: np.ndarray,
    reference_velocity: np.ndarray,
    reference_rotation: np.ndarray,
    reference_control: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Converts and validates the common deployment interface."""
    values = (
        _as_tensor(velocity, (1, 3), device, dtype),
        _as_tensor(rotation, (1, 3, 3), device, dtype),
        _as_tensor(reference_velocity, (1, 3), device, dtype),
        _as_tensor(reference_rotation, (1, 3, 3), device, dtype),
        _as_tensor(reference_control, (1, 4), device, dtype),
    )
    _validate_rotation(values[1], "rotation")
    _validate_rotation(values[3], "reference_rotation")
    return values


def _validate_rotation(matrix: Tensor, name: str) -> None:
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    orthogonality = torch.linalg.matrix_norm(
        matrix.transpose(-1, -2) @ matrix - identity
    )
    if bool((orthogonality > 1.0e-4).any()) or bool(
        (torch.linalg.det(matrix) <= 0.0).any()
    ):
        raise ValueError(f"{name} is not a valid SO(3) matrix")


def _validate_reference(
    domain: CcmDomain,
    rotation: Tensor,
    control: Tensor,
) -> None:
    tilt = torch.acos(rotation[..., 2, 2].clamp(-1.0, 1.0))
    if bool((tilt > domain.reference_tilt_angle_max).any()):
        raise RuntimeError("Reference attitude is outside the training domain")
    rate_limit = torch.as_tensor(
        domain.reference_body_rate_max,
        dtype=control.dtype,
        device=control.device,
    )
    thrust = control[..., 0]
    if bool(
        (thrust < domain.reference_thrust_min).any()
        or (thrust > domain.reference_thrust_max).any()
        or (control[..., 1:].abs() > rate_limit).any()
    ):
        raise RuntimeError("CTBR reference is outside the training domain")


def _control_numpy(control: Tensor) -> np.ndarray:
    result = control[0].cpu().numpy().astype(float)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Controller produced a non-finite CTBR command")
    return result


class _LieCcmRuntime:
    """Runs the ctbr_cnt controller through the ROS deployment interface."""

    def __init__(
        self,
        payload: dict[str, object],
        config: dict[str, object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load one standard Lie-CCM checkpoint."""
        self.device = device
        self.dtype = dtype
        self.domain = _domain_from_config(config)
        self.controller = _CtbrController(
            int(config.get("controller_hidden", config["hidden"])),
            self.domain.control_lower,
            self.domain.control_upper,
        ).to(device=self.device, dtype=self.dtype)
        self.controller.load_state_dict(payload["controller"], strict=True)
        self.controller.eval()

    @property
    def gravity(self) -> float:
        """Return gravitational acceleration used during training."""
        return self.domain.gravity

    def command(
        self,
        velocity: np.ndarray,
        rotation: np.ndarray,
        reference_velocity: np.ndarray,
        reference_rotation: np.ndarray,
        reference_control: np.ndarray,
    ) -> np.ndarray:
        """Validate the trained domain and return physical FLU CTBR."""

        (
            velocity_tensor,
            rotation_tensor,
            reference_velocity_tensor,
            reference_rotation_tensor,
            reference_control_tensor,
        ) = _runtime_inputs(
            velocity,
            rotation,
            reference_velocity,
            reference_rotation,
            reference_control,
            device=self.device,
            dtype=self.dtype,
        )

        velocity_error = velocity_tensor - reference_velocity_tensor
        rotation_error = so3_log(
            reference_rotation_tensor.transpose(-1, -2) @ rotation_tensor
        )
        rotation_angle = torch.linalg.vector_norm(rotation_error, dim=-1)
        if bool((velocity_error.abs() > self.domain.velocity_error_max).any()) or bool(
            (rotation_angle > self.domain.rotation_error_angle_max).any()
        ):
            raise RuntimeError("State is outside the checkpoint tracking-error domain")
        if bool(
            (reference_velocity_tensor.abs() > self.domain.velocity_reference_max).any()
        ):
            raise RuntimeError("Velocity reference is outside the training domain")
        _validate_reference(
            self.domain, reference_rotation_tensor, reference_control_tensor
        )

        with torch.inference_mode():
            control = self.controller(
                velocity_tensor,
                rotation_tensor,
                reference_velocity_tensor,
                reference_rotation_tensor,
                reference_control_tensor,
            )
        return _control_numpy(control)

    def warmup(self) -> None:
        """Initialize device kernels at hover equilibrium."""
        self.command(
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array((self.gravity, 0.0, 0.0, 0.0)),
        )


class _EgoCcmRuntime:
    """Runs an ego-centric checkpoint through the common ROS CTBR interface."""

    def __init__(
        self,
        payload: dict[str, object],
        config: dict[str, object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Load one ego-centric checkpoint."""
        self.device = device
        self.dtype = dtype
        self.domain = _ego_domain_from_config(config)
        self.controller = _EgoCtbrController(
            int(config["hidden"]),
            self.domain.control_lower,
            self.domain.control_upper,
        ).to(device=self.device, dtype=self.dtype)
        self.controller.load_state_dict(payload["controller"], strict=True)
        self.controller.eval()

    @property
    def gravity(self) -> float:
        """Return gravitational acceleration used during training."""
        return self.domain.gravity

    def command(
        self,
        velocity: np.ndarray,
        rotation: np.ndarray,
        reference_velocity: np.ndarray,
        reference_rotation: np.ndarray,
        reference_control: np.ndarray,
    ) -> np.ndarray:
        """Build ego state/reference features and return physical FLU CTBR."""

        (
            velocity,
            rotation,
            reference_velocity,
            reference_rotation,
            reference_control,
        ) = _runtime_inputs(
            velocity,
            rotation,
            reference_velocity,
            reference_rotation,
            reference_control,
            device=self.device,
            dtype=self.dtype,
        )

        rotation_error = so3_log(reference_rotation.transpose(-1, -2) @ rotation)
        if bool(
            (
                torch.linalg.vector_norm(rotation_error, dim=-1)
                > self.domain.rotation_error_angle_max
            ).any()
        ):
            raise RuntimeError("State is outside the checkpoint attitude-error domain")

        gamma = rotation[..., 2, :]
        velocity_error = (
            rotation.transpose(-1, -2) @ (velocity - reference_velocity)[..., None]
        ).squeeze(-1)
        if bool((velocity_error.abs() > self.domain.velocity_error_max).any()):
            raise RuntimeError("State is outside the checkpoint velocity-error domain")

        actual_tilt = torch.acos(gamma[..., 2].clamp(-1.0, 1.0))
        actual_pitch = -torch.asin(gamma[..., 0].clamp(-1.0, 1.0))
        if bool((actual_tilt > self.domain.actual_tilt_angle_max).any()) or bool(
            (actual_pitch.abs() > self.domain.actual_pitch_angle_max).any()
        ):
            raise RuntimeError("State is outside the checkpoint attitude domain")

        _validate_reference(self.domain, reference_rotation, reference_control)
        thrust = reference_control[..., 0]

        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        reference_yaw = torch.atan2(
            reference_rotation[..., 1, 0], reference_rotation[..., 0, 0]
        )
        yaw_error = torch.atan2(
            torch.sin(yaw - reference_yaw), torch.cos(yaw - reference_yaw)
        )
        state = torch.cat((gamma, velocity_error, yaw_error[..., None]), dim=-1)

        e3 = torch.tensor((0.0, 0.0, 1.0), dtype=self.dtype, device=self.device)
        reference_acceleration_world = (
            -self.gravity * e3 + thrust[..., None] * reference_rotation[..., :, 2]
        )
        reference_acceleration_body = (
            rotation.transpose(-1, -2) @ reference_acceleration_world[..., None]
        ).squeeze(-1)
        gamma_reference = reference_rotation[..., 2, :]
        denominator = (
            gamma_reference[..., 1].square() + gamma_reference[..., 2].square()
        ).clamp_min(1.0e-6)
        reference_yaw_rate = (
            reference_control[..., 2] * gamma_reference[..., 1]
            + reference_control[..., 3] * gamma_reference[..., 2]
        ) / denominator

        with torch.inference_mode():
            control = self.controller(
                state,
                reference_acceleration_body,
                reference_yaw_rate,
                reference_control,
            )
        return _control_numpy(control)

    def warmup(self) -> None:
        """Initialize device kernels at hover equilibrium."""
        self.command(
            np.zeros(3),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array((self.gravity, 0.0, 0.0, 0.0)),
        )


def load_runtime(
    checkpoint_path: Path,
    *,
    device_name: str,
    dtype_name: str,
) -> _LieCcmRuntime | _EgoCcmRuntime:
    """Select the runtime encoded by the checkpoint architecture."""
    device = torch.device(device_name)
    dtype = _dtype_from_name(dtype_name)
    payload, config = _read_payload(checkpoint_path, device)
    if config.get("architecture") == _EGO_ARCH:
        return _EgoCcmRuntime(payload, config, device, dtype)
    _validate_standard_config(config)
    return _LieCcmRuntime(payload, config, device, dtype)
