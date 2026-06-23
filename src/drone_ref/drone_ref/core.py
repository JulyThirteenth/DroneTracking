"""Controller-independent reference-path logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_ref.util import as_vec3, wrap_pi

_MIN_PATH_LENGTH = 1e-6
_CLOSE_LOOP_TOLERANCE = 1e-6
_MIN_YAW_DISTANCE = 1e-6


@dataclass(frozen=True)
class PathProgress:
    """Current arc-length progress along a reference path."""

    distance: float


class ReferencePath:
    """A 3-D ENU polyline parameterized by arc length."""

    def __init__(self, *, loop: bool = False) -> None:
        self._loop = bool(loop)
        self._points = np.empty((0, 3), dtype=float)
        self._arc_lengths = np.empty(0, dtype=float)
        self._last_distance: float | None = None

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()

    @property
    def length(self) -> float:
        if self._arc_lengths.size == 0:
            return 0.0

        return float(self._arc_lengths[-1])

    @property
    def valid(self) -> bool:
        return (
            self._points.shape[0] >= 2
            and np.isfinite(self.length)
            and self.length > _MIN_PATH_LENGTH
        )

    def set_path(self, points_enu: np.ndarray) -> None:
        """Set the reference path using points shaped ``(M, 3)``."""
        points = self._normalize_points(points_enu)
        points = self._remove_consecutive_duplicates(points)

        if points.shape[0] < 2:
            raise ValueError(
                "Reference path requires at least two distinct points"
            )

        if (
            self._loop
            and np.linalg.norm(points[-1] - points[0]) > _CLOSE_LOOP_TOLERANCE
        ):
            points = np.vstack((points, points[0]))

        segment_lengths = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        )

        arc_lengths = np.concatenate(
            (
                np.array([0.0], dtype=float),
                np.cumsum(segment_lengths),
            )
        )

        if arc_lengths[-1] <= _MIN_PATH_LENGTH:
            raise ValueError("Reference path has zero length")

        self._points = points
        self._arc_lengths = arc_lengths
        self._last_distance = None

    def update_progress(
        self,
        position_enu: np.ndarray,
    ) -> PathProgress | None:
        """Project the current position onto the path."""
        if not self.valid:
            return None

        closest_distance = self._closest_arc_length(as_vec3(position_enu))

        if self._last_distance is None:
            distance = closest_distance
        elif self._loop and self._has_wrapped(closest_distance):
            distance = closest_distance
        else:
            distance = max(
                self._last_distance,
                closest_distance,
            )

        self._last_distance = float(distance)

        return PathProgress(distance=float(distance))

    def sample(self, distance: float) -> np.ndarray:
        """Sample one ENU point at the requested arc length."""
        if not self.valid:
            raise RuntimeError("Reference path is not initialized")

        sample_distance = float(distance)

        if self._loop:
            sample_distance %= self.length
        else:
            sample_distance = float(np.clip(sample_distance, 0.0, self.length))

        return np.array(
            [
                np.interp(
                    sample_distance,
                    self._arc_lengths,
                    self._points[:, axis],
                )
                for axis in range(3)
            ],
            dtype=float,
        )

    def mpc_reference(
        self,
        *,
        progress: PathProgress | None,
        current_position_enu: np.ndarray,
        horizon: int,
        sample_distance: float,
    ) -> np.ndarray:
        """Return MPC reference points shaped ``(N+1, 3)``."""
        point_count = max(1, int(horizon) + 1)

        if progress is None or not self.valid:
            current_position = as_vec3(current_position_enu)

            return np.repeat(
                current_position.reshape(1, 3),
                point_count,
                axis=0,
            )

        distances = (
            progress.distance
            + max(0.0, float(sample_distance))
            * np.arange(point_count, dtype=float)
        )

        return np.vstack([self.sample(distance) for distance in distances])

    def yaw_reference(
        self,
        *,
        progress: PathProgress | None,
        lookahead_distance: float,
        previous_yaw: float,
    ) -> float:
        """Calculate an ENU yaw following the path tangent."""
        if progress is None or not self.valid:
            return wrap_pi(previous_yaw)

        point_now = self.sample(progress.distance)
        point_ahead = self.sample(
            progress.distance + max(0.0, float(lookahead_distance))
        )

        direction = point_ahead - point_now

        if np.linalg.norm(direction[:2]) <= _MIN_YAW_DISTANCE:
            return wrap_pi(previous_yaw)

        target_yaw = float(np.arctan2(direction[1], direction[0]))

        return float(previous_yaw + wrap_pi(target_yaw - previous_yaw))

    def _closest_arc_length(
        self,
        position: np.ndarray,
    ) -> float:
        starts = self._points[:-1]
        segments = np.diff(self._points, axis=0)

        squared_lengths = np.einsum(
            "ij,ij->i",
            segments,
            segments,
        )

        offsets = position.reshape(1, 3) - starts

        projections = np.divide(
            np.einsum("ij,ij->i", offsets, segments),
            squared_lengths,
            out=np.zeros_like(squared_lengths),
            where=squared_lengths > 1e-12,
        )

        projections = np.clip(projections, 0.0, 1.0)

        projected_points = starts + projections[:, None] * segments

        errors = position.reshape(1, 3) - projected_points
        squared_errors = np.einsum(
            "ij,ij->i",
            errors,
            errors,
        )

        segment_index = int(np.argmin(squared_errors))
        segment_length = float(np.sqrt(squared_lengths[segment_index]))

        return float(
            self._arc_lengths[segment_index]
            + projections[segment_index] * segment_length
        )

    def _has_wrapped(self, closest_distance: float) -> bool:
        assert self._last_distance is not None

        return (
            self._last_distance > 0.85 * self.length
            and closest_distance < 0.15 * self.length
        )

    @staticmethod
    def _normalize_points(
        points: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(points, dtype=float)

        if array.ndim != 2:
            raise ValueError(f"Path must be a 2-D array, got {array.shape}")

        if array.shape[1] != 3:
            raise ValueError(f"Path must have shape (M, 3), got {array.shape}")

        normalized = array.copy()

        if not np.all(np.isfinite(normalized)):
            raise ValueError("Path contains non-finite values")

        return normalized

    @staticmethod
    def _remove_consecutive_duplicates(
        points: np.ndarray,
    ) -> np.ndarray:
        if points.shape[0] <= 1:
            return points.copy()

        keep = np.concatenate(
            (
                np.array([True]),
                np.linalg.norm(
                    np.diff(points, axis=0),
                    axis=1,
                )
                > 1e-9,
            )
        )

        return points[keep]
