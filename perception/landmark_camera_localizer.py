"""
landmark_camera_localizer.py

Synthetic landmark-based camera localizer.
Designed to be replaced by a real visual front-end later.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, pi
from typing import List, Optional

import numpy as np


@dataclass
class LandmarkObservation:
    """Single landmark correspondence for camera-based localization."""

    world_xy: np.ndarray
    relative_xy_vehicle: np.ndarray


class LandmarkCameraLocalizer:
    """
    Simulates visual landmark observations and estimates vehicle XY from them.

    Assumption:
    - Landmark IDs are known in the synthetic world.
    - Current yaw estimate is available from inertial integration.
    """

    def __init__(
        self,
        max_range_m: float = 24.0,
        fov_deg: float = 110.0,
        rel_noise_std_m: float = 0.20,
        min_landmarks: int = 4,
        seed: int = 11,
    ) -> None:
        self.max_range_m = max_range_m
        self.fov_rad = np.deg2rad(fov_deg)
        self.rel_noise_std_m = rel_noise_std_m
        self.min_landmarks = min_landmarks
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _wrap_pi(angle_rad: float) -> float:
        while angle_rad > pi:
            angle_rad -= 2.0 * pi
        while angle_rad < -pi:
            angle_rad += 2.0 * pi
        return angle_rad

    def simulate_observations(
        self,
        true_x_m: float,
        true_y_m: float,
        true_yaw_rad: float,
        landmarks_xy: np.ndarray,
    ) -> List[LandmarkObservation]:
        """Generate synthetic landmark correspondences visible to the camera."""

        observations: List[LandmarkObservation] = []
        c_yaw = cos(true_yaw_rad)
        s_yaw = sin(true_yaw_rad)

        for world_xy in landmarks_xy:
            diff_x = float(world_xy[0] - true_x_m)
            diff_y = float(world_xy[1] - true_y_m)
            distance = np.hypot(diff_x, diff_y)
            if distance > self.max_range_m:
                continue

            bearing = self._wrap_pi(atan2(diff_y, diff_x) - true_yaw_rad)
            if abs(bearing) > self.fov_rad / 2.0:
                continue

            # Convert global delta into vehicle frame.
            rel_x = c_yaw * diff_x + s_yaw * diff_y
            rel_y = -s_yaw * diff_x + c_yaw * diff_y
            rel_xy = np.array([
                rel_x + self.rng.normal(0.0, self.rel_noise_std_m),
                rel_y + self.rng.normal(0.0, self.rel_noise_std_m),
            ])

            observations.append(
                LandmarkObservation(world_xy=np.asarray(world_xy, dtype=float), relative_xy_vehicle=rel_xy)
            )

        return observations

    def estimate_xy(
        self,
        observations: List[LandmarkObservation],
        yaw_est_rad: float,
    ) -> tuple[Optional[np.ndarray], float]:
        """
        Estimate vehicle XY from landmark correspondences and yaw estimate.

        Returns:
        - estimated_xy (2x1 ndarray) or None
        - confidence in [0, 1]
        """

        if len(observations) < self.min_landmarks:
            return None, 0.0

        c_yaw = cos(yaw_est_rad)
        s_yaw = sin(yaw_est_rad)
        rot = np.array([[c_yaw, -s_yaw], [s_yaw, c_yaw]], dtype=float)

        xy_candidates = []
        for obs in observations:
            # world = R(yaw)*relative + translation -> translation = world - R(yaw)*relative
            translation_xy = obs.world_xy.reshape(2, 1) - rot @ obs.relative_xy_vehicle.reshape(2, 1)
            xy_candidates.append(translation_xy)

        stacked = np.hstack(xy_candidates)
        estimated_xy = np.mean(stacked, axis=1, keepdims=True)

        residuals = stacked - estimated_xy
        mean_residual = float(np.mean(np.linalg.norm(residuals, axis=0)))
        raw_conf = len(observations) / (len(observations) + 6.0)
        residual_gate = float(np.clip(1.0 - mean_residual / 2.5, 0.0, 1.0))
        confidence = float(np.clip(raw_conf * residual_gate, 0.0, 1.0))

        return estimated_xy, confidence
