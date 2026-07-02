"""
synthetic_route_adapter.py

Synthetic data source for camera + GNSS + control signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Iterable

import numpy as np

try:
    from ..models import ControlInput, Pose2D, SensorFrame
    from ..perception.landmark_camera_localizer import LandmarkCameraLocalizer
    from .base_adapter import BaseSensorDataAdapter
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import ControlInput, Pose2D, SensorFrame
    from perception.landmark_camera_localizer import LandmarkCameraLocalizer
    from adapters.base_adapter import BaseSensorDataAdapter

# Reuse existing motion model to keep behavior aligned with the repository.
import sys
from pathlib import Path

components_root = Path(__file__).resolve().parents[3] / "components"
sys.path.append(str(components_root / "state"))
from state import State  # type: ignore


@dataclass
class SyntheticConfig:
    total_time_s: float = 36.0
    dt_s: float = 0.1
    gnss_noise_std_m: float = 1.0
    gnss_dropout_rate: float = 0.08
    yaw_obs_noise_deg: float = 1.2
    seed: int = 22


class SyntheticRouteAdapter(BaseSensorDataAdapter):
    """Generates a closed-ish synthetic route with noisy sensors."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.camera_localizer = LandmarkCameraLocalizer(seed=self.config.seed + 10)
        self.landmarks_xy = self._create_landmarks()

    def _create_landmarks(self) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(np.linspace(-35.0, 35.0, 14), np.linspace(-20.0, 20.0, 9))
        base = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        jitter = self.rng.normal(0.0, 0.35, size=base.shape)
        return base + jitter

    def _control_profile(self, t_s: float) -> ControlInput:
        accel = 0.35 * sin(0.27 * t_s) + 0.08 * sin(1.6 * t_s)
        yaw_rate = 0.14 * sin(0.18 * t_s) + 0.09 * sin(0.045 * t_s)
        return ControlInput(accel_mps2=accel, yaw_rate_rps=yaw_rate)

    def iter_frames(self) -> Iterable[SensorFrame]:
        state = np.array([[0.0], [0.0], [0.0], [6.5 / 3.6]], dtype=float)
        steps = int(self.config.total_time_s / self.config.dt_s)

        for step in range(steps):
            t_s = step * self.config.dt_s
            control = self._control_profile(t_s)
            input_u = np.array([[control.accel_mps2], [control.yaw_rate_rps]], dtype=float)
            state = State.motion_model(state, input_u, self.config.dt_s)

            gt = Pose2D(
                x_m=float(state[0, 0]),
                y_m=float(state[1, 0]),
                yaw_rad=float(state[2, 0]),
                speed_mps=float(state[3, 0]),
            )

            gnss_xy = None
            if self.rng.random() > self.config.gnss_dropout_rate:
                gnss_xy = np.array(
                    [
                        [gt.x_m + self.rng.normal(0.0, self.config.gnss_noise_std_m)],
                        [gt.y_m + self.rng.normal(0.0, self.config.gnss_noise_std_m)],
                    ],
                    dtype=float,
                )

            yaw_obs = gt.yaw_rad + self.rng.normal(0.0, np.deg2rad(self.config.yaw_obs_noise_deg))
            observations = self.camera_localizer.simulate_observations(
                true_x_m=gt.x_m,
                true_y_m=gt.y_m,
                true_yaw_rad=gt.yaw_rad,
                landmarks_xy=self.landmarks_xy,
            )
            camera_xy, camera_conf = self.camera_localizer.estimate_xy(observations, yaw_est_rad=yaw_obs)

            yield SensorFrame(
                timestamp_s=t_s,
                dt_s=self.config.dt_s,
                control=control,
                ground_truth=gt,
                gnss_xy=gnss_xy,
                camera_xy=camera_xy,
                camera_confidence=camera_conf,
            )
