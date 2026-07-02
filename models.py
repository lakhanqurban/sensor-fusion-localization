"""
models.py

Shared data models for the Sensor-Fusion Localization project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlInput:
    """Control command at one timestamp."""

    accel_mps2: float
    yaw_rate_rps: float


@dataclass
class Pose2D:
    """Planar vehicle pose and speed."""

    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float


@dataclass
class SensorFrame:
    """Unified sensor sample consumed by the benchmark pipeline."""

    timestamp_s: float
    dt_s: float
    control: ControlInput
    ground_truth: Pose2D
    gnss_xy: Optional[np.ndarray]
    camera_xy: Optional[np.ndarray]
    camera_confidence: float


@dataclass
class TrajectoryMetrics:
    """Aggregate metrics for one localization algorithm."""

    algorithm: str
    rmse_x_m: float
    rmse_y_m: float
    rmse_pos_m: float
