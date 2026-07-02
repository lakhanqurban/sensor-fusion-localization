"""
evaluator.py

Metric utilities for sensor-fusion localization benchmarking.
"""

from __future__ import annotations

import numpy as np

try:
    from .models import TrajectoryMetrics
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import TrajectoryMetrics


def compute_metrics(name: str, gt_xy: np.ndarray, est_xy: np.ndarray) -> TrajectoryMetrics:
    """Compute x/y and Euclidean RMSE for one trajectory estimate."""

    err = est_xy - gt_xy
    rmse_x = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    rmse_pos = float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1))))

    return TrajectoryMetrics(
        algorithm=name,
        rmse_x_m=rmse_x,
        rmse_y_m=rmse_y,
        rmse_pos_m=rmse_pos,
    )
