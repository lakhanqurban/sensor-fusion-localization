"""
replay_dataset_adapter.py

CSV replay adapter to support migration from synthetic to real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from ..models import ControlInput, Pose2D, SensorFrame
    from .base_adapter import BaseSensorDataAdapter
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import ControlInput, Pose2D, SensorFrame
    from adapters.base_adapter import BaseSensorDataAdapter


class ReplayDatasetAdapter(BaseSensorDataAdapter):
    """
    Replays pre-recorded data from CSV for benchmark compatibility.

    Expected CSV columns:
    timestamp_s,dt_s,accel_mps2,yaw_rate_rps,gt_x_m,gt_y_m,gt_yaw_rad,gt_speed_mps,
    gnss_x_m,gnss_y_m,camera_x_m,camera_y_m,camera_confidence

    Missing sensor values should be NaN.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV was not found: {self.csv_path}")

    def iter_frames(self) -> Iterable[SensorFrame]:
        data = np.genfromtxt(self.csv_path, delimiter=",", names=True)
        if data.size == 0:
            return

        if data.ndim == 0:
            rows = [data]
        else:
            rows = data

        for row in rows:
            gnss_xy = None
            if not np.isnan(row["gnss_x_m"]) and not np.isnan(row["gnss_y_m"]):
                gnss_xy = np.array([[float(row["gnss_x_m"])], [float(row["gnss_y_m"])]], dtype=float)

            camera_xy = None
            if not np.isnan(row["camera_x_m"]) and not np.isnan(row["camera_y_m"]):
                camera_xy = np.array([[float(row["camera_x_m"])], [float(row["camera_y_m"])]], dtype=float)

            confidence = 0.0 if np.isnan(row["camera_confidence"]) else float(row["camera_confidence"])

            yield SensorFrame(
                timestamp_s=float(row["timestamp_s"]),
                dt_s=float(row["dt_s"]),
                control=ControlInput(
                    accel_mps2=float(row["accel_mps2"]),
                    yaw_rate_rps=float(row["yaw_rate_rps"]),
                ),
                ground_truth=Pose2D(
                    x_m=float(row["gt_x_m"]),
                    y_m=float(row["gt_y_m"]),
                    yaw_rad=float(row["gt_yaw_rad"]),
                    speed_mps=float(row["gt_speed_mps"]),
                ),
                gnss_xy=gnss_xy,
                camera_xy=camera_xy,
                camera_confidence=confidence,
            )
