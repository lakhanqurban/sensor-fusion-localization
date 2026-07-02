"""
fusion.py

Lightweight camera + GNSS fusion for position observations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PositionObservationFusion:
    """Fuses GNSS and camera position measurements into one 2D observation."""

    def __init__(self, gnss_std_m: float = 1.0, camera_std_m: float = 0.7) -> None:
        self.gnss_var = max(gnss_std_m, 1e-6) ** 2
        self.camera_var = max(camera_std_m, 1e-6) ** 2

    def fuse(
        self,
        gnss_xy: Optional[np.ndarray],
        camera_xy: Optional[np.ndarray],
        camera_confidence: float,
    ) -> Optional[np.ndarray]:
        if gnss_xy is None and camera_xy is None:
            return None
        if gnss_xy is None:
            return camera_xy
        if camera_xy is None:
            return gnss_xy

        camera_conf = float(np.clip(camera_confidence, 0.0, 1.0))

        # Confidence scales camera contribution. Low confidence makes camera close to ignored.
        inv_var_gnss = 1.0 / self.gnss_var
        inv_var_camera = camera_conf / self.camera_var
        denom = inv_var_gnss + inv_var_camera
        if denom <= 1e-9:
            return gnss_xy

        fused = (inv_var_gnss * gnss_xy + inv_var_camera * camera_xy) / denom
        return fused
