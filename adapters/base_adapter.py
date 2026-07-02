"""
base_adapter.py

Adapter contract for sensor-frame data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

try:
    from ..models import SensorFrame
except ImportError:  # pragma: no cover - direct script execution fallback
    from models import SensorFrame


class BaseSensorDataAdapter(ABC):
    """Source-agnostic frame iterator for localization benchmark input."""

    @abstractmethod
    def iter_frames(self) -> Iterable[SensorFrame]:
        """Yield sensor frames in chronological order."""
        raise NotImplementedError
