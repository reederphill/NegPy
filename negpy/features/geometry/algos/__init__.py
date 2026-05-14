from typing import Optional, Protocol, Tuple
import numpy as np
from negpy.domain.types import ROI


class FrameDetector(Protocol):
    def detect(
        self,
        luma: np.ndarray,
        target_ratio_str: str,
        assist_luma: Optional[float],
    ) -> Tuple[ROI, float]:
        """Return (roi_in_luma_pixels, confidence 0..1). ROI is (y1, y2, x1, x2)."""
        ...
