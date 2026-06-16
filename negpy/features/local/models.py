from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class LocalAdjustmentsConfig:
    # (nx, ny, radius_normalized, strength_ev) — nx/ny in [0,1], radius as
    # fraction of the shorter image dimension, strength_ev in EV stops.
    spots: List[Tuple[float, float, float, float]] = field(default_factory=list)
    brush_size: float = 20.0   # Brush radius in image pixels
    strength: float = 0.3      # EV per pass, -1 to +1 (negative = darken)
