from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class RetouchSpot:
    dest_x: float  # normalized 0-1 in raw image space
    dest_y: float
    source_x: float  # normalized 0-1 in raw image space
    source_y: float
    radius: float  # pixels at original resolution


@dataclass(frozen=True)
class RetouchConfig:
    dust_remove: bool = False
    dust_threshold: float = 0.66
    dust_size: int = 4
    manual_spots: List[RetouchSpot] = field(default_factory=list)
    manual_dust_size: int = 6
