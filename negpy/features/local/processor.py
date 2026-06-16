from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.local.models import LocalAdjustmentsConfig
from negpy.features.local.logic import apply_local_adjustments


class LocalProcessor:
    """Applies dodge/burn local adjustments to linear-light RGB output."""

    def __init__(self, config: LocalAdjustmentsConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        return apply_local_adjustments(image, self.config)
