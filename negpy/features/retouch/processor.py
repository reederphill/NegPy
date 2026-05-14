from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.retouch.models import RetouchConfig, RetouchSpot
from negpy.features.retouch.logic import apply_dust_removal
from negpy.features.geometry.logic import map_coords_to_geometry


class RetouchProcessor:
    """
    Applies healing and automatic dust removal.
    """

    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        scale_factor = context.scale_factor

        orig_h, orig_w = context.original_size

        rot_params = context.metrics.get(
            "geometry_params",
            {
                "rotation": 0,
                "fine_rotation": 0.0,
                "flip_horizontal": False,
                "flip_vertical": False,
            },
        )
        rotation = rot_params.get("rotation", 0)
        fine_rotation = rot_params.get("fine_rotation", 0.0)
        flip_h = rot_params.get("flip_horizontal", False)
        flip_v = rot_params.get("flip_vertical", False)
        active_roi = context.metrics.get("active_roi", None)

        mapped_spots: list[RetouchSpot] = []
        for spot in self.config.manual_spots:
            mdx, mdy = map_coords_to_geometry(
                spot.dest_x,
                spot.dest_y,
                (orig_h, orig_w),
                rotation,
                fine_rotation,
                flip_h,
                flip_v,
                roi=active_roi,
            )
            msx, msy = map_coords_to_geometry(
                spot.source_x,
                spot.source_y,
                (orig_h, orig_w),
                rotation,
                fine_rotation,
                flip_h,
                flip_v,
                roi=active_roi,
            )
            mapped_spots.append(RetouchSpot(dest_x=mdx, dest_y=mdy, source_x=msx, source_y=msy, radius=spot.radius))

        img = apply_dust_removal(
            img,
            self.config.dust_remove,
            self.config.dust_threshold,
            self.config.dust_size,
            mapped_spots,
            scale_factor,
        )

        return img
