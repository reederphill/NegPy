import os

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum, StrEnum
from negpy.features.process.models import ProcessConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lab.models import LabConfig
from negpy.features.retouch.models import RetouchConfig, RetouchSpot
from negpy.features.toning.models import ToningConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.kernel.system.logging import get_logger
import negpy.kernel.system.paths as paths

logger = get_logger("domain.models")

# Map of old field names → new field names for backward-compatible deserialization.
# Add entries here when fields are renamed so old workspace files keep their data.
MIGRATIONS: Dict[str, str] = {
    "export_border_size": "border_size",
    "export_border_color": "border_color",
    "manual_dust_spots": "manual_spots",
}


class AspectRatio(StrEnum):
    FREE = "Free"
    ORIGINAL = "Original"
    R_3_2 = "3:2"
    R_4_3 = "4:3"
    R_5_4 = "5:4"
    R_6_7 = "6:7"
    R_1_1 = "1:1"
    R_65_24 = "65:24"
    # Verticals
    R_2_3 = "2:3"
    R_3_4 = "3:4"
    R_4_5 = "4:5"
    R_7_6 = "7:6"
    R_24_65 = "24:65"


class ExportFormat(StrEnum):
    JPEG = "JPEG"
    TIFF = "TIFF"


class ExportResolutionMode(StrEnum):
    ORIGINAL = "original"
    PRINT = "print"
    TARGET_PX = "target_px"


class ICCMode(Enum):
    OUTPUT = "Output"
    INPUT = "Input"


class ColorSpace(Enum):
    SAME_AS_SOURCE = "Same as Source"
    SRGB = "sRGB"
    ADOBE_RGB = "Adobe RGB"
    PROPHOTO = "ProPhoto RGB"
    WIDE = "Wide Gamut RGB"
    ACES = "ACES"
    P3_D65 = "P3 D65"
    REC2020 = "Rec 2020"
    XYZ = "XYZ"
    GREYSCALE = "Greyscale"


@dataclass(frozen=True)
class ExportConfig:
    """
    Export parameters (path, format, sizing).
    """

    userDir: str = field(default_factory=paths.get_default_user_dir)

    export_path: str = field(default_factory=lambda: os.path.join(paths.get_default_user_dir(), "export"))
    export_fmt: str = ExportFormat.JPEG
    export_color_space: str = ColorSpace.SAME_AS_SOURCE.value
    paper_aspect_ratio: str = AspectRatio.ORIGINAL
    export_print_size: float = 30.0
    export_dpi: int = 300
    export_resolution_mode: str = ExportResolutionMode.PRINT.value
    export_target_long_edge_px: int = 2000
    filename_pattern: str = "{{ original_name }}"
    overwrite: bool = True
    same_as_source: bool = False
    apply_icc: bool = False
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Complete state for a single image edit.
    """

    process: ProcessConfig = field(default_factory=ProcessConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lab: LabConfig = field(default_factory=LabConfig)
    retouch: RetouchConfig = field(default_factory=RetouchConfig)
    toning: ToningConfig = field(default_factory=ToningConfig)
    finish: FinishConfig = field(default_factory=FinishConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        """
        Flattens for serialization.
        """
        res = {}
        res.update(asdict(self.process))
        res.update(asdict(self.exposure))
        res.update(asdict(self.geometry))
        res.update(asdict(self.lab))
        # Serialize RetouchConfig manually so the spots list uses the compact wire
        # format ("dx"/"dy"/"sx"/"sy"/"r") rather than the full field names that
        # asdict() would emit — and to avoid asdict() expanding spots and then
        # immediately overwriting the result.
        r = self.retouch
        retouch_dict = {
            "dust_remove": r.dust_remove,
            "dust_threshold": r.dust_threshold,
            "dust_size": r.dust_size,
            "manual_dust_size": r.manual_dust_size,
            "manual_spots": [{"dx": s.dest_x, "dy": s.dest_y, "sx": s.source_x, "sy": s.source_y, "r": s.radius} for s in r.manual_spots],
        }
        res.update(retouch_dict)
        res.update(asdict(self.toning))
        res.update(asdict(self.finish))
        res.update(asdict(self.metadata))
        res.update(asdict(self.export))
        return res

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        """
        from DB/JSON.
        """

        # Apply field renames for backward compatibility.
        for old_key, new_key in MIGRATIONS.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)

        if "use_original_res" in data and "export_resolution_mode" not in data:
            data["export_resolution_mode"] = (
                ExportResolutionMode.ORIGINAL.value if data.pop("use_original_res") else ExportResolutionMode.PRINT.value
            )
        else:
            data.pop("use_original_res", None)

        config_classes = [
            ProcessConfig,
            ExposureConfig,
            GeometryConfig,
            LabConfig,
            RetouchConfig,
            ToningConfig,
            FinishConfig,
            MetadataConfig,
            ExportConfig,
        ]
        valid_keys = set()
        for cc in config_classes:
            valid_keys.update(cc.__dataclass_fields__.keys())

        unknown = set(data) - valid_keys
        if unknown:
            logger.warning("Dropping unknown config keys: %s", sorted(unknown))

        def filter_keys(config_cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
            valid = config_cls.__dataclass_fields__.keys()
            return {k: v for k, v in d.items() if k in valid}

        def deserialize_retouch(d: Dict[str, Any]) -> RetouchConfig:
            kwargs = filter_keys(RetouchConfig, d)
            raw_spots = kwargs.pop("manual_spots", None) or []
            spots: list[RetouchSpot] = []
            for entry in raw_spots:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    # Old format: [nx, ny, size] — migrate to RetouchSpot
                    nx, ny, size = float(entry[0]), float(entry[1]), float(entry[2])
                    spots.append(RetouchSpot(dest_x=nx, dest_y=ny, source_x=min(1.0, nx + 0.05), source_y=ny, radius=size))
                elif isinstance(entry, dict) and "dx" in entry:
                    spots.append(
                        RetouchSpot(
                            dest_x=float(entry["dx"]),
                            dest_y=float(entry["dy"]),
                            source_x=float(entry["sx"]),
                            source_y=float(entry["sy"]),
                            radius=float(entry["r"]),
                        )
                    )
                else:
                    logger.warning("Dropping unrecognized spot entry (unexpected format): %r", entry)
            return RetouchConfig(**kwargs, manual_spots=spots)

        return cls(
            process=ProcessConfig(**filter_keys(ProcessConfig, data)),
            exposure=ExposureConfig(**filter_keys(ExposureConfig, data)),
            geometry=GeometryConfig(**filter_keys(GeometryConfig, data)),
            lab=LabConfig(**filter_keys(LabConfig, data)),
            retouch=deserialize_retouch(data),
            toning=ToningConfig(**filter_keys(ToningConfig, data)),
            finish=FinishConfig(**filter_keys(FinishConfig, data)),
            metadata=MetadataConfig(**filter_keys(MetadataConfig, data)),
            export=ExportConfig(**filter_keys(ExportConfig, data)),
        )
