"""Tests for RetouchSpot dataclass and WorkspaceConfig retouch serialization/migration."""

import logging
import unittest

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.models import RetouchConfig, RetouchSpot


class TestRetouchSpot(unittest.TestCase):
    """Basic construction and field access for RetouchSpot."""

    def test_construction_and_fields(self):
        spot = RetouchSpot(dest_x=0.3, dest_y=0.4, source_x=0.35, source_y=0.4, radius=12.0)
        self.assertAlmostEqual(spot.dest_x, 0.3)
        self.assertAlmostEqual(spot.dest_y, 0.4)
        self.assertAlmostEqual(spot.source_x, 0.35)
        self.assertAlmostEqual(spot.source_y, 0.4)
        self.assertAlmostEqual(spot.radius, 12.0)

    def test_frozen(self):
        spot = RetouchSpot(dest_x=0.1, dest_y=0.2, source_x=0.15, source_y=0.2, radius=5.0)
        with self.assertRaises(Exception):
            spot.dest_x = 0.9  # type: ignore[misc]

    def test_equality(self):
        a = RetouchSpot(dest_x=0.1, dest_y=0.2, source_x=0.15, source_y=0.2, radius=5.0)
        b = RetouchSpot(dest_x=0.1, dest_y=0.2, source_x=0.15, source_y=0.2, radius=5.0)
        self.assertEqual(a, b)


class TestRetouchRoundTrip(unittest.TestCase):
    """to_dict() → from_flat_dict() preserves RetouchSpot data exactly."""

    def _make_config(self, spots: list[RetouchSpot]) -> WorkspaceConfig:
        from dataclasses import replace

        retouch = RetouchConfig(dust_remove=True, dust_threshold=0.5, dust_size=3, manual_spots=spots, manual_dust_size=8)
        return replace(WorkspaceConfig(), retouch=retouch)

    def test_roundtrip_no_spots(self):
        cfg = self._make_config([])
        flat = cfg.to_dict()
        restored = WorkspaceConfig.from_flat_dict(flat)
        self.assertEqual(restored.retouch.manual_spots, [])
        self.assertTrue(restored.retouch.dust_remove)

    def test_roundtrip_single_spot(self):
        spot = RetouchSpot(dest_x=0.25, dest_y=0.75, source_x=0.30, source_y=0.75, radius=10.0)
        cfg = self._make_config([spot])
        flat = cfg.to_dict()
        restored = WorkspaceConfig.from_flat_dict(flat)
        self.assertEqual(len(restored.retouch.manual_spots), 1)
        r = restored.retouch.manual_spots[0]
        self.assertAlmostEqual(r.dest_x, 0.25)
        self.assertAlmostEqual(r.dest_y, 0.75)
        self.assertAlmostEqual(r.source_x, 0.30)
        self.assertAlmostEqual(r.source_y, 0.75)
        self.assertAlmostEqual(r.radius, 10.0)

    def test_roundtrip_multiple_spots(self):
        spots = [
            RetouchSpot(dest_x=0.1, dest_y=0.2, source_x=0.15, source_y=0.2, radius=5.0),
            RetouchSpot(dest_x=0.5, dest_y=0.5, source_x=0.55, source_y=0.5, radius=8.0),
            RetouchSpot(dest_x=0.9, dest_y=0.1, source_x=0.95, source_y=0.1, radius=3.0),
        ]
        cfg = self._make_config(spots)
        flat = cfg.to_dict()
        restored = WorkspaceConfig.from_flat_dict(flat)
        self.assertEqual(len(restored.retouch.manual_spots), 3)
        for original, restored_spot in zip(spots, restored.retouch.manual_spots):
            self.assertAlmostEqual(original.dest_x, restored_spot.dest_x)
            self.assertAlmostEqual(original.dest_y, restored_spot.dest_y)
            self.assertAlmostEqual(original.source_x, restored_spot.source_x)
            self.assertAlmostEqual(original.source_y, restored_spot.source_y)
            self.assertAlmostEqual(original.radius, restored_spot.radius)

    def test_roundtrip_scalar_retouch_fields(self):
        """Scalar RetouchConfig fields survive the round trip."""
        spot = RetouchSpot(dest_x=0.1, dest_y=0.1, source_x=0.15, source_y=0.1, radius=7.0)
        cfg = self._make_config([spot])
        flat = cfg.to_dict()
        restored = WorkspaceConfig.from_flat_dict(flat)
        self.assertAlmostEqual(restored.retouch.dust_threshold, 0.5)
        self.assertEqual(restored.retouch.dust_size, 3)
        self.assertEqual(restored.retouch.manual_dust_size, 8)


class TestOldFormatMigration(unittest.TestCase):
    """3-element list format is migrated to RetouchSpot with correct source offset."""

    def test_old_list_migrates_to_spot(self):
        data = {"manual_spots": [[0.3, 0.6, 12.0]]}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(len(config.retouch.manual_spots), 1)
        spot = config.retouch.manual_spots[0]
        self.assertAlmostEqual(spot.dest_x, 0.3)
        self.assertAlmostEqual(spot.dest_y, 0.6)
        self.assertAlmostEqual(spot.source_x, 0.35)  # dest_x + 0.05
        self.assertAlmostEqual(spot.source_y, 0.6)
        self.assertAlmostEqual(spot.radius, 12.0)

    def test_old_list_source_x_clamped_at_1(self):
        """When dest_x + 0.05 > 1.0 the source_x is clamped to 1.0."""
        data = {"manual_spots": [[0.98, 0.5, 5.0]]}
        config = WorkspaceConfig.from_flat_dict(data)
        spot = config.retouch.manual_spots[0]
        self.assertAlmostEqual(spot.source_x, 1.0)

    def test_old_list_multiple_entries(self):
        data = {"manual_spots": [[0.1, 0.2, 4.0], [0.5, 0.5, 8.0]]}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(len(config.retouch.manual_spots), 2)
        self.assertAlmostEqual(config.retouch.manual_spots[0].source_x, 0.15)
        self.assertAlmostEqual(config.retouch.manual_spots[1].source_x, 0.55)

    def test_manual_dust_spots_key_renamed(self):
        """Old 'manual_dust_spots' key is migrated to 'manual_spots' before deserialization."""
        data = {"manual_dust_spots": [[0.2, 0.3, 6.0]]}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(len(config.retouch.manual_spots), 1)
        spot = config.retouch.manual_spots[0]
        self.assertAlmostEqual(spot.dest_x, 0.2)
        self.assertAlmostEqual(spot.dest_y, 0.3)
        self.assertAlmostEqual(spot.radius, 6.0)

    def test_manual_dust_spots_new_dict_format_after_rename(self):
        """Old key with new-format dict values also migrates cleanly."""
        data = {"manual_dust_spots": [{"dx": 0.4, "dy": 0.6, "sx": 0.45, "sy": 0.6, "r": 9.0}]}
        config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(len(config.retouch.manual_spots), 1)
        spot = config.retouch.manual_spots[0]
        self.assertAlmostEqual(spot.source_x, 0.45)


class TestUnrecognizedSpotWarning(unittest.TestCase):
    """Unrecognized spot entries emit a warning and are dropped."""

    def test_unknown_spot_format_warns(self):
        data = {"manual_spots": [{"unknown_key": 42}]}
        with self.assertLogs("negpy.domain.models", level=logging.WARNING) as cm:
            config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(config.retouch.manual_spots, [])
        self.assertTrue(any("unrecognized spot entry" in msg.lower() for msg in cm.output))

    def test_unknown_spot_skipped_valid_kept(self):
        """A bad entry is dropped; valid entries that follow are still loaded."""
        data = {
            "manual_spots": [
                {"unknown_key": "bad"},
                {"dx": 0.1, "dy": 0.2, "sx": 0.15, "sy": 0.2, "r": 5.0},
            ]
        }
        with self.assertLogs("negpy.domain.models", level=logging.WARNING):
            config = WorkspaceConfig.from_flat_dict(data)
        self.assertEqual(len(config.retouch.manual_spots), 1)
        self.assertAlmostEqual(config.retouch.manual_spots[0].dest_x, 0.1)


if __name__ == "__main__":
    unittest.main()
