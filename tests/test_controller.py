import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState, ToolMode
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestAppController(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        # Patch GPU-touching classes before AppController.__init__ so no real GPU is created
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        # Stop all background threads before the controller is GC'd
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_load_file_emits_zoom_reset(self):
        """Test that loading a file normally resets the zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng")

        mock_slot.assert_called_once_with(1.0)
        self.assertFalse(self.controller.state.hq_preview)

    def test_load_file_preserve_zoom(self):
        """Test that load_file with preserve_zoom=True skips resetting zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng", preserve_zoom=True)

        mock_slot.assert_not_called()

    def test_toggle_hq_preview_preserves_zoom(self):
        """Test that toggling HQ mode persists via session and preserves zoom."""
        self.controller.state.current_file_path = "dummy.dng"

        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.toggle_hq_preview()

        # Persistence delegated to session
        self.mock_session_manager.set_hq_preview.assert_called_once_with(True)

        # Zoom should NOT be reset
        mock_slot.assert_not_called()

    def test_preview_loaded_updates_state_and_emits_signal(self):
        """Successful preview loads should publish dimensions before rendering starts."""
        mock_slot = MagicMock()
        self.controller.preview_loaded.connect(mock_slot)
        self.controller.request_render = MagicMock()

        raw = object()
        dims = (1234, 5678)

        self.controller._on_preview_loaded("dummy.dng", raw, dims, "")

        self.assertIs(self.controller.state.preview_raw, raw)
        self.assertEqual(self.controller.state.original_res, dims)
        self.assertEqual(self.controller.state.current_file_path, "dummy.dng")
        mock_slot.assert_called_once_with()
        self.controller.request_render.assert_called_once_with()

    def test_apply_auto_crop_enables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(
            self.controller.state.config.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9), auto_crop_enabled=False, autocrop_ratio="3:2"
        )
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.apply_auto_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(saved_config.geometry.auto_crop_enabled)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_reset_crop_disables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9), auto_crop_enabled=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.reset_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.auto_crop_enabled)
        self.assertIsNone(saved_config.geometry.manual_crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_manual_crop_completion_disables_auto_crop(self):
        geometry = replace(self.controller.state.config.geometry, auto_crop_enabled=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.state.last_metrics = {"uv_grid": (0.0, 1.0, 0.0, 1.0)}
        self.controller.request_render = MagicMock()

        with patch("negpy.desktop.controller.CoordinateMapping.map_click_to_raw", side_effect=[(0.2, 0.3), (0.8, 0.9)]):
            self.controller.handle_crop_completed(0.2, 0.3, 0.8, 0.9)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.auto_crop_enabled)
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.2, 0.3, 0.8, 0.9))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_translated_updates_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_translated(0.3, 0.25, 0.7, 0.55)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.manual_crop_rect, (0.3, 0.25, 0.7, 0.55))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_translated_noop_when_no_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=None)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_translated(0.1, 0.1, 0.5, 0.5)

        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_handle_crop_translated_does_not_deactivate_tool(self):
        geometry = replace(self.controller.state.config.geometry, manual_crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MOVE
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_translated(0.3, 0.25, 0.7, 0.55)

        self.assertEqual(self.controller.state.active_tool, ToolMode.CROP_MOVE)


class TestBatchExportFiltering(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp/out")
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _captured_tasks(self):
        self.controller._run_export_tasks.assert_called_once()
        return self.controller._run_export_tasks.call_args.args[0]

    def test_export_all_with_no_filter(self):
        self.visible_indices = [0, 1, 2]
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2", "scan.tif"])

    def test_export_all_respects_filter(self):
        self.visible_indices = [0, 1]  # only IMG_*
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2"])

    def test_export_all_zero_matches_does_not_dispatch(self):
        self.visible_indices = []
        self.controller.request_batch_export()
        self.controller._run_export_tasks.assert_not_called()

    def test_export_all_preserves_display_order(self):
        self.visible_indices = [2, 0]  # reversed visible order from sort+filter
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif", "IMG_0001.cr2"])

    def test_export_all_override_settings_applies_current_export_to_all(self):
        self.visible_indices = [0, 1]
        self.controller.state.config = replace(
            self.controller.state.config,
            export=replace(self.controller.state.config.export, export_path="/orig"),
        )
        self.controller.request_batch_export(override_settings=True)
        tasks = self._captured_tasks()
        for t in tasks:
            self.assertEqual(t.params.export.export_path, "/tmp/out")


class TestExclusion(unittest.TestCase):
    """Tests for image exclusion from batch operations."""

    def setUp(self):
        self.mock_session = MagicMock(spec=DesktopSessionManager)
        self.mock_session.state = AppState()
        self.mock_session.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session)

        self.controller.normalization_requested = MagicMock()
        self.controller.normalization_progress = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.pipeline_thumb_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _make_files(self):
        return [
            {"name": "a.arw", "path": "/a.arw", "hash": "hash_a"},
            {"name": "b.arw", "path": "/b.arw", "hash": "hash_b"},
            {"name": "c.arw", "path": "/c.arw", "hash": "hash_c"},
        ]

    def test_batch_normalization_skips_excluded_files(self):
        state = self.controller.state
        state.uploaded_files = self._make_files()
        state.excluded_file_hashes = {"hash_b"}

        self.controller.request_batch_normalization()

        call_args = self.controller.normalization_requested.emit.call_args
        task = call_args[0][0]
        hashes = [f["hash"] for f in task.files]
        self.assertIn("hash_a", hashes)
        self.assertNotIn("hash_b", hashes)
        self.assertIn("hash_c", hashes)

    def test_batch_normalization_noop_when_all_excluded(self):
        state = self.controller.state
        state.uploaded_files = self._make_files()
        state.excluded_file_hashes = {"hash_a", "hash_b", "hash_c"}

        self.controller.request_batch_normalization()

        self.controller.normalization_requested.emit.assert_not_called()

    def test_batch_export_skips_excluded_files(self):
        state = self.controller.state
        state.uploaded_files = self._make_files()
        state.excluded_file_hashes = {"hash_a"}
        state.current_file_hash = None

        self.mock_session.repo.load_file_settings.return_value = None
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp/out")

        captured_tasks = []
        self.controller._run_export_tasks = MagicMock(side_effect=lambda tasks: captured_tasks.extend(tasks))

        self.controller.request_batch_export()

        exported_hashes = [t.file_info["hash"] for t in captured_tasks]
        self.assertNotIn("hash_a", exported_hashes)
        self.assertIn("hash_b", exported_hashes)
        self.assertIn("hash_c", exported_hashes)

    def test_sync_selected_settings_preserves_excluded_flag(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.desktop.session import DesktopSessionManager

        session = MagicMock(spec=DesktopSessionManager)
        session.state = AppState()
        session.repo = MagicMock()

        state = session.state
        state.uploaded_files = self._make_files()
        state.selected_file_idx = 0
        state.selected_indices = [0, 1]

        source_config = WorkspaceConfig()
        target_config = WorkspaceConfig(excluded=True)

        state.config = source_config
        session.repo.load_file_settings.return_value = target_config

        saved_configs = {}

        def capture_save(file_hash, config):
            saved_configs[file_hash] = config

        session.repo.save_file_settings.side_effect = capture_save

        # Call the real method
        DesktopSessionManager.sync_selected_settings(session)

        self.assertIn("hash_b", saved_configs)
        self.assertTrue(saved_configs["hash_b"].excluded, "sync must not overwrite target's excluded flag")


class TestToggleExcludeSelected(unittest.TestCase):
    """Tests for toggle_exclude_selected session method."""

    def setUp(self):
        from negpy.desktop.session import DesktopSessionManager

        self.session = MagicMock(spec=DesktopSessionManager)
        self.session.state = AppState()
        self.session.repo = MagicMock()
        self.session.state.uploaded_files = [
            {"name": "a.arw", "path": "/a.arw", "hash": "hash_a"},
            {"name": "b.arw", "path": "/b.arw", "hash": "hash_b"},
        ]
        self.session.state.selected_file_idx = 0
        self.session.state.selected_indices = [0]

        self.saved = {}
        self.session.repo.load_file_settings.return_value = None
        self.session.repo.save_file_settings.side_effect = lambda h, c: self.saved.update({h: c})

    def _toggle(self):
        from negpy.desktop.session import DesktopSessionManager

        DesktopSessionManager.toggle_exclude_selected(self.session)

    def test_toggle_adds_to_excluded_set(self):
        self._toggle()
        self.assertIn("hash_a", self.session.state.excluded_file_hashes)

    def test_toggle_twice_removes_from_excluded_set(self):
        self.session.repo.load_file_settings.side_effect = lambda h: self.saved.get(h)
        self._toggle()
        self._toggle()
        self.assertNotIn("hash_a", self.session.state.excluded_file_hashes)

    def test_toggle_updates_state_config_for_active_file(self):
        self._toggle()
        self.assertTrue(self.session.state.config.excluded)

    def test_toggle_does_not_update_state_config_for_non_active_file(self):
        self.session.state.selected_indices = [1]
        self._toggle()
        self.assertFalse(self.session.state.config.excluded)


if __name__ == "__main__":
    unittest.main()
