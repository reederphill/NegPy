from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QCheckBox, QLabel
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.theme import THEME


class LocalSidebar(BaseSidebar):
    """
    Single-brush local EV adjustment (dodge/burn by strength sign).
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.local

        # --- Tool button ---
        self.paint_btn = QPushButton(" Paint")
        self.paint_btn.setCheckable(True)
        self.paint_btn.setIcon(qta.icon("fa5s.paint-brush", color=THEME.text_primary))
        self.paint_btn.setToolTip("Click or drag on the canvas to apply local EV adjustments")
        self.layout.addWidget(self.paint_btn)

        # --- Brush controls ---
        img_w = self.state.original_res[0] if self.state.original_res[0] > 0 else 2000
        max_px = max(10, img_w // 10)
        default_px = max(1, min(int(conf.brush_size), max_px))
        self.brush_size_slider = CompactSlider(
            "Brush Size", 1, max_px, default_px, step=1, precision=1, unit=" px"
        )
        self.brush_size_slider.setToolTip("Brush radius in image pixels")
        self.layout.addWidget(self.brush_size_slider)

        self.strength_slider = CompactSlider(
            "Strength", -1.0, 1.0, conf.strength, step=0.1, precision=100,
            has_neutral=True, unit=" EV"
        )
        self.strength_slider.setToolTip(
            "EV added per pass — positive brightens (dodge), negative darkens (burn)"
        )
        self.layout.addWidget(self.strength_slider)

        # --- Status + clear ---
        status_row = QHBoxLayout()
        self.spot_count_label = QLabel("0 spots")
        self.spot_count_label.setStyleSheet(
            f"font-size: {THEME.font_size_base}px; color: {THEME.text_secondary};"
        )
        self.clear_btn = QPushButton(" Clear")
        self.clear_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))
        self.clear_btn.setToolTip("Remove all local adjustment spots")
        self.clear_btn.setEnabled(False)
        status_row.addWidget(self.spot_count_label)
        status_row.addStretch()
        status_row.addWidget(self.clear_btn)
        self.layout.addLayout(status_row)

        # --- Show overlay toggle ---
        self.show_mask_check = QCheckBox("Show Adjustment Map")
        self.show_mask_check.setToolTip(
            "Overlay dodge (yellow) and burn (blue) regions on the canvas"
        )
        self.layout.addWidget(self.show_mask_check)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.paint_btn.toggled.connect(self._on_paint_toggled)
        self.brush_size_slider.valueCommitted.connect(
            lambda v: self.update_config_section("local", render=False, persist=True, brush_size=max(1.0, float(v)))
        )
        self.strength_slider.valueChanged.connect(
            lambda v: self.update_config_section("local", render=False, strength=float(v))
        )
        self.strength_slider.valueCommitted.connect(
            lambda v: self.update_config_section("local", render=False, persist=True, strength=float(v))
        )
        self.clear_btn.clicked.connect(self.controller.clear_local)
        self.show_mask_check.toggled.connect(self._on_show_mask_toggled)

    def _on_paint_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.LOCAL_BRUSH if checked else ToolMode.NONE)

    def _on_show_mask_toggled(self, checked: bool) -> None:
        self.state.show_local_mask = checked
        if self.controller.canvas:
            self.controller.canvas.overlay.update()

    def sync_ui(self) -> None:
        conf = self.state.config.local
        self.block_signals(True)
        try:
            img_w = self.state.original_res[0] if self.state.original_res[0] > 0 else 2000
            max_px = max(10, img_w // 10)
            if self.brush_size_slider.slider.maximum() != max_px:
                self.brush_size_slider._max = float(max_px)
                self.brush_size_slider.slider.setRange(1, max_px)
                self.brush_size_slider.spin.setRange(1.0, float(max_px))
            self.brush_size_slider.setValue(float(max(1, min(int(conf.brush_size), max_px))))
            self.strength_slider.setValue(conf.strength)
            self.paint_btn.setChecked(self.state.active_tool == ToolMode.LOCAL_BRUSH)
            n = len(conf.spots)
            self.spot_count_label.setText(f"{n} spot{'s' if n != 1 else ''}")
            self.clear_btn.setEnabled(n > 0)
            self.show_mask_check.setChecked(getattr(self.state, "show_local_mask", False))
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in [
            self.paint_btn,
            self.brush_size_slider,
            self.strength_slider,
            self.show_mask_check,
        ]:
            w.blockSignals(blocked)
