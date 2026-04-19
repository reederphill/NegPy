from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from negpy.desktop.view.styles.theme import THEME


class Toast(QWidget):
    """Transient notification overlay anchored bottom-center of parent window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowFlags(Qt.WindowType.SubWindow)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)

        self.label = QLabel()
        self.label.setStyleSheet(f"color: {THEME.text_primary}; font-size: 15px; background: transparent;")
        layout.addWidget(self.label)

        self.setStyleSheet(f"background-color: rgba(26, 26, 26, 235);border: 1px solid {THEME.border_color};border-radius: 4px;")

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._start_fade)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(200)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_anim.finished.connect(self._on_fade_finished)

        self.setVisible(False)

    def show_text(self, text: str, duration_ms: int = 1500) -> None:
        self._hide_timer.stop()
        self._fade_anim.stop()
        self._opacity_effect.setOpacity(1.0)
        self.label.setText(text)
        self.adjustSize()
        self._reposition()
        self.raise_()
        self.setVisible(True)
        self._hide_timer.start(duration_ms)

    def _reposition(self) -> None:
        p = self.parentWidget()
        if p is None:
            return
        x = (p.width() - self.width()) // 2
        y = (p.height() - self.height()) // 2
        self.move(max(0, x), max(0, y))

    def _start_fade(self) -> None:
        self._fade_anim.stop()
        self._fade_anim.setStartValue(float(self._opacity_effect.opacity()))
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    def _on_fade_finished(self) -> None:
        if self._opacity_effect.opacity() < 0.01:
            self.setVisible(False)
