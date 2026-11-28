"""MapView's stroke tick: the Spray Can's timed dabs while its button is held (map_view.py's docstring has the split's rules).

A tick tool's press dabs once, then a timer dabs at the tile under the
nozzle, the cursor's last viewport position, so a pan, zoom or scroll under a
still cursor sprays the new tile. A move only moves the nozzle."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from descape import perf_trace

# Stroke tools the timer drives. Not in _INTERPOLATED_STROKE_TOOLS: a move does not dab.
_TICK_STROKE_TOOLS = frozenset({"spray"})


class _MapViewStrokeTickMixin:
    """MapView's stroke tick (map_view.py's docstring has the split's rules)."""

    def _on_stroke_tick(self) -> None:
        if not self._stroke_active or self._tool not in _TICK_STROKE_TOOLS or not self.isVisible():
            self.stop_stroke_tick()
            return
        # A busy span disables the view but not a QTimer (the _autosave_tick hazard); a modal owns the input.
        if not self.isEnabled() or QApplication.activeModalWidget() is not None:
            return
        # A release lost to a modal (Cancel on the close prompt mid-hold) must not spray forever.
        if not self._stroke_button_held():
            self._end_stroke()
            return
        if self._stroke_nozzle_pos is None:
            return
        pos = self.mapToScene(self._stroke_nozzle_pos)
        tile = self._pick_tile(pos) if self._pos_on_map(pos) else None
        if tile is None:
            return
        modifiers = QApplication.keyboardModifiers()
        if self._stroke_button == Qt.RightButton:
            modifiers = modifiers | Qt.ShiftModifier
        with perf_trace.span("step"):
            self._on_stroke_tiles([tile], modifiers)
        perf_trace.step()

    def stop_stroke_tick(self) -> None:
        """Idempotent, like resume_stroke_gc(): every stroke-closing path calls it."""
        self._stroke_tick_timer.stop()
        self._stroke_nozzle_pos = None

    def _stroke_button_held(self) -> bool:
        """A seam: tests press through mousePressEvent, which sets no real button state."""
        return bool(QApplication.mouseButtons() & self._stroke_button)
