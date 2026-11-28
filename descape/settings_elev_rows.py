"""Settings > Appearance's two elevation height rows (GH #288).

Stepped (with Flat + Isometric View) and Sloped each keep their own stop:
Sloped picks through an ID plane, so it can match the game's height, which
Stepped's occlusion keeps it from. Each row's value space is the 1-based stop
index, not the pct: setSingleStep() alone governs only arrow keys and the
wheel, so a drag would still produce off-stop pcts through
QStyle::sliderValueFromPosition.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider

from descape import iso_geometry, settings

_STEPPED_TIP = (
    "How tall one elevation level stands on screen in Stepped and in Flat with Isometric View. "
    "Flat with Isometric View always renders at elevation 0, so there this only shapes the diamond tiles. "
    f"The default is Tall ({iso_geometry.ELEV_STEP_DEFAULT_PCT}%). Heights above it are real headroom: "
    "the game's elevation changes are smooth ramps over several tiles, so a taller setting hides fewer "
    "tiles behind their neighbours than one sharp step that tall would."
)
_SLOPED_TIP = (
    "How tall one elevation level stands on screen in Sloped. "
    f"100% matches the game's own height, and is the default ({iso_geometry.SLOPED_ELEV_STEP_DEFAULT_PCT}%)."
)


def _value_text(pct: int, default_pct: int, tag: str) -> str:
    return f"{pct}% ({tag}, default)" if pct == default_pct else f"{pct}%"


class _ElevStepRowsMixin:
    """The rows' widgets, slots and persist methods; SettingsDialog builds them on its Appearance page."""

    def _add_elev_step_rows(self, layout) -> None:
        # Per build: Reset tab rebuilds the page, and the old page's labels are deleted with it.
        self._elev_row_labels = ([], [])
        self.elev_step_slider, self.elev_step_value_label, self.elev_step_default_button = self._elev_row(
            layout, "Stepped elevation height:", _STEPPED_TIP, settings.get_elev_step_pct(),
            iso_geometry.ELEV_STEP_DEFAULT_PCT, "Tall", self._on_elev_step_slider_changed,
            self._persist_elev_step_pct, self._reset_elev_step_pct,
        )
        self.sloped_elev_step_slider, self.sloped_elev_step_value_label, self.sloped_elev_step_default_button = (
            self._elev_row(
                layout, "Sloped elevation height:", _SLOPED_TIP, settings.get_sloped_elev_step_pct(),
                iso_geometry.SLOPED_ELEV_STEP_DEFAULT_PCT, "DE", self._on_sloped_elev_step_slider_changed,
                self._persist_sloped_elev_step_pct, self._reset_sloped_elev_step_pct,
            )
        )
        # One width per column, so the two sliders line up and keep their width as the value text changes.
        for labels in self._elev_row_labels:
            widths = []
            for label, text in labels:
                label.setText(text)
                widths.append(label.sizeHint().width())
            for label, _text in labels:
                label.setMinimumWidth(max(widths))
        self._update_elev_step_label(settings.get_elev_step_pct())
        self._update_sloped_elev_step_label(settings.get_sloped_elev_step_pct())

    def _elev_row(self, layout, text, tip, pct, default_pct, tag, on_changed, on_released, on_default):
        row = QHBoxLayout()
        name_label = QLabel(text)
        row.addWidget(name_label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, len(settings.ELEV_STEP_PCT_STOPS))
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setToolTip(tip)
        slider.setValue(settings.elev_step_index(pct))
        value_label = QLabel()
        default_button = QPushButton("Default")
        default_button.setToolTip(f"Back to {default_pct}%")
        default_index = settings.elev_step_index(default_pct)
        default_button.setEnabled(slider.value() != default_index)
        slider.valueChanged.connect(lambda value: default_button.setEnabled(value != default_index))
        slider.valueChanged.connect(on_changed)
        slider.sliderReleased.connect(on_released)
        default_button.clicked.connect(on_default)
        row.addWidget(slider, stretch=1)
        row.addWidget(value_label)
        row.addWidget(default_button)
        layout.addLayout(row)
        self._elev_row_labels[0].append((name_label, text))
        self._elev_row_labels[1].append((value_label, _value_text(default_pct, default_pct, tag)))
        return slider, value_label, default_button

    def _update_elev_step_label(self, pct: int) -> None:
        self.elev_step_value_label.setText(_value_text(pct, iso_geometry.ELEV_STEP_DEFAULT_PCT, "Tall"))

    def _update_sloped_elev_step_label(self, pct: int) -> None:
        self.sloped_elev_step_value_label.setText(_value_text(pct, iso_geometry.SLOPED_ELEV_STEP_DEFAULT_PCT, "DE"))

    def _on_elev_step_slider_changed(self, value: int) -> None:
        self._update_elev_step_label(settings.elev_step_pct_for_index(value))
        # Each apply is a full re-render, too slow to run per drag step: a drag
        # applies on sliderReleased, a key press (no press/release pair) on a debounce.
        if not self.elev_step_slider.isSliderDown():
            self._elev_step_apply_timer.start(200)

    def _on_sloped_elev_step_slider_changed(self, value: int) -> None:
        self._update_sloped_elev_step_label(settings.elev_step_pct_for_index(value))
        if not self.sloped_elev_step_slider.isSliderDown():
            self._sloped_elev_step_apply_timer.start(200)

    def _persist_elev_step_pct(self) -> None:
        self._elev_step_apply_timer.stop()
        value = settings.elev_step_pct_for_index(self.elev_step_slider.value())
        settings.set_elev_step_pct(value)
        self._log_and_refresh(f"Stepped elevation height: {value}%", "stepped")

    def _persist_sloped_elev_step_pct(self) -> None:
        self._sloped_elev_step_apply_timer.stop()
        value = settings.elev_step_pct_for_index(self.sloped_elev_step_slider.value())
        settings.set_sloped_elev_step_pct(value)
        self._log_and_refresh(f"Sloped elevation height: {value}%", "sloped")

    def _log_and_refresh(self, message: str, style: str) -> None:
        # _render_style is "stepped" for Flat + Isometric View too, which reads Stepped's stop.
        self._window._log_status(message)
        if self._window._render_style == style:
            self._window.refresh_map()

    def _reset_elev_step_pct(self) -> None:
        self.elev_step_slider.setValue(settings.elev_step_index(iso_geometry.ELEV_STEP_DEFAULT_PCT))
        self._persist_elev_step_pct()

    def _reset_sloped_elev_step_pct(self) -> None:
        self.sloped_elev_step_slider.setValue(settings.elev_step_index(iso_geometry.SLOPED_ELEV_STEP_DEFAULT_PCT))
        self._persist_sloped_elev_step_pct()
