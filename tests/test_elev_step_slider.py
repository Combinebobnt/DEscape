"""Settings > Appearance's two elevation height rows, Stepped's and Sloped's own (GH #288).

Each slider's value space is the stop index, not the pct (Track B-D-e of
the mip-level plan): an off-stop pct enumerates a shallower mip ladder than
the nearest stop would, and setSingleStep() alone doesn't prevent one -- it
governs arrow keys and the wheel, while a drag goes through
QStyle::sliderValueFromPosition and lands anywhere. Making the slider's own
range the 8 stop indices is what makes an off-stop value unrepresentable.

Same offscreen-ViewerWindow technique as tests/test_keybinds.py, including
constructing SettingsDialog directly rather than through
ViewerWindow._show_settings() (that method calls exec_(), which is modal and
would hang an offscreen run).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import iso_geometry, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


_dialog_and_window = conftest.dialog_and_window

# Each row: its widgets' attribute prefix, config key, memo global, getter, default, label suffix,
# guide line, and the render style it redraws.
ROWS = {
    "stepped": (
        "elev_step", "elev_step_pct", "_elev_step_pct", settings.get_elev_step_pct,
        iso_geometry.ELEV_STEP_DEFAULT_PCT, " (Tall, default)", "- Stepped elevation height:", "stepped",
    ),
    "sloped": (
        "sloped_elev_step", "sloped_elev_step_pct", "_sloped_elev_step_pct", settings.get_sloped_elev_step_pct,
        iso_geometry.SLOPED_ELEV_STEP_DEFAULT_PCT, " (DE, default)", "- Sloped elevation height:", "sloped",
    ),
}


def _other_key(row: str) -> str:
    return ROWS["sloped" if row == "stepped" else "stepped"][1]


def _close(dialog, window) -> None:
    dialog.close()
    window.edit_history.mark_saved()
    window.close()


@pytest.mark.parametrize("row", ROWS)
def test_slider_range_is_the_stop_indices_not_the_pct_range(row) -> None:
    dialog, window = _dialog_and_window()
    try:
        slider = getattr(dialog, f"{ROWS[row][0]}_slider")
        assert (slider.minimum(), slider.maximum()) == (1, len(settings.ELEV_STEP_PCT_STOPS))
        assert slider.singleStep() == 1
        assert slider.pageStep() == 1
        for index in range(slider.minimum(), slider.maximum() + 1):
            assert settings.elev_step_pct_for_index(index) in settings.ELEV_STEP_PCT_STOPS
    finally:
        _close(dialog, window)


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize(
    "saved_pct, expected_pct",
    [
        (48, 50),  # off-stop, snaps to the nearest
        (10, settings.ELEV_STEP_PCT_MIN),  # the old floor, snaps UP not to default
        (200, 200),  # already on a stop, unchanged
    ],
)
def test_slider_opens_on_the_saved_pct_s_snapped_stop(
    tmp_path, monkeypatch, row, saved_pct: int, expected_pct: int
) -> None:
    """Asserted against a concrete expected pct, not against the getter --
    comparing the widget to the same call the widget made would pass even
    with the conversion wired backwards. conftest's autouse
    _isolated_settings has already pointed CONFIG_PATH at this tmp_path, so
    writing the file here is what the dialog actually reads. The other row's
    key holds another stop, so a row reading the wrong key fails."""
    prefix, key, memo, *_rest = ROWS[row]
    (tmp_path / "config.yaml").write_text(f"{key}: {saved_pct}\n{_other_key(row)}: 175\n")
    monkeypatch.setattr(settings, memo, None)

    dialog, window = _dialog_and_window()
    try:
        expected_index = settings.ELEV_STEP_PCT_STOPS.index(expected_pct) + 1
        assert getattr(dialog, f"{prefix}_slider").value() == expected_index
        assert getattr(dialog, f"{prefix}_value_label").text().startswith(f"{expected_pct}%")
    finally:
        _close(dialog, window)


@pytest.mark.parametrize("row", ROWS)
def test_the_label_reads_a_pct_and_marks_its_own_default(row) -> None:
    prefix, _key, _memo, _get, default, suffix, *_rest = ROWS[row]
    dialog, window = _dialog_and_window()
    try:
        slider, label = getattr(dialog, f"{prefix}_slider"), getattr(dialog, f"{prefix}_value_label")
        default_index = settings.elev_step_index(default)
        slider.setValue(default_index)
        assert label.text() == f"{default}%{suffix}"

        other = 1 if default_index != 1 else 2
        slider.setValue(other)
        assert label.text() == f"{settings.elev_step_pct_for_index(other)}%"
    finally:
        _close(dialog, window)


def test_each_row_s_tooltip_and_guide_line_names_its_own_styles() -> None:
    """GH #288 split TASK-568's one row: Stepped's covers Flat with Isometric View and not Sloped,
    Sloped's carries the game's height (100% is one full half_h step)."""
    dialog, window = _dialog_and_window()
    try:
        stepped_tip, sloped_tip = dialog.elev_step_slider.toolTip(), dialog.sloped_elev_step_slider.toolTip()
    finally:
        _close(dialog, window)
    guide = (Path(__file__).resolve().parent.parent / "descape" / "help" / "settings.md").read_text(encoding="utf-8")
    lines = {row: next(line for line in guide.splitlines() if line.startswith(ROWS[row][6])) for row in ROWS}
    for text in (stepped_tip, lines["stepped"]):
        assert "Stepped" in text and "Isometric View" in text and "Sloped" not in text
    for text in (sloped_tip, lines["sloped"]):
        assert "Sloped" in text and "100% matches the game's own height" in text
    assert " -- " not in stepped_tip + sloped_tip


@pytest.mark.parametrize("on_screen", ["flat", "stepped", "sloped"])
@pytest.mark.parametrize("row", ROWS)
def test_an_apply_persists_only_its_own_key_and_redraws_only_its_style(
    tmp_path, monkeypatch, row, on_screen
) -> None:
    """A drag release persists the row's stop under its own key, leaves the other row's key
    unwritten, and redraws only when its style is the one rendered (ViewerWindow._render_style,
    which is "stepped" for Flat + Isometric View too)."""
    prefix, key, _memo, get, default, _suffix, _guide, style = ROWS[row]
    dialog, window = _dialog_and_window()
    try:
        monkeypatch.setattr(type(window), "_render_style", property(lambda _self: on_screen))
        redraws: list[None] = []
        monkeypatch.setattr(window, "refresh_map", lambda: redraws.append(None))
        slider = getattr(dialog, f"{prefix}_slider")
        wanted = 1 if settings.elev_step_index(default) != 1 else 2
        slider.setValue(wanted)
        slider.sliderReleased.emit()
        on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert on_disk[key] == settings.elev_step_pct_for_index(wanted) == get()
        assert _other_key(row) not in on_disk
        assert len(redraws) == int(on_screen == style), (row, on_screen, redraws)
        assert not getattr(dialog, f"_{prefix}_apply_timer").isActive()
    finally:
        _close(dialog, window)


@pytest.mark.parametrize("row", ROWS)
def test_each_row_s_default_button_lands_on_its_own_default(tmp_path, monkeypatch, row) -> None:
    """Disabled on the default stop and enabled off it; a click from another stop lands on the row's
    own default, persists only its key, leaves no timer armed and redraws as an apply does."""
    prefix, key, _memo, get, default, _suffix, _guide, style = ROWS[row]
    dialog, window = _dialog_and_window()
    try:
        monkeypatch.setattr(type(window), "_render_style", property(lambda _self: style))
        redraws: list[None] = []
        monkeypatch.setattr(window, "refresh_map", lambda: redraws.append(None))
        slider, button = getattr(dialog, f"{prefix}_slider"), getattr(dialog, f"{prefix}_default_button")
        default_index = settings.elev_step_index(default)
        assert slider.value() == default_index and not button.isEnabled()
        slider.setValue(1 if default_index != 1 else 2)
        assert button.isEnabled()
        button.click()
        assert slider.value() == default_index and not button.isEnabled()
        assert get() == default
        on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert on_disk[key] == default and _other_key(row) not in on_disk
        assert not getattr(dialog, f"_{prefix}_apply_timer").isActive()
        assert len(redraws) == 1
    finally:
        _close(dialog, window)


@pytest.mark.parametrize("how", ["close", "accept"])
def test_closing_the_dialog_applies_each_pending_slider_change_and_leaves_no_timer_armed(how) -> None:
    """TASK-463: a keyboard step arms a 200 ms debounce; the dialog outlives its close, so an
    armed timer fired later and wrote config.yaml after a test's settings isolation was gone.
    Passes when, after close() (a never-shown dialog, which skips done()) or accept(), no QTimer
    under the dialog is active and each slider's new value is already persisted."""
    from PyQt5.QtCore import QTimer

    dialog, window = _dialog_and_window()
    try:
        getters = {
            "elev_step_slider": lambda: settings.elev_step_index(settings.get_elev_step_pct()),
            "sloped_elev_step_slider": lambda: settings.elev_step_index(settings.get_sloped_elev_step_pct()),
            "graphics_quality_slider": settings.get_graphics_quality,
            "pan_speed_slider": settings.get_pan_speed,
            "wall_opacity_slider": settings.get_wall_opacity_pct,
        }
        wanted = {}
        for name in getters:
            slider = getattr(dialog, name)
            wanted[name] = slider.minimum() if slider.value() == slider.maximum() else slider.maximum()
            slider.setValue(wanted[name])
        assert dialog._elev_step_apply_timer.isActive(), "fixture assumption: a keyboard-path change debounces"
        assert dialog._sloped_elev_step_apply_timer.isActive(), "fixture assumption: a keyboard-path change debounces"
        getattr(dialog, how)()
        assert [t for t in dialog.findChildren(QTimer) if t.isActive()] == []
        assert {name: get() for name, get in getters.items()} == wanted
    finally:
        _close(dialog, window)


def test_the_two_sliders_line_up_and_keep_their_place_as_the_value_text_changes() -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    dialog, window = _dialog_and_window()
    try:
        dialog.show()
        tabs = dialog._tabs
        tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == "Appearance"))
        QApplication.processEvents()

        def spans():
            return [
                (slider.mapTo(dialog, QPoint(0, 0)).x(), slider.width())
                for slider in (dialog.elev_step_slider, dialog.sloped_elev_step_slider)
            ]

        before = spans()
        assert before[0] == before[1], f"Stepped's and Sloped's sliders differ: {before}"
        dialog._update_elev_step_label(75)  # "75%", shorter than its default's text
        QApplication.processEvents()
        assert spans() == before
    finally:
        _close(dialog, window)
