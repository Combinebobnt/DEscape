"""The "Stepped elevation height" slider's value space is the stop index,
not the pct (Track B-D-e of the mip-level plan).

The point of the phase: an off-stop pct enumerates a shallower mip ladder
than the nearest stop would, and setSingleStep() alone doesn't prevent one --
it governs arrow keys and the wheel, while a drag goes through
QStyle::sliderValueFromPosition and lands anywhere. Making the slider's own
range the 8 stop indices is what makes an off-stop value unrepresentable.

Same offscreen-ViewerWindow technique as tests/test_keybinds.py, including
constructing SettingsDialog directly rather than through
ViewerWindow._show_settings() (that method calls exec_(), which is modal and
would hang an offscreen run).
"""

from __future__ import annotations

import pytest

import conftest
from descape import settings

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _dialog_and_window():
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    return SettingsDialog(window), window


def test_slider_range_is_the_stop_indices_not_the_pct_range() -> None:
    dialog, window = _dialog_and_window()
    try:
        slider = dialog.elev_step_slider
        assert (slider.minimum(), slider.maximum()) == (1, len(settings.ELEV_STEP_PCT_STOPS))
        assert slider.singleStep() == 1
        assert slider.pageStep() == 1
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "saved_pct, expected_pct",
    [
        (48, 50),  # off-stop, snaps to the nearest
        (10, settings.ELEV_STEP_PCT_MIN),  # the old floor, snaps UP not to default
        (200, 200),  # already on a stop, unchanged
    ],
)
def test_slider_opens_on_the_saved_pct_s_snapped_stop(
    tmp_path, monkeypatch, saved_pct: int, expected_pct: int
) -> None:
    """Asserted against a concrete expected pct, not against
    get_elev_step_pct() -- comparing the widget to the same call the widget
    made would pass even with the conversion wired backwards. conftest's
    autouse _isolated_settings has already pointed CONFIG_PATH at this
    tmp_path and cleared the memo, so writing the file here is what the
    dialog actually reads."""
    (tmp_path / "config.yaml").write_text(f"elev_step_pct: {saved_pct}\n")
    monkeypatch.setattr(settings, "_elev_step_pct", None)

    dialog, window = _dialog_and_window()
    try:
        expected_index = settings.ELEV_STEP_PCT_STOPS.index(expected_pct) + 1
        assert dialog.elev_step_slider.value() == expected_index
        assert dialog.elev_step_value_label.text().startswith(f"{expected_pct}%")
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()


def test_every_reachable_slider_position_maps_to_a_legal_stop() -> None:
    """Exhaustive rather than sampled -- there are only 8 positions, and the
    guarantee being tested is that NO position yields an off-stop pct."""
    dialog, window = _dialog_and_window()
    try:
        slider = dialog.elev_step_slider
        for index in range(slider.minimum(), slider.maximum() + 1):
            pct = settings.elev_step_pct_for_index(index)
            assert pct in settings.ELEV_STEP_PCT_STOPS
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()


def test_the_label_reads_a_pct_and_still_marks_the_default() -> None:
    """_update_elev_step_label keeps taking a pct, so its "(Tall, default)"
    test against ELEV_STEP_DEFAULT_PCT survives the index conversion."""
    from descape import iso_geometry

    dialog, window = _dialog_and_window()
    try:
        default_index = settings.elev_step_index(iso_geometry.ELEV_STEP_DEFAULT_PCT)
        dialog.elev_step_slider.setValue(default_index)
        assert dialog.elev_step_value_label.text() == f"{iso_geometry.ELEV_STEP_DEFAULT_PCT}% (Tall, default)"

        other = 1 if default_index != 1 else 2
        dialog.elev_step_slider.setValue(other)
        assert dialog.elev_step_value_label.text() == f"{settings.elev_step_pct_for_index(other)}%"
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()
