"""Edit > Scatter Units in Region's real modal dialog (GH #105), opened by
ViewerWindow.scatter_units_in_region() with exec_() left alone.

tests/test_unit_edit_viewer.py patches ScatterDialog.exec_ to answer the dialog;
this file lets the real exec_() run its own event loop and answers the shown
dialog from a polling QTimer instead. The poller always closes the
ScatterDialog it finds, by OK or by reject, so a failed check can never leave
a modal loop blocking the run; with no dialog shown it stops after a bounded
number of ticks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_FISH_CONST = 456
_REF_HOUSE = 200
_ONE_TILE = (60, 60, 61, 61)
_MAX_TICKS = 200


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    return window


def _answer_real_dialog(window, act):
    """Polls for the shown ScatterDialog, hands it to `act` (which returns
    whether to press OK), then presses OK or rejects. Returns the record."""
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QDialogButtonBox

    from descape.scatter_dialog import ScatterDialog

    record: dict = {"ticks": 0, "seen": None, "error": None}
    timer = QTimer(window)
    timer.setInterval(10)

    def tick() -> None:
        record["ticks"] += 1
        shown = [d for d in window.findChildren(ScatterDialog) if d.isVisible()]
        if not shown:
            if record["ticks"] > _MAX_TICKS:
                timer.stop()
            return
        timer.stop()
        dialog = shown[0]
        record["seen"] = dialog
        press_ok = False
        try:
            record["modal"] = dialog.isModal()
            press_ok = act(dialog)
        except Exception as exc:  # noqa: BLE001 -- reported by the test, never raised inside Qt's loop
            record["error"] = repr(exc)
        if press_ok:
            dialog.buttons.button(QDialogButtonBox.Ok).click()
        else:
            dialog.reject()

    timer.timeout.connect(tick)
    timer.start()
    return record


def test_a_one_tile_region_with_a_house_selected_enables_ok_in_the_real_dialog() -> None:
    """GH #105 step 2. Pass conditions:
    - the real ScatterDialog opens modal from Edit > Scatter Units in Region;
    - switched to "Move the selected units", its count line reads "1 selected
      units, 1 eligible tiles in a 1x1 region (1 tiles)" and OK is enabled;
    - OK accepts: the 2x2 house cannot fit one tile, so it stays put, nothing
      is recorded and the status says nothing moved."""
    from PyQt5.QtWidgets import QDialogButtonBox

    window = _window()
    try:
        window.units_panel.select_object(_FISH_CONST)
        window.on_region_selected(_ONE_TILE)
        index = window.map_view._unit_index
        (house,) = [e for e in index.entries if e.unit.reference_id == _REF_HOUSE]
        window._selection = [(house.player_id, _REF_HOUSE)]
        window._refresh_selection_view()
        position = (house.unit.x, house.unit.y)
        records = len(window.edit_history.records)
        checked: dict = {}

        def act(dialog) -> bool:
            dialog.move_radio.setChecked(True)
            checked["count"] = dialog.count_label.text()
            checked["ok"] = dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
            return checked["ok"]

        record = _answer_real_dialog(window, act)
        window.scatter_units_in_region()

        assert record["error"] is None, record["error"]
        assert record["seen"] is not None, "the real dialog never showed"
        assert record["modal"]
        assert checked["count"] == "1 selected units, 1 eligible tiles in a 1x1 region (1 tiles)"
        assert checked["ok"], "OK is disabled on a one-tile region with a house selected"
        assert (house.unit.x, house.unit.y) == position
        assert len(window.edit_history.records) == records
        assert "nothing moved" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)
