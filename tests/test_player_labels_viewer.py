"""GH #130: every player selector shows "P<n> - <tribe name>" and the player's
swatch, through a real offscreen ViewerWindow. The label helper itself is
tests/test_player_labels.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"
_REF_ARCHER_P2 = 300  # units_120x120's P2 archer
_VILLAGER = 83
_NAME = "Horde Army"


def _window(path: Path):
    window = conftest.shown_window(1500, 900)
    window.load_scenario(path)
    assert window.scenario is not None, f"{path.name} failed to load"
    return window


def _spec(window, field_id: str):
    from descape import player_fields

    return {s.field_id: s for s in player_fields.specs_for(window.scenario)}[field_id]


def _row_by_data(combo, pid: int):
    index = combo.findData(pid)
    assert index >= 0, f"no row for pid {pid}"
    return index


def _swatch_rgb(icon) -> tuple[int, int, int]:
    assert not icon.isNull()
    pixel = icon.pixmap(16, 16).toImage().pixelColor(8, 8)
    return (pixel.red(), pixel.green(), pixel.blue())


def _scatter_dialog(window, monkeypatch):
    """The Scatter dialog scatter_units_in_region() builds, captured instead of shown."""
    from PyQt5.QtWidgets import QDialog

    from descape import viewer

    built = []

    class _Captured(viewer.ScatterDialog):
        def exec_(self):
            built.append(self)
            return QDialog.Rejected

    monkeypatch.setattr(viewer, "ScatterDialog", _Captured)
    window._region = (0, 0, 10, 10)
    window.units_panel.select_object(_VILLAGER)
    window.scatter_units_in_region()
    assert len(built) == 1, "the Scatter dialog was not built"
    return built[0]


def _selectors(window, monkeypatch, pid: int = 2) -> dict[str, tuple]:
    """(text, icon) of pid's row in every in-scope selector, keyed by the plan's ids."""
    units = window.units_panel
    combos = {
        "A": units.owner_combo,
        "B": units.unit_field_editors["player"],
        "I": window.stats_player_combo,
        "J": _scatter_dialog(window, monkeypatch).owner_combo,
    }
    out = {key: (c.itemText(_row_by_data(c, pid)), c.itemIcon(_row_by_data(c, pid))) for key, c in combos.items()}
    players = window.players_panel.player_combo
    out["C"] = (players.itemText(pid - 1), players.itemIcon(pid - 1))
    diplomacy = window.diplomacy_panel.player_combo
    d_index = window.diplomacy_panel._active_players.index(pid)
    out["D"] = (diplomacy.itemText(d_index), diplomacy.itemIcon(d_index))
    for key, actions in (("G", window.convert_source_actions), ("H", window.player_actions)):
        out[key] = (actions[pid].text(), actions[pid].icon())
    disables = window._build_disables_dialog().player_combo
    out["K"] = (disables.itemText(pid - 1), disables.itemIcon(pid - 1))
    return out


# Checkable menu actions stay text-only: Fusion draws an action's icon in place of its checkmark.
_TEXT_ONLY = frozenset({"G", "H"})


def _assert_texts(selectors, expected: str) -> None:
    wrong = {key: text for key, (text, _icon) in selectors.items() if text != expected}
    assert not wrong, f"expected {expected!r} in every selector, got {wrong}"
    null = [key for key, (_text, icon) in selectors.items() if icon.isNull() and key not in _TEXT_ONLY]
    assert not null, f"no swatch on {null}"
    iconned = [key for key in _TEXT_ONLY if not selectors[key][1].isNull()]
    assert not iconned, f"a checkable menu action carries an icon: {iconned}"


def test_a_tribe_name_edit_relabels_every_selector_and_undo_redo_follow(monkeypatch) -> None:
    window = _window(UNITS_FIXTURE)
    try:
        # Build C and D first, so the edit exercises the relabel-in-place path.
        window.mode_combo.setCurrentText("Diplomacy")
        window.mode_combo.setCurrentText("Players")
        _assert_texts(_selectors(window, monkeypatch), "P2")

        window.set_player_field(_spec(window, "tribe_name"), 2, _NAME)
        _assert_texts(_selectors(window, monkeypatch), f"P2 - {_NAME}")
        owner = window.units_panel.owner_combo
        assert owner.itemData(_row_by_data(owner, 2), 3) == f"P2 - {_NAME}"  # Qt.ToolTipRole
        # Other players and GAIA keep their bare labels.
        assert owner.itemText(_row_by_data(owner, 0)) == "GAIA"
        assert owner.itemText(_row_by_data(owner, 1)) == "P1"

        window.undo()
        _assert_texts(_selectors(window, monkeypatch), "P2")
        window.redo()
        _assert_texts(_selectors(window, monkeypatch), f"P2 - {_NAME}")
    finally:
        conftest.close_window(window)


def test_a_colour_edit_changes_the_swatches_but_not_the_text(monkeypatch) -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Diplomacy")
        window.mode_combo.setCurrentText("Players")
        window.set_player_field(_spec(window, "tribe_name"), 2, _NAME)
        before = _selectors(window, monkeypatch)
        old_rgb = tuple(window.scenario.player_colors[2])
        swatched = [key for key in before if key not in _TEXT_ONLY]
        assert {_swatch_rgb(before[key][1]) for key in swatched} == {old_rgb}

        from descape import player_fields

        color_spec = _spec(window, "color")
        own = player_fields.current_value(window.scenario, color_spec, 2)
        combo = window.players_panel.widget_for("color")
        new_value = next(
            combo.itemData(i) for i in range(combo.count())
            if combo.itemData(i) is not None and combo.itemData(i) != own
        )
        window.set_player_field(color_spec, 2, new_value)
        new_rgb = tuple(window.scenario.player_colors[2])
        assert new_rgb != old_rgb
        after = _selectors(window, monkeypatch)
        _assert_texts(after, f"P2 - {_NAME}")
        wrong = {key: _swatch_rgb(after[key][1]) for key in swatched if _swatch_rgb(after[key][1]) != new_rgb}
        assert not wrong, f"stale swatch (expected {new_rgb}): {wrong}"
    finally:
        conftest.close_window(window)


def test_a_player_count_change_keeps_the_index_mapped_selectors_rows() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Players")
        window.set_player_field(_spec(window, "tribe_name"), 3, _NAME)
        spin = window.players_panel.player_count_spin
        assert spin.isEnabled()
        spin.setValue(spin.value() + 1)
        combo = window.players_panel.player_combo
        assert combo.count() == 8
        assert [combo.itemText(i) for i in range(3)] == ["P1", "P2", f"P3 - {_NAME}"]
        disables = window._build_disables_dialog().player_combo
        assert disables.count() == 8
        assert disables.itemText(2) == f"P3 - {_NAME}"
    finally:
        conftest.close_window(window)


def test_relabelling_the_current_owner_rows_commits_nothing(monkeypatch) -> None:
    """setItemText on a combo's current row emits currentTextChanged, which
    no player combo may connect: a relabel must never look like an edit."""
    window = _window(UNITS_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Units")
        index = window.map_view._unit_index
        (entry,) = [e for e in index.entries if e.unit.reference_id == _REF_ARCHER_P2]
        window._selection = [(entry.player_id, entry.unit.reference_id)]
        window._refresh_selection_view()
        units = window.units_panel
        inspector = units.unit_field_editors["player"]
        assert inspector.currentData() == 2
        units.select_owner(2)
        commits = []
        monkeypatch.setattr(units, "_on_unit_field", lambda *args: commits.append(args))
        cursor = window.edit_history.cursor

        window.set_player_field(_spec(window, "tribe_name"), 2, _NAME)
        assert inspector.currentText() == f"P2 - {_NAME}"
        assert units.owner_combo.currentText() == f"P2 - {_NAME}"
        assert inspector.currentData() == 2
        assert units.owner_id() == 2
        assert commits == []
        assert window.edit_history.cursor == cursor + 1  # the tribe name edit alone
    finally:
        conftest.close_window(window)


def test_closing_the_document_resets_the_labels_and_drops_the_swatches() -> None:
    window = _window(UNITS_FIXTURE)
    try:
        window.set_player_field(_spec(window, "tribe_name"), 2, _NAME)
        window.edit_history.mark_saved()  # a dirty close would open a modal prompt
        window.close_scenario()
        owner = window.units_panel.owner_combo
        row = _row_by_data(owner, 2)
        assert owner.itemText(row) == "P2"
        assert owner.itemIcon(row).isNull()
        assert window.player_actions[2].text() == "P2"
    finally:
        conftest.close_window(window)


def _player_id_rows(panel):
    from descape.trigger_panel import PLAYER_ID_PRESENTATION

    return [w for spec, _k, _i, w in panel._rows if spec.presentation == PLAYER_ID_PRESENTATION]


def test_trigger_player_fields_show_the_label_live_and_on_rebuild() -> None:
    """F: the armour split trigger's first effect has source_player = 1
    (tools/gen_trigger_fixture.py)."""
    window = _window(TRIGGER_FIXTURE)
    try:
        window.mode_combo.setCurrentText("Triggers")
        panel = window.trigger_panel
        panel.select_trigger(1)
        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(2).child(0))
        (combo,) = _player_id_rows(panel)
        assert combo.currentData() == 1
        assert combo.currentText() == "P1"
        assert combo.itemText(_row_by_data(combo, 0)) == "GAIA"

        window.set_player_field(_spec(window, "tribe_name"), 1, _NAME)
        assert combo.currentText() == f"P1 - {_NAME}", "the on-screen combo was not relabelled"
        assert not combo.itemIcon(combo.currentIndex()).isNull()
        assert combo.currentData() == 1

        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(2).child(1))
        panel.entry_tree.setCurrentItem(panel.entry_tree.topLevelItem(2).child(0))
        (rebuilt,) = _player_id_rows(panel)
        assert rebuilt is not combo
        assert rebuilt.currentText() == f"P1 - {_NAME}", "a rebuilt form lost the label"
    finally:
        conftest.close_window(window)


def test_the_disables_dialog_draws_its_swatches() -> None:
    """K's old `pid in colors` test never matched a pid-indexed tuple."""
    window = _window(UNITS_FIXTURE)
    try:
        combo = window._build_disables_dialog().player_combo
        for i in range(combo.count()):
            assert _swatch_rgb(combo.itemIcon(i)) == tuple(window.scenario.player_colors[i + 1])
    finally:
        conftest.close_window(window)


def _example(name: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "examples" / name
    if not path.is_file():
        pytest.skip(f"{name} not in examples/")
    return path


@pytest.mark.corpus
def test_gh130_old_allies_labels_and_a_saved_rename_survive_reload(monkeypatch, tmp_path) -> None:
    """The issue's own file: P8's tribe name is "Horde Army". A rename saved
    and reopened reads back in every selector."""
    import shutil

    copy = tmp_path / "old-allies.aoe2scenario"
    shutil.copyfile(_example("old-allies-final-v2.aoe2scenario"), copy)
    window = _window(copy)
    try:
        owner = window.units_panel.owner_combo
        row = _row_by_data(owner, 8)
        assert owner.itemText(row) == "P8 - Horde Army"
        assert _swatch_rgb(owner.itemIcon(row)) == tuple(window.scenario.player_colors[8])

        window.mode_combo.setCurrentText("Diplomacy")
        window.mode_combo.setCurrentText("Players")
        window.set_player_field(_spec(window, "tribe_name"), 8, "Horde Army Renamed")
        _assert_texts(_selectors(window, monkeypatch, 8), "P8 - Horde Army Renamed")
        window.save()
        assert not window.edit_history.is_dirty, "the save did not complete"

        window.load_scenario(copy)
        window.mode_combo.setCurrentText("Diplomacy")
        window.mode_combo.setCurrentText("Players")
        _assert_texts(_selectors(window, monkeypatch, 8), "P8 - Horde Army Renamed")
    finally:
        conftest.close_window(window)
