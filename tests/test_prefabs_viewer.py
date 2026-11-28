"""The window's prefab wiring (TASK-269): Save as Prefab, the Prefabs tab's
rows, Use, Delete and the refusals Use inherits from Paste. prefabs.py's
own format rules are tests/test_prefabs.py. Every ViewerWindow() here must
call edit_history.mark_saved() before close()."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from descape import games, prefabs
from descape.region_clipboard import RegionBlock, RegionUnit
from descape.scenario_io import TEMPLATE_DIR

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ANCHOR = (40, 40)
_ROR_BLANK = TEMPLATE_DIR / games.get("ror").blank_template
_UNKNOWN_CONST = 65000


def _window(path: Path | None = None):
    window = conftest.terrain_edit_window() if path is None else conftest.blank_window(load=False)
    if path is not None:
        window.load_scenario(path)
        window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("select")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _copied_block(window) -> RegionBlock:
    """A 4x3 region with painted terrain, a layer, an elevation and a garrisoned castle."""
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(2)
    for y in range(3):
        for x in range(4):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    mm = window.scenario.map_manager
    mm.terrain[1].layer = 9
    edits = window._ensure_unit_edits()
    castle = edits.add(player=1, unit_const=82, x=2.5, y=1.5, z=0.0, rotation=0.0)
    villager = edits.add(player=1, unit_const=83, x=0.5, y=0.5, z=0.0, rotation=0.0)
    edits.set_garrisoned_in(villager, castle.reference_id)
    window._on_tool_selected("select")
    window.on_region_selected((0, 0, 4, 3))
    window.copy_region()
    return window._region_clipboard


def _paste(window) -> int:
    window.paste_terrain_check.setChecked(True)
    window.paste_elevation_check.setChecked(True)
    window.paste_units_check.setChecked(True)
    window.on_hover(_ANCHOR)
    before = len(window.edit_history.records)
    window.paste_region()
    return len(window.edit_history.records) - before


def _snapshot(window):
    mm = window.scenario.map_manager
    tiles = [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain]
    units = window.scenario.unit_manager.units
    by_id = {u.reference_id: u for player in units for u in player}
    rows = []
    for player, player_units in enumerate(units):
        for u in player_units:
            host = by_id.get(u.garrisoned_in_id)
            held_in = None if host is None else (host.x, host.y)
            rows.append((player, u.unit_const, u.x, u.y, u.z, u.rotation, u.status, u.initial_animation_frame, _caption(u), held_in))
    return tiles, sorted(rows)


def _caption(unit):
    """Caption and capture flag, read the way region_clipboard.py does (absent before 1.54/1.59)."""
    out = []
    for name in ("caption_string_id", "caption_string", "capture_flag"):
        try:
            out.append(getattr(unit, name))
        except Exception:  # noqa: BLE001 (the library's UnsupportedAttributeError)
            out.append(None)
    return tuple(out)


def _status(window) -> str:
    return window.status_log.toPlainText()


def test_a_saved_prefab_used_in_a_fresh_window_pastes_like_the_copy(monkeypatch) -> None:
    source = _window()
    try:
        block = _copied_block(source)
        monkeypatch.setattr(source, "_ask_prefab_name", lambda default: "Walled Base")
        source.save_prefab_action.trigger()
    finally:
        _close(source)
    (info,) = prefabs.list_prefabs()
    assert (info.name, info.unit_count) == ("Walled Base", 2)

    in_session = _window()
    try:
        in_session._push_clipboard(block)
        assert _paste(in_session) == 1
        expected = _snapshot(in_session)
    finally:
        _close(in_session)

    fresh = _window()
    try:
        fresh.prefabs_action.trigger()
        dialog = fresh._clipboard_dialog
        assert dialog.tabs.currentIndex() == 1
        assert dialog.prefab_tree.topLevelItemCount() == 1
        assert dialog.prefab_tree.topLevelItem(0).text(1) == "Walled Base"
        fresh._use_prefab(str(info.path))
        assert fresh._clipboard_history.active.label == "Walled Base"
        assert _paste(fresh) == 1
        assert _snapshot(fresh) == expected
    finally:
        _close(fresh)


@pytest.mark.parametrize(("dataset", "target"), [("ror", None), (games.DEFAULT_ID, _ROR_BLANK)], ids=["ror-into-aoe2", "aoe2-into-ror"])
def test_another_games_prefab_is_refused_at_paste(dataset, target) -> None:
    block = RegionBlock(2, 2, (2, 2, 2, 2), (0, 0, 0, 0), (-1, -1, -1, -1), (), dataset=dataset)
    path = prefabs.save_block(block, "Other")
    window = _window(target)
    try:
        window._show_prefabs()
        assert "(" in window._clipboard_dialog.prefab_tree.topLevelItem(0).text(1)  # the game note
        window._use_prefab(str(path))
        assert window._region_clipboard.dataset == dataset  # Use never restamps
        assert _paste(window) == 0
        assert "Paste refused" in _status(window)
        assert "does not define" not in _status(window)
    finally:
        _close(window)


def test_unknown_objects_are_counted_on_use() -> None:
    unit = RegionUnit(1, _UNKNOWN_CONST, 0.5, 0.5, 0.0, 0.0, 2, 0, -1, "", -1)
    block = RegionBlock(1, 1, (), (), (), (unit, dataclasses.replace(unit, unit_const=83)))
    window = _window()
    try:
        window._use_prefab(str(prefabs.save_block(block, "Modded")))
        assert "1 unit uses an object this game data does not define" in _status(window)
    finally:
        _close(window)


def test_save_as_prefab_needs_only_a_clipboard_entry() -> None:
    window = conftest.blank_window(load=False)
    try:
        assert not window.save_prefab_action.isEnabled()
        assert window.prefabs_action.isEnabled()
        window._push_clipboard(RegionBlock(1, 1, (2,), (0,), (-1,), ()))
        assert window.save_prefab_action.isEnabled()
    finally:
        _close(window)
    window = _window()
    try:
        window.mode_combo.setCurrentText("Triggers")
        assert window.mode == "triggers" and not window.save_prefab_action.isEnabled()
        window._push_clipboard(RegionBlock(1, 1, (2,), (0,), (-1,), ()))
        assert window.save_prefab_action.isEnabled()
    finally:
        _close(window)


def test_saving_onto_a_taken_name_asks_and_delete_keeps_the_session_entry(monkeypatch) -> None:
    window = _window()
    try:
        _copied_block(window)
        monkeypatch.setattr(window, "_ask_prefab_name", lambda default: "Base")
        window._save_entry_as_prefab()
        (first,) = prefabs.list_prefabs()

        answers = []
        monkeypatch.setattr(window, "_confirm_prefab", lambda title, text: answers.append(title) or False)
        window._save_entry_as_prefab()
        assert answers == ["Replace Prefab"] and len(prefabs.list_prefabs()) == 1
        monkeypatch.setattr(window, "_confirm_prefab", lambda title, text: True)
        window._save_entry_as_prefab()
        assert [i.path for i in prefabs.list_prefabs()] == [first.path]

        window._use_prefab(str(first.path))
        used = window._clipboard_history.active_id
        window._delete_prefab(str(first.path))
        assert not first.path.exists() and prefabs.list_prefabs() == []
        assert window._clipboard_history.active_id == used
    finally:
        _close(window)


def test_a_legacy_prefab_is_marked_on_a_de_document() -> None:
    path = prefabs.save_block(RegionBlock(1, 1, (2,), (0,), (-1,), (), legacy=True), "Old")
    window = _window()
    try:
        window._show_prefabs()
        assert window._clipboard_dialog.prefab_tree.topLevelItem(0).text(1) == "Old (older format)"
        window._use_prefab(str(path))
        assert _paste(window) == 0
    finally:
        _close(window)

