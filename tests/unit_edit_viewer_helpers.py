"""Shared fixtures and helpers for the tests/test_unit_edit_viewer_*.py modules:
phase 3.5b's place/move/delete/reassign driven through a real offscreen
ViewerWindow, on top of phase 3's selection wiring
(test_unit_selection_viewer.py) and phase 3.5a's headless model
(test_units_undo.py).

Uses the real units_120x120.aoe2scenario fixture, not the blank template's
SyntheticUnit stand-ins: UnitEditModel's construction gate needs real,
byte-verified unit structs (see unit_model.py's docstring), which a
synthetic dataclass can't satisfy.

Every test forces Flat terrain style before entering Units mode: MapView.
_pick_tile()'s Flat branch is a plain pixel/tile_px division, so scene
positions are trivial to construct by hand. Stepped/Sloped's own inverse
maths is covered by test_unit_pick.py, not duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QPointF, Qt

import conftest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


# A real object const from object_catalog's Units category -- Villager
# (unit_const 83), same as tools/gen_units_fixture.py uses for its own
# placed-unit coverage.
_PLACE_CONST = 83


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No unit edit viewer test wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    # Unchecked BEFORE the style switch, while still in Stepped -- iso_action
    # defaults checked (MapView._isometric's own default), so Flat would
    # otherwise render through the Flat+Isometric plan's real-iso path
    # instead of the plain top-down canvas these modules mean to exercise.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    _show_garrisoned(window)
    window.mode_combo.setCurrentText("Units")
    return window


def _window_with_style(style: str):
    """Like _window(), but for Stepped/Sloped instead of Flat -- Batch D's
    D5 scoped-patch wiring (dirty_screen_bbox_iso/_sloped via
    _patch_unit_edit_cache) has no Flat counterpart to exercise it against,
    since Flat's own branch there is patch_rects, not a bbox call."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.terrain_style_combo.setCurrentText(style)
    _show_garrisoned(window)
    window.mode_combo.setCurrentText("Units")
    return window


def _show_garrisoned(window) -> None:
    """GH #42 ships Filters > Show Garrisoned Units UNCHECKED, which would
    drop the fixture's garrisoned villager (reference_id 203, inside house
    200) out of the pick index and renumber every positional
    `index.entries[n]` in these modules. They test edit mechanics, not
    filtering, so it opts the whole fixture back into view; the tests that
    are about the hidden state turn it off again themselves."""
    window.show_garrisoned_action.setChecked(True)


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _unit_count(window) -> int:
    return sum(len(u) for u in window.scenario.unit_manager.units)


def _pos_for_tile(window, tile_x: int, tile_y: int) -> QPointF:
    tp = window.map_view._tile_pixels
    return QPointF(tile_x * tp + tp // 2, tile_y * tp + tp // 2)


def _empty_tile(window) -> tuple[int, int]:
    """A tile with no unit on it in the fixture -- for a pixel check that
    must start from a genuinely blank chunk, not one that happens to already
    hold a same-colour unit (see test_place_pixels_appear_where_none_existed's
    own docstring for why "was there a unit here already" is load-bearing)."""
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    return next((x, y) for x in range(60, 80) for y in range(60, 80) if (x, y) not in occupied)


def _chunk_for_tile(window, tile_x: int, tile_y: int) -> tuple[int, int]:
    tp = window.map_view._tile_pixels
    cp = window._cache.chunk_px
    return (tile_x * tp) // cp, (tile_y * tp) // cp


_BERRY_BUSH = 59  # FORAG, GAIA in every corpus placement


def _hide_garrisoned(window) -> None:
    """Back to the app's own default (_show_garrisoned's opposite): the
    occupant is not in the index and cannot be selected, which is the state
    every carry-along test has to work in."""
    window.show_garrisoned_action.setChecked(False)


def _fixture_host_and_occupant(window):
    """The fixture's House 200 and the Villager 203 inside it, moved onto the
    house's own point first -- the file places the villager one tile over,
    where a real occupant always shares its host's coordinates."""
    units = window.scenario.unit_manager.units[1]
    host = next(u for u in units if u.reference_id == _REF_HOUSE)
    occupant = next(u for u in units if u.reference_id == _REF_VILLAGER_P1)
    assert occupant.garrisoned_in_id == host.reference_id
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Co-locate", [1], fields_only=True):
        model.set_position(occupant, host.x, host.y, host.z)
    return host, occupant


def _place_tower(window):
    """A Watch Tower (5 places, holds villagers/foot/monks) on an empty tile,
    selected -- the fixture's own host is a House, which the game gives no
    garrison places at all."""
    model = window._ensure_unit_edits()
    tile = _empty_tile(window)
    with window._unit_edit(model, "Place tower", [1]):
        tower = model.add(1, 79, tile[0] + 0.5, tile[1] + 0.5)
    window._rebuild_unit_index()
    window._selection = [(1, tower.reference_id)]
    window._refresh_selection_view()
    return tower


def _entry(window, player_id: int, reference_id: int):
    return window.map_view._unit_index.entry_for_key((player_id, reference_id))


def _live(window, reference_id: int):
    return next(u for units in window.scenario.unit_manager.units for u in units if u.reference_id == reference_id)


def _tower_with_two_inside(window):
    tower = _place_tower(window)
    host = _entry(window, 1, tower.reference_id)
    window._garrison_add_const(host, 4)
    window._garrison_add_const(host, 83)
    occupants = [u for u in window.scenario.unit_manager.units[1] if u.garrisoned_in_id == tower.reference_id]
    assert len(occupants) == 2
    return tower, occupants


def _whole_canvas(window):
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()


def _rebuilt_canvas(window):
    """The same cache after a wholesale source rebuild, so the filter applies."""
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    window._cache.invalidate_units()
    window._cache.invalidate_region((0, 0, canvas_w, canvas_h), terrain_changed=False)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h).copy()


def _select_rows(window, reference_ids) -> None:
    tree = window.units_panel.garrison_tree
    tree.clearSelection()
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if item.data(0, Qt.UserRole) in reference_ids:
            item.setSelected(True)


def _second_tower(window):
    model = window._ensure_unit_edits()
    tile = _empty_tile(window)
    with window._unit_edit(model, "Place tower", [2]):
        other = model.add(2, 79, tile[0] + 0.5, tile[1] + 0.5)
    window._rebuild_unit_index()
    return other


_V159_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"


def _unit_layout(window) -> list[list[int]]:
    return [[id(u) for u in units] for units in window.scenario.unit_manager.units]


# Delete's wording, pinned because Cut Units (GH #192) shares its code with a different verb.


def _last_status_lines(window, count: int) -> list[str]:
    return window.status_log.toPlainText().splitlines()[-count:]


def _warnings(window) -> list[tuple[str, str]]:
    """Routes the window's warning dialogs into the returned list."""
    warned: list[tuple[str, str]] = []
    window._warn = lambda title, text: warned.append((title, text))
    return warned


def _orphan_reference_id(window, player: int, reference_id: int) -> tuple[int, int]:
    """Gives a unit the -1 reference_id the model refuses to remove; returns its new key."""
    unit = next(u for u in window.scenario.unit_manager.units[player] if u.reference_id == reference_id)
    unit.reference_id = -1
    window._rebuild_unit_index()
    return (player, -1)


_REF_ARCHER_P1 = 201  # type 70, angle_count 16 -- rotation IS an angle


_REF_VILLAGER_P1 = 203  # type 70, angle_count 16


_REF_WALL = 102  # GAIA wall -- rotation is a shape-variant index


_REF_HOUSE = 200  # angle_count 1 -- nothing to select, INERT


_REF_TREE_PINE = 101


_REF_ARCHER_P2 = 300


_REF_VILLAGER_P2 = 301


def _select(window, *reference_ids):
    """Selects the fixture units with these reference_ids, through the same
    reconciliation point every real selection path uses."""
    index = window.map_view._unit_index
    entries = [e for e in index.entries if e.unit.reference_id in reference_ids]
    assert len(entries) == len(reference_ids), "fixture unit missing from the index"
    window._selection = [(e.player_id, e.unit.reference_id) for e in entries]
    window._refresh_selection_view()
    return entries


_REF_TREE_OAK = 100  # GAIA oak, 42 variants, stored 7.0


_GATE_NE, _GATE_E, _GATE_SE = 64, 659, 88


def _make_gate(window, x: float = 10.0, y: float = 5.5):
    (entry,) = _select(window, _REF_WALL)
    entry.unit.unit_const, entry.unit.x, entry.unit.y = _GATE_NE, x, y
    window._after_unit_mutation()
    return _select(window, _REF_WALL)[0]


def _field(field_id):
    from descape import unit_fields

    return unit_fields.FIELDS_BY_ID[field_id]


def _owner_of(window, reference_id: int) -> int:
    units = window.scenario.unit_manager.units
    return next(p for p, us in enumerate(units) if any(u.reference_id == reference_id for u in us))


def _make_stack(window, count: int = 3) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Places `count` villagers at one identical point on an empty tile, in
    one wholesale edit. Returns (tile, keys top-first)."""
    tile = _empty_tile(window)
    model = window._ensure_unit_edits()
    units = []
    with window._unit_edit(model, "Stack", [1]):
        units.extend(model.add(1, _PLACE_CONST, tile[0] + 0.5, tile[1] + 0.5) for _ in range(count))
    keys = [(1, u.reference_id) for u in reversed(units)]
    return tile, keys


def _pos_in_tile(window, tile_x: int, tile_y: int, fx: float, fy: float) -> QPointF:
    """A scene position at a chosen sub-tile fraction. Flat only, which is
    what these modules force -- one continuous tile unit is tile_px pixels."""
    tp = window.map_view._tile_pixels
    return QPointF((tile_x + fx) * tp, (tile_y + fy) * tp)


_BARRACKS_CONST = 12  # 3x3


_HOUSE_CONST = 70  # 2x2


def _add_units(window, placements):
    """[(const, x, y)] added for player 1 in one wholesale edit; returns keys."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [1]):
        units = [model.add(1, const, x, y) for const, x, y in placements]
    return [(1, u.reference_id) for u in units]


def _free_tiles(window, tiles) -> None:
    occupied = {(int(u.x), int(u.y)) for units in window.scenario.unit_manager.units for u in units}
    assert not occupied & set(tiles), "the fixture already holds a unit on a test tile"


def _select_keys(window, keys) -> None:
    window._selection = list(keys)
    window._refresh_selection_view()


def _positions(window, keys):
    index = window.map_view._unit_index
    return [(index.entry_for_key(k).unit.x, index.entry_for_key(k).unit.y) for k in keys]
