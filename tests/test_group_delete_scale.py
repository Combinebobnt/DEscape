"""TASK-031.01: a box-selected group of a few hundred trees deletes in one
record, and undo and redo each give an exact unit layout.

tests/test_unit_edit_viewer.py's group-delete tests use two or three units.
This runs the same viewer path (marquee keys -> on_marquee_select ->
on_unit_delete -> undo/redo) at the in-app check's scale. The default-tier
case builds its own scenario through the real write path: 420 GAIA trees with
a gold mine every fifth placement, so survivors and deleted units interleave
in the GAIA list. The corpus case box-selects the densest GAIA block of
2_Joan_coop_2_v0_15, the file the in-app check names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_unit_edit_viewer import (
    FIXTURE_PATH,
    _close,
    _no_install_env,  # noqa: F401 -- autouse fixture
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TREE_CONSTS = (349, 350)  # the fixture's own GAIA trees
_GOLD_MINE = 66
# The synthetic forest: every other tile of a 42x20 block, from tile (40, 40).
_BLOCK = (40, 40, 82, 60)


def _flat_units_window(path: Path):
    """test_unit_edit_viewer._window() for any file: Flat top-down, garrisoned shown, Units mode."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(path)
    assert window.scenario is not None, f"{path.name} failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show_garrisoned_action.setChecked(True)
    window.mode_combo.setCurrentText("Units")
    return window


def _layout(window) -> list[list[tuple]]:
    """Every player list in order, each unit as its identity plus every field a delete could disturb."""
    return [
        [
            (
                id(u),
                u.reference_id,
                u.unit_const,
                u.x,
                u.y,
                u.z,
                u.rotation,
                u.initial_animation_frame,
                u.status,
                u.garrisoned_in_id,
            )
            for u in units
        ]
        for units in window.scenario.unit_manager.units
    ]


def _marquee(window, tx0: int, ty0: int, tx1: int, ty1: int) -> list[tuple[int, int]]:
    """A plain marquee over the half-open tile rect, through MapView's own hit test."""
    from PyQt5.QtCore import QRectF, Qt

    tp = window.map_view._tile_pixels
    rect = QRectF(tx0 * tp, ty0 * tp, (tx1 - tx0) * tp - 1, (ty1 - ty0) * tp - 1)
    keys = window.map_view._units_in_scene_rect(rect)
    window.on_marquee_select(keys, Qt.NoModifier)
    return keys


def _check_delete_undo_redo(window, keys: list[tuple[int, int]]) -> None:
    """Pass conditions, for the group now selected:
    - Delete removes exactly the selected units in one undo record, every survivor keeping its
      list order and fields, and no selected key left in the pick index;
    - undo gives back the exact pre-delete layout (same objects, order and fields) and index;
    - redo gives the exact post-delete layout again, and a second undo the exact original.
    """
    from PyQt5.QtCore import Qt

    assert sorted(window._selection) == sorted(keys)
    before = _layout(window)
    doomed = {id(window.map_view._unit_index.entry_for_key(k).unit) for k in keys}
    assert len(doomed) == len(keys)
    expected_after = [[row for row in units if row[0] not in doomed] for units in before]
    records_before = len(window.edit_history.records)

    window.on_unit_delete(Qt.NoModifier)

    after = _layout(window)
    assert after == expected_after
    assert sum(map(len, before)) - sum(map(len, after)) == len(keys)
    assert len(window.edit_history.records) == records_before + 1
    index = window.map_view._unit_index
    assert all(index.entry_for_key(k) is None for k in keys)

    window.undo()
    assert _layout(window) == before
    index = window.map_view._unit_index
    assert all(index.entry_for_key(k) is not None for k in keys)

    window.redo()
    assert _layout(window) == expected_after
    window.undo()
    assert _layout(window) == before


def _write_forest(out: Path) -> int:
    """The units fixture plus a GAIA forest over _BLOCK, written through write_scenario; returns its size."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.terrain_units import UnitAddSpec
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(FIXTURE_PATH)
    model = UnitEditModel(loaded)
    x0, y0, x1, y1 = _BLOCK
    specs = []
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            if (tx + ty) % 2:
                continue
            n = len(specs)
            const = _GOLD_MINE if n % 5 == 4 else _TREE_CONSTS[n % 2]
            variant = 0 if const == _GOLD_MINE else n % 4
            specs.append(UnitAddSpec(tx + 0.5, ty + 0.5, const, float(variant), variant))
    model.add_many(0, specs)
    write_scenario(loaded, out, backup=False, units=model)
    return len(specs)


# 200 stays under unit_model._REMOVE_BY_INDEX_MAX (256), so remove_many() deletes per index;
# 280 is past it, so remove_many() takes its one-filtering-pass branch.
@pytest.mark.parametrize("columns", [20, 28], ids=["200-by-index", "280-filter-pass"])
def test_deleting_a_few_hundred_box_selected_trees_undoes_and_redoes_exactly(tmp_path, columns: int) -> None:
    path = tmp_path / "forest.aoe2scenario"
    assert _write_forest(path) == 420
    window = _flat_units_window(path)
    try:
        x0, y0, x1, y1 = _BLOCK
        assert x0 + 3 + columns < x1
        # Starts 3 columns in and stops short of the far edge: units on both sides of the box survive.
        keys = _marquee(window, x0 + 3, y0, x0 + 3 + columns, y1)
        consts = [window.map_view._unit_index.entry_for_key(k).unit.unit_const for k in keys]
        assert len(keys) == 10 * columns
        assert sum(c in _TREE_CONSTS for c in consts) == 8 * columns and _GOLD_MINE in consts
        _check_delete_undo_redo(window, keys)
    finally:
        _close(window)


@pytest.mark.corpus
def test_deleting_the_densest_gaia_block_of_joan_2_undoes_and_redoes_exactly() -> None:
    from tests.conftest import ROOT

    path = ROOT / "examples" / "2_Joan_coop_2_v0_15.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    window = _flat_units_window(path)
    try:
        side = 24
        gaia = [(int(e.unit.x), int(e.unit.y)) for e in window.map_view._unit_index.entries if e.player_id == 0]
        width, height = window.map_view._map_width, window.map_view._map_height
        best = max(
            ((tx, ty) for tx in range(0, width - side, 4) for ty in range(0, height - side, 4)),
            key=lambda c: sum(c[0] <= x < c[0] + side and c[1] <= y < c[1] + side for x, y in gaia),
        )
        keys = _marquee(window, best[0], best[1], best[0] + side, best[1] + side)
        assert len(keys) >= 200, f"densest {side}x{side} block at {best} holds only {len(keys)} units"
        _check_delete_undo_redo(window, keys)
    finally:
        _close(window)
