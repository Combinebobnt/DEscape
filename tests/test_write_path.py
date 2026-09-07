"""Migrated from tools/verify_write_path.py (Phase 3 of the pytest migration
plan -- Ordering step 5, script #2).

Verifies v2's write path:

1. Zero-edit round trip -- load, write with no changes, and confirm the written
   file's decompressed body is byte-identical to the original. This is the core
   claim behind v2's design: patching the terrain struct array
   in place and recompressing reproduces the original file exactly when nothing
   changed, unlike re-serializing through AoE2ScenarioParser's own commit()/
   write_to_file() (which was tested directly against these same files and
   found to fail this exact check on 10 of 12).

2. Edit-then-reload -- paint one tile's terrain, raise one tile's elevation,
   and set another tile to an absolute elevation level; write; reload through
   the real load path (scenario_io.load_map_and_units, not just a byte
   comparison); confirm the edits are exactly what's now on disk.

3. edit_history.EditHistory invariants -- undo back to the load state matches
   a fresh independent load of the same file exactly; redo forward matches the
   post-edit state; a genuine no-op edit pushes no record; applying a new edit
   after an undo() truncates the discarded redo tail. Complements
   tests/test_edit_history.py's fake-tile unit tests with the same invariants
   exercised against a real, loaded scenario's real TerrainTiles.

See tests/test_scenario_io.py for the v1 (read-only) loader-shim safety net
-- kept separate since this exercises different, newer code with a different
failure mode.

Same fixture+corpus dual-tier split as tests/test_scenario_io.py: each check
runs once against the shipped blank template
descape/templates/blank_120x120.aoe2scenario (default tier -- also File > New's
source, see descape/viewer.py) and once parametrized against the real
examples/ corpus (corpus tier, preserving the original script's exact
behavior).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import iso_geometry
from descape.edit_history import EditHistory, tile_state
from descape.elevation_tools import set_tile_elevation
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario


def _pick_tile_indices(mm) -> tuple[int, int, int]:
    # TILE_A is always 0 (top-left corner). TILE_B and TILE_C used to be the
    # fixed indices 1 and 2 -- directly adjacent tiles in the row-major
    # terrain array -- which meant elevating one could pull the other along
    # via set_tile_elevation's neighbor-slope propagation (its own docstring
    # warns against this: "Do NOT build this by calling set_tile_elevation()
    # once per target"). Quarter- and three-quarter-map placement keeps B and
    # C far enough apart that no propagation cone (bounded by the elevation
    # delta, <= MAX_ELEVATION) can reach between them, even on the smallest
    # 120x120 fixture.
    w, h = mm.map_width, mm.map_height
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    return 0, y1 * w + x1, y2 * w + x2


def _pick_different_terrain(current: int) -> int:
    # Any two fixed, always-valid TerrainId values work here -- 2 (BEACH) and
    # 15 (GRASS_1) are both present in every DE structure version. Picking
    # whichever differs from `current` guarantees the edit is a real change
    # regardless of the tile's original terrain.
    return 15 if current == 2 else 2


def _pick_different_elevation(current: int) -> int:
    # +1 mod (MAX_ELEVATION + 1) always differs from `current` for any
    # current in [0, MAX_ELEVATION]. Import the constant rather than
    # restating it so this can't drift out of sync again.
    return (current + 1) % (iso_geometry.MAX_ELEVATION + 1)


def _check_zero_edit_identity(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    out = tmp_dir / f"{path.stem}.zero_edit{path.suffix}"
    write_scenario(s, out)
    written = out.read_bytes()
    original = path.read_bytes()
    if written == original:
        return True, "OK (byte-identical)"
    if written[: len(s.header_bytes)] != s.header_bytes:
        return False, "header bytes changed"
    body = _decompress_bytes(written[len(s.header_bytes) :])
    if body != s.decompressed_body:
        n = min(len(body), len(s.decompressed_body))
        i = next((k for k in range(n) if body[k] != s.decompressed_body[k]), n)
        return False, f"body diverged at byte {i} (lens {len(body)} vs {len(s.decompressed_body)})"
    n = min(len(written), len(original))
    i = next((k for k in range(n) if written[k] != original[k]), n)
    return False, (
        f"decompressed content matches but raw file bytes diverged at byte {i} "
        f"(lens {len(written)} vs {len(original)})"
    )


def _check_edit_reload(path: Path, tmp_dir: Path) -> tuple[bool, str]:
    s = load_map_and_units(path)
    mm = s.map_manager
    TILE_A, TILE_B, TILE_C = _pick_tile_indices(mm)

    tile_a = mm.terrain[TILE_A]
    new_terrain = _pick_different_terrain(tile_a.terrain_id)
    tile_a.terrain_id = new_terrain
    tile_a.layer = -1

    tile_b = mm.terrain[TILE_B]
    tx, ty = tile_b.xy
    new_elevation_b = _pick_different_elevation(tile_b.elevation)
    set_tile_elevation(mm, tx, ty, new_elevation_b)

    tile_c = mm.terrain[TILE_C]
    tx, ty = tile_c.xy
    new_elevation_c = _pick_different_elevation(tile_c.elevation)
    set_tile_elevation(mm, tx, ty, new_elevation_c)

    out = tmp_dir / f"{path.stem}.edited{path.suffix}"
    write_scenario(s, out)

    reloaded = load_map_and_units(out)
    got_a = reloaded.map_manager.terrain[TILE_A]
    got_b = reloaded.map_manager.terrain[TILE_B]
    got_c = reloaded.map_manager.terrain[TILE_C]

    problems = []
    if got_a.terrain_id != new_terrain or got_a.layer != -1:
        problems.append(f"tile A: expected terrain={new_terrain},layer=-1 got {tile_state(got_a)}")
    if got_b.elevation != new_elevation_b:
        problems.append(f"tile B: expected elevation={new_elevation_b} got {tile_state(got_b)}")
    if got_c.elevation != new_elevation_c:
        problems.append(f"tile C: expected elevation={new_elevation_c} got {tile_state(got_c)}")

    if problems:
        return False, "; ".join(problems)
    return True, "OK (edits round-tripped through a real reload)"


def _check_history_invariants(path: Path) -> tuple[bool, str]:
    original = load_map_and_units(path)
    original_states = [tile_state(t) for t in original.map_manager.terrain]

    s = load_map_and_units(path)
    mm = s.map_manager
    TILE_A, TILE_B, TILE_C = _pick_tile_indices(mm)
    tiles = mm.terrain
    hist = EditHistory()

    # -- op1: terrain paint on TILE_A --
    new_terrain = _pick_different_terrain(tiles[TILE_A].terrain_id)

    def paint():
        tiles[TILE_A].terrain_id = new_terrain
        tiles[TILE_A].layer = -1

    d1 = hist.apply("paint", tiles, paint)
    if not d1:
        return False, "op1 (a real terrain change) produced an empty diff"

    # -- op2: elevation raise on TILE_B, possibly touching neighbors too --
    tx, ty = tiles[TILE_B].xy
    new_elevation = _pick_different_elevation(tiles[TILE_B].elevation)
    d2 = hist.apply("raise", tiles, lambda: set_tile_elevation(mm, tx, ty, new_elevation))
    if not d2:
        return False, "op2 (a real elevation change) produced an empty diff"

    post_edit_states = [tile_state(t) for t in tiles]

    # -- genuine no-op: repaint TILE_A with the terrain it already has --
    def noop_paint():
        tiles[TILE_A].terrain_id = tiles[TILE_A].terrain_id

    d_noop = hist.apply("noop", tiles, noop_paint)
    if d_noop:
        return False, f"a genuine no-op edit pushed a non-empty diff: {d_noop}"
    if len(hist.records) != 2:
        return False, f"a no-op edit changed the record count to {len(hist.records)}, expected 2"

    # -- undo both real ops; must match the independently-loaded original --
    hist.undo(tiles)
    hist.undo(tiles)
    if not hist.can_undo and hist.can_redo:
        pass  # expected: fully undone, both ops redoable
    else:
        return False, f"unexpected can_undo/can_redo after undoing both ops: {hist.can_undo}/{hist.can_redo}"
    undone_states = [tile_state(t) for t in tiles]
    if undone_states != original_states:
        n = sum(1 for a, b in zip(undone_states, original_states) if a != b)
        return False, f"undo-to-start didn't match a fresh load: {n} tiles differ"

    # -- redo both; must match the post-edit snapshot taken above --
    hist.redo(tiles)
    hist.redo(tiles)
    if hist.can_redo:
        return False, "still redoable after redoing every record"
    redone_states = [tile_state(t) for t in tiles]
    if redone_states != post_edit_states:
        n = sum(1 for a, b in zip(redone_states, post_edit_states) if a != b)
        return False, f"redo-to-end didn't match the post-edit state: {n} tiles differ"

    # -- redo-tail truncation: undo once, apply a new op, old redo must be gone --
    hist.undo(tiles)  # back to just op1
    if len(hist.records) != 2 or hist.cursor != 1:
        return False, f"unexpected history state before truncation check: cursor={hist.cursor}, records={len(hist.records)}"
    tx, ty = tiles[TILE_C].xy
    other_elevation = _pick_different_elevation(tiles[TILE_C].elevation)
    d3 = hist.apply("set_level", tiles, lambda: set_tile_elevation(mm, tx, ty, other_elevation))
    if not d3:
        return False, "the truncation check's own edit was unexpectedly a no-op"
    if len(hist.records) != 2 or hist.records[-1].label != "set_level":
        return False, (
            f"applying after an undo() didn't truncate the redo tail: "
            f"records={[r.label for r in hist.records]}"
        )
    if hist.can_redo:
        return False, "redo still available immediately after a fresh apply()"

    return True, "OK (undo/redo/no-op/truncation all correct)"


def test_zero_edit_identity(tmp_path: Path) -> None:
    ok, detail = _check_zero_edit_identity(FIXTURE_PATH, tmp_path)
    assert ok, detail


@pytest.mark.corpus
def test_zero_edit_identity_corpus(scenario_path, tmp_path: Path) -> None:
    ok, detail = _check_zero_edit_identity(scenario_path, tmp_path)
    assert ok, detail


def test_edit_reload(tmp_path: Path) -> None:
    ok, detail = _check_edit_reload(FIXTURE_PATH, tmp_path)
    assert ok, detail


@pytest.mark.corpus
def test_edit_reload_corpus(scenario_path, tmp_path: Path) -> None:
    ok, detail = _check_edit_reload(scenario_path, tmp_path)
    assert ok, detail


def test_history_invariants() -> None:
    ok, detail = _check_history_invariants(FIXTURE_PATH)
    assert ok, detail


@pytest.mark.corpus
def test_history_invariants_corpus(scenario_path) -> None:
    ok, detail = _check_history_invariants(scenario_path)
    assert ok, detail
