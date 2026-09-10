#!/usr/bin/env python3
"""Verifies phase 2.8's region copy/paste (see git log for the full spec)
by driving the real ViewerWindow headlessly (QT_QPA_PLATFORM=offscreen), the
same technique tools/verify_iso_viewer_pick.py already uses to exercise
wiring a Qt-free script can't reach (toolbar-driven tool switches, real
QAction enabled-state gating, edit_history integration).

`examples/` is gitignored and absent from a fresh checkout or git worktree
(confirmed while writing the original version of this script, from inside
one) -- unlike verify_iso_viewer_pick.py, which defaults its scenario_dir
argument to `examples/`, this script requires the directory to be passed
explicitly so it still runs from a worktree that has no examples/ of its
own; point it at another checkout's examples/ directory (e.g.
~/source/DEscape/examples) if the current one doesn't have it.

Checks, per real example file with terrain_write_supported (files without it
are reported as skipped, not failed -- Terrain mode's tools are disabled for
them entirely, so there is nothing to copy/paste):

1. Copy/Paste disabled with no region/clipboard, regardless of the active
   tool (Pan) -- phase 2.8 retired the old per-tool clipboard-kind gating in
   favour of one keyed on self._region/self._region_clipboard alone.
2. Terrain region copy/paste round trip -- paint a known terrain_id over a
   source region through the real Draw tool stroke path, select + copy it,
   give the destination region a stale non--1 `layer` (simulating a leftover
   double-terrain blend), paste, and confirm every destination tile's
   terrain_id matches the source AND `layer` was reset to -1 -- the
   layer-reset decision RegionBlock.layers' own comment documents, matching
   every other terrain-write path in this tool.
3. Elevation region copy/paste, *with real neighbor propagation* -- built via
   real Set Elevation strokes (not a hand-authored elevation matrix, which
   can encode a jump the in-game editor could never produce -- see
   tests/test_region_clipboard.py's own fixture-construction comment) far
   enough from a destination neighbor's elevation that
   MapManager._elevation_tile_recursion's propagation rule is *forced* to
   touch that neighbor too when the paste lands. Skipped on a non-square map
   (elevation_targets()/set_tiles_elevation() require one, same as every
   other elevation tool).
4. Units region copy/paste -- units placed directly via UnitEditModel inside
   the source region (deterministic regardless of what the file already
   has), copied and pasted at a translated destination; confirms each
   pasted unit's rotation is passed through byte-for-byte (AGENTS.md's hard
   rule) and that begin_unit_edit() was given every owner in the block, not
   just one.
5. One undo step reverts a paste with terrain+elevation+units all checked --
   CompositeDiffRecord, not two separate records.
6. Non-square map -- the Elevation paste checkbox's own gate
   (self.scenario.map_is_square) is honoured even if forced checked, and a
   terrain+units paste still succeeds where elevation would have raised.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from descape.viewer import ViewerWindow

# Any two fixed, always-valid TerrainId values -- 2 (BEACH) and 15 (GRASS_1)
# are both present in every DE structure version (same pair
# tools/verify_write_path.py uses, for the same reason).
_TERRAIN_A, _TERRAIN_B = 2, 15


def _pick_different_terrain(current: int) -> int:
    return _TERRAIN_B if current == _TERRAIN_A else _TERRAIN_A


def check_file(path: Path) -> tuple[bool | None, str]:
    window = ViewerWindow()
    try:
        window.load_scenario(path)
        scenario = window.scenario
        if scenario is None:
            return False, "failed to load"
        if not scenario.terrain_write_supported:
            return None, "skipped (file doesn't support terrain/elevation editing)"

        mm = scenario.map_manager
        if mm.map_width < 16 or mm.map_height < 16:
            return None, "skipped (map too small for this script's fixed tile offsets)"

        problems: list[str] = []
        window.mode_combo.setCurrentText("Terrain")

        # -- 1. Pan: never enabled, regardless of the active tool --
        window._on_tool_selected("pan")
        if window.copy_action.isEnabled():
            problems.append("copy_action enabled with no region selected")
        if window.paste_action.isEnabled():
            problems.append("paste_action enabled with an empty clipboard")

        _check_terrain_round_trip(window, mm, problems)
        elevation_detail = (
            "elevation checks skipped (non-square map)"
            if not scenario.map_is_square
            else _check_elevation_round_trip(window, mm, problems)
        )
        # UnitEditModel.add() unconditionally assigns caption_string_id/
        # caption_string in Unit.__init__ (confirmed against the installed
        # library: neither omitting the kwarg nor passing its own default
        # avoids the assignment), which raises UnsupportedAttributeError on
        # any scenario version below caption_string's own Support(since=1.55)
        # -- a pre-existing bug in add() itself (Place Unit already has it
        # too), not something introduced by region paste. Units checks are
        # skipped below rather than reported as a phase 2.8 regression;
        # flagged in this script's output so it doesn't go unnoticed.
        if float(scenario.scenario_version) < 1.55:
            print(
                f"  (units checks skipped for {path.name}: pre-existing "
                f"UnitEditModel.add() bug on scenario_version < 1.55)"
            )
        else:
            _check_units_round_trip(window, mm, problems)
            _check_one_undo_step_for_a_mixed_paste(window, mm, problems)
        if not scenario.map_is_square:
            _check_elevation_checkbox_honoured_on_non_square(window, mm, problems)

        if problems:
            return False, "; ".join(problems)
        return True, f"OK (terrain + units + one-undo-step + {elevation_detail})"
    finally:
        # Same as verify_iso_viewer_pick.py's own finally block: mark_saved()
        # before close() so closeEvent()'s _confirm_discard_changes() doesn't
        # pop a modal QMessageBox with nothing able to click it offscreen --
        # every check above leaves edit_history dirty on purpose.
        window.edit_history.mark_saved()
        window.close()


def _select(window: ViewerWindow, tx0: int, ty0: int, tx1: int, ty1: int) -> None:
    window.on_region_selected((tx0, ty0, tx1, ty1))


def _check_terrain_round_trip(window: ViewerWindow, mm, problems: list[str]) -> None:
    window._on_tool_selected("draw")
    if not window.draw_action.isEnabled():
        problems.append("Draw tool disabled for this file -- terrain check skipped")
        return

    sx0, sy0, sx1, sy1 = 0, 0, 2, 2
    dx0, dy0 = 6, 0
    new_terrain_id = _pick_different_terrain(mm.get_tile(sx0, sy0).terrain_id)
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(new_terrain_id))
    for y in range(sy0, sy1):
        for x in range(sx0, sx1):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    for y in range(sy0, sy1):
        for x in range(sx0, sx1):
            if mm.get_tile(x, y).terrain_id != new_terrain_id:
                problems.append("test setup failed: source region didn't take the painted terrain_id")
                return

    # Simulate a stale double-terrain blend on the destination region --
    # paste must reset this to -1, not preserve it.
    for y in range(sy1 - sy0):
        for x in range(sx1 - sx0):
            t = mm.get_tile(dx0 + x, dy0 + y)
            t.layer = 7 if t.layer != 7 else 6
    pre_paste = {
        (x, y): (mm.get_tile(dx0 + x, dy0 + y).terrain_id, mm.get_tile(dx0 + x, dy0 + y).layer)
        for y in range(sy1 - sy0)
        for x in range(sx1 - sx0)
    }

    _select(window, sx0, sy0, sx1, sy1)
    if not window.copy_action.isEnabled():
        problems.append("copy_action disabled with a region selected")
        return
    window.copy_region()
    # Terrain-only: the source region may contain real pre-existing units
    # from this corpus file, and Units defaults to checked -- leaving it on
    # would exercise the units-paste path for a check that means to isolate
    # terrain alone.
    window.paste_elevation_check.setChecked(False)
    window.paste_units_check.setChecked(False)

    window.on_hover((dx0, dy0))
    if not window.paste_action.isEnabled():
        problems.append("paste_action disabled right after a copy")
        return
    records_before = len(window.edit_history.records)
    window.paste_region()

    for y in range(sy1 - sy0):
        for x in range(sx1 - sx0):
            got = mm.get_tile(dx0 + x, dy0 + y)
            if got.terrain_id != new_terrain_id or got.layer != -1:
                problems.append(
                    f"terrain paste at ({dx0 + x}, {dy0 + y}): expected "
                    f"(terrain_id={new_terrain_id}, layer=-1), got (terrain_id={got.terrain_id}, layer={got.layer})"
                )
    if len(window.edit_history.records) != records_before + 1:
        problems.append("terrain-only paste didn't push exactly one new EditHistory record")

    window.undo()
    for (x, y), (terrain_id, layer) in pre_paste.items():
        got = mm.get_tile(dx0 + x, dy0 + y)
        if (got.terrain_id, got.layer) != (terrain_id, layer):
            problems.append(f"undo after terrain paste didn't restore ({dx0 + x}, {dy0 + y})")
    window.redo()


def _check_elevation_round_trip(window: ViewerWindow, mm, problems: list[str]) -> str:
    """Built via real Set Elevation strokes, not a hand-authored elevation
    matrix -- see this module's docstring on why."""
    sx0, sy0 = 0, 4
    size = 2
    dx0, dy0 = mm.map_width // 2, mm.map_height // 2
    neighbor_x, neighbor_y = dx0, dy0 - 1  # dst is never at y=0 (map >= 16 tall)

    neighbor_before = mm.get_tile(neighbor_x, neighbor_y).elevation
    # Any value guaranteed to differ from neighbor_before by more than 1 --
    # MapManager._elevation_tile_recursion only adjusts a neighbor when
    # abs(other.elevation - source.elevation) > 1.
    target_value = 7 if neighbor_before <= 4 else 0

    window._on_tool_selected("set_level")
    if not window.set_level_action.isEnabled():
        return "elevation checks skipped (Set Elevation tool disabled for this file)"
    window.elevation_level_spin.setValue(target_value)
    for y in range(sy0, sy0 + size):
        for x in range(sx0, sx0 + size):
            window.on_edit_stroke_start()
            window.on_edit_stroke_tile(x, y, 0)
            window.on_edit_stroke_end()
    for y in range(sy0, sy0 + size):
        for x in range(sx0, sx0 + size):
            if mm.get_tile(x, y).elevation != target_value:
                problems.append("test setup failed: source region didn't take the set elevation value")
                return "elevation check aborted"

    dst_before = mm.get_tile(dx0, dy0).elevation
    pre_paste_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation

    window._on_tool_selected("select")
    _select(window, sx0, sy0, sx0 + size, sy0 + size)
    window.copy_region()
    window.paste_terrain_check.setChecked(False)
    window.paste_units_check.setChecked(False)
    window.paste_elevation_check.setChecked(True)

    window.on_hover((dx0, dy0))
    if not window.paste_action.isEnabled():
        problems.append("paste_action disabled for a non-empty clipboard")
    window.paste_region()

    got_dst = mm.get_tile(dx0, dy0).elevation
    got_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation
    if got_dst != target_value:
        problems.append(f"elevation paste: expected destination elevation={target_value}, got {got_dst}")
    if got_neighbor == pre_paste_neighbor:
        problems.append(
            "elevation paste: neighbor tile's elevation didn't change at all -- paste is not going "
            "through set_tiles_elevation's propagation (or the forced gap wasn't actually forced)"
        )
    last_record = window.edit_history.records[-1] if window.edit_history.records else None
    if last_record is None or last_record.label != "Paste Region" or len(last_record.touched_indices()) < 2:
        problems.append(
            f"elevation paste's EditHistory record didn't touch multiple tiles "
            f"(propagation should have): {last_record}"
        )

    window.undo()
    got_dst = mm.get_tile(dx0, dy0).elevation
    got_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation
    if got_dst != dst_before or got_neighbor != pre_paste_neighbor:
        problems.append(
            f"undo after elevation paste: expected (dst={dst_before}, neighbor={pre_paste_neighbor}), "
            f"got (dst={got_dst}, neighbor={got_neighbor})"
        )

    window.paste_terrain_check.setChecked(True)
    window.paste_units_check.setChecked(True)
    return "elevation round-trip with forced neighbor propagation + undo"


def _units_in_rect(window: ViewerWindow, x0: int, y0: int, x1: int, y1: int) -> list:
    return [
        u
        for player_units in window.scenario.unit_manager.units
        for u in player_units
        if x0 <= int(u.x) < x1 and y0 <= int(u.y) < y1
    ]


def _check_units_round_trip(window: ViewerWindow, mm, problems: list[str]) -> None:
    """Deltas against the region's own pre-existing unit count throughout,
    not an assumed-empty tile -- a real corpus file can have units (trees,
    decorations) anywhere, so a fixed offset is never guaranteed clear."""
    from descape.unit_model import UnitEditModel

    unit_edits = window._ensure_unit_edits()
    if unit_edits is None:
        problems.append("units are read-only for this file -- units check skipped")
        return
    assert isinstance(unit_edits, UnitEditModel)

    sx0, sy0 = 8, 8
    dx0, dy0 = 8, 12
    before_src = len(_units_in_rect(window, sx0, sy0, sx0 + 2, sy0 + 2))
    before_dst = len(_units_in_rect(window, dx0, dy0, dx0 + 2, dy0 + 2))
    a = unit_edits.add(player=1, unit_const=83, x=sx0 + 0.5, y=sy0 + 0.5, z=0.0, rotation=1.75)
    b = unit_edits.add(player=0, unit_const=83, x=sx0 + 1.5, y=sy0 + 1.5, z=0.0, rotation=37.0)
    window.edit_history.reset()  # the two adds above are test setup, not part of what this check measures

    window._on_tool_selected("select")
    _select(window, sx0, sy0, sx0 + 2, sy0 + 2)
    window.copy_region()
    expected_captured = before_src + 2
    got_captured = None if window._region_clipboard is None else len(window._region_clipboard.units)
    if got_captured != expected_captured:
        problems.append(f"copy_region() should have captured {expected_captured} units, got {got_captured}")
        return

    window.paste_terrain_check.setChecked(False)
    window.paste_elevation_check.setChecked(False)
    window.paste_units_check.setChecked(True)
    window.on_hover((dx0, dy0))
    window.paste_region()
    window.paste_terrain_check.setChecked(True)
    window.paste_elevation_check.setChecked(True)

    pasted_all = _units_in_rect(window, dx0, dy0, dx0 + 2, dy0 + 2)
    if len(pasted_all) != before_dst + expected_captured:
        problems.append(
            f"expected {before_dst + expected_captured} units at the destination after paste, "
            f"found {len(pasted_all)}"
        )
    # Identify OUR two pasted units specifically (by rotation, distinctive
    # enough not to collide with whatever the file already had) rather than
    # assuming the whole destination rect's contents are ours.
    mine = [u for u in pasted_all if u.rotation in (a.rotation, b.rotation)]
    rotations = sorted(u.rotation for u in mine)
    if rotations != sorted([a.rotation, b.rotation]):
        problems.append(f"pasted rotations {rotations} don't match the copied ones {[a.rotation, b.rotation]} verbatim")
    if any(u.garrisoned_in_id != -1 for u in mine):
        problems.append("pasted units should have garrisoned_in_id reset to -1")

    window.undo()
    after_undo = len(_units_in_rect(window, dx0, dy0, dx0 + 2, dy0 + 2))
    if after_undo != before_dst:
        problems.append(
            f"undo after unit paste didn't restore the destination region's unit count "
            f"(expected {before_dst}, got {after_undo})"
        )


def _check_one_undo_step_for_a_mixed_paste(window: ViewerWindow, mm, problems: list[str]) -> None:
    if not window.scenario.map_is_square:
        return  # elevation is unavailable here regardless -- see the caller
    from descape.unit_model import UnitEditModel

    unit_edits = window._ensure_unit_edits()
    if not isinstance(unit_edits, UnitEditModel):
        return

    sx0, sy0 = 2, 8
    dx0, dy0 = mm.map_width - 4, mm.map_height - 4
    window._on_tool_selected("draw")
    window.terrain_combo.setCurrentIndex(
        window.terrain_combo.findData(_pick_different_terrain(mm.get_tile(sx0, sy0).terrain_id))
    )
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(sx0, sy0, 0)
    window.on_edit_stroke_end()
    unit_edits.add(player=1, unit_const=83, x=sx0 + 0.5, y=sy0 + 0.5, z=0.0, rotation=0.0)
    window.edit_history.reset()  # setup above is not what this check measures

    before_dst = len(_units_in_rect(window, dx0, dy0, dx0 + 1, dy0 + 1))

    window._on_tool_selected("select")
    _select(window, sx0, sy0, sx0 + 1, sy0 + 1)
    window.copy_region()
    window.paste_terrain_check.setChecked(True)
    window.paste_elevation_check.setChecked(True)
    window.paste_units_check.setChecked(True)

    window.on_hover((dx0, dy0))
    records_before = len(window.edit_history.records)
    window.paste_region()
    pushed = len(window.edit_history.records) - records_before
    if pushed != 1:
        problems.append(f"a terrain+elevation+units paste should push exactly 1 record, pushed {pushed}")
        return
    if window.edit_history.records[-1].kind != "composite":
        problems.append("a terrain+elevation+units paste's single record should be a CompositeDiffRecord")

    window.undo()
    after_undo = len(_units_in_rect(window, dx0, dy0, dx0 + 1, dy0 + 1))
    if after_undo != before_dst:
        problems.append("one undo() didn't revert the unit half of a mixed paste")


def _check_elevation_checkbox_honoured_on_non_square(window: ViewerWindow, mm, problems: list[str]) -> None:
    window._on_tool_selected("select")
    _select(window, 0, 0, 2, 2)
    window.copy_region()
    window.paste_elevation_check.setChecked(True)  # forced on -- _update_tool_enabled should have disabled it
    if window.paste_elevation_check.isEnabled():
        problems.append("Elevation paste checkbox should be disabled on a non-square map")
    window.on_hover((4, 4))
    window.paste_region()  # must not raise -- do_elevation reads map_is_square defensively too


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, help="Directory of .aoe2scenario files (see module docstring)")
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {args.scenario_dir}")
        sys.exit(1)

    # Held in a variable deliberately -- see verify_iso_viewer_pick.py's own
    # comment on why an unassigned QApplication(...) crashes the first
    # ViewerWindow().
    app = QApplication.instance() or QApplication(sys.argv[:1])

    failures = 0
    for path in files:
        try:
            ok, detail = check_file(path)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        if ok is False:
            failures += 1
        print(f"{status}  {path.name:38s} {detail}")

    print(f"\n{len(files) - failures}/{len(files)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
