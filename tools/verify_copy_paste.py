#!/usr/bin/env python3
"""Verifies v2.7's copy/paste for edit tools (see git log for the full spec)
by driving the real ViewerWindow headlessly (QT_QPA_PLATFORM=offscreen), the
same technique tools/verify_iso_viewer_pick.py already uses to exercise
wiring a Qt-free script can't reach (toolbar-driven tool switches, real
QAction enabled-state gating, edit_history integration).

`examples/` is gitignored and absent from a fresh checkout or git worktree
(confirmed while writing this script, from inside one) -- unlike
verify_iso_viewer_pick.py, which defaults its scenario_dir argument to
`examples/`, this script requires the directory to be passed explicitly so
it still runs from a worktree that has no examples/ of its own; point it at
another checkout's examples/ directory (e.g.
~/source/DEscape/examples) if the current one doesn't have it.

Checks, per real example file with terrain_write_supported (files without it
are reported as skipped, not failed -- Edit mode's tools are disabled for
them entirely, so there is nothing to copy/paste):

1. Copy/Paste disabled while Pan is active, and while no map is loaded --
   never enabled for the one tool (Pan) with no per-tile data to copy.
2. Terrain copy/paste round trip -- paint a known terrain_id onto a source
   tile through the real Terrain tool stroke path, copy it, give the
   destination tile a stale non--1 `layer` (simulating a leftover
   double-terrain blend), paste, and confirm the destination's terrain_id
   matches the source AND `layer` was reset to -1 -- the layer-reset
   decision documented in ViewerWindow.paste_tile()'s mutate_fn, matching
   every other terrain-write path in this tool.
3. Elevation copy/paste round trip, *with real neighbor propagation* -- sets
   a source tile's elevation far enough from a destination tile's neighbor's
   elevation (a difference > 1) that MapManager._elevation_tile_recursion's
   propagation rule (confirmed directly against AoE2ScenarioParser's source:
   `abs(other.elevation - source_tile.elevation) > 1` triggers adjusting
   `other`) is *forced* to touch that neighbor too when the paste lands.
   Checking only the destination tile's own elevation would pass even for a
   buggy raw `tile.elevation =` write, which this must not regress to --
   this additionally asserts the neighbor changed and
   that the "Paste" EditHistory record's own diff touched more than one
   tile, which a raw assignment could never produce.
4. Paste is undoable -- for both the terrain and elevation pastes above,
   undo() restores the exact pre-paste tile states (destination tile, and
   for elevation, the propagated neighbor too).
5. Type-mismatch behavior -- the decision documented in
   ViewerWindow._update_tool_enabled() (disable Paste outright on a kind
   mismatch, rather than letting the clipboard's kind override the active
   tool): copy while on Terrain, switch to Elevate, confirm paste_action is
   disabled and that calling paste_tile() directly (bypassing the QAction)
   is still a defensive no-op; switch back to Terrain, confirm paste_action
   is enabled again.
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
        if mm.map_width < 12 or mm.map_height < 12:
            return None, "skipped (map too small for this script's fixed tile offsets)"

        problems: list[str] = []
        window.mode_combo.setCurrentText("Edit")

        # -- 1. Pan: never enabled, regardless of clipboard state --
        window._on_tool_selected("pan")
        if window.copy_action.isEnabled():
            problems.append("copy_action enabled while Pan is active")
        if window.paste_action.isEnabled():
            problems.append("paste_action enabled while Pan is active")

        # -- 2. Terrain copy/paste round trip --
        window._on_tool_selected("terrain")
        if not window.terrain_action.isEnabled():
            return None, "skipped (Terrain tool disabled for this file)"

        src_x, src_y = 0, 0
        dst_x, dst_y = 3, 0
        new_terrain_id = _pick_different_terrain(mm.get_tile(src_x, src_y).terrain_id)
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(new_terrain_id))
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(src_x, src_y, 0)
        window.on_edit_stroke_end()
        if mm.get_tile(src_x, src_y).terrain_id != new_terrain_id:
            problems.append("test setup failed: source tile didn't take the painted terrain_id")

        # Simulate a stale double-terrain blend on the destination tile --
        # paste must reset this to -1, not preserve it (decision #1). The
        # pre-paste snapshot used for the undo check below is taken *after*
        # this simulated mutation, matching what EditHistory.apply()'s own
        # begin_stroke() actually snapshots (current tile state at the
        # moment paste_tile() is called) -- not whatever the tile looked
        # like before this test staged the stale layer.
        dst_tile = mm.get_tile(dst_x, dst_y)
        dst_tile.layer = 7 if dst_tile.layer != 7 else 6
        pre_paste_terrain, pre_paste_layer = dst_tile.terrain_id, dst_tile.layer

        window.on_hover((src_x, src_y))
        if not window.copy_action.isEnabled():
            problems.append("copy_action disabled on Terrain tool with a map loaded")
        window.copy_tile()
        if window._clipboard != {"kind": "terrain", "terrain_id": new_terrain_id, "layer": -1}:
            problems.append(f"copy_tile() captured unexpected clipboard: {window._clipboard}")

        window.on_hover((dst_x, dst_y))
        if not window.paste_action.isEnabled():
            problems.append("paste_action disabled right after a matching-kind copy")
        records_before_paste = len(window.edit_history.records)
        window.paste_tile()

        got = mm.get_tile(dst_x, dst_y)
        if got.terrain_id != new_terrain_id:
            problems.append(f"terrain paste: expected terrain_id={new_terrain_id}, got {got.terrain_id}")
        if got.layer != -1:
            problems.append(f"terrain paste: expected layer reset to -1, got {got.layer}")
        if len(window.edit_history.records) != records_before_paste + 1:
            problems.append("terrain paste didn't push exactly one new EditHistory record")

        # -- 4a. Undo the terrain paste --
        window.undo()
        got = mm.get_tile(dst_x, dst_y)
        if (got.terrain_id, got.layer) != (pre_paste_terrain, pre_paste_layer):
            problems.append(
                f"undo after terrain paste: expected (terrain_id={pre_paste_terrain}, layer={pre_paste_layer}), "
                f"got (terrain_id={got.terrain_id}, layer={got.layer})"
            )
        window.redo()  # back to the pasted state, so later checks see a consistent history

        # -- 5. Type mismatch: clipboard is "terrain", switch to Elevate --
        window._on_tool_selected("elevation")
        if window.paste_action.isEnabled():
            problems.append("paste_action enabled with a terrain-kind clipboard while Elevate is active")
        records_before_mismatched_paste = len(window.edit_history.records)
        window.on_hover((dst_x, dst_y))
        window.paste_tile()  # direct call, bypassing the disabled QAction -- must still no-op
        if len(window.edit_history.records) != records_before_mismatched_paste:
            problems.append("paste_tile() mutated state despite a clipboard/tool kind mismatch")
        window._on_tool_selected("terrain")
        if not window.paste_action.isEnabled():
            problems.append("paste_action didn't re-enable after switching back to the matching tool")

        # -- 3. Elevation copy/paste, forcing real neighbor propagation --
        if not scenario.map_is_square:
            elevation_detail = "elevation checks skipped (non-square map)"
        else:
            elevation_detail = _check_elevation_paste(window, mm, problems)

        if problems:
            return False, "; ".join(problems)
        return True, f"OK (terrain + type-mismatch + {elevation_detail})"
    finally:
        # Same as verify_iso_viewer_pick.py's own finally block: mark_saved()
        # before close() so closeEvent()'s _confirm_discard_changes() doesn't
        # pop a modal QMessageBox with nothing able to click it offscreen --
        # every check above leaves edit_history dirty on purpose.
        window.edit_history.mark_saved()
        window.close()


def _check_elevation_paste(window: ViewerWindow, mm, problems: list[str]) -> str:
    """Elevation copy/paste with a forced neighbor-propagation check --
    factored out of check_file() only for readability, not reused
    elsewhere. Appends to `problems` in place; returns a short OK detail
    string for the caller's summary line."""
    src_x, src_y = 0, 1  # distinct from the terrain check's (0, 0)/(3, 0) tiles
    dst_x, dst_y = mm.map_width // 2, mm.map_height // 2
    neighbor_x, neighbor_y = dst_x, dst_y - 1  # dst is never at y=0 (map >= 12 tall)

    neighbor_before = mm.get_tile(neighbor_x, neighbor_y).elevation
    # Any value guaranteed to differ from neighbor_before by more than 1 --
    # MapManager._elevation_tile_recursion only adjusts a neighbor when
    # abs(other.elevation - source.elevation) > 1 (confirmed against
    # AoE2ScenarioParser's source; see this script's module docstring).
    target_value = 7 if neighbor_before <= 4 else 0

    window._on_tool_selected("set_level")
    if not window.set_level_action.isEnabled():
        return "elevation checks skipped (Set Elevation tool disabled for this file)"
    window.elevation_level_spin.setValue(target_value)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(src_x, src_y, 0)
    window.on_edit_stroke_end()
    if mm.get_tile(src_x, src_y).elevation != target_value:
        problems.append("test setup failed: source tile didn't take the set elevation value")

    dst_before = mm.get_tile(dst_x, dst_y).elevation
    pre_paste_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation

    window._on_tool_selected("elevation")  # Elevate and Set Elevation share the "elevation" clipboard kind
    window.on_hover((src_x, src_y))
    window.copy_tile()
    if window._clipboard != {"kind": "elevation", "value": target_value}:
        problems.append(f"copy_tile() captured unexpected clipboard: {window._clipboard}")

    window.on_hover((dst_x, dst_y))
    if not window.paste_action.isEnabled():
        problems.append("paste_action disabled for a matching elevation-kind clipboard")
    window.paste_tile()

    got_dst = mm.get_tile(dst_x, dst_y).elevation
    got_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation
    if got_dst != target_value:
        problems.append(f"elevation paste: expected destination elevation={target_value}, got {got_dst}")
    if got_neighbor == pre_paste_neighbor:
        problems.append(
            "elevation paste: neighbor tile's elevation didn't change at all -- paste is not going "
            "through set_tile_elevation's propagation (or the forced gap wasn't actually forced)"
        )
    last_record = window.edit_history.records[-1] if window.edit_history.records else None
    if last_record is None or last_record.label != "Paste" or len(last_record.touched_indices()) < 2:
        problems.append(
            f"elevation paste's EditHistory record didn't touch multiple tiles "
            f"(propagation should have): {last_record}"
        )

    # -- 4b. Undo the elevation paste: destination AND propagated neighbor
    # must both revert.
    window.undo()
    got_dst = mm.get_tile(dst_x, dst_y).elevation
    got_neighbor = mm.get_tile(neighbor_x, neighbor_y).elevation
    if got_dst != dst_before or got_neighbor != pre_paste_neighbor:
        problems.append(
            f"undo after elevation paste: expected (dst={dst_before}, neighbor={pre_paste_neighbor}), "
            f"got (dst={got_dst}, neighbor={got_neighbor})"
        )

    return "elevation round-trip with forced neighbor propagation + undo"


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
