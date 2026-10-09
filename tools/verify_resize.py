#!/usr/bin/env python3
"""Verifies File > Resize Map… (TASK-107, GH #45) by driving the real
ViewerWindow headlessly (QT_QPA_PLATFORM=offscreen), the same technique
tools/verify_copy_paste.py uses, so the action, the document replacement and
Save As are exercised rather than only scenario_resize's pure functions.

Like verify_copy_paste.py this script requires the scenario directory to be
passed explicitly, so it runs from a worktree without its own examples/.

Checks, per square example file whose resize the plan allows (others are
reported as skipped, with the plan's own refusal):

1. A BOTTOM_RIGHT shrink by 24 tiles through ViewerWindow._apply_resize():
   the document is the new size, dirty, with no undo step, and holds the
   plan's unit count.
2. Save As writes it, and a fresh load reads back the new size, the same
   unit count, every survivor at its old position minus 24 on both axes,
   and every kept terrain struct equal to the old one at (x + 24, y + 24).
3. When triggers were remapped, every coordinate group in the saved file is
   its old value minus 24, clamped to the map.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from descape import library_compat, trigger_geometry
from descape.scenario_io import TERRAIN_STRUCT_SIZE, load_map_and_units, parse_triggers
from descape.scenario_resize import TRIGGERS_REMAPPED, Anchor, resize_plan
from descape.viewer import ViewerWindow
from testkit.settings_isolation import isolate_settings

SHRINK = 24


def _shapes(loaded) -> list:
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    return [
        s
        for i, t in enumerate(manager.triggers)
        for s in trigger_geometry.shapes_for_trigger(t, vocabulary, trigger_index=i)
    ]


def _block(loaded) -> bytes:
    mm = loaded.map_manager
    off = loaded.terrain_block_offset
    return loaded.decompressed_body[off : off + TERRAIN_STRUCT_SIZE * mm.map_width * mm.map_height]


def check_file(path: Path) -> tuple[bool | None, str]:
    source = load_map_and_units(path)
    mm = source.map_manager
    size = mm.map_width - SHRINK
    plan = resize_plan(source, size, size, Anchor.BOTTOM_RIGHT)
    if plan.refusal is not None:
        return None, f"skipped ({plan.refusal})"
    shapes_before = _shapes(source) if plan.triggers == TRIGGERS_REMAPPED else None
    old_block = _block(source)
    before = {}
    for units in source.unit_manager.units:
        for u in units:
            before.setdefault(u.reference_id, []).append((u.x - SHRINK, u.y - SHRINK))

    problems: list[str] = []
    window = ViewerWindow()
    try:
        window.load_scenario(path)
        window._apply_resize(size, size, Anchor.BOTTOM_RIGHT)
        live = window.scenario.map_manager
        if (live.map_width, live.map_height) != (size, size):
            problems.append(f"document is {live.map_width}x{live.map_height}")
        if not window.edit_history.is_dirty or window.undo_action.isEnabled():
            problems.append("resized document should be dirty with nothing to undo")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "resized.aoe2scenario"
            window.scenario.path = dest
            window._untitled = False
            window.save()
            reloaded = load_map_and_units(dest)
            rm = reloaded.map_manager
            if (rm.map_width, rm.map_height) != (size, size):
                problems.append(f"saved file is {rm.map_width}x{rm.map_height}")
            count = sum(len(u) for u in reloaded.unit_manager.units)
            if count != plan.units_moved:
                problems.append(f"{count} units saved, plan said {plan.units_moved}")
            misplaced = [
                u.reference_id
                for units in reloaded.unit_manager.units
                for u in units
                if (u.x, u.y) not in before.get(u.reference_id, ())
            ]
            if misplaced:
                problems.append(f"{len(misplaced)} units not at old position - {SHRINK}")
            new_block = _block(reloaded)
            row = size * TERRAIN_STRUCT_SIZE
            for y in range(size):
                start = ((y + SHRINK) * mm.map_width + SHRINK) * TERRAIN_STRUCT_SIZE
                if new_block[y * row : (y + 1) * row] != old_block[start : start + row]:
                    problems.append(f"terrain row {y} is not the old row {y + SHRINK} cropped")
                    break
            if shapes_before is not None:
                shapes_after = _shapes(reloaded)
                bad = sum(
                    new.coords != tuple(min(max(c - SHRINK, 0), size - 1) for c in old.coords)
                    for old, new in zip(shapes_before, shapes_after, strict=False)
                )
                if bad or len(shapes_after) != len(shapes_before):
                    problems.append(f"{bad} trigger coordinate groups wrong")
    finally:
        window.edit_history.mark_saved()
        window.close()
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"OK ({mm.map_width} to {size}, {plan.units_deleted} deleted, {plan.units_moved} moved, "
        f"triggers {plan.triggers})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, help="Directory of .aoe2scenario files (see module docstring)")
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        print(f"No .aoe2scenario files found in {args.scenario_dir}")
        sys.exit(1)

    # Held in a variable: an unassigned QApplication(...) crashes the first ViewerWindow().
    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841

    failures = 0
    with tempfile.TemporaryDirectory() as config_dir:
        # load_scenario() records a recent file and close() the window size.
        isolate_settings(Path(config_dir))
        for path in files:
            try:
                ok, detail = check_file(path)
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            failures += ok is False
            print(f"{status}  {path.name:38s} {detail}")

    print(f"\n{len(files) - failures}/{len(files)} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
