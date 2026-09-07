#!/usr/bin/env python3
"""Measures the Tier B `rotation_is_variant` widening across examples/,
install-free for the corpus walk itself (scenario_io.load_map_and_units only)
but install-gated for the frame-count comparisons, the same split
scan_wall_rotation.py uses.

Only consts that BECOME variant under Tier B (new rotation_is_variant() true,
old hand-kept frozensets false) are in scope -- the wall/cliff consts already
variant under Tier A are unaffected by this widening and are not this script's
job to re-measure.

Per such const, reports: placement count, literal/radian split (against the
real .sld frame count when an install is configured, else angle_count), how
many placements actually change their resolved frame old-vs-new, and how many
store a literal integer past their file's real frame count (the
variant_index() wrap this same plan added).

Old resolved frame mirrors the shipped-before-this-plan code exactly: GAIA
(player 0) zeroed at these consts (they were not in the old variant set), and
every other placement went through angle_index() on its verbatim rotation --
neither GAIA-zeroing nor variant_index() applied to a const outside the old
set. New resolved frame is today's `unit_sprites._frame_for()`, unchanged by
this script.

Run any time the Tier B discriminator (unit_graphic_map.json's
`rotation_is_variant` field, or the trebuchet override) changes.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import render, unit_sprites
from descape.scenario_io import load_map_and_units

_OLD_VARIANT_CONSTS = unit_sprites._ROTATION_VARIANT_CONSTS | unit_sprites._CLIFF_VARIANT_CONSTS


def _old_resolved_frame(player_id: int, unit, angle_count: int, frame_count: int) -> int:
    """The frame index this placement resolved to before Tier B: verbatim
    rotation through angle_index(), except GAIA zeroed to rotation 0.0 --
    render.stored_rotation()'s own pre-Tier-B rule, replayed here rather than
    called, since the shipped function now implements the NEW rule."""
    old_rotation = 0.0 if player_id == 0 else float(unit.rotation)
    return unit_sprites.angle_index(old_rotation, angle_count) * frame_count


def scan_file(path: Path) -> dict:
    scenario = load_map_and_units(path)
    graphic_map = unit_sprites.graphic_map()

    per_const: dict[int, Counter] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            const = unit.unit_const
            if const in _OLD_VARIANT_CONSTS or not unit_sprites.rotation_is_variant(const):
                continue
            entry = graphic_map.get(const)
            if entry is None:
                continue
            angle_count = max(1, int(entry["angle_count"]))
            frame_count = max(1, int(entry["frame_count"]))
            real_frames = unit_sprites.sld_frame_count(str(entry["file_name"]))
            # The real VARIANT slot count, not the file's raw total -- same
            # division _frame_for() now applies, since a frame_count > 1
            # graphic (a decay animation, say) packs several on-disk frames
            # per variant. angle_count is the fallback with no install, same
            # as _frame_for()'s own default.
            variant_count = (
                max(1, real_frames // frame_count) if real_frames is not None else angle_count
            )

            new_rotation = render.stored_rotation(player_id, unit)
            literal = unit_sprites.is_literal_variant_index(new_rotation, variant_count)
            out_of_file = (
                not literal
                and abs(new_rotation - round(new_rotation)) < 1e-6
                and round(new_rotation) >= variant_count
            )
            new_frame = unit_sprites._frame_for(const, entry, new_rotation)
            old_frame = _old_resolved_frame(player_id, unit, angle_count, frame_count)

            counter = per_const.setdefault(const, Counter())
            counter["placements"] += 1
            counter["literal" if literal else "radian"] += 1
            counter["out_of_file"] += int(out_of_file)
            counter["frame_changed"] += int(old_frame != new_frame)
            if real_frames is not None:
                # The raw on-disk index _native_frame() itself gates on
                # (0 <= frame_index < sld.frame_count) -- compared directly
                # against real_frames, not against the divided variant_count
                # above, since this is the file-frame-table bound, not the
                # rotation-dispatch one.
                counter["no_sprite"] += int(
                    unit_sprites.sprite_for(const, new_rotation, player_id, 32) is None
                )
                counter["resolved_past_real_frames"] += int(new_frame >= real_frames)

    return per_const


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples")
    args = parser.parse_args()

    files = sorted(
        p for p in args.scenario_dir.iterdir()
        if p.suffix in (".aoe2scenario", ".scx2") or p.name == "play_Test"
    )
    if not files:
        print(f"No scenario files found under {args.scenario_dir}", file=sys.stderr)
        sys.exit(1)

    combined: dict[int, Counter] = {}
    for path in files:
        try:
            per_const = scan_file(path)
        except Exception as exc:  # noqa: BLE001 -- a scan tool, not production code
            print(f"{path.name}: FAILED: {exc}", file=sys.stderr)
            continue
        for const, counter in per_const.items():
            combined.setdefault(const, Counter()).update(counter)

    print(f"{'const':>6s} {'file_name':32s} {'n':>7s} {'lit':>7s} {'rad':>7s} {'chg':>7s} {'oof':>5s}")
    totals: Counter = Counter()
    graphic_map = unit_sprites.graphic_map()
    for const in sorted(combined):
        c = combined[const]
        name = str(graphic_map.get(const, {}).get("file_name", "?"))
        print(
            f"{const:6d} {name:32s} {c['placements']:7d} {c['literal']:7d} "
            f"{c['radian']:7d} {c['frame_changed']:7d} {c['out_of_file']:5d}"
        )
        totals.update(c)

    print()
    print(f"Total placements on a newly-variant const: {totals['placements']}")
    print(f"  of which resolved frame actually changes: {totals['frame_changed']}")
    print(f"  literal index past the file's real frame count: {totals['out_of_file']}")
    if "no_sprite" in totals:
        print(f"  degrade to no sprite (install-gated): {totals['no_sprite']}")
        print(f"  resolved index >= real frame count (install-gated): {totals['resolved_past_real_frames']}")
    else:
        print("  (no install configured -- no-sprite and past-real-frame-count checks skipped)")


if __name__ == "__main__":
    main()
