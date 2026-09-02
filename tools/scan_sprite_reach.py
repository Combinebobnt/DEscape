#!/usr/bin/env python3
"""How far a sprite's pixels reach past its own hotspot, in native sprite px.

Informational only, matching this project's bench_iso_backend.py /
bench_unit_sprites.py convention: always runs, never pass/fail. Exits 0 with a
clear message when no AoE2:DE install is configured, rather than failing --
nothing is bundled, so "no install" is a normal state here, not an error. The
pass/fail counterpart is tests/test_sprite_edit_bbox.py's `corpus` test, which
asserts each maximum below is still within the constant committed beside it.

**This is what sizes unit_sprites.MAX_SPRITE_REACH_LEFT/RIGHT/UP/DOWN**, which
descape/render.py's _sprite_reach_px turns into the per-side padding an edit's
dirty bbox needs so a sprite anchored on a dirty tile cannot escape it. Rerun
this after a game patch: a new or re-authored graphic taller than
b_west_wonder_britons_x1 would silently reintroduce the stale-fragment bug the
padding exists to prevent.

## What it reads, and why that bounds the real thing

Header boxes only -- no decode_frame anywhere -- so the whole scan is a few
seconds rather than the hours a full decode of 1,600 files would take. Per
frame it reads the MAIN layer's own bounding box (x1, y1, width, height) and
the frame's hotspot, and reports hotspot-relative reach on each side.

That box is an upper bound on what unit_sprites._cropped_to_ink actually
produces: the ink bbox is MAIN's opaque pixels, which cannot lie outside MAIN's
own box. Bounding rather than measuring exactly is the point -- the constants
must never under-state a real sprite, and a slightly generous bbox only costs
repaint area.

Every distinct file_name in descape/unit_graphic_map.json is walked, not a
sample: the 400-file sample this replaced under-stated the worst reach and, more
importantly, got the dominant direction wrong (it is UP, by a wide margin).

Unreadable files are counted, not skipped silently. They are the files absent
from this install, which load_sld already returns None for -- so they draw no
sprite at all and cannot contribute reach. The `14` layout variant used to be
counted here too; it became readable in 2026-08, which moved 59 referenced
files into the scan. Every maximum above survived that unchanged.

**Composite pieces (unit_graphic_map.json's optional "pieces" key) are scanned
at their own (dx, dy) offset from the unit's shared anchor, not at (0, 0).**
_sprite_reach_px widens an edit's dirty bbox around that shared anchor, so a
piece's contribution to the real widening a unit needs is its own MAIN-layer
reach shifted by its offset -- a piece sitting 192px to one side needs that
192px added to its own intrinsic reach on that side, or an edit near it could
under-widen and leave a stale fragment. Town centre pieces net (0, 0) (finding
5's cancellation) so this is a no-op for them; pastures are the real case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, unit_sprites
from descape.sld_decoder import LayerKind, load_sld

# The four directions, each as (label, constant name), in the order they are
# printed and in the order _sprite_reach_px returns them.
DIRECTIONS = (
    ("left", "MAX_SPRITE_REACH_LEFT"),
    ("up", "MAX_SPRITE_REACH_UP"),
    ("right", "MAX_SPRITE_REACH_RIGHT"),
    ("down", "MAX_SPRITE_REACH_DOWN"),
)


def frame_reach(frame, dx: int = 0, dy: int = 0) -> tuple[int, int, int, int] | None:
    """(left, up, right, down) native-pixel reach of this frame's MAIN layer
    from the point (dx, dy) away from its own hotspot -- (0, 0) for a plain
    sprite, a composite piece's own offset otherwise, so the reach is
    relative to the shared unit anchor _sprite_reach_px widens around, not
    the piece's own hotspot. None if the frame carries no MAIN layer.

    A layer box can extend past the frame canvas (165 layers of the surveyed
    install do -- see SLDLayer's docstring), so the box is used as stored
    rather than clipped to the canvas. Clipping would under-state reach on
    exactly the frames most likely to be the worst case.
    """
    main = next((layer for layer in frame.layers if layer.kind == LayerKind.MAIN), None)
    if main is None:
        return None
    hx, hy = frame.hotspot_x + dx, frame.hotspot_y + dy
    return (
        hx - main.x1,
        hy - main.y1,
        (main.x1 + main.width) - hx,
        (main.y1 + main.height) - hy,
    )


def scan(graphics_dir: Path, entries: list[tuple[str, int, int]]):
    """(maxima, worst_files, readable, unreadable) over every named .sld.

    `entries` is (file_name, dx, dy) -- (name, 0, 0) for a plain sprite, a
    composite piece's own offset otherwise. maxima and worst_files are both
    4-tuples in DIRECTIONS order. A file with no readable frame contributes
    nothing and counts as unreadable. Reading the same file at two different
    offsets (a pasture's corner-post graphic used by all four annexes) reads
    it from disk twice rather than caching -- this is an informational,
    few-seconds tool, not a hot path.
    """
    maxima = [0, 0, 0, 0]
    worst = ["", "", "", ""]
    readable = unreadable = 0
    for name, dx, dy in entries:
        sld = load_sld(graphics_dir / f"{name}.sld")
        if sld is None:
            unreadable += 1
            continue
        readable += 1
        for frame in sld.frames:
            reach = frame_reach(frame, dx, dy)
            if reach is None:
                continue
            for i in range(4):
                if reach[i] > maxima[i]:
                    maxima[i] = reach[i]
                    worst[i] = name if (dx, dy) == (0, 0) else f"{name} (piece, dx={dx} dy={dy})"
    return maxima, worst, readable, unreadable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--install", type=Path, help="AoE2:DE install root override")
    args = parser.parse_args()

    if args.install:
        asset_source.set_install_path_override(args.install)
    install = asset_source.get_install_path()
    if install is None:
        print(
            "No AoE2:DE install configured, so there are no sprite pixels to "
            "measure. Configure one (see README.md) or pass --install. "
            "Informational tool: this is not a failure."
        )
        return

    graphics_dir = Path(install) / unit_sprites.GRAPHICS_SUBPATH
    entries: set[tuple[str, int, int]] = set()
    for gm_entry in unit_sprites.graphic_map().values():
        entries.add((gm_entry["file_name"], 0, 0))
        for piece in gm_entry.get("pieces", ()):
            entries.add((piece["file_name"], piece["dx"], piece["dy"]))
    entries = sorted(entries)
    print(f"Sprite reach scan -- install {install}")
    print(f"{len(entries)} distinct (file, offset) entries referenced by unit_graphic_map.json")

    maxima, worst, readable, unreadable = scan(graphics_dir, entries)
    print(f"{readable} readable, {unreadable} unreadable (absent from this install)")
    print(f"\nWorst reach from the hotspot, native px at NATIVE_TILE_W = {unit_sprites.NATIVE_TILE_W}:\n")
    width = max(len(const) for _label, const in DIRECTIONS)
    for i, (_label, const) in enumerate(DIRECTIONS):
        print(f"{const:<{width}} = {maxima[i]:<5} # {worst[i]}")
    print("\nPaste into descape/unit_sprites.py if these have grown; see the constants there.")

    committed = [getattr(unit_sprites, const) for _label, const in DIRECTIONS]
    over = [
        f"{label} {maxima[i]} > {committed[i]}"
        for i, (label, _const) in enumerate(DIRECTIONS)
        if maxima[i] > committed[i]
    ]
    if over:
        print(f"\nSCANNED REACH EXCEEDS THE COMMITTED CONSTANTS: {', '.join(over)}")
    else:
        print("\nEvery scanned maximum is within its committed constant.")


if __name__ == "__main__":
    main()
