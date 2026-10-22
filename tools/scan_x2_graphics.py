#!/usr/bin/env python3
"""How the install's `_x2` (UHD) sprites line up with the `_x1` ones DEscape draws (GH #241).

Informational only, like scan_sprite_reach.py: always runs, never pass/fail,
and exits 0 with a message when no install is configured.

For every `_x1` stem unit_sprites.referenced_entries() names (particle effects
have no .sld and are skipped) it reports:

- whether the `_x2` file exists beside it;
- whether its frame count equals the `_x1` file's (unit_sprites only draws x2
  when it does, so a facing or variant can never change);
- the raw hotspot difference `|x2 hotspot - 2 * x1 hotspot|` per frame, in x2 px;
- x2 MAIN-box reach / 2 at the piece's own offset, against
  unit_sprites.MAX_SPRITE_REACH_*;
- total file bytes, x1 and x2.

`--ror` scans Return of Rome's referenced art instead: each x1 file where
unit_sprites finds it (the RoR root's copy, else the install's) and the x2
copy beside it, the pair the viewer draws.

Those are header reads only, as in scan_sprite_reach.py. A MAIN layer's
stored box is not its ink (block padding, and decay animations whose boxes
differ between the two files), so `--ink` adds the check that decides whether
x2 drawn at its own hotspot lands where x1 does: it decodes the standing
frame of up to four facings per stem in both files, crops each to its ink as
unit_sprites does, and compares where the ink sits from the hotspot (x2 / 2
against x1, x1 px): the ink box's worst side and the alpha-weighted centroid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import asset_source, game_dataset, unit_sprites
from descape.sld_decoder import LayerKind, SLDError, load_sld

# How many rows of each worst-first list are printed.
LIST_LIMIT = 25

RETURN_OF_ROME_SUBPATH = "modes/Pompeii/" + unit_sprites.GRAPHICS_SUBPATH

REACH_CONSTS = ("MAX_SPRITE_REACH_LEFT", "MAX_SPRITE_REACH_RIGHT", "MAX_SPRITE_REACH_UP", "MAX_SPRITE_REACH_DOWN")

BUCKETS = (("<= 0.5", -1.0, 0.5), ("<= 1", 0.5, 1.0), ("<= 2", 1.0, 2.0), ("> 2", 2.0, float("inf")))


def x2_stem(name: str) -> str | None:
    """`..._x1` -> `..._x2`, else None."""
    return name[:-3] + "_x2" if name.endswith("_x1") else None


def main_box(frame) -> tuple[int, int, int, int] | None:
    """(left, right, up, down) of the frame's MAIN box from its own hotspot, or None."""
    main = next((layer for layer in frame.layers if layer.kind == LayerKind.MAIN), None)
    if main is None:
        return None
    hx, hy = frame.hotspot_x, frame.hotspot_y
    return (hx - main.x1, main.x1 + main.width - hx, hy - main.y1, main.y1 + main.height - hy)


def raw_hotspot_difference(x1, x2) -> int:
    """Worst `|x2 hotspot - 2 * x1 hotspot|` over the frames, x2 px."""
    return max(
        (max(abs(f2.hotspot_x - 2 * f1.hotspot_x), abs(f2.hotspot_y - 2 * f1.hotspot_y))
         for f1, f2 in zip(x1.frames, x2.frames, strict=False)),
        default=0,
    )


def x2_reach(x2, dx: int, dy: int) -> tuple[float, float, float, float]:
    """(left, right, up, down) of every x2 frame's MAIN box from the piece anchor, in x1 px."""
    reach = [0.0, 0.0, 0.0, 0.0]
    for frame in x2.frames:
        box = main_box(frame)
        if box is None:
            continue
        # dx/dy are x1 px from the unit anchor to the piece's hotspot.
        left, right, up, down = box
        sides = (left / 2 + dx, right / 2 - dx, up / 2 + dy, down / 2 - dy)
        reach = [max(a, b) for a, b in zip(reach, sides, strict=True)]
    return tuple(reach)


def standing_frames(frame_count: int, angle_count: int, total: int) -> list[int]:
    """The standing frame of up to four facings spread over the angles."""
    angles = sorted({angle_count * k // 4 for k in range(4)})
    return [a * frame_count for a in angles if a * frame_count < total]


def ink(sld, index: int):
    """(left, right, up, down, centroid x, centroid y) of the frame's ink from its hotspot, or None."""
    try:
        frame = sld.decode_frame(index, kinds=(LayerKind.MAIN,), window_to_main=True)
    except SLDError:
        return None
    if frame.main is None:
        return None
    cropped = unit_sprites._cropped_to_ink(frame.main, None, frame.hotspot_x, frame.hotspot_y)
    if cropped is None:
        return None
    main, _pc, hx, hy = cropped
    h, w = main.shape[:2]
    alpha = main[..., 3].astype(np.float64)
    total = alpha.sum()
    cx = (alpha.sum(axis=0) * (np.arange(w) + 0.5)).sum() / total - hx
    cy = (alpha.sum(axis=1) * (np.arange(h) + 0.5)).sum() / total - hy
    return hx, w - hx, hy, h - hy, cx, cy


def ink_drift(x1, x2, indices: list[int]) -> tuple[float, float] | None:
    """(worst ink-box side drift, worst centroid drift), x1 px, over `indices`; None if no frame has ink in both."""
    box_drift = centroid_drift = None
    for index in indices:
        a, b = ink(x1, index), ink(x2, index)
        if a is None or b is None:
            continue
        sides = max(abs(s2 / 2 - s1) for s1, s2 in zip(a[:4], b[:4], strict=True))
        centroid = max(abs(b[4] / 2 - a[4]), abs(b[5] / 2 - a[5]))
        box_drift = sides if box_drift is None else max(box_drift, sides)
        centroid_drift = centroid if centroid_drift is None else max(centroid_drift, centroid)
    return None if box_drift is None else (box_drift, centroid_drift)


def print_buckets(values: list[float]) -> None:
    for label, low, high in BUCKETS:
        print(f"    {label:>6}: {sum(1 for v in values if low < v <= high)}")


def scan(graphics_dir: Path, check_ink: bool) -> dict:
    """Prints the report; returns {"usable": n, "reach": [left, right, up,
    down], "reach_worst": [file per side]} for tests/test_x2_sprites.py's
    corpus test."""
    particles = unit_sprites._particle_stems()
    offsets: dict[str, set[tuple[int, int]]] = {}
    layout: dict[str, tuple[int, int]] = {}
    per_civ = [e for art in unit_sprites.building_art().civ_art.values() for e in art.values()]
    for entry in [*unit_sprites.graphic_map().values(), *per_civ]:
        for item in [entry, *entry.get("pieces", ())]:
            layout.setdefault(item["file_name"], (max(1, int(item["frame_count"])), max(1, int(item["angle_count"]))))
    for name, dx, dy in unit_sprites.referenced_entries():
        if name not in particles:
            offsets.setdefault(name, set()).add((dx, dy))

    stems = sorted(offsets)
    no_x1_name = [n for n in stems if x2_stem(n) is None]
    missing_x1: list[str] = []
    missing_x2: list[str] = []
    unreadable_x2: list[str] = []
    count_mismatch: list[tuple[str, int, int]] = []
    usable = 0
    raws: list[tuple[int, str]] = []
    inks: list[tuple[float, float, str]] = []
    no_ink: list[str] = []
    reach = [0.0, 0.0, 0.0, 0.0]
    reach_worst = ["", "", "", ""]
    bytes_x1 = bytes_x2 = 0

    for position, name in enumerate(stems):
        if check_ink and position % 200 == 0:
            print(f"  ... {position}/{len(stems)}", file=sys.stderr)
        x2_name = x2_stem(name)
        if x2_name is None:
            continue
        # As unit_sprites draws them: x1 through _sld_path() (the RoR root's copy under --ror), x2 beside it.
        path1 = unit_sprites._sld_path(graphics_dir, name)
        path2 = path1.parent / f"{x2_name}.sld"
        x1 = load_sld(path1, keep_data=check_ink)
        if x1 is None:
            missing_x1.append(name)
            continue
        if not path2.is_file():
            missing_x2.append(name)
            continue
        bytes_x1 += path1.stat().st_size
        bytes_x2 += path2.stat().st_size
        x2 = load_sld(path2, keep_data=check_ink)
        if x2 is None:
            unreadable_x2.append(x2_name)
            continue
        if x2.frame_count != x1.frame_count:
            count_mismatch.append((name, x1.frame_count, x2.frame_count))
            continue
        usable += 1
        raws.append((raw_hotspot_difference(x1, x2), name))
        for dx, dy in offsets[name]:
            for i, side in enumerate(x2_reach(x2, dx, dy)):
                if side > reach[i]:
                    reach[i] = side
                    reach_worst[i] = x2_name if (dx, dy) == (0, 0) else f"{x2_name} (piece, dx={dx} dy={dy})"
        if check_ink:
            frame_count, angle_count = layout.get(name, (1, 1))
            drift = ink_drift(x1, x2, standing_frames(frame_count, angle_count, x1.frame_count))
            if drift is None:
                no_ink.append(name)
            else:
                inks.append((drift[1], drift[0], name))

    print(f"{len(stems)} distinct .sld stems referenced (particles skipped)")
    print(f"{len(no_x1_name)} do not end in _x1 (never get x2): {', '.join(no_x1_name[:LIST_LIMIT])}")
    print(f"{len(missing_x1)} x1 files absent or unreadable on this install")
    print(f"{len(missing_x2)} x1 files with no x2 file beside them")
    print(f"{len(unreadable_x2)} x2 files present but unreadable: {', '.join(unreadable_x2[:LIST_LIMIT])}")
    print(f"{len(count_mismatch)} x2 files whose frame count differs from x1 (x1 drawn)")
    for name, c1, c2 in count_mismatch[:LIST_LIMIT]:
        print(f"    {name}: x1 {c1}, x2 {c2}")
    print(f"{usable} x2 files usable (same frame count)")

    print("\nRaw hotspot difference |x2 - 2 * x1|, x2 px, worst frame per file:")
    print_buckets([float(r) for r, _n in raws])
    for raw, name in sorted(raws, reverse=True)[:LIST_LIMIT]:
        if raw > 0:
            print(f"    {raw:5d}  {name}")

    if check_ink:
        print(f"\nDecoded ink from the hotspot, x2 / 2 against x1, x1 px ({len(inks)} files, {len(no_ink)} with no ink in both):")
        print("  centroid drift (worst axis, worst checked facing):")
        print_buckets([c for c, _b, _n in inks])
        print("  ink-box drift (worst side, worst checked facing):")
        print_buckets([b for _c, b, _n in inks])
        print(f"  Worst {LIST_LIMIT} by centroid (centroid, box):")
        for centroid, box, name in sorted(inks, reverse=True)[:LIST_LIMIT]:
            print(f"    {centroid:6.2f} {box:6.1f}  {name}")

    print("\nx2 reach / 2 from the anchor, x1 px, against the committed constants:")
    for i, const in enumerate(REACH_CONSTS):
        committed = getattr(unit_sprites, const)
        flag = f"  OVER by {reach[i] - committed:.1f}" if reach[i] > committed else ""
        print(f"    {const:<22} x2 {reach[i]:7.1f}  committed {committed}{flag}  # {reach_worst[i]}")

    print(f"\nBytes on disk: x1 {bytes_x1 / 2**20:.1f} MB, x2 {bytes_x2 / 2**20:.1f} MB (files with both)")
    return {"usable": usable, "reach": reach, "reach_worst": reach_worst}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", type=Path, help="AoE2:DE install root override")
    parser.add_argument("--ink", action="store_true", help="also decode standing frames and compare their ink (slow)")
    parser.add_argument("--ror", action="store_true", help="scan Return of Rome's referenced art and its x2 copies")
    args = parser.parse_args()

    if args.install:
        asset_source.set_install_path_override(args.install)
    install = asset_source.get_install_path()
    if install is None:
        print(
            "No AoE2:DE install configured, so there are no sprites to compare. "
            "Configure one (see README.md) or pass --install. Informational tool: this is not a failure."
        )
        return
    graphics_dir = Path(install) / unit_sprites.GRAPHICS_SUBPATH
    if args.ror:
        game_dataset.set_active(game_dataset.ROR)
    print(f"x2 sprite scan -- install {install}, {'Return of Rome' if args.ror else 'AoE2'} art")
    for label, directory in (("AoE2", graphics_dir), ("RoR", Path(install) / RETURN_OF_ROME_SUBPATH)):
        count = sum(1 for _ in directory.glob("*_x2.sld")) if directory.is_dir() else "no folder"
        print(f"{label} _x2.sld files in {directory}: {count}")
    scan(graphics_dir, args.ink)


if __name__ == "__main__":
    main()
