#!/usr/bin/env python3
"""Review pack for TASK-134: a class-14 doodad (a 6x6 mountain) marks, outlines
and blocks on its real footprint, while its sprite keeps its own point and its
own depth slot.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md). Modelled
on tools/gen_blocker_review_pack.py (a footprint grown past one tile, the
pre-change render as the `one-tile` inject), with the capture of
tools/gen_fish_review_pack.py: the selection outline is a set of MapView
items, so this drives a real offscreen ViewerWindow in Stepped and rasterizes
scene rects 1:1 with testkit.qt_capture. Needs a configured AoE2:DE install
(sprites and terrain).

Frames, all Stepped on flat checkered ground:
- a: a mountain at (54.0, 54.0), sprites off: its mark.
- b: the same mountain, sprites on and selected: its outline under its art.
- c: a second mountain at (70.0, 40.0) with a tree on its near slope, tile
  (69, 41), sprites on. Depth order is ascending d = y - x: the tree's tile
  has d = -28 and the mountain's own tile -30, so the tree paints after (in
  front of) the mountain. The footprint's front corner (67, 42) has d = -25,
  so the building-like anchor paints the mountain over the tree. `--check`
  measures how much of the tree each run shows.
- d: checkered ground only, frame a's rect moved along +y: the specificity frame.

Injects: `vanish` (coarse: no placed objects), `one-tile` (the doodad table
emptied: the pre-TASK-134 1x1 mark and outline) and `front-anchor` (the
rejected building-like anchor: the sprite sorts at the footprint's front
corner, so the tree on the slope paints under the mountain).

The PACK_ID and every frame/check name are neutral on purpose: frame paths
appear in REVIEW.md, and nothing a reviewer reads may name the subject.

Run:
    tools/gen_doodad_review_pack.py --out-dir <neutral dir>
    tools/gen_doodad_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_doodad_review_pack.py --out-dir <neutral dir> --inject one-tile
    tools/gen_doodad_review_pack.py --check      # writes nothing

Writes <out-dir>/features/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from testkit import review_pack as rp
from testkit import settings_isolation

PACK_ID = "features"
OUT_DIR = grp.OUT_DIR
MNTN3 = 744  # 6x6 footprint, 4x4 clearance
TREE = 349
# (const, x, y); the mountains on whole coordinates, where an even span centres.
_LONE = (MNTN3, 54.0, 54.0)
_BACKED = (MNTN3, 70.0, 40.0)
_TREE = (TREE, 69.5, 41.5)
_PLACED = (_LONE, _BACKED, _TREE)
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1: one diamond cell per tile
# The specificity frame: frame a's rect moved this many tiles along +y, clear of every object
# (the rect is about 2.5 footprints tall, so less than this still overlaps frame a).
GROUND_SHIFT_TILES = 30


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    for const, x, y in _PLACED:
        model.add(0, const, x, y)
    out = tmp / "features.aoe2scenario"
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects


def _inject_vanish():
    """Coarse: no placed object draws anything, mark, art or outline (the same ground, empty)."""
    return [(sys.modules[__name__], "_HIDE_ALL", True)]


def _inject_one_tile():
    """The pre-TASK-134 footprint: every doodad back to 1x1 (mark, outline, occupied tiles)."""
    from descape import terrain_palette

    return [(terrain_palette, "DOODAD_TILE_SPANS", {}), (_render_module(), "DOODAD_TILE_SPANS", {})]


def _inject_front_anchor():
    """The rejected building-like anchor: a doodad's sprite sorts at its footprint's front corner."""
    return [(_render_module(), "_sprite_keeps_own_point", lambda unit_const: False)]


def _render_module():
    from descape import render

    return render


INJECTS = {
    "vanish": _inject_vanish,
    "one-tile": _inject_one_tile,
    "front-anchor": _inject_front_anchor,
}
COARSE_INJECT = "vanish"
_HIDE_ALL = False


@contextmanager
def _patched(patches):
    from descape import unit_sprites

    saved = [(mod, name, getattr(mod, name)) for mod, name, _new in patches]
    for mod, name, new in patches:
        setattr(mod, name, new)
    unit_sprites.clear_caches()
    try:
        yield
    finally:
        for mod, name, old in saved:
            setattr(mod, name, old)
        unit_sprites.clear_caches()


# ----------------------------------------------------------------- capture


def _int_rect(x0: float, y0: float, x1: float, y1: float):
    from PyQt5.QtCore import QRectF

    left, top = int(np.floor(x0)), int(np.floor(y0))
    return QRectF(left, top, int(np.ceil(x1)) - left, int(np.ceil(y1)) - top)


def _grab(mv, rect) -> np.ndarray:
    from PyQt5.QtWidgets import QApplication

    from testkit.qt_capture import scene_rect_to_array

    QApplication.processEvents()
    return scene_rect_to_array(mv.scene(), rect)


_WINDOW = None
_SCENARIO = None
_EMPTY = None
_RECTS = None


def _window():
    global _WINDOW, _SCENARIO, _EMPTY
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="features_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _SCENARIO = _scenario_file(tmp)
        _EMPTY = _empty_file(tmp)
        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
    return _WINDOW


def _empty_file(tmp: Path) -> Path:
    """The same checkered ground with nothing placed: the coarse inject's scene."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    out = tmp / "ground.aoe2scenario"
    write_scenario(loaded, out, backup=False)
    return out


def _load(sprites: bool, selected: bool) -> dict[tuple[float, float], object]:
    """Loads the scenario fresh (so patched span and sprite code re-renders) in
    Stepped with sprites on or off; selects the lone mountain when `selected`.
    Entries by stored (x, y)."""
    window = _window()
    window.show_sprites_action.setChecked(sprites)
    window.load_scenario(_EMPTY if _HIDE_ALL else _SCENARIO)
    if window.scenario is None:
        raise SystemExit("the generated scenario failed to load")
    window.mode_combo.setCurrentText("View")
    window.terrain_style_combo.setCurrentText("Stepped")
    window.mode_combo.setCurrentText("Units")
    mv = window.map_view
    entries = {(e.unit.x, e.unit.y): e for e in mv._unit_index.entries}
    chosen = [entries[_LONE[1:]]] if selected and _LONE[1:] in entries else []
    window._selection = [(e.player_id, e.unit.reference_id) for e in chosen]
    mv.set_unit_selection(chosen)
    return entries


def _rects():
    """Frame rects from the CLEAN outlines, so every run crops the same scene px."""
    global _RECTS
    if _RECTS is None:
        mv = _window().map_view
        entries = _load(sprites=True, selected=False)
        rects = {}
        for fid, placed in (("a", _LONE), ("c", _BACKED)):
            b = mv._unit_path(entries[placed[1:]]).boundingRect()
            # The art rises well above the footprint: room for it above, a little ground below.
            pad_x, head, foot = b.width() * 0.15, b.height() * 0.75, b.height() * 0.25
            rects[fid] = _int_rect(b.left() - pad_x, b.top() - head, b.right() + pad_x, b.bottom() + foot)
        rects["b"] = rects["a"]
        half_w, half_h = mv._iso_proj.half_w, mv._iso_proj.half_h
        rects["d"] = rects["a"].translated(-GROUND_SHIFT_TILES * half_w, GROUND_SHIFT_TILES * half_h)
        _RECTS = rects
    return _RECTS


def _built(inject: str | None, extra=()) -> list:
    return ([] if inject is None else INJECTS[inject]()) + list(extra)


def _frame_images(inject: str | None, extra=()) -> dict[str, np.ndarray]:
    rects = _rects()
    mv = _window().map_view
    out = {}
    with _patched(_built(inject, extra)):
        _load(sprites=False, selected=False)
        out["a"] = _grab(mv, rects["a"])
        _load(sprites=True, selected=True)
        for fid in ("b", "c", "d"):
            out[fid] = _grab(mv, rects[fid])
    return out


# --------------------------------------------------------------- measurement


def _transparent_tree():
    """The tree drawn fully transparent: what it changes is its visible ink."""
    from descape import unit_sprites

    real = unit_sprites.sprite_pieces_for

    def clear(unit_const, *args, **kwargs):
        pieces = real(unit_const, *args, **kwargs)
        if unit_const != TREE:
            return pieces
        out = []
        for piece in pieces:
            rgba = piece.draw.rgba.copy()
            rgba[..., 3] = 0
            draw = unit_sprites.SpriteDraw(rgba=rgba, hotspot_x=piece.draw.hotspot_x, hotspot_y=piece.draw.hotspot_y)
            out.append(unit_sprites.SpritePiece(draw=draw, dx=piece.dx, dy=piece.dy, slot=piece.slot))
        return out

    return unit_sprites, "sprite_pieces_for", clear


def _measure(inject: str | None, images: dict[str, np.ndarray]) -> list[str]:
    """What the answers rest on, measured off this run's frames: the coloured
    mark's extent in frame a, the outline's tile count and how far the art
    reaches past it in b, and the tree's visible ink in c."""
    window = _window()
    mv = window.map_view
    lines = []
    half_w = mv._iso_proj.half_w
    ground = _frame_images(COARSE_INJECT)
    with _patched(_built(inject)):
        entries = _load(sprites=True, selected=True)
        lone = entries.get(_LONE[1:])
        polys = mv._unit_polygons_for(lone) if lone is not None else []
        _load(sprites=True, selected=False)
        art = rp.changed_pixels(_grab(mv, _rects()["b"]), ground["b"])
    outline_w = 0.0
    if polys:
        px = [x for poly in polys for x, _y in poly]
        outline_w = (max(px) - min(px)) / (2 * half_w)
    art_cols = np.nonzero(art.any(axis=0))[0]
    art_w = (art_cols.max() - art_cols.min() + 1) / (2 * half_w) if len(art_cols) else 0.0
    lines.append(f"b: outline tiles {len(polys or [])}, outline {outline_w:.2f} tiles wide, art {art_w:.2f} tiles wide")
    if inject != COARSE_INJECT:
        bare = _frame_images(inject, [_transparent_tree()])
        tree_px = int(np.count_nonzero(rp.changed_pixels(images["c"], bare["c"])))
        lines.append(f"c: tree ink visible {tree_px}px")
    mark = rp.changed_pixels(images["a"], ground["a"])
    _ys, xs = np.nonzero(mark)
    if len(xs):
        lines.append(f"a: mark {len(xs)}px, {(xs.max() - xs.min() + 1) / (2 * half_w):.2f} tiles wide")
    else:
        lines.append("a: no mark")
    return lines


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "One object on checkered ground, angled view, drawn as a plain coloured shape."),
            ("b", "The same object drawn with its artwork, selected, angled view."),
            ("c", "A second object of the same kind with a smaller object near it, angled view."),
            ("d", "Checkered ground, same framing and zoom as the first frame."),
        )
    )
    checker = (
        "The ground is a checkerboard of two alternating ground textures, one diamond-shaped cell per map "
        "tile. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                checker + "An object is drawn here as a flat coloured shape lying on the ground. First describe "
                "the shape and list the checker cells it covers, counting them along each of its two diagonal "
                "directions. Then: how many cells does it cover? ONE for a single cell, THIRTY-SIX for a "
                "6-by-6 block, OTHER for any other count, NONE if there is no coloured shape."
            ),
            frames=("a",),
            answers=("ONE", "THIRTY-SIX", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                checker + "A selected object has a thin coloured outline drawn on the ground around it, with a "
                "translucent fill inside; the outline may be drawn as one shape or as several adjacent "
                "diamonds. The object's artwork is drawn over the ground and may cover part of the outline. "
                "First describe the outline's shape and list the checker cells it encloses, counting them "
                "along each of its two diagonal directions, including cells partly hidden by the artwork. "
                "Then: how many whole checker cells does the outline enclose? ONE for a single cell, "
                "THIRTY-SIX for a 6-by-6 block, OTHER for any other count, NONE if there is no outline."
            ),
            frames=("b",),
            answers=("ONE", "THIRTY-SIX", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                checker + "Compare the selected object's artwork with the outline on the ground under it. "
                "First describe where the artwork's lowest edge and its left-most and right-most points lie "
                "relative to the outline's four corners. Then: does the outline span roughly the width of the "
                "artwork's base? SPANS if the outline's left and right corners are each within about one "
                "checker cell of the artwork's base edges, SMALL if the artwork's base reaches clearly past "
                "the outline on both sides, NONE if there is no outline or no artwork."
            ),
            frames=("b",),
            answers=("SPANS", "SMALL", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                checker + "There is a large object and, near its lower edge, a smaller one (a tree). Find every "
                "part of the tree: its trunk, its foliage, and its base where it meets the ground or the large "
                "object. First describe which parts of the tree you can see and which the large object's "
                "artwork covers. Then: is the whole tree drawn in front of the large object? FRONT if the "
                "whole tree is visible and no part of it is covered by the large object, COVERED if the large "
                "object's artwork is drawn over some or all of the tree, including when you cannot find the "
                "tree at all."
            ),
            frames=("c",),
            answers=("FRONT", "COVERED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                "First describe everything drawn in this frame. Then: is there any object, outline or coloured "
                "mark in it, or only the checkered ground? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("d",),
            answers=("OBJECTS-PRESENT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object footprint review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_ID / grp._run_token(PACK_ID, inject, replica)
    pack_dir.mkdir(parents=True, exist_ok=True)
    images = _frame_images(inject)
    written, rendered = [], {}
    for frame in spec.frames:
        scaled, factor = rp.frame_to_band(images[frame.id])
        path = pack_dir / frame.filename
        seam_eyeball._save_png(scaled, path)
        written.append(path)
        rendered[frame.id] = (scaled.shape[1], scaled.shape[0], factor)
    review = pack_dir / "REVIEW.md"
    review.write_text(grp._review_markdown(spec, rendered, pack_dir))
    written.append(review)
    return written


def check() -> None:
    """Spec coupling, frame band, determinism, each inject's reach per frame,
    and the measurements each run's answers rest on."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    again = _frame_images(None)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean captures differ")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    for line in _measure(None, clean):
        print(f"measure: clean {line}")
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        if "d" in touched:
            failures.append(f"inject {name}: reached the bare-ground frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
        for line in _measure(name, injected):
            print(f"measure: {name} {line}")
    coarse = reach[COARSE_INJECT]
    for name, touched in reach.items():
        if name == COARSE_INJECT:
            continue
        for fid, (_px, comp) in touched.items():
            if comp >= coarse.get(fid, (0, 0))[1]:
                failures.append(f"inject {name}: {fid} component {comp}px is not under the coarse inject's")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of one run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
