#!/usr/bin/env python3
"""Review pack for GH #178: deep-water fish drawn at the in-game outline size
(Dorado 2x2, Whale 3x3) with their underwater body inside it, and Perch's body
at its .dat delta offset rather than 30 native px low.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_selection_review_pack.py, whose capture approach
it shares: the outline is a set of MapView selection items, so this drives a
real offscreen ViewerWindow in Stepped and rasterizes scene rects 1:1 with
testkit.qt_capture. Needs a configured AoE2:DE install (sprites and terrain).

The PACK_ID and every frame/check name are neutral on purpose: frame paths
appear in REVIEW.md, and nothing a reviewer reads may name the subject.

Perch's frame overhangs its 2x2 outline even at the right offset (17.6% of its
ink at native scale, all of it a small detached blob below-left), so its check
is not "inside or not". In this Stepped capture that blob does not show at all
(no changed px there in either run), and what separates the two renders is
which side of the outline's left-right corner line the body is on: rows
-13.5..+6.5 scene px around it with the offset, +6.5..+26.5 without (bottom tip
+32), measured 2026-10-09. `--check` re-measures every run and prints it.

Run:
    tools/gen_fish_review_pack.py --out-dir <neutral dir>
    tools/gen_fish_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_fish_review_pack.py --out-dir <neutral dir> --inject one-tile
    tools/gen_fish_review_pack.py --check      # writes nothing

Writes <out-dir>/objects/<opaque token>/ (build/review_pack/ by default, gitignored).
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

PACK_ID = "objects"
OUT_DIR = grp.OUT_DIR
DORADO, WHALE, PERCH = 455, 2625, 53
# frame id -> (const, x, y): whole coordinates for the 2x2s, .5 for the 3x3, as the corpus places them.
_PLACED = {"a": (DORADO, 54.0, 54.0), "b": (WHALE, 60.5, 60.5), "c": (PERCH, 66.0, 66.0)}
FISH = tuple(const for const, _x, _y in _PLACED.values())
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1: one diamond cell per tile
# The specificity frame: frame a's rect moved this many tiles along +y, clear of every object.
GROUND_SHIFT_TILES = 12


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    for const, x, y in _PLACED.values():
        model.add(0, const, x, y)
    out = tmp / "objects.aoe2scenario"
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects


def _transparent_fish():
    """Every fish piece drawn fully transparent: no body, and no mark fallback."""
    from descape import unit_sprites

    real = unit_sprites.sprite_pieces_for

    def clear(unit_const, *args, **kwargs):
        pieces = real(unit_const, *args, **kwargs)
        if unit_const not in FISH:
            return pieces
        out = []
        for piece in pieces:
            rgba = piece.draw.rgba.copy()
            rgba[..., 3] = 0
            draw = unit_sprites.SpriteDraw(rgba=rgba, hotspot_x=piece.draw.hotspot_x, hotspot_y=piece.draw.hotspot_y)
            out.append(unit_sprites.SpritePiece(draw=draw, dx=piece.dx, dy=piece.dy, slot=piece.slot))
        return out

    return unit_sprites, "sprite_pieces_for", clear


def _inject_vanish():
    """Coarse: no fish body and no selection outline anywhere."""
    return [_transparent_fish()]


def _inject_one_tile():
    """The pre-GH #178 span: every fish back to a 1x1 footprint and outline."""
    from descape import terrain_palette

    return [(terrain_palette, "FISH_TILE_SPANS", {})]


def _inject_no_offset():
    """The pre-amendment body position: Perch's .dat delta offset dropped."""
    from descape import unit_sprites

    return [(unit_sprites, "_body_offset", lambda entry: (0, 0))]


INJECTS = {
    "vanish": _inject_vanish,
    "one-tile": _inject_one_tile,
    "no-offset": _inject_no_offset,
}
COARSE_INJECT = "vanish"
_UNSELECTED = {"vanish"}


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
_RECTS = None


def _window():
    global _WINDOW, _SCENARIO
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="objects_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _SCENARIO = _scenario_file(tmp)
        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
    return _WINDOW


def _load(selected: bool) -> dict[int, object]:
    """Loads the scenario fresh (so patched sprite and span code re-renders) in
    Stepped, Units mode; selects the fish when `selected`. Entries by const."""
    window = _window()
    window.load_scenario(_SCENARIO)
    if window.scenario is None:
        raise SystemExit("the generated scenario failed to load")
    window.mode_combo.setCurrentText("View")
    window.terrain_style_combo.setCurrentText("Stepped")
    window.mode_combo.setCurrentText("Units")
    mv = window.map_view
    entries = {e.unit.unit_const: e for e in mv._unit_index.entries}
    chosen = [entries[c] for c in FISH] if selected else []
    window._selection = [(e.player_id, e.unit.reference_id) for e in chosen]
    mv.set_unit_selection(chosen)
    return entries


def _rects():
    """Frame rects from the CLEAN outlines, so every run crops the same scene px."""
    global _RECTS
    if _RECTS is None:
        mv = _window().map_view
        entries = _load(selected=True)
        rects = {}
        for fid, (const, _x, _y) in _PLACED.items():
            b = mv._unit_path(entries[const]).boundingRect()
            pad_x, head, foot = b.width() * 0.3, b.height() * 0.6, b.height() * 0.4
            rects[fid] = _int_rect(b.left() - pad_x, b.top() - head, b.right() + pad_x, b.bottom() + foot)
        half_w, half_h = mv._iso_proj.half_w, mv._iso_proj.half_h
        rects["d"] = rects["a"].translated(-GROUND_SHIFT_TILES * half_w, GROUND_SHIFT_TILES * half_h)
        _RECTS = rects
    return _RECTS


def _built(inject: str | None, extra=()) -> list:
    return ([] if inject is None else INJECTS[inject]()) + list(extra)


def _capture(inject: str | None, extra=()) -> tuple[dict[str, np.ndarray], dict[int, object]]:
    rects = _rects()
    mv = _window().map_view
    with _patched(_built(inject, extra)):
        entries = _load(selected=inject not in _UNSELECTED)
        out = {fid: _grab(mv, rect) for fid, rect in rects.items()}
        outlines = {const: mv._unit_polygons_for(entries[const]) for const in FISH}
    return out, outlines


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    return _capture(inject)[0]


# --------------------------------------------------------------- measurement


def _body_report(inject: str | None, images: dict[str, np.ndarray]) -> list[str]:
    """What each check's answer rests on, measured off this run's frames: the
    body is whatever turning the fish transparent changes."""
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPainterPath, QPolygonF

    bare, outlines = _capture(inject, [_transparent_fish()])
    rects = _rects()
    lines = []
    for fid, (const, _x, _y) in _PLACED.items():
        mask = rp.changed_pixels(images[fid], bare[fid])
        ys, xs = np.nonzero(mask)
        polys = outlines[const] or []
        if not len(xs) or not polys:
            lines.append(f"{fid}: body {len(xs)}px, outline tiles {len(polys)}")
            continue
        sx = xs + rects[fid].left() + 0.5
        sy = ys + rects[fid].top() + 0.5
        path = QPainterPath()
        for points in polys:
            path.addPolygon(QPolygonF([QPointF(px, py) for px, py in points]))
            path.closeSubpath()
        outside = sum(1 for px, py in zip(sx, sy, strict=True) if not path.contains(QPointF(px, py)))
        box = path.boundingRect()
        centre_y, tip_y = box.center().y(), box.bottom()
        top, bottom = sy.min() - centre_y, sy.max() - centre_y
        side = "LOWER-HALF" if top > 0 else "UPPER-HALF" if bottom < 0 else "STRADDLES"
        lines.append(
            f"{fid}: body {len(xs)}px, outline tiles {len(polys)}, outside {outside}px "
            f"({100 * outside / len(xs):.1f}%), body rows {top:+.1f}..{bottom:+.1f} from the "
            f"left-right corner line (bottom tip {tip_y - centre_y:+.1f}): {side}"
        )
    return lines


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "One selected object on checkered ground, angled view."),
            ("b", "A second selected object on checkered ground, angled view."),
            ("c", "A third selected object on checkered ground, angled view."),
            ("d", "Checkered ground, same framing and zoom as the first frame."),
        )
    )
    checker = (
        "The ground is a checkerboard of two alternating ground textures, one diamond-shaped cell per map "
        "tile. A selected object has a thin coloured outline drawn on the ground around it, with a "
        "translucent fill inside; the outline may be drawn as one shape or as several adjacent diamonds. "
    )
    count = (
        "First describe the outline's shape and list the checker cells it encloses, counting them along "
        "each of its two diagonal directions. Then: how many whole checker cells does the outline enclose? "
        "ONE for a single cell, FOUR for a 2-by-2 block, NINE for a 3-by-3 block, OTHER for any other "
        "count, NONE if there is no outline."
    )
    checks = (
        rp.Check(
            id="q1",
            question=checker + count,
            frames=("a",),
            answers=("ONE", "FOUR", "NINE", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=checker + count,
            frames=("b",),
            answers=("ONE", "FOUR", "NINE", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                checker + "Inside or around each outline there is a creature, drawn as part of the map "
                "(not the translucent fill, and not the ground pattern). First, in each frame, describe the "
                "creature and where its body lies relative to the outline: which corners and edges it is "
                "near, and whether any part of it crosses the outline line. Then: does any part of a "
                "creature's body cross outside its outline in either frame? CROSSES if some part does, "
                "INSIDE if every body lies wholly within its outline, NO-BODY if no creature is drawn."
            ),
            frames=("a", "b"),
            answers=("CROSSES", "INSIDE", "NO-BODY", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                checker + "Imagine a horizontal line through the outline's left-most and right-most corners: "
                "it splits the outline into an upper (far) half and a lower (near) half. This line is not "
                "drawn; any lines drawn inside the outline run diagonally and are not it. Then find the "
                "creature's body (not the translucent fill, and not the ground pattern). First describe "
                "where the body's highest and lowest points are relative to that imagined line and to the "
                "outline's top and bottom tips. Then: which side of the line is the body on? LOWER-HALF if "
                "the whole body is below the line, STRADDLES if part of it is above the line and part below, "
                "UPPER-HALF if the whole body is above the line, NO-BODY if no creature is drawn."
            ),
            frames=("c",),
            answers=("LOWER-HALF", "STRADDLES", "UPPER-HALF", "NO-BODY", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                "First describe everything drawn in this frame. Then: is there any object, creature, outline "
                "or coloured mark in it, or only the checkered ground? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("d",),
            answers=("OBJECTS-PRESENT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object outline review",
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
    and the body/outline measurements each run's answers rest on."""
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
    for line in _body_report(None, clean):
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
        for line in _body_report(name, injected):
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
