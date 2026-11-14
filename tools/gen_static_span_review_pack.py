#!/usr/bin/env python3
"""Review pack for TASK-277 (GH #333): a shipwreck, a flare, the hidden fish and
a hidden eye-candy const draw, outline and mark on their real footprint.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), with the
capture of tools/gen_doodad_review_pack.py: the selection outline is a set of
MapView items, so this drives a real offscreen ViewerWindow in Stepped and
rasterizes scene rects 1:1 with testkit.qt_capture. Needs a configured AoE2:DE
install (sprites and terrain).

Frames, all Stepped on flat checkered ground:
- a: Leviathan Shipwreck 2353 (4x4) at (54.0, 54.0), sprites on, selected.
- b: FLARE4 697 (2x2, invisible: its marker is the footprint) at (66.5, 54.5),
  off its span's parity, sprites on, selected.
- c: DOLP2 451 (2x2) at (54.0, 66.0), sprites on, selected.
- d: DOLP3 452 (2x2) at (66.0, 66.0), sprites on: its body draws at all.
- e: FLR_R 201 (2x2, no art) at (78.0, 54.0), sprites off: its plain mark.
- f: checkered ground only, frame a's rect moved along +y: the specificity frame.

Injects: `vanish` (coarse: no placed objects), `one-tile` (the pre-TASK-277
spans: static_spans emptied, the six hidden fish dropped from fish_spans),
`own-point` (the flare keeps its own point like a wreck does, so its marker
sits half a tile off its outline) and `empty-body` (DOLP3 without its
frame_offset: frame 0 of the Marlin body is empty).

The PACK_ID and every frame/check name are neutral on purpose: frame paths
appear in REVIEW.md, and nothing a reviewer reads may name the subject.

Run:
    tools/gen_static_span_review_pack.py --out-dir <neutral dir>
    tools/gen_static_span_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_static_span_review_pack.py --out-dir <neutral dir> --inject one-tile
    tools/gen_static_span_review_pack.py --check      # writes nothing

Writes <out-dir>/pieces/<opaque token>/ (build/review_pack/ by default, gitignored).
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
from testkit.render_family import RENDER, patch

PACK_ID = "pieces"
OUT_DIR = grp.OUT_DIR
WRECK, FLARE, DOLP2, DOLP3, FLR_R = 2353, 697, 451, 452, 201
HIDDEN_FISH = (451, 452, 454, 459, 460, 461)
# frame id -> (const, x, y); whole coordinates for even spans, except the flare, off parity on purpose.
_PLACED = {
    "a": (WRECK, 54.0, 54.0),
    "b": (FLARE, 66.5, 54.5),
    "c": (DOLP2, 54.0, 66.0),
    "d": (DOLP3, 66.0, 66.0),
    "e": (FLR_R, 78.0, 54.0),
}
_SELECTED = ("a", "b", "c")
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1: one diamond cell per tile
# The specificity frame: frame a's rect moved this many tiles along +y, clear of every object.
GROUND_SHIFT_TILES = 30


def _ground_file(tmp: Path, name: str, placed) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    for const, x, y in placed:
        model.add(0, const, x, y)
    out = tmp / name
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects
#
# An inject is a list of ("attr", obj, name, value) or ("item", mapping, key, value) patches;
# value _DROP removes the key. Tables are patched in place: render.py imports them by name.

_DROP = object()


def _inject_vanish():
    """Coarse: no placed object draws anything (the same ground, empty)."""
    return [("attr", sys.modules[__name__], "_HIDE_ALL", True)]


def _inject_one_tile():
    """The pre-TASK-277 spans: no static_spans, and the eight original fish only."""
    from descape import terrain_palette

    statics = [("item", terrain_palette.STATIC_TILE_SPANS, c, _DROP) for c in list(terrain_palette.STATIC_TILE_SPANS)]
    return statics + [("item", terrain_palette.FISH_TILE_SPANS, c, _DROP) for c in HIDDEN_FISH]


def _inject_own_point():
    """The flare keeps its own point, as a shipwreck's art does: no invisible-const exception."""
    from descape import terrain_palette

    keeps = lambda unit_const: unit_const in terrain_palette.DOODAD_TILE_SPANS or unit_const in terrain_palette.STATIC_TILE_SPANS  # noqa: E731
    return [("attr", RENDER, "_sprite_keeps_own_point", keeps)]


def _inject_empty_body():
    """DOLP3 without its frame_offset: frame 0 of the Marlin body, which is empty."""
    from descape import unit_sprites

    return [("item", unit_sprites.graphic_map()[DOLP3], "frame_offset", _DROP)]


INJECTS = {
    "vanish": _inject_vanish,
    "one-tile": _inject_one_tile,
    "own-point": _inject_own_point,
    "empty-body": _inject_empty_body,
}
COARSE_INJECT = "vanish"
_HIDE_ALL = False
# (inject, frame) -> largest changed component allowed, where the coarse inject changes less than the
# defect does: DOLP3's clean body is 253px, and with its empty frame it falls back to a 2x2 plain
# mark (4096px, measured 2026-11-03). Bounded by that mark, and to frame d alone.
_OWN_BOUND = {("empty-body", "d"): 4096}


@contextmanager
def _patched(patches):
    import pytest

    from descape import unit_sprites

    try:
        with pytest.MonkeyPatch.context() as mp:
            for kind, target, key, new in patches:
                if kind == "attr":
                    patch(mp, target, key, new)
                elif new is _DROP:
                    mp.delitem(target, key, raising=False)
                else:
                    mp.setitem(target, key, new)
            unit_sprites.clear_caches()
            yield
    finally:
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
        tmp = Path(tempfile.mkdtemp(prefix="pieces_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _SCENARIO = _ground_file(tmp, "pieces.aoe2scenario", _PLACED.values())
        _EMPTY = _ground_file(tmp, "ground.aoe2scenario", ())
        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
    return _WINDOW


def _load(sprites: bool, selected: bool) -> dict[tuple[float, float], object]:
    """Loads the scenario fresh (so patched span and sprite code re-renders) in
    Stepped with sprites on or off, selecting frames a-c's objects when
    `selected`. Entries by stored (x, y)."""
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
    chosen = [entries[_PLACED[f][1:]] for f in _SELECTED if selected and _PLACED[f][1:] in entries]
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
        for fid, (_const, x, y) in _PLACED.items():
            b = mv._unit_path(entries[(x, y)]).boundingRect()
            # Art rises above the footprint (the wreck's mast most): room above, a little ground around.
            pad_x, head, foot = b.width() * 0.6, b.height() * 1.2, b.height() * 0.6
            rects[fid] = _int_rect(b.left() - pad_x, b.top() - head, b.right() + pad_x, b.bottom() + foot)
        half_w, half_h = mv._iso_proj.half_w, mv._iso_proj.half_h
        rects["f"] = rects["a"].translated(-GROUND_SHIFT_TILES * half_w, GROUND_SHIFT_TILES * half_h)
        _RECTS = rects
    return _RECTS


def _built(inject: str | None) -> list:
    return [] if inject is None else INJECTS[inject]()


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    rects = _rects()
    mv = _window().map_view
    out = {}
    with _patched(_built(inject)):
        _load(sprites=False, selected=False)
        out["e"] = _grab(mv, rects["e"])
        _load(sprites=True, selected=True)
        for fid in ("a", "b", "c", "f"):
            out[fid] = _grab(mv, rects[fid])
        _load(sprites=True, selected=False)
        out["d"] = _grab(mv, rects["d"])
    return out


# --------------------------------------------------------------- measurement


def _measure(inject: str | None, images: dict[str, np.ndarray]) -> list[str]:
    """What the answers rest on, measured off this run: the outline tile count
    of frames a-c's objects, how far the flare's drawn ink centre sits from its
    outline's centre, and the ink frames d and e show over bare ground."""
    mv = _window().map_view
    half_w, half_h = mv._iso_proj.half_w, mv._iso_proj.half_h
    ground = _frame_images(COARSE_INJECT)
    lines = []
    with _patched(_built(inject)):
        entries = _load(sprites=True, selected=True)
        for fid in _SELECTED:
            entry = entries.get(_PLACED[fid][1:])
            polys = mv._unit_polygons_for(entry) if entry is not None else []
            lines.append(f"{fid}: outline tiles {len(polys or [])}")
        flare = entries.get(_PLACED["b"][1:])
        outline = mv._unit_polygons_for(flare) if flare is not None else []
        _load(sprites=True, selected=False)
        marker = rp.changed_pixels(_grab(mv, _rects()["b"]), ground["b"])  # unselected: the marker alone
    rect = _rects()["b"]
    scale = images["b"].shape[1] / rect.width()
    if outline:
        pts = np.array([p for poly in outline for p in poly], dtype=float)
        ocx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        ocy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        ys, xs = np.nonzero(marker)
        if len(xs):
            icx = rect.left() + (xs.min() + xs.max() + 1) / 2 / scale
            icy = rect.top() + (ys.min() + ys.max() + 1) / 2 / scale
            off = f"({(icx - ocx) / half_w:+.2f}, {(icy - ocy) / half_h:+.2f})"
            lines.append(f"b: marker centre off the outline centre by {off} half-tiles")
    for fid in ("d", "e"):
        mark = rp.changed_pixels(images[fid], ground[fid])
        _ys, xs = np.nonzero(mark)
        width = (xs.max() - xs.min() + 1) / (2 * half_w) if len(xs) else 0.0
        lines.append(f"{fid}: ink {len(xs)}px, {width:.2f} tiles wide")
    return lines


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "A large object drawn with its artwork, selected, angled view."),
            ("b", "A small object, selected, angled view."),
            ("c", "A creature, selected, angled view."),
            ("d", "A spot where a creature is placed, angled view."),
            ("e", "One object on checkered ground, angled view, drawn as a plain coloured shape."),
            ("f", "Checkered ground, same framing and zoom as the first frame."),
        )
    )
    checker = (
        "The ground is a checkerboard of two alternating ground textures, one diamond-shaped cell per map "
        "tile. "
    )
    outline = (
        "A selected object has a thin coloured outline drawn on the ground around it, with a translucent fill "
        "inside; the outline may be drawn as one shape or as several adjacent diamonds. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                checker + outline + "The object's artwork is drawn over the ground and may cover part of the "
                "outline. First describe the outline's shape and list the checker cells it encloses, counting "
                "them along each of its two diagonal directions, including cells partly hidden by the artwork. "
                "Then: how many whole checker cells does the outline enclose? ONE for a single cell, SIXTEEN "
                "for a 4-by-4 block, OTHER for any other count, NONE if there is no outline."
            ),
            frames=("a",),
            answers=("ONE", "SIXTEEN", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                checker + "Compare the selected object's artwork with the outline on the ground under it. "
                "First describe where the artwork's lowest edge and its left-most and right-most points lie "
                "relative to the outline's four corners. Then: does the outline span roughly the width of the "
                "artwork's base? SPANS if the outline's left and right corners are each within about one "
                "checker cell of the artwork's base edges, SMALL if the artwork's base reaches clearly past "
                "the outline on both sides, NONE if there is no outline or no artwork."
            ),
            frames=("a",),
            answers=("SPANS", "SMALL", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                checker + outline + "First describe the outline's shape and list the checker cells it "
                "encloses, counting them along each of its two diagonal directions. Then: how many whole "
                "checker cells does the outline enclose? ONE for a single cell, FOUR for a 2-by-2 block, OTHER "
                "for any other count, NONE if there is no outline."
            ),
            frames=("b",),
            answers=("ONE", "FOUR", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                checker + outline + "Inside or near the outline the object itself is drawn as a solid "
                "diamond-shaped symbol, more opaque than the outline's translucent fill. First describe the "
                "symbol and where each of its four corners lies relative to the outline's four corners. Then: "
                "does the symbol sit on the outline? ALIGNED if every corner of the symbol is within about a "
                "quarter of a checker cell of the matching outline corner, SHIFTED if the symbol is offset "
                "from the outline by about half a checker cell or more in some direction, NONE if there is no "
                "symbol or no outline."
            ),
            frames=("b",),
            answers=("ALIGNED", "SHIFTED", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                checker + outline + "A creature is drawn over the ground inside it, possibly faint or partly "
                "transparent, and may cover part of the outline. First describe the outline's shape and list "
                "the checker cells it encloses, counting them along each of its two diagonal directions. "
                "Then: how many whole checker cells does the outline enclose? ONE for a single cell, FOUR for "
                "a 2-by-2 block, OTHER for any other count, NONE if there is no outline."
            ),
            frames=("c",),
            answers=("ONE", "FOUR", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q6",
            question=(
                checker + "First describe everything drawn on the ground in this frame, however faint. "
                "Then: is a creature's body drawn here? BODY if you can see a creature or animal shape, "
                "NOTHING if there is only the checkered ground or a plain flat coloured shape."
            ),
            frames=("d",),
            answers=("BODY", "NOTHING", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q7",
            question=(
                checker + "An object is drawn here as a flat coloured shape lying on the ground. First describe "
                "the shape and list the checker cells it covers, counting them along each of its two diagonal "
                "directions. Then: how many cells does it cover? ONE for a single cell, FOUR for a 2-by-2 "
                "block, OTHER for any other count, NONE if there is no coloured shape."
            ),
            frames=("e",),
            answers=("ONE", "FOUR", "OTHER", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q8",
            question=(
                "First describe everything drawn in this frame. Then: is there any object, outline or coloured "
                "mark in it, or only the checkered ground? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("f",),
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
        if "f" in touched:
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
            if (name, fid) in _OWN_BOUND:
                # The defect replaces a small body with a bigger fallback mark, so vanish cannot bound it.
                if set(touched) != {fid} or comp > _OWN_BOUND[(name, fid)]:
                    failures.append(f"inject {name}: reaches {sorted(touched)} or {fid} {comp}px over its own bound")
                continue
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
