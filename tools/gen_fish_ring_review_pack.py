#!/usr/bin/env python3
"""Review pack for View > Layers > Fish Outlines: one ring per fish, of its
whole footprint, on its own tile heights, and none on anything else.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md). The ring
is a MapView scene item, so like tools/gen_fish_review_pack.py (whose capture
helpers it reuses) this drives a real offscreen ViewerWindow in Stepped on raised
ground and rasterizes scene rects 1:1. Needs a configured AoE2:DE install.

Counts and extents are pinned in tests/test_footprint_outlines.py; this pack
asks only what a test cannot see: whether each drawn ring reads as sitting
around its drawn creature, whether a large one reads as one shape, and whether
an ordinary object stays bare.

Run:
    tools/gen_fish_ring_review_pack.py --out-dir <neutral dir>
    tools/gen_fish_ring_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_fish_ring_review_pack.py --out-dir <neutral dir> --inject sunk
    tools/gen_fish_ring_review_pack.py --check      # writes nothing

Writes <out-dir>/marks/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_fish_review_pack as gfp
import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from testkit import review_pack as rp
from testkit import settings_isolation

PACK_ID = "marks"
OUT_DIR = grp.OUT_DIR
SHORE_FISH, WHALE, GOLD_MINE = 69, 2625, 66
# frame id -> (const, x, y). The Whale stands on a 2-high block, the rest on height 1.
_PLACED = {"a": (SHORE_FISH, 54.5, 54.5), "b": (WHALE, 60.5, 60.5), "c": (GOLD_MINE, 66.5, 66.5)}
_HIGH_BLOCK = range(57, 64)
CHECKER = gfp.CHECKER
GROUND_SHIFT_TILES = 12
FISH_LAYER = "fish_outlines"


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
        tile.elevation = 2 if tile.x in _HIGH_BLOCK and tile.y in _HIGH_BLOCK else 1
    model = UnitEditModel(loaded)
    for const, x, y in _PLACED.values():
        model.add(0, const, x, y)
    out = tmp / "marks.aoe2scenario"
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects


def _inject_shifted():
    """Coarse: every ring moved one whole tile down-right of its creature, in a
    pen four times as wide, so its delta outsizes every localized inject's."""
    from descape.map_view import MapView

    real = MapView._fish_ring_path

    def shifted(self, entries):
        return real(self, entries).translated(self._iso_proj.half_w, self._iso_proj.half_h)

    return [(MapView, "_fish_ring_path", shifted), ("pen", 4 * MapView.FISH_OUTLINE_PEN_WIDTH)]


def _inject_sunk():
    """Rings built from a flat height field while the art stands on raised
    ground: a stale overlay after a style switch looks exactly like this."""
    from descape.map_view import MapView

    real = MapView._fish_ring_path

    def sunk(self, entries):
        saved = self._iso_elevations
        self._iso_elevations = None if saved is None else np.zeros_like(saved)
        try:
            return real(self, entries)
        finally:
            self._iso_elevations = saved

    return [(MapView, "_fish_ring_path", sunk)]


def _inject_cells():
    """Every footprint tile ringed on its own: Merge tiles not applied."""
    from descape import unit_pick

    return [(unit_pick, "merge_footprint_polygons", list)]


def _inject_mine():
    """The gold mine counted as a fish."""
    from descape import unit_kind

    real = unit_kind.fish_consts
    return [(unit_kind, "fish_consts", lambda: real() | {GOLD_MINE})]


INJECTS = {"shifted": _inject_shifted, "sunk": _inject_sunk, "cells": _inject_cells, "mine": _inject_mine}
COARSE_INJECT = "shifted"


# ----------------------------------------------------------------- capture


_WINDOW = None
_SCENARIO = None
_RECTS = None


def _window():
    global _WINDOW, _SCENARIO
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="marks_pack_"))
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


def _load(layer_on: bool) -> dict[int, object]:
    """Loads the scenario fresh in Stepped, View mode, Footprint Outlines off and
    nothing selected, with the fish layer as asked. Entries by const."""
    window = _window()
    window.layer_actions[FISH_LAYER].setChecked(False)
    window.load_scenario(_SCENARIO)
    if window.scenario is None:
        raise SystemExit("the generated scenario failed to load")
    window.mode_combo.setCurrentText("View")
    window.terrain_style_combo.setCurrentText("Stepped")
    if window.footprint_action.isChecked():
        window.footprint_action.setChecked(False)
    window.layer_actions[FISH_LAYER].setChecked(layer_on)
    index = window.map_view._unit_index  # View mode builds it only for the layer
    return {} if index is None else {e.unit.unit_const: e for e in index.entries}


def _rects():
    """Frame rects from the clean footprints, so every run crops the same scene px."""
    global _RECTS
    if _RECTS is None:
        mv = _window().map_view
        entries = _load(layer_on=True)
        rects = {}
        for fid, (const, _x, _y) in _PLACED.items():
            b = mv._unit_path(entries[const]).boundingRect()
            pad_x, foot = b.width() * 0.5, b.height() * 0.8
            head = max(b.height() * 1.2, 70.0)
            rects[fid] = gfp._int_rect(b.left() - pad_x, b.top() - head, b.right() + pad_x, b.bottom() + foot)
        half_w, half_h = mv._iso_proj.half_w, mv._iso_proj.half_h
        rects["d"] = rects["a"].translated(-GROUND_SHIFT_TILES * half_w, GROUND_SHIFT_TILES * half_h)
        _RECTS = rects
    return _RECTS


def _capture(inject: str | None) -> dict[str, np.ndarray]:
    rects = _rects()
    mv = _window().map_view
    built = [] if inject is None else INJECTS[inject]()
    widths = [p[1] for p in built if p[0] == "pen"]
    with gfp._patched([p for p in built if p[0] != "pen"]):
        _load(layer_on=True)
        if widths:
            pen = mv._fish_item.pen()
            pen.setWidth(widths[0])
            mv._fish_item.setPen(pen)
        try:
            return {fid: gfp._grab(mv, rect) for fid, rect in rects.items()}
        finally:
            mv._fish_item.setPen(mv._fish_pen)


def _capture_layer_off() -> dict[str, np.ndarray]:
    """The clean scene with the layer off: what every ring measurement diffs against."""
    rects = _rects()
    mv = _window().map_view
    _load(layer_on=False)
    return {fid: gfp._grab(mv, rect) for fid, rect in rects.items()}


# --------------------------------------------------------------- measurement


def _ring_report(images: dict[str, np.ndarray], bare: dict[str, np.ndarray]) -> list[str]:
    """The ring is whatever switching the layer off changes."""
    lines = []
    for fid in sorted(images):
        mask = rp.changed_pixels(images[fid], bare[fid])
        n = int(np.count_nonzero(mask))
        box = rp.delta_bbox(mask) if n else None
        lines.append(f"{fid}: ring {n}px, {rp.components(mask) if n else 0} component(s), bbox {box}")
    return lines


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "A small creature on raised checkered ground, angled view."),
            ("b", "A large creature on a higher block of checkered ground, angled view."),
            ("c", "A resource object on raised checkered ground, angled view."),
            ("d", "Checkered ground, same framing and zoom as the first frame."),
        )
    )
    ground = (
        "The ground is a checkerboard of two alternating ground textures, one diamond-shaped cell per map "
        "tile; raised ground is drawn as a block whose top is lifted above its surroundings, with a "
        "darker side face below its edge. An outline here is a thin, bright, light-blue line with no fill. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                ground + "First, in each frame, describe the creature, the cell or cells under it, and any "
                "outline: which cells it traces and where it lies relative to the creature's base. Then: in "
                "either frame, does an outline sit on cells other than the ones the creature stands on (for "
                "example shifted below or beside it, or on a lower level than the cells under the creature)? "
                "Ignore the creature's own body, any shadow and the side faces of the raised ground. AROUND "
                "if every outline traces the cells under its creature, DISPLACED if some outline traces "
                "other cells, NO-OUTLINE if neither frame has an outline."
            ),
            frames=("a", "b"),
            answers=("AROUND", "DISPLACED", "NO-OUTLINE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                ground + "First describe the outline: its overall shape, and every line of the same colour "
                "inside its border, if any. Then: is it one closed shape with nothing of that colour inside "
                "it, or is its inside divided by lines of that colour into smaller diamonds? Ignore the "
                "checker pattern's own cell edges, which are where two ground textures meet rather than "
                "drawn lines. ONE-SHAPE, DIVIDED, or NO-OUTLINE if there is no outline."
            ),
            frames=("b",),
            answers=("ONE-SHAPE", "DIVIDED", "NO-OUTLINE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                ground + "First describe the object and everything drawn on the ground around it. Then: is "
                "there a light-blue outline around the object or on the cells under it? OUTLINED or BARE."
            ),
            frames=("c",),
            answers=("OUTLINED", "BARE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
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
    images = _capture(inject)
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


def _option_masks(name: str, injected, bare, rings) -> list[str]:
    """The mask each answer option rests on, and each localized inject's size
    bound (measured 2026-11-26: sunk drops 8/16 px, cells ~2x the ring px)."""
    ring = {fid: rp.changed_pixels(injected[fid], bare[fid]) for fid in injected}
    clean_px = {fid: int(rings[fid].sum()) for fid in rings}
    out = []
    if name == "sunk":
        for fid in ("a", "b"):
            drop = rp.delta_bbox(ring[fid])[1] - rp.delta_bbox(rings[fid])[1]
            if not 4 <= drop <= 24:
                out.append(f"inject sunk: {fid} ring drops {drop}px, outside 4..24")
    elif name == "cells":
        if not 1.5 * clean_px["b"] <= ring["b"].sum() <= 2.5 * clean_px["b"]:
            out.append(f"inject cells: b ring {int(ring['b'].sum())}px is not divided (clean {clean_px['b']})")
    elif name == "mine" and (rp.components(ring["c"]) != 1 or not 0.5 * clean_px["a"] <= ring["c"].sum() <= 2 * clean_px["a"]):
        out.append(f"inject mine: c ring {int(ring['c'].sum())}px is not one 1-tile ring")
    return out


def check() -> None:
    """Spec coupling, leaks, frame band, determinism, each inject's reach per
    frame (localized under coarse on shared frames), the ring masks each
    answer option rests on, and each localized inject's size bound."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += [f"leak: {hit}" for hit in rp.leaks(spec)]
    clean = _capture(None)
    again = _capture(None)
    bare = _capture_layer_off()
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean captures differ")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    for line in _ring_report(clean, bare):
        print(f"measure: clean {line}")
    rings = {fid: rp.changed_pixels(clean[fid], bare[fid]) for fid in clean}
    for fid in ("a", "b"):
        if not rings[fid].any() or rp.components(rings[fid]) != 1:
            failures.append(f"clean {fid}: the ring is not one component")
    for fid in ("c", "d"):
        if rings[fid].any():
            failures.append(f"clean {fid}: a ring where none belongs")
    expected = {"shifted": {"a", "b"}, "sunk": {"a", "b"}, "cells": {"b"}, "mine": {"c"}}
    reach = {}
    for name in INJECTS:
        injected = _capture(name)
        failures += _option_masks(name, injected, bare, rings)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if set(touched) != expected[name]:
            failures.append(f"inject {name}: touched {sorted(touched)}, expected {sorted(expected[name])}")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
        for line in _ring_report(injected, bare):
            print(f"measure: {name} {line}")
    coarse = reach[COARSE_INJECT]
    for name, touched in reach.items():
        if name == COARSE_INJECT:
            continue
        for fid, (_px, comp) in touched.items():
            if fid in coarse and comp >= coarse[fid][1]:
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
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory {grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
