#!/usr/bin/env python3
"""Review pack for the zoomed-out look past the coarsest level (TASK-031.70):
MapCanvasItem.paint() blits that level box-reduced k times there.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_selection_review_pack.py, whose window and
inject scaffolding this reuses. gen_review_pack.py renders off-engine and
cannot reach the reduction, which only exists inside paint(), and
testkit.qt_capture.scene_rect_to_array is 1:1 (level 0), so every capture
here is one scene.render() of a scene rect into a target `zoom` times
larger, the way tests/test_mip_viewer.py's _render_scene_rect does it. Every
reduced crop asserts item._last_reduction > 0; the base twins and the
close-up reference assert 0. Drives a real offscreen ViewerWindow on
examples/old-allies-final-v2.aoe2scenario (1920x1080, the zoom bench's
window) and needs a configured AoE2:DE install and the native kernel.

Run:
    tools/gen_coarse_reduce_review_pack.py --out-dir <neutral dir>
    tools/gen_coarse_reduce_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_coarse_reduce_review_pack.py --out-dir <neutral dir> --inject k0
    tools/gen_coarse_reduce_review_pack.py --check      # writes nothing; prints each inject's reach

Writes <out-dir>/p/<opaque token>/ (default out-dir build/review_pack, gitignored).
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

PACK_ID = "coarse_reduce"
# The directory under --out-dir: neutral, since every frame path is printed in REVIEW.md.
PACK_DIR = "p"
OUT_DIR = grp.OUT_DIR
SCENARIO = ROOT / "examples" / "old-allies-final-v2.aoe2scenario"
WINDOW = (1920, 1080)
# Device px of every crop; 6x nearest-neighbour puts it at 1500x1020.
CROP = (250, 170)
# Percent of fit: 50 is k = 2 and 75 is k = 1 on this map at this window (residual 0.20 and 0.30).
FAR, MID, NEAR = 0.5, 0.75, 2.0
# The edit: an L of this terrain id (snow), painted after the reduced entries exist.
EDIT_TERRAIN = 32


# ----------------------------------------------------------------- injects


def _inject_k0():
    """k forced to 0: today's unreduced bilinear blit at the same zoom."""
    from descape import viewer_canvas

    return viewer_canvas, "reduction_for", lambda coarsest, scale: 0


def _wrap_painted_reduction(transform):
    """render_reduced_rect() with `transform(block, x0, chunk_px)` applied to the
    block paint() asked for, never to the k-1 builds nested under it."""
    from descape.render_cache import _ChunkCacheBase

    real = _ChunkCacheBase.render_reduced_rect
    depth = [0]

    def wrapped(self, x0, y0, x1, y1, mip, k):
        depth[0] += 1
        try:
            block = real(self, x0, y0, x1, y1, mip, k)
        finally:
            depth[0] -= 1
        return block if depth[0] else transform(block, max(0, x0), self.chunk_px)

    return _ChunkCacheBase, "render_reduced_rect", wrapped


def _inject_shift():
    """A one-pixel source shift: in every odd reduced chunk column the content
    moves one reduced pixel right, so it is offset and its edges are seams."""

    def shift(block, x0, size):
        out = block.copy()
        for c0 in range(x0 - x0 % size, x0 + block.shape[1], size):
            if (c0 // size) % 2 == 0:
                continue
            lo, hi = max(c0, x0) - x0, min(c0 + size, x0 + block.shape[1]) - x0
            out[:, lo + 1 : hi] = block[:, lo : hi - 1]
        return out

    return _wrap_painted_reduction(shift)


def _inject_ghost():
    """A half-intensity copy of every reduced pixel two reduced pixels to the right."""

    def ghost(block, _x0, _size):
        out = block.copy()
        out[:, 2:] = ((block[:, 2:].astype(np.uint16) + block[:, :-2]) // 2).astype(np.uint8)
        return out

    return _wrap_painted_reduction(ghost)


def _inject_stale():
    """_write_slice() without its stamp: a reduced entry built before an edit stays valid."""
    from descape.render_cache import _ChunkCacheBase

    def write_slice(self, chunk, mip, cx, cy, ix0, iy0, ix1, iy1, pixels):
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        chunk[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = pixels

    return _ChunkCacheBase, "_write_slice", write_slice


INJECTS = {
    "k0": _inject_k0,
    "shift": _inject_shift,
    "ghost": _inject_ghost,
    "stale": _inject_stale,
}
COARSE_INJECT = "k0"
LOCAL_INJECT = "stale"


@contextmanager
def _patched(patches):
    saved = [(obj, name, getattr(obj, name)) for obj, name, _new in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


# ----------------------------------------------------------------- capture


def _render(mv, centre, zoom: float) -> np.ndarray:
    """One paint of the CROP-sized device rect around scene point `centre` at `zoom`."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtWidgets import QApplication

    from testkit.qt_capture import qimage_rgb888_to_array

    QApplication.processEvents()
    tw, th = CROP
    w, h = tw / zoom, th / zoom
    rect = QRectF(centre[0] - w / 2, centre[1] - h / 2, w, h)
    image = QImage(tw, th, QImage.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    mv.scene().render(painter, QRectF(0, 0, tw, th), rect)
    painter.end()
    return qimage_rgb888_to_array(image.convertToFormat(QImage.Format_RGB888))


def _warm(mv, zoom: float) -> None:
    """One paint of the whole map at `zoom`, so every reduced entry it shows exists."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    bound = mv._canvas_item.boundingRect()
    image = QImage(max(1, round(bound.width() * zoom)), max(1, round(bound.height() * zoom)), QImage.Format_RGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    mv.scene().render(painter, QRectF(0, 0, image.width(), image.height()), bound)
    painter.end()


def _tile_centre(mv, x: int, y: int) -> tuple[float, float]:
    c = mv._tile_polygon(x, y).boundingRect().center()
    return c.x(), c.y()


def _plain_blocks(scenario) -> list[tuple[int, int]]:
    """10x10 tile blocks with at most 3 units on them, one terrain id over at
    least 95 of their tiles and no elevation step, emptiest first. old-allies
    has no block without a unit; its plainest are open water."""
    from collections import Counter

    mm = scenario.map_manager
    taken = Counter((int(u.x) // 10, int(u.y) // 10) for units in scenario.unit_manager.units for u in units)
    out = []
    for by in range(2, mm.map_height // 10 - 2):
        for bx in range(2, mm.map_width // 10 - 2):
            if taken[(bx, by)] > 3:
                continue
            tiles = [mm.terrain[(by * 10 + j) * mm.map_width + bx * 10 + i] for j in range(10) for i in range(10)]
            if Counter(t.terrain_id for t in tiles).most_common(1)[0][1] >= 95 and len({t.elevation for t in tiles}) == 1:
                out.append((bx, by))
    out.sort(key=lambda b: (taken[b], b))
    if len(out) < 3:
        raise SystemExit(f"needs 3 plain 10x10 blocks, found {len(out)}")
    return out


def _wall_tile(scenario) -> tuple[int, int]:
    from collections import Counter

    from descape import unit_kind

    walls = unit_kind.wall_consts()
    tiles = [(int(u.x), int(u.y)) for units in scenario.unit_manager.units for u in units if u.unit_const in walls]
    (bx, by), _n = Counter((x // 10, y // 10) for x, y in tiles).most_common(1)[0]
    return bx * 10 + 5, by * 10 + 5


def _forest_edge_tile(scenario) -> tuple[int, int]:
    """The centre of the densest 10x10 tree block with a near-empty 4-neighbour,
    nudged 5 tiles toward that neighbour so the crop straddles the edge."""
    from collections import Counter

    from descape import terrain_palette

    trees = set(terrain_palette.TREE_UNIT_IDS)
    blocks = Counter(
        (int(u.x) // 10, int(u.y) // 10) for units in scenario.unit_manager.units for u in units if u.unit_const in trees
    )
    mm = scenario.map_manager
    best = None
    for (bx, by), _n in sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0])):
        if not (2 <= bx < mm.map_width // 10 - 2 and 2 <= by < mm.map_height // 10 - 2):
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if blocks.get((bx + dx, by + dy), 0) <= 5:
                best = (bx * 10 + 5 + 5 * dx, by * 10 + 5 + 5 * dy)
                break
        if best:
            return best
    raise SystemExit("no forest edge found")


def _edit(window, block: tuple[int, int]) -> None:
    """An L of EDIT_TERRAIN in plain block `block`, through the viewer's repaint funnel."""
    mm = window.scenario.map_manager
    bx, by = block[0] * 10, block[1] * 10
    tiles = [(bx + i, by + j) for j in range(1, 3) for i in range(1, 9)] + [(bx + i, by + j) for j in range(3, 9) for i in range(1, 3)]
    indices = []
    for x, y in tiles:
        index = y * mm.map_width + x
        mm.terrain[index].terrain_id = EDIT_TERRAIN
        indices.append(index)
    window._apply_dirty(indices)


def _expect_k(item, want, what: str, inject: str | None) -> None:
    got = item._last_reduction
    if want == "reduced":
        ok = got == 0 if inject == "k0" else got > 0
    else:
        ok = got == want
    if not ok:
        raise SystemExit(f"{what}: paint ran with k = {got}, expected {want} (inject {inject})")


def _set_style(window, label: str, grid: bool) -> None:
    from PyQt5.QtWidgets import QApplication

    window.terrain_style_combo.setCurrentText(label)
    window.iso_action.setChecked(True)
    if window.grid_action.isChecked() != grid:
        window.grid_action.setChecked(grid)
    for _ in range(5):
        QApplication.processEvents()


def _capture(window, inject: str | None) -> dict[str, np.ndarray]:
    from descape import composite_backend, viewer_canvas

    if composite_backend.native is None:
        raise SystemExit(f"needs the native kernel ({composite_backend.describe()})")
    window.load_scenario(SCENARIO)
    if window.scenario is None:
        raise SystemExit(f"{SCENARIO} failed to load")
    window.distance_ticks_action.setChecked(True)
    mv = window.map_view
    scenario = window.scenario
    plain = _plain_blocks(scenario)
    out: dict[str, np.ndarray] = {}
    base = [(viewer_canvas, "reduction_for", lambda coarsest, scale: 0)]
    with _patched([] if inject is None else [INJECTS[inject]()]):
        _set_style(window, "Stepped", grid=False)
        item = mv._canvas_item
        fit = mv._fit_baseline_scale()
        for zoom in (fit * FAR, fit * MID):
            _warm(mv, zoom)
        chunk_scene = item._cache.chunk_px * item._cache.mip_scale(item._cache.mip_levels()[0])
        wall = _tile_centre(mv, *_wall_tile(scenario))
        forest = _tile_centre(mv, *_forest_edge_tile(scenario))
        # The specificity frame: centred in an even k = 1 chunk column, so no reduced chunk edge is in it.
        k1_chunk = chunk_scene * 2
        m = round((item.boundingRect().center().x() / k1_chunk - 0.5) / 2)
        spec = ((2 * m + 0.5) * k1_chunk, _tile_centre(mv, plain[0][0] * 10 + 5, plain[0][1] * 10 + 5)[1])
        bound = item.boundingRect()
        top = min((_tile_centre(mv, x, y) for x, y in ((0, 0), (239, 0), (0, 239), (239, 239))), key=lambda c: c[1])
        seam_x = round(bound.center().x() / (chunk_scene * 4)) * chunk_scene * 4  # a k = 2 chunk edge
        for fid, centre, pct in (
            ("a1", wall, FAR), ("b1", forest, MID), ("g1", spec, MID),
            ("h1", (seam_x, bound.center().y()), FAR),
            ("e1", (seam_x, top[1] + (seam_x - top[0]) / 2), FAR),
        ):
            out[fid] = _render(mv, centre, fit * pct)
            _expect_k(item, "reduced", fid, inject)
        with _patched(base):
            for fid, centre, pct in (("a2", wall, FAR), ("b2", forest, MID)):
                out[fid] = _render(mv, centre, fit * pct)
                _expect_k(item, 0, fid, inject)
        block = plain[1]
        _edit(window, block)
        edit = _tile_centre(mv, block[0] * 10 + 5, block[1] * 10 + 5)
        out["f1"] = _render(mv, edit, fit * NEAR)
        _expect_k(item, 0, "f1", inject)
        out["f2"] = _render(mv, edit, fit * MID)
        _expect_k(item, "reduced", "f2", inject)

        _set_style(window, "Flat", grid=True)
        item = mv._canvas_item
        fit = mv._fit_baseline_scale()
        for zoom in (fit * FAR, fit * MID):
            _warm(mv, zoom)
        bound = item.boundingRect()
        chunk_scene = item._cache.chunk_px * item._cache.mip_scale(item._cache.mip_levels()[0])
        grid_spot = _tile_centre(mv, plain[2][0] * 10 + 5, plain[2][1] * 10 + 5)
        out["c1"] = _render(mv, grid_spot, fit * FAR)
        _expect_k(item, "reduced", "c1", inject)
        with _patched(base):
            out["c2"] = _render(mv, grid_spot, fit * FAR)
            _expect_k(item, 0, "c2", inject)
        seam_x = round(bound.center().x() / (chunk_scene * 2)) * chunk_scene * 2  # a k = 1 chunk edge
        out["d1"] = _render(mv, (seam_x, bound.center().y()), fit * MID)
        _expect_k(item, "reduced", "d1", inject)
    window.grid_action.setChecked(False)
    return out


_WINDOW = None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(tmp)
        if not SCENARIO.is_file():
            raise SystemExit(f"needs {SCENARIO}")
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(*WINDOW)
        _WINDOW.show()
    return _WINDOW


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    images = _capture(_window(), inject)
    _window().edit_history.mark_saved()
    return images


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a1", "Zoomed far out over a line of walls, angled view."),
            ("a2", "The same crop as a1, rendered a second way."),
            ("b1", "Zoomed out over the edge of a forest, angled view."),
            ("b2", "The same crop as b1, rendered a second way."),
            ("c1", "Zoomed far out over open ground with a tile grid drawn on it, a second map view style."),
            ("c2", "The same crop as c1, rendered a second way."),
            ("d1", "Zoomed out over ground with a tile grid, the second view style."),
            ("e1", "Zoomed far out at the top corner of the map, with the tick strip along its border."),
            ("f1", "Close-up of an area of ground."),
            ("f2", "The same area as f1, zoomed out."),
            ("g1", "Zoomed out over open ground, angled view."),
            ("h1", "Zoomed far out over the middle of the map, angled view."),
        )
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                "First, for each frame, describe what it shows. Then look for a straight line running exactly "
                "vertically (top to bottom) across the frame where the picture is discontinuous: texture, objects "
                "or lines do not continue across it but jump sideways, break, or show a visible straight boundary. "
                "Exclude things that are themselves straight in the map: walls, tile grid lines and the map's own "
                "border, which run diagonally, not vertically. Name each frame where you see such a vertical "
                "line. BREAK if any frame has one, CONTINUOUS if none does."
            ),
            frames=("h1", "d1", "e1", "g1"),
            answers=("BREAK", "CONTINUOUS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                "This frame shows the map's top corner: the map image, its border edge and a strip of short "
                "tick marks outside that edge. First describe the edge and the tick strip on each side of the "
                "corner. Then: along the whole visible border, does the map image's edge keep the same distance "
                "from the tick strip, or does that distance change abruptly at some point (a step where the "
                "image edge jumps closer to or farther from the ticks)? Exclude the corner's own change of "
                "direction. STEADY or STEPPED."
            ),
            frames=("e1",),
            answers=("STEADY", "STEPPED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "First describe the thin lines and sharp edges in each frame (wall outlines, tree edges, grid "
                "lines, edges of buildings). Then: is any such line or edge accompanied by a fainter duplicate "
                "of itself running beside it, offset sideways by roughly one or two enlarged pixel blocks, so "
                "the line looks doubled or smeared wider than a single soft line? Ordinary softness, where a "
                "line simply fades out over about one enlarged pixel block, does not count. Name each frame "
                "where you see it. DOUBLED if any frame has it, SINGLE if none does."
            ),
            frames=("a1", "b1", "c1"),
            answers=("DOUBLED", "SINGLE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                "Three pairs of frames: a1 and a2, b1 and b2, c1 and c2. Each pair is the same crop rendered two "
                "ways. For each pair, first describe the differences you see. Then say which of the two shows "
                "more aliasing: thin lines broken into dashes or dots, jagged stair-steps, or fine texture turned "
                "into speckle or false patterns. Answer SECOND if the second frame of every pair shows more, "
                "FIRST if the first of every pair does, MIXED if the pairs disagree, SAME if no pair shows any "
                "difference."
            ),
            frames=("a1", "a2", "b1", "b2", "c1", "c2"),
            answers=("SECOND", "FIRST", "MIXED", "SAME", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                "f1 and f2 show the same area of the map, f1 closer up. First describe any distinctly coloured "
                "patch of ground in f1 and its shape. Then: does f2 show that same patch, with the same shape, "
                "at its place? SAME if it does, MISSING if f2 does not show it at all, DIFFERENT if f2 shows a "
                "patch of another shape or colour there."
            ),
            frames=("f1", "f2"),
            answers=("SAME", "MISSING", "DIFFERENT", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map render review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game, zoomed out so that the whole "
            "map is small on screen. Each frame is a crop, enlarged by a whole-number factor with no smoothing, "
            "so one rendered pixel is one square block. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_DIR / grp._run_token(PACK_ID, inject, replica)
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
    """Spec coupling, frame band, determinism, and each inject's reach per frame."""
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
    failures.extend(
        f"pair {fid}: the reduced and unreduced renders are identical"
        for fid in ("a", "b", "c")
        if not rp.changed_pixels(clean[f"{fid}1"], clean[f"{fid}2"]).any()
    )
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                diff = int(np.abs(clean[frame.id].astype(int) - injected[frame.id].astype(int)).max())
                touched[frame.id] = (n, rp.largest_component_px(mask), diff)
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max/d{d}" for fid, (px, comp, d) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    # Every inject stays off the twins it does not concern, and shift off the specificity frame.
    for name in ("shift", "ghost", "stale"):
        failures.extend(f"inject {name}: reached base twin {fid}" for fid in ("a2", "b2", "c2", "f1") if fid in reach[name])
    if "g1" in reach["shift"]:
        failures.append("inject shift: reached the specificity frame g1")
    if set(reach[LOCAL_INJECT]) != {"f2"}:
        failures.append(f"inject {LOCAL_INJECT}: reached {sorted(reach[LOCAL_INJECT])}, expected f2 only")
    coarse = reach[COARSE_INJECT]
    if "f2" in coarse and reach[LOCAL_INJECT]["f2"][1] >= coarse["f2"][1]:
        failures.append(f"{LOCAL_INJECT} is not smaller than {COARSE_INJECT} on f2")
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
