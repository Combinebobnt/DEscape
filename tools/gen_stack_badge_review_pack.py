#!/usr/bin/env python3
"""Review pack for GH #100's stacked-unit badge: its position on the tile
(Settings > Appearance, default bottom-right) and its background colour.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), shaped
like tools/gen_selection_review_pack.py: the badge is a MapView scene item, so
this drives a real offscreen ViewerWindow and crops the viewport, which is
where the badge's device-space box is drawn. The ground is a per-tile
two-tone checkerboard so a box's own tile is visible.

Run:
    tools/gen_stack_badge_review_pack.py
    tools/gen_stack_badge_review_pack.py --replica 1
    tools/gen_stack_badge_review_pack.py --inject old-anchor
    tools/gen_stack_badge_review_pack.py --check      # writes nothing

Writes build/review_pack/stack_badge/<opaque token>/, gitignored.
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

PACK_ID = "stack_badge"
OUT_DIR = grp.OUT_DIR
VILLAGER = 83
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1, the marker pack's pair
# A lone 2-stack, and three 2-stacks on touching tiles (the Castle-footprint case).
# Each trio stack has its own owner, so its cell-covering mark has its own colour.
_LONE = (40, 41)
_TRIO = [(60, 61), (61, 61), (60, 62)]
_TRIO_OWNERS = (1, 2, 6)
_CUSTOM_BACKGROUND = "#c02040"
# Device-pixel crop, the same for every frame so every frame gets the same upscale.
_CROP_W, _CROP_H = 360, 270
# frame id -> (style, iso view, position, background, badges on, target tiles)
_FRAMES = {
    "a_flat": ("Flat", False, "bottom_right", "#000000", True, [_LONE]),
    "a_iso": ("Flat", True, "bottom_right", "#000000", True, [_LONE]),
    "a_stepped": ("Stepped", False, "bottom_right", "#000000", True, [_LONE]),
    "a_sloped": ("Sloped", False, "bottom_right", "#000000", True, [_LONE]),
    "b_stepped": ("Stepped", False, "bottom_right", "#000000", True, _TRIO),
    "b_flat": ("Flat", False, "bottom_right", "#000000", True, _TRIO),
    "c_iso": ("Flat", True, "above", "#000000", True, [_LONE]),
    "d_one": ("Stepped", False, "above", "#000000", True, [_LONE]),
    "d_two": ("Stepped", False, "above", _CUSTOM_BACKGROUND, True, [_LONE]),
    "ground": ("Stepped", False, "bottom_right", "#000000", False, [_LONE]),
}


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.terrain_id = CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    for owner, (tx, ty) in zip((1, *_TRIO_OWNERS), [_LONE, *_TRIO], strict=True):
        for _ in range(2):
            model.add(owner, VILLAGER, tx + 0.5, ty + 0.5)
    out = tmp / "stacks.aoe2scenario"
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects


def _inject_old_anchor():
    """The pre-GH #100 badge: bounding-rect top-centre, box above it, whatever the setting."""
    from PyQt5.QtCore import QPointF

    from descape.map_view import MapView
    from descape.viewer_canvas import StackBadgeItem

    def old(self, tile_x, tile_y):
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return None
        rect = polygon.boundingRect()
        return QPointF(rect.center().x(), rect.top()), rect.center()

    real_placement = StackBadgeItem.set_placement
    return [
        (MapView, "_stack_badge_anchor", old),
        (StackBadgeItem, "set_placement", lambda self, tucked: real_placement(self, False)),
    ]


def _inject_vertex():
    """Iso tucked anchors on the side's first point (a vertex four tiles share), not its midpoint."""
    from PyQt5.QtCore import QPointF

    from descape.map_view import _ELEVATED_STYLES, MapView

    real = MapView._stack_badge_anchor

    def vertex(self, tile_x, tile_y):
        got = real(self, tile_x, tile_y)
        iso = self._isometric or self._terrain_style in _ELEVATED_STYLES
        if got is None or not iso or self._stack_badge_position == "above":
            return got
        points = self._tile_edge_points(tile_x, tile_y, self._STACK_BADGE_SIDES[self._stack_badge_position])
        return QPointF(points[0]), got[1]

    return [(MapView, "_stack_badge_anchor", vertex)]


def _inject_style_size():
    """Sloped draws a larger badge than the other views."""
    from descape.map_view import MapView

    real = MapView._rebuild_stack_badges

    def sized(self):
        real(self)
        item = self._stack_badge_item
        if item is not None:
            item._font.setPixelSize(item.FONT_PX * 3 // 2 if self._terrain_style == "sloped" else item.FONT_PX)
            item.update()

    return [(MapView, "_rebuild_stack_badges", sized)]


def _inject_above_edge():
    """`above` in an iso view anchors on the upper-left edge's midpoint, not the top vertex."""
    from descape.map_view import _ELEVATED_STYLES, MapView

    real = MapView._stack_badge_anchor

    def edge(self, tile_x, tile_y):
        got = real(self, tile_x, tile_y)
        iso = self._isometric or self._terrain_style in _ELEVATED_STYLES
        if got is None or not iso or self._stack_badge_position != "above":
            return got
        points = self._tile_edge_points(tile_x, tile_y, "up_left")
        return (points[0] + points[-1]) / 2.0, got[1]

    return [(MapView, "_stack_badge_anchor", edge)]


def _inject_opaque_custom():
    """A non-black background paints fully opaque."""
    from PyQt5.QtGui import QColor

    from descape.viewer_canvas import StackBadgeItem

    real = StackBadgeItem.set_background_color

    def opaque(self, color):
        real(self, color)
        if (color.red(), color.green(), color.blue()) != (0, 0, 0):
            self._background = QColor(color.red(), color.green(), color.blue(), 255)

    return [(StackBadgeItem, "set_background_color", opaque)]


INJECTS = {
    "old-anchor": _inject_old_anchor,
    "vertex": _inject_vertex,
    "style-size": _inject_style_size,
    "above-edge": _inject_above_edge,
    "opaque-custom": _inject_opaque_custom,
}
COARSE_INJECT = "old-anchor"


@contextmanager
def _patched(inject: str | None):
    patches = [] if inject is None else INJECTS[inject]()
    saved = [(obj, name, obj.__dict__[name]) for obj, name, _new in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


# ----------------------------------------------------------------- capture


def _show(window, frame_id: str) -> None:
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    style, iso, position, background, badges_on, tiles = _FRAMES[frame_id]
    if style == "Flat":
        window.iso_action.setChecked(iso)
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Units")
    window.stack_badges_action.setChecked(badges_on)
    mv = window.map_view
    settings.set_overlay_color("unit_stack_background", background)
    mv.apply_overlay_colors()
    # Rebuilds the anchors under whichever inject is live.
    mv.set_stack_badge_position(position)
    rect = QRectF()
    for tx, ty in tiles:
        for dx in (-2, 2):
            for dy in (-2, 2):
                rect = rect.united(mv._tile_polygon(tx + dx, ty + dy).boundingRect())
    mv.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _grab(window, frame_id: str) -> np.ndarray:
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication

    from testkit.qt_capture import qimage_rgb888_to_array

    _show(window, frame_id)
    mv = window.map_view
    tiles = _FRAMES[frame_id][5]
    centres = [mv._tile_polygon(*t).boundingRect().center() for t in tiles]
    mid = QPointF(sum(c.x() for c in centres) / len(centres), sum(c.y() for c in centres) / len(centres))
    view = mv.viewportTransform().map(mid)
    QApplication.processEvents()
    image = mv.viewport().grab().toImage().convertToFormat(QImage.Format_RGB888)
    array = qimage_rgb888_to_array(image)
    x0 = round(view.x()) - _CROP_W // 2
    y0 = round(view.y()) - _CROP_H // 2
    assert x0 >= 0 and y0 >= 0 and x0 + _CROP_W <= array.shape[1] and y0 + _CROP_H <= array.shape[0], frame_id
    return np.ascontiguousarray(array[y0 : y0 + _CROP_H, x0 : x0 + _CROP_W])


_WINDOW = None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="stack_badge_pack_"))
        settings_isolation.pin_install_path()
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
        _WINDOW.load_scenario(_scenario_file(tmp))
        if _WINDOW.scenario is None:
            raise SystemExit("the generated stacks scenario failed to load")
        # Coloured marks, not sprites: a standing sprite would cover the box's own ground.
        _WINDOW.show_sprites_action.setChecked(False)
    return _WINDOW


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    window = _window()
    with _patched(inject):
        return {frame_id: _grab(window, frame_id) for frame_id in _FRAMES}


# --------------------------------------------------------------- the spec


_CHECKER = (
    "The ground is a checkerboard of two alternating ground colours, one cell per map tile "
    "(diamond-shaped cells in angled views, squares in top-down ones). Flat single-colour shapes in "
    "other colours are map objects, each one covering exactly the cell it stands on. "
)


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a_flat", "A small numbered box by a map object, seen straight from above."),
            ("a_iso", "The same, in an angled view."),
            ("a_stepped", "The same, in a second angled view."),
            ("a_sloped", "The same, in a third angled view."),
            ("b_stepped", "Three map objects on neighbouring cells, each with a numbered box, angled view."),
            ("b_flat", "The same three, seen straight from above."),
            ("c_iso", "A numbered box by a map object, angled view."),
            ("d_one", "A numbered box by a map object, angled view."),
            ("d_two", "The same, with a differently coloured box."),
            ("ground", "Checkered ground with a map object, angled view."),
        )
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                _CHECKER + "Each frame has one small box with a number in it. First describe, for each frame, "
                "which cell the box overlaps and where within that cell it sits. Then: in EVERY frame, is the "
                "box wholly inside one cell, pressed against that cell's lower-right side (the lower-right edge "
                "of a diamond, the bottom-right corner of a square)? LOWER-RIGHT if so, ELSEWHERE if in any "
                "frame the box crosses a cell boundary or sits against a different side or corner."
            ),
            frames=("a_flat", "a_iso", "a_stepped", "a_sloped"),
            answers=("LOWER-RIGHT", "ELSEWHERE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                _CHECKER + "Three map objects stand on three neighbouring cells, and there are three small "
                "numbered boxes. First describe, for each box, which coloured cell or cells it overlaps. Then: does each "
                "box lie wholly inside a single cell, with no two boxes in the same cell? DISTINCT if so, "
                "MIXED if any box crosses a cell boundary or two boxes share a cell."
            ),
            frames=("b_stepped", "b_flat"),
            answers=("DISTINCT", "MIXED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "Each frame is enlarged by the same factor and has one small box with a number in it. First "
                "estimate each box's width and height in pixels. Then: are all four boxes the same size, "
                "within a few pixels? SAME-SIZE or DIFFERENT-SIZE. Judge the box, not the object near it."
            ),
            frames=("a_flat", "a_iso", "a_stepped", "a_sloped"),
            answers=("SAME-SIZE", "DIFFERENT-SIZE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                _CHECKER + "There is one small numbered box and one map object. First describe where the "
                "box sits relative to the corners of the diamond cell the map object covers. Then: is the box "
                "centred directly above that diamond's TOP corner, its bottom just above the corner? TOP-CORNER, "
                "or ELSEWHERE if it is centred over an edge, a side corner or anywhere else."
            ),
            frames=("c_iso",),
            answers=("TOP-CORNER", "ELSEWHERE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                _CHECKER + "Each frame has one small coloured box with a number, lying across the boundary "
                "between two cells of different colours. First describe, for each box, whether and how the two "
                "ground colours can be told apart through the box's fill. Then: does the ground show through "
                "both boxes about equally? SAME if so, DIFFERENT if one box lets the ground show and the other "
                "is a flat solid colour that hides it."
            ),
            frames=("d_one", "d_two"),
            answers=("SAME", "DIFFERENT", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q6",
            question=(
                "First describe everything drawn in this frame. Then: is there a small box with a number in it "
                "anywhere in the frame? BOX or NO-BOX."
            ),
            frames=("ground",),
            answers=("BOX", "NO-BOX", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map number box review",
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
        if "ground" in touched:
            failures.append(f"inject {name}: reached the bare-ground frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[COARSE_INJECT]
    smaller = [
        (name, fid)
        for name, t in reach.items() if name != COARSE_INJECT
        for fid, (_px, comp) in t.items() if fid in coarse and comp < coarse[fid][1]
    ]
    if not smaller:
        failures.append(f"no inject is smaller than {COARSE_INJECT} on a shared frame")
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
