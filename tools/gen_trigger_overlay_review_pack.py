#!/usr/bin/env python3
"""Review pack for the trigger map overlay (GH #41): does a trigger area's
outline sit on the tiles the area names, in Stepped and Sloped, where the
ground under its edge is raised?

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_selection_review_pack.py, whose shape it follows:
the overlay is MapView scene items, not chunk pixels, so this drives a real
offscreen ViewerWindow and rasterizes scene rects 1:1 with testkit.qt_capture.
The named tiles are painted a third terrain by the scenario itself, so the
reviewer compares the outline against ground the overlay code never touched.
Needs a configured AoE2:DE install (terrain textures).

Run:
    tools/gen_trigger_overlay_review_pack.py
    tools/gen_trigger_overlay_review_pack.py --replica 1
    tools/gen_trigger_overlay_review_pack.py --inject quad
    tools/gen_trigger_overlay_review_pack.py --check      # writes nothing

Writes build/review_pack/trigger_overlay/<opaque token>/, gitignored.
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

PACK_ID = "trigger_overlay"
OUT_DIR = grp.OUT_DIR
# DESERT_SAND / GRASS_1 per-tile checkerboard (GOTCHAS), the named tiles SNOW.
CHECKER = (14, 0)
PATCH = 32
# Area A (inclusive corners) crosses onto a plateau: its right half is raised.
AREA_A = (50, 52, 59, 59)
PLATEAU = (55, 44, 66, 67, 3)  # x0, y0, x1, y1 inclusive, elevation
# Area B lies on flat ground: the specificity control for the elevation defect.
AREA_B = (20, 80, 27, 85)
_STYLES = {"stepped": "Stepped", "sloped": "Sloped"}


def _elevation(x: int, y: int) -> int:
    x0, y0, x1, y1, level = PLATEAU
    return level if x0 <= x <= x1 and y0 <= y <= y1 else 0


def _inside(area, x: int, y: int) -> bool:
    x1, y1, x2, y2 = area
    return x1 <= x <= x2 and y1 <= y <= y2


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in loaded.map_manager.terrain:
        tile.elevation = _elevation(tile.x, tile.y)
        patch = _inside(AREA_A, tile.x, tile.y) or _inside(AREA_B, tile.x, tile.y)
        tile.terrain_id = PATCH if patch else CHECKER[(tile.x + tile.y) % 2]
    out = tmp / "areas.aoe2scenario"
    write_scenario(loaded, out, backup=False)
    return out


def _shapes():
    from descape.trigger_geometry import AREA_FIELDS, TriggerShape

    return [
        TriggerShape(0, "effect", 0, "area", AREA_FIELDS, AREA_A),
        TriggerShape(0, "effect", 1, "area", AREA_FIELDS, AREA_B),
    ]


# ----------------------------------------------------------------- injects


def _inject_quad():
    """The plan's rejected design: one straight quad through the ring's four
    outer corners, ignoring the elevation along each edge."""
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPainterPath

    from descape.map_view import MapView

    real = MapView._rect_ring_path

    def quad(self, rect):
        ring = real(self, rect)
        if ring.isEmpty():
            return ring
        tx0, ty0, tx1, ty1 = rect
        corners = []
        for tx, ty, side, index in (
            (tx0, ty0, "up_left", 0), (tx1 - 1, ty0, "up_right", 0),
            (tx1 - 1, ty1 - 1, "right", 0), (tx0, ty1 - 1, "left", 0),
        ):
            points = self._tile_edge_points(tx, ty, side)
            # A copy: an indexed QPolygonF point dies with its temporary polygon.
            corners.append(QPointF(points[index]))
        path = QPainterPath()
        path.moveTo(corners[0])
        for point in corners[1:]:
            path.lineTo(point)
        path.closeSubpath()
        return path

    return MapView, "_rect_ring_path", quad


def _inject_offset():
    """Every area drawn two tiles further along both +x and +y."""
    from descape.map_view import MapView

    real = MapView._rect_ring_path

    def shifted(self, rect):
        tx0, ty0, tx1, ty1 = rect
        return real(self, (tx0 + 2, ty0 + 2, tx1 + 2, ty1 + 2))

    return MapView, "_rect_ring_path", shifted


def _inject_short():
    """The inclusive far x corner read as exclusive: one tile short on that one side."""
    from descape.map_view import MapView

    real = MapView._rect_ring_path

    def short(self, rect):
        tx0, ty0, tx1, ty1 = rect
        return real(self, (tx0, ty0, max(tx0 + 1, tx1 - 1), ty1))

    return MapView, "_rect_ring_path", short


INJECTS = {"quad": _inject_quad, "offset": _inject_offset, "short": _inject_short}
COARSE_INJECT = "offset"
LOCALIZED_INJECT = "short"


@contextmanager
def _patched(inject: str | None):
    patches = [] if inject is None else [INJECTS[inject]()]
    saved = [(obj, name, obj.__dict__[name]) for obj, name, _new in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


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


def _area_bounds(mv, area):
    """Scene bounds of the area's own tiles, from the overlay-independent tile polygons."""
    x1, y1, x2, y2 = area
    bounds = None
    for tx, ty in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        rect = mv._tile_polygon(tx, ty).boundingRect()
        bounds = rect if bounds is None else bounds.united(rect)
    return bounds


def _capture(window, inject: str | None) -> dict[str, np.ndarray]:
    mv = window.map_view
    out: dict[str, np.ndarray] = {}
    with _patched(inject):
        for style, label in _STYLES.items():
            window.terrain_style_combo.setCurrentText(label)
            mv.show_trigger_overlay(_shapes(), None)
            frames = (("a", AREA_A),) if style == "sloped" else (("a", AREA_A), ("b", AREA_B))
            for suffix, area in frames:
                b = _area_bounds(mv, area)
                # Tall headroom above: the plateau lifts the far edge.
                pad = b.width() * 0.12
                rect = _int_rect(b.left() - pad, b.top() - pad * 2.5, b.right() + pad, b.bottom() + pad)
                out[f"{style}_{suffix}"] = _grab(mv, rect)
    mv.clear_trigger_overlay()
    return out


_WINDOW = None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="trigger_overlay_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
        _WINDOW.load_scenario(_scenario_file(tmp))
        if _WINDOW.scenario is None:
            raise SystemExit("the generated area scenario failed to load")
    return _WINDOW


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    return _capture(_window(), inject)


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("stepped_a", "A patch of pale ground on a checkered map, seen at an angle; some ground is raised."),
            ("sloped_a", "The same patch in a second map view style with smooth slopes."),
            ("stepped_b", "A second patch of pale ground on a checkered map, seen at an angle."),
        )
    )
    ground = (
        "The ground is a checkerboard of two alternating ground textures, one cell per map tile, "
        "except for a rectangular patch of pale ground. A green line is drawn over the map, with a "
        "faint green tint inside it. "
    )
    exclusion = (
        "Ignore the faint green tint and the line's own thickness. A departure means the green line "
        "leaves the patch's edge by a visible gap, runs through the checkered ground outside the "
        "patch, cuts across the pale patch itself, or skips a step where the edge climbs onto raised "
        "ground. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                ground + "In each frame, first describe where the green line runs relative to the pale "
                "patch's edge along each of the patch's four sides, including where the edge crosses "
                "raised ground. Then: does the green line run along the pale patch's edge on all four "
                "sides in EVERY frame? " + exclusion + "FOLLOWS if it does everywhere, DEPARTS if it "
                "leaves the edge anywhere."
            ),
            frames=("stepped_a", "sloped_a"),
            answers=("FOLLOWS", "DEPARTS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                ground + "First describe where the green line runs relative to the pale patch's edge "
                "along each of its four sides. Then: does the green line run along the pale patch's "
                "edge on all four sides? " + exclusion + "FOLLOWS or DEPARTS."
            ),
            frames=("stepped_b",),
            answers=("FOLLOWS", "DEPARTS", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map outline review",
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
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    if "stepped_b" in reach["quad"]:
        failures.append("inject quad reached stepped_b, the flat-ground control")
    failures.extend(f"inject quad did not reach {fid}" for fid in ("stepped_a", "sloped_a") if fid not in reach["quad"])
    coarse, local = reach[COARSE_INJECT], reach[LOCALIZED_INJECT]
    shared = [fid for fid in local if fid in coarse]
    if not shared:
        failures.append(f"{LOCALIZED_INJECT} shares no frame with {COARSE_INJECT}")
    failures.extend(
        f"{LOCALIZED_INJECT} is not smaller than {COARSE_INJECT} on {fid}"
        for fid in shared
        if local[fid][1] >= coarse[fid][1]
    )
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
