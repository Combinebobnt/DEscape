#!/usr/bin/env python3
"""Renders the terrace-edge seam line
(descape/iso_geometry.py's seam_edge_indices/seam_apex_indices,
descape/render.py's SEAM_SHADE) for a manual eyeball pass -- always writes
PNGs, never pass/fail. Modelled on tools/gen_elevation_reference.py's
--check style and tools/gen_iso_reference_pngs.py's "always writes, never
pass/fail" contract.

Per pct (25/50/100/200) a triptych against a 4-terrace pyramid centred on
_ANCHOR: seam on, seam off (render.SEAM_SHADE neutralised to 1.0, the same
no-baseline A/B tests/test_seam_line.py and tests/test_seam_viewer.py both
already use), and an amplified difference. The main sweep captures through
a real offscreen ViewerWindow (QT_QPA_PLATFORM=offscreen) at the app's own
default tile_px -- "the actual paint dispatch a user sees"
(tests/test_seam_viewer.py's own phrase), not just this module's compositor
in isolation. --no-viewer, or PyQt5 simply not being importable, falls back
to the same off-engine descape.render.composite_rect_iso() primitive the
coarse-mip sweep below always uses.

A second sweep covers tile_px 8 and 16 explicitly across the same pct
values -- the coarse mips are where the apex gap was visible and where
nothing has ever been eyeballed.
Always off-engine, never through a live zoomed viewer: 8 sits below
settings.GRAPHICS_QUALITY_MIN's own floor of 16 on this map size (a 120x120
map only ever reaches tile_px 16 via graphics_quality=1, "Potatest"), and
driving Qt zoom to land on a specific mip level is exactly the fragile
dance tests/test_mip_viewer.py exists to pin down precisely -- not worth
reproducing here when composite_rect_iso() is, byte-identically, the exact
function IsoChunkCache.render_rect() calls per mip chunk (confirmed while
root-causing the map-extent-outline bug during commit 2).

Writes build/seam_eyeball/, gitignored. No test reads it. No golden images
anywhere -- there is no baseline to compare against, only the seam-on vs
seam-off A/B each triptych already carries.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import iso_geometry as ig
from descape import render
from descape import settings
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import qt_capture

OUT_DIR = ROOT / "build" / "seam_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

PCT_VALUES = (25, 50, 100, 200)
MIP_TILE_PX_VALUES = (8, 16)

# Matches tests/test_seam_viewer.py's own _ANCHOR=(60, 60), not tests/
# test_seam_line.py's (10, 10) -- MapView.set_source() draws a decorative
# "map extent" outline (iso_geometry.ground_outline_corners(), a Qt-scene-
# only polygon) that render_terrain_iso() never draws, and a probe near
# (10, 10) on this same 120x120 template sits close enough to the map's own
# (0, 0) corner for that outline's edge to cut through it. See that file's
# own docstring for the measured 161-stray-pixel confirmation.
_ANCHOR = (60, 60)
PYRAMID_RADIUS = 3
PYRAMID_MAX_ELEV = 3  # step=1 * PYRAMID_RADIUS rings

_QAPP = None


def _apply_pyramid(scenario, center=_ANCHOR, step: int = 1) -> None:
    """Same shape as tests/test_seam_line.py's _pyramid_scenario / tests/
    test_seam_viewer.py's _set_pyramid, applied in place to an
    already-loaded scenario so both the off-engine and through-viewer
    paths below build the identical fixture from the identical source."""
    cx, cy = center
    for tile in scenario.map_manager.terrain:
        rings = max(0, PYRAMID_RADIUS - max(abs(tile.x - cx), abs(tile.y - cy)))
        tile.elevation = min(ig.MAX_ELEVATION, step * rings)


def _pyramid_scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    _apply_pyramid(scenario)
    return scenario


def _crop_bbox_px(proj, center, radius: int, max_elev: int, pad_tiles: int = 1):
    """Screen-pixel crop bounds covering every tile within `radius` of
    `center` at both elevation 0 and max_elev, plus skirt/contact-shadow
    headroom below -- same technique as tools/gen_elevation_reference.py's
    own _crop_bbox_px and tests/test_seam_viewer.py's _bbox_for_tiles."""
    cx, cy = center
    xs, ys = [], []
    for tx in range(cx - radius, cx + radius + 1):
        for ty in range(cy - radius, cy + radius + 1):
            for e in (0, max_elev):
                sx, sy = ig.tile_screen_origin(tx, ty, e, proj)
                xs += [sx, sx + 2 * proj.half_w]
                ys += [sy, sy + 2 * proj.half_h]
    ys.append(max(ys) + max_elev * proj.elev_step)
    pad = pad_tiles * proj.tile_px
    return max(0, min(xs) - pad), max(0, min(ys) - pad), max(xs) + pad, max(ys) + pad


def _amplify_diff(a: np.ndarray, b: np.ndarray, factor: int = 4) -> np.ndarray:
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * factor, 0, 255).astype(np.uint8)


def _render_at(scenario, tile_px: int, elev_step_pct: int):
    """Off-engine full-canvas render at an explicit (tile_px, elev_step_pct)
    pair, decoupled from settings.get_graphics_quality()/get_elev_step_pct()
    entirely -- see the module docstring for why composite_rect_iso() is a
    legitimate stand-in for a live zoomed mip render."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    proj = ig.canvas_size_and_origin(
        w, h, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=elev_step_pct
    )
    elevations = np.zeros((h, w), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    skirt_headroom = (ig.MAX_ELEVATION - ig.MIN_ELEVATION) * proj.elev_step
    img = render.composite_rect_iso(
        scenario,
        0,
        0,
        proj.canvas_w,
        proj.canvas_h + skirt_headroom,
        elevations,
        proj,
        tile_px,
        units_by_tile={},
        building_bboxes={},
        with_units=False,
    )
    return img, proj


def _triptych_off_engine(scenario, tile_px: int, elev_step_pct: int, crop_radius: int = PYRAMID_RADIUS):
    original = render.SEAM_SHADE
    try:
        with_seam, proj = _render_at(scenario, tile_px, elev_step_pct)
        render.SEAM_SHADE = 1.0
        render._seam_factors.cache_clear()
        without_seam, _ = _render_at(scenario, tile_px, elev_step_pct)
    finally:
        render.SEAM_SHADE = original
        render._seam_factors.cache_clear()

    x0, y0, x1, y1 = _crop_bbox_px(proj, _ANCHOR, crop_radius, PYRAMID_MAX_ELEV)
    x1, y1 = min(x1, with_seam.shape[1]), min(y1, with_seam.shape[0])
    on = with_seam[y0:y1, x0:x1]
    off = without_seam[y0:y1, x0:x1]
    return on, off, _amplify_diff(on, off)


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


# 1:1 ONLY -- see testkit.qt_capture for why any other target size silently
# degrades exactness to a near-match.
_scene_rect_to_array = qt_capture.scene_rect_to_array


def _stepped_window(elev_step_pct: int, graphics_quality: int):
    """Real, shown ViewerWindow with the pyramid fixture loaded in Stepped
    mode, offscreen. Local duplicate of tests/conftest.py's own
    stepped_window(), for the same reason _scene_rect_to_array() is.

    Settings are pinned via the MODULE GLOBALS directly, not
    settings.set_elev_step_pct()/set_graphics_quality() -- unlike a pytest
    run there is no autouse fixture redirecting CONFIG_PATH here, so the
    real setters would write straight through to this developer's own
    config.yaml (this happened once already, to an ad-hoc debug
    snippet)."""
    _ensure_qapp()
    import descape.settings as settings_module
    from descape.viewer import ViewerWindow
    from PyQt5.QtWidgets import QApplication

    settings_module._elev_step_pct = elev_step_pct
    settings_module._graphics_quality = graphics_quality

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit(f"{BLANK_TEMPLATE_PATH.name} failed to load")
    _apply_pyramid(window.scenario)
    window._render_current()
    window.show()
    QApplication.processEvents()
    return window


def _triptych_via_viewer(elev_step_pct: int, crop_radius: int = PYRAMID_RADIUS):
    from PyQt5.QtCore import QRectF

    original = render.SEAM_SHADE
    window = _stepped_window(elev_step_pct, settings.GRAPHICS_QUALITY_DEFAULT)
    try:
        proj = window.map_view._iso_proj
        if proj is None:
            raise SystemExit("Stepped mode produced no projection")
        x0, y0, x1, y1 = _crop_bbox_px(proj, _ANCHOR, crop_radius, PYRAMID_MAX_ELEV)
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)

        on = _scene_rect_to_array(window.map_view.scene(), rect)

        render.SEAM_SHADE = 1.0
        render._seam_factors.cache_clear()
        window._render_current()
        off = _scene_rect_to_array(window.map_view.scene(), rect)
    finally:
        render.SEAM_SHADE = original
        render._seam_factors.cache_clear()
        window.edit_history.mark_saved()
        window.close()
    return on, off, _amplify_diff(on, off)


def _grabs(out_dir: Path) -> list[Path]:
    """Full-window ViewerWindow.grab() and SettingsDialog.grab(), offscreen
    -- chrome included, not a crop, since these exist to show the feature
    in its real UI context rather than to isolate the seam itself."""
    from descape.viewer import SettingsDialog

    written = []
    window = _stepped_window(100, settings.GRAPHICS_QUALITY_DEFAULT)
    try:
        window_path = out_dir / "viewer_window.png"
        window.grab().save(str(window_path))
        written.append(window_path)

        dialog = SettingsDialog(window)
        try:
            dialog_path = out_dir / "settings_dialog.png"
            dialog.grab().save(str(dialog_path))
            written.append(dialog_path)
        finally:
            dialog.close()
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _save_png(img: np.ndarray, out_path: Path) -> None:
    from PIL import Image

    Image.fromarray(img, mode="RGB").save(out_path)


def _write_triptych(out_dir: Path, prefix: str, on: np.ndarray, off: np.ndarray, diff: np.ndarray) -> list[Path]:
    written = []
    for suffix, img in (("seam_on", on), ("seam_off", off), ("diff", diff)):
        path = out_dir / f"{prefix}_{suffix}.png"
        _save_png(img, path)
        written.append(path)
    return written


def generate(out_dir: Path, with_viewer: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for pct in PCT_VALUES:
        if with_viewer:
            on, off, diff = _triptych_via_viewer(pct)
        else:
            on, off, diff = _triptych_off_engine(_pyramid_scenario(), render.SMALL_MAP_TILE_PIXELS, pct)
        written += _write_triptych(out_dir, f"pct{pct}", on, off, diff)

    scenario = _pyramid_scenario()
    for tile_px in MIP_TILE_PX_VALUES:
        for pct in PCT_VALUES:
            on, off, diff = _triptych_off_engine(scenario, tile_px, pct)
            written += _write_triptych(out_dir, f"mip{tile_px}_pct{pct}", on, off, diff)

    if with_viewer:
        written += _grabs(out_dir)

    return written


def check() -> None:
    """--check: Qt-free correctness pass, no PyQt5/ViewerWindow at all.
    Commit 2 left only ~7s of this plan's
    <=20s default-tier budget for commit 3, and tests/test_seam_viewer.py
    already covers the through-viewer path byte-identically -- so this
    renders one small triptych off-engine and asserts the seam actually
    painted something the no-op control didn't."""
    failures = 0
    on, off, diff = _triptych_off_engine(_pyramid_scenario(), tile_px=16, elev_step_pct=100, crop_radius=2)

    if on.shape != off.shape or on.shape != diff.shape:
        print(f"FAIL: triptych shape mismatch {on.shape} vs {off.shape} vs {diff.shape}")
        failures += 1
    if np.array_equal(on, off):
        print("FAIL: seam-on and seam-off renders are identical -- the seam painted nothing")
        failures += 1
    if not diff.any():
        print("FAIL: amplified diff is all-zero")
        failures += 1

    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print("OK: seam eyeball triptych renders and differs from its no-op control")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument(
        "--no-viewer", action="store_true", help="Skip the real offscreen ViewerWindow capture (Qt-free run)"
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing any files")
    args = parser.parse_args()

    if args.check:
        check()
        return

    with_viewer = not args.no_viewer and PYQT5_AVAILABLE
    if not args.no_viewer and not PYQT5_AVAILABLE:
        print("PyQt5 not importable -- falling back to a Qt-free run (same as --no-viewer)")

    written = generate(args.out_dir, with_viewer)
    for path in written:
        # relative_to(ROOT) raises for an --out-dir outside the repo -- every
        # PNG is already written by this point, so fall back to the absolute
        # path rather than crash on reporting after the real work succeeded.
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
