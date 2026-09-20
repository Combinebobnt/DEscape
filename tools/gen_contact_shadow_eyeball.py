#!/usr/bin/env python3
"""Renders the Stepped contact shadow (descape/render.py's CONTACT_RAMP_GAIN,
CONTACT_RAMP_MIN, CONTACT_WEDGE_WEIGHT) for a manual A/B eyeball pass --
always writes PNGs, never pass/fail. Modelled on tools/gen_seam_eyeball.py,
and reuses its pyramid fixture, crop and render helpers.

Per elev_step_pct in PCT_VALUES, one contact sheet against the 4-terrace
pyramid: the shipped constants as control in the top-left cell, then every
CONTACT_RAMP_GAIN in GAIN_VALUES x CONTACT_WEDGE_WEIGHT in WEDGE_WEIGHTS.
Each cell is also written on its own. The main sweep captures through a real
offscreen ViewerWindow at the app's default tile_px; --no-viewer, or PyQt5
not importable, falls back to the off-engine composite_rect_iso() render. A
second sweep is always off-engine at tile_px 128, where the old fixed wedge
ramp drew a 3-row barb against a 1-row band junction.

Also prints, per cell, two numbers so the choice is not eyeball-only: the
8-connected component count of the darkened footprint (seam + band + wedge,
measured against a render with CONTACT_SHADE and SEAM_SHADE neutralised to
1.0), where 3 is ideal (one per terrace ring); and single-step coverage of
the neighbour diamond at tile_px 64, whose ceiling is ~35%.

Writes build/contact_shadow_eyeball/, gitignored. No test reads it and there
are no golden images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import iso_geometry as ig
from descape import render
from testkit import settings_isolation

OUT_DIR = ROOT / "build" / "contact_shadow_eyeball"

PCT_VALUES = (10, 25, 50, 100, 200)
GAIN_VALUES = (0.75, 1.0, 1.25)
WEDGE_WEIGHTS = ("taper", "even")
COARSE_TILE_PX = 128
_GAP_PX = 6


def _variants() -> list[tuple[str, float, str]]:
    shipped = ("shipped", render.CONTACT_RAMP_GAIN, render.CONTACT_WEDGE_WEIGHT)
    return [shipped] + [(f"gain{g}_{w}", g, w) for w in WEDGE_WEIGHTS for g in GAIN_VALUES]


def _set_constants(gain: float, weight: str, contact_shade: float, seam_shade: float) -> None:
    render.CONTACT_RAMP_GAIN = gain
    render.CONTACT_WEDGE_WEIGHT = weight
    render.CONTACT_SHADE = contact_shade
    render.SEAM_SHADE = seam_shade
    render._shadow_factors.cache_clear()
    render._seam_factors.cache_clear()


class _Constants:
    """Restores every patched render.py constant and clears both factor
    caches on exit, so a later render never sees a stale variant."""

    def __enter__(self):
        self.saved = (render.CONTACT_RAMP_GAIN, render.CONTACT_WEDGE_WEIGHT, render.CONTACT_SHADE, render.SEAM_SHADE)
        return self

    def __exit__(self, *exc):
        _set_constants(*self.saved)


def _components(mask: np.ndarray) -> int:
    """8-connected component count, iterative flood fill (no scipy here)."""
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        count += 1
        seen[y0, x0] = True
        stack = [(int(y0), int(x0))]
        while stack:
            y, x = stack.pop()
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
    return count


def _coverage(tile_px: int, rise_px: int) -> float:
    """Both bands' still-darkening pixels over one diamond -- the metric
    behind CONTACT_RAMP_GAIN's comment."""
    darkened = 0
    for side in ("up_left", "up_right"):
        if ig.shadow_quad_indices(tile_px, rise_px, side)[0].size:
            darkened += int(np.count_nonzero(render._shadow_factors(tile_px, rise_px, side) < 1.0))
    return darkened / ig.diamond_indices(tile_px)[0].size


def _crop(img: np.ndarray, proj) -> np.ndarray:
    x0, y0, x1, y1 = seam_eyeball._crop_bbox_px(proj, seam_eyeball._ANCHOR, seam_eyeball.PYRAMID_RADIUS, seam_eyeball.PYRAMID_MAX_ELEV)
    return img[y0 : min(y1, img.shape[0]), x0 : min(x1, img.shape[1])]


def _cell_off_engine(scenario, tile_px: int, pct: int, gain: float, weight: str):
    """(crop, darkened mask) for one variant, off-engine."""
    with _Constants() as saved:
        _set_constants(gain, weight, saved.saved[2], saved.saved[3])
        img, proj = seam_eyeball._render_at(scenario, tile_px, pct)
        _set_constants(gain, weight, 1.0, 1.0)
        bare, _ = seam_eyeball._render_at(scenario, tile_px, pct)
    on, off = _crop(img, proj), _crop(bare, proj)
    return on, np.any(on != off, axis=-1)


def _cell_via_viewer(window, pct: int, gain: float, weight: str):
    from PyQt5.QtCore import QRectF

    proj = window.map_view._iso_proj
    if proj is None:
        raise SystemExit("Stepped mode produced no projection")
    x0, y0, x1, y1 = seam_eyeball._crop_bbox_px(proj, seam_eyeball._ANCHOR, seam_eyeball.PYRAMID_RADIUS, seam_eyeball.PYRAMID_MAX_ELEV)
    rect = QRectF(x0, y0, x1 - x0, y1 - y0)
    with _Constants() as saved:
        _set_constants(gain, weight, saved.saved[2], saved.saved[3])
        window._render_current()
        on = seam_eyeball._scene_rect_to_array(window.map_view.scene(), rect)
        _set_constants(gain, weight, 1.0, 1.0)
        window._render_current()
        off = seam_eyeball._scene_rect_to_array(window.map_view.scene(), rect)
    return on, np.any(on != off, axis=-1)


def _sheet(cells: list[np.ndarray], columns: int) -> np.ndarray:
    h = max(c.shape[0] for c in cells)
    w = max(c.shape[1] for c in cells)
    rows = -(-len(cells) // columns)
    sheet = np.full((rows * (h + _GAP_PX), columns * (w + _GAP_PX), 3), 255, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, columns)
        y, x = r * (h + _GAP_PX), c * (w + _GAP_PX)
        sheet[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
    return sheet


def _write_sweep(out_dir: Path, prefix: str, pct: int, render_cell, tile_px: int) -> list[Path]:
    written, cells = [], []
    for name, gain, weight in _variants():
        on, mask = render_cell(pct, gain, weight)
        with _Constants() as saved:
            _set_constants(gain, weight, saved.saved[2], saved.saved[3])
            rise = ig.canvas_size_and_origin(8, 8, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=pct).elev_step
            cov = _coverage(tile_px, rise) if rise > 0 else 0.0
        print(f"{prefix} pct={pct:3d} {name:16s} components={_components(mask):2d} coverage={cov:6.1%}")
        path = out_dir / f"{prefix}_pct{pct}_{name}.png"
        seam_eyeball._save_png(on, path)
        written.append(path)
        cells.append(on)
    # Row 1: shipped, then an empty slot; rows 2-3: taper / even across GAIN_VALUES.
    blank = np.full_like(cells[0], 255)
    layout = [cells[0], blank, blank, *cells[1:]]
    sheet_path = out_dir / f"{prefix}_pct{pct}_sheet.png"
    seam_eyeball._save_png(_sheet(layout, len(GAIN_VALUES)), sheet_path)
    written.append(sheet_path)
    return written


def generate(out_dir: Path, with_viewer: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if with_viewer:
        import tempfile

        # Before any ViewerWindow exists: load_scenario() records a recent
        # file, which would otherwise write to the real config.yaml.
        settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="contact_shadow_eyeball_")))
    written: list[Path] = []
    scenario = seam_eyeball._pyramid_scenario()
    default_px = render.SMALL_MAP_TILE_PIXELS

    for pct in PCT_VALUES:
        if with_viewer:
            from descape import settings

            window = seam_eyeball._stepped_window(pct, settings.GRAPHICS_QUALITY_DEFAULT)
            try:
                written += _write_sweep(
                    out_dir, "viewer", pct, lambda p, g, w, win=window: _cell_via_viewer(win, p, g, w), default_px
                )
            finally:
                window.edit_history.mark_saved()
                window.close()
        else:
            written += _write_sweep(
                out_dir, f"px{default_px}", pct, lambda p, g, w: _cell_off_engine(scenario, default_px, p, g, w), default_px
            )

    for pct in PCT_VALUES:
        written += _write_sweep(
            out_dir,
            f"px{COARSE_TILE_PX}",
            pct,
            lambda p, g, w: _cell_off_engine(scenario, COARSE_TILE_PX, p, g, w),
            COARSE_TILE_PX,
        )

    for tile_px in (default_px, COARSE_TILE_PX):
        for pct in PCT_VALUES:
            written.append(_write_corner_pair(out_dir, tile_px, pct))
    return written


_CORNER_A = (60, 60)
_CORNER_ZOOM = 4


def _inner_corner_scenario():
    """A = (x, y) and B = (x+1, y+1) one level up, so both cast onto
    N = (x+1, y): the smallest inner corner, where the tip pass draws."""
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

    ax, ay = _CORNER_A
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 1 if (tile.x, tile.y) in ((ax, ay), (ax + 1, ay + 1)) else 0
    return scenario


def _write_corner_pair(out_dir: Path, tile_px: int, pct: int) -> Path:
    """Inner-corner vertex with the tip pass on (left) and off (right), at
    _CORNER_ZOOM x. The two must differ only in the receiving tile's two apex
    columns; a straight run is covered byte-identically by the tests."""
    scenario = _inner_corner_scenario()
    on, proj = seam_eyeball._render_at(scenario, tile_px, pct)
    real = ig.shadow_tip_indices
    empty = np.zeros(0, dtype=np.int64)
    ig.shadow_tip_indices = lambda _tp, _r, _side: (empty, empty, empty, empty)
    render._shadow_factors.cache_clear()
    try:
        off, _ = seam_eyeball._render_at(scenario, tile_px, pct)
    finally:
        ig.shadow_tip_indices = real
        render._shadow_factors.cache_clear()
    nx, ny = ig.tile_screen_origin(_CORNER_A[0] + 1, _CORNER_A[1], 0, proj)
    y0, y1 = max(0, ny - proj.half_h), ny + 3 * proj.half_h
    x0, x1 = max(0, nx - proj.half_w), nx + 3 * proj.half_w
    pair = [np.repeat(np.repeat(img[y0:y1, x0:x1], _CORNER_ZOOM, 0), _CORNER_ZOOM, 1) for img in (on, off)]
    changed = int(np.count_nonzero(np.any(on != off, axis=-1)))
    print(f"corner px{tile_px} pct={pct:3d} tip pass changed {changed} px")
    path = out_dir / f"corner_px{tile_px}_pct{pct}.png"
    seam_eyeball._save_png(_sheet(pair, 2), path)
    return path


def check() -> None:
    """--check: Qt-free, writes nothing. One small off-engine cell must
    darken something, and every patched constant must be restored after."""
    before = (render.CONTACT_RAMP_GAIN, render.CONTACT_WEDGE_WEIGHT, render.CONTACT_SHADE, render.SEAM_SHADE)
    scenario = seam_eyeball._pyramid_scenario()
    on, mask = _cell_off_engine(scenario, 16, 100, GAIN_VALUES[-1], WEDGE_WEIGHTS[0])
    failures = []
    if on.size == 0 or not mask.any():
        failures.append("the shadow darkened nothing against its no-op control")
    if _components(mask) < 1:
        failures.append("component count is zero on a non-empty mask")
    after = (render.CONTACT_RAMP_GAIN, render.CONTACT_WEDGE_WEIGHT, render.CONTACT_SHADE, render.SEAM_SHADE)
    if after != before:
        failures.append(f"render constants not restored: {before} -> {after}")
    if failures:
        raise SystemExit("; ".join(failures))
    print("OK: contact shadow eyeball cell renders, differs from its no-op control, and restores constants")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--no-viewer", action="store_true", help="Skip the real offscreen ViewerWindow capture")
    parser.add_argument("--check", action="store_true", help="Validate without writing any files")
    args = parser.parse_args()

    if args.check:
        check()
        return

    with_viewer = not args.no_viewer and seam_eyeball.PYQT5_AVAILABLE
    if not args.no_viewer and not seam_eyeball.PYQT5_AVAILABLE:
        print("PyQt5 not importable -- falling back to a Qt-free run (same as --no-viewer)")

    for path in generate(args.out_dir, with_viewer):
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
