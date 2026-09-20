#!/usr/bin/env python3
"""Screenshots View > Grid for a manual eyeball pass -- always writes PNGs,
never pass/fail. Modelled on tools/gen_stack_badge_eyeball.py, including the
settings-isolation trap testkit.settings_isolation exists for.

Two fixtures, because the two halves of the feature need different terrain:

- the blank template, for the ground-plane lattice in Flat (with and without
  Isometric View), Stepped and Sloped, at the darkest and lightest stops, plus
  one fit-to-map capture where the LOD ladder has dropped the minors;
- an elevated `examples/` scenario, for Follow Terrain Elevation on vs off in
  Stepped and Sloped. Draping is invisible on flat ground, so the first
  fixture cannot show it at all.

Writes build/grid_eyeball/, gitignored. No test reads it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from testkit import settings_isolation

OUT_DIR = ROOT / "build" / "grid_eyeball"
ELEVATED_SCENARIO = ROOT / "examples" / "2_Joan_coop_2_v0_15.aoe2scenario"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

# Tile rects the captures frame: a close look where every line is separable,
# and (blank template only) the whole map, where the LOD ladder has spoken.
_NEAR_TILES = (40, 40, 70, 70)

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


def _open_window(path: Path):
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(1400, 1000)
    window.show()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        raise SystemExit(f"{path} failed to load")
    window.grid_action.setChecked(True)
    return window


def _frame(window, tiles) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    map_view = window.map_view
    x0, y0, x1, y1 = tiles
    rect = None
    for tx, ty in ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        polygon = map_view._tile_polygon(tx, ty)
        bounds = polygon.boundingRect()
        rect = bounds if rect is None else rect.united(bounds)
    map_view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.map_view.viewport().grab().save(str(out_path))


def _ground_plane_captures(out_dir: Path) -> list[Path]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape import grid_overlay
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    written: list[Path] = []
    window = _open_window(BLANK_TEMPLATE_PATH)
    window.grid_follow_action.setChecked(False)
    try:
        for style, isometric in (("Stepped", True), ("Sloped", True), ("Flat", True), ("Flat", False)):
            window.terrain_style_combo.setCurrentText(style)
            if style == "Flat":
                window.iso_action.setChecked(isometric)
            QApplication.processEvents()
            for blend in (grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX):
                window.map_view.set_grid_appearance(blend, grid_overlay.THICKNESS_DEFAULT)
                _frame(window, _NEAR_TILES)
                view = "iso" if isometric else "top"
                side = "dark" if blend < 0 else "light"
                path = out_dir / f"ground_{style.lower()}_{view}_blend_{side}{abs(blend)}.png"
                _grab(window, path)
                written.append(path)
                print(f"  {path.name}: lod={window.map_view._grid_item.last_lod}")

        window.map_view.fitInView(window.map_view._map_rect, Qt.KeepAspectRatio)
        path = out_dir / "ground_flat_top_fit_to_map.png"
        _grab(window, path)
        written.append(path)
        print(f"  {path.name}: lod={window.map_view._grid_item.last_lod}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _drape_captures(out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    if not ELEVATED_SCENARIO.exists():
        print(f"skipping the drape captures: {ELEVATED_SCENARIO} is not present")
        return []

    written: list[Path] = []
    window = _open_window(ELEVATED_SCENARIO)
    try:
        for style in ("Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            for follow in (True, False):
                window.grid_follow_action.setChecked(follow)
                _frame(window, (60, 60, 90, 90))
                path = out_dir / f"drape_{style.lower()}_{'follow' if follow else 'ground'}.png"
                _grab(window, path)
                written.append(path)
                print(f"  {path.name}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _ground_plane_captures(out_dir) + _drape_captures(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")

    import tempfile

    _ensure_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        written = generate(args.out_dir)

    for path in written:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
