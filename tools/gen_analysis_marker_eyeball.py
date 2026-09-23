#!/usr/bin/env python3
"""Screenshots Tools > Map Analysis' on-map markers for a manual eyeball
pass -- always writes PNGs, never pass/fail. Modelled on
tools/gen_stack_badge_eyeball.py, including the settings-isolation trap
testkit.settings_isolation exists for.

One synthetic scenario on the blank template carries, deliberately:
- an elevation bump (one tile at 2 among 0s) on a BLACK / DESERT_SAND
  checkerboard: error markers, one reading "!! x4", over dark and bright
  terrain side by side;
- a second bump on plain grass, far from the first, so fit-to-view shows
  two separate clusters;
- a player-1 villager garrisoned in a unit id nothing carries: a warning;
- a player-1 villager alone on a 3x3 island in water: an info marker.

Captured at fit-to-view and at close zoom in Flat, Stepped and Sloped, plus
one close capture with the first bump's row focused (the ring) and one after
the dialog closed (no markers).

Writes build/analysis_marker_eyeball/, gitignored. No test reads it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from testkit import settings_isolation

OUT_DIR = ROOT / "build" / "analysis_marker_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_VILLAGER = 83
_BLACK, _DESERT_SAND, _WATER = 47, 14, 1
_BUMP_A = (20, 20)
_BUMP_B = (70, 60)
_GARRISONED = (40.5, 30.5)
_ISLAND = (30, 70, 33, 73)  # x0, y0, x1, y1, exclusive
_CLOSE_TILES = (14, 14, 27, 27)

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


@dataclass
class _SyntheticUnit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1


def _open_window():
    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(1400, 1000)
    window.show()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit("blank template failed to load")

    mm = window.scenario.map_manager
    width = mm.map_width

    def tile(x, y):
        return mm.terrain[y * width + x]

    for y in range(_BUMP_A[1] - 6, _BUMP_A[1] + 6):
        for x in range(_BUMP_A[0] - 6, _BUMP_A[0] + 6):
            tile(x, y).terrain_id = _BLACK if (x + y) % 2 else _DESERT_SAND
    tile(*_BUMP_A).elevation = 2
    tile(*_BUMP_B).elevation = 2
    x0, y0, x1, y1 = _ISLAND
    for y in range(y0 - 3, y1 + 3):
        for x in range(x0 - 3, x1 + 3):
            tile(x, y).terrain_id = _WATER if not (x0 <= x < x1 and y0 <= y < y1) else tile(x, y).terrain_id
    units = window.scenario.unit_manager.units
    units[1].append(_SyntheticUnit(x=_GARRISONED[0], y=_GARRISONED[1], unit_const=_VILLAGER, reference_id=901,
                                   garrisoned_in_id=4242))
    units[1].append(_SyntheticUnit(x=x0 + 1.5, y=y0 + 1.5, unit_const=_VILLAGER, reference_id=902))
    window._render_current(reset_view=True)
    return window


def _frame(window, tiles) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    map_view = window.map_view
    x0, y0, x1, y1 = tiles
    rect = None
    for tx, ty in ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        bounds = map_view._tile_polygon(tx, ty).boundingRect()
        rect = bounds if rect is None else rect.united(bounds)
    map_view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _fit(window) -> None:
    from PyQt5.QtWidgets import QApplication

    window.map_view.set_isometric(window.map_view._isometric)
    QApplication.processEvents()


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.map_view.viewport().grab().save(str(out_path))


def _drawn(window) -> int:
    item = window.map_view.analysis_marker_item()
    return -1 if item is None else item.markers_drawn


def generate(out_dir: Path) -> list[Path]:
    from PyQt5.QtCore import Qt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window()
    try:
        window.analysis_action.trigger()
        print(f"anchors: {window.map_view._analysis_markers}")
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            for label in ("fit", "close"):
                if label == "fit":
                    _fit(window)
                else:
                    _frame(window, _CLOSE_TILES)
                path = out_dir / f"{style.lower()}_{label}.png"
                _grab(window, path)
                written.append(path)
                print(f"{style} {label}: markers_drawn={_drawn(window)}")

        tree = window._analysis_dialog.tree
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                finding = group.child(j).data(0, Qt.UserRole)
                if finding is not None and finding.tile == _BUMP_A:
                    tree.setCurrentItem(group.child(j))
                    break
        _frame(window, _CLOSE_TILES)
        path = out_dir / "sloped_close_focused.png"
        _grab(window, path)
        written.append(path)
        print(f"focus: {window.map_view._analysis_focus}")

        window._analysis_dialog.close()
        path = out_dir / "sloped_close_dialog_closed.png"
        _grab(window, path)
        written.append(path)
        print(f"after close: item={window.map_view.analysis_marker_item()}")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")

    import tempfile

    _ensure_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        # Pinned first, from the real config: legibility over real terrain
        # textures is half of what these captures answer.
        if settings_isolation.pin_install_path() is None:
            print("no AoE2:DE install visible -- terrain shows as flat colours")
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
