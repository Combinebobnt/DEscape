#!/usr/bin/env python3
"""Screenshots the stacked-unit count badges (View > Show Stacked-Unit
Badges) for a manual eyeball pass -- always writes PNGs, never pass/fail.
Modelled on tools/gen_units_panel_eyeball.py, including the settings-isolation
trap testkit.settings_isolation exists for.

One synthetic scenario on the blank template carries, deliberately:
- two villagers at an identical point (10.5, 10.5): one badge reading 2;
- a Castle at (20, 20) with three villagers on three different tiles of its
  footprint: three separate badges reading 2, not one pile;
- two villagers on tile (12, 14) at different sub-tile points: no badge;
- a lone villager at (15.5, 11.5): no badge.

Captured in Flat, Stepped and Sloped at two zooms (the badge must be the same
device size in both), plus one fit-to-map capture below the LOD gate and one
with the toggle off (neither shows any badge).

Writes build/stack_badge_eyeball/, gitignored. No test reads it.
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

OUT_DIR = ROOT / "build" / "stack_badge_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_VILLAGER = 83
_CASTLE = 82
# Tile rects (x0, y0, x1, y1) framed by the near and far zoom captures.
_NEAR_TILES = (8, 8, 24, 24)
_FAR_TILES = (0, 0, 40, 40)

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

    units = window.scenario.unit_manager.units
    ref = iter(range(101, 200))
    placements = [
        (10.5, 10.5), (10.5, 10.5),  # identical point
        (12.25, 14.25), (12.75, 14.75),  # same tile, different sub-tile points
        (15.5, 11.5),  # lone
        (18.5, 18.5), (19.5, 21.5), (21.5, 20.5),  # on three Castle tiles
    ]
    for x, y in placements:
        units[1].append(_SyntheticUnit(x=x, y=y, unit_const=_VILLAGER, reference_id=next(ref)))
    units[2].append(_SyntheticUnit(x=20.0, y=20.0, unit_const=_CASTLE, reference_id=next(ref)))
    window.mode_combo.setCurrentText("Units")
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


def generate(out_dir: Path) -> list[Path]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window()
    try:
        for style in ("Flat", "Stepped", "Sloped"):
            if style == "Flat":
                window.iso_action.setChecked(False)
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            groups = window.map_view._stack_groups
            print(f"{style}: {len(groups)} groups at {sorted(groups)}")
            for label, tiles in (("near", _NEAR_TILES), ("far", _FAR_TILES)):
                _frame(window, tiles)
                path = out_dir / f"{style.lower()}_{label}.png"
                _grab(window, path)
                written.append(path)
                print(f"  {label}: badges_drawn={window.map_view._stack_badge_item.badges_drawn}")

        window.map_view.fitInView(window.map_view._map_rect, Qt.KeepAspectRatio)
        path = out_dir / "sloped_fit_to_map_below_lod_gate.png"
        _grab(window, path)
        written.append(path)
        print(f"fit-to-map: badges_drawn={window.map_view._stack_badge_item.badges_drawn}")

        _frame(window, _NEAR_TILES)
        window.stack_badges_action.setChecked(False)
        path = out_dir / "sloped_near_badges_off.png"
        _grab(window, path)
        written.append(path)
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
