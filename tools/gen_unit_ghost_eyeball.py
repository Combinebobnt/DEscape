#!/usr/bin/env python3
"""Screenshots the mid-drag move preview's ghost for a manual eyeball pass --
always writes PNGs, never pass/fail. Modelled on
tools/gen_stack_badge_eyeball.py, including the settings-isolation trap
testkit.settings_isolation exists for.

The ghost is a live visual: no automated layer proves it looks right, and the
default test tier cannot see the sprite half of it at all (conftest's settings
isolation hides the AoE2:DE install on purpose). So each capture is taken
mid-gesture -- press, move past UNIT_DRAG_THRESHOLD_PX, grab, and only then
release -- against a REAL example scenario, so real .sld art resolves.

Per style (Flat, Stepped, Sloped), with sprites on and off:
- a multi-tile building dragged several tiles, which exercises the composite-
  pieces path (a town centre is four pieces, not one);
- a 1x1 unit dragged the same way;
- one capture after the release, to show the ghost actually clears.

Flat is expected to show the coloured MARK in both sprite states:
composite_rect_flat has no sprite compositor at all, so honouring the toggle
there would preview art the map cannot draw.

Writes build/unit_ghost_eyeball/, gitignored. No test reads it.
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

OUT_DIR = ROOT / "build" / "unit_ghost_eyeball"
DEFAULT_SCENARIO = ROOT / "examples" / "0_June_Event_Scenario.aoe2scenario"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_DRAG_TILES = 6

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


def _isolate_config(tmp_dir: Path) -> None:
    """The install path is pinned into the environment FIRST, from the real
    config: unlike every sibling tool here, this one is about sprite art, and
    isolating the config without that would silently capture the coloured
    mark everywhere and look like a bug in the ghost."""
    import descape.asset_source as asset_source_module

    install = asset_source_module.get_install_path()
    if install is not None:
        os.environ["AOE2DE_INSTALL_PATH"] = str(install)
    else:
        print("no AoE2:DE install visible -- every capture will show the coloured mark")

    settings_isolation.isolate_settings(tmp_dir)


def _open_window(scenario_path: Path):
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(1400, 1000)
    window.show()
    window.load_scenario(scenario_path)
    if window.scenario is None:
        window.close()
        raise SystemExit(f"{scenario_path} failed to load")
    window.mode_combo.setCurrentText("Units")
    return window


def _subjects(window):
    """(label, entry) for a multi-tile building and a 1x1 unit -- the two
    shapes the ghost resolves differently (a composite's several pieces
    versus a single draw).

    Player-owned rather than GAIA, where the file has one. Not cosmetic: a
    capture whose subject is a lone rock in a forest is one a human cannot
    actually eyeball, and these PNGs exist for exactly that pass. GAIA is the
    fallback only so a unit-free player never leaves a style uncaptured."""
    from descape import render
    from descape.unit_filter import GAIA_PLAYER_ID

    index = window.map_view._unit_index
    picks: dict[tuple[str, bool], object] = {}
    for entry in index.entries:
        span = render.tile_span(entry.unit.unit_const, render.NON_BUILDING_SPAN)
        label = "building" if span[0] > 2 else ("unit" if span == (1, 1) else None)
        if label is None:
            continue
        picks.setdefault((label, entry.player_id == GAIA_PLAYER_ID), entry)
    out = []
    for label in ("building", "unit"):
        entry = picks.get((label, False)) or picks.get((label, True))
        if entry is not None:
            out.append((label, entry))
    return out


def _frame(window, entry) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    view = window.map_view
    rect = None
    for dx in range(-2, _DRAG_TILES + 3):
        for dy in range(-2, _DRAG_TILES + 3):
            polygon = view._tile_polygon(entry.own_x + dx, entry.own_y + dy)
            if polygon is None:
                continue
            bounds = polygon.boundingRect()
            rect = bounds if rect is None else rect.united(bounds)
    if rect is not None:
        view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _viewport_pos(window, tile_x: int, tile_y: int):
    from PyQt5.QtCore import QPointF

    polygon = window.map_view._tile_polygon(tile_x, tile_y)
    centre = polygon.boundingRect().center()
    return QPointF(window.map_view.mapFromScene(centre))


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.map_view.viewport().grab().save(str(out_path))


def _drag_and_grab(window, entry, out_path: Path, release: bool) -> bool:
    """Press, move, grab MID-GESTURE, then optionally release. Returns whether
    a ghost was actually on screen at grab time, so a capture that shows
    nothing is distinguishable from one that was never taken."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QMouseEvent

    view = window.map_view
    press = _viewport_pos(window, entry.own_x, entry.own_y)
    move = _viewport_pos(window, entry.own_x + _DRAG_TILES, entry.own_y + _DRAG_TILES)

    def event(kind, pos, button, buttons):
        return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)

    view.mousePressEvent(event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
    view.mouseMoveEvent(event(QEvent.MouseMove, move, Qt.NoButton, Qt.LeftButton))
    showing = view._unit_ghost_item is not None
    if not release:
        _grab(window, out_path)
    view.mouseReleaseEvent(event(QEvent.MouseButtonRelease, move, Qt.LeftButton, Qt.NoButton))
    if release:
        _grab(window, out_path)
    return showing


def generate(out_dir: Path, scenario_path: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window(scenario_path)
    try:
        for style in ("Flat", "Stepped", "Sloped"):
            if style == "Flat":
                window.iso_action.setChecked(False)
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            for sprites in (True, False):
                window.show_sprites_action.setEnabled(True)
                window.show_sprites_action.setChecked(sprites)
                QApplication.processEvents()
                tag = "sprites_on" if sprites else "sprites_off"
                for label, entry in _subjects(window):
                    _frame(window, entry)
                    path = out_dir / f"{style.lower()}_{tag}_{label}.png"
                    showing = _drag_and_grab(window, entry, path, release=False)
                    written.append(path)
                    print(f"{path.name}: ghost_on_screen={showing}")
                    window.undo()
                    QApplication.processEvents()

        # One capture after the release, so a ghost that never clears is
        # visible rather than merely untested.
        label, entry = _subjects(window)[0]
        _frame(window, entry)
        path = out_dir / "sloped_after_release_no_ghost.png"
        _drag_and_grab(window, entry, path, release=True)
        written.append(path)
        print(f"{path.name}: ghost_item_after_release={window.map_view._unit_ghost_item}")
        window.undo()
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO, help="Scenario to drag units in")
    args = parser.parse_args()
    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable")
    _ensure_qapp()
    _isolate_config(args.out_dir)
    for path in generate(args.out_dir, args.scenario):
        print(path)


if __name__ == "__main__":
    main()
