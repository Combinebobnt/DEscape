#!/usr/bin/env python3
"""Screenshots View > Player Cameras for a manual eyeball pass -- always
writes PNGs, never pass/fail. Modelled on tools/gen_range_ring_eyeball.py,
including the settings-isolation trap testkit.settings_isolation exists for.

One fixture: `examples/F7_3_York (865).aoe2scenario`, which stores a real
set of starting views (P1 at (165, 105); P2..P6 all at (171, 78), measured,
not assumed -- five markers really do share one tile there). Captured in
Flat, Stepped and Sloped around each of those two tiles, plus a Players-mode
pair showing the emphasis moving from P1 to P2.

What the captures answer, and the tests do not: pen weight and legibility
over real terrain and sprites, whether the glyph reads as a camera at its
device size, whether it stands ON its tile at that tile's own elevation in
the two elevated styles, and what five overlapping markers look like. The
countable halves -- how many markers, which players, which one is
emphasised, device size across styles -- are asserted in
tests/test_camera_markers.py instead.

Writes build/camera_marker_eyeball/, gitignored. No test reads it.
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

OUT_DIR = ROOT / "build" / "camera_marker_eyeball"
SCENARIO = ROOT / "examples" / "F7_3_York (865).aoe2scenario"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

# Tiles wide enough to show the marker against its surroundings, tight
# enough that a 20 px glyph is not a speck.
_FRAME_RADIUS = 10
# The close frame, for judging the glyph itself: at a 20 px device size the
# context frame above answers "is it on the right tile" and nothing about
# pen weight or whether it reads as a camera.
_CLOSE_RADIUS = 3

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
    window.player_cameras_action.setChecked(True)
    return window


def _frame(window, centre: tuple[int, int], radius: int = _FRAME_RADIUS) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    map_view = window.map_view
    cx, cy = centre
    rect = None
    for tx, ty in (
        (cx - radius, cy - radius),
        (cx + radius, cy - radius),
        (cx - radius, cy + radius),
        (cx + radius, cy + radius),
    ):
        bounds = map_view._tile_polygon(tx, ty).boundingRect()
        rect = bounds if rect is None else rect.united(bounds)
    map_view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.map_view.viewport().grab().save(str(out_path))


def _player_views(window) -> dict[int, tuple[int, int]]:
    return {
        item.player_id: item.tile() for item in window.map_view.camera_marker_items()
    }


def generate(out_dir: Path) -> list[Path]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window(SCENARIO)
    try:
        views = _player_views(window)
        print(f"  markers: {views}")
        tiles = {f"p{player_id}": tile for player_id, tile in sorted(views.items())}
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            if style == "Flat":
                window.iso_action.setChecked(True)
            QApplication.processEvents()
            for label, tile in (("p1", tiles.get("p1")), ("shared", tiles.get("p2"))):
                if tile is None:
                    continue
                _frame(window, tile)
                path = out_dir / f"york_{label}_{style.lower()}.png"
                _grab(window, path)
                written.append(path)

        # Close frames, Stepped only: the glyph itself rather than where it
        # sits. Same 20 px marker, many fewer tiles around it.
        window.terrain_style_combo.setCurrentText("Stepped")
        QApplication.processEvents()
        for label, tile in (("p1", tiles.get("p1")), ("shared", tiles.get("p2"))):
            if tile is None:
                continue
            _frame(window, tile, _CLOSE_RADIUS)
            path = out_dir / f"york_{label}_close.png"
            _grab(window, path)
            written.append(path)

        # Emphasis, in Stepped: the marker the Players panel points at,
        # against the same frame with a different player selected.
        window.terrain_style_combo.setCurrentText("Stepped")
        window.mode_combo.setCurrentText("Players")
        QApplication.processEvents()
        for player_id in (1, 2):
            window.players_panel.select_player(player_id)
            QApplication.processEvents()
            _frame(window, tiles[f"p{player_id}"], _CLOSE_RADIUS)
            path = out_dir / f"york_emphasis_p{player_id}.png"
            _grab(window, path)
            written.append(path)

        # Off: the same frame with nothing drawn, so "is that mark ours"
        # has an answer that does not depend on memory.
        window.player_cameras_action.setChecked(False)
        QApplication.processEvents()
        _frame(window, tiles["p1"])
        path = out_dir / "york_p1_toggle_off.png"
        _grab(window, path)
        written.append(path)

        # Raised: both York view tiles sit at elevation 0, so the frames above
        # cannot show the glyph standing on raised terrain. Set Level 4 under
        # P1's view through the real tool path, then capture both elevated
        # styles. In memory only; nothing is saved.
        window.player_cameras_action.setChecked(True)
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("set_level")
        window.elevation_level_spin.setValue(4)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(tiles["p1"][0], tiles["p1"][1], Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        for style in ("Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            _frame(window, tiles["p1"], _CLOSE_RADIUS)
            path = out_dir / f"york_p1_raised_{style.lower()}.png"
            _grab(window, path)
            written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")
    if not SCENARIO.exists():
        raise SystemExit(f"{SCENARIO} is not present")

    import tempfile

    _ensure_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        # Pinned FIRST, from the real config, like gen_range_ring_eyeball:
        # half of what these captures answer is whether the glyph stays
        # legible over real sprite art rather than over coloured marks.
        if settings_isolation.pin_install_path() is None:
            print("no AoE2:DE install visible -- units will show as coloured marks")
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
