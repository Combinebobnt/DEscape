#!/usr/bin/env python3
"""Screenshots View > Range Rings for a manual eyeball pass -- always writes
PNGs, never pass/fail. Modelled on tools/gen_grid_eyeball.py, including the
settings-isolation trap testkit.settings_isolation exists for.

Two fixtures, because the two questions the ring raises need different
terrain:

- the blank template, with a Castle (range 8, 4x4, so a 10-tile ring) and a
  House (range 0) injected on flat ground, captured in Flat (with and without
  Isometric View), Stepped and Sloped. This is the shape question: a true
  circle in Flat before the view transform, a 2:1 ellipse in the iso styles,
  an even pen weight all the way round, and nothing at all around the House;
- an elevated `examples/` scenario with a Castle injected on the highest
  ground in view, captured in Stepped and Sloped. This is the placement
  question: the ring sits at the BUILDING's own height, while its radius
  stays measured on the ground plane;
- the flat-ground Castle again with a stand-in trigger effect setting its
  range to 12 (GH #49 slice 2), in the same four views: a dashed ring at 14
  tiles beside the solid one at 10, so the dashed one's ink box is 1.4 times
  the solid one's on each axis around the same centre.

The ring deliberately draws over the sprites, unlike the game's own, which
composites its circle into the terrain -- see map_view._draw_unit_range_rings.

Writes build/range_ring_eyeball/, gitignored. No test reads it.
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

OUT_DIR = ROOT / "build" / "range_ring_eyeball"
ELEVATED_SCENARIO = ROOT / "examples" / "2_Joan_coop_2_v0_15.aoe2scenario"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

CASTLE_CONST = 82  # 4x4, range 8.0
HOUSE_CONST = 70  # 2x2, range 0.0
# Half the widest ring plus margin, so the whole ellipse is inside the frame.
_FRAME_RADIUS = 14

_QAPP = None


@dataclass
class _Unit:
    """The fields render/unit_pick read off a placed unit, same synthetic
    stand-in tests/test_unit_selection_viewer.py uses."""

    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1


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
    window.range_rings_action.setChecked(True)
    return window


def _place(window, unit_const: int, tile: tuple[int, int], reference_id: int) -> None:
    """Anchors a building at `tile`, at the `tile + span/2` placement real
    files use, so the footprint starts on `tile` itself."""
    from descape.terrain_palette import BUILDING_TILE_SPANS

    span_x, span_y = BUILDING_TILE_SPANS[unit_const]
    window.scenario.unit_manager.units[1].append(
        _Unit(
            x=tile[0] + span_x / 2,
            y=tile[1] + span_y / 2,
            unit_const=unit_const,
            reference_id=reference_id,
        )
    )


def _select(window, unit_const: int):
    from PyQt5.QtWidgets import QApplication

    window.mode_combo.setCurrentText("Units")
    QApplication.processEvents()
    entry = next(
        e for e in window.map_view._unit_index.entries if e.unit.unit_const == unit_const
    )
    window._selection = [(entry.player_id, entry.unit.reference_id)]
    window.map_view.set_unit_selection([entry])
    window.units_panel.show_unit(entry)
    QApplication.processEvents()
    return entry


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
        polygon = map_view._tile_polygon(tx, ty)
        bounds = polygon.boundingRect()
        rect = bounds if rect is None else rect.united(bounds)
    map_view.fitInView(rect, Qt.KeepAspectRatio)
    QApplication.processEvents()


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.map_view.viewport().grab().save(str(out_path))


def _flat_ground_captures(out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import BLANK_TEMPLATE_PATH

    written: list[Path] = []
    window = _open_window(BLANK_TEMPLATE_PATH)
    castle_tile, house_tile = (40, 40), (70, 40)
    _place(window, CASTLE_CONST, castle_tile, 201)
    _place(window, HOUSE_CONST, house_tile, 202)
    try:
        for style, isometric in (("Stepped", True), ("Sloped", True), ("Flat", True), ("Flat", False)):
            window.terrain_style_combo.setCurrentText(style)
            if style == "Flat":
                window.iso_action.setChecked(isometric)
            QApplication.processEvents()
            _select(window, CASTLE_CONST)
            _frame(window, castle_tile)
            view = "iso" if isometric else "top"
            path = out_dir / f"castle_{style.lower()}_{view}.png"
            _grab(window, path)
            written.append(path)

            _select(window, HOUSE_CONST)
            _frame(window, house_tile)
            path = out_dir / f"house_{style.lower()}_{view}.png"
            _grab(window, path)
            written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


TRIGGER_RANGE = 12.0
# The trigger ring is 14 tiles, so the frame needs more room than _FRAME_RADIUS.
TRIGGER_FRAME_RADIUS = 18


def trigger_set_effect(value: float = TRIGGER_RANGE, unit_const: int = CASTLE_CONST):
    """A Change Object Range SET `value` for every P1 `unit_const`, as
    trigger_range.collect_range_effects() would report it."""
    from descape import trigger_range
    from descape.trigger_fields import UNSET

    return trigger_range.RangeEffect(
        sequence=0, trigger_index=0, trigger_name="Eyeball", effect_index=0,
        effect_type=trigger_range.CHANGE_OBJECT_RANGE, effect_name="change_object_range",
        operation=trigger_range.SET, quantity=value, source_player=1, unit_const=unit_const,
        group=UNSET, object_type=UNSET, selected_ids=frozenset(), area=None,
        has_conditions=False, looping=False, enabled=True,
    )


VIEWS = (("Stepped", True), ("Sloped", True), ("Flat", True), ("Flat", False))


def set_view(window, style: str, isometric: bool) -> None:
    from PyQt5.QtWidgets import QApplication

    window.terrain_style_combo.setCurrentText(style)
    if style == "Flat":
        window.iso_action.setChecked(isometric)
    QApplication.processEvents()


def _trigger_captures(out_dir: Path) -> list[Path]:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    written: list[Path] = []
    window = _open_window(BLANK_TEMPLATE_PATH)
    castle_tile = (40, 40)
    _place(window, CASTLE_CONST, castle_tile, 201)
    effects = (trigger_set_effect(),)
    # Stands in for a parsed file: every selection refresh re-reads it, so a direct push would be overwritten.
    window._trigger_range_effects = lambda: effects
    try:
        for style, isometric in VIEWS:
            set_view(window, style, isometric)
            _select(window, CASTLE_CONST)
            _frame(window, castle_tile, TRIGGER_FRAME_RADIUS)
            view = "iso" if isometric else "top"
            path = out_dir / f"castle_trigger_{style.lower()}_{view}.png"
            _grab(window, path)
            written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _highest_tile(window) -> tuple[int, int]:
    """The highest tile with room for a 4x4 and a ring around it."""
    import numpy as np

    elevations = window.map_view._iso_elevations
    margin = _FRAME_RADIUS + 2
    inner = elevations[margin:-margin, margin:-margin]
    flat_index = int(np.argmax(inner))
    y, x = np.unravel_index(flat_index, inner.shape)
    return int(x) + margin, int(y) + margin


def _elevated_captures(out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    if not ELEVATED_SCENARIO.exists():
        print(f"skipping the elevated captures: {ELEVATED_SCENARIO} is not present")
        return []

    written: list[Path] = []
    window = _open_window(ELEVATED_SCENARIO)
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        QApplication.processEvents()
        tile = _highest_tile(window)
        print(f"  elevated Castle at tile {tile}, elevation {window.map_view._iso_elevations[tile[1], tile[0]]}")
        _place(window, CASTLE_CONST, tile, 301)
        for style in ("Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            QApplication.processEvents()
            _select(window, CASTLE_CONST)
            _frame(window, tile)
            path = out_dir / f"elevated_castle_{style.lower()}.png"
            _grab(window, path)
            written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _flat_ground_captures(out_dir) + _elevated_captures(out_dir) + _trigger_captures(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")

    import tempfile

    _ensure_qapp()
    with tempfile.TemporaryDirectory() as tmp:
        # Pinned FIRST, from the real config, like gen_unit_ghost_eyeball:
        # half of what these captures answer is whether the ring stays
        # legible over real sprite art rather than over coloured marks.
        if settings_isolation.pin_install_path() is None:
            print("no AoE2:DE install visible -- buildings will show as coloured marks")
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
