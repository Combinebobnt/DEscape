#!/usr/bin/env python3
"""Screenshots UnitsPanel (the units-sidebar-catalog work) for a manual
eyeball pass -- always writes PNGs, never pass/fail. Modelled on
tools/gen_trigger_panel_eyeball.py, including the trap
testkit.settings_isolation exists for: redirect CONFIG_PATH before any
ViewerWindow exists, or closeEvent()'s unconditional
settings.set_window_size() writes straight through to this developer's real
config.yaml.

Captures, at MIN_USEFUL_WIDTH and a comfortable width:
1. Nothing selected -- "No unit selected", stats block hidden.
2. An Archer selected -- both grids, five stat rows, editable rotation, no
   rotation note (a real facing const, not a variant one -- see
   _ANGLE_ROTATION_CONST's own comment for why this isn't a building).
3. A tree selected -- Hit points only, combat rows gone, rotation note visible.
4. Three units selected -- the count readout.
5. Catalog filtered to a long name (elision check, no horizontal clipping).
6. The vertical splitter dragged to its floor (the _fit_inspector_height
   check -- the two notes must scroll, never overlap).

Writes build/units_panel_eyeball/, gitignored. No test reads it. No golden
images -- there is no baseline, only visual inspection.
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

OUT_DIR = ROOT / "build" / "units_panel_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

# 4/284 are pinned ground truth in tests/test_unit_stats_table.py (Archer:
# all five rows; a tree: HP only). Archer, not Town Center, for the "full
# stats, editable rotation, no caveat note" state: unit_rotation.
# rotation_is_angle(109) is actually False (buildings don't have a facing
# any more than a tree does -- AGENTS.md's creatable-only whitelist excludes
# them), so a Town Centre screenshot would have shown the rotation note
# anyway. Caught by generating and looking at the image, exactly what this
# pass is for. The long name is a real object catalog entry -- the elision
# check needs a real 36-character name, not a synthetic one.
_ANGLE_ROTATION_CONST = 4  # Archer
_TREE_CONST = 284
_LONG_NAME_FILTER = "FLAGSHIP OF NEARCHOS DOCKED MOVEABLE"

_MIN_WIDTH = 380  # UnitsPanel.MIN_USEFUL_WIDTH, read fresh at capture time
_COMFORTABLE_WIDTH = 500

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
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.show()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit("blank template failed to load")

    units = window.scenario.unit_manager.units
    units[1].append(_SyntheticUnit(x=4.5, y=4.5, unit_const=_ANGLE_ROTATION_CONST, reference_id=101))
    units[0].append(_SyntheticUnit(x=8.5, y=8.5, unit_const=_TREE_CONST, reference_id=102))
    units[1].append(_SyntheticUnit(x=12.5, y=12.5, unit_const=_ANGLE_ROTATION_CONST, reference_id=103))
    units[1].append(_SyntheticUnit(x=16.5, y=16.5, unit_const=_ANGLE_ROTATION_CONST, reference_id=104))

    window.mode_combo.setCurrentText("Units")
    window._rebuild_unit_index()
    QApplication.processEvents()
    return window


def _set_left_width(window, width: int) -> None:
    sizes = window.content_splitter.sizes()
    total = sum(sizes) or (width + 800)
    window.content_splitter.setSizes([width, total - width])


def _grab(window, out_path: Path) -> None:
    from PyQt5.QtWidgets import QApplication

    QApplication.processEvents()
    window.units_panel.grab().save(str(out_path))


def _entry_for(window, reference_id: int):
    index = window.map_view._unit_index
    return next(e for e in index.entries if e.unit.reference_id == reference_id)


def _capture_at_width(window, width: int, out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    _set_left_width(window, width)
    QApplication.processEvents()

    # 1. Nothing selected.
    window.units_panel.clear()
    path = out_dir / f"w{width}_1_nothing_selected.png"
    _grab(window, path)
    written.append(path)

    # 2. An Archer selected -- five stat rows, editable rotation, no
    # rotation note (a real facing, so the caveat doesn't apply).
    window.units_panel.show_unit(_entry_for(window, 101))
    path = out_dir / f"w{width}_2_angle_rotation_const_selected.png"
    _grab(window, path)
    written.append(path)

    # 3. A tree selected -- Hit points only, rotation note visible.
    window.units_panel.show_unit(_entry_for(window, 102))
    path = out_dir / f"w{width}_3_tree_selected.png"
    _grab(window, path)
    written.append(path)

    # 4. Three units selected -- the count readout.
    window.units_panel.show_selection_count(3)
    path = out_dir / f"w{width}_4_three_selected.png"
    _grab(window, path)
    written.append(path)

    # 5. Catalog filtered to a long name -- the elision check.
    window.units_panel.catalog_view.filter_edit.setText(_LONG_NAME_FILTER)
    window.units_panel.catalog_view.select(2448)
    path = out_dir / f"w{width}_5_long_name_filtered.png"
    _grab(window, path)
    written.append(path)
    window.units_panel.catalog_view.filter_edit.setText("")

    # 6. Splitter dragged to its floor -- _fit_inspector_height's own check.
    window.units_panel.show_unit(_entry_for(window, 102))  # the tree: both notes visible
    from descape import settings

    splitter = window.units_panel.splitter
    sizes = splitter.sizes()
    total = sum(sizes) or 900
    splitter.setSizes([total - settings.MIN_SPLIT_PANE, settings.MIN_SPLIT_PANE])
    QApplication.processEvents()
    path = out_dir / f"w{width}_6_splitter_at_floor.png"
    _grab(window, path)
    written.append(path)
    splitter.setSizes(sizes)

    return written


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window()
    try:
        print(
            f"UnitsPanel.MIN_USEFUL_WIDTH={window.units_panel.MIN_USEFUL_WIDTH}, "
            f"catalog rows={window.units_panel.catalog_view.tree.topLevelItemCount()} top-level groups"
        )
        for width in (_MIN_WIDTH, _COMFORTABLE_WIDTH):
            written += _capture_at_width(window, width, out_dir)
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
