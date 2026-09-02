#!/usr/bin/env python3
"""Screenshots TriggerPanel (Phase 4c, trigger organization) for a manual
eyeball pass -- always writes PNGs, never pass/fail. This covers the
trigger-organization plan's own verification step: "Screenshot the panel
at 340 px on F7_3_York (18 sections) and old-allies-final-v2 (30 sections):
expanded, collapsed, tag-filtered, and under File order (flat) ... check
indent doesn't clip names at MIN_USEFUL_WIDTH."

Requires PyQt5 -- unlike tools/gen_seam_eyeball.py there is no off-engine
fallback, since the thing under review (grouped-tree rendering, indent,
clipping) only exists in the real widget.

Writes build/trigger_panel_eyeball/, gitignored. No test reads it. No
golden images -- there is no baseline, only visual inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "build" / "trigger_panel_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

# (output-name prefix, examples/ filename, expected section count per the
# plan's census -- logged, not asserted, since this script has no pass/fail).
TARGETS = (
    ("f7_3_york", "F7_3_York (865).aoe2scenario", 18),
    ("old_allies_final_v2", "old-allies-final-v2.aoe2scenario", 30),
)

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


def _isolate_config(tmp_dir: Path) -> None:
    """Redirect CONFIG_PATH before any ViewerWindow exists, or
    closeEvent()'s unconditional settings.set_window_size() writes straight
    through to this developer's real config.yaml -- the exact trap
    tests/conftest.py's _isolated_settings fixture exists to close, which a
    standalone tool doesn't get for free. See descape/settings.py's
    _SETTINGS_MEMOIZED_GLOBALS-shaped module globals: they must be reset too,
    since a prior import may have already memoized real values."""
    import descape.asset_source as asset_source_module
    import descape.settings as settings_module

    fake_config_path = tmp_dir / "config.yaml"
    asset_source_module.CONFIG_PATH = fake_config_path
    settings_module.CONFIG_PATH = fake_config_path
    for name in (
        "_zoom_centered_on_cursor",
        "_graphics_quality",
        "_dark_mode",
        "_elev_step_pct",
        "_window_size",
        "_split_sizes",
        "_log_height",
        "_distance_ticks",
        "_distance_tick_interval",
        "_keybinds",
    ):
        setattr(settings_module, name, None)


def _open_window(path: Path):
    from descape.viewer import ViewerWindow
    from PyQt5.QtWidgets import QApplication

    window = ViewerWindow()
    window.show()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        raise SystemExit(f"{path.name} failed to load")
    window.mode_combo.setCurrentText("Triggers")
    QApplication.processEvents()
    return window


def _grab(window, out_path: Path) -> None:
    window.trigger_panel.grab().save(str(out_path))


def _capture_one(prefix: str, filename: str, expected_sections: int, out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    path = ROOT / "examples" / filename
    if not path.is_file():
        raise SystemExit(f"missing corpus file: {path}")

    written: list[Path] = []
    window = _open_window(path)
    panel = window.trigger_panel
    try:
        print(f"{filename}: grouped={panel._grouped} top_level_rows={panel.tree.topLevelItemCount()} "
              f"(plan expects {expected_sections} sections)")

        # 1. expanded -- the populate-time default (every section starts
        # expanded, see trigger_panel.py's setExpanded(True) call sites).
        expanded_path = out_dir / f"{prefix}_expanded.png"
        _grab(window, expanded_path)
        written.append(expanded_path)

        # 2. collapsed.
        panel.tree.collapseAll()
        QApplication.processEvents()
        collapsed_path = out_dir / f"{prefix}_collapsed.png"
        _grab(window, collapsed_path)
        written.append(collapsed_path)
        panel.tree.expandAll()
        QApplication.processEvents()

        # 3. tag-filtered -- pick the first real tag, if the file has one.
        if panel.tag_combo.count() > 1:
            panel.tag_combo.setCurrentIndex(1)
            QApplication.processEvents()
            tag_path = out_dir / f"{prefix}_tag_filtered.png"
            _grab(window, tag_path)
            written.append(tag_path)
            panel.tag_combo.setCurrentIndex(0)
            QApplication.processEvents()
        else:
            print(f"  (skipping tag_filtered -- {filename} has no [tag]-prefixed trigger names)")

        # 4. File order (trigger ID) -- grouping only exists under Display
        # order, so this must render flat.
        panel.sort_combo.setCurrentText("File order (trigger ID)")
        QApplication.processEvents()
        file_order_path = out_dir / f"{prefix}_file_order.png"
        _grab(window, file_order_path)
        written.append(file_order_path)
        if panel._grouped:
            print(f"  WARNING: {filename} still reports _grouped=True under File order")
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


TRIGGER_FIXTURE = ROOT / "tests" / "fixtures" / "triggers_120x120.aoe2scenario"
# trigger_id 60's own first condition (destroy_object, unit_object=14201) --
# the small fixture carries no Unit/Unit[] field at all.
UNIT_PRESENTATION_FILE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"
UNIT_PRESENTATION_TRIGGER_ID = 60


def _select_effect(panel, trigger_list_index: int, child_index: int, scroll_to: str = "") -> None:
    from PyQt5.QtWidgets import QApplication

    panel.select_trigger(trigger_list_index)
    QApplication.processEvents()
    effects = panel.entry_tree.topLevelItem(2)
    panel.entry_tree.setCurrentItem(effects.child(child_index))
    QApplication.processEvents()
    if scroll_to:
        # The property form scrolls independently of the panel -- without
        # this the field under review can sit below the fold, off the
        # grabbed image entirely, the same trap 4b.6a's own overlap defect
        # was found by screenshotting rather than by a widget assertion.
        widget = next(w for s, _k, _i, w in panel._rows if s.name == scroll_to)
        panel.property_area.ensureWidgetVisible(widget)
        QApplication.processEvents()


def _capture_reference_states(out_dir: Path) -> list[Path]:
    """4d (trigger constant pickers): a reference row in each of its distinct
    widget states -- catalog (CatalogLineEdit), document (a trigger/variable
    combo), read-only (every widget disabled), and raw Unit (the unchanged
    spinbox, deferred to phase 3.5b). Plus the CatalogBrowseDialog itself.
    """
    from PyQt5.QtWidgets import QApplication

    written: list[Path] = []
    window = _open_window(TRIGGER_FIXTURE)
    panel = window.trigger_panel
    try:
        # catalog: "Fixture: armour split" (list index 1), first effect --
        # object_list_unit_id = 4 (ARCHER).
        _select_effect(panel, 1, 0, scroll_to="object_list_unit_id")
        path = out_dir / "reference_catalog.png"
        _grab(window, path)
        written.append(path)

        # document: "Fixture: references" (list index 2), first effect --
        # trigger_id = 0.
        _select_effect(panel, 2, 0, scroll_to="trigger_id")
        path = out_dir / "reference_document.png"
        _grab(window, path)
        written.append(path)

        # browse dialog, opened from the catalog row above.
        _select_effect(panel, 1, 0, scroll_to="object_list_unit_id")
        from descape.constant_picker import CatalogBrowseDialog, CatalogLineEdit

        catalog_widget = next(
            w for s, _k, _i, w in panel._rows if isinstance(w, CatalogLineEdit)
        )
        browse_dialog = CatalogBrowseDialog(catalog_widget._catalog, catalog_widget._default_category)
        browse_dialog.select(catalog_widget.value())
        browse_dialog.show()
        QApplication.processEvents()
        path = out_dir / "reference_browse_dialog.png"
        browse_dialog.grab().save(str(path))
        browse_dialog.close()
        written.append(path)

        # read-only: same document row, with the file gated off.
        window.scenario.trigger_write_supported = False
        panel.show_scenario(window.scenario)
        _select_effect(panel, 2, 0, scroll_to="trigger_id")
        path = out_dir / "reference_read_only.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()

    # raw Unit: a real corpus file's own condition -- the small fixture has
    # no Unit/Unit[] field to show this state on.
    window = _open_window(UNIT_PRESENTATION_FILE)
    panel = window.trigger_panel
    try:
        manager = panel._manager()
        trigger = next(t for t in manager.triggers if t.trigger_id == UNIT_PRESENTATION_TRIGGER_ID)
        list_index = list(manager.triggers).index(trigger)
        panel.select_trigger(list_index)
        QApplication.processEvents()
        conditions = panel.entry_tree.topLevelItem(1)
        panel.entry_tree.setCurrentItem(conditions.child(0))
        QApplication.processEvents()
        unit_widget = next(w for s, _k, _i, w in panel._rows if s.name == "unit_object")
        panel.property_area.ensureWidgetVisible(unit_widget)
        QApplication.processEvents()
        path = out_dir / "reference_raw_unit.png"
        _grab(window, path)
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for prefix, filename, expected_sections in TARGETS:
        written += _capture_one(prefix, filename, expected_sections, out_dir)
    written += _capture_reference_states(out_dir)
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
        _isolate_config(Path(tmp))
        written = generate(args.out_dir)

    for path in written:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
