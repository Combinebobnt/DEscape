#!/usr/bin/env python3
"""Screenshots the toolbar (both rows) for a manual eyeball pass -- always
writes PNGs, never pass/fail. Covers the toolbar-overflow plan's own
Stages 1-2 verification step: dump both rows at 800/1024/1280 px in View,
Terrain+Draw and Terrain+Set Elevation, confirming the empty params row
holds its height and that mode-inapplicable tools disappear rather than
grey out.

Requires PyQt5 -- there is no off-engine fallback, since window chrome only
exists in the real widget.

Writes build/toolbar_eyeball/, gitignored. No test reads it. No golden
images -- there is no baseline, only visual inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "build" / "toolbar_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

WIDTHS = (800, 1024, 1280)

# (prefix, mode label, tool_id or None for whichever tool the mode leaves
# active -- Pan for View).
STATES = (
    ("view", "View", None),
    ("terrain_draw", "Terrain", "draw"),
    ("terrain_set_level", "Terrain", "set_level"),
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
    through to this developer's real config.yaml -- see
    tools/gen_trigger_panel_eyeball.py's own copy of this function for why."""
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


def _open_window(width: int):
    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow
    from PyQt5.QtWidgets import QApplication

    window = ViewerWindow()
    window.resize(width, 700)
    window.show()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit("blank template failed to load")
    QApplication.processEvents()
    return window


def _capture(prefix: str, mode: str, tool_id: str | None, width: int, out_dir: Path) -> Path:
    from PyQt5.QtWidgets import QApplication

    window = _open_window(width)
    try:
        window.mode_combo.setCurrentText(mode)
        if tool_id is not None:
            window._on_tool_selected(tool_id)
        QApplication.processEvents()
        out_path = out_dir / f"{prefix}_{width}.png"
        window.grab().save(str(out_path))
    finally:
        window.edit_history.mark_saved()
        window.close()
    return out_path


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for prefix, mode, tool_id in STATES:
        for width in WIDTHS:
            written.append(_capture(prefix, mode, tool_id, width, out_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        raise SystemExit("PyQt5 not importable -- this script has no Qt-free fallback")

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
