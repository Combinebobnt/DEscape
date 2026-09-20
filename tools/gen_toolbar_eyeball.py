#!/usr/bin/env python3
"""Screenshots the main toolbar (both rows) for a manual eyeball pass on the
width-driven "More Tools" tool-overflow button -- always writes PNGs, never
pass/fail. Confirms the empty tool-params row holds its height across
tools/widths, and that the More Tools button's text tracks the active tool
when that tool is overflowed.

Follows tools/gen_trigger_panel_eyeball.py's shape exactly: its own
_ensure_qapp(), testkit.settings_isolation, and no golden-image baseline -- this repo
has no golden-image gate for window chrome anywhere, and establishing one is
out of scope for this plan.

Writes build/toolbar_eyeball/, gitignored. No test reads it.
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

OUT_DIR = ROOT / "build" / "toolbar_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_WIDTHS = (800, 1024, 1280)
# (prefix, mode text, tool_id to select once in that mode -- "" for none).
_STATES = (
    ("view", "View", ""),
    ("terrain_draw", "Terrain", "draw"),
    ("terrain_set_elevation", "Terrain", "set_level"),
)

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


def _open_window():
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit("blank template failed to load")
    window.show()
    QApplication.processEvents()
    return window


def _grab_toolbars(window, out_path: Path) -> None:
    """Both toolbar rows, stacked as one image -- there's no single widget
    that spans both (they're separate QToolBars either side of
    addToolBarBreak()), so grab each and paste them together."""
    from PyQt5.QtGui import QImage, QPainter

    main_pix = window.main_toolbar.grab()
    param_toolbar = next(
        tb for tb in window.findChildren(type(window.main_toolbar)) if tb.windowTitle() == "Tool Options"
    )
    param_pix = param_toolbar.grab()

    width = max(main_pix.width(), param_pix.width())
    height = main_pix.height() + param_pix.height()
    combined = QImage(width, height, QImage.Format_ARGB32)
    combined.fill(0xFFFFFFFF)
    painter = QPainter(combined)
    painter.drawPixmap(0, 0, main_pix)
    painter.drawPixmap(0, main_pix.height(), param_pix)
    painter.end()
    combined.save(str(out_path))


def generate(out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    window = _open_window()
    try:
        for prefix, mode_text, tool_id in _STATES:
            window.mode_combo.setCurrentText(mode_text)
            QApplication.processEvents()
            if tool_id:
                # .trigger() (not _on_tool_selected() directly) so the
                # QAction's own checked state -- and so its toolbar
                # button's highlight -- matches what a real click leaves
                # behind, for an honest screenshot.
                getattr(window, f"{tool_id}_action").trigger()
                QApplication.processEvents()
            for width in _WIDTHS:
                window.resize(width, 800)
                QApplication.processEvents()
                more_tools_state = (
                    "overflow" if window.more_tools_action.isVisible() else "no_overflow"
                )
                path = out_dir / f"{prefix}_{width}_{more_tools_state}.png"
                _grab_toolbars(window, path)
                written.append(path)
                print(
                    f"{prefix} @ {width}px: more_tools={more_tools_state} "
                    f"button_text={window.more_tools_button.text()!r} "
                    f"menu_count={len(window.more_tools_menu.actions())}"
                )
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
