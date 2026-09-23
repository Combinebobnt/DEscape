#!/usr/bin/env python3
"""Screenshots the Terrain-mode sidebar picker page (TerrainPanel, GH #56) in
a real ViewerWindow for a manual eyeball pass, at the panel's
MIN_USEFUL_WIDTH and a wider width, and measures what a screenshot cannot
settle: that the swatches are actually different textures rather than 131
identical green squares, and the page's layout numbers (tree column vs
viewport width, note wrap, hover line height).

Run outside pytest deliberately. tests/conftest.py's _isolated_settings
hides the configured AoE2DE install, which is the right default for the
test suite (the no-install fallback is what it pins) but means the real
.dds swatches are never exercised there. This tool is where they are;
--no-install captures the fallback path instead.

Writes build/terrain_browser_eyeball/, gitignored. No golden images.
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

from testkit import settings_isolation

OUT_DIR = ROOT / "build" / "terrain_browser_eyeball"

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_QAPP = None


def _ensure_qapp() -> None:
    global _QAPP
    if _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(sys.argv[:1])


_WIDER_WIDTH = 460
# A far-corner tile (3-digit coordinates) for the longest plausible hover string.
_HOVER_TILE = (119, 119)
_LONGEST_NAME_TERRAIN = 81  # BEACH_NON_NAVIGABLE_WET_GRAVEL


def _open_window():
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(1400, 900)
    window.show()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    if window.scenario is None:
        window.close()
        raise SystemExit("blank template failed to load")
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    mm = window.scenario.map_manager
    x, y = (min(c, mm.map_width - 1) for c in _HOVER_TILE)
    # The catalog's longest name under the hover tile, so the one-line readout has something to elide.
    mm.terrain[y * mm.map_width + x].terrain_id = _LONGEST_NAME_TERRAIN
    window.on_hover((x, y))
    QApplication.processEvents()
    return window


def _set_left_width(window, width: int) -> None:
    from PyQt5.QtWidgets import QApplication

    sizes = window.content_splitter.sizes()
    total = sum(sizes) or (width + 800)
    window.content_splitter.setSizes([width, total - width])
    QApplication.processEvents()
    QApplication.processEvents()


def _layout_report(window, label: str) -> None:
    panel = window.terrain_panel
    tree = panel.view.tree
    line = panel.hover_label.fontMetrics().lineSpacing()
    print(
        f"  [{label}] panel width={panel.width()} tree column={tree.header().sectionSize(0)} "
        f"viewport={tree.viewport().width()} hscroll={'shown' if tree.horizontalScrollBar().isVisible() else 'hidden'} "
        f"note height={panel.note_label.height()} (line {line}) hover height={panel.hover_label.height()} "
        f"hover text={panel.hover_label.text()!r}"
    )


def _capture(out_dir: Path) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    window = _open_window()
    written: list[Path] = []
    panel = window.terrain_panel
    try:
        print(f"TerrainPanel.MIN_USEFUL_WIDTH={panel.MIN_USEFUL_WIDTH}, default terrain id={panel.terrain_id()}")
        for width in (panel.MIN_USEFUL_WIDTH, _WIDER_WIDTH):
            _set_left_width(window, width)
            panel.view.filter_edit.setText("")
            if panel.view.show_hidden_checkbox is not None:
                panel.view.show_hidden_checkbox.setChecked(False)
            QApplication.processEvents()
            _layout_report(window, f"w{width} opened")
            path = out_dir / f"w{width}_1_opened_on_default.png"
            panel.grab().save(str(path))
            written.append(path)

            panel.view.filter_edit.setText("snow")
            QApplication.processEvents()
            path = out_dir / f"w{width}_2_filtered_snow.png"
            panel.grab().save(str(path))
            written.append(path)

            panel.view.filter_edit.setText("")
            panel.view.show_hidden_checkbox.setChecked(True)
            panel.view.tree.expandAll()
            QApplication.processEvents()
            _layout_report(window, f"w{width} all expanded")
            path = out_dir / f"w{width}_3_all_expanded_with_unused.png"
            panel.grab().save(str(path))
            written.append(path)
            panel.view.tree.collapseAll()

        _set_left_width(window, panel.MIN_USEFUL_WIDTH)
        path = out_dir / "window_terrain_draw.png"
        window.grab().save(str(path))
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written


def _measure(out_dir: Path) -> int:
    """The assertions a screenshot cannot make, per category representative:
    the swatches are pairwise distinct in mean RGB, and each one carries
    real texture detail rather than being a flat square.

    The second check is the load-bearing one. terrain_palette's fallback
    colour table was baked FROM these same texture averages, so a flat
    fallback square and a real swatch have the same mean -- mean alone
    cannot tell "the install broke" from "the install is fine". Per-pixel
    spread can: a flat fill has exactly zero.

    Returns the number of failures (0 is a pass).
    """
    from descape import asset_source, terrain_browser, terrain_catalog

    if not asset_source.is_available():
        print("no AoE2DE install configured -- swatches are flat palette colours")

    by_category: dict[str, int] = {}
    for entry in terrain_catalog.terrains():
        by_category.setdefault(entry.category, entry.id)

    expect_detail = asset_source.is_available()
    means: dict[str, tuple[float, float, float]] = {}
    failures = 0
    for category, terrain_id in by_category.items():
        image = terrain_browser.swatch_pixmap(terrain_id).toImage()
        samples = [
            ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF)
            for y in range(image.height())
            for x in range(image.width())
            for p in [image.pixel(x, y)]
        ]
        count = len(samples)
        mean = tuple(sum(s[c] for s in samples) / count for c in range(3))
        means[category] = mean
        spread = max(max(s[c] for s in samples) - min(s[c] for s in samples) for c in range(3))
        flag = ""
        if expect_detail and spread == 0:
            flag = "  FLAT -- expected real texture detail here"
            failures += 1
        print(
            f"  {category:<20} id={terrain_id:<4} "
            f"mean RGB={tuple(round(c) for c in mean)} spread={spread}{flag}"
        )
    if not expect_detail:
        print("  (spread 0 is correct with no install -- every swatch is a flat fill)")

    categories = list(means)
    for i, a in enumerate(categories):
        for b in categories[i + 1 :]:
            if all(abs(x - y) < 2 for x, y in zip(means[a], means[b], strict=True)):
                # With no install these are README's hand-guessed flat colours, which do repeat.
                print(f"  COLLISION: {a} and {b} are the same colour{'' if expect_detail else ' (guessed palette, not counted)'}")
                failures += expect_detail
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-install", action="store_true", help="hide the AoE2DE install: capture the flat-colour fallback"
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        print("PyQt5 not importable -- nothing to capture", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="descape-eyeball-") as tmp:
        # redirect_asset_source=False: get_install_path() reads asset_source's
        # own CONFIG_PATH, and the real install is where the .dds swatches
        # this tool exists to look at come from.
        if args.no_install:
            os.environ.pop("AOE2DE_INSTALL_PATH", None)
        settings_isolation.isolate_settings(Path(tmp), redirect_asset_source=args.no_install)
        _ensure_qapp()
        for path in _capture(args.out):
            print(f"wrote {path}")
        print("category swatch means:")
        collisions = _measure(args.out)
    if collisions:
        print(f"{collisions} swatch check(s) failed -- see the flags above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
