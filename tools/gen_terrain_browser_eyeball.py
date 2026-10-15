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


_GRID_WIDE_WIDTH = 600


def _pixel_spread(image) -> int:
    samples = [image.pixel(x, y) for y in range(image.height()) for x in range(image.width())]
    return max(max((p >> s) & 0xFF for p in samples) - min((p >> s) & 0xFF for p in samples) for s in (16, 8, 0))


def _grid_report(window, label: str, expect_detail: bool) -> int:
    """GH #161 grid checks a screenshot cannot settle; returns the failure count.

    Every shown cell has an icon (with real texture detail when an install is
    configured), every caption's wrapped text rect fits its cell, headings
    are left-aligned in WindowText, and the armed cell carries Highlight on
    all four sides with its swatch untinted. Offscreen Qt geometry."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QFontMetrics, QPalette

    from descape import asset_source
    from descape.swatch_grid import _TEXT_FLAGS, PAD

    grid = window.terrain_panel.view.grid
    grid.flush_swatches()
    image = grid.viewport().grab().toImage()
    palette = grid.palette()
    metrics = QFontMetrics(grid.caption_font())
    failures = []
    shown = [v for v in grid.values() if not grid.cell_item(v).isHidden()]
    for value in shown:
        cell = grid.cell_item(value)
        icon = cell.icon()
        if icon.isNull():
            failures.append(f"no icon on {cell.text()}")
        elif expect_detail and _pixel_spread(icon.pixmap(64, 64).toImage()) == 0:
            # g_bla.dds and g_kf1.dds really are solid black; only a fallback fill is a failure.
            if asset_source.get_terrain_thumbnail(value, 64) is None:
                failures.append(f"flat fallback swatch on {cell.text()}")
            else:
                print(f"  [{label}] note: {cell.text()}'s real texture is a single colour")
        rect = grid.visualItemRect(cell)
        box = grid.caption_rect(rect, cell.text())
        used = metrics.boundingRect(QRect(box.left(), box.top(), box.width(), 100_000), _TEXT_FLAGS, cell.text())
        if not rect.contains(used):
            failures.append(f"caption overflows on {cell.text()}: {used} outside {rect}")
    text = palette.color(QPalette.WindowText).rgb()
    viewport = grid.viewport().rect()
    for heading in grid.heading_items():
        rect = grid.visualItemRect(heading)
        if heading.isHidden() or not rect.intersects(viewport):
            continue
        xs = [x for y in range(rect.top(), rect.bottom()) for x in range(rect.left(), rect.right())
              if image.valid(x, y) and image.pixel(x, y) == text]
        if not xs:
            failures.append(f"heading {heading.text()!r} has no WindowText pixels")
        elif min(xs) - rect.left() > PAD + 4:
            failures.append(f"heading {heading.text()!r} starts {min(xs) - rect.left()} px in, not left-aligned")
    armed = grid.current_value()
    cell = grid.cell_item(armed) if armed is not None else None
    if cell is not None and not cell.isHidden() and grid.visualItemRect(cell).intersects(viewport):
        rect = grid.visualItemRect(cell)
        highlight = palette.color(QPalette.Highlight).rgb()
        mid = rect.center()
        sides = [(rect.left(), mid.y()), (mid.x(), rect.top()), (rect.right(), mid.y()), (mid.x(), rect.bottom())]
        if not all(image.pixel(x, y) == highlight for x, y in sides):
            failures.append("armed cell is missing Highlight on a side")
        swatch = grid.swatch_rect(rect)
        source = cell.icon().pixmap(grid.swatch_px(), grid.swatch_px()).toImage()
        drawn = image.copy(swatch)
        if any(drawn.pixel(x, y) != source.pixel(x, y) for x, y in ((0, 0), (swatch.width() // 2, swatch.height() // 2))):
            failures.append("armed swatch is tinted")
    else:
        print(f"  [{label}] armed cell not on screen; frame check skipped")
    tops = [grid.visualItemRect(grid.cell_item(v)).top() for v in shown]
    per_row = max(tops.count(t) for t in set(tops)) if tops else 0
    print(
        f"  [{label}] cells shown={len(shown)} cell_w={grid.cell_width()} per_row={per_row} "
        f"content height={grid.verticalScrollBar().maximum() + grid.viewport().height()} "
        f"hscroll max={grid.horizontalScrollBar().maximum()} failures={len(failures)}"
    )
    for failure in failures:
        print(f"    FAIL {failure}")
    return len(failures)


def _capture_grid(out_dir: Path) -> tuple[list[Path], int]:
    """Grid mode at MIN_USEFUL_WIDTH and 600 px, at each swatch size, once with
    show-hidden ticked, and once under a dark theme (offscreen Qt)."""
    from PyQt5.QtWidgets import QApplication

    from descape import asset_source, settings
    from descape.viewer_dialogs import apply_theme

    expect_detail = asset_source.is_available()
    window = _open_window()
    written: list[Path] = []
    failures = 0
    panel = window.terrain_panel
    try:
        panel.set_display_mode("grid")
        panel.set_terrain(next(v for v in panel.view.grid.values() if v != panel.terrain_id()))
        for width in (panel.MIN_USEFUL_WIDTH, _GRID_WIDE_WIDTH):
            _set_left_width(window, width)
            for px in settings.TERRAIN_GRID_SWATCH_SIZES:
                panel.set_grid_swatch_px(px)
                QApplication.processEvents()
                failures += _grid_report(window, f"grid w{width} {px}px", expect_detail)
                path = out_dir / f"grid_w{width}_{px}px.png"
                panel.grab().save(str(path))
                written.append(path)
        _set_left_width(window, panel.MIN_USEFUL_WIDTH)
        panel.set_grid_swatch_px(settings.TERRAIN_GRID_SWATCH_PX_DEFAULT)
        panel.view.show_hidden_checkbox.setChecked(True)
        panel.view.grid.scrollToBottom()
        QApplication.processEvents()
        failures += _grid_report(window, "grid show-hidden", expect_detail)
        path = out_dir / "grid_show_hidden_bottom.png"
        panel.grab().save(str(path))
        written.append(path)
        panel.view.show_hidden_checkbox.setChecked(False)
        panel.set_terrain(panel.terrain_id())
        panel.view.grid.set_current_value(panel.terrain_id())
        settings.set_theme("dark")
        apply_theme(QApplication.instance(), "dark", {})
        QApplication.processEvents()
        failures += _grid_report(window, "grid dark", expect_detail)
        path = out_dir / "grid_dark.png"
        panel.grab().save(str(path))
        written.append(path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    return written, failures


def _grid_timing() -> int:
    """Shows the grid on the real install with cold caches and lets the event
    loop fill it: prints the time from first show to the last icon and the
    longest single tick."""
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import asset_source

    if not asset_source.is_available():
        print("--grid-timing needs a configured AoE2DE install", file=sys.stderr)
        return 1
    from descape.terrain_panel import TerrainPanel

    asset_source.clear_install_caches()
    panel = TerrainPanel(display_mode="grid")
    panel.resize(440, 900)
    grid = panel.view.grid
    ticks: list[float] = []
    real_tick = grid._tick

    def timed_tick() -> None:
        start = time.perf_counter()
        real_tick()
        ticks.append(time.perf_counter() - start)

    grid._timer.timeout.disconnect()
    grid._timer.timeout.connect(timed_tick)
    start = time.perf_counter()
    panel.show()
    while True:
        QApplication.processEvents()
        if not grid.fill_pending() and all(
            grid.cell_item(v).isHidden() or not grid.cell_item(v).icon().isNull() for v in grid.values()
        ):
            break
    total = time.perf_counter() - start
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    print(
        f"grid fill: {total:.2f} s from show to last icon over {len(ticks)} ticks, "
        f"longest tick {max(ticks) * 1000:.1f} ms, load {load:.1f}"
    )
    panel.close()
    return 0


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
    parser.add_argument(
        "--grid-timing", action="store_true",
        help="time the grid's progressive fill on the real install with cold caches (GH #161), then exit",
    )
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
        if args.grid_timing:
            return _grid_timing()
        for path in _capture(args.out):
            print(f"wrote {path}")
        print("grid mode:")
        grid_paths, grid_failures = _capture_grid(args.out)
        for path in grid_paths:
            print(f"wrote {path}")
        print("category swatch means:")
        collisions = _measure(args.out) + grid_failures
    if collisions:
        print(f"{collisions} swatch check(s) failed -- see the flags above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
