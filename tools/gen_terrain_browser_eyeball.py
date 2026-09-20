#!/usr/bin/env python3
"""Screenshots TerrainBrowseDialog for a manual eyeball pass, and measures
the one thing a screenshot cannot settle: that the swatches are actually
different textures rather than 131 identical green squares.

Run outside pytest deliberately. tests/conftest.py's _isolated_settings
hides the configured AoE2DE install, which is the right default for the
test suite (the no-install fallback is what it pins) but means the real
.dds swatches -- the whole point of this browser -- are never exercised
there. This tool is where they are.

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


def _capture(out_dir: Path, terrain_id: int) -> list[Path]:
    from PyQt5.QtWidgets import QApplication

    from descape.terrain_browser import TerrainBrowseDialog

    written: list[Path] = []
    dialog = TerrainBrowseDialog(terrain_id)
    dialog.resize(520, 560)
    try:
        QApplication.processEvents()
        path = out_dir / "opened_on_current.png"
        dialog.grab().save(str(path))
        written.append(path)

        dialog.filter_edit.setText("snow")
        QApplication.processEvents()
        path = out_dir / "filtered_snow.png"
        dialog.grab().save(str(path))
        written.append(path)

        dialog.filter_edit.setText("")
        dialog.show_hidden_checkbox.setChecked(True)
        dialog.tree.expandAll()
        QApplication.processEvents()
        path = out_dir / "all_expanded_with_unused.png"
        dialog.grab().save(str(path))
        written.append(path)
    finally:
        dialog.close()
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
                print(f"  COLLISION: {a} and {b} are the same colour")
                failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain-id", type=int, default=0, help="terrain the dialog opens on (default GRASS_1)"
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
        settings_isolation.isolate_settings(Path(tmp), redirect_asset_source=False)
        _ensure_qapp()
        for path in _capture(args.out, args.terrain_id):
            print(f"wrote {path}")
        print("category swatch means:")
        collisions = _measure(args.out)
    if collisions:
        print(f"{collisions} swatch check(s) failed -- see the flags above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
