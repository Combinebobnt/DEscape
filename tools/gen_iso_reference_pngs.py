#!/usr/bin/env python3
"""Renders 3 elevation-heavy real example files (2_Joan_coop_2,
C2_ElCid_coop_4, C2_ElCid_coop_1) through descape.render's isometric
compositor for a manual eyeball pass -- always writes PNGs, never
pass/fail. See tools/verify_iso_render.py's own module docstring, check
6, for what these three files were originally picked to show.

Extracted from tools/verify_iso_render.py's generate_reference_pngs()
(the pytest migration plan's Ordering step 4) -- it rode along in that
script's checks list without a check_ prefix or a real pass/fail verdict,
so it belongs here as a generator tool, not as a migrated test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import render
from descape.scenario_io import load_map_and_units

# The 3 elevation-heavy real files this tool has always rendered -- matched
# by stem prefix since example filenames carry a version suffix
# (e.g. "2_Joan_coop_2_v0_15.aoe2scenario") that isn't fixed across re-saves.
REFERENCE_STEMS = ("2_Joan_coop_2", "C2_ElCid_coop_4", "C2_ElCid_coop_1")


def generate(scenario_dir: Path, out_dir: Path) -> list[Path]:
    files_by_name = {p.stem: p for p in scenario_dir.glob("*.aoe2scenario")}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in REFERENCE_STEMS:
        matches = [p for stem, p in files_by_name.items() if stem.startswith(name)]
        if not matches:
            raise SystemExit(f"no example file found matching {name!r} in {scenario_dir}")
        path = matches[0]
        scenario = load_map_and_units(path)
        out_path = out_dir / f"{path.stem}_iso.png"
        render.save_png(scenario, str(out_path), isometric=True)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "build" / "iso_preview", help="Output directory")
    args = parser.parse_args()

    written = generate(args.scenario_dir, args.out_dir)
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
