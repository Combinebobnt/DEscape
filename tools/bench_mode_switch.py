#!/usr/bin/env python3
"""Mode switch to Units: what entering Units mode costs, and how much of it is
the pick index (`unit_index`) versus the Units panel page switch (`left_page`).

Per run, on a fresh load of the scenario in one terrain style, with View >
Footprint Outlines off and then on:

- **first**: the first entry to Units mode after the load (no index yet).
- **reentry**: leave to Terrain and come back, with no unit edit in between.
- **after_edit**: leave to Terrain, undo a Place there (a unit edit made
  outside Units mode), and come back.

Each entry prints the wall time of the mode switch (the combo's
setCurrentText(), which runs the whole switch synchronously) and the Perf
Trace `perf op mode-switch` line's `unit_index` and `left_page` phases.

The platform is whatever Qt picks: in-app numbers came from Xvfb, so run it
there and name the platform with every number. One shell command, Xvfb on TCP:

    Xvfb :57 -listen tcp -nolisten unix -screen 0 1920x1200x24 & XPID=$!; sleep 1; \\
    DISPLAY=127.0.0.1:57 .venv/bin/python3 tools/bench_mode_switch.py; \\
    kill $XPID; wait $XPID

`--root DIR` imports descape (and testkit) from another checkout, so one copy
of this script times a baseline worktree and a lane branch alike. Settings are
isolated (pin_install_path() + isolate_settings(), then the module globals
pinned), so nothing here writes the real config.yaml.

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail. Read medians.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ENTRIES = ("first", "reentry", "after_edit")
COLUMNS = ("switch", "unit_index", "left_page")
_TOWN_CENTRE = 109


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "?"


def _op_phases(line: str) -> dict[str, float]:
    """`name ms` pairs from a `perf op` line's phase section."""
    parts = line.split(" | ")
    if len(parts) < 2:
        return {}
    return {name: float(ms) for name, ms in re.findall(r"(\S+) (-?\d+(?:\.\d+)?)", parts[1])}


def main() -> None:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", nargs="?", type=Path, default=root_default / "examples" / "old-allies-final-v2.aoe2scenario"
    )
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--runs", type=int, default=5, help="Runs per outlines setting (default 5)")
    parser.add_argument("--style", default="Stepped", choices=("Flat", "Stepped", "Sloped"))
    parser.add_argument("--outlines", default="off,on", help="Comma-separated: off, on (default off,on)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    outline_states = [s.strip() for s in args.outlines.split(",") if s.strip()]
    if not outline_states or any(s not in ("off", "on") for s in outline_states):
        parser.error("--outlines takes off and/or on")

    root = (args.root or root_default).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    settings_isolation.pin_install_path()
    settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="bench-mode-switch-")))
    from PyQt5.QtWidgets import QApplication

    import descape.settings as settings_module
    from descape import debug_log, perf_trace

    settings_module._footprint_outlines = False
    settings_module._stack_badges = True
    perf_trace.enable(True)
    qt_window.ensure_qapp()
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    print(
        f"descape from {root} ({sha.stdout.strip() or '?'}); {args.scenario.name}; {args.style}; "
        f"platform {QApplication.platformName()}; {args.runs} runs"
    )

    def enter(window, mode: str) -> dict[str, float]:
        perf_trace.flush_pending_op()
        debug_log.clear()
        t0 = time.perf_counter()
        window.mode_combo.setCurrentText(mode)
        switch_ms = (time.perf_counter() - t0) * 1000
        perf_trace.flush_pending_op()
        lines = [ln for ln in debug_log.get_log_text().splitlines() if "perf op mode-switch" in ln]
        phases = _op_phases(lines[-1]) if lines else {}
        QApplication.processEvents()
        assert window.mode == mode.lower(), f"mode switch to {mode} refused"
        nan = float("nan")
        return {"switch": switch_ms, "unit_index": phases.get("unit_index", nan), "left_page": phases.get("left_page", nan)}

    window = qt_window.stepped_window(args.scenario, show=True)
    try:
        if args.style != "Stepped":
            window.terrain_style_combo.setCurrentText(args.style)
            QApplication.processEvents()
        for state in outline_states:
            window.footprint_action.setChecked(state == "on")
            QApplication.processEvents()
            rows: dict[str, list[dict[str, float]]] = {name: [] for name in ENTRIES}
            print(f"\noutlines {state}")
            for run in range(args.runs):
                window.edit_history.mark_saved()
                window.load_scenario(args.scenario)
                QApplication.processEvents()
                assert window.map_view._unit_index is None or state == "on", "a fresh load kept an index"
                load = _loadavg()
                row = {"first": enter(window, "Units")}
                view = window.map_view
                count = sum(len(units) for units in window.scenario.unit_manager.units)
                window.units_panel.select_object(_TOWN_CENTRE)
                mid = (window.scenario.map_manager.map_width // 2, window.scenario.map_manager.map_height // 2)
                window.on_unit_place(view._tile_polygon(*mid).boundingRect().center(), None)
                assert sum(len(u) for u in window.scenario.unit_manager.units) == count + 1, "the place did not happen"
                enter(window, "Terrain")
                row["reentry"] = enter(window, "Units")
                enter(window, "Terrain")
                window.undo()
                QApplication.processEvents()
                assert sum(len(u) for u in window.scenario.unit_manager.units) == count, "the undo did not happen"
                row["after_edit"] = enter(window, "Units")
                enter(window, "Terrain")
                for name in ENTRIES:
                    rows[name].append(row[name])
                cells = "  ".join(
                    f"{name} " + " ".join(f"{row[name][c]:6.1f}" for c in COLUMNS) for name in ENTRIES
                )
                print(f"  run {run + 1}  {cells}  | loadavg {load}")
            print(f"  columns per entry: {' '.join(COLUMNS)} (ms)")
            for name in ENTRIES:
                med = "  ".join(
                    f"{c} {statistics.median(r[c] for r in rows[name]):.1f} "
                    f"[{min(r[c] for r in rows[name]):.1f}-{max(r[c] for r in rows[name]):.1f}]"
                    for c in COLUMNS
                )
                print(f"  median {name:<10} {med}")
    finally:
        window.edit_history.mark_saved()
        window.close()


if __name__ == "__main__":
    main()
