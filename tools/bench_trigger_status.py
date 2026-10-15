#!/usr/bin/env python3
"""Trigger status pass (GH #166): what it costs on a file's first Triggers-mode
populate, and how much of that populate it is.

Per scenario, once: the trigger and entry counts, and `parse_triggers()`
seconds (gc on, as the app runs it). Then per run, medians printed:

- **pure**: `trigger_status.evaluate_trigger()` over every trigger, through one
  fresh StatusContext (per-context caches cold, as on a first populate).
- **panel**: the panel's own pass, `TriggerPanel._compute_all_statuses()`,
  with no memo: what a first populate runs.
- **first**: `show_scenario()` on a shown TriggerPanel for a document it is
  not showing, so the pass runs in full.
- **memo**: `show_scenario()` again on the same document (the memoized
  repopulate).
- **nopass**: the first populate with the pass stubbed out: the rest of the
  populate.

The context resolves unit ids the way the viewer's does, through a
unit_references.ReferenceIndex. Every line names the Qt platform and the load
average; read medians, and compare two checkouts on one machine at one time.

`--root DIR` imports descape (and testkit) from another checkout, so one copy
of this script times a baseline worktree and a lane branch alike. Settings are
isolated (pin_install_path() + isolate_settings()), so nothing here writes the
real config.yaml.

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COLUMNS = ("pure", "panel", "first", "memo", "nopass")


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "?"


def main() -> None:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenarios", nargs="*", type=Path, default=[root_default / "examples" / "old-allies-final-v2.aoe2scenario"]
    )
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--runs", type=int, default=5, help="Runs per scenario (default 5)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    root = (args.root or root_default).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    settings_isolation.pin_install_path()
    settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="bench-trigger-status-")))
    from PyQt5.QtWidgets import QApplication

    from descape import library_compat, trigger_status
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.trigger_panel import TriggerPanel
    from descape.unit_references import build_reference_index

    qt_window.ensure_qapp()
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    print(f"descape from {root} ({sha.stdout.strip() or '?'}); platform {QApplication.platformName()}; {args.runs} runs")

    for path in args.scenarios:
        loaded = load_map_and_units(path)
        t0 = time.perf_counter()
        manager = parse_triggers(loaded)
        parse_s = time.perf_counter() - t0
        if manager is None:
            print(f"\n{path.name}: parse_triggers() returned None, skipped")
            continue
        triggers = manager.triggers
        entries = sum(len(t.conditions) + len(t.effects) for t in triggers)
        vocabulary = None
        if library_compat.vocabulary_is_available(loaded.scenario_version):
            vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        index = build_reference_index(loaded)

        def context_for(vocab, trigger_ids, index=index):
            return trigger_status.StatusContext(
                vocab, trigger_ids, unit_exists=lambda ref_id: index.get(ref_id) is not None
            )

        print(
            f"\n{path.name}: {len(triggers)} triggers, {entries} entries, vocabulary "
            f"{'yes' if vocabulary is not None else 'no'}; parse_triggers() {parse_s:.1f} s | loadavg {_loadavg()}"
        )
        panel = TriggerPanel()
        panel.status_context = context_for
        # The viewer wires this to its edit model; None (the bare panel's default) disables the memo.
        panel.edited_trigger_indices = frozenset
        panel.resize(1500, 900)
        panel.show()
        QApplication.processEvents()
        rows: list[dict[str, float]] = []
        try:
            for run in range(args.runs):
                load = _loadavg()
                row = {}
                ids = frozenset(range(len(triggers)))
                context = context_for(vocabulary, ids)
                t0 = time.perf_counter()
                for trigger in triggers:
                    trigger_status.evaluate_trigger(trigger, context)
                row["pure"] = (time.perf_counter() - t0) * 1000

                panel.show_scenario(loaded)
                QApplication.processEvents()
                t0 = time.perf_counter()
                panel._compute_all_statuses(manager)
                row["panel"] = (time.perf_counter() - t0) * 1000

                panel.show_scenario(None)
                QApplication.processEvents()
                t0 = time.perf_counter()
                panel.show_scenario(loaded)
                row["first"] = (time.perf_counter() - t0) * 1000
                QApplication.processEvents()

                t0 = time.perf_counter()
                panel.show_scenario(loaded)
                row["memo"] = (time.perf_counter() - t0) * 1000
                QApplication.processEvents()

                panel.show_scenario(None)
                QApplication.processEvents()
                panel._compute_all_statuses = lambda _manager, same_document=False, p=panel: p._reset_statuses()
                try:
                    t0 = time.perf_counter()
                    panel.show_scenario(loaded)
                    row["nopass"] = (time.perf_counter() - t0) * 1000
                finally:
                    del panel._compute_all_statuses
                QApplication.processEvents()
                panel.show_scenario(None)
                QApplication.processEvents()
                rows.append(row)
                cells = "  ".join(f"{c} {row[c]:7.1f}" for c in COLUMNS)
                print(f"  run {run + 1}  {cells}  | loadavg {load}")
            print("  median " + "  ".join(f"{c} {statistics.median(r[c] for r in rows):.1f}" for c in COLUMNS) + " (ms)")
        finally:
            panel.close()


if __name__ == "__main__":
    main()
