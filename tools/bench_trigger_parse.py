#!/usr/bin/env python3
"""The first trigger parse: what `scenario_io.parse_triggers()` costs on a
document whose triggers were never read, standalone and in the app.

Standalone (the default), per scenario, in a fresh process each: the load,
the viewer's own `gc.collect(); gc.freeze()`, then the parse, printing the
trigger and entry counts, the parse seconds split into sections (the
library's `_create_and_load_section()` for Triggers and Files) and construct
(`TriggerManager.construct()`), the process CPU seconds (steadier than wall
clock on a loaded machine; `--app` prints them too), RSS before and after, one full
`gc.collect()` afterwards, timed, and the load average.

`--app` instead times the first `mode_combo.setCurrentText("Triggers")` on a
shown ViewerWindow, once the load's warms have drained and their gc hold
(`gc_hold`) has released, with Perf Trace's
`panel` phase, then the same RSS and `gc.collect()` figures. `--saves N`
(with `--app`) then renames one trigger and saves to a temporary path, N
times, printing RSS after each save.

`--app` runs offscreen unless QT_QPA_PLATFORM says otherwise (testkit's
default); for Xvfb, set QT_QPA_PLATFORM=xcb and run on TCP as
`tools/bench_mode_switch.py` documents. Name the platform with every number.

`--root DIR` imports descape (and testkit) from another checkout, so one copy
of this script times a baseline worktree and a branch alike. Settings are
isolated, so nothing here writes the real config.yaml.

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail.
"""

from __future__ import annotations

import argparse
import gc
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "?"


def _rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def _timed_collect() -> float:
    t0 = time.perf_counter()
    gc.collect()
    return time.perf_counter() - t0


def _counts(manager) -> tuple[int, int]:
    triggers = manager.triggers
    return len(triggers), sum(len(t.conditions) + len(t.effects) for t in triggers)


def _standalone(scenario: Path) -> None:
    from descape import scenario_io

    loaded = scenario_io.load_map_and_units(str(scenario))
    gc.collect()
    gc.freeze()
    lib = loaded._scenario
    spent = {"sections": 0.0, "construct": 0.0}
    original_section = lib._create_and_load_section
    manager_cls = scenario_io.TriggerManager
    original_construct = manager_cls.__dict__.get("construct")
    inherited_construct = manager_cls.construct

    def section(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_section(*args, **kwargs)
        finally:
            spent["sections"] += time.perf_counter() - t0

    def construct(cls, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return inherited_construct.__func__(cls, *args, **kwargs)
        finally:
            spent["construct"] += time.perf_counter() - t0

    lib._create_and_load_section = section
    manager_cls.construct = classmethod(construct)
    rss0 = _rss_mb()
    load = _loadavg()
    try:
        t0, c0 = time.perf_counter(), time.process_time()
        manager = scenario_io.parse_triggers(loaded)
        parse_s, cpu_s = time.perf_counter() - t0, time.process_time() - c0
    finally:
        del lib._create_and_load_section
        if original_construct is None:
            del manager_cls.construct
        else:
            manager_cls.construct = original_construct
    rss1 = _rss_mb()
    collect_s = _timed_collect()
    triggers, entries = _counts(manager) if manager is not None else (0, 0)
    print(
        f"{scenario.name}: {triggers:,} triggers, {entries:,} entries; parse {parse_s:.2f}s, cpu {cpu_s:.2f}s "
        f"(sections {spent['sections']:.2f}, construct {spent['construct']:.2f}); "
        f"RSS {rss0:.0f} -> {rss1:.0f} MB; gc.collect after {collect_s * 1000:.0f} ms; "
        f"read {loaded.trigger_read_supported}; loadavg {load}"
    )


def _app(scenario: Path, saves: int) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import debug_log, gc_hold, perf_trace, scenario_write, trigger_fields
    from testkit import qt_window

    perf_trace.enable(True)
    window = qt_window.stepped_window(scenario, show=True)
    try:
        deadline = time.perf_counter() + 600
        warmers = (window._level_warmer, window._load_warmer, window._margin_warmer)
        while any(w.is_active for w in warmers) and time.perf_counter() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        # Until the warms' gc hold releases too: inside one gen 2 never collects, unlike a click after a pause.
        while gc_hold.is_engaged() and time.perf_counter() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        for _ in range(50):
            QApplication.processEvents()
        perf_trace.flush_pending_op()
        debug_log.clear()
        rss0 = _rss_mb()
        load = _loadavg()
        t0, c0 = time.perf_counter(), time.process_time()
        window.mode_combo.setCurrentText("Triggers")
        switch_s, cpu_s = time.perf_counter() - t0, time.process_time() - c0
        perf_trace.flush_pending_op()
        lines = [ln for ln in debug_log.get_log_text().splitlines() if "perf op mode-switch" in ln]
        panel = re.search(r"\bpanel (\d+(?:\.\d+)?)", lines[-1]) if lines else None
        rss1 = _rss_mb()
        collect_s = _timed_collect()
        manager = window.scenario._trigger_manager
        triggers, entries = _counts(manager) if manager is not None else (0, 0)
        print(
            f"{scenario.name} --app ({QApplication.platformName()}): {triggers:,} triggers, {entries:,} entries; "
            f"first Triggers entry {switch_s:.2f}s, cpu {cpu_s:.2f}s (panel {panel.group(1) if panel else '?'} ms); "
            f"RSS {rss0:.0f} -> {rss1:.0f} MB; gc.collect after {collect_s * 1000:.0f} ms; "
            f"mode {window.mode}; loadavg {load}"
        )
        name_spec = next(spec for spec in trigger_fields.TRIGGER_FIELDS if spec.name == "name")
        out_dir = Path(tempfile.mkdtemp(prefix="bench-trigger-parse-"))
        for n in range(saves):
            window.set_trigger_field(0, name_spec, f"bench rename {n}")
            t0 = time.perf_counter()
            scenario_write.write_scenario(
                window.scenario, str(out_dir / f"save{n}.aoe2scenario"), backup=False, **window._edit_model_kwargs()
            )
            window.edit_history.mark_saved()
            QApplication.processEvents()
            print(f"  save {n + 1}: {time.perf_counter() - t0:.2f}s, RSS {_rss_mb():.0f} MB")
    finally:
        window.edit_history.mark_saved()
        window.close()


def main() -> None:
    root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", type=Path, default=[root_default / "examples" / "old-allies-final-v2.aoe2scenario"])
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--app", action="store_true", help="time the first Triggers-mode entry in a shown window")
    parser.add_argument("--saves", type=int, default=0, help="with --app: rename-and-save N times, RSS after each")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.saves and not args.app:
        parser.error("--saves needs --app")
    root = (args.root or root_default).resolve()
    if not args.one:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        print(f"descape from {root} ({sha.stdout.strip() or '?'})", flush=True)
        for scenario in args.scenarios:
            # A fresh process per file: the parse's class-level state and frozen heap never carry over.
            cmd = [sys.executable, __file__, "--one", "--root", str(root), "--saves", str(args.saves), str(scenario)]
            subprocess.run(cmd + (["--app"] if args.app else []), check=False)
        return

    sys.path.insert(0, str(root))
    from testkit import settings_isolation

    settings_isolation.pin_install_path()
    settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="bench-trigger-parse-")))
    if args.app:
        from testkit import qt_window

        qt_window.ensure_qapp()
        _app(args.scenarios[0], args.saves)
    else:
        _standalone(args.scenarios[0])


if __name__ == "__main__":
    main()
