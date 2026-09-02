#!/usr/bin/env python3
"""
Censuses trigger names across a corpus of scenario files against the
`[tag]`/divider conventions descape/trigger_organize.py builds on, and
diagnoses where the divider heuristic disagrees with a quick eyeball.

Why this exists: `is_divider()` was fitted on 1570 names from 16 examples/
files and that fit is provisional -- this script re-measures it against a
much larger private scenario collection (not shipped in this repo) before
any panel code gets written, producing two diagnostic lists: near-misses
(names starting with >=2 divider characters that the predicate rejects) and
the accepted divider names themselves, so both false-negative and
false-positive risk can be eyeballed before the heuristic is frozen.

One file per subprocess, deliberately -- same reason as
tools/census_trigger_versions.py: parse_triggers()'s library-side field
gating is class-level and process-global, so a mixed-version sweep in one
process would measure load order, not the files. Read-only, always; never
writes to any scenario file, and never hardcodes a path into the private
planning repo (the one-way-reference rule -- this tool is public, its
inputs are not).

Usage:
    tools/census_trigger_names.py examples/
    tools/census_trigger_names.py ~/some/private/collection --out names.json

Directories are recursed for *.aoe2scenario; named files are taken as-is.
--out writes a JSON artifact ({file: [names in display order]}) for files
that parsed -- meant to be committed wherever the caller's own repo keeps
census artifacts (this repo's own copy lives in the private planning repo,
never here, since real scenario names are third-party content -- see
tools/gen_trigger_fixture.py's docstring for the same licensing question).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape.trigger_organize import is_divider, leading_divider_run, parse_tag
from testkit.scenario_targets import collect_files

RECORD_PREFIX = "##CENSUS##"

STATUS_OK = "OK"
STATUS_UNPARSEABLE = "UNPARSEABLE"


def measure(path: Path) -> dict:
    """Load one file and return its trigger names in display order, or a
    skip reason. Runs in the worker process only."""
    from descape.scenario_io import load_map_and_units, parse_triggers

    record: dict = {"path": str(path)}
    try:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
    except Exception as e:
        record.update(status=STATUS_UNPARSEABLE, detail=f"{type(e).__name__}: {e}")
        return record
    if manager is None:
        record.update(status=STATUS_UNPARSEABLE, detail="parse_triggers() returned None")
        return record

    order = list(manager.trigger_display_order)
    names = [manager.triggers[i].name or "" for i in order]
    record.update(status=STATUS_OK, names=names)
    return record


def run_worker(path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one", str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            return json.loads(line[len(RECORD_PREFIX) :])
    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"path": str(path), "status": "WORKERFAIL", "detail": f"exit {proc.returncode}: {detail}"}


def near_miss(name: str) -> bool:
    """A name starting with >=2 divider characters that is_divider() rejects
    -- the false-negative side of the heuristic (a real header that won't
    group)."""
    return leading_divider_run(name) >= 2 and not is_divider(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", type=Path, nargs="*", help="Directories to recurse, and/or single files")
    parser.add_argument("--out", type=Path, help="Write a JSON artifact of {file: names} for parsed files")
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one is not None:
        print(RECORD_PREFIX + json.dumps(measure(args.one)))
        return

    if not args.targets:
        parser.error("no targets given")

    files = collect_files(args.targets, parser.error)
    if not files:
        print(f"No scenario files found in {', '.join(str(t) for t in args.targets)}")
        sys.exit(1)

    ok = 0
    unparseable = 0
    unexpected = 0
    total_names = 0
    divider_names: list[str] = []
    near_misses: list[str] = []
    tag_counts: dict[str, int] = {}
    artifact: dict[str, list[str]] = {}

    for path in files:
        try:
            record = run_worker(path)
        except subprocess.TimeoutExpired:
            record = {"path": str(path), "status": "WORKERFAIL", "detail": "timed out"}

        status = record["status"]
        if status == STATUS_OK:
            ok += 1
            names = record["names"]
            artifact[record["path"]] = names
            for name in names:
                total_names += 1
                if is_divider(name):
                    divider_names.append(name.strip())
                elif near_miss(name):
                    near_misses.append(name)
                tag = parse_tag(name)
                if tag is not None:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        elif status == STATUS_UNPARSEABLE:
            unparseable += 1
        else:
            unexpected += 1
            print(f"WORKERFAIL {record['path']}: {record.get('detail', '?')}", file=sys.stderr)

    print(f"{len(files)} file(s): {ok} parsed, {unparseable} unparseable, {unexpected} worker failures")
    print(f"{total_names} trigger names across parsed files")
    print(f"{len(divider_names)} divider names, {len(tag_counts)} distinct tags")

    print(f"\nnear-misses ({len(near_misses)}) -- >=2 leading divider chars, rejected:")
    for name in sorted(set(near_misses)):
        print(f"  {name!r}")

    print(f"\naccepted divider names ({len(divider_names)}):")
    for name in sorted(set(divider_names)):
        print(f"  {name!r}")

    if args.out is not None:
        args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
        print(f"\nwrote {len(artifact)} file(s) to {args.out}")

    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
