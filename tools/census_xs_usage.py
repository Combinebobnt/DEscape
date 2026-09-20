#!/usr/bin/env python3
"""
Censuses XS script usage across a corpus of scenario files: the trigger-level
Script Call bodies (effect `message`, condition `xs_function`) and the
file-level attachment (Map `script_name`, Files `script_file_content`).

Why this exists: docs/XS_SCRIPTING.md and the decision to keep the file-level
surface read-only both rest on how real scenarios use XS. This makes those
counts reproducible instead of taken on trust.

Script Call is matched by numeric type: effect 55 and condition 25, the same
ids in every shipped DE structure version.

One file per subprocess, deliberately, for the same cross-version
class-poisoning reason tools/census_trigger_versions.py documents: loading a
pre-1.54 scenario mutates library class state for every later file in the
process.

Read-only, always. Much of the real corpus lives under a Proton prefix; this
script only ever loads.

Usage:
    .venv/bin/python3 tools/census_xs_usage.py examples/ examples/play_Test
    .venv/bin/python3 tools/census_xs_usage.py ~/games

Exit code is non-zero only on an unexpected failure (a worker subprocess that
crashed or printed nothing parseable). A file that cannot be loaded, or whose
Triggers section cannot be parsed, is a counted result category.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testkit.scenario_targets import collect_files

RECORD_PREFIX = "##CENSUS##"

SCRIPT_CALL_EFFECT = 55
SCRIPT_CALL_CONDITION = 25

STATUS_OK = "OK"
STATUS_LOAD_FAIL = "LOADFAIL"
STATUS_NO_TRIGGERS = "NOTRIG"


def measure(path: Path) -> dict:
    """Load one file and return its census record. Runs in the worker process only."""
    from descape.scenario_io import load_map_and_units, parse_triggers, xs_attachment

    record: dict = {"path": str(path)}
    try:
        s = load_map_and_units(path)
    except Exception as e:
        record.update(status=STATUS_LOAD_FAIL, detail=f"{type(e).__name__}: {e}")
        return record

    record["scenario_version"] = s.scenario_version
    script_name, _ = xs_attachment(s)
    record["script_name"] = script_name

    manager = parse_triggers(s)
    if manager is None:
        record["status"] = STATUS_NO_TRIGGERS
        return record

    _, embedded = xs_attachment(s)
    effects = conditions = payload = 0
    for trigger in manager.triggers:
        for effect in trigger.effects:
            if effect.effect_type == SCRIPT_CALL_EFFECT and effect.message:
                effects += 1
                payload += len(effect.message)
        for condition in trigger.conditions:
            if condition.condition_type == SCRIPT_CALL_CONDITION and condition.xs_function:
                conditions += 1
                payload += len(condition.xs_function)
    record.update(
        status=STATUS_OK,
        embedded_chars=embedded,
        effects=effects,
        conditions=conditions,
        payload_chars=payload,
    )
    return record


def run_worker(path: Path) -> dict:
    """Measure one file in a fresh interpreter; never loads anything in this process."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one", str(path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,  # the caller reads returncode/stdout and reports itself
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            return json.loads(line[len(RECORD_PREFIX) :])
    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"path": str(path), "status": "WORKERFAIL", "detail": f"exit {proc.returncode}: {detail}"}


def format_line(record: dict) -> str:
    status = record["status"]
    body = f"v{record.get('scenario_version', '?'):<5s}"
    if status == STATUS_OK:
        body += (
            f" effects={record['effects']:<3d} conditions={record['conditions']:<3d}"
            f" chars={record['payload_chars']:<6d}"
            f" script_name={record['script_name']!r} embedded={record['embedded_chars']}"
        )
    elif "script_name" in record:
        body += f" script_name={record['script_name']!r}"
    if "detail" in record:
        body += f"  {record['detail']}"
    return f"{status:<10s} {body}  {record['path']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("targets", type=Path, nargs="*", help="Directories to recurse, and/or single files")
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)  # internal worker mode
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

    records = []
    for path in files:
        try:
            record = run_worker(path)
        except subprocess.TimeoutExpired:
            record = {"path": str(path), "status": "WORKERFAIL", "detail": "timed out"}
        records.append(record)
        print(format_line(record), flush=True)

    ok = [r for r in records if r["status"] == STATUS_OK]
    loadable = [r for r in records if r["status"] in (STATUS_OK, STATUS_NO_TRIGGERS)]
    with_xs = [r for r in ok if r["effects"] or r["conditions"]]

    print(f"\n{len(files)} file(s)")
    for status in (STATUS_OK, STATUS_NO_TRIGGERS, STATUS_LOAD_FAIL, "WORKERFAIL"):
        count = sum(1 for r in records if r["status"] == status)
        if count:
            print(f"  {status:<10s} {count:>4d}")
    print(f"\nloadable files:              {len(loadable)}")
    print(f"  with trigger-level XS:     {len(with_xs)}")
    print(f"  Script Call effects:       {sum(r['effects'] for r in ok)}")
    print(f"  Script Call conditions:    {sum(r['conditions'] for r in ok)}")
    print(f"  total XS payload (chars):  {sum(r['payload_chars'] for r in ok)}")
    print(f"  non-empty script_name:     {sum(1 for r in loadable if r.get('script_name'))}")
    print(f"  non-empty embedded script: {sum(1 for r in ok if r.get('embedded_chars'))}")

    sys.exit(1 if any(r["status"] == "WORKERFAIL" for r in records) else 0)


if __name__ == "__main__":
    main()
