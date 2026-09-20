#!/usr/bin/env python3
"""
Censuses PlayerDataTwo's `ai_type` against its companion `ai_names` and the
embedded per-file text, to test whether the field DEscape's Players panel
labels "Player Type" is really the AI *mode* selector (0 = custom,
1 = standard, 2 = none) rather than the in-game Human/Computer/Either row.

Why this exists: `ai_type` is read-only in the Players panel with its three
observed values labelled `Unknown (0..2)`, because nobody had identified the
byte. The format spec for the AoK/AoC-era SCX layout names it, but DE
collapsed that era's three-string AI sub-struct into a single
`AIStruct = {unknown, ai_per_file_text}`, and an imported semantic can shift
exactly at that joint. So the claim is measured here before anything is
renamed: if `ai_type == 0` really means "custom AI", those players should be
the ones carrying a non-empty `ai_names` entry and a non-empty embedded
`.per` text, and `ai_type` in {1, 2} should carry neither.

`human`/`active` come along from DataHeader's `player_data_1` because the
competing reading is that `ai_type` *is* the Human/Computer/Either control;
a file where the two disagree discriminates between the two readings.

Per-file text is recorded as a LENGTH, never as the text: a custom AI embeds
its whole `.per` source, which runs to megabytes across a corpus.

Every array is emitted at its raw 16-wide index, not mapped to P1..P8 + GAIA.
Where GAIA sits inside `ai_type`/`ai_names` is assumed by analogy in
descape/player_fields.py rather than measured, and this tool exists to
measure things, so it must not bake that assumption into its own output.

One file per subprocess, deliberately -- same reason as
tools/census_trigger_names.py: the library's field gating is class-level and
process-global, so a mixed-version sweep in one interpreter would report load
order rather than files. Read-only, always; never writes to any scenario file,
and never hardcodes a path into the private planning repo (the one-way-
reference rule -- this tool is public, its inputs are not).

Usage:
    tools/census_player_ai.py examples/
    tools/census_player_ai.py ~/some/private/collection --out player_ai.json

Directories are recursed for *.aoe2scenario; named files are taken as-is, so
the .scn/.scx members of a mixed collection need naming explicitly (and
pre-DE files will not load at all). --out writes a JSON artifact meant for
wherever the caller's own repo keeps census artifacts -- never this one, since
real scenario names are third-party content.
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

STATUS_OK = "OK"
STATUS_UNPARSEABLE = "UNPARSEABLE"

# The predicted table this census is a go/no-go gate on. Anything outside it
# on a non-default row means the AoK-to-DE struct collapse changed the meaning.
CUSTOM, STANDARD, NONE = 0, 1, 2
_MODE_NAMES = {CUSTOM: "custom", STANDARD: "standard", NONE: "none"}
_RESERVED_NAMES = {"", "standard", "none", "random"}

# Rows with active == 0 carry no information about the field: they are either
# GAIA or the unwritten 9..15 filler slots, whose stored ai_type is whatever the
# writer left there rather than a chosen mode. The literal-table check below
# still counts them, because the plan's pass criterion was written before this
# was known; every distribution block restricts to the active rows and says so.


def _retrievers(scenario, section_name: str) -> dict | None:
    section = scenario.sections.get(section_name)
    return None if section is None else section.retriever_map


def _values(retrievers: dict | None, name: str) -> list | None:
    if retrievers is None:
        return None
    retriever = retrievers.get(name)
    if retriever is None or retriever.data is None:
        return None
    return list(retriever.data)


def _struct_field(entries: list | None, field: str) -> list | None:
    """`field` out of each entry of a parsed struct array, or None if the
    array or the field is absent on this version."""
    if entries is None:
        return None
    out = []
    for entry in entries:
        retriever = entry.retriever_map.get(field)
        if retriever is None:
            return None
        out.append(retriever.data)
    return out


def measure(path: Path) -> dict:
    """Read one file's AI-mode columns, or a skip reason. Worker process only."""
    from descape.scenario_io import load_map_and_units

    record: dict = {"path": str(path)}
    try:
        loaded = load_map_and_units(path)
    except Exception as e:
        record.update(status=STATUS_UNPARSEABLE, detail=f"{type(e).__name__}: {e}")
        return record

    scenario = loaded._scenario
    pd2 = _retrievers(scenario, "PlayerDataTwo")
    header = _retrievers(scenario, "DataHeader")
    per_text = _struct_field(_values(pd2, "ai_files"), "ai_per_file_text")
    player_data_1 = _values(header, "player_data_1")

    record.update(
        status=STATUS_OK,
        scenario_version=loaded.scenario_version,
        ai_type=_values(pd2, "ai_type"),
        ai_name=_values(pd2, "ai_names"),
        per_text_len=None if per_text is None else [len(t or "") for t in per_text],
        human=_struct_field(player_data_1, "human"),
        active=_struct_field(player_data_1, "active"),
        civilization=_struct_field(player_data_1, "civilization"),
        tribe_name=_values(header, "tribe_names"),
    )
    return record


def run_worker(path: Path) -> dict:
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


def row_violation(mode: int, name: str, text_len: int) -> str | None:
    """Why this (ai_type, ai_name, per_text_len) row contradicts the predicted
    table, or None if it fits. A row only fits `custom` if it carries the
    evidence of a custom AI; the reserved names are what the in-game
    Personality list offers for the other two modes."""
    named = name.strip().lower() not in _RESERVED_NAMES
    if mode == CUSTOM:
        if not named:
            return "ai_type 0 (custom) with no custom ai_name"
        if text_len == 0:
            return "ai_type 0 (custom) with no embedded per-text"
        return None
    if mode in (STANDARD, NONE):
        if named:
            return f"ai_type {mode} ({_MODE_NAMES[mode]}) with custom ai_name {name!r}"
        if text_len:
            return f"ai_type {mode} ({_MODE_NAMES[mode]}) with {text_len} bytes of per-text"
        return None
    return f"ai_type {mode} outside the documented 0/1/2 range"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", type=Path, nargs="*", help="Directories to recurse, and/or single files")
    parser.add_argument("--out", type=Path, help="Write a JSON artifact of the per-file columns")
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
    incomplete: list[str] = []
    mode_counts: dict[int, int] = {}
    violations: list[str] = []
    mode_by_human: dict[tuple[int, int], int] = {}
    artifact: dict[str, dict] = {}
    active_names: dict[int, dict[str, int]] = {}
    active_lens: dict[int, list[int]] = {}
    inactive_modes: dict[int, int] = {}

    for path in files:
        try:
            record = run_worker(path)
        except subprocess.TimeoutExpired:
            record = {"path": str(path), "status": "WORKERFAIL", "detail": "timed out"}

        status = record["status"]
        if status == STATUS_UNPARSEABLE:
            unparseable += 1
            continue
        if status != STATUS_OK:
            unexpected += 1
            print(f"WORKERFAIL {record['path']}: {record.get('detail', '?')}", file=sys.stderr)
            continue

        ok += 1
        name = Path(record["path"]).name
        artifact[record["path"]] = {k: v for k, v in record.items() if k != "path"}

        modes, names, lens = record["ai_type"], record["ai_name"], record["per_text_len"]
        if modes is None or names is None or lens is None:
            incomplete.append(name)
            continue

        active, human = record["active"], record["human"]
        for i, mode in enumerate(modes):
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            why = row_violation(mode, names[i] or "", lens[i])
            if why is not None:
                violations.append(f"{name} [{i}]: {why} (ai_name={names[i]!r}, per_text_len={lens[i]})")
            if active is not None and i < len(active) and active[i]:
                bucket = active_names.setdefault(mode, {})
                bucket[names[i] or ""] = bucket.get(names[i] or "", 0) + 1
                active_lens.setdefault(mode, []).append(lens[i])
                if human is not None and i < len(human):
                    key = (mode, int(bool(human[i])))
                    mode_by_human[key] = mode_by_human.get(key, 0) + 1
            else:
                inactive_modes[mode] = inactive_modes.get(mode, 0) + 1

    print(f"{len(files)} file(s): {ok} parsed, {unparseable} unparseable, {unexpected} worker failures")
    if incomplete:
        print(f"{len(incomplete)} parsed file(s) missing one of ai_type/ai_names/ai_files: {incomplete}")
    print(f"ai_type value counts across all 16 indices: {dict(sorted(mode_counts.items()))}")
    # The competing reading is that ai_type IS the Human/Computer/Either row, which
    # would make this table diagonal. Active rows only: a filler slot's ai_type and
    # human are both whatever the writer left, so counting those measures nothing.
    print(f"(ai_type, human) on active rows: {dict(sorted(mode_by_human.items()))}")

    non_default = sum(count for mode, count in mode_counts.items() if mode != STANDARD)
    print(f"\n{len(violations)} row(s) contradict the predicted table; {non_default} non-default row(s):")
    for line in violations[:40]:
        print(f"  {line}")
    if len(violations) > 40:
        print(f"  ... and {len(violations) - 40} more")

    active_rows = sum(len(v) for v in active_lens.values())
    print(f"\nactive == 1 rows only ({active_rows} of {sum(mode_counts.values())}):")
    for mode in sorted(active_names):
        lens = sorted(active_lens[mode])
        median = lens[len(lens) // 2] if lens else 0
        print(f"  ai_type={mode}: {len(lens)} rows, per_text_len median={median}, {len(active_names[mode])} distinct ai_name(s)")
        for ai_name, count in sorted(active_names[mode].items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
            print(f"      {count:>4}x {ai_name!r}")
    print(f"  (inactive rows, excluded above, by ai_type: {dict(sorted(inactive_modes.items()))})")

    collisions = {
        ai_name: sorted(mode for mode, bucket in active_names.items() if ai_name in bucket)
        for ai_name in {n for bucket in active_names.values() for n in bucket}
    }
    collisions = {n: modes for n, modes in collisions.items() if len(modes) > 1}
    print(f"\n{len(collisions)} ai_name(s) appearing under more than one ai_type on active rows:")
    for ai_name, modes in sorted(collisions.items()):
        print(f"  {ai_name!r}: {modes}")

    if violations:
        verdict = "FAILS -- the predicted table does not hold as written"
    elif non_default == 0:
        verdict = "HOLDS VACUOUSLY -- no discriminating rows in this corpus"
    else:
        verdict = "HOLDS -- every non-default row fits the predicted table"
    print(f"\nGate (literal predicted table): {verdict}")
    print("Name-class partition on active rows: " + ("clean" if not collisions else f"{len(collisions)} cross-ai_type name collision(s)"))

    if args.out is not None:
        args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
        print(f"wrote {len(artifact)} file(s) to {args.out}")

    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
