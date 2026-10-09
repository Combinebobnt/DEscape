#!/usr/bin/env python3
"""
Censuses which condition and effect fields real scenarios actually set, per
type, to seed descape/trigger_requirements.json (GH #166).

For each type name and each field it lists, counts the corpus entries that
list the field in their version (`listed`) and the ones whose value counts
as set (`set`), using descape/trigger_status.is_set(), the same predicate the
status checks use. The `area` and `location` pseudo-fields are counted the
same way through trigger_status.token_state(). Each type also gets its total
`uses`, the number of files using it, and the scenario versions seen.

--status also runs trigger_status over every entry with the committed
requirements table and reports, per type, how many entries it flags PROBLEM
and why, plus every flagged entry. That is the table's false-positive audit:
each flagged entry is either a table fix or a genuinely broken trigger.

One file per subprocess, deliberately, the same as
tools/census_trigger_names.py: the library's field gating is class-level and
process-global, so a mixed-version sweep in one process would measure load
order, not the files. Read-only, always; never writes to a scenario file.
The output path is an argument and never hardcoded here, since the corpus and
the artifact it produces are third-party content kept outside this repo.

Usage:
    tools/census_trigger_fields.py examples/ --out fields.json
    tools/census_trigger_fields.py examples/ more_scenarios/ --status --jobs 2 --out status.json

Directories are recursed for *.aoe2scenario; named files are taken as-is.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testkit.scenario_targets import collect_files

RECORD_PREFIX = "##CENSUS##"

STATUS_OK = "OK"
STATUS_UNPARSEABLE = "UNPARSEABLE"
STATUS_NO_VOCABULARY = "NO_VOCABULARY"
STATUS_WORKERFAIL = "WORKERFAIL"

KINDS = (("condition", "conditions", "condition_type"), ("effect", "effects", "effect_type"))
PSEUDO_TOKENS = ("area", "location")
OBJECTIVE_FLAGS = ("display_as_objective", "header", "display_on_screen", "enabled")


def _type_record() -> dict:
    return {"uses": 0, "fields": {}, "pseudo": {}, "field_kinds": {}}


def _bump(pair_map: dict, key: str, is_set: bool) -> None:
    pair = pair_map.setdefault(key, [0, 0])
    pair[0] += 1
    pair[1] += int(is_set)


def measure(path: Path, with_status: bool) -> dict:
    """Load one file and return its per-type field counts (and, with
    `with_status`, its flagged entries), or a skip reason. Worker only."""
    from descape import library_compat, trigger_status
    from descape.scenario_io import load_map_and_units, parse_triggers
    from descape.unit_references import build_reference_index

    record: dict = {"path": str(path)}
    try:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
    except Exception as e:
        record.update(status=STATUS_UNPARSEABLE, detail=f"{type(e).__name__}: {e}")
        return record
    record["version"] = loaded.scenario_version
    if manager is None:
        record.update(status=STATUS_UNPARSEABLE, detail="parse_triggers() returned None")
        return record
    if not library_compat.vocabulary_is_available(loaded.scenario_version):
        record.update(status=STATUS_NO_VOCABULARY, triggers=len(manager.triggers))
        return record

    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    index = build_reference_index(loaded)
    context = trigger_status.StatusContext(
        vocabulary=vocabulary,
        trigger_ids=frozenset(range(len(manager.triggers))),
        unit_exists=lambda ref_id: ref_id in index.by_id,
    )
    types: dict[str, dict[str, dict]] = {"conditions": {}, "effects": {}}
    unknown: dict[str, dict[str, int]] = {"conditions": {}, "effects": {}}
    flagged: list[dict] = []
    no_effects: list[dict] = []
    triggers ={"total": len(manager.triggers), "dividers": 0, "no_effects": 0, "problem": 0}

    for trigger_index, trigger in enumerate(manager.triggers):
        for kind, kind_key, type_attribute in KINDS:
            for entry in list(trigger_status.read_value(trigger, kind_key) or []):
                definition = trigger_status.definition_for(kind, entry, vocabulary)
                if definition is None:
                    type_id = str(trigger_status.read_value(entry, type_attribute))
                    unknown[kind_key][type_id] = unknown[kind_key].get(type_id, 0) + 1
                    continue
                row = types[kind_key].setdefault(definition.name, _type_record())
                row["uses"] += 1
                for spec in trigger_status.field_specs_for(kind, definition, vocabulary):
                    value = trigger_status.read_value(entry, spec.attribute)
                    _bump(row["fields"], spec.name, trigger_status.is_set(spec, value))
                    row["field_kinds"][spec.name] = spec.kind
                for token in PSEUDO_TOKENS:
                    state = trigger_status.token_state(kind, token, entry, context)
                    if state is not None:
                        _bump(row["pseudo"], token, state)

        if not with_status:
            continue
        result = trigger_status.evaluate_trigger(trigger, context)
        rollup = result.trigger.status
        # A divider row is never styled, but its entry rows are, so they are still audited.
        divider = rollup is trigger_status.Status.NONE
        triggers["dividers"] += divider
        if rollup is trigger_status.Status.PROBLEM:
            triggers["problem"] += 1
        if not divider and not list(trigger_status.read_value(trigger, "effects") or []):
            triggers["no_effects"] += 1
            no_effects.append(
                {
                    "trigger": trigger_index,
                    "trigger_name": str(trigger_status.read_value(trigger, "name") or ""),
                    "conditions": len(list(trigger_status.read_value(trigger, "conditions") or [])),
                    # An objective-display trigger may carry no effects on purpose.
                    **{flag: bool(trigger_status.read_value(trigger, flag)) for flag in OBJECTIVE_FLAGS},
                }
            )
        for kind, kind_key, flagged_type_attribute in KINDS:
            entries = list(trigger_status.read_value(trigger, kind_key) or [])
            for entry_index, (entry, status) in enumerate(zip(entries, getattr(result, kind_key), strict=True)):
                if status.status is not trigger_status.Status.PROBLEM:
                    continue
                definition = trigger_status.definition_for(kind, entry, vocabulary)
                flagged.append(
                    {
                        "trigger": trigger_index,
                        "trigger_name": str(trigger_status.read_value(trigger, "name") or ""),
                        "divider": divider,
                        "kind": kind_key,
                        "entry": entry_index,
                        "type": definition.name if definition is not None else None,
                        "type_id": trigger_status.read_value(entry, flagged_type_attribute),
                        "reasons": list(status.reasons),
                    }
                )

    record.update(status=STATUS_OK, types=types, unknown=unknown)
    if with_status:
        record.update(triggers=triggers, flagged=flagged, no_effects=no_effects)
    return record


def run_worker(path: Path, with_status: bool) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--one", str(path)]
    if with_status:
        command.append("--status")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return {"path": str(path), "status": STATUS_WORKERFAIL, "detail": "timed out"}
    for line in proc.stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            return json.loads(line[len(RECORD_PREFIX) :])
    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"path": str(path), "status": STATUS_WORKERFAIL, "detail": f"exit {proc.returncode}: {detail}"}


def _reason_shape(reason: str) -> str:
    """A reason with its ids replaced by N, so counts aggregate across entries."""
    return re.sub(r"-?\d+", "N", reason)


def merge(records: list[dict], with_status: bool) -> dict:
    """The corpus-wide artifact from per-file worker records."""
    out: dict = {
        "files": {},
        "conditions": {},
        "effects": {},
        "unknown_types": {"conditions": {}, "effects": {}},
    }
    by_status: dict[str, int] = {}
    file_rows = []
    for record in records:
        status = record["status"]
        by_status[status] = by_status.get(status, 0) + 1
        file_rows.append({k: record[k] for k in ("path", "status", "version", "detail") if k in record})
        if status != STATUS_OK:
            continue
        for kind_key in ("conditions", "effects"):
            for name, row in record["types"][kind_key].items():
                merged = out[kind_key].setdefault(
                    name, {"uses": 0, "files": 0, "versions": [], "fields": {}, "pseudo": {}, "field_kinds": {}}
                )
                merged["uses"] += row["uses"]
                merged["files"] += 1
                if record["version"] not in merged["versions"]:
                    merged["versions"].append(record["version"])
                for section in ("fields", "pseudo"):
                    for field, (listed, set_count) in row[section].items():
                        pair = merged[section].setdefault(field, {"listed": 0, "set": 0})
                        pair["listed"] += listed
                        pair["set"] += set_count
                for field, kind in row["field_kinds"].items():
                    merged["field_kinds"].setdefault(field, kind)
            for type_id, count in record["unknown"][kind_key].items():
                bucket = out["unknown_types"][kind_key]
                bucket[type_id] = bucket.get(type_id, 0) + count

    out["files"] = {"by_status": by_status, "list": file_rows}
    if with_status:
        out["status"] = _merge_status(records)
    return out


def _merge_status(records: list[dict]) -> dict:
    totals = {"triggers": 0, "dividers": 0, "no_effects": 0, "problem_triggers": 0, "problem_entries": 0}
    per_type: dict[str, dict[str, dict]] = {"conditions": {}, "effects": {}}
    flagged = []
    no_effects = []
    for record in records:
        if record["status"] != STATUS_OK:
            continue
        no_effects += [{"path": record["path"], **item} for item in record["no_effects"]]
        triggers = record["triggers"]
        totals["triggers"] += triggers["total"]
        totals["dividers"] += triggers["dividers"]
        totals["no_effects"] += triggers["no_effects"]
        totals["problem_triggers"] += triggers["problem"]
        for item in record["flagged"]:
            totals["problem_entries"] += 1
            name = item["type"] if item["type"] is not None else f"type {item['type_id']}"
            row = per_type[item["kind"]].setdefault(name, {"problems": 0, "reasons": {}})
            row["problems"] += 1
            for reason in item["reasons"]:
                shape = _reason_shape(reason)
                row["reasons"][shape] = row["reasons"].get(shape, 0) + 1
            flagged.append({"path": record["path"], **item})
    return {"totals": totals, "per_type": per_type, "flagged": flagged, "no_effect_triggers": no_effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", type=Path, nargs="*", help="Directories to recurse, and/or single files")
    parser.add_argument("--out", type=Path, help="Write the JSON artifact here")
    parser.add_argument("--status", action="store_true", help="Also audit trigger_status over every entry")
    parser.add_argument("--jobs", type=int, default=1, help="Worker subprocesses at once (default 1)")
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one is not None:
        print(RECORD_PREFIX + json.dumps(measure(args.one, args.status)))
        return

    if not args.targets:
        parser.error("no targets given")
    files = collect_files(args.targets, parser.error)
    if not files:
        print(f"No scenario files found in {', '.join(str(t) for t in args.targets)}")
        sys.exit(1)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        records = list(pool.map(lambda path: run_worker(path, args.status), files))

    artifact = merge(records, args.status)
    counts = artifact["files"]["by_status"]
    print(f"{len(files)} file(s): " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    for kind_key in ("conditions", "effects"):
        rows = artifact[kind_key]
        print(f"{kind_key}: {len(rows)} type(s) used, {sum(r['uses'] for r in rows.values())} entries")
    for record in records:
        if record["status"] == STATUS_WORKERFAIL:
            print(f"WORKERFAIL {record['path']}: {record.get('detail', '?')}", file=sys.stderr)
    if args.status:
        totals = artifact["status"]["totals"]
        print("status: " + ", ".join(f"{k}={v}" for k, v in totals.items()))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    sys.exit(1 if counts.get(STATUS_WORKERFAIL) else 0)


if __name__ == "__main__":
    main()
