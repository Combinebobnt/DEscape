#!/usr/bin/env python3
"""
Reports where a scenario file stops matching a structure definition.

Why this exists: AoE2ScenarioParser ships structure definitions for DE v1.36 through
v1.58 only, so older files raise UnknownScenarioStructureError before a single section
is parsed. Supporting one of those versions means authoring a structure.json for it,
and that is an iterative job: start from the nearest shipped version, find the first
field that disagrees with the bytes, fix it, repeat. This tool is the measuring
instrument for that loop, so each round is measured rather than guessed.

Point it at a candidate structure with --structure and it reports, per file, the first
section and retriever that diverges, the byte offset parsing reached, and a hex window
around that offset with the plausible integer/string readings annotated. A str32 whose
length field reads as 0x6E6F6974 is instantly diagnosable as a shifted ASCII run.

--trace dumps the full per-section offset table on a file that parses cleanly. Diffing
that table between a known-good file and a candidate is how you find where two field
runs desynchronise, rather than guessing which field to add or drop.

One file per subprocess, deliberately, for the same reason
tools/census_trigger_versions.py does it: AoE2ScenarioParser gates fields with
Support(since=...) ranges compared against the scenario version, and a load permanently
swaps class-level properties (e.g. Unit.caption_string_id) for every later scenario in
the same process. A mixed-version sweep in one interpreter reports load order, not
files. See tests/README.md's "Process-global hazard" note.

Read-only, always. Much of the real corpus lives under a Proton prefix, which
descape/scenario_io.py's FORBIDDEN_WRITE_MARKER exists to protect since it is the
user's only copy. This script has no output path parameter at all.

Usage:
    tools/map_structure_divergence.py --structure descape/versions/DE/v1.21/structure.json \
        ~/games/steam/steamapps/workshop/content/221380
    tools/map_structure_divergence.py --trace examples/

With no --structure, each file is measured against the library's own definition for its
version, which is the tool's self-test: a known-good file must report no divergence.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testkit.scenario_targets import collect_files

# Marks the worker's one machine-readable line, so library chatter on stdout (progress
# lines, multi-thousand-line hex dumps on failure) can't be mistaken for the result.
RECORD_PREFIX = "##DIVERGE##"

STATUS_CLEAN = "CLEAN"  # every section parsed; see `remainder` for trailing bytes
STATUS_DIVERGE = "DIVERGE"  # a section raised, and we located where
STATUS_NOSTRUCT = "NOSTRUCT"  # no structure available for this file's version
STATUS_READFAIL = "READFAIL"  # not a scenario file, or unreadable

DEFAULT_HEX_WINDOW = 48


def _annotate(buf: bytes, offset: int) -> dict:
    """Plausible readings of the bytes at `offset`, to make a misalignment legible.
    A wrong offset usually shows up as an absurd u32 or an ASCII run read as a length."""
    out: dict = {}
    if 0 <= offset <= len(buf) - 4:
        (u,) = struct.unpack_from("<I", buf, offset)
        (s,) = struct.unpack_from("<i", buf, offset)
        (f,) = struct.unpack_from("<f", buf, offset)
        out["u32"] = u
        out["s32"] = s
        out["f32"] = f
        # A str32 reads this same u32 as a length. Show what it would swallow.
        if 0 < u <= 4096 and offset + 4 + u <= len(buf):
            out["as_str32"] = repr(bytes(buf[offset + 4 : offset + 4 + min(u, 64)]))
        out["is_ascii_run"] = all(32 <= b < 127 for b in buf[offset : offset + 4])
    if 0 <= offset <= len(buf) - 8:
        (d,) = struct.unpack_from("<d", buf, offset)
        out["f64"] = d
    return out


def _hex_window(buf: bytes, offset: int, width: int) -> dict:
    lo = max(0, offset - width)
    hi = min(len(buf), offset + width)
    return {
        "before": buf[lo:offset].hex(" "),
        "after": buf[offset:hi].hex(" "),
        "window_start": lo,
    }


def _failed_retriever(section) -> str | None:
    """The first retriever with no data is where set_data_from_generator stopped.
    Retriever._data starts None and is filled in structure order, and the library
    attaches a section to scenario.sections before filling it."""
    if section is None:
        return None
    for name, retriever in section.retriever_map.items():
        if retriever.data is None:
            return name
    return None


def _retriever_offsets(section, start: int) -> list[dict]:
    """Per-retriever byte spans within one section, in structure order.

    Sums len(retriever.get_data_as_bytes()) rather than re-parsing: this reproduces
    the offsets AoE2ScenarioParser already computed, without instrumenting the
    generator. Stops at the first retriever whose .data is None, the same point
    _failed_retriever locates, so it works on a section that only partially
    loaded too -- the section-total offset table alone can't say which field
    inside e.g. DataHeader desyncs, but this can.

    Also stops (rather than propagating) if get_data_as_bytes() itself raises --
    confirmed reachable: a misaligned-but-non-None read upstream can leave a
    retriever holding a value its own serializer refuses (e.g. an OverflowError
    from a bogus 53345-byte Filename), which is a symptom worth truncating the
    trace at, not a reason to lose the whole file's report over."""
    if section is None:
        return []
    offsets = []
    offset = start
    for name, retriever in section.retriever_map.items():
        if retriever.data is None:
            break
        try:
            length = len(retriever.get_data_as_bytes())
        except Exception:
            break
        offsets.append({"retriever": name, "start": offset, "end": offset + length})
        offset += length
    return offsets


def _load_structure(scenario, structure_path: Path | None) -> None:
    """Install a structure on `scenario`, from an explicit file or the library's own.
    AoE2Scenario.structure is a plain writable dict, so assigning is exact rather than
    a workaround; parse fresh per call because the library mutates the dict it returns."""
    if structure_path is not None:
        scenario.structure = json.loads(structure_path.read_text(encoding="utf-8"))
    else:
        scenario._load_structure()


def measure(path: Path, structure_path: Path | None, hex_width: int, trace: bool) -> dict:
    """Walk one file against one structure. Runs in the worker process only."""
    from AoE2ScenarioParser import settings
    from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
    from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
    from AoE2ScenarioParser.scenarios.aoe2_scenario import (
        _decompress_bytes,
        _get_file_version,
        _get_scenario_variant,
    )

    settings.PRINT_STATUS_UPDATES = False
    record: dict = {"path": str(path)}

    try:
        raw = path.read_bytes()
        igen = IncrementalGenerator(str(path), raw)
        version = _get_file_version(igen)
        variant = _get_scenario_variant(igen)
    except Exception as e:
        record.update(status=STATUS_READFAIL, detail=f"{type(e).__name__}: {e}")
        return record

    record["scenario_version"] = version
    scenario = AoE2DEScenario("DE", version, source_location=str(path), name="", variant=variant)

    try:
        _load_structure(scenario, structure_path)
    except Exception as e:
        record.update(status=STATUS_NOSTRUCT, detail=f"{type(e).__name__}: {e}")
        return record

    record["structure"] = str(structure_path) if structure_path else f"library v{version}"

    # The library prints a multi-thousand-line hex dump before re-raising, which is
    # exactly the noise this tool exists to replace with one located line.
    noise = io.StringIO()
    sections: list[dict] = []
    stage = "FileHeader"
    buf = raw
    offset = 0

    try:
        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            scenario._load_header_section(igen)
            sections.append({"section": "FileHeader", "start": 0, "end": igen.progress})
            header_end = igen.progress
            decompressed = _decompress_bytes(igen.get_remaining_bytes())
            data_igen = IncrementalGenerator("Scenario Data", decompressed)
            buf = decompressed
            for name in scenario.structure:
                if name == "FileHeader":
                    continue
                stage = name
                start = data_igen.progress
                scenario._create_and_load_section(name, data_igen)
                entry = {"section": name, "start": start, "end": data_igen.progress}
                if trace:
                    entry["retrievers"] = _retriever_offsets(scenario.sections.get(name), start)
                sections.append(entry)
        record.update(
            status=STATUS_CLEAN,
            header_length=header_end,
            body_bytes=len(decompressed),
            walked=data_igen.progress,
            remainder=len(decompressed) - data_igen.progress,
        )
        if trace:
            record["sections"] = sections
        return record
    except Exception as e:
        # progress may be unset if the failure was in the header itself.
        try:
            offset = data_igen.progress
        except UnboundLocalError:
            offset = igen.progress
            buf = raw
        record.update(
            status=STATUS_DIVERGE,
            section=stage,
            retriever=_failed_retriever(scenario.sections.get(stage)),
            offset=offset,
            buffer="body" if stage != "FileHeader" else "header",
            buffer_bytes=len(buf),
            detail=f"{type(e).__name__}: {str(e)[:160]}",
            hex=_hex_window(buf, offset, hex_width),
            reading=_annotate(buf, offset),
            sections=sections,
        )
        if trace and stage != "FileHeader":
            record["retrievers"] = _retriever_offsets(scenario.sections.get(stage), start)
        return record


def run_worker(path: Path, args) -> dict:
    """Measure one file in a fresh interpreter; never loads anything in this process."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(path)]
    if args.structure:
        cmd += ["--structure", str(args.structure)]
    if args.trace:
        cmd.append("--trace")
    cmd += ["--hex-window", str(args.hex_window)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            return json.loads(line[len(RECORD_PREFIX) :])
    detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"path": str(path), "status": "WORKERFAIL", "detail": f"exit {proc.returncode}: {detail}"}


def _retriever_rows(retrievers: list[dict], indent: str) -> str:
    return "\n".join(
        f"{indent}{r['retriever']:<32s} {r['start']:>9d} .. {r['end']:>9d}  ({r['end'] - r['start']:>9d} bytes)"
        for r in retrievers
    )


def report(record: dict) -> str:
    status = record["status"]
    name = Path(record["path"]).name
    version = record.get("scenario_version", "?")
    if status == STATUS_CLEAN:
        head = (
            f"{status:<9s} v{version:<5s} body={record['body_bytes']:>9d} "
            f"walked={record['walked']:>9d} remainder={record['remainder']:>9d}  {name}"
        )
        if "sections" in record:
            rows = []
            for s in record["sections"]:
                rows.append(
                    f"      {s['section']:<18s} {s['start']:>9d} .. {s['end']:>9d}"
                    f"  ({s['end'] - s['start']:>9d} bytes)"
                )
                if s.get("retrievers"):
                    rows.append(_retriever_rows(s["retrievers"], "          "))
            head += "\n" + "\n".join(rows)
        return head
    if status != STATUS_DIVERGE:
        return f"{status:<9s} v{version:<5s} {record.get('detail', '')}  {name}"

    lines = [
        f"{status:<9s} v{version:<5s} {record['section']}.{record['retriever']} "
        f"@ {record['buffer']} offset {record['offset']} of {record['buffer_bytes']}  {name}",
        f"      {record['detail']}",
    ]
    reading = record.get("reading", {})
    if reading:
        bits = [f"u32={reading.get('u32')}", f"s32={reading.get('s32')}"]
        if reading.get("is_ascii_run"):
            bits.append("ASCII-RUN (likely a shifted string)")
        if "as_str32" in reading:
            bits.append(f"as_str32={reading['as_str32']}")
        lines.append("      reading: " + "  ".join(bits))
    hexw = record.get("hex", {})
    if hexw:
        lines.append(f"      before: {hexw['before']}")
        lines.append(f"      after : {hexw['after']}")
    if record.get("sections"):
        last = record["sections"][-1]
        lines.append(f"      last clean section: {last['section']} ended at {last['end']}")
    if record.get("retrievers"):
        lines.append(f"      {record['section']} retrievers up to the failure:")
        lines.append(_retriever_rows(record["retrievers"], "          "))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", type=Path, nargs="*", help="Directories to recurse, and/or files")
    parser.add_argument(
        "--structure",
        type=Path,
        help="Candidate structure.json to measure against. Default: the library's own "
        "definition for each file's version, which is this tool's self-test.",
    )
    parser.add_argument(
        "--trace", action="store_true", help="On a clean parse, dump the per-section offset table"
    )
    parser.add_argument("--hex-window", type=int, default=DEFAULT_HEX_WINDOW)
    parser.add_argument("--one", type=Path, help=argparse.SUPPRESS)  # internal worker mode
    args = parser.parse_args()

    if args.one is not None:
        record = measure(args.one, args.structure, args.hex_window, args.trace)
        print(RECORD_PREFIX + json.dumps(record))
        return

    if not args.targets:
        parser.error("no targets given")
    if args.structure is not None and not args.structure.is_file():
        parser.error(f"no such structure file: {args.structure}")

    files = collect_files(args.targets, parser.error)
    if not files:
        print(f"No scenario files found in {', '.join(str(t) for t in args.targets)}")
        sys.exit(1)

    records = []
    unexpected = 0
    for path in files:
        try:
            record = run_worker(path, args)
        except subprocess.TimeoutExpired:
            record = {"path": str(path), "status": "WORKERFAIL", "detail": "timed out"}
        records.append(record)
        if record["status"] == "WORKERFAIL":
            unexpected += 1
        print(report(record), flush=True)

    by_status = Counter(r["status"] for r in records)
    print(f"\n{len(files)} file(s)")
    for status, count in sorted(by_status.items()):
        print(f"  {status:<9s} {count:>4d}")

    where = Counter(
        f"{r['section']}.{r['retriever']}" for r in records if r["status"] == STATUS_DIVERGE
    )
    if where:
        print("\nfirst divergence, by field")
        for field, count in where.most_common():
            print(f"  {count:>4d}  {field}")

    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
