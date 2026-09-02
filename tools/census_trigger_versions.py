#!/usr/bin/env python3
"""
Censuses the (scenario version, trigger version) pairs across a corpus of scenario
files, and reports which of them AoE2ScenarioParser would refuse to load outright.

Why this exists: a trigger editor needs the library to parse the
Triggers section, which is exactly what descape/scenario_io.py exists to avoid. The
library's gate is AoE2Scenario._validate_latest_trigger_data_version()
(AoE2ScenarioParser/scenarios/aoe2_scenario.py:214):

    if self.scenario_version == '1.54' and trigger_version < 4.1:

-- scenario version 1.54 *only*. A 1.37/trigger-2.2 or 1.41/trigger-2.4 file loads
through the library fine; only the narrow 1.54-with-old-triggers combination is
blocked. This script measures how much of a real corpus falls in that band, so v4's
scope rests on a count rather than an impression.

No new parsing code is needed to get the trigger version: the library itself peeks it
as the first 8 bytes of the Triggers section, and load_map_and_units()'s trigger_tail
begins at exactly that offset -- so struct.unpack("<d", tail[:8])[0] is the whole
measurement.

One file per subprocess, deliberately. tools/verify_batch_api.py's docstring documents
a real library bug where loading any pre-1.54 scenario in a process permanently swaps
a class-level property (Unit.caption_string_id) for every later >=1.54 scenario in the
same process. A mixed-version sweep of hundreds of files hits that squarely, and the
results would be artifacts of load order rather than of the files. Each file is loaded
in a fresh interpreter (--one, the internal single-file mode) so no cross-file state
can exist at all.

Read-only, always. Much of the real corpus lives under a Proton prefix -- the path
scenario_io.FORBIDDEN_WRITE_MARKER = "compatdata" (descape/scenario_io.py:74) exists to
protect, since it is the user's only copy of those files. This script only ever loads.

Usage:
    tools/census_trigger_versions.py examples/ examples/play_Test
    tools/census_trigger_versions.py ~/games

Directories are recursed for *.aoe2scenario. Files named explicitly are accepted
whatever they're called -- examples/play_Test is a real 1.54/3.9 scenario with no
extension, which the glob alone would miss (hence the two-argument form above, which
is what covers all 17 example files).

--reverse processes the file list back-to-front. Output content is identical either
way (only line order changes), which is the check that the per-subprocess isolation
really is doing its job:

    tools/census_trigger_versions.py examples/ examples/play_Test | sort > /tmp/a
    tools/census_trigger_versions.py --reverse examples/ examples/play_Test | sort > /tmp/b
    diff /tmp/a /tmp/b

Exit code is non-zero only on an *unexpected* failure (a worker subprocess that
crashed or printed nothing parseable). A file the library simply cannot load -- a
non-DE .scn/.scx, or a scenario version with no versions/DE/vX.YY/ structure
definition -- is a result category, counted and reported, not an error.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testkit.scenario_targets import collect_files

# Marks the worker's one machine-readable line, so library chatter on stdout (version
# warnings, progress lines) can't be mistaken for the result.
RECORD_PREFIX = "##CENSUS##"

# A real trigger version is a small positive double (2.2, 2.4, 3.9, 4.1, 4.5).
# Anything outside this band means trigger_tail does not begin where we think it
# does for that structure version -- report it as an outlier instead of feeding
# garbage into the histogram.
TRIGGER_VERSION_MIN = 1.0
TRIGGER_VERSION_MAX = 20.0

# Statuses. OK/BLOCKED mean the file was measured; the rest are result categories.
STATUS_OK = "OK"
STATUS_BLOCKED = "BLOCKED"
STATUS_LOAD_FAIL = "LOADFAIL"
STATUS_SHORT_TAIL = "SHORTTAIL"
STATUS_ODD_VERSION = "ODDVER"


def measure(path: Path) -> dict:
    """Load one file and return its census record. Runs in the worker process only."""
    from descape.scenario_io import load_map_and_units

    record: dict = {"path": str(path)}
    try:
        s = load_map_and_units(path)
    except Exception as e:
        record.update(status=STATUS_LOAD_FAIL, detail=f"{type(e).__name__}: {e}")
        return record

    record.update(
        scenario_version=s.scenario_version,
        map_size=f"{s.map_manager.map_width}x{s.map_manager.map_height}",
        tail_bytes=len(s.trigger_tail),
    )

    if len(s.trigger_tail) < 8:
        record.update(status=STATUS_SHORT_TAIL, detail="tail shorter than one double")
        return record

    (trigger_version,) = struct.unpack("<d", s.trigger_tail[:8])
    record["trigger_version"] = trigger_version
    if not TRIGGER_VERSION_MIN <= trigger_version <= TRIGGER_VERSION_MAX:
        record.update(status=STATUS_ODD_VERSION, detail=f"implausible value {trigger_version!r}")
        return record

    blocked = s.scenario_version == "1.54" and trigger_version < 4.1
    record["status"] = STATUS_BLOCKED if blocked else STATUS_OK
    return record


def run_worker(path: Path) -> dict:
    """Measure one file in a fresh interpreter; never loads anything in this process."""
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
    return {
        "path": str(path),
        "status": "WORKERFAIL",
        "detail": f"exit {proc.returncode}: {detail}",
    }


def format_line(record: dict) -> str:
    status = record["status"]
    version = record.get("scenario_version", "?")
    trigger = record.get("trigger_version")
    trigger_str = f"{trigger:.1f}" if isinstance(trigger, float) else "?"
    body = (
        f"v{version:<5s} trig {trigger_str:<4s} "
        f"{record.get('map_size', '?'):>9s}  tail={record.get('tail_bytes', 0):>9d}"
    )
    if "detail" in record:
        body += f"  {record['detail']}"
    return f"{status:<9s} {body}  {record['path']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "targets", type=Path, nargs="*", help="Directories to recurse, and/or single files"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Process the file list back-to-front"
    )
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
    if args.reverse:
        files.reverse()

    records = []
    unexpected = 0
    for path in files:
        try:
            record = run_worker(path)
        except subprocess.TimeoutExpired:
            record = {"path": str(path), "status": "WORKERFAIL", "detail": "timed out"}
        records.append(record)
        if record["status"] == "WORKERFAIL":
            unexpected += 1
        print(format_line(record), flush=True)

    by_status = Counter(r["status"] for r in records)
    pairs = Counter(
        (r["scenario_version"], round(r["trigger_version"], 2))
        for r in records
        if r["status"] in (STATUS_OK, STATUS_BLOCKED)
    )

    print(f"\n{len(files)} file(s)")
    for status in (
        STATUS_OK,
        STATUS_BLOCKED,
        STATUS_LOAD_FAIL,
        STATUS_SHORT_TAIL,
        STATUS_ODD_VERSION,
        "WORKERFAIL",
    ):
        if by_status[status]:
            print(f"  {status:<9s} {by_status[status]:>4d}")

    print("\nscenario/trigger version histogram")
    for (scenario_version, trigger_version), count in sorted(pairs.items()):
        blocked = scenario_version == "1.54" and trigger_version < 4.1
        mark = "  <- blocked by the library" if blocked else ""
        print(f"  {scenario_version} / {trigger_version:<4} {count:>4d}{mark}")

    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
