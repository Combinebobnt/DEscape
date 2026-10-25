#!/usr/bin/env python3
"""Regenerates the committed descape/*.json tables that are derived from an
AoE2:DE install, and reports whether they are still current.

Ten generators (GENERATORS below) each parse the game's empires2_x2_p1.dat
and write one or two tables. This tool runs them together into a scratch
directory, starts replacing the committed tables only once every generator
has succeeded, and then records what they were generated from in
tools/game_data_stamp.json: the .dat's sha256 and size, the Steam build id,
the genieutils-py version, and each table's sha256.

Each table, and the stamp, is replaced atomically, but the set is not: a
write failing at one table leaves the tables before it new and the rest, and
the stamp, old. --status and tests/test_regen_game_data.py both flag that
state (a table that differs from the stamp); rerun the regeneration.

    python3 tools/regen_game_data.py              regenerate every table
    python3 tools/regen_game_data.py --only gen_unit_stats.py
    python3 tools/regen_game_data.py --status     is a regeneration due?
    python3 tools/regen_game_data.py --check      does one reproduce the tables?
    python3 tools/regen_game_data.py --dataset ror   only Return of Rome's tables

Two datasets (GH #67), each with its own .dat, tables and stamp section:
aoe2 (resources/_common/dat/, descape/, the stamp's top level) and ror, Return
of Rome (modes/Pompeii/resources/_common/dat/, descape/ror/, the stamp's "ror"
section). Each generator runs with --dataset ror for the second. --dataset
picks aoe2, ror or all (the default); under all, an install without the RoR
.dat looks at the AoE2 tables only, with a note.

--only regenerates a subset and updates only those tables' stamp entries. It
refuses when the install's .dat or build id is not the one the stamp records,
since that would mix two game builds in one set of tables: run the full
regeneration first. Commit a regenerated table together with the stamp
(tests/test_regen_game_data.py fails on a table whose hash the stamp does not
carry).

--status runs no generator and needs no genieutils-py: it compares the
install's .dat and build id, and each committed table, with the stamp. It is
the quick patch detector, not proof: three generators also read install files
beyond the .dat (texture names, .sld headers, the palette). --check is the
proof: --status plus a full regeneration compared with the committed tables.
It writes nothing outside its scratch directory.

Exit codes. --status and --check: 0 current, 1 stale (a missing stamp, a
table or .dat field that differs, or a generator failing under --check), 2
could not look (no install visible, no .dat in it, or genieutils-py missing
for --check). Regenerating: 0 written, 1 nothing written because a generator
failed or --only was refused, 2 could not run.

The install is --install, else the one DEscape itself is configured with
(AOE2DE_INSTALL_PATH, then config.yaml). Table bytes are compared, hashed and
written with CRLF folded to LF, so a Windows run agrees with an LF checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, dat_derive
from descape.content_roots import AOE2DE_STEAM_APP_ID

TOOLS_DIR = ROOT / "tools"
TABLES_DIR = ROOT / "descape"
STAMP_PATH = TOOLS_DIR / "game_data_stamp.json"
DAT_SUBPATH = Path("resources") / "_common" / "dat" / "empires2_x2_p1.dat"
ROR_DAT_SUBPATH = Path("modes") / "Pompeii" / DAT_SUBPATH
ROR_STAMP_KEY = "ror"
ROR_TABLES_SUBDIR = "ror"

# The one registry, shared with the runtime derivation of a data mod's tables (GH #58).
GENERATORS = dat_derive.GENERATORS

DATASET_LABELS = {"aoe2": "Age of Empires II", "ror": "Return of Rome"}

EXIT_OK = 0
EXIT_STALE = 1
EXIT_CANNOT = 2

STAMP_COMMENT = (
    "What the committed descape/*.json game-data tables were generated from, and each table's sha256 "
    "(CRLF folded to LF). `refreshed` is the day a full regeneration first recorded this .dat and build id. "
    "Written by tools/regen_game_data.py: regenerate, never edit by hand."
)
_STALE_VERDICT = "game data is stale: run tools/regen_game_data.py"
_BUILDID = re.compile(r'"buildid"\s+"(\d+)"')


def normalise(data: bytes) -> bytes:
    """The one form table bytes are hashed, compared and written in."""
    return data.replace(b"\r\n", b"\n")


def table_sha256(path: Path) -> str:
    return hashlib.sha256(normalise(path.read_bytes())).hexdigest()


def dat_fingerprint(install: Path, dat_subpath: Path = DAT_SUBPATH) -> dict[str, object] | None:
    dat_path = install / dat_subpath
    if not dat_path.is_file():
        return None
    digest = hashlib.sha256()
    size = 0
    with dat_path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def steam_buildid(install: Path) -> str | None:
    """The build id in the Steam library's app manifest, or None (a Store install has none)."""
    resolved = install.resolve()
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name == "steamapps":
            try:
                text = (ancestor / f"appmanifest_{AOE2DE_STEAM_APP_ID}.acf").read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            match = _BUILDID.search(text)
            return match.group(1) if match else None
    return None


def genieutils_version() -> str | None:
    try:
        return importlib.metadata.version("genieutils-py")
    except importlib.metadata.PackageNotFoundError:
        return None


def _today() -> str:
    return datetime.datetime.now(datetime.UTC).date().isoformat()


@dataclass(frozen=True)
class Context:
    """Everything a run reads or writes, so a test can point all of it at scratch files."""

    tables_dir: Path = TABLES_DIR
    stamp_path: Path = STAMP_PATH
    tools_dir: Path = TOOLS_DIR
    generators: tuple[tuple[str, tuple[str, ...]], ...] = GENERATORS
    genieutils_version: Callable[[], str | None] = genieutils_version
    today: Callable[[], str] = _today
    out: Callable[[str], None] = print
    # Per dataset: dataset_context() replaces these for ror, and tables_dir with descape/ror/.
    dat_subpath: Path = DAT_SUBPATH
    generator_args: tuple[str, ...] = ()
    stamp_key: str | None = None


def dataset_context(ctx: Context, dataset: str) -> Context:
    """ctx for one dataset: aoe2 as given, ror under tables_dir/ror with its own .dat and stamp section."""
    if dataset == "aoe2":
        return ctx
    return replace(
        ctx,
        tables_dir=ctx.tables_dir / ROR_TABLES_SUBDIR,
        dat_subpath=ROR_DAT_SUBPATH,
        generator_args=("--dataset", "ror"),
        stamp_key=ROR_STAMP_KEY,
    )


def outputs_of(generators: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[str, ...]:
    return tuple(name for _script, names in generators for name in names)


def script_for(table: str, generators: Sequence[tuple[str, tuple[str, ...]]] = GENERATORS) -> str | None:
    return next((script for script, names in generators if table in names), None)


def load_stamp(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stamp_text(stamp: dict) -> str:
    return json.dumps(stamp, indent=2, sort_keys=True) + "\n"


def load_section(ctx: Context) -> dict | None:
    """ctx's dataset's part of the stamp: the top level without the ror section, or that section."""
    stamp = load_stamp(ctx.stamp_path)
    if stamp is None:
        return None
    if ctx.stamp_key is None:
        return {key: value for key, value in stamp.items() if key != ROR_STAMP_KEY}
    section = stamp.get(ctx.stamp_key)
    return section if isinstance(section, dict) else None


def merged_stamp(ctx: Context, section: dict) -> dict:
    """The whole stamp with ctx's dataset's part replaced by `section`."""
    stamp = load_stamp(ctx.stamp_path) or {}
    if ctx.stamp_key is not None:
        return {**stamp, ctx.stamp_key: section}
    return {**section, **({ROR_STAMP_KEY: stamp[ROR_STAMP_KEY]} if ROR_STAMP_KEY in stamp else {})}


def _write_atomic(path: Path, data: bytes) -> None:
    # One fixed sibling and no cleanup: a failed write leaves at most this file.
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def stale_against_stamp(
    stamp: dict | None,
    fingerprint: dict[str, object],
    buildid: str | None,
    ctx: Context,
) -> list[str]:
    """One line per way the install or a committed table disagrees with the stamp, each led by
    the stamp field or the table it is about. Empty means current."""
    if stamp is None:
        return [f"{_shown(ctx.stamp_path)}: no readable stamp"]
    recorded = stamp.get("dat") if isinstance(stamp.get("dat"), dict) else {}
    stale = [
        f"dat.{key}: install {fingerprint[key]}, stamp {recorded.get(key)}"
        for key in ("sha256", "size")
        if recorded.get(key) != fingerprint[key]
    ]
    recorded_build = stamp.get("steam_buildid")
    if buildid is not None and recorded_build is not None and buildid != recorded_build:
        stale.append(f"steam_buildid: install {buildid}, stamp {recorded_build}")
    tables = stamp.get("tables") if isinstance(stamp.get("tables"), dict) else {}
    registered = outputs_of(ctx.generators)
    for name in registered:
        path = ctx.tables_dir / name
        if not path.is_file():
            stale.append(f"{name}: missing from {_shown(ctx.tables_dir)}")
        elif name not in tables:
            stale.append(f"{name}: not in the stamp")
        elif tables[name] != table_sha256(path):
            stale.append(f"{name}: differs from the stamp")
    stale.extend(f"{name}: in the stamp, but no generator writes it" for name in sorted(set(tables) - set(registered)))
    return stale


def only_refusal(stamp: dict | None, fingerprint: dict[str, object], buildid: str | None) -> str | None:
    """Why a partial regeneration may not run against this install, or None."""
    if stamp is None:
        return "there is no stamp to compare the install with"
    recorded = stamp.get("dat") if isinstance(stamp.get("dat"), dict) else {}
    if recorded.get("sha256") != fingerprint["sha256"]:
        return f"the install's .dat sha256 is {fingerprint['sha256']}, the stamp records {recorded.get('sha256')}"
    recorded_build = stamp.get("steam_buildid")
    if buildid is not None and recorded_build is not None and buildid != recorded_build:
        return f"the install's Steam buildid is {buildid}, the stamp records {recorded_build}"
    return None


def run_generators(generators: Sequence[tuple[str, tuple[str, ...]]], install: Path, scratch: Path, ctx: Context) -> list[str]:
    """Runs each generator into `scratch`, echoing its output. Returns the scripts that failed."""
    failed = []
    for script, names in generators:
        ctx.out(f"== {script}")
        run = subprocess.run(
            [sys.executable, str(ctx.tools_dir / script), str(install), "--out-dir", str(scratch), *ctx.generator_args],
            capture_output=True, text=True, errors="replace", check=False,
        )
        for stream in (run.stdout, run.stderr):
            if stream.strip():
                ctx.out(stream.rstrip("\n"))
        unwritten = [name for name in names if not (scratch / name).is_file()]
        if run.returncode != 0:
            ctx.out(f"{script}: exited {run.returncode}")
            failed.append(script)
        elif unwritten:
            ctx.out(f"{script}: exited 0 without writing {', '.join(unwritten)}")
            failed.append(script)
    return failed


@contextlib.contextmanager
def _scratch_dir(keep: bool, out: Callable[[str], None]) -> Iterator[Path]:
    if keep:
        kept = Path(tempfile.mkdtemp(prefix="regen_game_data_"))
        try:
            yield kept
        finally:
            out(f"kept the generated tables in {kept}")
        return
    with tempfile.TemporaryDirectory(prefix="regen_game_data_") as tmp:
        yield Path(tmp)


def _read_install(install: Path | None, ctx: Context) -> tuple[dict[str, object], str | None] | None:
    if install is None:
        ctx.out("no AoE2:DE install visible: pass --install or set AOE2DE_INSTALL_PATH")
        return None
    fingerprint = dat_fingerprint(install, ctx.dat_subpath)
    if fingerprint is None:
        ctx.out(f"not found: {install / ctx.dat_subpath}")
        return None
    buildid = steam_buildid(install)
    ctx.out(f"install: {install}")
    ctx.out(f"  {ctx.dat_subpath.as_posix()}: sha256 {fingerprint['sha256']}, {fingerprint['size']} bytes")
    ctx.out(f"  Steam buildid: {buildid if buildid is not None else 'none (no steamapps app manifest)'}")
    return fingerprint, buildid


def _describe_stamp(stamp: dict | None, ctx: Context) -> None:
    if stamp is not None:
        ctx.out(f"stamp: {_shown(ctx.stamp_path)}, refreshed {stamp.get('refreshed')}, genieutils-py {stamp.get('genieutils')}")


def _stale_and_said(stamp: dict | None, seen: tuple[dict[str, object], str | None], ctx: Context) -> list[str]:
    stale = stale_against_stamp(stamp, *seen, ctx)
    if not stale:
        ctx.out(f"  the install and all {len(outputs_of(ctx.generators))} tables in {_shown(ctx.tables_dir)} match the stamp")
    return stale


def _verdict(stale: list[str], ctx: Context) -> int:
    if stale:
        for line in stale:
            ctx.out(f"STALE {line}")
        ctx.out(_STALE_VERDICT)
        return EXIT_STALE
    ctx.out("game data is current")
    return EXIT_OK


def _missing_genieutils(ctx: Context) -> int:
    ctx.out("genieutils-py is not installed, so no generator can run: pip install -r requirements-dev.txt")
    return EXIT_CANNOT


def status(install: Path | None, ctx: Context) -> int:
    seen = _read_install(install, ctx)
    if seen is None:
        return EXIT_CANNOT
    stamp = load_section(ctx)
    _describe_stamp(stamp, ctx)
    return _verdict(_stale_and_said(stamp, seen, ctx), ctx)


def check(install: Path | None, ctx: Context, *, keep: bool = False) -> int:
    seen = _read_install(install, ctx)
    if seen is None:
        return EXIT_CANNOT
    stamp = load_section(ctx)
    _describe_stamp(stamp, ctx)
    version = ctx.genieutils_version()
    if version is None:
        return _missing_genieutils(ctx)
    stale = _stale_and_said(stamp, seen, ctx)
    with _scratch_dir(keep, ctx.out) as scratch:
        failed = run_generators(ctx.generators, install, scratch, ctx)
        stale.extend(f"{script}: the generator failed (its output is above)" for script in failed)
        ctx.out(f"tables in {_shown(ctx.tables_dir)} against a fresh regeneration:")
        for script, names in ctx.generators:
            if script in failed:
                continue
            for name in names:
                committed = ctx.tables_dir / name
                same = committed.is_file() and normalise(committed.read_bytes()) == normalise((scratch / name).read_bytes())
                ctx.out(f"  {name}: {'identical' if same else 'differs'}")
                if not same:
                    stale.append(f"{name}: differs from a fresh regeneration")
    if stamp is not None and stamp.get("genieutils") != version:
        ctx.out(f"note: this run used genieutils-py {version}, the stamp records {stamp.get('genieutils')}")
    return _verdict(stale, ctx)


def regenerate(install: Path | None, ctx: Context, *, only: Sequence[str] = (), keep: bool = False) -> int:
    from descape.scenario_io import is_workshop_subscribed

    for target in (ctx.tables_dir, ctx.stamp_path):
        if is_workshop_subscribed(target):
            ctx.out(f"refusing to write into a subscribed-mods (mods/subscribed/) folder: {target}")
            return EXIT_CANNOT
    seen = _read_install(install, ctx)
    if seen is None:
        return EXIT_CANNOT
    fingerprint, buildid = seen
    stamp = load_section(ctx)
    _describe_stamp(stamp, ctx)
    version = ctx.genieutils_version()
    if version is None:
        return _missing_genieutils(ctx)
    selected = tuple(entry for entry in ctx.generators if not only or entry[0] in only)
    if only:
        refusal = only_refusal(stamp, fingerprint, buildid)
        if refusal is not None:
            ctx.out(f"refusing --only: {refusal}")
            ctx.out("A partial regeneration would mix two game builds. Run tools/regen_game_data.py with no --only first.")
            return EXIT_STALE

    generated: dict[str, bytes] = {}
    with _scratch_dir(keep, ctx.out) as scratch:
        failed = run_generators(selected, install, scratch, ctx)
        if failed:
            ctx.out(f"nothing written: {', '.join(failed)} failed (output above)")
            return EXIT_STALE
        for name in outputs_of(selected):
            generated[name] = normalise((scratch / name).read_bytes())

    ctx.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, data in generated.items():
        target = ctx.tables_dir / name
        if target.is_file() and normalise(target.read_bytes()) == data:
            ctx.out(f"  {name}: unchanged")
        else:
            _write_atomic(target, data)
            ctx.out(f"  {name}: updated")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in generated.items()}

    if only:
        # Only the regenerated tables' entries move: `refreshed` and the rest describe the full run.
        new_stamp = {**stamp, "tables": {**stamp.get("tables", {}), **hashes}}
        if stamp.get("genieutils") != version:
            ctx.out(f"note: this run used genieutils-py {version}, the stamp still records {stamp.get('genieutils')}")
    else:
        same_build = stamp is not None and stamp.get("dat") == fingerprint and stamp.get("steam_buildid") == buildid
        new_stamp = {
            "dat": fingerprint,
            "steam_buildid": buildid,
            "genieutils": version,
            "refreshed": stamp["refreshed"] if same_build and stamp.get("refreshed") else ctx.today(),
            "tables": hashes,
        }
        if ctx.stamp_key is None:
            new_stamp["_comment"] = STAMP_COMMENT
    if new_stamp == stamp:
        ctx.out(f"  {_shown(ctx.stamp_path)}: unchanged")
    else:
        ctx.stamp_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(ctx.stamp_path, stamp_text(merged_stamp(ctx, new_stamp)).encode("utf-8"))
        ctx.out(f"  {_shown(ctx.stamp_path)}: updated")
    ctx.out(f"regenerated {len(generated)} of {len(outputs_of(ctx.generators))} tables")
    return EXIT_OK


def _only_scripts(names: Sequence[str], generators: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[list[str], list[str]]:
    """(--only names as registry script names, the ones the registry does not have)."""
    known = {script for script, _names in generators}
    scripts, unknown = [], []
    for name in names:
        script = Path(name).name
        script = script if script.endswith(".py") else f"{script}.py"
        (scripts if script in known else unknown).append(script)
    return scripts, unknown


def main(argv: Sequence[str] | None = None, *, ctx: Context | None = None) -> int:
    ctx = ctx if ctx is not None else Context()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", type=Path, help="AoE2:DE install root (default: the configured install)")
    parser.add_argument("--tables-dir", type=Path, help="Directory holding the tables (default: descape/)")
    parser.add_argument("--stamp", type=Path, help="The stamp file (default: tools/game_data_stamp.json)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="Compare the install and the tables with the stamp; run nothing")
    mode.add_argument("--check", action="store_true", help="--status, plus a full regeneration compared with the tables")
    mode.add_argument("--only", nargs="+", metavar="SCRIPT", help="Regenerate only these generators' tables, e.g. gen_unit_stats.py")
    parser.add_argument("--keep", action="store_true", help="Leave the scratch directory of generated tables and print its path")
    parser.add_argument(
        "--dataset", choices=("aoe2", "ror", "all"), default="all",
        help="Which game's tables: aoe2, ror (Return of Rome) or all (default)",
    )
    args = parser.parse_args(argv)

    regenerating = not (args.status or args.check)
    if args.keep and args.status:
        parser.error("--keep does nothing with --status, which generates no tables")
    only: list[str] = []
    if args.only:
        only, unknown = _only_scripts(args.only, ctx.generators)
        if unknown:
            parser.error(f"--only: no such generator {', '.join(unknown)} (choose from {', '.join(s for s, _n in ctx.generators)})")

    ctx = replace(
        ctx,
        tables_dir=args.tables_dir if args.tables_dir is not None else ctx.tables_dir,
        stamp_path=args.stamp if args.stamp is not None else ctx.stamp_path,
    )
    if regenerating and ctx.stamp_path.resolve() == STAMP_PATH.resolve() and ctx.tables_dir.resolve() != TABLES_DIR.resolve():
        parser.error("regenerating into --tables-dir needs --stamp too, or the repo's own stamp would describe tables kept elsewhere")
    install = args.install if args.install is not None else asset_source.get_install_path()
    datasets = ["aoe2", "ror"] if args.dataset == "all" else [args.dataset]
    if args.dataset == "all" and install is not None and not (install / ROR_DAT_SUBPATH).is_file():
        if (install / DAT_SUBPATH).is_file():
            ctx.out(f"note: no Return of Rome .dat at {install / ROR_DAT_SUBPATH}: looking at the AoE2 tables only")
        datasets = ["aoe2"]
    worst = EXIT_OK
    for dataset in datasets:
        if len(datasets) > 1:
            ctx.out(f"=== {DATASET_LABELS[dataset]} ===")
        dctx = dataset_context(ctx, dataset)
        if args.status:
            result = status(install, dctx)
        elif args.check:
            result = check(install, dctx, keep=args.keep)
        else:
            result = regenerate(install, dctx, only=only, keep=args.keep)
        worst = max(worst, result)
        if result == EXIT_CANNOT:
            break
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
