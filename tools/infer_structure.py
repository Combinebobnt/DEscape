#!/usr/bin/env python3
"""Infers a new scenario version's structure from game-saved files, for maintainers.

When AoE2:DE patches the scenario format and no structure definition exists
for the new version, `infer` searches for the layout change from the newest
known structure and writes `structure.json` plus `INFERRED.md` (inputs,
each edit with its evidence and rivals, notes a release needs) to
build/structure_infer/v<version>/. The search is tools/structure_infer.py;
the library itself checks the result before anything is written.

`suggest-reference DIR` ranks the base version's files in DIR by how many
fields carry a non-default value, so the file the maintainer re-saves in the
patched editor pins down the most positions.

Read-only on every input. Writes only into --out, only once every check
passes, never into a mods/subscribed/ folder. Exit codes: 0 INFERRED, 1
AMBIGUOUS or REFUSED (nothing written), 2 could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import game_dataset, games
from descape import structure_walk as sw
from descape.scenario_io import is_workshop_subscribed
from testkit.scenario_targets import collect_files
from tools import structure_infer as si

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "build" / "structure_infer"
OUTPUT_FILES = ("structure.json", "INFERRED.md")
# Units and Triggers fields count this many times in suggest-reference: most bumps insert there.
SUGGEST_WEIGHT = {"Units": 3, "Triggers": 3}

PROCEDURE = """maintainer procedure:
  1. suggest-reference DIR (holding files of the newest known version): re-save the top file, unedited, in the patched editor.
  2. infer NEW_FILES... --reference <that file> <its re-save>; samples may be left out with --reference.
  3. INFERRED: read INFERRED.md in --out. AMBIGUOUS: save one in-game edit of the new feature as a second file, add --diff-pair.
  4. REFUSED: author the structure by hand, measuring each round with tools/map_structure_divergence.py.
  5. Ship it as a release: copy structure.json into descape/versions/DE/v<version>/ with tests and a PROVENANCE.md."""


class CannotRun(Exception):
    """An input or option the tool can't use: exit 2."""


def _header_variant(f: si.ScenarioFile, structure: dict) -> int | None:
    """FileHeader.unknown_value_2 under the base header, or None when the walk doesn't reach it."""
    try:
        walked = sw.walk({sw.HEADER_SECTION: structure[sw.HEADER_SECTION]}, f.raw, 0, [sw.HEADER_SECTION])
    except sw.WalkError as e:
        walked = e.partial
    col = walked.columns.get((sw.HEADER_SECTION, "unknown_value_2")) if walked is not None else None
    return int(col.value(0)) if col is not None and len(col) else None


def game_of(files: Sequence[si.ScenarioFile], structure: dict) -> tuple[games.Game, list[str]]:
    """The one game every new-version input belongs to, and any notes."""
    notes = []
    found: dict[str, list[str]] = {}
    for f in files:
        variant = _header_variant(f, structure)
        if variant is None:
            notes.append(
                f"{f.path.name}: FileHeader.unknown_value_2 not reached under the base header; taken as {games.default().display_name}."
            )
        found.setdefault(games.for_variant(variant).id, []).append(f.path.name)
    if len(found) > 1:
        listed = "; ".join(f"{games.get(g).display_name}: {', '.join(names)}" for g, names in found.items())
        raise CannotRun(f"the inputs come from more than one game ({listed})")
    return games.get(next(iter(found))), notes


def terrain_table(game: games.Game) -> Path:
    """The game's own terrain table, never a loaded data mod's."""
    return game_dataset.table_path("terrain_classes.json", dataset=game.id)


def terrain_ids(game: games.Game) -> set[int]:
    path = terrain_table(game)
    try:
        return {int(k) for k in json.loads(path.read_text(encoding="utf-8"))["classes"]}
    except (OSError, ValueError, KeyError) as e:
        raise CannotRun(f"{path}: {e}") from e


def _check_out(out: Path, force: bool) -> None:
    if is_workshop_subscribed(out):
        raise CannotRun(f"{out} is inside a mods/subscribed/ folder, which the game's mod manager re-syncs")
    if out.exists() and not out.is_dir():
        raise CannotRun(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()) and not force:
        raise CannotRun(f"{out} is not empty (--force replaces {' and '.join(OUTPUT_FILES)} in it)")


def write_outputs(out: Path, texts: dict[str, str]) -> None:
    """Every file to a temp beside it first, then each os.replace'd into place, structure.json
    last and its old copy removed first, so a failed replace never leaves an old report beside a
    new structure or a new report beside an old one that still loads."""
    out.mkdir(parents=True, exist_ok=True)
    temps: list[tuple[str, Path]] = []
    try:
        for name, text in texts.items():
            fd, tmp = tempfile.mkstemp(dir=out, prefix=f".{name}.", suffix=".tmp")
            temps.append((name, Path(tmp)))
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            os.chmod(tmp, 0o644)
        (out / "structure.json").unlink(missing_ok=True)
        for name, tmp in sorted(temps, key=lambda t: t[0] == "structure.json"):
            os.replace(tmp, out / name)
    finally:
        for _, tmp in temps:
            tmp.unlink(missing_ok=True)


def run_infer(args: argparse.Namespace) -> int:
    if not args.samples and args.reference is None:
        raise CannotRun("no sample files (name some, or give --reference OLD NEW)")
    samples = list(args.samples) or [args.reference[1]]
    try:
        base_version, structure = si.resolve_base(args.base)
        new_files = [si.load_file(p) for p in [*samples, *(args.reference[1:] if args.reference else []), *(args.diff_pair or [])]]
    except si.InferInputError as e:
        raise CannotRun(str(e)) from e
    version = new_files[0].version
    out = args.out if args.out is not None else DEFAULT_OUT / f"v{version}"
    _check_out(out, args.force)
    game, notes = game_of(new_files, structure)
    try:
        result = si.infer(
            (base_version, structure),
            samples,
            reference=tuple(args.reference) if args.reference else None,
            diff_pair=tuple(args.diff_pair) if args.diff_pair else None,
            diff_expect_new=args.diff_expect_new,
            terrain_ids=terrain_ids(game),
            accept_ambiguous=args.accept_ambiguous,
        )
    except si.InferInputError as e:
        raise CannotRun(str(e)) from e
    result.notes[:0] = [f"Terrain ids from {game.display_name}'s own table.", *notes]
    report = si.render_report(result)
    print("\n".join(report))
    print()
    if result.status != si.STATUS_INFERRED:
        if result.status == si.STATUS_AMBIGUOUS:
            print("AMBIGUOUS: nothing written. Add a differential pair (--diff-pair), or --accept-ambiguous to record the open spans.")
        else:
            print(f"REFUSED: nothing written. {result.refusal}")
        return 1
    texts = {"structure.json": json.dumps(result.structure, indent=4) + "\n", "INFERRED.md": "\n".join(report) + "\n"}
    write_outputs(out, texts)
    print(f"INFERRED: {version} from {base_version}, wrote {out / 'structure.json'} and INFERRED.md")
    return 0


def _norm(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


def non_default_paths(walked: sw.Walk, compiled: sw.CompiledStructure, structure: dict) -> set[tuple[str, ...]]:
    """Every scalar field path that carries a value other than its structure default somewhere."""
    out = set()
    for block, (spec, prefixes) in si.block_prefixes(compiled).items():
        retrievers = si.block_dict(structure, block)["retrievers"]
        for f in spec.fields:
            if f.kind == "struct":
                continue
            default = _norm(retrievers[f.name].get("default"))
            for prefix in prefixes:
                col = walked.columns.get((*prefix, f.name))
                if col is not None and any(_norm(v) != default for v in col.value_list()):
                    out.add((*prefix, f.name))
    return out


def run_suggest(args: argparse.Namespace) -> int:
    try:
        base_version, structure = si.resolve_base(args.base)
    except si.InferInputError as e:
        raise CannotRun(str(e)) from e

    def missing(message: str) -> None:
        raise CannotRun(message)

    compiled = sw.compile_structure(structure)
    ranked: list[tuple[int, Path]] = []
    skipped: list[str] = []
    for path in collect_files([args.dir], missing):
        try:
            f = si.load_file(path)
        except si.InferInputError:
            continue
        if f.version != base_version:
            continue
        try:
            walked = sw.walk_file(compiled, f.raw)
        except (sw.WalkError, sw.StructureError) as e:
            skipped.append(f"{path.name}: {e}")
            continue
        paths = set()
        for w in (walked.header, walked.body):
            paths |= non_default_paths(w, compiled, structure)
        ranked.append((sum(SUGGEST_WEIGHT.get(p[0], 1) for p in paths), path))
    for line in skipped:
        print(f"skipped, does not walk under {base_version}: {line}")
    if not ranked:
        print(f"no file of version {base_version} in {args.dir}")
        return 1
    ranked.sort(key=lambda t: (-t[0], str(t[1])))
    print(f"{len(ranked)} file(s) of version {base_version}, best reference first.")
    print("Score: fields with a non-default value somewhere, Units and Triggers fields counting 3.")
    for score, path in ranked:
        print(f"{score:6d}  {path}")
    print(f"Re-save {ranked[0][1].name}, unedited, in the patched editor, then: infer --reference {ranked[0][1]} <the re-save>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=PROCEDURE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "infer", help="infer the samples' version's structure", epilog=PROCEDURE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("samples", type=Path, nargs="*", metavar="SAMPLE", help="game-saved files of the new version")
    p.add_argument("--base", help="the version (or a structure.json path) to start from; default the newest known")
    p.add_argument(
        "--reference", type=Path, nargs=2, metavar=("OLD", "NEW"), help="a base-version file and the patched game's re-save of it"
    )
    p.add_argument("--diff-pair", type=Path, nargs=2, metavar=("A", "B"), help="two new-version files differing by one in-game edit")
    p.add_argument("--diff-expect-new", action="store_true", help="the differential pair's edit must land in an inferred field")
    p.add_argument("--out", type=Path, help="output folder (default build/structure_infer/v<version>/)")
    p.add_argument("--force", action="store_true", help="replace the outputs in a non-empty --out")
    p.add_argument("--accept-ambiguous", action="store_true", help="write an ambiguous result, its open spans recorded in INFERRED.md")
    s = sub.add_parser("suggest-reference", help="rank the base version's files in DIR as a reference to re-save")
    s.add_argument("dir", type=Path, metavar="DIR")
    s.add_argument("--base", help="the version to rank files of; default the newest known")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "infer" and args.diff_expect_new and args.diff_pair is None:
        print("could not run: --diff-expect-new needs --diff-pair", file=sys.stderr)
        return 2
    try:
        return run_infer(args) if args.command == "infer" else run_suggest(args)
    except CannotRun as e:
        print(f"could not run: {e}", file=sys.stderr)
        return 2
    except (sw.WalkError, sw.StructureError) as e:
        # A reference OLD the base can't walk, or a base structure that doesn't compile.
        print(f"could not run: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except Exception:  # anything else is a bug or a malformed input, never AMBIGUOUS/REFUSED's exit 1
        print(f"could not run: unexpected error\n{traceback.format_exc()}", file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    sys.exit(main())
