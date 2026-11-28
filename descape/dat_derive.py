"""Derives the game-data tables from one empires2_x2_p1.dat (GH #58).

The tables DEscape ships (descape/*.json, descape/ror/*.json) and a data mod's
come from one code path: the tools/gen_*.py generators in GENERATORS, which
tools/regen_game_data.py also runs. derive() runs them all in this process with
DatFile.parse memoised, so the .dat parses once (~10 s, not ten times), and
points them at `dat` through their --dat argument. Their other inputs (.sld
headers, the palette, texture names) still come from the install, since a data
mod ships none of them.

derive() runs them lenient by default (GH #262, `--lenient`, tools/_game_root.py):
a guard pinning an invariant measured on the stock .dat degrades only what it
guards (a plain entry, dropped pieces, an omitted row) and returns a warning
instead of failing the whole mod: a DeriveWarning record (GH #323), its text
the generator's own (for a bug report), its kind and subject what
descape/data_mod_menu.py words for the user. regen_game_data.py runs them
strict, so the shipped tables and their failures are unchanged;
derive(lenient=False) does too.
A missing install file or an unreadable .dat still fails either way.

genieutils-py is imported only inside derive(). The viewer never calls derive()
itself: a parse holds the GIL for seconds, so it runs in a subprocess,
`python -m descape.dat_derive <dat> <install> <out>` from source and the frozen
exe's `--derive-dat` (packaging/entry_frozen.py), both through main().

main() reports progress on stdout (GH #359), one progress_line() per step,
which the viewer reads back with parse_progress_line(); the generators' own
prints share the stream and parse to None.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import runpy
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

# The one registry: every tools/gen_*.py that parses the .dat, and the tables it writes.
GENERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gen_garrison_table.py", ("garrison_table.json",)),
    ("gen_object_catalog.py", ("object_catalog.json",)),
    ("gen_terrain_blend_map.py", ("terrain_blend_map.json",)),
    ("gen_terrain_classes.py", ("terrain_classes.json",)),
    ("gen_terrain_texture_map.py", ("terrain_texture_map.json",)),
    ("gen_terrain_unit_map.py", ("terrain_unit_map.json",)),
    ("gen_tree_unit_ids.py", ("tree_unit_ids.json",)),
    ("gen_unit_graphic_map.py", ("unit_graphic_map.json", "building_art_map.json")),
    ("gen_unit_render_data.py", ("unit_render_data.json",)),
    ("gen_unit_stats.py", ("unit_stats.json",)),
)
TABLE_NAMES: tuple[str, ...] = tuple(name for _script, names in GENERATORS for name in names)
# Imported by every generator; part of what derive() runs, so part of source_digest().
GAME_ROOT_HELPER = "_game_root.py"
GAME_ROOT_MODULE = "_game_root"
# The 1999 and 1997 games' .dat readers and the archive reader, which _game_root and the generators import too.
READER_HELPERS = ("_dat57.py", "_dat37.py", "_drs.py")


def tools_dir() -> Path:
    """Where the generators live: the repo's tools/, or the frozen bundle's copy."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        return Path(bundle) / "tools"
    return Path(__file__).resolve().parent.parent / "tools"


def source_files(tools: Path | None = None) -> list[Path]:
    """Every source file a derivation runs: the generators, their helpers, this
    module. The frozen build bundles all of them as data (packaging/descape.spec);
    its --self-check calls source_digest() to prove it."""
    tools = tools_dir() if tools is None else tools
    helpers = [tools / name for name in (GAME_ROOT_HELPER, *READER_HELPERS)]
    return [tools / script for script, _names in GENERATORS] + helpers + [Path(__file__).resolve()]


def source_digest(tools: Path | None = None) -> str:
    """sha256 over source_files()' bytes (CRLF folded), so editing any of them
    changes a cache key. Read once per process: the files do not change under it."""
    return _source_digest(str(tools_dir() if tools is None else tools))


@functools.cache
def _source_digest(tools: str) -> str:
    digest = hashlib.sha256()
    for path in source_files(Path(tools)):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DeriveWarning:
    """One lenient guard's warning: `text` is `<script>: message (fallback)`;
    `kind` one of tools/_game_root.WARNING_KINDS (or "other", the cache's
    fallback for an item it cannot read); `subject` the unit const or terrain
    id it is about, a tuple of them, or None."""

    text: str
    kind: str
    subject: int | tuple[int, ...] | None = None


@functools.cache
def warning_kinds() -> frozenset[str]:
    """tools/_game_root.WARNING_KINDS, loaded from tools_dir() under a private
    name so the generators' own cached copy is untouched."""
    spec = importlib.util.spec_from_file_location("_descape_warning_kinds", tools_dir() / GAME_ROOT_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return frozenset(module.WARNING_KINDS)


class DeriveError(RuntimeError):
    """A generator failed or wrote nothing; the message names it."""


def unit_ids(dat_file) -> set[int]:
    """Every unit const a parsed .dat defines (civ 0's list, as the generators read it)."""
    return {const for const, unit in enumerate(dat_file.civs[0].units) if unit is not None}


def derive(
    dat: Path,
    install: Path,
    out_dir: Path,
    *,
    tools: Path | None = None,
    parsed: dict | None = None,
    lenient: bool = True,
    progress=None,
) -> list[DeriveWarning]:
    """Runs every generator against `dat` (AoE2 dataset, install files from
    `install`), writing TABLE_NAMES into `out_dir`, and returns the lenient
    guards' DeriveWarnings, each text prefixed with its generator's script name. Raises
    DeriveError, or whatever genieutils raises on a .dat it cannot read.
    `parsed` is the memo of DatFile.parse results by path, shared with the
    caller. `lenient=False` fails on the first guard, as regen_game_data.py does.
    `progress(i, script)` is called as each generator starts, i counting from 0."""
    from genieutils.datfile import DatFile

    tools = tools_dir() if tools is None else tools
    memo = {} if parsed is None else parsed
    real_parse = DatFile.__dict__["parse"]

    def parse_once(cls, input_file):
        key = str(Path(input_file).resolve())
        if key not in memo:
            memo[key] = real_parse.__func__(cls, input_file)
        return memo[key]

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_argv, saved_path = sys.argv, list(sys.path)
    saved_helper = sys.modules.get(GAME_ROOT_MODULE)
    DatFile.parse = classmethod(parse_once)
    sys.path.insert(0, str(tools))
    helper = None
    warnings: list[DeriveWarning] = []
    try:
        helper = _tools_helper(tools)
        for i, (script, names) in enumerate(GENERATORS):
            if progress is not None:
                progress(i, script)
            sys.argv = [script, str(install), "--out-dir", str(out_dir), "--dat", str(dat)]
            if lenient:
                sys.argv.append("--lenient")
            # Per run, so a later generator's repeat of an earlier text is kept under its own label.
            helper.WARNINGS.clear()
            try:
                runpy.run_path(str(tools / script), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise DeriveError(f"{script} exited with {exc.code}") from exc
            warnings += [
                DeriveWarning(f"{script}: {warning['text']}", warning["kind"], warning["subject"])
                for warning in helper.WARNINGS
            ]
            missing = [name for name in names if not (out_dir / name).is_file()]
            if missing:
                raise DeriveError(f"{script} wrote no {', '.join(missing)}")
    finally:
        if helper is not None:
            helper.set_lenient(False)
            helper.WARNINGS.clear()
        if saved_helper is not None:
            sys.modules[GAME_ROOT_MODULE] = saved_helper
        DatFile.parse = real_parse
        sys.argv, sys.path[:] = saved_argv, saved_path
    return warnings


def _tools_helper(tools: Path):
    """`tools`' own _game_root, the module the runpy'd generators import: one
    cached from another folder (a test's tools copy) is set aside until derive() ends."""
    cached = sys.modules.get(GAME_ROOT_MODULE)
    if cached is not None and Path(cached.__file__).resolve() != (tools / GAME_ROOT_HELPER).resolve():
        del sys.modules[GAME_ROOT_MODULE]
    import _game_root

    return _game_root


def parse_unit_ids(dat: Path, parsed: dict | None = None) -> set[int]:
    """unit_ids() of `dat`, through the same memo as derive()."""
    from genieutils.datfile import DatFile

    key = str(Path(dat).resolve())
    if parsed is not None and key in parsed:
        return unit_ids(parsed[key])
    data = DatFile.parse(str(dat))
    if parsed is not None:
        parsed[key] = data
    return unit_ids(data)


def dat_version(dat: Path) -> str | None:
    """The format tag a .dat opens with ("VER 8.9"), or None when it cannot be read as one."""
    try:
        with Path(dat).open("rb") as handle:
            head = zlib.decompressobj(wbits=-15).decompress(handle.read(1 << 16), 8)
    except (OSError, zlib.error):
        return None
    text = head.rstrip(b"\0").decode("ascii", "replace")
    return text if text.startswith("VER ") else None


def failure_message(exc: BaseException, dat: Path) -> str:
    """One line for a failed derive. genieutils fails an older .dat (VER 7.2
    mods) with a bare AssertionError, so an empty message names the format instead."""
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    version = dat_version(dat) or "not a readable .dat"
    return f"genieutils-py cannot read this .dat ({version}, {type(exc).__name__})"


def main(argv: list[str] | None = None) -> int:
    """The derive subprocess: `<mod dat> <install> <out dir>`. Exit 0 once
    descape.dat_cache.build_entry() has filled <out dir>, 1 on any failure
    (failure_message() on stderr's last line), 2 on bad arguments."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: dat_derive <dat> <install> <out dir>", file=sys.stderr)
        return 2
    from descape import dat_cache

    try:
        dat_cache.build_entry(Path(args[0]), Path(args[1]), Path(args[2]), progress=dat_cache.print_progress)
    except Exception as exc:  # noqa: BLE001 -- a subprocess boundary: report and exit non-zero
        print(failure_message(exc, Path(args[0])), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
