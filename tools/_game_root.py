"""Where a game-data generator reads its files, per dataset (GH #67).

Every tools/gen_*.py that regen_game_data.py runs takes the AoE2:DE install
root and `--dataset {aoe2,ror}`. Return of Rome keeps its own
empires2_x2_p1.dat under `modes/Pompeii/`, which mirrors the install's layout;
any other file a generator reads (textures, palettes, particles, .sld
headers) is taken from the RoR root when it has one and from the install
otherwise, the same order descape/game_dataset.py resolves at runtime.

The CLASSIC datasets (aoc, aoe1) take their own 1990s install instead: their
.dat and DRS archives are found case-insensitively along ordered art folders
(classic_archives(), classic_graphics()). aoe1 stays in PENDING, accepted by
--dataset but not in DATASETS, until descape/games.py registers it.

HD Edition (hd) has a DE-shaped resources/ tree, so its .dat and
textures are found as AoE2's are, but its art is loose SLPs. SLP_ART is every
dataset whose graphics are SLPs; slp_art() and slp_palette() give a generator
that dataset's lookup and palette whichever kind of install holds them.

`--lenient` (GH #262) is for a data mod's .dat, which descape/dat_derive.py
passes and regen_game_data.py never does. Each generator calls set_lenient()
after parsing its arguments; a guard that pins an invariant measured on the
stock .dat then goes through refuse(), which raises SystemExit as before when
strict, and when lenient records a warning in WARNINGS and returns, so the
caller degrades only the object or row it guards.

A warning is a record (GH #323): `text` is `message (fallback)`, the
developer's wording a mod author's bug report needs; `kind` is one of
WARNING_KINDS, what the user sees differently; `subject` is what it is about,
an int id, a non-empty tuple of ids or None (unnamed, counted once; an empty
tuple records nothing). Each kind's subject is a unit const or a
terrain id by the kind (descape/data_mod_menu.py's WARNING_TEXT, which holds
the user's wording; tools import nothing from descape/).
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

AOE2 = "aoe2"
ROR = "ror"
# The 1999 game (Age of Kings / The Conquerors): its root is its own install, not a DE one.
AOC = "aoc"
# The original Age of Empires (1997) with Rise of Rome, read from that game's own install.
AOE1 = "aoe1"
# Age of Empires II: HD Edition, read from its own install's resources/ tree (TASK-431).
HD = "hd"
DATASETS = (AOE2, ROR, AOC, HD)
# Accepted by --dataset before descape/games.py registers them; registering moves one into DATASETS.
PENDING = (AOE1,)
# Read from a raw 1990s install: DRS archives and SLP art, no resources/ tree.
CLASSIC = (AOC, AOE1)
# Graphics are SLPs (a classic install's DRS archives, HD Edition's loose files), read with _drs.
SLP_ART = (*CLASSIC, HD)
# HD Edition's loose SLP folders and its JASC palette, under its install.
HD_DRS_SUBPATH = Path("resources") / "_common" / "drs"
HD_PALETTE_SUBPATH = HD_DRS_SUBPATH / "interface" / "50500.bina"
ROR_SUBPATH = Path("modes") / "Pompeii"
DAT_SUBPATH = Path("resources") / "_common" / "dat" / "empires2_x2_p1.dat"
# The Conquerors' .dat under its install, the 1.0c patch's first; descape/games.py's AOC.dat_subpaths.
AOC_DAT_SUBPATHS = (Path("Data") / "empires2_x1_p1.dat", Path("Data") / "empires2_x1.dat")
AOC_DATA_SUBPATH = Path("Data")
# Rise of Rome's .dat, a superset of the base game's data/Empires.dat.
AOE1_DAT_SUBPATHS = (Path("data2") / "empires.dat",)
# A classic install's art folders, the first holding an archive winning.
_ART_DIRS = {AOC: (AOC_DATA_SUBPATH,), AOE1: (Path("data2"), Path("data"))}
# The archive every classic graphics lookup needs; the 1999 game's others are _drs.AOC_GRAPHICS_ORDER.
GRAPHICS_DRS = "graphics.drs"
# The .dat version strings _dat57 and _dat37 read; every other one goes to genieutils-py.
DAT57_VERSION = "VER 5.7"
DAT37_VERSION = "VER 3.7"

_lenient = False
# What a lenient warning means to the user; the viewer's WARNING_TEXT words each one.
WARNING_KINDS = (
    "parts_dropped",
    "not_drawn",
    "no_garrison_flags",
    "age_art",
    "civ_art",
    "footprint",
    "placement_unchecked",
    "invisible_rule",
    "terrain_class",
    "terrain_blend",
)
# {"text", "kind", "subject"} records, insertion-ordered and de-duplicated on
# text: per-civ loops repeat a message.
WARNINGS: list[dict] = []


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=DATASETS + PENDING,
        default=AOE2,
        help=(
            "Which game's .dat to read: aoe2 (default), ror (Return of Rome, under modes/Pompeii/), "
            "aoc (the 1999 Age of Kings / The Conquerors), aoe1 (Age of Empires with Rise of Rome) or "
            "hd (Age of Empires II: HD Edition); for the last three the install argument is that game's own install"
        ),
    )
    parser.add_argument(
        "--dat",
        type=Path,
        default=None,
        help="Read this .dat instead of the dataset's own, e.g. a data mod's (GH #58); other files still come from the install",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="a data mod's .dat: a broken invariant degrades that object with a warning instead of failing (GH #262)",
    )


def set_lenient(on: bool) -> None:
    """Called by each generator after parse_args(), so a strict run never inherits an earlier lenient one."""
    global _lenient
    _lenient = bool(on)


def lenient() -> bool:
    return _lenient


def refuse(message: str, fallback: str, kind: str, subject: int | tuple[int, ...] | None = None) -> None:
    """Strict: raise SystemExit(message). Lenient: record `message (fallback)`
    with its kind and subject in WARNINGS and return, and the caller takes
    `fallback`. An empty tuple `subject` means nothing is affected: strict
    still raises, lenient records nothing, not even meta.json text (TASK-701). An unknown `kind`
    raises ValueError either way."""
    if kind not in WARNING_KINDS:
        raise ValueError(f"unknown warning kind {kind!r}")
    if not _lenient:
        raise SystemExit(message)
    if subject == ():
        return
    text = f"{message} ({fallback})"
    if all(warning["text"] != text for warning in WARNINGS):
        if isinstance(subject, tuple):
            subject = tuple(int(member) for member in subject)
        elif subject is not None:
            subject = int(subject)
        WARNINGS.append({"text": text, "kind": kind, "subject": subject})


def game_root(install: Path, dataset: str) -> Path:
    """The root holding `dataset`'s own files: its resources/ tree, or the 1999 install itself."""
    return install / ROR_SUBPATH if dataset == ROR else install


def dat_subpaths(dataset: str) -> tuple[Path, ...]:
    """`dataset`'s .dat candidates under its root, in the order tried."""
    return {AOC: AOC_DAT_SUBPATHS, AOE1: AOE1_DAT_SUBPATHS}.get(dataset, (DAT_SUBPATH,))


def find_path(root: Path, subpath: str | Path) -> Path | None:
    """`root / subpath` with each part matched case-insensitively, as a 1990s
    install copied from Windows spells them (`Empires.dat`, `Terrain.drs`);
    an exact match wins. None when no such file or folder exists."""
    found = Path(root)
    for part in Path(subpath).parts:
        if (found / part).exists():
            found = found / part
            continue
        if not found.is_dir():
            return None
        matches = sorted(child for child in found.iterdir() if child.name.lower() == part.lower())
        if not matches:
            return None
        found = matches[0]
    return found


def dat_path(install: Path, dataset: str, dat: Path | None = None) -> Path:
    """`dat` when given, else the dataset's own .dat under its root: the first
    candidate that exists, else the first. A classic install's is matched case-insensitively."""
    if dat is not None:
        return dat
    root = game_root(install, dataset)
    for subpath in dat_subpaths(dataset):
        found = find_path(root, subpath) if dataset in CLASSIC else root / subpath
        if found is not None and found.is_file():
            return found
    return root / dat_subpaths(dataset)[0]


def classic_archives(install: Path, dataset: str, *names: str):
    """A classic install's archives `names` (e.g. "terrain.drs") as one _drs.LayeredDrs:
    each art folder in order, each name in order within it, matched case-insensitively, so
    a member comes from the first archive holding it (Rise of Rome's data2/ over data/).
    Absent archives are skipped; FileNotFoundError when none is there."""
    import _drs

    archives = []
    for art_dir in _ART_DIRS[dataset]:
        for name in names:
            found = find_path(install, art_dir / name)
            if found is not None and found.is_file():
                archives.append(_drs.DrsArchive(found))
    if not archives:
        raise FileNotFoundError(f"{install}: no {' or '.join(names)} in {', '.join(map(str, _ART_DIRS[dataset]))}")
    return _drs.LayeredDrs(*archives)


def classic_graphics(install: Path, dataset: str):
    """The classic install's graphics lookup: classic_archives() over its graphics archives, the
    1999 game's in the engine's order. FileNotFoundError without a graphics.drs among them."""
    import _drs

    names = _drs.AOC_GRAPHICS_ORDER if dataset == AOC else (GRAPHICS_DRS,)
    layered = classic_archives(install, dataset, *names)
    if not any(archive.path.name.lower() == GRAPHICS_DRS for archive in layered.archives):
        raise FileNotFoundError(f"{install}: no {GRAPHICS_DRS} in {', '.join(map(str, _ART_DIRS[dataset]))}")
    return layered


def slp_art(install: Path, dataset: str):
    """An SLP_ART dataset's graphics lookup (has/read/ids by kind "slp"): classic_graphics()
    for a classic install, HD Edition's loose folders in _drs.HD_GRAPHICS_ORDER. FileNotFoundError without any."""
    import _drs

    if dataset in CLASSIC:
        return classic_graphics(install, dataset)
    if dataset != HD:
        raise ValueError(f"dataset {dataset!r} has no SLP art")
    stack = _drs.LooseSlpStack(*(install / HD_DRS_SUBPATH / name for name in _drs.HD_GRAPHICS_ORDER))
    if not stack.ids("slp"):
        raise FileNotFoundError(f"{install}: no .slp in {', '.join(map(str, stack.folders))}")
    return stack


# The classic palette: interfac.drs's JASC-PAL member, the one HD Edition keeps as a loose 50500.bina.
CLASSIC_PALETTE_ID = 50500


def slp_palette(install: Path, dataset: str) -> list[tuple[int, int, int]]:
    """An SLP_ART dataset's JASC palette 50500 as (r, g, b) tuples. FileNotFoundError when absent."""
    import _drs

    if dataset in CLASSIC:
        return _drs.read_jasc_palette(classic_archives(install, dataset, "interfac.drs").read("bina", CLASSIC_PALETTE_ID))
    if dataset != HD:
        raise ValueError(f"dataset {dataset!r} has no SLP palette")
    path = install / HD_PALETTE_SUBPATH
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return _drs.read_jasc_palette(path.read_bytes())


def dat_version(path: Path) -> str:
    """The version string a .dat opens with ("VER 5.7"), read by inflating its first bytes only."""
    with Path(path).open("rb") as handle:
        head = zlib.decompressobj(-zlib.MAX_WBITS).decompress(handle.read(4096), 8)
    return head.split(b"\0", 1)[0].decode("latin-1")


def parse_dat(path: Path):
    """The parsed .dat: _dat57's reader for a 1999 one, _dat37's for a 1997 one, else
    genieutils-py's. DatFile.parse is looked up per call, so descape/dat_derive.py's memo applies."""
    version = dat_version(path)
    if version == DAT57_VERSION:
        import _dat57

        return _dat57.parse(path)
    if version == DAT37_VERSION:
        import _dat37

        return _dat37.parse(path)
    from genieutils.datfile import DatFile

    return DatFile.parse(str(path))


def resource(install: Path, dataset: str, subpath: str | Path) -> Path:
    """`subpath` under the dataset's root when it exists there, else under the install."""
    candidate = game_root(install, dataset) / subpath
    return candidate if dataset == AOE2 or candidate.exists() else install / subpath


class LayeredFolder:
    """A folder lookup over several folders in priority order: `folder / name`
    is the first folder's file that exists, else the last folder's path."""

    def __init__(self, *folders: Path) -> None:
        self.folders = folders

    def __truediv__(self, name: str) -> Path:
        for folder in self.folders:
            if (folder / name).exists():
                return folder / name
        return self.folders[-1] / name

    def __str__(self) -> str:
        return " over ".join(str(folder) for folder in self.folders)

    def is_dir(self) -> bool:
        return any(folder.is_dir() for folder in self.folders)

    def glob(self, pattern: str) -> list[Path]:
        """Each name once, from the first folder holding it."""
        found: dict[str, Path] = {}
        for folder in self.folders:
            for path in folder.glob(pattern):
                found.setdefault(path.name, path)
        return [found[name] for name in sorted(found)]


def folder(install: Path, dataset: str, subpath: str | Path) -> Path | LayeredFolder:
    """`subpath` as a folder: the install's under AoE2, the RoR root's layered over the install's under RoR."""
    if dataset == ROR:
        return LayeredFolder(game_root(install, dataset) / subpath, install / subpath)
    return install / subpath


def by_dataset(dataset: str, aoe2, ror):
    """`aoe2` or `ror`: a generator's hand-measured const list for this dataset."""
    return ror if dataset == ROR else aoe2
