"""Where a game-data generator reads its files, per dataset (GH #67).

Every tools/gen_*.py that regen_game_data.py runs takes the AoE2:DE install
root and `--dataset {aoe2,ror}`. Return of Rome keeps its own
empires2_x2_p1.dat under `modes/Pompeii/`, which mirrors the install's layout;
any other file a generator reads (textures, palettes, particles, .sld
headers) is taken from the RoR root when it has one and from the install
otherwise, the same order descape/game_dataset.py resolves at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

AOE2 = "aoe2"
ROR = "ror"
DATASETS = (AOE2, ROR)
ROR_SUBPATH = Path("modes") / "Pompeii"
DAT_SUBPATH = Path("resources") / "_common" / "dat" / "empires2_x2_p1.dat"


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=AOE2,
        help="Which game's .dat to read: aoe2 (default) or ror (Return of Rome, under modes/Pompeii/)",
    )
    parser.add_argument(
        "--dat",
        type=Path,
        default=None,
        help="Read this .dat instead of the dataset's own, e.g. a data mod's (GH #58); other files still come from the install",
    )


def game_root(install: Path, dataset: str) -> Path:
    """The root holding `dataset`'s own resources/ tree."""
    return install / ROR_SUBPATH if dataset == ROR else install


def dat_path(install: Path, dataset: str, dat: Path | None = None) -> Path:
    """`dat` when given, else the dataset's own .dat under the install."""
    return dat if dat is not None else game_root(install, dataset) / DAT_SUBPATH


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
