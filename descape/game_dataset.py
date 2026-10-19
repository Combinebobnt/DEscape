"""Which game's data DEscape reads: Age of Empires II or Return of Rome (GH #67).

Return of Rome (RoR) has its own empires2_x2_p1.dat, and the same unit const
means a different unit there (93 is Spearman in AoE2 and Hoplite in RoR). So
every table generated from the .dat (tools/regen_game_data.py) has a RoR copy
under descape/ror/, and RoR art lives under the install's modes/Pompeii/,
which mirrors the install's own layout and falls back to it file by file.

One process-wide active dataset, AoE2 by default. The viewer sets it from the
open file's variant (from_variant()) or the Settings > Advanced > Game data choice.
table_path() picks a generated table's copy, resource_path() an install file's,
and cache() keys a loader's cache by dataset. A switch still clears the
install-derived caches (asset_source.clear_install_caches()), so those re-read
after a switch either way. Qt-free, and importing it reads no file.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, update_wrapper
from pathlib import Path

AOE2 = "aoe2"
ROR = "ror"
DATASETS = (AOE2, ROR)
DISPLAY_NAMES = {AOE2: "Age of Empires II", ROR: "Return of Rome"}

# ScenarioVariant.ROR, the file header's unknown_value_2 for a RoR scenario.
ROR_VARIANT = 2
# The RoR root, relative to the install: its own resources/ and widgetui/ trees.
ROR_SUBPATH = "modes/Pompeii"

TABLES_DIR = Path(__file__).resolve().parent
ROR_TABLES_DIR = TABLES_DIR / "ror"
# Tables not derived from either .dat: both datasets read the one copy.
SHARED_TABLES = frozenset({"cliff_connectivity.json", "trigger_requirements.json"})

_active = AOE2
_switch_hooks: list[Callable[[], None]] = []


def active() -> str:
    return _active


def is_ror() -> bool:
    return _active == ROR


def set_active(dataset: str) -> bool:
    """Makes `dataset` the active one; True when it changed. A change drops
    every install-derived cache (names, art, icons) and runs the on_switch()
    hooks, which refill the tables other modules imported at load time."""
    global _active
    if dataset not in DATASETS:
        raise ValueError(f"unknown game dataset {dataset!r}")
    if dataset == _active:
        return False
    _active = dataset
    from descape import asset_source

    asset_source.clear_install_caches()
    for hook in tuple(_switch_hooks):
        hook()
    return True


def on_switch(hook: Callable[[], None]) -> Callable[[], None]:
    """Registers `hook` to run after every dataset change; returns it, so it works as a decorator."""
    _switch_hooks.append(hook)
    return hook


def from_variant(variant) -> str:
    """The dataset for a file header's variant (ScenarioVariant or its int):
    ROR for Return of Rome, AOE2 for everything else, None included."""
    try:
        return ROR if variant is not None and int(variant) == ROR_VARIANT else AOE2
    except (TypeError, ValueError):
        return AOE2


def table_path(default: Path | str, dataset: str | None = None) -> Path:
    """The active dataset's copy of a generated table, or `dataset`'s. `default`
    is the AoE2 path (or just the file name); under RoR the same name in
    descape/ror/, except SHARED_TABLES, which stay where `default` points."""
    path = Path(default)
    if len(path.parts) == 1:
        path = TABLES_DIR / path
    if (dataset or _active) == ROR and path.name not in SHARED_TABLES:
        return ROR_TABLES_DIR / path.name
    return path


def ror_root(install: Path | None) -> Path | None:
    return None if install is None else install / ROR_SUBPATH


def resource_path_in(install: Path | None, subpath: str | Path) -> Path | None:
    """An install file or folder for the active dataset: under RoR the RoR
    root's copy when it exists, else the install's own (which may not exist
    either; the caller checks). None without an install."""
    if install is None:
        return None
    if _active == ROR:
        candidate = install / ROR_SUBPATH / subpath
        if candidate.exists():
            return candidate
    return install / subpath


def resource_path(subpath: str | Path) -> Path | None:
    """resource_path_in() under the configured install."""
    from descape import asset_source

    return resource_path_in(asset_source.get_install_path(), subpath)


def resource_dirs(subpath: str | Path) -> list[Path]:
    """Every folder a lookup by loose name should search, in priority order:
    the RoR root's copy first under RoR, then the install's. Only existing folders."""
    from descape import asset_source

    install = asset_source.get_install_path()
    if install is None:
        return []
    roots = [install / ROR_SUBPATH, install] if _active == ROR else [install]
    return [root / subpath for root in roots if (root / subpath).is_dir()]


def cache(maxsize: int | None = 128):
    """functools.lru_cache, one cache per dataset: the switch itself neither
    clears nor re-parses it, though set_active()'s clear_install_caches()
    drops the install-derived ones. cache_clear() drops every dataset's."""

    def decorate(fn):
        per_dataset: dict[str, Callable] = {}

        def wrapper(*args, **kwargs):
            cached = per_dataset.get(_active)
            if cached is None:
                cached = per_dataset[_active] = lru_cache(maxsize=maxsize)(fn)
            return cached(*args, **kwargs)

        def cache_clear() -> None:
            for cached in per_dataset.values():
                cached.cache_clear()

        def cache_info():
            """The active dataset's cache's lru_cache info."""
            cached = per_dataset.get(_active)
            if cached is None:
                cached = per_dataset[_active] = lru_cache(maxsize=maxsize)(fn)
            return cached.cache_info()

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        return update_wrapper(wrapper, fn)

    return decorate
