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

A data mod (GH #58) is an overlay on the AoE2 dataset: set_data_mod() names a
folder of tables derived from the mod's own .dat (descape/dat_cache.py), which
table_path() then returns in place of the committed ones. It applies only while
AoE2 is active, since a data mod replaces the AoE2 .dat; under Return of Rome it
is inert. cache_key() is the dataset plus the whole mod overlay, so cache() keeps
one cache per pair, and generation() moves on every change to either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DataModTables:
    """An active data mod's derived tables (GH #58). `key` is its cache entry's
    name, unique per mod .dat and derivation inputs; `root` the mod folder, for
    its strings; `title` what the window title shows; `mod_only_ids` the unit
    consts its .dat defines and the install's does not."""

    key: str
    tables_dir: Path
    root: Path
    title: str
    mod_only_ids: frozenset[int] = frozenset()


_active = AOE2
_data_mod: DataModTables | None = None
# cache_key()'s value, kept current by _set_state(): cache() reads it on every call.
_cache_key: object = AOE2
_generation = 0
_switch_hooks: list[Callable[[], None]] = []


def active() -> str:
    return _active


def is_ror() -> bool:
    return _active == ROR


def data_mod() -> DataModTables | None:
    """The data mod whose tables are in effect: the one set_data_mod() named, while AoE2 is active."""
    return _data_mod if _active == AOE2 else None


def cache_key() -> object:
    """What cache() keys on: the dataset, or the dataset and the whole data mod in effect."""
    return _cache_key


def _set_state(dataset: str, mod: DataModTables | None) -> bool:
    """Sets both; runs the switch and returns True when cache_key() changed."""
    global _active, _data_mod, _cache_key
    _active, _data_mod = dataset, mod
    effective = data_mod()
    key = _active if effective is None else (_active, effective)
    if key == _cache_key:
        return False
    _cache_key = key
    _switched()
    return True


def generation() -> int:
    """Moves on every change of cache_key(), for caches keyed on the tables in effect (render's sprite memo)."""
    return _generation


def _switched() -> None:
    global _generation
    _generation += 1
    from descape import asset_source

    asset_source.clear_install_caches()
    for hook in tuple(_switch_hooks):
        hook()


def set_active(dataset: str) -> bool:
    """Makes `dataset` the active one; True when it changed. A change drops
    every install-derived cache (names, art, icons) and runs the on_switch()
    hooks, which refill the tables other modules imported at load time."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown game dataset {dataset!r}")
    if dataset == _active:
        return False
    return _set_state(dataset, _data_mod)


def set_data_mod(mod: DataModTables | None) -> bool:
    """Makes `mod` the AoE2 data mod (None for the install's own tables); True
    when the tables in effect changed. Kept while Return of Rome is active, inert there."""
    return _set_state(_active, mod)


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
    descape/ror/, under a data mod in its tables folder (dataset None only),
    except SHARED_TABLES, which stay where `default` points."""
    path = Path(default)
    if len(path.parts) == 1:
        path = TABLES_DIR / path
    if path.name in SHARED_TABLES:
        return path
    if (dataset or _active) == ROR:
        return ROR_TABLES_DIR / path.name
    mod = data_mod() if dataset is None else None
    return path if mod is None else mod.tables_dir / path.name


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
    """functools.lru_cache, one cache per cache_key(): the switch itself neither
    clears nor re-parses it, though set_active()'s clear_install_caches()
    drops the install-derived ones. cache_clear() drops every dataset's."""

    def decorate(fn):
        per_dataset: dict[object, Callable] = {}

        def _cached() -> Callable:
            key = _cache_key
            cached = per_dataset.get(key)
            if cached is None:
                cached = per_dataset[key] = lru_cache(maxsize=maxsize)(fn)
            return cached

        def wrapper(*args, **kwargs):
            return _cached()(*args, **kwargs)

        def cache_clear() -> None:
            for cached in per_dataset.values():
                cached.cache_clear()

        def cache_info():
            """The active dataset's cache's lru_cache info."""
            return _cached().cache_info()

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        return update_wrapper(wrapper, fn)

    return decorate
