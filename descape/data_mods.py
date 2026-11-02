"""Discovering the player's data mods and activating one (GH #58).

A data mod is a mod folder that ships a whole resources/_common/dat/
empires2_x2_p1.dat, which replaces the install's (it is not an overlay). The
game selects one in the lobby rather than enabling it, so mod-status.json lists
every data mod as disabled and discovery ignores the Enabled flag. Nothing in a
scenario names its mod, so the viewer remembers one per scenario path.

A mod's identity is its mod-status Path form, `status_path`
("subscribed//93961_Exploding Villagers"), the same across profile ids; when
several profile ids carry one, the active profile's copy wins
(content_roots.active_profile_id()), else the first by id.

Activation goes through descape/dat_cache.py: tables_for() answers a cache hit
at once; a miss is derived by the caller in a subprocess
(derive_command()), never in the GUI's process. Qt-free and read-only toward
the profile, like content_roots.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from descape import content_roots, dat_cache, game_dataset

DAT_DIR = "dat"
DAT_NAME = "empires2_x2_p1.dat"
REPO_ROOT = Path(__file__).resolve().parent.parent
# The frozen exe's argument for the derive subprocess (packaging/entry_frozen.py).
FROZEN_DERIVE_FLAG = "--derive-dat"
# GH #265: the table sections unchanged_ids() compares per object, each with the
# top-level fields it ignores: civ ids name the mod's own civs, and a graphic
# id shifts when a mod adds graphics while the art itself stays the same.
UNCHANGED_COMPARED = (
    ("object_catalog.json", "objects", frozenset({"civs"})),
    ("unit_stats.json", "stats", frozenset()),
    ("unit_stats.json", "line_of_sight", frozenset()),
    ("unit_graphic_map.json", "graphics", frozenset({"graphic_id"})),
)


@dataclass(frozen=True)
class DataMod:
    status_path: str
    title: str
    workshop_id: str | None
    root: Path
    dat: Path


def discover(roots: list[content_roots.ContentRoot] | None = None) -> list[DataMod]:
    """Every data mod across the profile's mods, enabled or not, one per
    status_path, sorted by title. `roots` defaults to every content root."""
    if roots is None:
        roots = content_roots.content_roots(include_disabled=True)
    mod_kinds = (content_roots.KIND_MOD_SUBSCRIBED, content_roots.KIND_MOD_LOCAL)
    candidates = [root for root in roots if root.kind in mod_kinds and root.status_path]
    profile = content_roots.profile_path()
    preferred = None if profile is None else content_roots.active_profile_id(profile)
    candidates.sort(key=lambda root: root.profile_id != preferred)
    found: dict[str, DataMod] = {}
    for root, path in content_roots.find_resources(DAT_DIR, (".dat",), candidates):
        if path.name.casefold() != DAT_NAME or root.status_path in found:
            continue
        found[root.status_path] = DataMod(
            root.status_path, root.title or root.mod_id or root.status_path, root.workshop_id, root.path, path
        )
    return sorted(found.values(), key=lambda mod: (mod.title.casefold(), mod.status_path))


def identity(mod: DataMod) -> dict:
    """What remembers `mod` (settings' per-scenario choice, an autosave row):
    its status_path, plus its title for a message once it is gone and its
    workshop id, which find() falls back on."""
    return {"status_path": mod.status_path, "title": mod.title, "workshop_id": mod.workshop_id}


def find(remembered: dict, mods: list[DataMod]) -> DataMod | None:
    """The mod in `mods` that identity() dict names: by status_path, else by
    workshop id (a mod that moved between profile folders), else None."""
    for mod in mods:
        if mod.status_path == remembered.get("status_path"):
            return mod
    workshop = remembered.get("workshop_id")
    if workshop:
        return next((mod for mod in mods if mod.workshop_id == workshop), None)
    return None


def genieutils_available() -> bool:
    """Whether a cache miss can be derived at all (genieutils-py importable)."""
    return importlib.util.find_spec("genieutils") is not None


def entry_key(mod: DataMod, install: Path) -> str:
    """The cache entry name for `mod` against `install` (its .dat hashed once per size and mtime)."""
    return dat_cache.entry_key(dat_cache.entry_inputs(mod.dat, install))


def tables_for(mod: DataMod, key: str) -> game_dataset.DataModTables | None:
    """The overlay for `mod` when its cache entry `key` is complete, else None (a miss)."""
    entry = dat_cache.lookup(key)
    if entry is None:
        return None
    return game_dataset.DataModTables(key, entry.path, mod.root, mod.title, entry.mod_only_ids, entry.warnings)


def entry_dir(key: str) -> Path:
    return dat_cache.CACHE_ROOT / key


def derive_command(mod: DataMod, install: Path, key: str) -> tuple[list[str], Path]:
    """(argv, cwd) for the derive subprocess that fills entry_dir(key)."""
    args = [str(mod.dat), str(install), str(entry_dir(key))]
    if getattr(sys, "frozen", False):
        return [sys.executable, FROZEN_DERIVE_FLAG, *args], Path(sys.executable).parent
    return [sys.executable, "-m", "descape.dat_derive", *args], REPO_ROOT


def mod_only_ids() -> frozenset[int]:
    """The active data mod's unit consts the install's .dat lacks; empty with no mod."""
    mod = game_dataset.data_mod()
    return frozenset() if mod is None else mod.mod_only_ids


def _section(name: str, key: str, dataset: str | None = None) -> dict:
    path = game_dataset.table_path(name, dataset)
    return json.loads(path.read_text()).get(key, {}) if path.is_file() else {}


def _without(entry, ignored: frozenset[str]):
    return {k: v for k, v in entry.items() if k not in ignored} if isinstance(entry, dict) else entry


@game_dataset.cache(maxsize=1)
def unchanged_ids() -> frozenset[int]:
    """The active data mod's unit consts that are AoE2's own object, untouched
    (GH #265): object_catalog.same_object_in_aoe2(), and each UNCHANGED_COMPARED
    entry equal to the committed AoE2 table's but for its ignored fields, and
    a name string the mod's strings files do not define. Empty with no mod."""
    mod = game_dataset.data_mod()
    if mod is None:
        return frozenset()
    from descape import asset_source, object_catalog

    objects = _section("object_catalog.json", "objects")
    keys = {key for key in objects if int(key) not in mod.mod_only_ids and object_catalog.same_object_in_aoe2(int(key))}
    for name, section, ignored in UNCHANGED_COMPARED:
        ours, aoe2 = _section(name, section), _section(name, section, game_dataset.AOE2)
        keys = {key for key in keys if _without(ours.get(key), ignored) == _without(aoe2.get(key), ignored)}
    renamed = asset_source.data_mod_string_ids()
    return frozenset(int(key) for key in keys if objects[key].get("string_id") not in renamed)
