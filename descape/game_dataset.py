"""Which game's data DEscape reads: Age of Empires II or Return of Rome (GH #67).

Each game is a descriptor in descape/games.py; this module holds the active
content stack and resolves tables, install files and strings through it
(docs/GAME_DATA.md, "The content stack").

Return of Rome (RoR) has its own empires2_x2_p1.dat, and the same unit const
means a different unit there (93 is Spearman in AoE2 and Hoplite in RoR). So
every table generated from the .dat (tools/regen_game_data.py) has a RoR copy
under descape/ror/, and RoR art lives under the install's modes/Pompeii/,
which mirrors the install's own layout and falls back to it file by file.

One process-wide ContentStack, AoE2 with no data mod by default, replaced
whole on every change. The viewer sets the game from the open file's own
(games.for_scenario()) or the Settings > Advanced > Game data choice, and
lock_reason() says when that shows a document view-only. table_path()
picks a generated table's copy, resource_path() an install file's, and
cache() keys a loader's cache by the stack's tables_key. A change runs
asset_source.clear_install_caches() with the layers it changed (TABLES, ROOTS),
then the on_switch() hooks. Qt-free, and importing it reads no file.

A data mod (GH #58) replaces a game's .dat: set_data_mod() names a folder of
tables derived from the mod's own .dat (descape/dat_cache.py), which
table_path() then returns in place of the committed ones, and the mod's
strings files go on top of the game's. It is in effect only while a game with
Game.data_mods is active (AoE2); under Return of Rome it stays set but inert.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, update_wrapper
from pathlib import Path

from descape import games

# The two games' ids, as plain strings for set_active() callers.
AOE2 = "aoe2"
ROR = "ror"

TABLES_DIR = Path(__file__).resolve().parent
# Tables not derived from any .dat: every dataset reads the one copy.
SHARED_TABLES = frozenset({"cliff_connectivity.json", "trigger_requirements.json"})

# A game's strings files sit in this folder under its root, Game.strings_files by name.
STRINGS_DIR_TEMPLATE = "resources/{lang}/strings/key-value"
# GH #58: a data mod's own strings, in this order over the game's (one mod ships both).
MOD_STRINGS_FILES = ("key-value-strings-utf8.txt", "key-value-modded-strings-utf8.txt")

# The layers a stack change names to the caches (docs/GAME_DATA.md, "Cache layers").
TABLES = "tables"  # which generated tables are in effect
ROOTS = "roots"  # which install roots files resolve under
DATA_MOD_SWITCH = frozenset({TABLES})
GAME_SWITCH = frozenset({TABLES, ROOTS})


def __getattr__(name: str):
    # The old constants, read from the registry on each access so a game registered later shows.
    if name == "DATASETS":
        return games.ids()
    if name == "DISPLAY_NAMES":
        return {game.id: game.display_name for game in games.GAMES}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class DataModTables:
    """An active data mod's derived tables (GH #58). `key` is its cache entry's
    name, unique per mod .dat and derivation inputs; `root` the mod folder, for
    its strings; `title` what the window title shows; `mod_only_ids` the unit
    consts its .dat defines and the install's does not; `warnings` what the
    lenient derive simplified (GH #262); `side_key` a hash of its side files
    (GH #263), compared so that editing one is a new overlay for cache()."""

    key: str
    tables_dir: Path
    root: Path
    title: str
    mod_only_ids: frozenset[int] = frozenset()
    # Outside the hash game_dataset.cache takes on every call; `key` already identifies the derive.
    warnings: tuple[str, ...] = field(default=(), compare=False)
    side_key: str = ""


@dataclass(frozen=True)
class ContentStack:
    """What content resolves through: a game and the data mod set over it
    (docs/GAME_DATA.md, "The content stack"). Art mods stay mod_graphics state.
    Immutable: a change builds a new one and swaps it in with one assignment."""

    game: games.Game
    # What set_data_mod() named, kept across a game switch; in effect only where game.data_mods.
    wanted_mod: DataModTables | None = None
    # Derived in __post_init__, so cache() reads tables_key without hashing anything new per call.
    data_mod: DataModTables | None = field(init=False, repr=False, compare=False)
    tables_key: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mod = self.wanted_mod if self.game.data_mods else None
        object.__setattr__(self, "data_mod", mod)
        object.__setattr__(self, "tables_key", self.game.id if mod is None else (self.game.id, mod))

    def roots(self, install: Path) -> tuple[Path, ...]:
        """The install roots files resolve along: the game's own, then each fallback's."""
        return tuple(of.root(install) for of in games.chain(self.game))

    def strings_paths(self, install: Path, lang: str) -> list[Path]:
        """Every strings file for lang, existing or not, in merge order: the
        bottom fallback's first, the game's own, then the data mod's. A later
        file wins on a shared key."""
        folder = STRINGS_DIR_TEMPLATE.format(lang=lang)
        own = [of.root(install) / folder / name for of in reversed(games.chain(self.game)) for name in of.strings_files]
        return own + self.mod_strings_paths(lang)

    def mod_strings_paths(self, lang: str) -> list[Path]:
        """The data mod's strings files for lang, existing or not; [] with none in effect."""
        if self.data_mod is None:
            return []
        from descape import content_roots

        folder = content_roots.child_dir(self.data_mod.root, *STRINGS_DIR_TEMPLATE.format(lang=lang).split("/"))
        return [] if folder is None else [folder / name for name in MOD_STRINGS_FILES]


_stack = ContentStack(games.get(AOE2))
_generation = 0
_switch_hooks: list[Callable[[], None]] = []


def stack() -> ContentStack:
    """The active content stack. Read it once per lookup: a switch replaces it whole."""
    return _stack


def active() -> str:
    return _stack.game.id


def game() -> games.Game:
    """The active game's descriptor. A file's own game is games.for_scenario()."""
    return _stack.game


def data_mod() -> DataModTables | None:
    """The data mod whose tables are in effect: the one set_data_mod() named, while a game that takes one is active."""
    return _stack.data_mod


def cache_key() -> object:
    """What cache() keys on: the game id, or the id and the whole data mod in effect."""
    return _stack.tables_key


def _swap(new: ContentStack) -> bool:
    """Installs `new` in one assignment; runs the switch and returns True when cache_key() changed."""
    global _stack
    old, _stack = _stack, new
    if new.tables_key == old.tables_key:
        return False
    _switched(DATA_MOD_SWITCH if new.game.id == old.game.id else GAME_SWITCH)
    return True


def generation() -> int:
    """Moves on every change of cache_key(), for caches keyed on the tables in effect (render's sprite memo)."""
    return _generation


def _switched(changed: frozenset[str]) -> None:
    global _generation
    _generation += 1
    from descape import asset_source

    asset_source.clear_install_caches(changed=changed)
    for hook in tuple(_switch_hooks):
        hook()


def only_changed(changed: frozenset[str] | None, layers: frozenset[str]) -> bool:
    """Whether a clear that names `changed` stays inside `layers`, so a cache
    that reads none of them keeps. None, an install or profile change, is
    everything; a layer the cache does not list leaves it cleared."""
    return changed is not None and changed <= layers


def set_active(dataset: str) -> bool:
    """Makes `dataset` the active one; True when it changed. A change drops
    every install-derived cache (names, art, icons) and runs the on_switch()
    hooks, which refill the tables other modules imported at load time."""
    of = games.get(dataset)  # raises ValueError for an unknown one
    if of.id == _stack.game.id:
        return False
    return _swap(ContentStack(of, _stack.wanted_mod))


@contextmanager
def activated(dataset: str) -> Iterator[None]:
    """`dataset` active for the block, the previous one again after it,
    even on a raise. No switch at all when it is already active; the data
    mod stays set throughout, and one set inside the block survives it. A
    switch drops what set_active() drops, so keep the block free of anything
    that paints with the other game's tables."""
    previous = _stack.game.id
    if dataset == previous:
        yield
        return
    set_active(dataset)
    try:
        yield
    finally:
        set_active(previous)


def set_data_mod(mod: DataModTables | None) -> bool:
    """Makes `mod` the data mod (None for the game's own tables); True when
    the tables in effect changed. Kept while a game without data mods (Return
    of Rome) is active, inert there. A change names only TABLES, so decoded
    art stays (TASK-478)."""
    return _swap(ContentStack(_stack.game, mod))


def on_switch(hook: Callable[[], None]) -> Callable[[], None]:
    """Registers `hook` to run after every dataset change; returns it, so it works as a decorator."""
    _switch_hooks.append(hook)
    return hook


def from_variant(variant) -> str:
    """The dataset for a file header's variant (ScenarioVariant or its int):
    ROR for Return of Rome, AOE2 for everything else, None included."""
    return games.for_variant(variant).id


def lock_reason(file_game: str, shown_game: str) -> str | None:
    """Why a document of `file_game` shown with `shown_game`'s data may not be
    written, or None (docs/GAME_DATA.md, "View-only games and the lock"). A
    game that is not writable is view-only whatever data shows it; otherwise
    a file is editable only with its own game's data shown (GH #67)."""
    own, shown = games.get(file_game), games.get(shown_game)
    if not own.writable:
        return f"Read-only: DEscape opens {own.display_name} files view-only."
    if shown.id != own.id:
        return (
            f"Read-only: showing {shown.display_name} game data for a "
            f"{own.display_name} file. Set Settings > Advanced > Game data to Auto to edit it."
        )
    return None


def table_path(default: Path | str, dataset: str | None = None) -> Path:
    """The active dataset's copy of a generated table, or `dataset`'s. `default`
    is the AoE2 path (or just the file name); under another game the same name
    in its own tables folder (descape/ror/), under a data mod in its tables
    folder (dataset None only), except SHARED_TABLES, which stay where
    `default` points."""
    path = Path(default)
    if len(path.parts) == 1:
        path = TABLES_DIR / path
    if path.name in SHARED_TABLES:
        return path
    if dataset is None:
        current = _stack  # one read, so the game and the mod come from the same stack
        of, mod = current.game, current.data_mod
    else:
        of, mod = games.get(dataset), None
    # The top provider gives the whole set: the data mod's folder, else the game's own.
    if mod is not None:
        return mod.tables_dir / path.name
    return path if of.id == games.DEFAULT_ID else of.tables_dir / path.name


def resource_path_in(install: Path | None, subpath: str | Path) -> Path | None:
    """An install file or folder for the active dataset: the first copy that
    exists along its root chain (under RoR the RoR root's, then the
    install's), else the last root's own path (which may not exist either;
    the caller checks). None without an install."""
    if install is None:
        return None
    *own, last = _stack.roots(install)
    for root in own:
        candidate = root / subpath
        if candidate.exists():
            return candidate
    return last / subpath


def resource_path(subpath: str | Path) -> Path | None:
    """resource_path_in() under the configured install."""
    from descape import asset_source

    return resource_path_in(asset_source.get_install_path(), subpath)


def resource_dirs(subpath: str | Path) -> list[Path]:
    """Every folder a lookup by loose name should search, in priority order
    along the active dataset's root chain: the RoR root's copy first under
    RoR, then the install's. Only existing folders."""
    from descape import asset_source

    install = asset_source.get_install_path()
    if install is None:
        return []
    return [root / subpath for root in _stack.roots(install) if (root / subpath).is_dir()]


def has_own_strings(lang: str, game: games.Game | None = None) -> bool:
    """Whether the configured install ships `game`'s own strings files for
    lang (default the active game's), under its own root. A game without them
    reads only its fallback's strings (TASK-507). Cached until
    asset_source.clear_install_caches()."""
    from descape import asset_source

    install = asset_source.get_install_path()
    return install is not None and _ships_strings(install, game or _stack.game, lang)


@lru_cache(maxsize=8)
def _ships_strings(install: Path, game: games.Game, lang: str) -> bool:
    folder = game.root(install) / STRINGS_DIR_TEMPLATE.format(lang=lang)
    return any((folder / name).is_file() for name in game.strings_files)


def clear_caches() -> None:
    """Drops what the resolver remembers of the install; asset_source.clear_install_caches() calls it."""
    _ships_strings.cache_clear()


# Every cache()'s per-key dict, so forget() can reach them.
_caches: list[dict[object, Callable]] = []


def forget(dataset: str) -> None:
    """Drops every cache() entry kept for `dataset`, with or without a data mod.
    For games.unregister(): the next game registered under that id must not read this one's tables."""
    for per_dataset in _caches:
        for key in [key for key in per_dataset if key == dataset or (isinstance(key, tuple) and key[0] == dataset)]:
            del per_dataset[key]


def cache(maxsize: int | None = 128):
    """functools.lru_cache, one cache per cache_key(): the switch itself neither
    clears nor re-parses it, though a switch's clear_install_caches()
    drops the install-derived ones (a data-mod switch only the table-keyed ones). cache_clear() drops every dataset's."""

    def decorate(fn):
        per_dataset: dict[object, Callable] = {}
        _caches.append(per_dataset)

        def _cached() -> Callable:
            key = _stack.tables_key
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
