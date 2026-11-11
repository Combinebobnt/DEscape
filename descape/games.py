"""The games DEscape knows, one descriptor each (docs/GAME_DATA.md).

A Game is plain data: what a game is called, which scenario files are its
own, where its files sit in the install, which folder holds its generated
tables, and which of DEscape's readers it uses. Adding a game starts here,
with a new descriptor in GAMES; game_dataset resolves tables and install
files from the active one, and the Settings and New Map lists, the
regeneration tool and the per-game tests all read this registry.

Qt-free, imports nothing from descape, and holds no state but the registry
list, which only tests change (register() / unregister()). Everything
derived from it is computed per call, so a game registered later shows
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# The values of Game.reader, Game.art and Game.install that have code behind them.
IMPLEMENTED_READERS = frozenset({"aoe2scenarioparser"})
IMPLEMENTED_ART = frozenset({"sld"})
AOE2DE_INSTALL = "aoe2de"


@dataclass(frozen=True)
class Game:
    """One game's descriptor. Field by field in docs/GAME_DATA.md."""

    # Persisted (config.yaml game_data, New Map's choice, the stamp file): never rename one.
    id: str
    display_name: str
    # The folder of this game's generated tables (dat_derive.TABLE_NAMES).
    tables_dir: Path
    # Appended to the window title for this game's files.
    title_tag: str | None = None
    # Which configured install holds the game. Only AOE2DE_INSTALL resolves to a path so far.
    install: str = AOE2DE_INSTALL
    # The game's own root under the install, mirroring the install's layout; None is the install itself.
    root_subpath: str | None = None
    # The game whose files and strings fill in what this one's root lacks.
    fallback: str | None = None
    # The file header variants that are this game's; None is the default game, which takes every other.
    variants: frozenset[int] | None = None
    # This game's section of tools/game_data_stamp.json; None is the top level.
    stamp_key: str | None = None
    # Strings files under <root>/resources/<lang>/strings/key-value/, read in this order.
    strings_files: tuple[str, ...] = ("key-value-strings-utf8.txt",)
    # The widgetui materials file naming this game's own icon files, when it has one.
    icon_materials: str | None = None
    # A zoomed-out level reads terrain textures shrunk with it.
    texture_scales_with_level: bool = False
    # AoE2ScenarioParser's unit, building, hero and tech enums name this game's ids.
    library_enums: bool = True
    # File > Data Mod applies: a data mod replaces this game's .dat.
    data_mods: bool = True
    # File > New Map's donor under descape/templates/; None keeps the game out of New Map.
    blank_template: str | None = None
    # False makes every document of this game view-only.
    writable: bool = True
    # The scenario reader and the art format, one implemented value each so far.
    reader: str = "aoe2scenarioparser"
    art: str = "sld"

    def root(self, install: Path) -> Path:
        """The folder holding this game's own resources/ tree under `install`."""
        return install if self.root_subpath is None else install / self.root_subpath


AOE2 = Game(
    id="aoe2",
    display_name="Age of Empires II",
    tables_dir=_PACKAGE_DIR,
    blank_template="blank_120x120.aoe2scenario",
)
ROR = Game(
    id="ror",
    display_name="Return of Rome",
    tables_dir=_PACKAGE_DIR / "ror",
    title_tag=" [Return of Rome]",
    root_subpath="modes/Pompeii",
    fallback="aoe2",
    # ScenarioVariant.ROR, the file header's unknown_value_2.
    variants=frozenset({2}),
    stamp_key="ror",
    strings_files=("key-value-pompeii-strings-utf8.txt",),
    icon_materials="morematerials.json",
    texture_scales_with_level=True,
    library_enums=False,
    data_mods=False,
    blank_template="blank_ror_120x120.aoe2scenario",
)

# Registry order is the Settings and New Map order.
GAMES: list[Game] = [AOE2, ROR]
# The game of every file no other game's `variants` claims.
DEFAULT_ID = "aoe2"


def ids() -> tuple[str, ...]:
    return tuple(game.id for game in GAMES)


def get(game_id: str) -> Game:
    for game in GAMES:
        if game.id == game_id:
            return game
    raise ValueError(f"unknown game dataset {game_id!r}")


def default() -> Game:
    return get(DEFAULT_ID)


def for_variant(variant) -> Game:
    """The game of a file header's variant (ScenarioVariant or its int): the
    one claiming it, else the default. None and junk are the default's."""
    try:
        value = None if variant is None else int(variant)
    except (TypeError, ValueError):
        value = None
    if value is not None:
        for game in GAMES:
            if game.variants is not None and value in game.variants:
                return game
    return default()


def for_scenario(loaded) -> Game:
    """A loaded scenario's own game, whatever game data is active."""
    return for_variant(loaded.game_variant)


def chain(game: Game) -> tuple[Game, ...]:
    """`game`, then each game it falls back to, in lookup order."""
    found = [game]
    while found[-1].fallback is not None:
        following = get(found[-1].fallback)
        if following in found:
            raise ValueError(f"game {game.id!r} falls back in a cycle through {following.id!r}")
        found.append(following)
    return tuple(found)


def register(game: Game) -> None:
    """Test hook: adds `game` at the end of the registry."""
    if game.id in ids():
        raise ValueError(f"game {game.id!r} is already registered")
    GAMES.append(game)


def unregister(game_id: str) -> None:
    """Test hook: removes what register() added."""
    GAMES.remove(get(game_id))
