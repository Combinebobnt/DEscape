"""The games DEscape knows, one descriptor each (docs/GAME_DATA.md).

A Game is plain data: what a game is called, which scenario files are its
own, where its files sit in the install, which folder holds its generated
tables, and which of DEscape's readers it uses. Adding a game starts here,
with a new descriptor in GAMES; game_dataset resolves tables and install
files from the active one, and the Settings and New Map lists, the
regeneration tool and the per-game tests all read this registry.

What a game's files cannot tell DEscape (its wall and cliff consts, its civ
list) is a PerGame table in the module that uses it, one entry per game.

Qt-free, imports nothing from descape at load time, and holds no state but
the registry list, which only tests change (register() / unregister()), and
the list of PerGame tables. Everything derived from it is computed per call,
so a game registered later shows everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# The values of Game.reader, Game.art and Game.install that have code behind them.
IMPLEMENTED_READERS = frozenset({"aoe2scenarioparser"})
IMPLEMENTED_ART = frozenset({"sld", "slp"})
AOE2DE_INSTALL = "aoe2de"
# The 1999 game's own install (Age of Kings / The Conquerors), set apart from the AoE2:DE one.
AOC_INSTALL = "aoc_install"
# Age of Empires II: HD Edition's own install (TASK-431).
HD_INSTALL = "hd_install"
IMPLEMENTED_INSTALLS = frozenset({AOE2DE_INSTALL, AOC_INSTALL, HD_INSTALL})
# The 1997 game's own install (Age of Empires with Rise of Rome); not implemented until AOE1 is registered.
AOE1_INSTALL = "aoe1_install"
# A DE game's .dat under its root.
DE_DAT_SUBPATH = "resources/_common/dat/empires2_x2_p1.dat"


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
    # Which configured install holds the game, one of IMPLEMENTED_INSTALLS.
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
    # AoE2ScenarioParser's unit, building, hero and tech enums name this game's ids.
    library_enums: bool = True
    # File > Data Mod applies: a data mod replaces this game's .dat.
    data_mods: bool = True
    # File > New Map's donor under descape/templates/; None keeps the game out of New Map.
    blank_template: str | None = None
    # False makes every document of this game view-only.
    writable: bool = True
    # The scenario reader and the art format: IMPLEMENTED_READERS, IMPLEMENTED_ART.
    reader: str = "aoe2scenarioparser"
    art: str = "sld"
    # The scenario_format labels (ScenarioFormat.game) that are this game's, checked before `variants`.
    formats: frozenset[str] = frozenset()
    # The game whose data shows this game's files while its own install is not configured.
    stand_in: str | None = None
    # Strings files are read from whichever language folder the root has, not only DEscape's own.
    strings_any_language: bool = False
    # The .dat, relative to the generators' root for this game, tried in order.
    dat_subpaths: tuple[str, ...] = (DE_DAT_SUBPATH,)
    # Runtime reads a folder built from the install (mirroring DE's layout), not the install itself.
    derived_root: bool = False
    # scenario_format labels no game claims whose files this game's data shows only when picked by hand.
    hand_pick_formats: frozenset[str] = frozenset()

    def root(self, install: Path) -> Path:
        """The folder holding this game's own resources/ tree under `install`."""
        return install if self.root_subpath is None else install / self.root_subpath


AOE2 = Game(
    id="aoe2",
    display_name="Age of Empires II",
    tables_dir=_PACKAGE_DIR,
    # The antiquity civs' units, buildings and techs are named in the paphos file only.
    strings_files=("key-value-strings-utf8.txt", "key-value-paphos-strings-utf8.txt"),
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
    # scenario_format.AOE1: Age of Empires (1997) and Rise of Rome files, shown with this game's data.
    formats=frozenset({"aoe1-1.15"}),
    stamp_key="ror",
    strings_files=("key-value-pompeii-strings-utf8.txt",),
    icon_materials="morematerials.json",
    library_enums=False,
    data_mods=False,
    blank_template="blank_ror_120x120.aoe2scenario",
)
# The 1999 game: Age of Kings files too, whose consts and civs The Conquerors kept. View-only.
AOC = Game(
    id="aoc",
    display_name="Age of Kings / The Conquerors",
    tables_dir=_PACKAGE_DIR / "aoc",
    title_tag=" [The Conquerors]",
    install=AOC_INSTALL,
    variants=frozenset(),
    stamp_key="aoc",
    library_enums=False,
    data_mods=False,
    writable=False,
    art="slp",
    # scenario_format.AOK, AOC and UP; games.py imports nothing from descape, tests/test_games.py pins them.
    formats=frozenset({"scn-aok", "scn-aoc", "scn-up"}),
    stand_in="aoe2",
    strings_any_language=True,
    dat_subpaths=("Data/empires2_x1_p1.dat", "Data/empires2_x1.dat"),
    derived_root=True,
)
# The 1997 game with Rise of Rome. Unregistered: registering it needs its install kind and every PerGame entry.
AOE1 = Game(
    id="aoe1",
    display_name="Age of Empires (Rise of Rome)",
    tables_dir=_PACKAGE_DIR / "aoe1",
    title_tag=" [Age of Empires]",
    install=AOE1_INSTALL,
    variants=frozenset(),
    stamp_key="aoe1",
    library_enums=False,
    data_mods=False,
    writable=False,
    art="slp",
    # Its 1.15 files stay Return of Rome's (ROR.formats) until its art draws.
    stand_in="ror",
    strings_any_language=True,
    dat_subpaths=("data2/empires.dat",),
    derived_root=True,
)
# Age of Empires II: HD Edition (TASK-431). View-only. Its header-2 bands only; its header-3 .aoe2scenario
# stays AoE2:DE's (no game claims it), and this game's data shows one only by a hand pick in Settings.
HD = Game(
    id="hd",
    display_name="Age of Empires II: HD Edition",
    tables_dir=_PACKAGE_DIR / "hd",
    title_tag=" [HD Edition]",
    install=HD_INSTALL,
    variants=frozenset(),
    stamp_key="hd",
    library_enums=False,
    data_mods=False,
    writable=False,
    art="slp",
    # scenario_format.HD: data versions 1.21, 1.24 and 1.26.
    formats=frozenset({"scn-hd"}),
    stand_in="aoe2",
    strings_any_language=True,
    derived_root=True,
    # scenario_format.HD_DE.
    hand_pick_formats=frozenset({"hd-aoe2scenario"}),
)

# Registry order is the Settings and New Map order.
GAMES: list[Game] = [AOE2, ROR, AOC, HD]
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


def for_format(label: str | None) -> Game | None:
    """The game claiming a scenario_format label through Game.formats, or None."""
    if label is not None:
        for game in GAMES:
            if label in game.formats:
                return game
    return None


def for_scenario(loaded) -> Game:
    """A loaded scenario's own game, whatever game data is active: the one
    claiming its format, else its header variant's. A duck-typed scenario
    without either is the default game's. Which data shows it is
    game_dataset.file_game()."""
    fmt = getattr(loaded, "source_format", None)
    claimed = for_format(getattr(fmt, "game", None))
    return claimed if claimed is not None else for_variant(getattr(loaded, "game_variant", None))


def chain(game: Game) -> tuple[Game, ...]:
    """`game`, then each game it falls back to, in lookup order."""
    found = [game]
    while found[-1].fallback is not None:
        following = get(found[-1].fallback)
        if following in found:
            raise ValueError(f"game {game.id!r} falls back in a cycle through {following.id!r}")
        found.append(following)
    return tuple(found)


class PerGame(dict):
    """A hand-measured value per game id (docs/GAME_DATA.md, "Per-game
    tables"): no default and no inheritance, so a registered game missing
    from one fails tests/test_games.py's completeness test, which lists
    `what` for each. A dict, so a test patches one game's entry with
    monkeypatch.setitem()."""

    def __init__(self, what: str, values: dict[str, object]):
        super().__init__(values)
        self.what = what
        TABLES.append(self)

    def active(self):
        """The active game's value, read at call time."""
        global _game_dataset
        if _game_dataset is None:
            # Bound on first use, not at import: game_dataset imports this module. Per-unit render paths call this.
            from descape import game_dataset as _game_dataset
        return self[_game_dataset.game().id]

    def for_game(self, game: Game):
        """`game`'s value, for a file's own game (for_scenario()) rather than the active one."""
        return self[game.id]


# Every PerGame built so far, for the completeness test.
TABLES: list[PerGame] = []
_game_dataset = None


def register(game: Game) -> None:
    """Test hook: adds `game` at the end of the registry."""
    if game.id in ids():
        raise ValueError(f"game {game.id!r} is already registered")
    GAMES.append(game)


def unregister(game_id: str) -> None:
    """Test hook: removes what register() added."""
    GAMES.remove(get(game_id))
