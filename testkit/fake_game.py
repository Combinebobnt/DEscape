"""A throwaway game in descape.games' registry, for tests of code that reads it.

`registered()` gives every PerGame table the default game's entry (shared, not copied) for it,
and always takes the game and those entries back out, leaves AoE2 active
when the test left the fake one active, and drops what game_dataset.cache()
kept for it, so no later test sees it or its tables.
"""

from __future__ import annotations

import importlib
import pkgutil
import shutil
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from descape import dat_derive, game_dataset, games, scenario_io

FAKE_ID = "fake"
FAKE_VARIANT = 77


def fake(tables_dir: Path, **overrides) -> games.Game:
    """A third game shaped like Return of Rome: its own root, variant and stamp section, AoE2 as fallback."""
    game = games.Game(
        id=FAKE_ID,
        display_name="Fake Game",
        tables_dir=tables_dir,
        root_subpath="modes/Fake",
        fallback=games.DEFAULT_ID,
        variants=frozenset({FAKE_VARIANT}),
        stamp_key=FAKE_ID,
    )
    return replace(game, **overrides)


def view_only(directory: Path, **overrides) -> games.Game:
    """fake() over a copy of AoE2's generated tables in `directory`, with writable=False:
    activating it refills every imported table with real rows, as AoE2's."""
    tables = directory / "fake_tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(games.AOE2.tables_dir / name, tables / name)
    return fake(tables, writable=False, **overrides)


def scenario_copy(directory: Path, source: Path = scenario_io.BLANK_TEMPLATE_PATH, name: str = "fake_blank.aoe2scenario") -> Path:
    """`source` with its header variant (unknown_value_2) set to FAKE_VARIANT, in `directory`; the new path."""
    raw = bytearray(source.read_bytes())
    loaded = scenario_io.load_map_and_units(source)
    start, end = scenario_io._header_field_spans(loaded.header_bytes, loaded._scenario.sections["FileHeader"].retriever_map)[
        "unknown_value_2"
    ]
    assert end - start == 4
    struct.pack_into("<I", raw, start, FAKE_VARIANT)
    path = directory / name
    path.write_bytes(bytes(raw))
    return path


def import_every_module() -> None:
    """Imports every descape module, so games.TABLES holds every PerGame table."""
    import descape

    for module in pkgutil.walk_packages(descape.__path__, "descape."):
        importlib.import_module(module.name)


@contextmanager
def registered(game: games.Game) -> Iterator[games.Game]:
    """`game` in the registry for the block, each PerGame table sharing the default game's entry
    (the same object: mutate it in place and AoE2's changes too)."""
    import_every_module()
    games.register(game)
    filled = []
    try:
        for table in games.TABLES:
            table[game.id] = table[games.DEFAULT_ID]
            filled.append(table)
        yield game
    finally:
        try:
            if game_dataset.active() == game.id:
                game_dataset.set_active(games.DEFAULT_ID)
        finally:
            for table in filled:
                del table[game.id]
            games.unregister(game.id)
            game_dataset.forget(game.id)
