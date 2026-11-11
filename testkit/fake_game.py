"""A throwaway game in descape.games' registry, for tests of code that reads it.

`registered()` always takes the game back out, leaves AoE2 active when the
test left the fake one active, and drops what game_dataset.cache() kept for
it, so no later test sees it or its tables.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from descape import game_dataset, games

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


@contextmanager
def registered(game: games.Game) -> Iterator[games.Game]:
    games.register(game)
    try:
        yield game
    finally:
        try:
            if game_dataset.active() == game.id:
                game_dataset.set_active(games.DEFAULT_ID)
        finally:
            games.unregister(game.id)
            game_dataset.forget(game.id)
