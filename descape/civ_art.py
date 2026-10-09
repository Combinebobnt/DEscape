"""Which civ's building art, and which age's, each player draws (GH #48).

A building's art depends on its owner's architecture and starting age, so the
renderer resolves a placed building const through a per-player
`(art_civ, age)` pair (`unit_sprites.resolve_entry()`). This module derives
that pair from the scenario's own player fields. Qt-free.

- `art_civ` is a .dat civ index (1..62): the architecture if it names a real
  civ, else the civilization if that does, else None, meaning today's
  (Gaia-table) art. Random values, unknown tokens and a missing
  `architecture_set` (a 1.37 file) fall through to the next choice.
- `age` is the StartingAge value, 2..5: 6 (Post-Imperial) draws as 5, and
  anything else (0 and 4294967295 appear on inactive players) as 2.
- GAIA is always `GAIA_ART`, so Gaia's buildings keep today's art.

Civilization values are ints below scenario version 1.56 (CivilizationOld,
equal to the .dat civ index) and str16 tokens such as 'HUN-CIV' from 1.56 on
(Civilization). A token maps through its member name to CivilizationOld; the
three civs CivilizationOld stops short of (.dat 60-62) use `_EXTRA_CIV_INDEX`,
which tests/test_civ_art.py checks against building_art_map.json's own civ
names.
"""

from __future__ import annotations

from collections.abc import Mapping

from AoE2ScenarioParser.datasets.object_support import Civilization, CivilizationOld

GAIA_ART: tuple[int | None, int] = (None, 2)
"""(art_civ, age) for GAIA, and for any player whose values resolve nowhere."""

MAX_CIV_INDEX = 62
"""The .dat's last civ index (Danes)."""

DARK_AGE = 2
IMPERIAL_AGE = 5
POST_IMPERIAL_AGE = 6

# Civilization member names past CivilizationOld's last member (Tupi, 59).
_EXTRA_CIV_INDEX: dict[str, int] = {"SAXONS": 60, "VARANGIANS": 61, "DANES": 62}

# The same three as raw tokens, for a library whose Civilization lacks them too.
_EXTRA_CIV_TOKENS: dict[str, int] = {"SAXONS-CIV": 60, "VARANGIANS-CIV": 61, "DANES-CIV": 62}

PLAYER_COUNT = 9
"""player_id 0 (GAIA) .. 8, the length of every per-player render tuple."""


def civ_index(value: object) -> int | None:
    """The .dat civ index a raw civilization/architecture value names, or None
    when it names no concrete civ (GAIA, a random choice, an unknown token)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= MAX_CIV_INDEX else None
    if not isinstance(value, str):
        return None
    try:
        name = Civilization(value).name
    except ValueError:
        return _EXTRA_CIV_TOKENS.get(value)
    if name in _EXTRA_CIV_INDEX:
        return _EXTRA_CIV_INDEX[name]
    try:
        index = CivilizationOld[name].value
    except KeyError:
        return None
    return index if 1 <= index <= MAX_CIV_INDEX else None


def art_age(value: object) -> int:
    """The StartingAge a building draws at: 2..5 as-is, Post-Imperial as
    Imperial, anything else as Dark Age."""
    if isinstance(value, bool) or not isinstance(value, int):
        return DARK_AGE
    if value == POST_IMPERIAL_AGE:
        return IMPERIAL_AGE
    return value if DARK_AGE <= value <= IMPERIAL_AGE else DARK_AGE


def resolve(civilization: object, architecture: object, starting_age: object) -> tuple[int | None, int]:
    """One player's (art_civ, age) from its three raw values."""
    art_civ = civ_index(architecture)
    if art_civ is None:
        art_civ = civ_index(civilization)
    return art_civ, art_age(starting_age)


_FIELDS = ("civilization", "architecture", "starting_age")


def player_art(
    loaded, pending: Mapping[str, Mapping[int, object]] | None = None,
) -> tuple[tuple[int | None, int], ...]:
    """(art_civ, age) for player_id 0..8 on `loaded`, the stored values
    overlaid with `pending` ({field_id: {player_id: value}}, the viewer's
    pending Players-mode edits). A field the file does not carry reads as
    missing, which falls through like a random value."""
    from descape import player_fields

    specs = {s.field_id: s for s in player_fields.specs_for(loaded) if s.field_id in _FIELDS}
    pending = pending or {}
    result = [GAIA_ART]
    for player_id in range(1, PLAYER_COUNT):
        values = []
        for field_id in _FIELDS:
            overlay = pending.get(field_id, {})
            if player_id in overlay:
                values.append(overlay[player_id])
            elif field_id in specs:
                values.append(player_fields.current_value(loaded, specs[field_id], player_id))
            else:
                values.append(None)
        result.append(resolve(*values))
    return tuple(result)
