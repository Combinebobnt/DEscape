"""GH #230: whether a player's civ can have a unit, building or tech.

Qt-free. The one rule the button grid and Map Analysis share. It reads
tools/gen_object_catalog.py's `civs` key, which is derived from each civ's
tech tree, so it is a model of the .dat, not a measurement per civ. An id
without the key (generic units, and anything with no train or research
location) counts as available to every civ.

Measured in game (2026-10-16, Spanish, Imperial): Change Train Location of a
unit the civ lacks (Mule Cart, Longbowman) gives no button, one it has
(Conquistador) shows, and the Mule Cart shows once Enable/Disable Object
enables it. So a trigger that enables an id makes it available whatever the
civ, and an id in the Options Disables list that no trigger enables is not.
A tech a trigger enables or researches also makes the units it gives (the
catalog's `gives`: enabled or upgraded into) available: a file that enables
Elite Kamayuk for a Muisca player can then move the Elite Kamayuk.

A civ of None means unfiltered: a random or unset civilization, or a
player the file has no civilization for.

A player starting in a later age already has every generic tech of an
earlier one (measured 2026-10-16: Loom at a Feudal start, Bloodlines at
Imperial, Chemistry only at Post-Imperial), so such a tech has no button
(researched_at_start(), from the catalog's tech `age`). Unique techs were not
measured and are left out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from AoE2ScenarioParser.datasets.effects import EffectId

from descape import civ_art, command_buttons, disables_fields, game_dataset
from descape.command_buttons import TECH, UNIT

Item = tuple[int, str]
"""(id, kind), as the button grid's cells hold them: kind is UNIT for objects
(units and buildings), TECH for techs."""

ALL_PLAYERS = -1

_DISABLES_KINDS = {"units": UNIT, "buildings": UNIT, "techs": TECH}
# Effects that enable or research a tech, so its `gives` units become available.
_TECH_EFFECTS = frozenset(
    {
        EffectId.ENABLE_DISABLE_TECHNOLOGY,
        EffectId.RESEARCH_TECHNOLOGY,
        EffectId.INITIATE_RESEARCH,
        EffectId.RESEARCH_LOCAL_TECHNOLOGY,
    }
)


@dataclass(frozen=True)
class Overrides:
    """What the file itself turns on or off for one player."""

    # Enabled or researched by a trigger, plus the units such a tech gives.
    enabled: frozenset[Item] = field(default_factory=frozenset)
    # Disabled by one of those effects.
    disabled: frozenset[Item] = field(default_factory=frozenset)
    # In the Options Disables list.
    listed: frozenset[Item] = field(default_factory=frozenset)

    def turned_off(self, item: Item) -> bool:
        """Disabled by a trigger or the Disables list."""
        return item in self.disabled or item in self.listed


NO_OVERRIDES = Overrides()


def civs_of(id_: int, kind: str) -> tuple[int, ...] | None:
    """The civ indices that can have `id_`, or None when every civ can."""
    civs = command_buttons.catalog_row(id_, kind).get("civs")
    return None if civs is None else tuple(civs)


def has(civ: int | None, id_: int, kind: str) -> bool:
    """Whether `civ`'s tech tree gives it `id_`. A civ of None is unfiltered."""
    if civ is None:
        return True
    civs = civs_of(id_, kind)
    return civs is None or civ in civs


def available(civ: int | None, overrides: Overrides, id_: int, kind: str) -> bool:
    """has(), unless the file enables `id_` (available) or lists it in the
    Disables list without enabling it (not available)."""
    item = (id_, kind)
    if item in overrides.enabled:
        return True
    if item in overrides.listed:
        return False
    return has(civ, id_, kind)


# StartingAge values; anything else (0 and 4294967295 on inactive players) is no start age.
AGE_NAMES = {2: "Dark", 3: "Feudal", 4: "Castle", 5: "Imperial", 6: "Post-Imperial"}

# Why missing() finds no button: the civ lacks it, the Disables list has it, or the start age researched it.
MISSING_CIV = "civ"
MISSING_LISTED = "disables"
MISSING_START = "start"


def researched_at_start(start_age: int | None, tech: int) -> bool:
    """Whether a player starting in `start_age` already has generic `tech`."""
    if start_age is None:
        return False
    row = command_buttons.catalog_row(tech, TECH)
    return row.get("civ", -1) == -1 and "age" in row and start_age > row["age"]


def missing(civ: int | None, overrides: Overrides, id_: int, kind: str, start_age: int | None = None) -> str:
    """"" when `id_` gets a button, else MISSING_START, MISSING_LISTED or MISSING_CIV."""
    if kind == TECH and researched_at_start(start_age, id_):
        return MISSING_START
    if available(civ, overrides, id_, kind):
        return ""
    return MISSING_LISTED if (id_, kind) in overrides.listed else MISSING_CIV


def keeper(civ: int | None, overrides: Overrides, start_age: int | None = None) -> Callable[[int, str], bool]:
    """missing() as the `keep(id_, kind)` predicate command_buttons takes."""
    return lambda id_, kind: not missing(civ, overrides, id_, kind, start_age)


def _player_value(loaded, player: int, field_id: str):
    if not isinstance(player, int) or not 1 <= player < civ_art.PLAYER_COUNT:
        return None
    from descape import player_fields

    spec = next((s for s in player_fields.specs_for(loaded) if s.field_id == field_id), None)
    if spec is None:
        return None
    try:
        return player_fields.current_value(loaded, spec, player)
    except (ValueError, IndexError, AttributeError):
        return None


def player_civ_value(loaded, player: int):
    """`player`'s raw civilization value, or None (GAIA, unset, unreadable)."""
    return _player_value(loaded, player, "civilization")


def player_start_age(loaded, player: int) -> int | None:
    """`player`'s StartingAge (2 Dark .. 6 Post-Imperial), or None."""
    value = _player_value(loaded, player, "starting_age")
    return value if value in AGE_NAMES and not isinstance(value, bool) else None


def player_civ(loaded, player: int) -> int | None:
    """The .dat civ index `player` plays, or None for a random, unset or
    unknown civilization. The civilization, never the architecture, read
    with the file's own game's civ tokens (civ_art.player_art() does the same)."""
    ror = game_dataset.from_variant(getattr(loaded, "game_variant", None)) == game_dataset.ROR
    return civ_art.civ_index(player_civ_value(loaded, player), ror=ror)


def civ_name(loaded, player: int) -> str:
    """The player's civilization as text ("Spanish"), or "" when unreadable."""
    from descape import player_fields

    value = player_civ_value(loaded, player)
    return "" if value is None else player_fields.resolve_civilization_name(value)


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _for_player(effect, player: int) -> bool:
    source = _int(getattr(effect, "source_player", None))
    return source in (player, ALL_PLAYERS)


def file_overrides(manager, loaded, player: int) -> Overrides:
    """`player`'s Overrides from the triggers in `manager` (None when they
    did not parse) and the Disables list in `loaded` (None to skip it).
    An effect with source_player -1 counts for every player."""
    enabled: set[Item] = set()
    disabled: set[Item] = set()
    for trigger in getattr(manager, "triggers", None) or ():
        for effect in trigger.effects:
            effect_type = _int(getattr(effect, "effect_type", None))
            if effect_type == EffectId.ENABLE_DISABLE_OBJECT:
                kind, id_ = UNIT, _int(getattr(effect, "object_list_unit_id", None))
            elif effect_type in _TECH_EFFECTS:
                kind, id_ = TECH, _int(getattr(effect, "technology", None))
            else:
                continue
            if id_ is None or id_ < 0 or not _for_player(effect, player):
                continue
            switch = effect_type in (EffectId.ENABLE_DISABLE_OBJECT, EffectId.ENABLE_DISABLE_TECHNOLOGY)
            if switch and _int(getattr(effect, "enabled", None)) != 1:
                disabled.add((id_, kind))
                continue
            enabled.add((id_, kind))
            if kind == TECH:
                enabled.update((unit, UNIT) for unit in command_buttons.catalog_row(id_, TECH).get("gives", ()))
    listed: set[Item] = set()
    if loaded is not None and 1 <= player <= disables_fields.NUM_EDITABLE_PLAYERS:
        for category, kind in _DISABLES_KINDS.items():
            listed.update((id_, kind) for id_ in disables_fields.current_ids(loaded, category, player))
    return Overrides(frozenset(enabled), frozenset(disabled), frozenset(listed))
