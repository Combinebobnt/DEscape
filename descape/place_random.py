"""Pseudo catalog picks that place one of several real objects at random (GH #164).

The Units panel lists "Villager (Random)" beside the male (83) and female
(293) villagers. Its value is a negative sentinel, so it can never be a real
object id. The panel rolls a real const from it before each placement and
re-rolls after one, so every placement path (the hover ghost, the click,
place_rotation, the status line) still sees only real consts, and the ghost
shows the object the click then places. Scatter rolls once per placed copy.

Only the Units panel offers these rows: object_catalog.objects(), Find and
Replace and every other picker stay real-const only.

Qt-free and install-free, so the default tier covers it without a display.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import object_catalog

# Negative, so it can never collide with a real object id.
RANDOM_VILLAGER = -1


@dataclass(frozen=True)
class _RandomPick:
    label: str
    consts: tuple[int, ...]
    # The const that stands for the pick wherever one object must be shown.
    preview: int
    search_text: str


RANDOM_PICKS = {
    RANDOM_VILLAGER: _RandomPick(
        "Villager (Random)", (83, 293), 83, "villager (random) random male female"
    ),
}

# The catalog category the rows sit in.
GROUP = "Units"


def is_random(value) -> bool:
    return value in RANDOM_PICKS


def choices(value: int) -> tuple[int, ...]:
    """The real consts a placement of `value` picks from: itself for a real const."""
    pick = RANDOM_PICKS.get(value)
    return (value,) if pick is None else pick.consts


def roll(value: int, rng) -> int:
    """One real const for `value`, drawn with `rng.choice`; a real const is returned as is."""
    options = choices(value)
    return options[0] if len(options) == 1 else rng.choice(options)


def label(value: int) -> str:
    """The catalog name of a random pick. KeyError for a real const."""
    return RANDOM_PICKS[value].label


def preview_const(value: int) -> int:
    """The real const shown for a random pick (its sprite, portrait and stats). KeyError for a real const."""
    return RANDOM_PICKS[value].preview


def catalog_entries() -> tuple:
    """The rows as object_catalog.CatalogEntry, for the Units catalog browser
    (GH #68): in GROUP, under the preview const's class, never editor-hidden."""
    entries = []
    for value, pick in RANDOM_PICKS.items():
        preview = object_catalog.entry(object_catalog.objects(), pick.preview)
        class_id = None if preview is None else preview.class_id
        entries.append(object_catalog.CatalogEntry(value, pick.label, GROUP, class_id, False, pick.search_text))
    return tuple(entries)
