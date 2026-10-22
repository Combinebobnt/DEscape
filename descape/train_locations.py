"""GH #230: every Change Train Location, Add Train Location and Change
Research Location effect in a file, and what they do to a building's panel.

Qt-free. The button grid applies the other effects for the same player when
"Show all triggers' changes" is ticked, and Map Analysis reads the same list.
Effects apply in file order (trigger id, then effect index): the order they
fire in game is not knowable at edit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from AoE2ScenarioParser.datasets.effects import EffectId

from descape import civ_art, civ_availability, command_buttons
from descape.civ_availability import NO_OVERRIDES, Item, Overrides
from descape.command_buttons import TECH, UNIT

CHANGE_TRAIN = "change_train_location"
ADD_TRAIN = "add_train_location"
CHANGE_RESEARCH = "change_research_location"
# effect name -> (field naming what moves, its kind)
EFFECT_MOVERS = {
    CHANGE_TRAIN: ("object_list_unit_id", UNIT),
    ADD_TRAIN: ("object_list_unit_id", UNIT),
    CHANGE_RESEARCH: ("technology", TECH),
}
BUILDING_FIELD = "object_list_unit_id_2"
SLOT_FIELD = "button_location"
PLAYER_FIELD = "source_player"

_EFFECT_BY_ID = {
    EffectId.CHANGE_TRAIN_LOCATION.value: CHANGE_TRAIN,
    EffectId.ADD_TRAIN_LOCATION.value: ADD_TRAIN,
    EffectId.CHANGE_TECHNOLOGY_LOCATION.value: CHANGE_RESEARCH,
}
TITLES = {
    CHANGE_TRAIN: "Change Train Location",
    ADD_TRAIN: "Add Train Location",
    CHANGE_RESEARCH: "Change Research Location",
}
# The two that move the unit or tech off its old slot; Add keeps it there too.
MOVES_AWAY = frozenset({CHANGE_TRAIN, CHANGE_RESEARCH})


@dataclass(frozen=True)
class LocationEdit:
    trigger_id: int
    trigger_name: str
    index: int
    effect: str
    player: int
    mover: int
    building: int
    value: int

    @property
    def kind(self) -> str:
        return EFFECT_MOVERS[self.effect][1]

    @property
    def item(self) -> Item:
        return (self.mover, self.kind)

    @property
    def label(self) -> str:
        """'Trigger 4 "Spanish setup": Change Train Location'."""
        return f'Trigger {self.trigger_id} "{self.trigger_name}": {TITLES[self.effect]}'


def _int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def location_edits(manager) -> tuple[LocationEdit, ...]:
    """Every location effect in `manager` (None when triggers did not parse), in file order."""
    edits = []
    triggers = sorted(getattr(manager, "triggers", None) or (), key=lambda t: _int(t.trigger_id))
    for trigger in triggers:
        for index, effect in enumerate(trigger.effects):
            name = _EFFECT_BY_ID.get(_int(getattr(effect, "effect_type", None)))
            if name is None:
                continue
            mover_field, _kind = EFFECT_MOVERS[name]
            edits.append(
                LocationEdit(
                    trigger_id=_int(trigger.trigger_id),
                    trigger_name=str(trigger.name),
                    index=index,
                    effect=name,
                    player=_int(getattr(effect, PLAYER_FIELD, None)),
                    mover=_int(getattr(effect, mover_field, None)),
                    building=_int(getattr(effect, BUILDING_FIELD, None)),
                    value=_int(getattr(effect, SLOT_FIELD, None)),
                )
            )
    return tuple(edits)


def for_player(edits, player: int) -> tuple[LocationEdit, ...]:
    """The edits that apply to `player`: its own and those for every player (-1)."""
    return tuple(edit for edit in edits if edit.player in (player, civ_availability.ALL_PLAYERS))


@dataclass(frozen=True)
class PlayerContext:
    """What the button grid needs about the effect's player."""

    player: int = -1
    # The .dat civ index, or None when unfiltered.
    civ: int | None = None
    # The civilization as text ("Spanish", "Random"), or "" with no player.
    civ_text: str = ""
    overrides: Overrides = NO_OVERRIDES
    # The other location effects for this player, in file order.
    edits: tuple[LocationEdit, ...] = ()
    # StartingAge (2..6), or None when unknown.
    start_age: int | None = None

    def missing(self, id_: int, kind: str) -> str:
        return civ_availability.missing(self.civ, self.overrides, id_, kind, self.start_age)

    def keep(self, id_: int, kind: str) -> bool:
        return not self.missing(id_, kind)

    def missing_text(self, id_: int, kind: str, name: str) -> str:
        """Why `name` gets no button, as a clause ("Spanish can't train Mule Cart"), or ""."""
        return missing_text(self.missing(id_, kind), kind, name, self.civ_text, self.player, self.start_age)


NO_CONTEXT = PlayerContext()


def player_context(manager, loaded, player: int, exclude: tuple[int, int] | None = None) -> PlayerContext:
    """`player`'s PlayerContext, leaving out the effect at `exclude`
    ((trigger id, effect index): the one being edited)."""
    if not 1 <= player < civ_art.PLAYER_COUNT:
        return PlayerContext(player=player)
    edits = tuple(
        edit
        for edit in for_player(location_edits(manager), player)
        if (edit.trigger_id, edit.index) != exclude
    )
    return PlayerContext(
        player=player,
        civ=civ_availability.player_civ(loaded, player),
        civ_text=civ_availability.civ_name(loaded, player),
        overrides=civ_availability.file_overrides(manager, loaded, player),
        edits=edits,
        start_age=civ_availability.player_start_age(loaded, player),
    )


def missing_text(reason: str, kind: str, name: str, civ_text: str, player: int, start_age: int | None) -> str:
    """civ_availability.missing()'s `reason` as a clause for a tooltip or a finding."""
    if reason == civ_availability.MISSING_START:
        age = civ_availability.AGE_NAMES.get(start_age, "")
        return f"player {player} starts in the {age} Age, which already has {name}"
    if reason == civ_availability.MISSING_LISTED:
        return f"{name} is in player {player}'s Disables list"
    if reason == civ_availability.MISSING_CIV:
        verb = "research" if kind == TECH else "train"
        return f"{civ_text} can't {verb} {name}"
    return ""


def enable_hint(reason: str, kind: str) -> str:
    """How to give the button back, or "" when nothing can (the tech is already researched)."""
    if reason == civ_availability.MISSING_START:
        return ""
    return "Enable/Disable Technology" if kind == TECH else "Enable/Disable Object"


def lands_here(edit: LocationEdit, building: int, page: str | None) -> bool:
    """Whether `edit` puts its button on `building`'s panel (on `page` of the builder menus)."""
    if edit.building != building or command_buttons.slot(edit.value) is None:
        return False
    if command_buttons.is_builder(building):
        return edit.kind == UNIT and command_buttons.villager_page(edit.mover) == page
    return True


def _sorted_cell(items) -> list[Item]:
    """Units first (they cover techs), each kind base first."""
    units = command_buttons.ordered([i for i, k in items if k == UNIT], UNIT)
    techs = command_buttons.ordered([i for i, k in items if k == TECH], TECH)
    return [(i, UNIT) for i in units] + [(i, TECH) for i in techs]


@dataclass
class Applied:
    """apply()'s result: the panel, plus which edits touched which button."""

    cells: dict[int, list[Item]]
    # button -> the edits that put a button there
    placed: dict[int, list[LocationEdit]]
    # button -> the edits that moved an item away from there
    removed: dict[int, list[LocationEdit]]
    # button -> the edits landing there whose unit or tech `keep` rejects (no button in game)
    unavailable: dict[int, list[LocationEdit]] = field(default_factory=dict)


def apply(cells, building: int, page: str | None, edits, keep=None) -> Applied:
    """`cells` ({button: (id, kind) items}) after `edits` in order. A Change
    takes its unit or tech off every slot, an edit landing on `building`
    adds it at its value, and edits `keep` rejects (the civ can't have it)
    change nothing but are listed in `unavailable` where they land."""
    result = Applied({button: list(items) for button, items in cells.items()}, {}, {})
    for edit in edits:
        if edit.effect not in EFFECT_MOVERS or edit.mover < 0:
            continue
        if keep is not None and not keep(edit.mover, edit.kind):
            if lands_here(edit, building, page):
                result.unavailable.setdefault(edit.value, []).append(edit)
            continue
        item = edit.item
        if edit.effect in MOVES_AWAY:
            for button, items in result.cells.items():
                if item in items:
                    items.remove(item)
                    result.removed.setdefault(button, []).append(edit)
        if lands_here(edit, building, page):
            items = result.cells.setdefault(edit.value, [])
            if item not in items:
                items.append(item)
            result.cells[edit.value] = _sorted_cell(items)
            result.placed.setdefault(edit.value, []).append(edit)
    result.cells = {button: items for button, items in result.cells.items() if items}
    return result
