"""Qt-free reader for the trigger effects that change a building's maximum
range, for View > Range Rings > Trigger Changes and the Inspector line (GH #49
slice 2).

Read-only: nothing here writes a trigger. Effects are frozen plain values and
hold no library object past collect_range_effects(): a later parse of another
scenario disables fields on the same library classes (trigger_geometry's rule).

Nothing is simulated. Conditions, firing order across triggers, loops and the
disabled-at-start flag are not evaluated: each trigger's own matching effects
are applied once, in that trigger's display order, to the base range, giving
one value per trigger ("if this trigger fires once").

Which placed building an effect reaches is inference about the game, kept in
one function, effect_applies(), so an in-game pass can flip a row in one edit:

- 51, 79 (type-wide): `source_player` is its owner and `object_list_unit_id`
  is its const.
- 104 (class-wide): `source_player` is its owner, `object_group2` is its class,
  and `object_type2` (when set) is its object type.
- 32, 105, 106 with `selected_object_ids`: its `reference_id` is listed. Every
  other filter is ignored.
- 32, 105, 106 without ids: `source_player` is its owner, and each SET filter
  passes: area contains its own tile (inclusive), `object_list_unit_id` is its
  const, `object_group` is its class, `object_type` is its object type. No
  filter set means every object of that player.

Operations: SET q, ADD base+q, SUBTRACT base-q, MULTIPLY base*q, DIVIDE base/q
(q == 0 skips the effect). `operation` unset is ADD on 32 (the effect predates
the field) and skips the effect on the others. A result below 0 clamps to 0.
The by-variable effects (79, 106) are counted, never applied.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from descape import game_dataset, object_catalog, trigger_model
from descape.trigger_fields import UNSET
from descape.trigger_geometry import AREA_FIELDS, coordinate

CHANGE_OBJECT_RANGE = 32
MODIFY_ATTRIBUTE = 51
MODIFY_ATTRIBUTE_BY_VARIABLE = 79
MODIFY_ATTRIBUTE_FOR_CLASS = 104
MODIFY_OBJECT_ATTRIBUTE = 105
MODIFY_OBJECT_ATTRIBUTE_BY_VARIABLE = 106

# ObjectAttribute.MAXIMUM_RANGE: the attribute-carrying effects count only for it.
MAXIMUM_RANGE_ATTRIBUTE = 12

ATTRIBUTE_EFFECTS = frozenset(
    {
        MODIFY_ATTRIBUTE,
        MODIFY_ATTRIBUTE_BY_VARIABLE,
        MODIFY_ATTRIBUTE_FOR_CLASS,
        MODIFY_OBJECT_ATTRIBUTE,
        MODIFY_OBJECT_ATTRIBUTE_BY_VARIABLE,
    }
)
RANGE_EFFECTS = ATTRIBUTE_EFFECTS | {CHANGE_OBJECT_RANGE}
TYPE_WIDE_EFFECTS = frozenset({MODIFY_ATTRIBUTE, MODIFY_ATTRIBUTE_BY_VARIABLE})
CLASS_WIDE_EFFECTS = frozenset({MODIFY_ATTRIBUTE_FOR_CLASS})
BY_VARIABLE_EFFECTS = frozenset({MODIFY_ATTRIBUTE_BY_VARIABLE, MODIFY_OBJECT_ATTRIBUTE_BY_VARIABLE})

# Operation enum values.
SET, ADD, SUBTRACT, MULTIPLY, DIVIDE = 1, 2, 3, 4, 5
OPERATIONS = frozenset({SET, ADD, SUBTRACT, MULTIPLY, DIVIDE})
_OPERATION_WORDS = {
    SET: "set",
    ADD: "add",
    SUBTRACT: "subtract",
    MULTIPLY: "multiply by",
    DIVIDE: "divide by",
}

# The object types a trigger's object_type filter takes, from the .dat's combat_level.
_OBJECT_TYPES = frozenset({1, 2, 3, 4})


class _State:
    """A no-data state for describe(), named in its repr."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


# Triggers not parsed yet (never forced), and a file whose triggers can't be read.
UNREAD = _State("UNREAD")
UNREADABLE = _State("UNREADABLE")

NOT_SIMULATED = (
    "Each trigger's own range effects are applied once to the base range. Conditions, "
    "the order triggers fire in, loops and the disabled-at-start flag are not simulated."
)
UNREAD_TOOLTIP = (
    "Triggers are read only once Triggers mode (or Find's Triggers tab) has opened them, "
    "so selecting a building never waits on a slow read."
)


@dataclass(frozen=True)
class RangeEffect:
    """One effect that may change maximum range, as plain values.

    `sequence` is its position in collect_range_effects()' output, so a
    lookup can restore trigger and display order after an index merge.
    `group` and `object_type` hold `object_group2`/`object_type2` for 104.
    `quantity` is None when unset or set from a variable."""

    sequence: int
    trigger_index: int
    trigger_name: str
    effect_index: int
    effect_type: int
    effect_name: str
    operation: int
    quantity: float | None
    source_player: int
    unit_const: int
    group: int
    object_type: int
    selected_ids: frozenset[int]
    area: tuple[int, int, int, int] | None
    has_conditions: bool
    looping: bool
    enabled: bool

    @property
    def by_variable(self) -> bool:
        return self.effect_type in BY_VARIABLE_EFFECTS


@dataclass(frozen=True)
class TriggerRange:
    """One trigger's result for one building: `value` after its matching
    effects, and those effects in display order."""

    trigger_index: int
    trigger_name: str
    value: float
    effects: tuple[RangeEffect, ...]


@dataclass(frozen=True)
class RangeEffectIndex:
    """Effects with selected ids keyed by each listed reference id, the rest by
    source player, so one unit's lookup touches only its candidates."""

    by_reference: Mapping[int, tuple[RangeEffect, ...]]
    by_player: Mapping[int, tuple[RangeEffect, ...]]
    effect_count: int

    def candidates(self, *, player_id: int, reference_id: int) -> tuple[RangeEffect, ...]:
        found = self.by_reference.get(reference_id, ()) + self.by_player.get(player_id, ())
        return tuple(sorted(found, key=lambda e: e.sequence))


def _read(obj: Any, attribute: str) -> Any:
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _int(obj: Any, attribute: str, listed: frozenset[str]) -> int:
    """A listed int field, UNSET when not listed, unset or not an int."""
    if attribute not in listed:
        return UNSET
    value = _read(obj, attribute)
    if isinstance(value, bool) or value is None:
        return UNSET
    try:
        return int(value)
    except (TypeError, ValueError):
        return UNSET


def _quantity(effect: Any, listed: frozenset[str]) -> float | None:
    """effect.quantity, the value the property form shows (quantity_float's
    alias), or None for unset (None, -1, not a number)."""
    if not listed & {"quantity", "quantity_float"}:
        return None
    value = _read(effect, "quantity")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number == UNSET else number


def _selected_ids(effect: Any, listed: frozenset[str]) -> frozenset[int]:
    if "selected_object_ids" not in listed:
        return frozenset()
    value = _read(effect, "selected_object_ids") or ()
    if isinstance(value, int):
        value = (value,)
    try:
        return frozenset(v for v in value if isinstance(v, int) and not isinstance(v, bool) and v >= 0)
    except TypeError:
        return frozenset()


def _area(effect: Any, listed: frozenset[str]) -> tuple[int, int, int, int] | None:
    """The area only when trigger_geometry would draw it: all four set."""
    if not all(f in listed for f in AREA_FIELDS):
        return None
    coords = tuple(coordinate(effect, f) for f in AREA_FIELDS)
    if any(c < 0 for c in coords):
        return None
    return coords  # type: ignore[return-value]


def collect_range_effects(manager: Any, vocabulary: Any) -> tuple[RangeEffect, ...]:
    """Every effect in `manager` that may change maximum range, one pass:
    32 always, 51/79/104/105/106 only with `object_attributes` 12. Per
    trigger, in display order. Reads only the attributes the effect's
    VocabularyEntry lists."""
    found: list[RangeEffect] = []
    for trigger_index, trigger in enumerate(_read(manager, "triggers") or ()):
        effects = _read(trigger, "effects") or ()
        if not effects:
            continue
        name = _read(trigger, "name")
        trigger_name = name if isinstance(name, str) else ""
        has_conditions = bool(_read(trigger, "conditions"))
        looping = bool(_read(trigger, "looping"))
        enabled = _read(trigger, "enabled")
        enabled = True if enabled is None else bool(enabled)
        for effect_index in trigger_model.entry_display_order(trigger, "effect"):
            effect = effects[effect_index]
            effect_type = _read(effect, "effect_type")
            if effect_type not in RANGE_EFFECTS:
                continue
            definition = vocabulary.effects.get(effect_type)
            if definition is None:
                continue
            listed = frozenset(definition.attributes)
            if effect_type in ATTRIBUTE_EFFECTS and _int(effect, "object_attributes", listed) != MAXIMUM_RANGE_ATTRIBUTE:
                continue
            class_wide = effect_type in CLASS_WIDE_EFFECTS
            found.append(
                RangeEffect(
                    sequence=len(found),
                    trigger_index=trigger_index,
                    trigger_name=trigger_name,
                    effect_index=effect_index,
                    effect_type=effect_type,
                    effect_name=definition.name,
                    operation=_int(effect, "operation", listed),
                    quantity=None if effect_type in BY_VARIABLE_EFFECTS else _quantity(effect, listed),
                    source_player=_int(effect, "source_player", listed),
                    unit_const=_int(effect, "object_list_unit_id", listed),
                    group=_int(effect, "object_group2" if class_wide else "object_group", listed),
                    object_type=_int(effect, "object_type2" if class_wide else "object_type", listed),
                    selected_ids=_selected_ids(effect, listed),
                    area=_area(effect, listed),
                    has_conditions=has_conditions,
                    looping=looping,
                    enabled=enabled,
                )
            )
    return tuple(found)


@game_dataset.cache(maxsize=1)
def _catalog_fields() -> dict[int, tuple[int | None, int]]:
    """unit_const -> (class, combat_level) from the active dataset's catalog."""
    return {
        const: (row.get("class"), int(row.get("combat_level", 0) or 0))
        for const, row in object_catalog.dat_objects().items()
    }


def class_of(unit_const: int) -> int | None:
    """object_catalog.json's `class` for a const, None when unknown."""
    fields = _catalog_fields().get(unit_const)
    return None if fields is None else fields[0]


def object_type_of(unit_const: int) -> int | None:
    """The trigger object type of a const: its baked .dat `combat_level` when
    1 to 4, else None, which matches no object_type filter rather than a
    guessed one. Return of Rome's catalog does not carry `combat_level` yet
    (regenerated without its .dat), so there every const gives None."""
    fields = _catalog_fields().get(unit_const)
    if fields is None:
        return None
    return fields[1] if fields[1] in _OBJECT_TYPES else None


def _area_contains(area: tuple[int, int, int, int], tile: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = area
    x, y = tile
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def effect_applies(
    effect: RangeEffect, *, player_id: int, unit_const: int, reference_id: int, own_tile: tuple[int, int]
) -> bool:
    """Whether `effect` reaches this placed building. The module docstring's
    table, the one place its game assumptions live."""
    if effect.effect_type in TYPE_WIDE_EFFECTS:
        return effect.source_player == player_id and effect.unit_const == unit_const
    if effect.effect_type in CLASS_WIDE_EFFECTS:
        if effect.source_player != player_id or effect.group == UNSET or effect.group != class_of(unit_const):
            return False
        return effect.object_type == UNSET or effect.object_type == object_type_of(unit_const)
    if effect.selected_ids:
        return reference_id in effect.selected_ids
    if effect.source_player != player_id:
        return False
    if effect.area is not None and not _area_contains(effect.area, own_tile):
        return False
    if effect.unit_const != UNSET and effect.unit_const != unit_const:
        return False
    if effect.group != UNSET and effect.group != class_of(unit_const):
        return False
    return effect.object_type == UNSET or effect.object_type == object_type_of(unit_const)


def apply_operation(effect: RangeEffect, value: float) -> float | None:
    """`value` after `effect`, or None when the effect is skipped: unset
    quantity, unknown or unset operation (except 32, where unset is ADD), or a
    divide by zero."""
    q = effect.quantity
    if q is None:
        return None
    operation = effect.operation
    if operation not in OPERATIONS:
        if effect.effect_type != CHANGE_OBJECT_RANGE or operation != UNSET:
            return None
        operation = ADD
    if operation == SET:
        return q
    if operation == ADD:
        return value + q
    if operation == SUBTRACT:
        return value - q
    if operation == MULTIPLY:
        return value * q
    if q == 0:
        return None
    return value / q


def index_effects(effects: Iterable[RangeEffect]) -> RangeEffectIndex:
    by_reference: dict[int, list[RangeEffect]] = {}
    by_player: dict[int, list[RangeEffect]] = {}
    count = 0
    for effect in effects:
        count += 1
        if effect.selected_ids and effect.effect_type not in TYPE_WIDE_EFFECTS | CLASS_WIDE_EFFECTS:
            for ref in effect.selected_ids:
                by_reference.setdefault(ref, []).append(effect)
        else:
            by_player.setdefault(effect.source_player, []).append(effect)
    return RangeEffectIndex(
        by_reference={k: tuple(v) for k, v in by_reference.items()},
        by_player={k: tuple(v) for k, v in by_player.items()},
        effect_count=count,
    )


def _matching(index: RangeEffectIndex, unit: dict) -> list[RangeEffect]:
    candidates = index.candidates(player_id=unit["player_id"], reference_id=unit["reference_id"])
    return [e for e in candidates if effect_applies(e, **unit)]


def trigger_ranges(
    index: RangeEffectIndex,
    base: float,
    *,
    player_id: int,
    unit_const: int,
    reference_id: int,
    own_tile: tuple[int, int],
) -> tuple[TriggerRange, ...]:
    """One TriggerRange per trigger with at least one applied match, in
    trigger order: that trigger's effects applied in display order to `base`,
    clamped at 0."""
    unit = {"player_id": player_id, "unit_const": unit_const, "reference_id": reference_id, "own_tile": own_tile}
    per_trigger: dict[int, list[RangeEffect]] = {}
    for effect in _matching(index, unit):
        if not effect.by_variable:
            per_trigger.setdefault(effect.trigger_index, []).append(effect)
    ranges = []
    for trigger_index, effects in per_trigger.items():
        value = float(base)
        applied = []
        for effect in effects:
            result = apply_operation(effect, value)
            if result is not None:
                value = result
                applied.append(effect)
        if applied:
            ranges.append(TriggerRange(trigger_index, applied[0].trigger_name, max(0.0, value), tuple(applied)))
    return tuple(ranges)


def unresolved_count(
    index: RangeEffectIndex,
    *,
    player_id: int,
    unit_const: int,
    reference_id: int,
    own_tile: tuple[int, int],
) -> int:
    """How many matching effects set the range from a variable (never drawn)."""
    unit = {"player_id": player_id, "unit_const": unit_const, "reference_id": reference_id, "own_tile": own_tile}
    return sum(1 for e in _matching(index, unit) if e.by_variable)


def format_value(value: float) -> str:
    """14.0 -> "14", 7.5 -> "7.5"."""
    return f"{value:g}"


def _effect_phrase(effect: RangeEffect) -> str:
    name = effect.effect_name.replace("_", " ")
    operation = effect.operation
    if operation == UNSET and effect.effect_type == CHANGE_OBJECT_RANGE:
        operation = ADD
    word = _OPERATION_WORDS.get(operation, "?")
    return f"{name}, {word} {format_value(effect.quantity or 0.0)}"


def _flags(effect: RangeEffect) -> list[str]:
    flags = []
    if effect.has_conditions:
        flags.append("has conditions")
    if effect.looping:
        flags.append("loops")
    if not effect.enabled:
        flags.append("disabled at start")
    return flags


def describe(
    ranges: tuple[TriggerRange, ...], unresolved: int, *, state: Any = None, ringable: bool
) -> tuple[str, str]:
    """(Inspector line, tooltip) for one selected unit. `state` is UNREAD,
    UNREADABLE or None (read); `ringable` says whether the unit could take a
    ring at all (a building). Empty strings hide the line."""
    plural = "effect" if unresolved == 1 else "effects"
    variable_line = f"Range set from a variable by {unresolved} trigger {plural} (not drawn)"
    if ranges:
        values: list[str] = []
        for r in ranges:
            text = format_value(r.value)
            if text not in values:
                values.append(text)
        rows = []
        for r in ranges:
            phrases = "; ".join(_effect_phrase(e) for e in r.effects)
            parts = [f'{format_value(r.value)}: "{r.trigger_name}"', phrases, *_flags(r.effects[0])]
            rows.append(", ".join(parts))
        if unresolved:
            rows.append(variable_line)
        return f"Range with triggers: {', '.join(values)}", "\n".join([*rows, NOT_SIMULATED])
    if unresolved:
        return variable_line, NOT_SIMULATED
    if ringable and state is UNREAD:
        return "Trigger range changes not read yet: open Triggers mode once", UNREAD_TOOLTIP
    if ringable and state is UNREADABLE:
        return "This file's triggers can't be read, so trigger range changes are not shown", ""
    return "", ""
