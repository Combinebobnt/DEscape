"""Qt-free completeness status for triggers, conditions and effects (GH #166):
OK when every required field is set and every reference resolves, PROBLEM
otherwise, with one short human reason per finding for the row's tooltip.

That is a completeness check, not a promise about runtime behaviour.

Two layers, each with its reason:

- **Universal checks** run on every type with no table: type 0 `none`, a
  dangling placed-unit reference, a dangling trigger id, an incoherent
  quantity cluster and a half-set coordinate group. They need no per-type
  knowledge, so they cannot false-positive on a type nobody has reviewed. A
  type id the vocabulary does not list is UNCHECKED, not PROBLEM: there is
  nothing to check it against (the one corpus case looks deliberate).
- **The required-field table** (`trigger_requirements.json`) is keyed by
  kind and type *name*, since names stay stable across version JSONs while
  ids do not. A row's `all` tokens must each be set; if it has `any` groups,
  at least one group must be fully set. A token is a field name or one of
  the pseudo-fields `area` and `location`. A token this version does not
  list for the type is skipped, never failed; an `any` group left with no
  listed tokens is dropped rather than passing vacuously, and a row whose
  every group is dropped has no `any` constraint in that version.

"Set" is `is_set()`, keyed on the field's FieldSpec kind, because `-1` is not
universally "missing": `source_player` defaults to 1, `object_state` to 2,
and condition `difficulty_level`'s `-1` is EXTREME. Which fields may be
required at all is the table test's job, not this module's.

Values are read like trigger_fields._live_value(): through `spec.attribute`
(`quantity_float` aliases to `quantity`) and inside try/except, since the
library's version-gated properties raise and the public `quantity` getter
raises on a broken armour pair. None reads as unset for every kind.

The caller builds the StatusContext, so this module needs no Qt and no
document. With no vocabulary every entry is UNCHECKED, and so is every
non-divider trigger (evaluate_trigger() never applies the zero-effects rule
then).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from AoE2ScenarioParser.datasets.conditions import ConditionId
from AoE2ScenarioParser.datasets.effects import EffectId

from . import trigger_fields, trigger_geometry, trigger_organize, unit_references
from .library_compat import TriggerVocabulary, VocabularyEntry
from .trigger_fields import FieldSpec

CONDITION = trigger_geometry.ENTRY_CONDITION
EFFECT = trigger_geometry.ENTRY_EFFECT

_TABLE_JSON_PATH = Path(__file__).resolve().parent / "trigger_requirements.json"

AREA = "area"
LOCATION = "location"
PSEUDO_FIELDS = frozenset({AREA, LOCATION})

_TYPE_ATTRIBUTE = {CONDITION: "condition_type", EFFECT: "effect_type"}
_TABLE_KEY = {CONDITION: "conditions", EFFECT: "effects"}

# The types get_trigger_referencing_ce() covers. Keyed on the field name, never
# on presentation: condition 79's trigger_id presentation is "", a plain INT.
_TRIGGER_ID_FIELD = "trigger_id"
TRIGGER_ID_TYPES = {
    CONDITION: frozenset({int(ConditionId.TRIGGER_ACTIVE)}),
    EFFECT: frozenset({int(EffectId.ACTIVATE_TRIGGER), int(EffectId.DEACTIVATE_TRIGGER)}),
}

_GROUP_NAMES = {
    trigger_geometry.AREA_FIELDS: "area",
    trigger_geometry.LOCATION_FIELDS: "location",
    trigger_geometry.WALL_FIELDS: "wall",
}


class Status(Enum):
    OK = "ok"
    PROBLEM = "problem"
    UNCHECKED = "unchecked"  # no vocabulary for this scenario version, or a type id it does not list
    NONE = "none"  # a section divider row, never styled


@dataclass(frozen=True)
class EntryStatus:
    status: Status
    reasons: tuple[str, ...] = ()


UNCHECKED = EntryStatus(Status.UNCHECKED)


@dataclass(frozen=True)
class Requirement:
    """One table row. `any` holds the alternative groups, each a token tuple."""

    all: tuple[str, ...] = ()
    any: tuple[tuple[str, ...], ...] = ()
    why: str = ""


@dataclass(frozen=True)
class StatusContext:
    """What a status pass reads besides the entry itself.

    `unit_exists` is called per placed-unit id and is backed by the caller's
    unit_references.ReferenceIndex. `requirements` overrides the committed
    table (kind key -> type name -> Requirement); None means load_table().
    """

    vocabulary: TriggerVocabulary | None
    trigger_ids: frozenset[int] = frozenset()
    unit_exists: Callable[[int], bool] = lambda _ref: True
    requirements: Mapping[str, Mapping[str, Requirement]] | None = None
    # (kind, type id) -> {field name: FieldSpec}; per context, so one vocabulary. field_specs() was 75% of a pass.
    _specs: dict[tuple[str, int], dict[str, FieldSpec]] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class TriggerStatuses:
    """One trigger's rollup plus its entries' statuses, in stored order."""

    trigger: EntryStatus
    conditions: tuple[EntryStatus, ...]
    effects: tuple[EntryStatus, ...]


# -- the table -----------------------------------------------------------------


def parse_table(raw: Mapping[str, Any]) -> dict[str, dict[str, Requirement]]:
    """{"conditions": {name: Requirement}, "effects": {...}} from the JSON shape."""
    table: dict[str, dict[str, Requirement]] = {}
    for kind_key in _TABLE_KEY.values():
        rows = {}
        for name, row in raw.get(kind_key, {}).items():
            rows[name] = Requirement(
                all=tuple(row.get("all", ())),
                any=tuple(tuple(group) for group in row.get("any", ())),
                why=str(row.get("why", "")),
            )
        table[kind_key] = rows
    return table


@lru_cache(maxsize=1)
def load_table() -> dict[str, dict[str, Requirement]]:
    return parse_table(json.loads(_TABLE_JSON_PATH.read_text(encoding="utf-8")))


# -- the "set" predicate ---------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_set(spec: FieldSpec, value: Any) -> bool:
    """Whether `value` counts as set for a field of `spec.kind`. Shared by the
    checks and tools/census_trigger_fields.py, so the two cannot disagree."""
    if value is None:
        return False
    kind = spec.kind
    if kind == trigger_fields.BOOL:
        return True
    if kind == trigger_fields.STR:
        return bool(str(value).strip())
    if kind == trigger_fields.INT_LIST:
        if not isinstance(value, (list, tuple)):
            return False
        return len(value) > 0 and list(value) != [trigger_fields.UNSET]
    if kind == trigger_fields.FLOAT:
        return value != float(trigger_fields.UNSET)
    # INT, ENUM, REFERENCE, and UNSUPPORTED (never a table token; census only).
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and list(value) != [trigger_fields.UNSET]
    return value != trigger_fields.UNSET


def read_value(entry: Any, attribute: str) -> Any:
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- version-gated properties and the broken-pair quantity getter raise
        return None


def _coordinate(entry: Any, attribute: str) -> int:
    value = read_value(entry, attribute)
    if not _is_number(value):
        return trigger_fields.UNSET
    return int(value)


def _unit_ids(value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v != trigger_fields.UNSET)


# -- per-entry status ------------------------------------------------------------


def _vocab(kind: str, vocabulary: TriggerVocabulary) -> tuple[Mapping[int, VocabularyEntry], Mapping[str, str]]:
    if kind == CONDITION:
        return vocabulary.conditions, vocabulary.condition_presentation
    return vocabulary.effects, vocabulary.effect_presentation


def definition_for(kind: str, entry: Any, vocabulary: TriggerVocabulary | None) -> VocabularyEntry | None:
    if vocabulary is None:
        return None
    entries, _ = _vocab(kind, vocabulary)
    return entries.get(read_value(entry, _TYPE_ATTRIBUTE[kind]))


def entry_label(kind: str, index: int, entry: Any, vocabulary: TriggerVocabulary | None) -> str:
    """`effect 3 (create object)`: the 1-based position and type name a
    trigger rollup prefixes each child reason with."""
    definition = definition_for(kind, entry, vocabulary)
    if definition is None:
        name = f"type {read_value(entry, _TYPE_ATTRIBUTE[kind])}"
    else:
        name = definition.name.replace("_", " ")
    return f"{kind} {index + 1} ({name})"


def _label(name: str) -> str:
    return name.replace("_", " ")


def _universal_reasons(
    kind: str, entry: Any, definition: VocabularyEntry, presentation: Mapping[str, str], context: StatusContext
) -> list[str]:
    reasons = [
        f"unit {ref_id} ({_label(attribute)}) is not placed on the map"
        for attribute, ids in unit_references.references_in(entry, definition, presentation).items()
        for ref_id in ids
        if not context.unit_exists(ref_id)
    ]

    if definition.id in TRIGGER_ID_TYPES[kind] and _TRIGGER_ID_FIELD in definition.attributes:
        target = read_value(entry, _TRIGGER_ID_FIELD)
        if _is_number(target) and target != trigger_fields.UNSET and target not in context.trigger_ids:
            reasons.append(f"trigger {target} does not exist")

    if kind == EFFECT:
        incoherence = trigger_fields.cluster_incoherence(entry)
        if incoherence:
            reasons.append(incoherence)

    for fields in trigger_geometry.coordinate_fields(definition):
        values = [_coordinate(entry, f) for f in fields]
        if any(v >= 0 for v in values) and any(v < 0 for v in values):
            reasons.append(f"{_GROUP_NAMES[fields]} is only partly set")
    return reasons


def _token_state(
    token: str, entry: Any, definition: VocabularyEntry, specs: Mapping[str, FieldSpec], context: StatusContext
) -> bool | None:
    """True/False for set/unset, or None when this version does not list the
    token's field(s) for the type, so the token is skipped."""
    listed = set(definition.attributes)
    if token == AREA:
        fields = [f for f in trigger_geometry.AREA_FIELDS if f in listed]
        if not fields:
            return None
        return all(_coordinate(entry, f) >= 0 for f in fields)
    if token == LOCATION:
        fields = [f for f in trigger_geometry.LOCATION_FIELDS if f in listed]
        has_reference = trigger_geometry.LOCATION_OBJECT_FIELD in listed
        if not fields and not has_reference:
            return None
        if fields and all(_coordinate(entry, f) >= 0 for f in fields):
            return True
        if has_reference:
            ids = _unit_ids(read_value(entry, trigger_geometry.LOCATION_OBJECT_FIELD))
            # A stale reference must not read as a set location.
            return bool(ids) and all(context.unit_exists(ref_id) for ref_id in ids)
        return False
    spec = specs.get(token)
    if token not in listed or spec is None:
        return None
    return is_set(spec, read_value(entry, spec.attribute))


def token_state(kind: str, token: str, entry: Any, context: StatusContext) -> bool | None:
    """Whether one table token is set on `entry`, or None when it would be
    skipped (unlisted in this version, or no vocabulary). For the census."""
    definition = definition_for(kind, entry, context.vocabulary)
    if definition is None:
        return None
    return _token_state(token, entry, definition, _specs_by_name(kind, definition, context), context)


def field_specs_for(kind: str, definition: VocabularyEntry, vocabulary: TriggerVocabulary) -> tuple[FieldSpec, ...]:
    """trigger_fields.field_specs() for one type: the specs is_set() is keyed on."""
    _, presentation = _vocab(kind, vocabulary)
    return trigger_fields.field_specs(definition, presentation, _TYPE_ATTRIBUTE[kind])


def _specs_by_name(kind: str, definition: VocabularyEntry, context: StatusContext) -> dict[str, FieldSpec]:
    key = (kind, definition.id)
    specs = context._specs.get(key)
    if specs is None:
        specs = {spec.name: spec for spec in field_specs_for(kind, definition, context.vocabulary)}
        context._specs[key] = specs
    return specs


def _table_reasons(
    kind: str, requirement: Requirement, entry: Any, definition: VocabularyEntry, context: StatusContext
) -> list[str]:
    specs = _specs_by_name(kind, definition, context)
    reasons = [
        f"missing {_label(token)}"
        for token in requirement.all
        if _token_state(token, entry, definition, specs, context) is False
    ]

    groups = []
    for group in requirement.any:
        states = [_token_state(token, entry, definition, specs, context) for token in group]
        kept = [(token, state) for token, state in zip(group, states, strict=True) if state is not None]
        if kept:
            groups.append(kept)
    if groups and not any(all(state for _token, state in group) for group in groups):
        options = " or ".join(" + ".join(_label(token) for token, _state in group) for group in groups)
        reasons.append(f"needs {options}")
    return reasons


def entry_status(kind: str, entry: Any, context: StatusContext) -> EntryStatus:
    """The status of one condition (`kind` CONDITION) or effect (EFFECT).
    UNCHECKED with no vocabulary, or for a type id the vocabulary does not list."""
    vocabulary = context.vocabulary
    if vocabulary is None:
        return UNCHECKED
    entries, presentation = _vocab(kind, vocabulary)
    type_attribute = _TYPE_ATTRIBUTE[kind]
    type_id = read_value(entry, type_attribute)
    if type_id == 0:
        return EntryStatus(Status.PROBLEM, (f"no {kind} type chosen",))
    definition = entries.get(type_id)
    if definition is None:
        return UNCHECKED

    reasons = _universal_reasons(kind, entry, definition, presentation, context)
    table = context.requirements if context.requirements is not None else load_table()
    requirement = table.get(_TABLE_KEY[kind], {}).get(definition.name)
    if requirement is not None:
        reasons += _table_reasons(kind, requirement, entry, definition, context)
    if reasons:
        return EntryStatus(Status.PROBLEM, tuple(reasons))
    return EntryStatus(Status.OK)


# -- rollups ---------------------------------------------------------------------


def group_status(labelled: Iterable[tuple[str, EntryStatus]]) -> EntryStatus:
    """The rollup of several entries: any PROBLEM wins (each reason prefixed
    with its entry's label), then any UNCHECKED, else OK. An empty group is OK."""
    reasons: list[str] = []
    unchecked = False
    for label, status in labelled:
        if status.status is Status.PROBLEM:
            reasons += [f"{label}: {reason}" for reason in status.reasons]
        elif status.status is Status.UNCHECKED:
            unchecked = True
    if reasons:
        return EntryStatus(Status.PROBLEM, tuple(reasons))
    if unchecked:
        return UNCHECKED
    return EntryStatus(Status.OK)


def _is_divider(trigger: Any) -> bool:
    return trigger_organize.is_divider(str(read_value(trigger, "name") or ""))


_OBJECTIVE_FLAGS = ("display_as_objective", "header", "display_on_screen")


def needs_effects(trigger: Any) -> bool:
    """Whether zero effects makes `trigger` a PROBLEM: not for a divider, and
    not for an objective, an objective header or an on-screen display
    (Trigger.display_as_objective / .header / .display_on_screen), which may
    exist only to show text (user decisions, 2026-10-07)."""
    if _is_divider(trigger):
        return False
    return not any(read_value(trigger, flag) for flag in _OBJECTIVE_FLAGS)


def trigger_status(trigger: Any, entry_statuses: Sequence[tuple[str, EntryStatus]]) -> EntryStatus:
    """A trigger's rollup over its labelled condition and effect statuses.
    A divider is NONE; zero effects is a PROBLEM unless the trigger is shown
    as an objective, an objective header or on screen (needs_effects()). `enabled` is
    ignored: status is about completeness, not whether the trigger fires."""
    if _is_divider(trigger):
        return EntryStatus(Status.NONE)
    rolled = group_status(entry_statuses)
    if not list(read_value(trigger, "effects") or []) and needs_effects(trigger):
        return EntryStatus(Status.PROBLEM, ("has no effects", *rolled.reasons))
    return rolled


def evaluate_trigger(trigger: Any, context: StatusContext) -> TriggerStatuses:
    """Every entry's status plus the trigger rollup. With no vocabulary every
    entry and every non-divider trigger is UNCHECKED, zero effects included."""
    conditions = list(read_value(trigger, "conditions") or [])
    effects = list(read_value(trigger, "effects") or [])
    if context.vocabulary is None:
        rollup = trigger_status(trigger, []) if _is_divider(trigger) else UNCHECKED
        return TriggerStatuses(rollup, (UNCHECKED,) * len(conditions), (UNCHECKED,) * len(effects))
    condition_statuses = tuple(entry_status(CONDITION, c, context) for c in conditions)
    effect_statuses = tuple(entry_status(EFFECT, e, context) for e in effects)
    labelled = [
        *(
            (entry_label(CONDITION, i, c, context.vocabulary), s)
            for i, (c, s) in enumerate(zip(conditions, condition_statuses, strict=True))
        ),
        *(
            (entry_label(EFFECT, i, e, context.vocabulary), s)
            for i, (e, s) in enumerate(zip(effects, effect_statuses, strict=True))
        ),
    ]
    return TriggerStatuses(trigger_status(trigger, labelled), condition_statuses, effect_statuses)
