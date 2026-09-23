"""
Field specs for the trigger property editor: what widget each condition,
effect, or trigger field needs, derived from the library's own per-version
vocabulary JSON.

Deliberately Qt-free. TriggerPanel builds widgets from these specs and writes
values back with plain setattr(), which keeps the derivation testable in the
default tier without a QApplication and keeps the panel itself dumb.

Three things about the underlying data shape drove the design.

1. **Kind is per (type, field), not per field name.** The same attribute name
   is scalar on one type and list-valued on another: `quantity` defaults to -1
   on most effects and to [] on the armour/attack ones. So a spec is always
   derived from one VocabularyEntry's own default_attributes, never from a
   global field-name table.
2. **Condition/Effect attributes are instance attributes**, assigned in
   __init__, not class properties. getattr/setattr on a live entry is the whole
   read/write path; hasattr() against the class says nothing, and says it
   differently depending on whether depoison() has run.
3. **The presentation map names an enum, not a widget.** The library's own
   attr_presentation turns those names into display strings; here they pick
   between a combo box (small enum), a raw int editor with a resolved name
   beside it (the big id datasets, which need a real picker), and a plain
   spinbox.

Enum classes are imported from the public AoE2ScenarioParser.datasets modules
rather than attr_presentation's private _datasets dict, so
tests/test_private_api_guard.py needs no new names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from AoE2ScenarioParser.datasets import buildings, players, techs, trigger_lists, units
from AoE2ScenarioParser.datasets.effects import EffectId
from AoE2ScenarioParser.objects.data_objects.effect import (
    _get_armour_attack_source,
    _is_float_quantity_effect,
)

from . import object_catalog
from .library_compat import VocabularyEntry

# -- kinds -------------------------------------------------------------------

INT = "int"
FLOAT = "float"
BOOL = "bool"
STR = "str"
ENUM = "enum"
INT_LIST = "int_list"
REFERENCE = "reference"
UNSUPPORTED = "unsupported"

# -- multi-line modes (FieldSpec.multiline) ----------------------------------

# An XS script body: monospace, un-wrapped, CR-separated on the way back out.
XS = "xs"
# User-facing prose: proportional, word-wrapped, and written back with
# whichever newline token the stored value already used.
PROSE = "prose"

# The library's "this field is not set" value across every numeric trigger
# field. A spinbox shows it as "(unset)" at its minimum rather than as -1.
UNSET = -1


class _Mixed:
    """MIXED's type, so a stray one reads as MIXED in a traceback."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MIXED"


# "The selected entries disagree on this field" (GH #60). Unlike UNSET and
# None, never a stored value, so it is never written back.
MIXED = _Mixed()

# Shown but never written. quantity_float left this set: it is editable exactly
# when it is the live cluster slot (apply_quantity_cluster_rule()).
DISPLAY_ONLY: frozenset[str] = frozenset()

# Fields whose library attribute differs from the vocabulary name. Effect has
# no public quantity_float; its value lives in Effect.quantity.
_ATTRIBUTE_ALIASES: Mapping[str, str] = {"quantity_float": "quantity"}

# Small enums that fit a combo box. Measured across every shipped
# versions/DE/v*/ JSON, not just the version this repo's fixture uses.
_ENUM_TYPES: Mapping[str, type[Enum]] = {
    "ActionType": trigger_lists.ActionType,
    "AttackStance": trigger_lists.AttackStance,
    "Attribute": trigger_lists.Attribute,
    "ButtonLocation": trigger_lists.ButtonLocation,
    "ColorMood": trigger_lists.ColorMood,
    "Comparison": trigger_lists.Comparison,
    "DamageClass": trigger_lists.DamageClass,
    "DecisionOption": trigger_lists.DecisionOption,
    "DiplomacyState": trigger_lists.DiplomacyState,
    "ObjectAttribute": trigger_lists.ObjectAttribute,
    "ObjectClass": trigger_lists.ObjectClass,
    "ObjectModifyAttributeState": trigger_lists.ObjectModifyAttributeState,
    "ObjectState": trigger_lists.ObjectState,
    "ObjectType": trigger_lists.ObjectType,
    "Operation": trigger_lists.Operation,
    "PanelLocation": trigger_lists.PanelLocation,
    "PlayerColorId": players.PlayerColorId,
    "PlayerId": players.PlayerId,
    "TimeUnit": trigger_lists.TimeUnit,
    "UnitAIAction": trigger_lists.UnitAIAction,
    "VictoryTimerType": trigger_lists.VictoryTimerType,
    "VisibilityState": trigger_lists.VisibilityState,
}

# Every presentation that stays a REFERENCE kind rather than becoming a plain
# int, combo, or unsupported field. All seven need a resolved name shown
# beside (or in place of) the raw id, but split three ways for 4d's picker
# (TriggerPanel._build_widget()): TriggerId/VariableId resolve against the
# open document (see DOCUMENT_REFERENCES and object_catalog.py's
# trigger_choices()/variable_choices()); UnitInfo/BuildingInfo/TechInfo get a
# real picker (see CATALOG_PRESENTATIONS); Unit/Unit[] are placed-unit
# reference_ids, not type constants: resolved by descape/unit_references.py
# against the open document and picked from the map, never from a dataset.
# None here just marks "no library Enum for from_id()", not "no picker".
_REFERENCE_DATASETS: Mapping[str, type[Enum] | None] = {
    "TriggerId": None,
    "VariableId": None,
    "Unit": None,
    "Unit[]": None,
    "UnitInfo": units.UnitInfo,
    "TechInfo": techs.TechInfo,
    "BuildingInfo": buildings.BuildingInfo,
}

# UnitInfo and BuildingInfo both legitimately hold ids from any of the four
# object datasets (object_list holds buildings/heroes/others too) -- the
# library treats them as one combined lookup (object_catalog.py), where this
# module previously mapped each presentation to a single dataset and silently
# returned "" for most real values. TechInfo has no such gap: it is the only
# dataset a "TechInfo" presentation ever draws from.
_COMBINED_OBJECT_PRESENTATIONS = frozenset({"UnitInfo", "BuildingInfo"})


@dataclass(frozen=True)
class CatalogPresentation:
    """Which object_catalog.py catalog a REFERENCE presentation picks from,
    and which category its CatalogBrowseDialog should open on."""

    catalog: Callable[[], tuple]
    default_category: str = ""


# Every presentation CatalogLineEdit picks from, keyed only on names already
# in _REFERENCE_DATASETS -- test_every_presentation_is_covered's no-unused-
# entries half would fail on a presentation this build never actually sees.
CATALOG_PRESENTATIONS: Mapping[str, CatalogPresentation] = {
    "UnitInfo": CatalogPresentation(object_catalog.objects, "Units"),
    "BuildingInfo": CatalogPresentation(object_catalog.objects, "Buildings"),
    "TechInfo": CatalogPresentation(object_catalog.techs),
}

# Reference presentations resolved against the open document's own trigger or
# variable list rather than any id dataset -- object_catalog.py's
# trigger_choices()/variable_choices()/trigger_name_for()/variable_name_for().
DOCUMENT_REFERENCES = frozenset({"TriggerId", "VariableId"})

# Presentation values that map to a widget without naming a dataset.
_SCALAR_PRESENTATIONS: Mapping[str, str] = {"": INT, "bool": BOOL, "str": STR}


@dataclass(frozen=True)
class FieldSpec:
    """One editable field of a condition, effect, or trigger.

    `choices` is (label, value) pairs for ENUM only. `sentinel` is the value
    meaning "unset", or None for kinds that have no such value. `attribute`
    is the library attribute read and written, when it differs from `name`
    ("" means the same). `multiline`
    is XS, PROSE or "" -- a mode rather than a bool, since the two multi-line
    kinds need different widgets (monospace and un-wrapped vs. proportional
    and wrapping) and different newline handling, while `if spec.multiline:`
    still reads as "this field is multi-line".
    """

    name: str
    kind: str
    choices: tuple[tuple[str, int], ...] = ()
    sentinel: int | None = UNSET
    presentation: str = ""
    read_only: bool = False
    multiline: str = ""
    attribute: str = ""

    def __post_init__(self) -> None:
        if not self.attribute:
            object.__setattr__(self, "attribute", self.name)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")


# The trigger's own fields, in the order the in-game editor shows them. Fixed
# rather than derived: TriggerStruct's JSON carries storage fields
# (description_stid, condition_order) that are not user-editable properties.
TRIGGER_FIELDS: tuple[FieldSpec, ...] = (
    # `name` stays one line: 1569 corpus values, median 18 chars, none with a
    # newline. The other two are real prose (description: 407 values, max 235,
    # 16 of them multi-line across all three newline tokens).
    FieldSpec("name", STR, sentinel=None),
    FieldSpec("short_description", STR, sentinel=None, multiline=PROSE),
    FieldSpec("description", STR, sentinel=None, multiline=PROSE),
    FieldSpec("enabled", BOOL, sentinel=None),
    FieldSpec("looping", BOOL, sentinel=None),
    FieldSpec("header", BOOL, sentinel=None),
    FieldSpec("display_as_objective", BOOL, sentinel=None),
    FieldSpec("display_on_screen", BOOL, sentinel=None),
    FieldSpec("mute_objectives", BOOL, sentinel=None),
)


def enum_choices(presentation: str) -> tuple[tuple[str, int], ...]:
    """(label, value) pairs for a combo-box presentation, ordered by value.

    Iterating the class yields canonical members only, so an alias cannot
    produce two rows with the same value.
    """
    enum_type = _ENUM_TYPES.get(presentation)
    if enum_type is None:
        return ()
    pairs = [(member.name.replace("_", " "), int(member.value)) for member in enum_type]
    return tuple(sorted(pairs, key=lambda pair: pair[1]))


# Above this many members a combo stops being a choice and becomes a scroll.
# 20 is the user's own line: ColorMood (19) is the largest that stays a combo,
# UnitAIAction (25) the smallest that becomes a picker.
PICKER_MIN_CHOICES = 20


def wants_picker(spec: FieldSpec) -> bool:
    """Whether an editable field gets the type-ahead ValueLineEdit picker
    instead of a QComboBox. Only ENUM fields qualify; the presentation stays
    in _ENUM_TYPES and `choices` stays populated either way."""
    return spec.kind == ENUM and len(spec.choices) >= PICKER_MIN_CHOICES


def resolve_reference(presentation: str, value: Any) -> str:
    """Display name for a reference field's raw id, or "" if it does not
    resolve. The id datasets carry tuple values, so this goes through their
    from_id() rather than looking the value up directly.

    UnitInfo/BuildingInfo go through object_catalog's merged lookup instead:
    object_list legitimately holds buildings, heroes and other objects too, so
    resolving through a single dataset left most real values blank.
    """
    if presentation not in _REFERENCE_DATASETS:
        return ""
    if not isinstance(value, int) or isinstance(value, bool):
        return ""
    if value == UNSET:
        return ""
    if presentation in _COMBINED_OBJECT_PRESENTATIONS:
        return object_catalog.combined_object_name(value)
    dataset = _REFERENCE_DATASETS.get(presentation)
    if dataset is None:
        return ""
    try:
        member = dataset.from_id(value)
    except (KeyError, ValueError, TypeError):
        # Measured against UnitInfo/TechInfo/BuildingInfo: an out-of-range id
        # raises KeyError, a negative one ValueError, a non-int TypeError.
        return ""
    return getattr(member, "name", "").replace("_", " ")


def _kind_for(attribute: str, default: Any, presentation: str) -> tuple[str, str]:
    """(kind, presentation) for one attribute of one vocabulary entry."""
    if isinstance(default, list):
        return (INT_LIST, presentation)
    scalar = _SCALAR_PRESENTATIONS.get(presentation)
    if scalar is not None:
        return (scalar, presentation)
    if presentation in _REFERENCE_DATASETS:
        return (REFERENCE, presentation)
    if presentation in _ENUM_TYPES:
        return (ENUM, presentation)
    # A presentation this build has never seen. Shown read-only rather than
    # guessed at, and pinned by test_every_presentation_is_covered so a library
    # bump surfaces here instead of in the UI.
    return (UNSUPPORTED, presentation)


# (vocabulary entry name, attribute) pairs rendered as a multi-line editor: the
# XS script bodies. Keyed on the entry's name, not its id, since names are what
# stay stable across the shipped version JSONs.
MULTILINE_FIELDS = frozenset({("script_call", "message"), ("script_call", "xs_function")})

# The same, for user-facing prose. Set from the corpus: these are the message
# fields that actually hold sentences (display_instructions alone has 510
# values, median 89 chars, max 256). Every *_name message, change_variable,
# modify_attribute and sound_name stays single-line -- each is an identifier
# under 30 characters, not prose. change_technology_description is absent from
# the corpus but is the same field class as change_object_description, and
# exists from v1.40 on.
PROSE_FIELDS = frozenset(
    {
        ("display_instructions", "message"),
        ("send_chat", "message"),
        ("display_timer", "message"),
        ("change_object_description", "message"),
        ("change_technology_description", "message"),
    }
)


def _multiline_mode(entry_name: str, attribute: str, kind: str) -> str:
    """XS, PROSE or "" for one (type, field) pair."""
    if kind != STR:
        return ""
    if (entry_name, attribute) in MULTILINE_FIELDS:
        return XS
    if (entry_name, attribute) in PROSE_FIELDS:
        return PROSE
    return ""


def field_specs(
    entry: VocabularyEntry,
    presentation_map: Mapping[str, str],
    type_attribute: str,
) -> tuple[FieldSpec, ...]:
    """Specs for every field of one condition or effect type, in the order the
    vocabulary lists them.

    `type_attribute` (condition_type / effect_type) is dropped: it is the
    entry's identity, not one of its fields, and changing it is the picker's
    job in 4b.6b rather than the property form's.
    """
    specs = []
    for attribute in entry.attributes:
        if attribute == type_attribute:
            continue
        default = entry.default_attributes.get(attribute)
        kind, presentation = _kind_for(attribute, default, presentation_map.get(attribute, ""))
        if attribute == "quantity_float":
            # By name: its version-JSON default is an int -1, so _kind_for()
            # cannot tell it from an INT.
            kind = FLOAT
        specs.append(
            FieldSpec(
                name=attribute,
                kind=kind,
                choices=enum_choices(presentation) if kind == ENUM else (),
                sentinel=None if kind in (STR, BOOL, INT_LIST) else UNSET,
                presentation=presentation,
                read_only=attribute in DISPLAY_ONLY or kind == UNSUPPORTED,
                multiline=_multiline_mode(entry.name, attribute, kind),
                attribute=_ATTRIBUTE_ALIASES.get(attribute, attribute),
            )
        )
    return tuple(specs)


# -- retyping an existing condition or effect (GH #37) -----------------------

# Four Effect fields whose stored meaning depends on effect_type: `quantity` is
# bit-split with the armour/attack pair, and the library re-derives which slot
# is authoritative from (effect_type, object_attributes). Carrying them across
# types is the same hazard entry_structural_edit()'s "copy" op avoids by
# deepcopying rather than copying fields across.
#
# Effect-only. 14 condition types list `quantity` as an ordinary field, where
# it is a plain number with no second slot behind it.
#
# `variable` is deliberately absent: it is a plain slot, and only the private
# _variable_ref serializer merges it with armour_attack_class, which is never
# carried, so the fresh entry's own default governs that merge.
_RETYPE_EXCLUDED = frozenset(
    {"quantity", "quantity_float", "armour_attack_quantity", "armour_attack_class"}
)


def _live_value(entry: Any, attribute: str):
    """getattr that tolerates the library's version-gated properties, which
    raise UnsupportedAttributeError rather than being absent. Same rule as
    TriggerPanel._read(): an unreadable field is simply not carried."""
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- see docstring
        return None


# -- editing several entries at once (GH #60) --------------------------------


def shared_specs(spec_lists: Sequence[Sequence[FieldSpec]]) -> tuple[FieldSpec, ...]:
    """The fields every entry in a selection genuinely shares, in the first
    entry's order.

    By FieldSpec equality, never by field name: `quantity` is INT on most
    effects and INT_LIST on the armour ones, `message` is PROSE on some types
    and STR on others, and apply_quantity_cluster_rule() locks different
    cluster fields on two effects of the same type. One widget over two
    fields that only share a name would write the wrong kind of value.
    """
    lists = [tuple(specs) for specs in spec_lists]
    if not lists:
        return ()
    others = [set(specs) for specs in lists[1:]]
    return tuple(spec for spec in lists[0] if all(spec in other for other in others))


def shared_value(
    entries: Sequence[Any],
    attribute: str,
    reader: Callable[[Any, str], Any] = _live_value,
):
    """The value every entry holds for `attribute`, or MIXED when they
    differ (or there are none). Compared with ==, so lists compare by value.
    `reader` is the caller's version-tolerant getattr."""
    if not entries:
        return MIXED
    first = reader(entries[0], attribute)
    for entry in entries[1:]:
        if reader(entry, attribute) != first:
            return MIXED
    return first


def retype_carryover(
    entry: Any,
    old_definition: VocabularyEntry | None,
    new_definition: VocabularyEntry,
    type_attribute: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """(values to re-apply, names that did not carry) for a retype.

    The caller builds a *fresh* entry of the new type through the library's own
    construction path and then applies the returned values, so this only has to
    answer "which of the old entry's set fields still mean the same thing".

    A candidate is skipped when its live value is None or equal to the old
    type's own default: there is nothing meaningful to carry, and carrying a
    default would clobber the new type's own non-(-1) default (change_object_
    attack defaults armour_attack_quantity to 1).

    `dropped` names every field the old type listed that held a real value and
    did not carry, excluded ones included, so the user is always told.

    An unknown old type (`old_definition is None`) carries nothing rather than
    raising: such an entry is already renderable, and the retype is exactly how
    a user gets out of it.
    """
    if old_definition is None:
        return ({}, ())

    excluded = _RETYPE_EXCLUDED if type_attribute == "effect_type" else frozenset()
    # Effect types 77/78 (create_object_attack/_armor) omit effect_type from
    # their own `attributes` list. Harmless and deliberately left alone: the
    # type id is read off the object, not off this list.
    shared = set(new_definition.attributes) & set(old_definition.attributes)
    applied: dict[str, Any] = {}
    dropped: list[str] = []
    for attribute in old_definition.attributes:
        if attribute == type_attribute:
            continue
        value = _live_value(entry, attribute)
        if value is None or value == old_definition.default_attributes.get(attribute):
            continue
        if attribute in shared and attribute not in excluded:
            # Copied, never aliased: the library mutates lists handed to it in
            # place, which is restore()'s trap 6.
            applied[attribute] = list(value) if isinstance(value, list) else value
        else:
            dropped.append(attribute)
    return (applied, tuple(dropped))


def retype_entry(
    trigger: Any,
    kind: str,
    entry_index: int,
    type_id: int,
    definitions: Mapping[int, VocabularyEntry],
    type_attribute: str,
) -> tuple[str, ...]:
    """Replace one condition or effect with a fresh entry of `type_id`,
    carrying over what retype_carryover() says still applies. Returns the
    names that did not carry.

    Build-then-replace, not a write to the type attribute:

    - `_add_condition`/`_add_effect` is the library's own construction path.
      It builds from the module-level default_attributes that
      _initialise_version_dependencies rewrites per load, so the result is
      right for this file's version, and an unsupported type raises
      UnsupportedAttributeError rather than writing junk.
    - It leaves no stale value in a slot the new type does not list. Every
      field is serialized regardless of type, so a stale one would reach the
      file.
    - It sidesteps Effect.effect_type's setter, which recomputes the
      armour/attack flag.

    Here rather than in the viewer so the corpus tier can run it against every
    shipped scenario version without a QApplication. Qt-free like the rest of
    this module; it only ever touches library objects.
    """
    entries = trigger.conditions if kind == "condition" else trigger.effects
    old = entries[entry_index]
    applied, dropped = retype_carryover(
        old,
        definitions.get(_live_value(old, type_attribute)),
        definitions[type_id],
        type_attribute,
    )
    if kind == "condition":
        trigger._add_condition(type_id)
    else:
        trigger._add_effect(type_id)
    # Append, pop, replace-at-index: the length never changes, so
    # condition_order/effect_order's lazy getter -- a no-op at equal length --
    # has nothing to rebuild and the user's display order survives. Nothing
    # reads that getter in between.
    # An explicit index, not a bare pop(): UuidList.pop's signature defaults
    # __index to `...` rather than -1, so the no-argument form raises.
    fresh = entries.pop(len(entries) - 1)
    # After construction, never before: object_attributes has a setter that
    # recomputes the armour/attack flag.
    before = live_quantity_slot(fresh) if kind == "effect" else ""
    for name, value in applied.items():
        setattr(fresh, name, value)
    if kind == "effect":
        # A carried object_attributes can flip the fresh entry's live slot onto
        # one its own construction never populated.
        listed = definitions[type_id].attributes
        for name, value in cluster_fixup(fresh, before, live_quantity_slot(fresh), listed):
            setattr(fresh, name, value)
    # Through UuidList.__setitem__, which re-anchors the fresh object's _uuid.
    entries[entry_index] = fresh
    return dropped


# -- the quantity cluster ----------------------------------------------------
#
# The library packs three logical fields into one file field and picks the
# packing at runtime from (effect_type, object_attributes). Whichever slot it
# will serialize must hold a number, and the others must be cleared.

ARMOUR_ATTACK = "armour_attack"
VARIABLE = "variable"
QUANTITY = "quantity"
QUANTITY_FLOAT = "quantity_float"

_ARMOUR_ATTACK_FIELDS = ("armour_attack_quantity", "armour_attack_class")
_CLUSTER_FIELDS = ("quantity", "quantity_float", *_ARMOUR_ATTACK_FIELDS)

# The vocabulary fields each live slot makes editable. `variable` is live in
# both of its states and is not a cluster field.
_LIVE_FIELDS: Mapping[str, frozenset[str]] = {
    ARMOUR_ATTACK: frozenset(_ARMOUR_ATTACK_FIELDS),
    VARIABLE: frozenset({"armour_attack_class"}),
    QUANTITY: frozenset({"quantity"}),
    QUANTITY_FLOAT: frozenset({"quantity_float"}),
}


def live_quantity_slot(entry: Any) -> str:
    """Which of the cluster's fields the library will actually serialize:
    ARMOUR_ATTACK, VARIABLE, QUANTITY_FLOAT or QUANTITY.

    Asks the library's own gates rather than sniffing values, since the values
    are exactly what is wrong after an object_attributes switch. Naming
    inversion: the library's 'quantity' source means the armour/attack pair is
    live and packs into the quantity slot.
    """
    effect_type = getattr(entry, "effect_type", None)
    object_attributes = getattr(entry, "object_attributes", None)
    source = _get_armour_attack_source(effect_type, object_attributes)
    if source == "quantity":
        return ARMOUR_ATTACK
    if source == "variable":
        return VARIABLE
    if _is_float_quantity_effect(effect_type, object_attributes):
        return QUANTITY_FLOAT
    return QUANTITY


def apply_quantity_cluster_rule(specs: Sequence[FieldSpec], entry: Any) -> tuple[FieldSpec, ...]:
    """Mark every cluster field read-only except the live slot's. Writing an
    inert one emits IncorrectArmorAttackUsageWarning or lands where
    serialization never reads it."""
    live = _LIVE_FIELDS[live_quantity_slot(entry)]
    return tuple(
        replace(spec, read_only=True)
        if spec.name in _CLUSTER_FIELDS and spec.name not in live
        else spec
        for spec in specs
    )


def _as_number(value: Any, slot: str) -> int | float:
    """The coercion rule, one place: an int slot takes int(), the float slot
    float(), and anything that is not a number becomes -1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = UNSET
    return float(value) if slot == QUANTITY_FLOAT else int(value)


def cluster_fixup(
    entry: Any, before: str, after: str, listed: Sequence[str]
) -> tuple[tuple[str, Any], ...]:
    """The ordered (attribute, value) writes that re-establish the cluster
    invariant after a live-slot change from `before` to `after`. Pure.

    Call after object_attributes (or the type) is already assigned. Carries the
    amount and drops the armour class; never the merged packed value. Reads raw
    fields, since on a broken pair the public quantity getter is what raises.
    Only fields in `listed` (the entry type's per-version attributes) are ever
    written. Entering writes come before clears: setting a newly live field is
    what must not warn, and a clear to None never does.
    """
    if before == after:
        return ()
    if before == ARMOUR_ATTACK:
        amount = getattr(entry, "_armour_attack_quantity", None)
    elif before in (QUANTITY, QUANTITY_FLOAT):
        amount = getattr(entry, "_quantity", None)
    else:
        amount = None

    writes: list[tuple[str, Any]] = []
    if after == ARMOUR_ATTACK:
        writes += [("armour_attack_quantity", _as_number(amount, after)), ("armour_attack_class", 0)]
    elif after == VARIABLE:
        writes.append(("armour_attack_class", 0))
    elif after in (QUANTITY, QUANTITY_FLOAT):
        writes.append(("quantity", _as_number(amount, after)))

    if before == ARMOUR_ATTACK and after != ARMOUR_ATTACK:
        writes.append(("armour_attack_quantity", None))
        if after != VARIABLE:
            writes.append(("armour_attack_class", None))
    elif before == VARIABLE and after not in (VARIABLE, ARMOUR_ATTACK):
        writes.append(("armour_attack_class", None))

    # Effect.quantity backs both quantity rows, so either listed name counts.
    names = set(listed)
    if names & {"quantity", "quantity_float"}:
        names.add("quantity")
    return tuple((name, value) for name, value in writes if name in names)


# The effect types whose live slot object_attributes can switch. Every other
# type's slot is fixed by its type, so construction always populates it.
_SLOT_SWITCHING_EFFECTS = frozenset(
    int(effect)
    for effect in (
        EffectId.MODIFY_ATTRIBUTE,
        EffectId.MODIFY_ATTRIBUTE_FOR_CLASS,
        EffectId.MODIFY_OBJECT_ATTRIBUTE,
        EffectId.MODIFY_ATTRIBUTE_BY_VARIABLE,
        EffectId.MODIFY_VARIABLE_BY_ATTRIBUTE,
        EffectId.MODIFY_OBJECT_ATTRIBUTE_BY_VARIABLE,
    )
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def cluster_incoherence(effect: Any) -> str:
    """Why this effect's live cluster slot cannot serialize, or "". Reads the
    raw fields: on a broken pair the public quantity getter itself raises."""
    if getattr(effect, "effect_type", None) not in _SLOT_SWITCHING_EFFECTS:
        return ""
    slot = live_quantity_slot(effect)
    if slot == ARMOUR_ATTACK:
        raw = {name: getattr(effect, f"_{name}", None) for name in _ARMOUR_ATTACK_FIELDS}
    elif slot == VARIABLE:
        raw = {"armour_attack_class": getattr(effect, "_armour_attack_class", None)}
    else:
        raw = {"quantity": getattr(effect, "_quantity", None)}
    for name, value in raw.items():
        # [] is the library's own marker for a field this version stores with
        # zero repeat; it serializes, so it is not refused here.
        if value == []:
            continue
        if slot == QUANTITY_FLOAT:
            ok = _is_int(value) or isinstance(value, float)
        else:
            ok = _is_int(value)
        if not ok:
            return f"its live {slot.replace('_', ' ')} slot holds {name}={value!r}"
    return ""


# -- int-list formatting -----------------------------------------------------


def format_int_list(value: Any) -> str:
    """A list field's editor text. Non-list values render empty rather than
    raising, since the same attribute is scalar on other types."""
    if not isinstance(value, (list, tuple)):
        return ""
    return ", ".join(str(item) for item in value)


def parse_int_list(text: str) -> list[int]:
    """Inverse of format_int_list(). Raises ValueError on anything that is not
    a comma or whitespace separated run of integers, so the caller can reject
    the edit rather than write a half-parsed list."""
    cleaned = text.replace(",", " ").split()
    return [int(item) for item in cleaned]


# -- XS script text ----------------------------------------------------------


def xs_to_display(value: Any) -> str:
    """An XS field's editor text. The game stores lines separated by a bare CR,
    which a text widget cannot hold, so every separator becomes LF."""
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def xs_from_display(text: str) -> str:
    """Inverse of xs_to_display() for the CR-only values the game writes. Not
    an involution on a stored CRLF or lone LF, which is why the panel only
    calls this on text the user actually changed."""
    return text.replace("\r\n", "\n").replace("\n", "\r")
