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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from AoE2ScenarioParser.datasets import buildings, players, techs, trigger_lists, units

from . import object_catalog
from .library_compat import VocabularyEntry

# -- kinds -------------------------------------------------------------------

INT = "int"
BOOL = "bool"
STR = "str"
ENUM = "enum"
INT_LIST = "int_list"
REFERENCE = "reference"
UNSUPPORTED = "unsupported"

# The library's "this field is not set" value across every numeric trigger
# field. A spinbox shows it as "(unset)" at its minimum rather than as -1.
UNSET = -1

# Derived and has no public setter, so it is shown but never written.
DISPLAY_ONLY = frozenset({"quantity_float"})

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
# reference_ids, not type constants, so a picker for them is map-selection
# integration, deferred to phase 3.5b -- they stay a plain spinbox.
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
    meaning "unset", or None for kinds that have no such value.
    """

    name: str
    kind: str
    choices: tuple[tuple[str, int], ...] = ()
    sentinel: int | None = UNSET
    presentation: str = ""
    read_only: bool = False

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")


# The trigger's own fields, in the order the in-game editor shows them. Fixed
# rather than derived: TriggerStruct's JSON carries storage fields
# (description_stid, condition_order) that are not user-editable properties.
TRIGGER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("name", STR, sentinel=None),
    FieldSpec("short_description", STR, sentinel=None),
    FieldSpec("description", STR, sentinel=None),
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
    except Exception:
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
        specs.append(
            FieldSpec(
                name=attribute,
                kind=kind,
                choices=enum_choices(presentation) if kind == ENUM else (),
                sentinel=None if kind in (STR, BOOL, INT_LIST) else UNSET,
                presentation=presentation,
                read_only=attribute in DISPLAY_ONLY or kind == UNSUPPORTED,
            )
        )
    return tuple(specs)


# -- the armour/attack live-value rule ---------------------------------------

_ARMOUR_ATTACK_FIELDS = ("armour_attack_quantity", "armour_attack_class")


def armour_attack_source(entry: Any) -> str:
    """Which slot the library treats as authoritative for this effect,
    "armour_attack" or "quantity".

    For effect types listing both, writing the wrong one emits
    IncorrectArmorAttackUsageWarning and can corrupt the other, and the library
    decides at runtime rather than by type. Its own `_armour_attack_source` is
    private, so this reads the parsed values instead: an int in either
    armour/attack field means that pair is the source, [] or None means
    `quantity` is.
    """
    for name in _ARMOUR_ATTACK_FIELDS:
        value = getattr(entry, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return "armour_attack"
    return "quantity"


def apply_armour_attack_rule(specs: Sequence[FieldSpec], entry: Any) -> tuple[FieldSpec, ...]:
    """Mark whichever of the quantity / armour-attack pair is not authoritative
    read-only. A no-op for every type that does not list both."""
    names = {spec.name for spec in specs}
    if "quantity" not in names or not names.intersection(_ARMOUR_ATTACK_FIELDS):
        return tuple(specs)
    inert = (
        _ARMOUR_ATTACK_FIELDS
        if armour_attack_source(entry) == "quantity"
        else ("quantity",)
    )
    return tuple(
        FieldSpec(
            name=spec.name,
            kind=spec.kind,
            choices=spec.choices,
            sentinel=spec.sentinel,
            presentation=spec.presentation,
            read_only=True,
        )
        if spec.name in inert
        else spec
        for spec in specs
    )


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
