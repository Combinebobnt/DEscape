"""descape/trigger_requirements.json, the required-field table behind the
trigger status colours (GH #166): its integrity rules, checked against every
shipped vocabulary. Default tier; reads only the committed table and the
library's version JSONs.

A wrong row shows a false red, so each rule here closes one way a token
could never read as set (a BOOL, a default other than unset, an enum with a
-1 member) or could silently never be read (an unknown name)."""

from __future__ import annotations

import json
from enum import Enum

import pytest
from AoE2ScenarioParser.datasets import trigger_lists
from AoE2ScenarioParser.datasets.conditions import ConditionId
from AoE2ScenarioParser.datasets.effects import EffectId

from descape import library_compat, trigger_fields, trigger_geometry, trigger_status
from descape.trigger_status import AREA, LOCATION, PSEUDO_FIELDS

VERSIONS = library_compat.vocabulary_versions()
KINDS = (
    ("conditions", trigger_status.CONDITION, "condition_type"),
    ("effects", trigger_status.EFFECT, "effect_type"),
)
_PSEUDO_COMPONENTS = {
    AREA: trigger_geometry.AREA_FIELDS,
    LOCATION: (*trigger_geometry.LOCATION_FIELDS, trigger_geometry.LOCATION_OBJECT_FIELD),
}
_ROW_KEYS = frozenset({"all", "any", "why"})


def _raw_table() -> dict:
    def no_duplicates(pairs):
        keys = [key for key, _value in pairs]
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        assert not duplicated, f"duplicate keys in trigger_requirements.json: {duplicated}"
        return dict(pairs)

    return json.loads(trigger_status._TABLE_JSON_PATH.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


TABLE = trigger_status.load_table()


def _tokens(row: trigger_status.Requirement) -> list[str]:
    return [*row.all, *(token for group in row.any for token in group)]


def _definitions():
    """(version, kind key, kind, raw entry JSON, VocabularyEntry, presentation map)
    for every type of every shipped vocabulary."""
    for version in VERSIONS:
        raw = library_compat.vocabulary_json(version)
        vocabulary = library_compat.load_vocabulary(version)
        for kind_key, kind, _type_attribute in KINDS:
            entries = vocabulary.conditions if kind == trigger_status.CONDITION else vocabulary.effects
            shared = (
                vocabulary.condition_presentation if kind == trigger_status.CONDITION else vocabulary.effect_presentation
            )
            for type_id, definition in entries.items():
                yield version, kind_key, kind, raw[kind_key][str(type_id)], definition, vocabulary, shared


def _required_fields(token: str, definition) -> list[str]:
    """The listed vocabulary fields a token reads in one version."""
    if token in PSEUDO_FIELDS:
        return [field for field in _PSEUDO_COMPONENTS[token] if field in definition.attributes]
    return [token] if token in definition.attributes else []


def _presentation(raw_entry: dict, shared, field: str) -> str:
    """The field's presentation, per-entry override first. _parse_vocabulary_json
    drops those overrides, so they are read from the raw JSON here."""
    return raw_entry.get("attribute_presentation", {}).get(field, shared.get(field, ""))


def _enum_values(presentation: str) -> set[int]:
    """Every member value of the enum a presentation names, or empty."""
    values = {value for _label, value in trigger_fields.enum_choices(presentation)}
    library_enum = getattr(trigger_lists, presentation, None) if presentation else None
    if isinstance(library_enum, type) and issubclass(library_enum, Enum):
        values |= {int(member.value) for member in library_enum}
    return values


def test_the_table_has_exactly_one_row_per_vocabulary_name() -> None:
    raw = _raw_table()
    assert set(raw) == {"conditions", "effects"}
    for kind_key, _kind, _type_attribute in KINDS:
        names = {definition.name for _v, key, _k, _r, definition, _voc, _s in _definitions() if key == kind_key}
        assert len(names) == {"conditions": 41, "effects": 103}[kind_key]
        assert set(raw[kind_key]) == names, kind_key


def test_rows_hold_only_all_any_and_why() -> None:
    for kind_key, rows in _raw_table().items():
        for name, row in rows.items():
            assert set(row) <= _ROW_KEYS, (kind_key, name)
            assert all(isinstance(token, str) for token in row.get("all", ())), (kind_key, name)
            for group in row.get("any", ()):
                assert group and all(isinstance(token, str) for token in group), (kind_key, name)


def test_every_token_is_a_pseudo_field_or_a_listed_field_in_some_version() -> None:
    listed: dict[tuple[str, str], set[str]] = {}
    for _version, kind_key, kind, _raw, definition, vocabulary, _shared in _definitions():
        specs = trigger_status.field_specs_for(kind, definition, vocabulary)
        listed.setdefault((kind_key, definition.name), set()).update(spec.name for spec in specs)
    for kind_key, rows in TABLE.items():
        for name, row in rows.items():
            for token in _tokens(row):
                assert token in PSEUDO_FIELDS or token in listed[(kind_key, name)], (kind_key, name, token)


def test_no_bool_or_unsupported_field_is_required() -> None:
    for version, kind_key, kind, _raw, definition, vocabulary, _shared in _definitions():
        row = TABLE[kind_key][definition.name]
        specs = {spec.name: spec for spec in trigger_status.field_specs_for(kind, definition, vocabulary)}
        for token in _tokens(row):
            for field in _required_fields(token, definition):
                kind_of = specs[field].kind
                assert kind_of not in (trigger_fields.BOOL, trigger_fields.UNSUPPORTED), (
                    version, kind_key, definition.name, field, kind_of
                )


def test_every_required_field_defaults_to_unset() -> None:
    """A default that reads as set (source_player's 1, object_state's 2) would
    mean a fresh entry could never be flagged. -1 for numbers, "" for text and
    [] for lists all read unset; a missing default reads None, also unset."""
    checked = 0
    for version, kind_key, kind, _raw, definition, vocabulary, _shared in _definitions():
        row = TABLE[kind_key][definition.name]
        specs = {spec.name: spec for spec in trigger_status.field_specs_for(kind, definition, vocabulary)}
        for token in _tokens(row):
            for field in _required_fields(token, definition):
                default = definition.default_attributes.get(field)
                assert not trigger_status.is_set(specs[field], default), (
                    version, kind_key, definition.name, field, default
                )
                checked += 1
    assert checked > 0


def test_the_enum_resolver_sees_per_entry_overrides() -> None:
    """The guard below is only as good as this: difficulty_level's quantity is
    a plain INT in the shared map and EXTREME (-1) only in its own override."""
    raw = library_compat.vocabulary_json("1.59")["conditions"][str(int(ConditionId.DIFFICULTY_LEVEL))]
    vocabulary = library_compat.load_vocabulary("1.59")
    presentation = _presentation(raw, vocabulary.condition_presentation, "quantity")
    assert presentation == "DifficultyLevel"
    assert -1 in _enum_values(presentation)


def test_no_required_field_has_an_enum_with_a_minus_one_member() -> None:
    for version, kind_key, _kind, raw_entry, definition, _vocabulary, shared in _definitions():
        row = TABLE[kind_key][definition.name]
        for token in _tokens(row):
            for field in _required_fields(token, definition):
                presentation = _presentation(raw_entry, shared, field)
                assert -1 not in _enum_values(presentation), (version, kind_key, definition.name, field, presentation)


def test_no_cluster_field_is_required_on_a_slot_switching_effect() -> None:
    """cluster_incoherence() owns those fields: which one is live depends on
    object_attributes, so a table token would read a dead slot."""
    cluster = {*trigger_fields._CLUSTER_FIELDS}
    names = {
        definition.name
        for _v, kind_key, _k, _r, definition, _voc, _s in _definitions()
        if kind_key == "effects" and definition.id in trigger_fields._SLOT_SWITCHING_EFFECTS
    }
    assert len(names) == len(trigger_fields._SLOT_SWITCHING_EFFECTS)
    for name in names:
        assert not cluster & set(_tokens(TABLE["effects"][name])), name


def test_every_row_with_a_requirement_says_why() -> None:
    for kind_key, rows in TABLE.items():
        for name, row in rows.items():
            if row.all or row.any:
                assert row.why.strip(), (kind_key, name)


@pytest.mark.parametrize(
    ("kind_key", "type_id"),
    [
        ("effects", int(EffectId.ACTIVATE_TRIGGER)),
        ("effects", int(EffectId.DEACTIVATE_TRIGGER)),
        ("conditions", int(ConditionId.TRIGGER_ACTIVE)),
    ],
)
def test_the_trigger_referencing_types_require_trigger_id(kind_key, type_id) -> None:
    """remove_triggers() resets a reference to a deleted trigger to -1, which
    only this row catches; the universal check sees ids that do not exist."""
    names = {
        definition.name
        for _v, key, _k, _r, definition, _voc, _s in _definitions()
        if key == kind_key and definition.id == type_id
    }
    assert len(names) == 1
    assert "trigger_id" in TABLE[kind_key][names.pop()].all
