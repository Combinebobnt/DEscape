"""Covers descape/trigger_fields.py, the Qt-free half of the 4b.6 property
editor. No QApplication and no scenario file: specs are derived from the
library's own per-version JSON, which is also what lets the panel populate
before anything is open.

The load-bearing test is test_every_presentation_is_covered. Every other test
here pins a rule against one hand-picked type; that one sweeps every shipped
versions/DE/v*/ JSON, so a library bump that adds a presentation name fails
here with a clear message instead of silently rendering an unsupported field
read-only in the UI.
"""

from __future__ import annotations

import json

import pytest

from descape import library_compat, trigger_fields
from descape.trigger_fields import FieldSpec


def _every_shipped_vocabulary():
    """(version, kind, raw json) for every condition/effect file the installed
    library ships. The corpus tier is not needed: these are library data files,
    not scenarios."""
    for version_dir in sorted(library_compat.VERSIONS_DIR.glob("v*")):
        for kind in ("conditions", "effects"):
            path = version_dir / f"{kind}.json"
            if path.is_file():
                yield (version_dir.name, kind, json.loads(path.read_text(encoding="utf-8")))


# -- the sweep ---------------------------------------------------------------


def test_every_presentation_is_covered() -> None:
    """No shipped version names a presentation this module cannot place.

    Deliberately not scoped to the fixture's own version: a field that only
    appears in v1.36 or v1.58 still has to render, and the whole point of
    reading the JSON rather than the library's module-level dicts is that all
    of them are available without a file open.
    """
    known = (
        set(trigger_fields._ENUM_TYPES)
        | set(trigger_fields._REFERENCE_DATASETS)
        | set(trigger_fields._SCALAR_PRESENTATIONS)
    )
    unknown = {}
    for version, kind, raw in _every_shipped_vocabulary():
        for field, presentation in raw.get("-1", {}).get("attribute_presentation", {}).items():
            if presentation not in known:
                unknown.setdefault(presentation, []).append(f"{version}/{kind}:{field}")
    assert unknown == {}, f"unhandled presentation names: {unknown}"


def test_no_shipped_type_derives_an_unsupported_field() -> None:
    """The above proves the table covers every *name*; this proves the
    derivation actually reaches a widget for every real (type, field) pair."""
    offenders = []
    for version, kind, raw in _every_shipped_vocabulary():
        presentation = raw.get("-1", {}).get("attribute_presentation", {})
        type_attribute = "condition_type" if kind == "conditions" else "effect_type"
        for key, value in raw.items():
            if key == "-1":
                continue
            entry = library_compat.VocabularyEntry(
                id=int(key),
                name=value["name"],
                attributes=tuple(value.get("attributes", ())),
                default_attributes=dict(value.get("default_attributes", {})),
            )
            for spec in trigger_fields.field_specs(entry, presentation, type_attribute):
                if spec.kind == trigger_fields.UNSUPPORTED:
                    offenders.append(f"{version}/{kind}/{value['name']}.{spec.name}")
    assert offenders == [], f"fields with no widget: {offenders}"


# -- kind derivation ---------------------------------------------------------


def _entry(attributes, defaults=None, name="test_type", entry_id=1):
    return library_compat.VocabularyEntry(
        id=entry_id,
        name=name,
        attributes=tuple(attributes),
        default_attributes=dict(defaults or {}),
    )


def test_the_type_attribute_is_not_a_field() -> None:
    """It is the entry's identity. Changing it is the 4b.6b picker's job, and
    a spinbox on it would let the user turn a Timer into a nonsense type."""
    entry = _entry(["condition_type", "timer"], {"condition_type": 1, "timer": -1})
    names = [spec.name for spec in trigger_fields.field_specs(entry, {}, "condition_type")]
    assert names == ["timer"]


def test_kind_comes_from_the_presentation_map() -> None:
    entry = _entry(
        ["timer", "enabled", "message", "comparison", "trigger_id"],
        {"timer": -1, "enabled": -1, "message": "", "comparison": -1, "trigger_id": -1},
    )
    presentation = {
        "timer": "",
        "enabled": "bool",
        "message": "str",
        "comparison": "Comparison",
        "trigger_id": "TriggerId",
    }
    kinds = {
        spec.name: spec.kind
        for spec in trigger_fields.field_specs(entry, presentation, "condition_type")
    }
    assert kinds == {
        "timer": trigger_fields.INT,
        "enabled": trigger_fields.BOOL,
        "message": trigger_fields.STR,
        "comparison": trigger_fields.ENUM,
        "trigger_id": trigger_fields.REFERENCE,
    }


def test_list_ness_is_per_type_not_per_field_name() -> None:
    """The trap this module's per-entry derivation exists for. `quantity` is
    scalar on most effects and list-valued on the armour/attack ones, so a
    global field-name table would render one of the two wrong.
    """
    scalar = _entry(["quantity"], {"quantity": -1})
    listed = _entry(["quantity"], {"quantity": []})
    assert trigger_fields.field_specs(scalar, {}, "effect_type")[0].kind == trigger_fields.INT
    assert trigger_fields.field_specs(listed, {}, "effect_type")[0].kind == trigger_fields.INT_LIST


def test_a_list_field_keeps_its_kind_over_its_presentation() -> None:
    """armour_attack_class presents as DamageClass but defaults to [] on the
    types that use it as a list. The list shape wins; a combo box cannot hold
    two values."""
    entry = _entry(["armour_attack_class"], {"armour_attack_class": []})
    spec = trigger_fields.field_specs(entry, {"armour_attack_class": "DamageClass"}, "effect_type")[0]
    assert spec.kind == trigger_fields.INT_LIST
    assert spec.presentation == "DamageClass"


def test_an_unknown_presentation_is_read_only_not_guessed() -> None:
    entry = _entry(["mystery"], {"mystery": -1})
    spec = trigger_fields.field_specs(entry, {"mystery": "SomethingNew"}, "effect_type")[0]
    assert spec.kind == trigger_fields.UNSUPPORTED
    assert spec.read_only is True


def test_quantity_float_is_display_only() -> None:
    """It is derived and has no public setter."""
    entry = _entry(["quantity_float"], {"quantity_float": -1})
    spec = trigger_fields.field_specs(entry, {"quantity_float": ""}, "effect_type")[0]
    assert spec.read_only is True


def test_sentinels_belong_only_to_numeric_kinds() -> None:
    entry = _entry(
        ["timer", "message", "enabled", "selected_object_ids"],
        {"timer": -1, "message": "", "enabled": -1, "selected_object_ids": []},
    )
    presentation = {"timer": "", "message": "str", "enabled": "bool", "selected_object_ids": "Unit[]"}
    sentinels = {
        spec.name: spec.sentinel
        for spec in trigger_fields.field_specs(entry, presentation, "effect_type")
    }
    assert sentinels == {
        "timer": trigger_fields.UNSET,
        "message": None,
        "enabled": None,
        "selected_object_ids": None,
    }


# -- enum choices and reference resolution -----------------------------------


def test_enum_choices_are_ordered_by_value() -> None:
    choices = trigger_fields.enum_choices("Comparison")
    assert choices[0] == ("EQUAL", 0)
    assert [value for _, value in choices] == sorted(value for _, value in choices)


def test_enum_choices_have_no_duplicate_values() -> None:
    """Iterating an Enum yields canonical members only, so an alias cannot
    produce two rows the combo box would show as the same thing."""
    for presentation in trigger_fields._ENUM_TYPES:
        values = [value for _, value in trigger_fields.enum_choices(presentation)]
        assert len(values) == len(set(values)), f"{presentation} has alias duplicates"


def test_enum_choices_is_empty_for_a_reference() -> None:
    """A reference must not silently become a 498-entry combo box."""
    assert trigger_fields.enum_choices("UnitInfo") == ()
    assert trigger_fields.enum_choices("TriggerId") == ()


def test_reference_resolves_through_from_id() -> None:
    """The id datasets carry tuple values, so a direct value lookup fails."""
    assert trigger_fields.resolve_reference("UnitInfo", 486) == "BROWN BEAR"
    assert trigger_fields.resolve_reference("TechInfo", 16) == "ANARCHY"


def test_reference_resolution_tolerates_the_unresolvable() -> None:
    """A raw id editor must never fail to render over an id no dataset knows."""
    assert trigger_fields.resolve_reference("UnitInfo", 999999) == ""
    assert trigger_fields.resolve_reference("UnitInfo", trigger_fields.UNSET) == ""
    assert trigger_fields.resolve_reference("TriggerId", 3) == ""
    assert trigger_fields.resolve_reference("UnitInfo", None) == ""


def test_unitinfo_resolves_a_building_the_old_single_dataset_lookup_missed() -> None:
    """Slice 0's confirmed defect: 109 is TOWN CENTER in BuildingInfo, absent
    from UnitInfo, and object_list legitimately holds buildings too."""
    assert trigger_fields.resolve_reference("UnitInfo", 109) == "TOWN CENTER"
    assert trigger_fields.resolve_reference("BuildingInfo", 109) == "TOWN CENTER"


def test_catalog_presentations_cover_exactly_the_three_id_datasets() -> None:
    assert set(trigger_fields.CATALOG_PRESENTATIONS) == {"UnitInfo", "BuildingInfo", "TechInfo"}
    for presentation in trigger_fields.CATALOG_PRESENTATIONS:
        assert presentation in trigger_fields._REFERENCE_DATASETS, (
            f"{presentation} must stay a REFERENCE-kind presentation or "
            "test_every_presentation_is_covered's no-unused-entries half breaks"
        )


def test_catalog_presentations_point_at_a_non_empty_catalog() -> None:
    for presentation, mapped in trigger_fields.CATALOG_PRESENTATIONS.items():
        catalog = mapped.catalog()
        assert catalog, f"{presentation}'s catalog must not be empty"


def test_document_references_are_the_two_intra_document_presentations() -> None:
    assert trigger_fields.DOCUMENT_REFERENCES == {"TriggerId", "VariableId"}
    for presentation in trigger_fields.DOCUMENT_REFERENCES:
        assert presentation in trigger_fields._REFERENCE_DATASETS
        assert presentation not in trigger_fields.CATALOG_PRESENTATIONS


def test_unit_presentations_are_neither_catalog_nor_document_references() -> None:
    """Unit/Unit[] are placed-unit reference_ids, deferred to phase 3.5b, so
    TriggerPanel._build_widget() must keep falling through to the plain
    spinbox for them -- checked here by confirming neither dispatch table
    claims them, which is what that fallthrough depends on."""
    for presentation in ("Unit", "Unit[]"):
        assert presentation in trigger_fields._REFERENCE_DATASETS
        assert presentation not in trigger_fields.CATALOG_PRESENTATIONS
        assert presentation not in trigger_fields.DOCUMENT_REFERENCES


# -- the armour/attack rule --------------------------------------------------


class _FakeEffect:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


_AA_SPECS = (
    FieldSpec("quantity", trigger_fields.INT),
    FieldSpec("armour_attack_quantity", trigger_fields.INT),
    FieldSpec("armour_attack_class", trigger_fields.INT),
)


def test_armour_attack_source_reads_the_parsed_values() -> None:
    aa = _FakeEffect(armour_attack_quantity=3, armour_attack_class=4, quantity=[])
    plain = _FakeEffect(armour_attack_quantity=[], armour_attack_class=[], quantity=7)
    assert trigger_fields.armour_attack_source(aa) == "armour_attack"
    assert trigger_fields.armour_attack_source(plain) == "quantity"


def test_armour_attack_rule_locks_the_inert_side() -> None:
    """Writing the non-authoritative slot emits IncorrectArmorAttackUsageWarning
    and can corrupt the other, so the UI must not offer it."""
    aa = _FakeEffect(armour_attack_quantity=3, armour_attack_class=4, quantity=[])
    locked = {
        spec.name: spec.read_only for spec in trigger_fields.apply_armour_attack_rule(_AA_SPECS, aa)
    }
    assert locked == {"quantity": True, "armour_attack_quantity": False, "armour_attack_class": False}

    plain = _FakeEffect(armour_attack_quantity=[], armour_attack_class=[], quantity=7)
    locked = {
        spec.name: spec.read_only
        for spec in trigger_fields.apply_armour_attack_rule(_AA_SPECS, plain)
    }
    assert locked == {"quantity": False, "armour_attack_quantity": True, "armour_attack_class": True}


def test_armour_attack_rule_is_a_no_op_without_both_sides() -> None:
    specs = (FieldSpec("quantity", trigger_fields.INT), FieldSpec("timer", trigger_fields.INT))
    effect = _FakeEffect(quantity=5)
    assert trigger_fields.apply_armour_attack_rule(specs, effect) == specs


# -- int lists ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("", []), ("4", [4]), ("4, 5, 6", [4, 5, 6]), ("4 5 6", [4, 5, 6]), (" 4 ,5 ", [4, 5])],
)
def test_parse_int_list_accepts_commas_and_whitespace(text, expected) -> None:
    assert trigger_fields.parse_int_list(text) == expected


@pytest.mark.parametrize("text", ["4, x", "4.5", "--"])
def test_parse_int_list_rejects_a_half_parsed_list(text) -> None:
    """Rejecting the whole edit is the point: writing [4] for "4, x" would
    silently drop the user's second value."""
    with pytest.raises(ValueError):
        trigger_fields.parse_int_list(text)


def test_int_list_round_trips_through_the_editor_text() -> None:
    assert trigger_fields.parse_int_list(trigger_fields.format_int_list([4, 5, 6])) == [4, 5, 6]


def test_format_int_list_tolerates_a_scalar() -> None:
    """The same attribute is scalar on other types, so this is reachable
    whenever a spec and a live entry disagree."""
    assert trigger_fields.format_int_list(-1) == ""
    assert trigger_fields.format_int_list(None) == ""
