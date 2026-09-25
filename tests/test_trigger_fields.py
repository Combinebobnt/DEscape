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

import pytest

from descape import library_compat, trigger_fields
from descape.trigger_fields import FieldSpec


def _every_shipped_vocabulary():
    """(version, kind, raw json) for every condition/effect file the installed
    library or this repo ships, as the UI sees it (vocabulary_json() drops
    repo-only attributes). The corpus tier is not needed: these are data
    files, not scenarios."""
    for version in library_compat.vocabulary_versions():
        for kind, raw in library_compat.vocabulary_json(version).items():
            yield (f"v{version}", kind, raw)


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
            offenders.extend(
                f"{version}/{kind}/{value['name']}.{spec.name}"
                for spec in trigger_fields.field_specs(entry, presentation, type_attribute)
                if spec.kind == trigger_fields.UNSUPPORTED
            )
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


def test_quantity_float_is_a_float_field_backed_by_quantity() -> None:
    """Effect has no public quantity_float: a setattr on that name lands on a
    dead instance attribute serialization ignores. Its read-only flag comes
    from the cluster rule, not from DISPLAY_ONLY."""
    entry = _entry(["quantity_float"], {"quantity_float": -1})
    spec = trigger_fields.field_specs(entry, {"quantity_float": ""}, "effect_type")[0]
    assert spec.kind == trigger_fields.FLOAT
    assert spec.attribute == "quantity"
    assert spec.read_only is False


def test_a_spec_s_attribute_defaults_to_its_name() -> None:
    assert FieldSpec("timer", trigger_fields.INT).attribute == "timer"


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


def test_wants_picker_selects_exactly_the_six_large_enums() -> None:
    """Pinned by name so a library enum growing past the threshold shows up
    here as a deliberate change, not a silent widget swap."""
    qualifying = {
        presentation
        for presentation in trigger_fields._ENUM_TYPES
        if trigger_fields.wants_picker(
            FieldSpec("f", trigger_fields.ENUM, trigger_fields.enum_choices(presentation), presentation=presentation)
        )
    }
    assert qualifying == {
        "Attribute",
        "ObjectAttribute",
        "ObjectClass",
        "DamageClass",
        "ActionType",
        "UnitAIAction",
    }
    assert len(trigger_fields.enum_choices("ColorMood")) == trigger_fields.PICKER_MIN_CHOICES - 1


def test_wants_picker_is_enum_only() -> None:
    many = tuple((f"choice {i}", i) for i in range(trigger_fields.PICKER_MIN_CHOICES * 2))
    assert trigger_fields.wants_picker(FieldSpec("f", trigger_fields.ENUM, many))
    assert not trigger_fields.wants_picker(FieldSpec("f", trigger_fields.INT, many))
    assert not trigger_fields.wants_picker(FieldSpec("f", trigger_fields.REFERENCE, presentation="UnitInfo"))


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
    assert {"TriggerId", "VariableId"} == trigger_fields.DOCUMENT_REFERENCES
    for presentation in trigger_fields.DOCUMENT_REFERENCES:
        assert presentation in trigger_fields._REFERENCE_DATASETS
        assert presentation not in trigger_fields.CATALOG_PRESENTATIONS


def test_unit_presentations_are_neither_catalog_nor_document_references() -> None:
    """Unit/Unit[] are placed-unit reference_ids, resolved against the open
    document and picked from the map, so TriggerPanel._build_widget() must
    keep falling through to its spinbox (Unit) and line edit (Unit[]) for
    them -- checked here by confirming neither dispatch table claims them,
    which is what that fallthrough depends on."""
    for presentation in ("Unit", "Unit[]"):
        assert presentation in trigger_fields._REFERENCE_DATASETS
        assert presentation not in trigger_fields.CATALOG_PRESENTATIONS
        assert presentation not in trigger_fields.DOCUMENT_REFERENCES


# -- the quantity cluster ----------------------------------------------------

_ARMOR, _ATTACK, _HIT_POINTS = 8, 9, 0
_MODIFY_ATTRIBUTE, _MODIFY_ATTRIBUTE_BY_VARIABLE, _DISPLAY_INSTRUCTIONS = 51, 79, 20


class _FakeEffect:
    """Raw fields set under their private names too, as the library stores
    them, since cluster_fixup() and cluster_incoherence() read those."""

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)
            if key in ("quantity", "armour_attack_quantity", "armour_attack_class"):
                setattr(self, f"_{key}", value)


_CLUSTER_SPECS = (
    FieldSpec("quantity", trigger_fields.INT),
    FieldSpec("quantity_float", trigger_fields.FLOAT, attribute="quantity"),
    FieldSpec("armour_attack_quantity", trigger_fields.INT),
    FieldSpec("armour_attack_class", trigger_fields.INT),
)
_LISTED = ("effect_type", "object_attributes", "quantity", "quantity_float",
           "armour_attack_quantity", "armour_attack_class")


def _loaded_effects():
    """The fixture's armour-split trigger. Loading also initialises the
    library's module-global vocabulary, which its float gate reads."""
    from pathlib import Path

    from descape.scenario_io import load_map_and_units, parse_triggers

    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    return parse_triggers(load_map_and_units(fixture)).triggers[1].effects


def test_live_quantity_slot_asks_the_library_gate_not_the_values() -> None:
    """The old value-sniffing rule answered wrong in exactly the corrupted
    state an attribute switch creates: pair still set, object_attributes gone."""
    aa = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_ARMOR)
    plain = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS,
                        armour_attack_quantity=2, armour_attack_class=3)
    var = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE_BY_VARIABLE, object_attributes=_ATTACK)
    assert trigger_fields.live_quantity_slot(aa) == trigger_fields.ARMOUR_ATTACK
    assert trigger_fields.live_quantity_slot(plain) == trigger_fields.QUANTITY
    assert trigger_fields.live_quantity_slot(var) == trigger_fields.VARIABLE


def test_live_quantity_slot_sees_a_float_attribute() -> None:
    effect = _loaded_effects()[1]
    effect.object_attributes = 13  # WORK_RATE
    assert trigger_fields.live_quantity_slot(effect) == trigger_fields.QUANTITY_FLOAT


def _locked(effect) -> dict:
    return {spec.name: spec.read_only for spec in trigger_fields.apply_quantity_cluster_rule(_CLUSTER_SPECS, effect)}


def test_the_cluster_rule_leaves_only_the_live_slot_editable() -> None:
    aa = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_ARMOR)
    assert _locked(aa) == {"quantity": True, "quantity_float": True,
                           "armour_attack_quantity": False, "armour_attack_class": False}
    plain = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS)
    assert _locked(plain) == {"quantity": False, "quantity_float": True,
                              "armour_attack_quantity": True, "armour_attack_class": True}
    var = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE_BY_VARIABLE, object_attributes=_ARMOR)
    assert _locked(var) == {"quantity": True, "quantity_float": True,
                            "armour_attack_quantity": True, "armour_attack_class": False}


def test_the_cluster_rule_unlocks_quantity_float_on_a_float_attribute() -> None:
    effect = _loaded_effects()[1]
    effect.object_attributes = 13
    assert _locked(effect) == {"quantity": True, "quantity_float": False,
                               "armour_attack_quantity": True, "armour_attack_class": True}


def test_the_cluster_rule_leaves_other_fields_alone() -> None:
    specs = (FieldSpec("quantity", trigger_fields.INT), FieldSpec("timer", trigger_fields.INT))
    effect = _FakeEffect(effect_type=_DISPLAY_INSTRUCTIONS, object_attributes=-1)
    assert trigger_fields.apply_quantity_cluster_rule(specs, effect) == specs


_AA, _Q, _QF, _VAR = (trigger_fields.ARMOUR_ATTACK, trigger_fields.QUANTITY,
                      trigger_fields.QUANTITY_FLOAT, trigger_fields.VARIABLE)


@pytest.mark.parametrize(
    "raw,before,after,expected",
    [
        # Same slot, including ARMOR -> ATTACK: nothing to re-establish.
        ({"armour_attack_quantity": 2, "armour_attack_class": 3}, _AA, _AA, ()),
        # Leaving the pair carries the amount, never the packed 196610.
        ({"armour_attack_quantity": 2, "armour_attack_class": 3}, _AA, _Q,
         (("quantity", 2), ("armour_attack_quantity", None), ("armour_attack_class", None))),
        ({"armour_attack_quantity": 2, "armour_attack_class": 3}, _AA, _QF,
         (("quantity", 2.0), ("armour_attack_quantity", None), ("armour_attack_class", None))),
        # Entering it carries the amount and drops the class to 0.
        ({"quantity": 45}, _Q, _AA, (("armour_attack_quantity", 45), ("armour_attack_class", 0))),
        ({"quantity": 2.5}, _QF, _AA, (("armour_attack_quantity", 2), ("armour_attack_class", 0))),
        # Both quantity slots are Effect.quantity, but the kind must follow.
        ({"quantity": 2.5}, _QF, _Q, (("quantity", 2),)),
        ({"quantity": 7}, _Q, _QF, (("quantity", 7.0),)),
        # A non-number carries as the unset sentinel.
        ({"quantity": None}, _Q, _AA, (("armour_attack_quantity", -1), ("armour_attack_class", 0))),
        ({"armour_attack_quantity": []}, _AA, _Q,
         (("quantity", -1), ("armour_attack_quantity", None), ("armour_attack_class", None))),
        # The variable slot: nothing to carry, variable itself untouched.
        ({"quantity": 5}, _Q, _VAR, (("armour_attack_class", 0),)),
        ({"armour_attack_class": 4}, _VAR, _Q, (("quantity", -1), ("armour_attack_class", None))),
    ],
)
def test_cluster_fixup_re_establishes_the_invariant(raw, before, after, expected) -> None:
    effect = _FakeEffect(**raw)
    assert trigger_fields.cluster_fixup(effect, before, after, _LISTED) == expected


def test_cluster_fixup_writes_only_listed_fields() -> None:
    """The *_BY_VARIABLE types list armour_attack_class and variable only."""
    effect = _FakeEffect(armour_attack_class=4)
    listed = ("effect_type", "object_attributes", "armour_attack_class", "variable")
    assert trigger_fields.cluster_fixup(effect, _VAR, _Q, listed) == (("armour_attack_class", None),)


def test_cluster_incoherence_names_an_empty_live_slot() -> None:
    broken = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_ARMOR,
                         armour_attack_quantity=None, armour_attack_class=None, quantity=45)
    assert "armour_attack_quantity=None" in trigger_fields.cluster_incoherence(broken)
    fine = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS, quantity=45)
    assert trigger_fields.cluster_incoherence(fine) == ""
    floaty = _FakeEffect(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS, quantity=2.5)
    assert "quantity=2.5" in trigger_fields.cluster_incoherence(floaty)
    fixed_slot = _FakeEffect(effect_type=_DISPLAY_INSTRUCTIONS, object_attributes=-1, quantity=None)
    assert trigger_fields.cluster_incoherence(fixed_slot) == ""


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


# -- XS script fields --------------------------------------------------------


def test_the_multiline_mode_is_exactly_the_named_pairs_in_every_version() -> None:
    """Swept over every shipped version, not the fixture's: XS only exists
    from v1.40, and a field-count census scoped to one version has been wrong
    here before.

    Asserts the *mode*, not a bool: the XS pairs and the prose pairs are both
    multi-line but want different widgets, so a pair landing in the wrong set
    would pass a truthiness check and still render wrong.
    """
    found = set()
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
                pair = (entry.name, spec.name)
                if pair in trigger_fields.MULTILINE_FIELDS:
                    expected = trigger_fields.XS
                elif pair in trigger_fields.PROSE_FIELDS:
                    expected = trigger_fields.PROSE
                else:
                    expected = ""
                if expected:
                    found.add(pair)
                    if spec.kind != trigger_fields.STR:
                        offenders.append(f"{version}/{kind}/{entry.name}.{spec.name} is not STR")
                if spec.multiline != expected:
                    offenders.append(
                        f"{version}/{kind}/{entry.name}.{spec.name}: "
                        f"{spec.multiline!r} != {expected!r}"
                    )
    assert offenders == [], f"wrong multiline mode: {offenders}"
    # Every named pair reaches a real spec in at least one shipped version --
    # a misspelled name in either frozenset would otherwise match nothing and
    # fail silently. Not every version: script_call arrived in v1.40, and so
    # did change_technology_description.
    assert found == trigger_fields.MULTILINE_FIELDS | trigger_fields.PROSE_FIELDS


def test_the_trigger_s_own_prose_fields_are_prose_and_its_name_is_not() -> None:
    modes = {spec.name: spec.multiline for spec in trigger_fields.TRIGGER_FIELDS}
    assert modes["description"] == trigger_fields.PROSE
    assert modes["short_description"] == trigger_fields.PROSE
    assert modes["name"] == ""


def test_the_cluster_rule_keeps_the_multiline_mode() -> None:
    specs = (
        FieldSpec("quantity", trigger_fields.INT),
        FieldSpec("armour_attack_quantity", trigger_fields.INT),
        FieldSpec("message", trigger_fields.STR, sentinel=None, multiline=trigger_fields.XS),
    )

    class _Entry:
        effect_type = _MODIFY_ATTRIBUTE
        object_attributes = _HIT_POINTS

    ruled = trigger_fields.apply_quantity_cluster_rule(specs, _Entry())
    assert [spec.read_only for spec in ruled] == [False, True, False]
    assert ruled[2].multiline == trigger_fields.XS


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "void r(){\rif(xsCreateFile(false)){\rxsWriteString(\"hello world\");\rxsCloseFile();\r}\r}",
        "void f()\r{\r  // a line comment\r  int a = 0;\r}\r",
        "\r\r",
    ],
)
def test_a_cr_separated_script_round_trips_byte_exactly(stored) -> None:
    display = trigger_fields.xs_to_display(stored)
    assert "\r" not in display
    assert display.count("\n") == stored.count("\r")
    assert trigger_fields.xs_from_display(display) == stored


def test_crlf_and_lf_display_as_lines_and_write_back_as_cr() -> None:
    """Not an involution: a stored CRLF or LF comes back as CR. The panel's
    latch, not this translation, is what keeps an untouched field byte-identical."""
    assert trigger_fields.xs_to_display("a\r\nb") == "a\nb"
    assert trigger_fields.xs_to_display("a\nb") == "a\nb"
    assert trigger_fields.xs_from_display("a\r\nb") == "a\rb"
    assert trigger_fields.xs_from_display("a\nb") == "a\rb"


def test_xs_to_display_tolerates_none() -> None:
    assert trigger_fields.xs_to_display(None) == ""


# -- the retype carry-over rule (GH #37) -------------------------------------


class _Live:
    """A stand-in for a live Condition/Effect: plain instance attributes, which
    is exactly what the library gives these (see the module docstring's point
    2)."""

    def __init__(self, **values):
        self.__dict__.update(values)


def test_a_shared_field_with_a_real_value_carries() -> None:
    old = _entry(["condition_type", "amount_or_quantity", "source_player"],
                 {"condition_type": 1, "amount_or_quantity": -1, "source_player": -1})
    new = _entry(["condition_type", "amount_or_quantity"],
                 {"condition_type": 2, "amount_or_quantity": -1}, entry_id=2)
    entry = _Live(condition_type=1, amount_or_quantity=7, source_player=-1)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "condition_type")
    assert applied == {"amount_or_quantity": 7}
    assert dropped == ()


def test_a_field_only_the_old_type_has_is_dropped_and_named() -> None:
    old = _entry(["condition_type", "timer", "source_player"],
                 {"condition_type": 1, "timer": -1, "source_player": -1})
    new = _entry(["condition_type", "source_player"],
                 {"condition_type": 2, "source_player": -1}, entry_id=2)
    entry = _Live(condition_type=1, timer=30, source_player=3)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "condition_type")
    assert applied == {"source_player": 3}
    assert dropped == ("timer",)


def test_a_value_equal_to_the_old_default_does_not_carry() -> None:
    """Nothing meaningful to carry, and carrying it would clobber a new type's
    own non-(-1) default -- change_object_attack defaults
    armour_attack_quantity to 1, not -1."""
    old = _entry(["effect_type", "quantity_stub"], {"effect_type": 1, "quantity_stub": -1})
    new = _entry(["effect_type", "quantity_stub"], {"effect_type": 2, "quantity_stub": 1},
                 entry_id=2)
    entry = _Live(effect_type=1, quantity_stub=-1)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "effect_type")
    assert applied == {}
    assert dropped == ()


def test_the_excluded_cluster_never_carries_on_an_effect() -> None:
    """quantity is bit-split with the armour/attack pair and the library
    re-derives which slot is authoritative from the effect type."""
    names = ["effect_type", "quantity", "quantity_float", "armour_attack_quantity",
             "armour_attack_class", "source_player"]
    defaults = dict.fromkeys(names, -1)
    old = _entry(names, {**defaults, "effect_type": 1})
    new = _entry(names, {**defaults, "effect_type": 2}, entry_id=2)
    entry = _Live(effect_type=1, quantity=5, quantity_float=5.0, armour_attack_quantity=2,
                  armour_attack_class=3, source_player=1)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "effect_type")
    assert applied == {"source_player": 1}
    assert set(dropped) == {"quantity", "quantity_float", "armour_attack_quantity",
                            "armour_attack_class"}


def test_a_conditions_quantity_does_carry() -> None:
    """The exclusion set is effect-only: 14 condition types list `quantity` as
    an ordinary field with no second slot behind it."""
    old = _entry(["condition_type", "quantity"], {"condition_type": 1, "quantity": -1})
    new = _entry(["condition_type", "quantity"], {"condition_type": 2, "quantity": -1},
                 entry_id=2)
    entry = _Live(condition_type=1, quantity=4)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "condition_type")
    assert applied == {"quantity": 4}
    assert dropped == ()


def test_a_list_value_is_copied_not_aliased() -> None:
    """restore()'s trap 6: the library mutates lists handed to it in place."""
    old = _entry(["effect_type", "selected_object_ids"],
                 {"effect_type": 1, "selected_object_ids": []})
    new = _entry(["effect_type", "selected_object_ids"],
                 {"effect_type": 2, "selected_object_ids": []}, entry_id=2)
    stored = [11, 12]
    entry = _Live(effect_type=1, selected_object_ids=stored)
    applied, _dropped = trigger_fields.retype_carryover(entry, old, new, "effect_type")
    assert applied["selected_object_ids"] == [11, 12]
    assert applied["selected_object_ids"] is not stored
    applied["selected_object_ids"].append(13)
    assert stored == [11, 12]


def test_an_unknown_old_type_carries_nothing_without_raising() -> None:
    """_describe() already renders such an entry; retyping is how a user gets
    out of it, so it must not be the one op that raises."""
    new = _entry(["effect_type", "source_player"], {"effect_type": 2, "source_player": -1})
    entry = _Live(effect_type=999, source_player=4)
    applied, dropped = trigger_fields.retype_carryover(entry, None, new, "effect_type")
    assert applied == {}
    assert dropped == ()


def test_an_unreadable_field_is_simply_not_carried() -> None:
    """A version-gated property raises rather than being absent."""

    class _Gated(_Live):
        @property
        def gated(self):
            raise RuntimeError("unsupported in this scenario version")

    old = _entry(["effect_type", "gated", "source_player"],
                 {"effect_type": 1, "gated": -1, "source_player": -1})
    new = _entry(["effect_type", "gated", "source_player"],
                 {"effect_type": 2, "gated": -1, "source_player": -1}, entry_id=2)
    entry = _Gated(effect_type=1, source_player=2)
    applied, dropped = trigger_fields.retype_carryover(entry, old, new, "effect_type")
    assert applied == {"source_player": 2}
    assert dropped == ()


# -- intersecting several entries' forms (GH #60) ----------------------------


def test_shared_specs_intersects_by_spec_not_by_name() -> None:
    """The same name can be a different field on two types (INT here,
    INT_LIST there), and the cluster rule locks a name on one entry but not
    the other. A name-keyed intersection would put one widget over both."""
    from dataclasses import replace

    plain = FieldSpec("quantity", trigger_fields.INT)
    listed = FieldSpec("quantity", trigger_fields.INT_LIST)
    locked = replace(plain, read_only=True)
    shared = FieldSpec("source_player", trigger_fields.INT)

    assert trigger_fields.shared_specs([(plain, shared), (listed, shared)]) == (shared,)
    assert trigger_fields.shared_specs([(plain, shared), (locked, shared)]) == (shared,)
    assert trigger_fields.shared_specs([(plain, shared), (plain, shared)]) == (plain, shared)


def test_shared_specs_keeps_the_first_entrys_order() -> None:
    a = FieldSpec("a", trigger_fields.INT)
    b = FieldSpec("b", trigger_fields.INT)
    c = FieldSpec("c", trigger_fields.INT)
    assert trigger_fields.shared_specs([(a, b, c), (c, b, a)]) == (a, b, c)
    assert trigger_fields.shared_specs([(c, b, a), (a, b), (b, a, c)]) == (b, a)


def test_shared_specs_of_one_entry_is_that_entrys_own_form() -> None:
    """N = 1 goes through the same code as N > 1."""
    specs = (FieldSpec("a", trigger_fields.INT), FieldSpec("b", trigger_fields.STR))
    assert trigger_fields.shared_specs([specs]) == specs
    assert trigger_fields.shared_specs([]) == ()


def test_shared_value_is_mixed_only_when_the_values_differ() -> None:
    agreeing = [_FakeEffect(quantity=5), _FakeEffect(quantity=5), _FakeEffect(quantity=5)]
    differing = [_FakeEffect(quantity=5), _FakeEffect(quantity=5), _FakeEffect(quantity=6)]
    assert trigger_fields.shared_value(agreeing, "quantity") == 5
    assert trigger_fields.shared_value(differing, "quantity") is trigger_fields.MIXED
    assert trigger_fields.shared_value([], "quantity") is trigger_fields.MIXED
    # UNSET and None are real stored values, never MIXED.
    assert trigger_fields.shared_value([_FakeEffect(q=-1), _FakeEffect(q=-1)], "q") == trigger_fields.UNSET


def test_shared_value_compares_lists_by_value() -> None:
    """Equal-but-distinct lists agree; identity would call them MIXED and blank
    a field that could then never be group-set."""
    same = [_FakeEffect(ids=[1, 2]), _FakeEffect(ids=[1, 2])]
    other = [_FakeEffect(ids=[1, 2]), _FakeEffect(ids=[2, 1])]
    assert trigger_fields.shared_value(same, "ids") == [1, 2]
    assert trigger_fields.shared_value(other, "ids") is trigger_fields.MIXED


def test_shared_value_reads_through_a_version_tolerant_getattr() -> None:
    """The library's version-gated properties raise rather than being absent."""

    class _Gated:
        @property
        def gated(self):
            raise RuntimeError("unsupported on this version")

    assert trigger_fields.shared_value([_Gated(), _Gated()], "gated") is None
    assert trigger_fields.shared_value([_FakeEffect(x=1)], "x", reader=lambda e, a: 7) == 7
