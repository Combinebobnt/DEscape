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


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


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


def _structure_types(version: str) -> dict[str, dict[str, str]]:
    """{"conditions"/"effects": {retriever: datatype}} from `version`'s
    structure.json, the library's or this repo's."""
    import json

    for base in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR):
        path = base / f"v{version}" / "structure.json"
        if path.is_file():
            structs = json.loads(path.read_text(encoding="utf-8"))["Triggers"]["structs"]["TriggerStruct"]["structs"]
            return {
                kind: {name: r["type"] for name, r in structs[struct]["retrievers"].items()}
                for kind, struct in (("conditions", "ConditionStruct"), ("effects", "EffectStruct"))
            }
    raise FileNotFoundError(version)


def test_every_string_stored_field_is_a_str_spec() -> None:
    """TASK-396: the library's JSON presents create_decision's message_option1/2
    as "" (a number) though they are str32, so an int typed there failed the save."""
    offenders = []
    for version in library_compat.vocabulary_versions():
        vocab = library_compat.load_vocabulary(version)
        types = _structure_types(version)
        for kind, entries, presentation, type_attribute in (
            ("conditions", vocab.conditions, vocab.condition_presentation, "condition_type"),
            ("effects", vocab.effects, vocab.effect_presentation, "effect_type"),
        ):
            for entry in entries.values():
                offenders.extend(
                    f"v{version}/{entry.name}.{spec.name}: {spec.kind}"
                    for spec in trigger_fields.field_specs(entry, presentation, type_attribute)
                    if types[kind].get(spec.name, "").startswith("str") and spec.kind != trigger_fields.STR
                )
    assert offenders == [], f"str-stored fields not given a text field: {offenders}"


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
    """armour_attack_class presents as DamageClass; on an entry that defaults
    it to [] (the library's 77/78 did before TASK-395's fix) the list shape
    wins, since a combo box cannot hold two values."""
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


# -- TASK-059: allow_in_fog, a BOOL that keeps its sentinel --------------------

_FOG = FieldSpec("allow_in_fog", trigger_fields.BOOL, sentinel=trigger_fields.UNSET)
_PLAIN_BOOL = FieldSpec("inverted", trigger_fields.BOOL, sentinel=None)


def test_allow_in_fog_is_a_bool_by_name_with_the_unset_sentinel() -> None:
    """Its presentation is "", which alone would make it an INT spinbox."""
    entry = _entry(["condition_type", "allow_in_fog", "inverted"], {"condition_type": 27})
    specs = trigger_fields.field_specs(entry, {"allow_in_fog": "", "inverted": "bool"}, "condition_type")
    assert [(s.name, s.kind, s.sentinel) for s in specs] == [
        ("allow_in_fog", trigger_fields.BOOL, trigger_fields.UNSET),
        ("inverted", trigger_fields.BOOL, None),
    ]


@pytest.mark.parametrize(("value", "checked"), [(-1, False), (0, False), (1, True), (2, True)])
def test_a_sentinel_bool_shows_unset_as_unticked(value: int, checked: bool) -> None:
    assert trigger_fields.checkbox_checked(_FOG, value) is checked


@pytest.mark.parametrize(("value", "checked"), [(0, False), (1, True), (-1, True)])
def test_a_plain_bool_keeps_bool_of_its_value(value: int, checked: bool) -> None:
    """inverted's -1 stays ticked: whether the game reads it so is unmeasured."""
    assert trigger_fields.checkbox_checked(_PLAIN_BOOL, value) is checked


def test_form_value_maps_the_sentinel_to_zero_for_a_sentinel_bool_only() -> None:
    assert trigger_fields.form_value(_FOG, -1) == 0
    assert trigger_fields.form_value(_FOG, 0) == 0
    assert trigger_fields.form_value(_FOG, 1) == 1
    assert trigger_fields.form_value(_PLAIN_BOOL, -1) == -1
    assert trigger_fields.form_value(FieldSpec("timer", trigger_fields.INT), -1) == -1
    assert trigger_fields.form_value(_FOG, trigger_fields.MIXED) is trigger_fields.MIXED


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


_PRE_141_ORDER = [
    "name",
    "short_description",
    "description",
    "enabled",
    "looping",
    "header",
    "display_as_objective",
    "display_on_screen",
    "mute_objectives",
]


def test_the_trigger_header_fields_are_specs_with_their_library_attributes() -> None:
    """GH #141. The stids keep UNSET (-1 is legal and shows "(unset)");
    description_order is a u32 with no sentinel."""
    specs = {spec.name: spec for spec in trigger_fields.TRIGGER_FIELDS}
    expected = {
        "short_description_string_table_id": (trigger_fields.INT, "short_description_stid", trigger_fields.UNSET),
        "description_string_table_id": (trigger_fields.INT, "description_stid", trigger_fields.UNSET),
        "description_order": (trigger_fields.INT, "description_order", None),
        "execute_on_load": (trigger_fields.BOOL, "execute_on_load", None),
    }
    for name, (kind, attribute, sentinel) in expected.items():
        assert (specs[name].kind, specs[name].attribute, specs[name].sentinel) == (kind, attribute, sentinel), name
    names = [spec.name for spec in trigger_fields.TRIGGER_FIELDS]
    assert [name for name in names if name in _PRE_141_ORDER] == _PRE_141_ORDER
    assert names[names.index("short_description") + 1] == "short_description_string_table_id"
    assert names[names.index("description") + 1] == "description_string_table_id"
    assert names[names.index("header") + 1] == "description_order"
    assert names[-1] == "execute_on_load"


def test_every_trigger_attribute_is_one_the_library_links() -> None:
    from AoE2ScenarioParser.objects.data_objects.trigger import Trigger

    linked = {link.name for link in library_compat._iter_links(Trigger._link_list)}
    assert {spec.attribute for spec in trigger_fields.TRIGGER_FIELDS} <= linked


def test_the_header_fields_carry_tooltips() -> None:
    specs = {spec.name: spec for spec in trigger_fields.TRIGGER_FIELDS}
    for name in ("header", "description_order", "description_string_table_id", "short_description_string_table_id"):
        assert specs[name].tooltip, name
    assert "higher numbers are listed first" in specs["description_order"].tooltip


def test_execute_on_load_is_gated_by_the_library_s_own_support() -> None:
    """The gate is the library link's Support, so a library bump that moves it
    moves the form too. Pinned here so that move is noticed."""
    from AoE2ScenarioParser.objects.data_objects.trigger import Trigger

    link = next(link for link in library_compat._iter_links(Trigger._link_list) if link.name == "execute_on_load")
    assert link.support.since == pytest.approx(1.55)
    names_154 = [spec.name for spec in trigger_fields.trigger_specs("1.54")]
    names_155 = [spec.name for spec in trigger_fields.trigger_specs("1.55")]
    assert "execute_on_load" not in names_154
    assert names_155 == [spec.name for spec in trigger_fields.TRIGGER_FIELDS]
    assert names_154 == names_155[:-1]


def test_an_unparseable_version_shows_every_trigger_field() -> None:
    """Support.supports() calls float(); a version that isn't dotted numbers
    takes the tolerant default rather than raising."""
    assert trigger_fields.trigger_specs("new") == trigger_fields.TRIGGER_FIELDS
    assert trigger_fields.trigger_specs("") == trigger_fields.TRIGGER_FIELDS


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


def _v159_trigger():
    """(loaded, a trigger) of the 1.59 fixture. Loading also initialises the
    library's module-global vocabulary, which _add_condition() builds from.
    The caller keeps `loaded` alive: the library's scenario store does not."""
    from pathlib import Path

    from descape.scenario_io import load_map_and_units, parse_triggers

    loaded = load_map_and_units(Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario")
    return loaded, parse_triggers(loaded).triggers[1]


def _fresh(trigger, kind: str, type_id: int):
    """One fresh entry through the library's own construction path, left out of the trigger."""
    entries = trigger.conditions if kind == "condition" else trigger.effects
    (trigger._add_condition if kind == "condition" else trigger._add_effect)(type_id)
    return entries.pop(len(entries) - 1)


def test_retyping_a_fresh_entry_reports_nothing_dropped() -> None:
    """TASK-059's trap 2: Object Attacked lists no default for five fields that
    a fresh one holds at type 0's -1, which retype used to name as dropped."""
    vocabulary = library_compat.load_vocabulary("1.59")
    by_name = {entry.name: entry.id for entry in vocabulary.conditions.values()}
    attacked, timer = by_name["object_attacked"], by_name["timer"]
    definition = vocabulary.conditions[attacked]
    missing = {a for a in definition.attributes if a not in definition.default_attributes}
    assert {"unit_object", "object_list"} <= missing, "fixture assumption: fields with no listed default"

    _loaded, trigger = _v159_trigger()
    trigger._add_condition(attacked)
    index = len(trigger.conditions) - 1
    assert trigger_fields.retype_entry(trigger, "condition", index, timer, vocabulary.conditions, "condition_type") == ()

    trigger._add_condition(attacked).unit_object = 200
    index = len(trigger.conditions) - 1
    dropped = trigger_fields.retype_entry(trigger, "condition", index, timer, vocabulary.conditions, "condition_type")
    assert dropped == ("unit_object",)


def test_a_fresh_entry_holds_type_0_s_value_where_its_type_lists_no_default() -> None:
    """The fallback retype_carryover() takes is right only if this holds: for
    every 1.54+ (type, field) pair with no listed default, the library's fresh
    entry carries the type-0 value. Unreadable cluster slots (None) are never
    compared, and must stay inside the effect exclusion set."""
    from AoE2ScenarioParser.datasets import conditions as condition_data
    from AoE2ScenarioParser.datasets import effects as effect_data
    from AoE2ScenarioParser.scenarios.aoe2_scenario import _initialise_version_dependencies

    _loaded, trigger = _v159_trigger()
    modules = (condition_data, effect_data)
    names = ("condition_names", "effect_names", "default_attributes", "attributes", "attribute_presentation")
    saved = {(m, n): dict(getattr(m, n)) for m in modules for n in names if hasattr(m, n)}
    versions = [v for v in library_compat.vocabulary_versions() if tuple(map(int, v.split("."))) >= (1, 54)]
    assert "1.59" in versions
    pairs, wrong = 0, []
    try:
        for version in versions:
            vocabulary = library_compat.load_vocabulary(version)
            _initialise_version_dependencies("DE", version)
            for kind, entries, type_attribute in (
                ("condition", vocabulary.conditions, "condition_type"),
                ("effect", vocabulary.effects, "effect_type"),
            ):
                base = entries[0].default_attributes
                for type_id, entry in entries.items():
                    unlisted = [a for a in entry.attributes if a != type_attribute and a not in entry.default_attributes]
                    if type_id == 0 or not unlisted:
                        continue
                    fresh = _fresh(trigger, kind, type_id)
                    for attribute in unlisted:
                        pairs += 1
                        value = trigger_fields._live_value(fresh, attribute)
                        unread = value is None and kind == "effect" and attribute in trigger_fields._RETYPE_EXCLUDED
                        if value != base.get(attribute) and not unread:
                            wrong.append((version, kind, type_id, attribute, value, base.get(attribute)))
    finally:
        for (module, name), value in saved.items():
            getattr(module, name).clear()
            getattr(module, name).update(value)
    assert pairs > 150
    assert wrong == []


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


# -- unit fields captioned after their player (GH #212) ----------------------

_UNIT_FIELDS = ("object_list_unit_id", "object_list_unit_id_2")

# Every effect whose only unit field is its source player's. Spelled out, where
# field_caption() is a rule: a library bump that adds a matching type fails
# here and gets looked at instead of being captioned silently.
_SOURCE_UNIT_TYPES = frozenset(
    {
        "attack_move",
        "build_object",
        "change_object_armor",
        "change_object_attack",
        "change_object_caption",
        "change_object_cost",
        "change_object_description",
        "change_object_hp",
        "change_object_name",
        "change_object_player_color",
        "change_object_player_name",
        "change_object_range",
        "change_object_speed",
        "change_object_stance",
        "change_ownership",
        "counts_units_into_variable",
        "create_object",
        "create_object_armor",
        "create_object_attack",
        "damage_object",
        "disable_object_deletion",
        "disable_object_selection",
        "disable_unit_attackable",
        "disable_unit_targeting",
        "enable/disable_object",
        "enable_object_deletion",
        "enable_object_selection",
        "enable_unit_attackable",
        "enable_unit_targeting",
        "freeze_object",
        "heal_object",
        "kill_object",
        "modify_attribute",
        "modify_attribute_by_variable",
        "modify_object_attribute",
        "modify_object_attribute_by_variable",
        "modify_variable_by_attribute",
        "patrol",
        "place_foundation",
        "remove_object",
        "set_building_gather_point",
        "set_object_cost",
        "stop_object",
        "task_object",
        "teleport_object",
        "train_unit",
        "unload",
    }
)

# Effects with a unit field that keep the name-derived caption: the instruction
# panel's icon, and the three two-unit effects that act on one player.
_UNCAPTIONED_UNIT_TYPES = frozenset(
    {
        "display_instructions",
        "change_object_icon",
        "create_garrisoned_object",
        "research_local_technology",
    }
)

# GH #169: the command-button effects, captioned by what moves where.
_LOCATION_TYPES = frozenset({"add_train_location", "change_train_location", "change_research_location"})

# (effect type, field) -> caption, for every pair whose label is not its name.
_EXPECTED_CAPTIONS = {
    **{(name, "object_list_unit_id"): "source player unit" for name in _SOURCE_UNIT_TYPES},
    ("replace_object", "object_list_unit_id"): "source player unit",
    ("replace_object", "object_list_unit_id_2"): "target player unit",
    **{(name, "object_list_unit_id"): "unit to train" for name in ("add_train_location", "change_train_location")},
    **{(name, "object_list_unit_id_2"): "trained at" for name in ("add_train_location", "change_train_location")},
    ("change_research_location", "object_list_unit_id_2"): "researched at",
    **{(name, "button_location"): "button slot" for name in _LOCATION_TYPES},
}


def test_the_captioned_fields_are_exactly_the_named_set_in_every_version() -> None:
    """Swept over every shipped version, conditions included, like the
    multiline sweep above: the caption is a rule over a type's own attribute
    list, so each version's list has to be asked."""
    assert len(_SOURCE_UNIT_TYPES) == 47
    found = set()
    unit_types = set()
    offenders = []
    for version, kind, raw in _every_shipped_vocabulary():
        presentation = raw.get("-1", {}).get("attribute_presentation", {})
        type_attribute = "condition_type" if kind == "conditions" else "effect_type"
        names = set()
        captioned = {}
        for key, value in raw.items():
            if key == "-1":
                continue
            entry = library_compat.VocabularyEntry(
                id=int(key),
                name=value["name"],
                attributes=tuple(value.get("attributes", ())),
                default_attributes=dict(value.get("default_attributes", {})),
            )
            names.add(entry.name)
            for spec in trigger_fields.field_specs(entry, presentation, type_attribute):
                if spec.name in _UNIT_FIELDS:
                    unit_types.add((kind, entry.name))
                if spec.label != spec.name.replace("_", " "):
                    captioned[(entry.name, spec.name)] = spec.label
        expected = (
            {pair: caption for pair, caption in _EXPECTED_CAPTIONS.items() if pair[0] in names}
            if kind == "effects"
            else {}
        )
        if captioned != expected:
            wrong = sorted(set(captioned.items()) ^ set(expected.items()))
            offenders.append(f"{version}/{kind}: {wrong}")
        found |= set(captioned)
    assert offenders == [], f"wrong captions: {offenders}"
    # Every named pair reaches a real spec in at least one shipped version.
    assert found == set(_EXPECTED_CAPTIONS)
    # And every type with a unit field is accounted for by name, the uncaptioned ones too.
    named = _SOURCE_UNIT_TYPES | _UNCAPTIONED_UNIT_TYPES | _LOCATION_TYPES | {"replace_object"}
    assert unit_types == {("effects", name) for name in named}
    assert len(named) == (
        len(_SOURCE_UNIT_TYPES) + len(_UNCAPTIONED_UNIT_TYPES) + len(_LOCATION_TYPES) + 1
    ), "a type is named twice"


def _unit_spec(caption: str = "") -> FieldSpec:
    return FieldSpec("object_list_unit_id", trigger_fields.REFERENCE, presentation="UnitInfo", caption=caption)


def test_a_caption_is_not_part_of_a_specs_identity() -> None:
    """Two types that caption the same field differently still share it in a
    multi-selection, which shared_specs() decides by equality (GH #60)."""
    plain = _unit_spec()
    captioned = _unit_spec("source player unit")
    assert plain.label == "object list unit id"
    assert captioned.label == "source player unit"
    assert captioned == plain
    assert hash(captioned) == hash(plain)
    assert len({plain, captioned, _unit_spec("target player unit")}) == 1


def test_shared_specs_drops_a_caption_the_selection_disagrees_on() -> None:
    plain = _unit_spec()
    source = _unit_spec("source player unit")
    target = _unit_spec("target player unit")
    player = FieldSpec("source_player", trigger_fields.INT)

    def labels(*lists) -> list[str]:
        return [spec.label for spec in trigger_fields.shared_specs(lists)]

    neutral = ["object list unit id", "source player"]
    assert labels((source, player), (plain, player)) == neutral
    assert labels((plain, player), (source, player)) == neutral
    assert labels((source, player), (target, player)) == neutral
    # The disagreement can sit in any list, not just the second.
    assert labels((source, player), (source, player), (target, player)) == neutral
    assert labels((source, player), (source, player)) == ["source player unit", "source player"]
    assert labels((source, player), (source, player), (source, player)) == ["source player unit", "source player"]
    # The neutral spec is still the same field to everything keyed on equality.
    assert trigger_fields.shared_specs([(source, player), (plain, player)]) == (source, player)


def test_replace_object_and_change_object_icon_still_share_their_unit_field() -> None:
    """The real pair behind the rule above: one captions the field, the other
    does not, and the form over both shows the name-derived caption."""
    vocabulary = library_compat.load_vocabulary("1.59")
    by_name = {entry.name: entry for entry in vocabulary.effects.values()}
    spec_lists = [
        trigger_fields.field_specs(by_name[name], vocabulary.effect_presentation, "effect_type")
        for name in ("replace_object", "change_object_icon")
    ]
    assert [spec.label for spec in spec_lists[0] if spec.name in _UNIT_FIELDS] == [
        "source player unit",
        "target player unit",
    ]
    shared = {spec.name: spec.label for spec in trigger_fields.shared_specs(spec_lists)}
    assert shared["object_list_unit_id"] == "object list unit id"
    assert shared["object_list_unit_id_2"] == "object list unit id 2"


def test_task370_a_none_defaulted_unit_list_shares_with_a_list_defaulted_one() -> None:
    """task_object defaults selected_object_ids to None, kill_object to []; a
    selection of both still shows the one field (GH #60)."""
    vocabulary = library_compat.load_vocabulary("1.59")
    by_name = {entry.name: entry for entry in vocabulary.effects.values()}
    assert by_name["task_object"].default_attributes.get("selected_object_ids") is None, "fixture assumption"
    assert isinstance(by_name["kill_object"].default_attributes.get("selected_object_ids"), list), "fixture assumption"
    spec_lists = [
        trigger_fields.field_specs(by_name[name], vocabulary.effect_presentation, "effect_type")
        for name in ("task_object", "kill_object")
    ]
    shared = {spec.name: spec for spec in trigger_fields.shared_specs(spec_lists)}
    assert "selected_object_ids" in shared
    assert shared["selected_object_ids"].kind == trigger_fields.INT_LIST
    assert shared["selected_object_ids"].sentinel is None


def test_field_label_is_the_caption_or_the_name_derived_default() -> None:
    kill = _entry(["effect_type", "object_list_unit_id", "source_player", "message"], name="kill_object")
    assert trigger_fields.field_label(kill, "object_list_unit_id") == "source player unit"
    assert trigger_fields.field_label(kill, "source_player") == "source player"
    assert trigger_fields.field_label(kill, "message") == "message"
    # An unknown type has no definition to ask.
    assert trigger_fields.field_label(None, "object_list_unit_id") == "object list unit id"
    icon = _entry(["effect_type", "object_list_unit_id", "source_player"], name="display_instructions")
    assert trigger_fields.field_label(icon, "object_list_unit_id") == "object list unit id"
    # The rule needs a source player, and a second unit field switches it off.
    no_player = _entry(["effect_type", "object_list_unit_id"], name="kill_object")
    assert trigger_fields.field_label(no_player, "object_list_unit_id") == "object list unit id"
    two_units = _entry(
        ["effect_type", "object_list_unit_id", "object_list_unit_id_2", "source_player"], name="change_object_icon"
    )
    assert trigger_fields.field_label(two_units, "object_list_unit_id") == "object list unit id"
    assert trigger_fields.field_label(two_units, "object_list_unit_id_2") == "object list unit id 2"


# -- GH #169: the command-button effects -------------------------------------


def test_button_location_lists_no_button_and_both_pages() -> None:
    choices = trigger_fields.enum_choices("ButtonLocation")
    assert len(choices) == 31
    assert [value for _label, value in choices] == [0, *range(1, 16), *range(21, 36)]
    labels = {value: label for label, value in choices}
    assert labels[0] == "no button"
    assert labels[1] == "row 1, col 1"
    assert labels[7] == "row 2, col 2"
    assert labels[15] == "row 3, col 5"
    assert labels[24] == "page 2, row 1, col 4"
    assert labels[35] == "page 2, row 3, col 5"


def test_button_location_stays_a_combo_with_a_tooltip() -> None:
    """31 rows is over PICKER_MIN_CHOICES, but the button grid is its picker."""
    vocabulary = library_compat.load_vocabulary("1.59")
    by_name = {entry.name: entry for entry in vocabulary.effects.values()}
    for name in _LOCATION_TYPES:
        specs = trigger_fields.field_specs(by_name[name], vocabulary.effect_presentation, "effect_type")
        spec = next(spec for spec in specs if spec.name == "button_location")
        assert spec.kind == trigger_fields.ENUM
        assert len(spec.choices) >= trigger_fields.PICKER_MIN_CHOICES
        assert not trigger_fields.wants_picker(spec)
        assert "command panel" in spec.tooltip
        assert all(not other.tooltip for other in specs if other.name != "button_location")
