"""descape.trigger_summary: the Detail column's sentences (GH #388). Default
tier, synthetic entries over the real 1.59 vocabulary and the committed
trigger_summaries.json."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import library_compat, message_markup, player_labels, trigger_summary
from descape.trigger_summary import FULL, RAW, Context, ObjectFilter, Options, Segment

VOCABULARY = library_compat.load_vocabulary("1.59")
NAMES = player_labels.labels_for(("", "Britons", "Franks", "Goths", "Mongols", "", "", "Byzantines", ""))
_CASTLE, _ARCHER, _LOOM, _HIT_POINTS = 82, 4, 22, 0


def _definition(kind: str, name: str):
    entries = VOCABULARY.conditions if kind == "condition" else VOCABULARY.effects
    return next(d for d in entries.values() if d.name == name)


def _entry(kind: str, name: str, **fields):
    return SimpleNamespace(**{**_definition(kind, name).default_attributes, **fields})


def _presentation(kind: str):
    return VOCABULARY.condition_presentation if kind == "condition" else VOCABULARY.effect_presentation


def _summary(kind: str, name: str, context: Context | None = None, options: Options | None = None, **fields):
    entry = _entry(kind, name, **fields)
    return trigger_summary.summarize(
        kind, entry, _definition(kind, name), _presentation(kind), context or Context(labels=NAMES), options or Options()
    )


def _text(kind, name, context=None, options=None, **fields) -> str:
    return _summary(kind, name, context, options, **fields).text


_AREA = {"area_x1": 21, "area_y1": 215, "area_x2": 24, "area_y2": 218}


# -- the four examples of the plan ---------------------------------------------


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        (
            "short",
            [
                "P7 1 Castle at 21,215 24,218",
                "P2 to P1, visible",
                "P4 Loom",
                'P7 Castle "Hit points" set to 1',
            ],
        ),
        (
            "full",
            [
                "At least 1 P7 Castle in 21,215 - 24,218",
                "P2 sees P1 as visible",
                "P4 researches Loom",
                'Set "Hit points" of P7 Castle to 1',
            ],
        ),
    ],
)
def test_the_four_examples_read_as_designed(style, expected) -> None:
    options = Options(sentence_style=style)
    got = [
        _text("condition", "objects_in_area", options=options, quantity=1, object_list=_CASTLE, source_player=7, **_AREA),
        _text("effect", "set_player_visibility", options=options, source_player=2, target_player=1, visibility_state=0),
        _text("effect", "research_technology", options=options, source_player=4, technology=_LOOM, force_research_technology=0),
        _text(
            "effect", "modify_attribute", options=options,
            quantity=1, object_list_unit_id=_CASTLE, source_player=7, operation=1, object_attributes=_HIT_POINTS,
        ),
    ]
    assert got == expected


def test_a_trigger_reference_reads_as_its_name_with_a_link() -> None:
    context = Context(labels=NAMES, trigger_name={4: "Wave 2"}.get)
    summary = _summary("effect", "activate_trigger", context, trigger_id=4)
    assert summary.text == '"Wave 2"'
    assert summary.segments == (Segment('"Wave 2"', (trigger_summary.LINK_TRIGGER, 4)),)
    assert _text("effect", "deactivate_trigger", context, Options(sentence_style=FULL), trigger_id=4) == 'Turn off trigger "Wave 2"'
    assert _text("effect", "activate_trigger", context, trigger_id=9) == "trigger 9 (missing)"
    assert _summary("effect", "activate_trigger", context, Options(link_triggers=False), trigger_id=4).segments[0].link is None


def test_an_inverted_condition_reads_not() -> None:
    context = Context(labels=NAMES, trigger_name={4: "Wave 2"}.get)
    assert _text("condition", "trigger_active", context, trigger_id=4, inverted=1) == 'not "Wave 2"'
    assert _text("condition", "trigger_active", context, Options(sentence_style=FULL), trigger_id=4, inverted=1) == (
        'Not: Trigger "Wave 2" is on'
    )


# -- options -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("player_text", "expected"),
    [("number", "P2 to P1, visible"), ("full", "P2 - Franks to P1 - Britons, visible"), ("name", "Franks to Britons, visible")],
)
def test_player_text_styles(player_text, expected) -> None:
    options = Options(player_text=player_text)
    assert _text("effect", "set_player_visibility", options=options, source_player=2, target_player=1, visibility_state=0) == expected


def test_a_player_with_no_name_reads_as_its_number_in_the_name_style() -> None:
    fields = {"source_player": 5, "target_player": 0, "visibility_state": 0}
    assert _text("effect", "set_player_visibility", options=Options(player_text="name"), **fields) == "P5 to GAIA, visible"


def test_the_swatch_follows_the_player_colour_option() -> None:
    fields = {"source_player": 2, "target_player": 1, "visibility_state": 0}
    with_swatch = _summary("effect", "set_player_visibility", **fields).segments
    assert [s.player for s in with_swatch if s.player is not None] == [2, 1]
    without = _summary("effect", "set_player_visibility", options=Options(player_colour="off"), **fields).segments
    assert all(s.player is None for s in without)
    text = _summary("effect", "set_player_visibility", options=Options(player_colour="text"), **fields).segments
    assert [(s.player, s.mark) for s in text if s.player is not None] == [(2, "text"), (1, "text")]


def test_off_flags_hide_unless_asked_and_ids_follow_names() -> None:
    fields = {"source_player": 4, "technology": _LOOM, "force_research_technology": 0}
    assert _text("effect", "research_technology", **fields) == "P4 Loom"
    assert _text("effect", "research_technology", options=Options(show_off_flags=True), **fields) == "P4 Loom, not forced"
    assert _text("effect", "research_technology", options=Options(show_ids=True), **fields) == f"P4 Loom #{_LOOM}"
    assert _text("effect", "research_technology", **{**fields, "force_research_technology": 1}) == "P4 Loom, forced"


def test_long_text_is_capped_and_a_cap_of_zero_never_cuts() -> None:
    message = "Defend the castle until the reinforcements arrive from the west"
    capped = _text("effect", "display_instructions", source_player=1, message=message, display_time=10)
    assert capped == 'P1 "Defend the castle until the reinforceme…", 10 s'
    assert message in _text("effect", "display_instructions", options=Options(text_cap=0), source_player=1, message=message)


def test_links_follow_their_options() -> None:
    fields = {"quantity": 1, "object_list": _CASTLE, "source_player": 7, **_AREA}
    links = [s.link for s in _summary("condition", "objects_in_area", **fields).segments if s.link]
    assert [link[0] for link in links] == [trigger_summary.LINK_MATCH, trigger_summary.LINK_AREA]
    assert links[1] == (trigger_summary.LINK_AREA, 21, 215, 24, 218)
    off = Options(link_coords=False, link_objects=False)
    assert not [s.link for s in _summary("condition", "objects_in_area", options=off, **fields).segments if s.link]


def test_raw_mode_reads_the_attribute_pairs() -> None:
    summary = _summary("effect", "set_player_visibility", options=Options(mode=RAW), source_player=2, target_player=1, visibility_state=0)
    assert summary.text == "source_player=2, target_player=1, visibility_state=0"
    assert summary.raw == summary.text


def test_options_ignore_unknown_keys() -> None:
    assert Options.from_mapping({"sentence_style": "full", "no_such_option": 1}) == Options(sentence_style="full")


# -- objects -------------------------------------------------------------------


def test_the_object_filter_reads_selected_objects_first() -> None:
    placed = {500: SimpleNamespace(player_id=1, unit_const=_ARCHER, bounds=(0, 0, 1, 1), own_tile=(5, 5))}
    context = Context(labels=NAMES, unit=placed.get)
    one = _summary("effect", "kill_object", context, selected_object_ids=[500])
    assert one.text == "P1 Archer #500"
    assert one.segments[0].link == (trigger_summary.LINK_UNIT, 500)
    several = _summary("effect", "kill_object", context, selected_object_ids=[500, 501, 502])
    assert several.segments == (Segment("3 selected P1 objects", (trigger_summary.LINK_UNITS, (500, 501, 502)), 1),)
    assert _text("effect", "kill_object", context, selected_object_ids=[777]) == "P1 object 777 (not on the map)"


def test_the_object_filter_reads_class_area_and_max() -> None:
    fields = {"source_player": 3, "object_group": 6, **_AREA, "max_units_affected": 5}
    assert _text("effect", "kill_object", **fields) == "P3 class Infantry in 21,215 24,218 (max 5)"
    assert _text("effect", "kill_object", options=Options(sentence_style=FULL), **fields) == (
        "Kill up to 5 P3 class Infantry in 21,215 - 24,218"
    )
    match = _summary("effect", "kill_object", **fields).segments[0].link
    assert match == (trigger_summary.LINK_MATCH, ObjectFilter(3, -1, 6, -1, (21, 215, 24, 218), "P3 class Infantry"))


@pytest.mark.parametrize(
    ("operation", "short", "full"),
    [
        (1, "P1 Archer HP set to 5", "Set the HP of all P1 Archer to 5"),
        (2, "P1 Archer HP +5", "Add 5 to the HP of all P1 Archer"),
        (3, "P1 Archer HP -5", "Take 5 from the HP of all P1 Archer"),
        (4, "P1 Archer HP x5", "Multiply the HP of all P1 Archer by 5"),
        (5, "P1 Archer HP /5", "Divide the HP of all P1 Archer by 5"),
    ],
)
def test_each_operation_reads_in_both_styles(operation, short, full) -> None:
    fields = {"quantity": 5, "object_list_unit_id": _ARCHER, "source_player": 1, "operation": operation}
    assert _text("effect", "change_object_hp", **fields) == short
    assert _text("effect", "change_object_hp", options=Options(sentence_style=FULL), **fields) == full


def test_the_armour_slot_reads_amount_and_class() -> None:
    fields = {
        "armour_attack_quantity": 2, "armour_attack_class": 3, "object_list_unit_id": _ARCHER, "source_player": 1, "operation": 2,
    }
    assert _text("effect", "change_object_armor", **fields) == "P1 Archer armour +2 Pierce"


# -- unset, extras and the fallback --------------------------------------------


def test_an_unset_field_reads_a_question_mark_and_drops_its_optional_part() -> None:
    assert _text("effect", "set_player_visibility", source_player=2, target_player=-1, visibility_state=0) == "P2 to ?, visible"
    assert _text("effect", "unload", source_player=1, object_list_unit_id=_ARCHER) == "P1 Archer"


def test_a_set_field_the_sentence_leaves_out_is_appended() -> None:
    assert _text("effect", "declare_victory", source_player=2, enabled=0) == "P2 + not enabled"
    assert _text("effect", "send_chat", source_player=1, message="Hi", sound_name="horn") == 'P1 "Hi", sound "horn"'
    assert _text("effect", "add_train_location", object_list_unit_id=_ARCHER, source_player=1, object_list_unit_id_2=12,
                 button_location=1, hotkey=16).endswith(" + hotkey: 16")


def test_a_type_with_no_template_reads_label_value_pairs(monkeypatch) -> None:
    monkeypatch.setattr(trigger_summary, "load_table", lambda: {"conditions": {}, "effects": {}})
    assert _text("effect", "set_player_visibility", source_player=2, target_player=1, visibility_state=0) == (
        "source player: P2, target player: P1, visibility state: visible"
    )


# -- the template parser and the matcher ----------------------------------------


def test_the_parser_finds_nested_placeholders_and_rejects_unbalanced_text() -> None:
    assert trigger_summary.placeholders("{op|{variable}[ {armour_attack_class}]|{objtype} {object_attributes}}") == {
        "op", "variable", "armour_attack_class", "objtype", "object_attributes",
    }
    for bad in ("{source_player", "[{quantity}", "x}", "{not a name}"):
        with pytest.raises(ValueError):
            trigger_summary.parse(bad)


def test_the_matcher_applies_owner_const_class_type_and_area() -> None:
    ref = SimpleNamespace(player_id=7, unit_const=_CASTLE, own_tile=(22, 216))
    class_of, type_of = (lambda const: 3), (lambda const: 2)
    assert trigger_summary.matches(ObjectFilter(7, _CASTLE, area=(21, 215, 24, 218)), ref, class_of, type_of)
    assert not trigger_summary.matches(ObjectFilter(6), ref, class_of, type_of)
    assert not trigger_summary.matches(ObjectFilter(7, _ARCHER), ref, class_of, type_of)
    assert not trigger_summary.matches(ObjectFilter(7, group=6), ref, class_of, type_of)
    assert not trigger_summary.matches(ObjectFilter(7, object_type=4), ref, class_of, type_of)
    assert not trigger_summary.matches(ObjectFilter(7, area=(0, 0, 5, 5)), ref, class_of, type_of)


# -- template modifiers --------------------------------------------------------


def test_a_yes_no_choice_reads_one_of_its_texts() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1}
    assert _text("effect", "enable/disable_object", enabled=1, **fields) == "P1 Archer on"
    assert _text("effect", "enable/disable_object", enabled=0, **fields) == "P1 Archer off"
    assert _text("effect", "enable/disable_object", options=Options(sentence_style=FULL), enabled=0, **fields) == "Disable P1 Archer"


@pytest.mark.parametrize(
    ("name", "fields", "expected"),
    [
        ("difficulty_level", {"quantity": 1}, "Hard"),
        ("diplomacy_state", {"quantity": 3, "source_player": 1, "target_player": 2}, "P1 to P2: enemy"),
        ("technology_state", {"quantity": 3, "source_player": 1, "technology": _LOOM}, "P1 Loom done"),
    ],
)
def test_a_named_enum_reads_its_word(name, fields, expected) -> None:
    assert _text("condition", name, **fields) == expected


def test_a_default_hiding_field_shows_only_off_its_default() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1}
    default_state = _definition("effect", "remove_object").default_attributes["object_state"]
    assert _text("effect", "remove_object", object_state=default_state, **fields) == "P1 Archer"
    assert _text("effect", "remove_object", object_state=3, **fields) == "P1 Archer, dead"
    attribute = {**fields, "object_attributes": _HIT_POINTS, "operation": 1, "quantity": 5}
    assert _text("effect", "modify_object_attribute", object_filter=0, **attribute) == 'P1 Archer "Hit points" set to 5'
    assert _text("effect", "modify_object_attribute", object_filter=1, **attribute) == 'P1 Archer "Hit points" set to 5, modified'


def test_either_falls_back_and_appends_the_branch_it_did_not_show() -> None:
    assert _text("effect", "send_chat", source_player=1, string_id=12345) == "P1 string 12345"
    assert _text("effect", "send_chat", source_player=1, message="Hi", string_id=12345) == 'P1 "Hi" + string id: 12345'


def test_a_float_quantity_reads_without_f32_noise() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1, "operation": 4, "object_attributes": 5}
    assert _text("effect", "modify_attribute", quantity=1.2000000476837158, **fields) == 'P1 Archer "Movement speed" x1.2'


def test_an_unnamed_variable_reads_its_number() -> None:
    assert _text("condition", "variable_value", variable=3, comparison=4, quantity=10) == "var 3 >= 10"
    assert _text("condition", "variable_value", options=Options(sentence_style=FULL), variable=3, comparison=4, quantity=10) == (
        "Variable 3 is at least 10"
    )


def test_full_style_says_all_and_starts_with_a_capital() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1}
    assert _text("effect", "kill_object", options=Options(sentence_style=FULL), **fields) == "Kill all P1 Archer"
    assert _text("effect", "disable_object_selection", options=Options(sentence_style=FULL), **fields) == (
        "All P1 Archer cannot be selected"
    )


def test_an_object_beside_an_area_links_only_the_ones_in_it() -> None:
    fields = {"quantity": 1, "object_list": _CASTLE, "source_player": 7, **_AREA}
    link = next(s.link for s in _summary("condition", "objects_in_area", **fields).segments if s.link and s.link[0] == "match")
    assert link == (trigger_summary.LINK_MATCH, ObjectFilter(7, _CASTLE, area=(21, 215, 24, 218), label="P7 Castle"))


@pytest.mark.parametrize("version", library_compat.vocabulary_versions())
def test_every_type_of_every_version_summarises_at_its_defaults(version) -> None:
    vocabulary = library_compat.load_vocabulary(version)
    for kind, entries, presentation in (
        ("condition", vocabulary.conditions, vocabulary.condition_presentation),
        ("effect", vocabulary.effects, vocabulary.effect_presentation),
    ):
        for definition in entries.values():
            entry = SimpleNamespace(**definition.default_attributes)
            for style in ("short", "full"):
                trigger_summary.summarize(kind, entry, definition, presentation, options=Options(sentence_style=style))


def test_a_player_the_sentence_does_not_show_leads_it() -> None:
    placed = {500: SimpleNamespace(player_id=1, unit_const=_ARCHER, bounds=(0, 0, 1, 1), own_tile=(5, 5))}
    context = Context(labels=NAMES, unit=placed.get)
    assert _text("effect", "kill_object", context, source_player=2, selected_object_ids=[500]) == "P2 P1 Archer #500"
    assert _text("effect", "kill_object", context, source_player=1, selected_object_ids=[500]) == "P1 Archer #500"
    lead = _summary("effect", "kill_object", context, source_player=2, selected_object_ids=[500]).segments[0]
    assert (lead.text, lead.player) == ("P2", 2)
    # No player field, or one left unset: nothing leads.
    assert _text("condition", "timer", timer=60) == "60 s"
    assert _text("effect", "kill_object", context, source_player=-1, selected_object_ids=[500]) == "P1 Archer #500"


def test_create_attack_reads_give_in_full_style() -> None:
    fields = {"armour_attack_quantity": 2, "armour_attack_class": 4, "object_list_unit_id": _ARCHER, "source_player": 2}
    full = Options(sentence_style=FULL)
    assert _text("effect", "create_object_attack", options=full, operation=2, **fields) == "Give all P2 Archer 2 Melee attack"
    assert _text("effect", "create_object_attack", options=full, operation=1, **fields) == "Give all P2 Archer 2 Melee attack, set"
    assert _text("effect", "create_object_attack", operation=2, **fields) == "P2 Archer attack +2 Melee"


def test_a_cost_lists_only_the_slots_in_use() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1, "resource_1": 1, "resource_1_quantity": 150}
    assert _text("effect", "change_object_cost", **fields) == "P1 Archer: 150 Wood"
    assert _text("effect", "change_object_cost", resource_2=3, resource_2_quantity=0, **fields) == "P1 Archer: 150 Wood, 0 Gold"


def test_placed_ids_and_how_many_selected_ids_to_list() -> None:
    placed = {500: SimpleNamespace(player_id=1, unit_const=_ARCHER, bounds=(0, 0, 1, 1), own_tile=(5, 5))}
    context = Context(labels=NAMES, unit=placed.get)
    ids = [500, 501, 502, 503, 504]
    assert _text("effect", "kill_object", context, Options(show_unit_ids=False), selected_object_ids=[500]) == "P1 Archer"
    assert _text("effect", "kill_object", context, selected_object_ids=ids) == "5 selected P1 objects"
    assert _text("effect", "kill_object", context, Options(unit_id_limit=3), selected_object_ids=ids) == (
        "5 selected P1 objects (#500, #501, #502, …)"
    )
    assert _text("effect", "kill_object", context, Options(unit_id_limit=5), selected_object_ids=ids) == (
        "5 selected P1 objects (#500, #501, #502, #503, #504)"
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("area", "P1 Archer in 21,215 24,218"),
        ("objects", "3 selected P1 objects"),
        ("objects_first", "3 selected P1 objects in 21,215 24,218"),
        ("area_first", "P1 Archer in 21,215 24,218 and 3 selected P1 objects"),
    ],
)
def test_selected_objects_with_an_area_read_per_the_option(choice, expected) -> None:
    fields = {"source_player": 1, "object_list_unit_id": _ARCHER, "selected_object_ids": [500, 501, 502], **_AREA}
    assert _text("effect", "kill_object", options=Options(selected_with_area=choice), **fields) == expected


def test_objects_in_area_and_object_attacked_lead_with_the_player() -> None:
    placed = {500: SimpleNamespace(player_id=2, unit_const=_CASTLE, bounds=(0, 0, 1, 1), own_tile=(5, 5))}
    context = Context(labels=NAMES, unit=placed.get)
    assert _text("condition", "objects_in_area", quantity=5, object_list=_ARCHER, source_player=1, **_AREA) == (
        "P1 5 Archer at 21,215 24,218"
    )
    assert _text("condition", "objects_in_area", quantity=5, object_list=_ARCHER, source_player=-1, **_AREA) == (
        "5 Archer at 21,215 24,218"
    )
    assert _text("condition", "object_attacked", context, unit_object=500, quantity=5, object_list=_ARCHER, source_player=1) == (
        "P2 Castle #500 by P1 5 Archer"
    )


def test_the_weapon_mode_flag_and_damage_read_plainly() -> None:
    fields = {"quantity": 5, "object_list": _ARCHER, "source_player": 1, **_AREA}
    assert _text("condition", "objects_in_area", include_changeable_weapon_objects=1, **fields) == (
        "P1 5 Archer at 21,215 24,218, incl. other weapon mode"
    )
    assert _text("condition", "objects_in_area", options=Options(show_off_flags=True), include_changeable_weapon_objects=0, **fields) == (
        "P1 5 Archer at 21,215 24,218, not incl. other weapon mode"
    )
    fields = {"quantity": 5, "object_list_unit_id": _ARCHER, "source_player": 1}
    assert _text("effect", "damage_object", **fields) == "5 to P1 Archer"
    assert _text("effect", "heal_object", **fields) == "5 to P1 Archer"
    assert _text("effect", "damage_object", options=Options(show_hp=True), **fields) == "5 HP to P1 Archer"
    assert _text("effect", "heal_object", options=Options(show_hp=True, sentence_style=FULL), **fields) == "Heal all P1 Archer by 5 HP"


def test_colour_tags_colour_the_text_or_show_as_written() -> None:
    message = "<AQUA>Bernabe: I knew <odd> it"
    coloured = _summary("effect", "send_chat", source_player=1, message=message)
    assert coloured.text == 'P1 "Bernabe: I knew <odd> it"'
    assert {s.rgb for s in coloured.segments if "Bernabe" in s.text} == {message_markup.TEXT_COLORS["AQUA"]}
    assert _text("effect", "send_chat", options=Options(text_tags="raw"), source_player=1, message=message) == f'P1 "{message}"'
    capped = _text("effect", "send_chat", options=Options(text_cap=8), source_player=1, message=message)
    assert capped == 'P1 "Bernabe…"', "the cap counts what shows, not the tag"


def test_an_effect_naming_no_object_reads_its_player_alone() -> None:
    fields = {"source_player": 1, "operation": 1, "object_attributes": _HIT_POINTS, "quantity": 100}
    summary = _summary("effect", "modify_attribute", **fields)
    assert summary.text == 'P1 "Hit points" set to 100'
    assert not [s.link for s in summary.segments if s.link]
    assert _text("effect", "kill_object", source_player=1, **_AREA) == "P1 objects in 21,215 24,218"


def test_an_object_to_be_created_links_to_where_it_will_appear() -> None:
    fields = {"object_list_unit_id": _ARCHER, "source_player": 1, "location_x": 30, "location_y": 40}
    links = [s.link for s in _summary("effect", "create_object", **fields).segments if s.link]
    assert links == [(trigger_summary.LINK_TILE, 30, 40), (trigger_summary.LINK_TILE, 30, 40)]
    assert not [s.link for s in _summary("effect", "create_object", options=Options(link_objects=False), **fields).segments
                if s.link and s.text == "P1 Archer"]


def test_a_replacement_links_to_the_object_it_replaces() -> None:
    placed = {500: SimpleNamespace(player_id=1, unit_const=_ARCHER, bounds=(0, 0, 1, 1), own_tile=(5, 5))}
    context = Context(labels=NAMES, unit=placed.get)
    fields = {"source_player": 1, "target_player": 2, "object_list_unit_id_2": _CASTLE, "selected_object_ids": [500]}
    summary = _summary("effect", "replace_object", context, **fields)
    assert summary.text == "P1 Archer #500 with P2 Castle"
    assert {s.link for s in summary.segments if s.link} == {(trigger_summary.LINK_UNIT, 500)}
    assert [s.text for s in summary.segments if s.link] == ["P1 Archer #500", "P2", " ", "Castle"]
