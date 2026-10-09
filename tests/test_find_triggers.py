"""descape.find_triggers: Find and Replace's Qt-free trigger search and its
replace planning (GH #144 Part B).

Default tier, no QApplication, on tests/fixtures/find_triggers_120x120.aoe2scenario
(tools/gen_trigger_fixture.py --find; its builder documents every shape) plus
v159_units_triggers.aoe2scenario for a 1.59 vocabulary. No fixture carries a
pre-1.44 file, whose unit_object/next_object have presentation "" and so are
not references there; that rule is unit_references.references_in()'s, pinned
in tests/test_unit_references.py, which find_triggers calls rather than forks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from descape import find_triggers as ft
from descape.find_text import FindPatternError, compile_pattern
from descape.find_triggers import TriggerFindCriteria
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIND_FIXTURE = FIXTURES / "find_triggers_120x120.aoe2scenario"
V159_FIXTURE = FIXTURES / "v159_units_triggers.aoe2scenario"

_ARCHER = 4
_HOUSE = 70
_VILLAGER = 83
_KNIGHT = 38
_ARCHER_A = 500
_ARCHER_B = 501
_VILLAGER_REF = 503
_RETARGET = 2


def _loaded(path=FIND_FIXTURE):
    loaded = load_map_and_units(path)
    assert parse_triggers(loaded) is not None
    return loaded


def _find(loaded, **kwargs):
    return ft.find_triggers(loaded, TriggerFindCriteria(**kwargs))


def _where(hits):
    return {(h.trigger_index, h.kind, h.entry_index, h.attribute) for h in hits}


# -- text scopes ------------------------------------------------------------------


def test_each_text_scope() -> None:
    loaded = _loaded()
    assert _where(_find(loaded, text="Find", text_scopes=frozenset({"names"}))) == {
        (0, "trigger", -1, "name"),
        (3, "trigger", -1, "name"),
    }
    assert _where(_find(loaded, text="second line", text_scopes=frozenset({"descriptions"}))) == {
        (0, "trigger", -1, "description")
    }
    # create_decision's option messages are str32 the vocabulary calls INT.
    assert _where(_find(loaded, text="burn", text_scopes=frozenset({"messages"}))) == {
        (0, "effect", 1, "message_option2")
    }
    assert _where(_find(loaded, text="archers to", text_scopes=frozenset({"messages"}))) == {
        (0, "effect", 0, "message")
    }
    xs = _find(loaded, text="findCounter", text_scopes=frozenset({"xs"}))
    assert _where(xs) == {(0, "effect", 3, "message")}
    assert xs[0].field_class == "xs"
    assert "\n" in xs[0].value_display and "\r" not in xs[0].value_display


def test_identifiers_stay_hidden_while_their_scope_is_off() -> None:
    loaded = _loaded()
    assert _find(loaded, text="find_var") == []
    assert _find(loaded, text="find_sound") == []
    found = _find(loaded, text="find_", text_scopes=frozenset({"identifiers"}))
    assert _where(found) == {(0, "effect", 0, "sound_name"), (0, "effect", 2, "message")}


def test_empty_text_gives_no_text_hits_and_no_facet_lists_triggers() -> None:
    loaded = _loaded()
    rows = _find(loaded)
    assert [(h.trigger_index, h.field_class) for h in rows] == [(i, "trigger") for i in range(4)]
    assert [h.trigger_index for h in _find(loaded, enabled=False)] == [3]
    assert compile_pattern("", False, False) is None


def test_case_insensitive_spans_index_the_original_text() -> None:
    """"ß".casefold() is "ss": folding the text would shift every span after it."""
    pattern = compile_pattern("ss", False, False)
    text = "Straße Strasse"
    spans = [m.span() for m in pattern.finditer(text)]
    assert [text[a:b] for a, b in spans] == ["ss"]
    pattern = compile_pattern("BRIDGE", False, False)
    loaded = _loaded()
    hit = next(h for h in ft.find_triggers(loaded, TriggerFindCriteria(text="BRIDGE")) if h.attribute == "message")
    a, b = hit.spans[0]
    assert hit.value_display[a:b] == "bridge"


def test_type_hits_across_every_type_field() -> None:
    loaded = _loaded()
    hits = _find(loaded, type_consts=frozenset({_ARCHER, _HOUSE}))
    where = _where(hits)
    assert (_RETARGET, "condition", 0, "object_list") in where
    assert (_RETARGET, "effect", 0, "object_list_unit_id") in where
    assert (_RETARGET, "effect", 5, "object_list_unit_id_2") in where
    assert (_RETARGET, "effect", 6, "building_list") in where
    assert all(h.field_class == "type" for h in hits)


def test_object_ref_hits_for_both_selected_object_ids_shapes() -> None:
    loaded = _loaded()
    hits = _find(loaded, object_refs=frozenset({_ARCHER_B, _VILLAGER_REF}))
    where = _where(hits)
    # patrol's selected_object_ids is INT_LIST-shaped, task_object's and build_object's REFERENCE-shaped.
    assert (_RETARGET, "effect", 2, "selected_object_ids") in where
    assert (_RETARGET, "effect", 3, "selected_object_ids") in where
    assert (_RETARGET, "effect", 6, "selected_object_ids") in where
    assert ft.trigger_ref_counts(_loaded(V159_FIXTURE)) == {200: 2}


def test_facets_and_per_trigger() -> None:
    loaded = _loaded()
    # Trigger 2 has archer type fields and refs to 500; trigger 0 has neither.
    hits = _find(loaded, type_consts=frozenset({_ARCHER}), object_refs=frozenset({_ARCHER_A}))
    assert {h.trigger_index for h in hits} == {_RETARGET}
    assert {h.field_class for h in hits} == {"type", "object_ref"}
    assert _find(loaded, text="bridge", object_refs=frozenset({_ARCHER_A})) == []


def test_trigger_ref_counts_never_forces_a_parse() -> None:
    loaded = load_map_and_units(FIND_FIXTURE)
    assert ft.trigger_ref_counts(loaded) is None
    assert loaded._trigger_manager is None
    parse_triggers(loaded)
    assert ft.trigger_ref_counts(loaded) == {_ARCHER_A: 5, _ARCHER_B: 1, _VILLAGER_REF: 2}


# -- text replace -------------------------------------------------------------------


def _text_plan(loaded, text, replacement, *, regex=False, scopes=ft.DEFAULT_SCOPES):
    hits = _find(loaded, text=text, regex=regex, text_scopes=scopes)
    pattern = compile_pattern(text, regex, False)
    return hits, ft.plan_text_replace(loaded, hits, pattern, replacement, regex)


def test_regex_template_and_literal_backslash() -> None:
    loaded = _loaded()
    _hits, plan = _text_plan(loaded, r"Hold the (\w+)", r"Guard the \1", regex=True)
    assert [w.new_value for w in plan.writes] == ["Guard the bridge.\r\nArchers to the walls."]
    _hits, plan = _text_plan(loaded, "Hold", r"Keep\1")
    assert plan.writes[0].new_value.startswith("Keep\\1 the bridge.")
    with pytest.raises(FindPatternError):
        _text_plan(loaded, r"Hold", r"\9", regex=True)


def test_prose_keeps_its_crlf_token() -> None:
    loaded = _loaded()
    _hits, plan = _text_plan(loaded, "the walls", "the gate", scopes=frozenset({"messages"}))
    assert plan.writes[0].new_value == "Hold the bridge.\r\nArchers to the gate."
    # Adding a line break in display form writes the field's own token.
    _hits, plan = _text_plan(loaded, "bridge.", "bridge.\nNew line", scopes=frozenset({"messages"}))
    assert plan.writes[0].new_value == "Hold the bridge.\r\nNew line\r\nArchers to the walls."


def test_refusals() -> None:
    loaded = _loaded()
    _hits, plan = _text_plan(loaded, "bridge", "a\x00b")
    assert plan.writes == [] and [r for _h, r in plan.refusals] == ["result contains a NUL character"]
    _hits, plan = _text_plan(loaded, "Burn it", "Burn\nit", scopes=frozenset({"messages"}))
    assert [r for _h, r in plan.refusals] == ["newline in a single-line field"]
    _hits, plan = _text_plan(loaded, "Retarget", "Re\ntarget", scopes=frozenset({"names"}))
    assert [r for _h, r in plan.refusals] == ["newline in a single-line field"]
    _hits, plan = _text_plan(loaded, r"\[Find\]", "[Other]", regex=True, scopes=frozenset({"names"}))
    assert [r for _h, r in plan.refusals] == ["would move the trigger to another tag or section"]
    _hits, plan = _text_plan(loaded, "Retarget", "--- Retarget ---", scopes=frozenset({"names"}))
    assert [r for _h, r in plan.refusals] == ["would move the trigger to another tag or section"]
    # Within the tag: allowed.
    _hits, plan = _text_plan(loaded, "Messages", "Notes", scopes=frozenset({"names"}))
    assert [w.new_value for w in plan.writes] == ["[Find] Notes"]


def test_a_no_op_write_is_dropped() -> None:
    loaded = _loaded()
    hits, plan = _text_plan(loaded, "bridge", "bridge")
    assert hits and plan.writes == [] and plan.refusals == []
    assert plan.touched == []


# -- type replace ---------------------------------------------------------------------


def test_type_replace_writes_only_checked_live_old_values() -> None:
    loaded = _loaded()
    hits = _find(loaded, type_consts=frozenset({_ARCHER}))
    plan = ft.plan_type_replace(loaded, hits, frozenset({_ARCHER}), _KNIGHT)
    assert {(w.kind, w.entry_index, w.attribute) for w in plan.writes} >= {
        ("condition", 0, "object_list"),
        ("effect", 5, "object_list_unit_id_2"),
    }
    assert all(w.new_value == _KNIGHT for w in plan.writes)
    assert plan.touched == [_RETARGET]
    unknown = ft.plan_type_replace(loaded, hits, frozenset({_ARCHER}), 999_999)
    assert unknown.writes == [] and len(unknown.refusals) == len(hits)


# -- follow an object Replace ---------------------------------------------------------


def test_follow_replace_sorts_entries_into_tiers() -> None:
    loaded = _loaded()
    plan = ft.plan_follow_replace(loaded, {_ARCHER_A: (_ARCHER, _KNIGHT)})
    tiers = {(e.kind, e.entry_index, e.attribute): e.tier for e in plan.entries}
    assert tiers[("effect", 0, "object_list_unit_id")] == ft.TIER_PAIRED
    assert tiers[("effect", 1, "object_list_unit_id")] == ft.TIER_FILTER_ONLY
    assert tiers[("effect", 2, "object_list_unit_id")] == ft.TIER_MIXED
    # The object_attacked condition and the task_object target are report-only, never paired.
    assert tiers[("condition", 0, "object_list")] == ft.TIER_REPORT
    assert tiers[("effect", 3, "object_list_unit_id")] == ft.TIER_REPORT
    assert tiers[("effect", 4, "object_list_unit_id")] == ft.TIER_CLASS
    assert tiers[("effect", 5, "object_list_unit_id_2")] == ft.TIER_REPORT
    assert [(w.entry_index, w.new_value) for w in plan.writes()] == [(0, _KNIGHT)]
    assert [(w.entry_index, w.new_value) for w in plan.writes(include_filter_only=True)] == [(0, _KNIGHT), (1, _KNIGHT)]
    assert plan.filter_only_summary() == ["Create Object: 1 effect name Archer"]


def test_follow_replace_of_both_archers_pairs_the_patrol() -> None:
    loaded = _loaded()
    plan = ft.plan_follow_replace(loaded, {_ARCHER_A: (_ARCHER, _KNIGHT), _ARCHER_B: (_ARCHER, _KNIGHT)})
    tiers = {(e.kind, e.entry_index): e.tier for e in plan.entries if e.attribute == "object_list_unit_id"}
    assert tiers[("effect", 2)] == ft.TIER_PAIRED


# -- CSV -----------------------------------------------------------------------------


def test_hits_to_csv_round_trip_and_compatdata(tmp_path) -> None:
    hits = _find(_loaded(), text="bridge")
    out = tmp_path / "hits.csv"
    assert ft.hits_to_csv(hits, out) == len(hits)
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == ft.CSV_COLUMNS
    assert rows[1][1] == "[Find] Messages"
    blocked = tmp_path / "compatdata" / "hits.csv"
    blocked.parent.mkdir()
    with pytest.raises(WriteBlockedError):
        ft.hits_to_csv(hits, blocked)
    assert not blocked.exists()
