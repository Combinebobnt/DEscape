"""Pre-DE trigger 1.6 sections (GH #316): the normalizer, the legacy
vocabularies and the read-only Triggers view. Hand-packed bytes only; real
files are the corpus tier's (tests/test_legacy_corpus.py, test_legacy_oracle.py)."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

from descape import (
    find_triggers,
    legacy_triggers,
    library_compat,
    scenario_format,
    scenario_io,
    trigger_clipboard,
    trigger_fields,
    trigger_model,
)

import conftest

_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]
ROOT = Path(__file__).resolve().parent.parent


def _p(fmt, *values):
    return struct.pack("<" + fmt, *values)


def _s(text: bytes) -> bytes:
    return _p("i", len(text)) + text


def _effect(effect_type, values, message=b"", sound=b"", selected=()):
    out = _p("ii", effect_type, len(values)) + b"".join(_p("i", v) for v in values) + _s(message) + _s(sound)
    return out + b"".join(_p("i", v) for v in selected)


def _condition(condition_type, values):
    return _p("ii", condition_type, len(values)) + b"".join(_p("i", v) for v in values)


def _trigger(name, description, effects, conditions, *, enabled=1, looping=0, stid=-1, objective=0, order=0):
    out = _p("I", enabled) + _p("b", looping) + _p("i", stid) + _p("B", objective) + _p("I", order) + _p("i", 0)
    out += _s(description) + _s(name)
    out += _p("i", len(effects)) + b"".join(effects) + b"".join(_p("i", i) for i in range(len(effects)))
    out += _p("i", len(conditions)) + b"".join(conditions) + b"".join(_p("i", i) for i in range(len(conditions)))
    return out


def _effect_fields(count, *, quantity=-1, selected=0, source=1):
    """The first `count` of the 23 legacy effect fields: quantity is field 1, the selected count 4, source player 7."""
    values = [-1] * 23
    values[1], values[4], values[7] = quantity, selected, source
    return values[:count]


def _tail(*, es_only=False, ai_files=True, unknown_effect=False):
    """Two triggers, display order unlike ID order, effects at counts 23, 14 and 0, conditions at 16 and 6."""
    first = _trigger(
        b"Greet",
        b"Say hello",
        [_effect(3, _effect_fields(23, source=2), message=b"Hello", sound=b"bell"),
         _effect(22, _effect_fields(14, selected=2), selected=(7, 9))],
        [_condition(10, [30] + [-1] * 15)],
        objective=1,
        order=5,
    )
    second = _trigger(
        b"Win",
        b"",
        [_effect(99 if unknown_effect else 13, []), _effect(28, _effect_fields(23, quantity=-5))],
        [_condition(3, [4, -1, -1, -1, -1, 2])],
        looping=1,
        stid=60052,
    )
    out = _p("d", 1.6) + _p("b", 0) + _p("i", 2) + first + second + _p("ii", 1, 0)
    out += _p("II", int(ai_files), int(es_only)) + (bytes(396) if es_only else b"")
    if ai_files:
        out += _p("I", 1) + _s(b"ai.per") + _s(b"(defrule)")
    return out


def _parsed(tail: bytes | None = None, *, data_version: str = "1.22"):
    """A DE fixture made a legacy trigger-1.6 document: its structure, tail and vocabulary all legacy."""
    game = {"1.20": scenario_format.AOK, "1.22": scenario_format.AOC}[data_version]
    loaded = conftest.legacy_loaded(game=game)
    loaded._scenario.structure = json.loads(library_compat.legacy_structure_path(data_version).read_text(encoding="utf-8"))
    # A real header-2 file says 1.21, and the library's links gate their fields on it.
    loaded.scenario_version = loaded._scenario.scenario_version = "1.21"
    loaded.trigger_tail = _tail() if tail is None else tail
    loaded.trigger_version = legacy_triggers.TRIGGER_VERSION
    loaded.trigger_read_supported = None
    scenario_io.reactivate(loaded)
    return loaded


def test_normalize_consumes_exactly_and_refuses_a_byte_off():
    for kwargs in ({}, {"es_only": True}, {"ai_files": False}):
        tail = _tail(**kwargs)
        normalized = legacy_triggers.normalize(tail)
        assert normalized.endswith(bytes(1028) + _p("I", 0)), kwargs
        for bad in (tail[:-1], tail + b"\0"):
            with pytest.raises(legacy_triggers.LegacyTriggerFormatError):
                legacy_triggers.normalize(bad)
    with pytest.raises(legacy_triggers.LegacyTriggerFormatError, match="version"):
        legacy_triggers.normalize(_p("d", 1.3) + _tail()[8:])
    assert issubclass(legacy_triggers.LegacyTriggerFormatError, ValueError), "the misparse tuple must catch it"


def test_a_normalized_section_parses_with_padded_fields_and_legacy_names():
    loaded = _parsed()
    manager = scenario_io.parse_triggers(loaded)
    assert manager is not None and loaded.trigger_read_supported is True
    assert loaded.trigger_write_supported is False and loaded.triggers_section_end == -1
    assert [t.name for t in manager.triggers] == ["Greet", "Win"]
    assert list(manager.trigger_display_order) == [1, 0]
    greet, win = manager.triggers
    assert (greet.description, greet.display_as_objective, greet.description_order) == ("Say hello", True, 5)
    assert (win.looping, win.description_stid) == (True, 60052)
    chat, freeze = greet.effects
    assert (chat.message, chat.sound_name, chat.source_player) == ("Hello", "bell", 2)
    assert list(freeze.selected_object_ids) == [7, 9]
    assert freeze.area_x1 == -1 and freeze.instruction_panel_position == -1, "past a short count reads as unset"
    assert win.effects[1].quantity == -5
    assert win.conditions[0].quantity == 4 and win.conditions[0].source_player == 2
    assert win.conditions[0].area_x1 == -1
    hits = find_triggers.find_triggers(loaded, find_triggers.TriggerFindCriteria(text="hello"))
    assert {(h.trigger_name, h.attribute) for h in hits} >= {("Greet", "message"), ("Greet", "description")}
    vocabulary = library_compat.load_vocabulary(loaded.vocabulary_version)
    assert (vocabulary.effects[freeze.effect_type].name, vocabulary.effects[freeze.effect_type].label) == (
        "freeze_object",
        "freeze unit",
    ), "DE's name keys the rule tables; the old one is shown"
    assert trigger_model.exec_order_value(loaded) == 1


def test_an_unknown_type_id_parses_and_a_bad_section_degrades_to_unread():
    loaded = _parsed(_tail(unknown_effect=True))
    manager = scenario_io.parse_triggers(loaded)
    assert manager.triggers[1].effects[0].effect_type == 99
    assert 99 not in library_compat.load_vocabulary(loaded.vocabulary_version).effects
    broken = _parsed(_tail()[:-1])
    assert scenario_io.parse_triggers(broken) is None and broken.trigger_read_supported is False


def test_legacy_vocabularies_match_the_generator_and_their_bands():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import gen_legacy_vocabulary
    finally:
        sys.path.remove(str(ROOT / "tools"))
    for path, text in gen_legacy_vocabulary.build().items():
        assert path.read_text(encoding="utf-8") == text, f"{path} is stale: rerun tools/gen_legacy_vocabulary.py"
    aok = library_compat.load_vocabulary("legacy-1.20")
    aoc = library_compat.load_vocabulary("legacy-1.22")
    assert max(aok.effects) == 23 and sorted(aoc.effects) == list(range(34))
    assert max(aok.conditions) == max(aoc.conditions) == 19
    for vocabulary, labels in (
        (aok, {22: "freeze unit", 23: "use advanced buttons", 5: "tribute"}),
        (aoc, {28: "change object attack", 29: "stop unit", 33: "change object speed"}),
    ):
        assert {i: vocabulary.effects[i].label for i in labels} == labels
    assert aok.conditions[17].label == "researching tech"
    for type_id in (28, 31):
        assert "quantity" in aoc.effects[type_id].attributes
        assert not {"armour_attack_quantity", "armour_attack_class"} & set(aoc.effects[type_id].attributes)
    assert "legacy-1.22" not in library_compat.vocabulary_versions(), "Trigger Details' preview takes the newest DE one"


def test_legacy_trigger_specs_list_only_what_a_legacy_trigger_stores():
    names = [spec.attribute for spec in trigger_fields.trigger_specs("legacy-1.22")]
    assert sorted(names) == sorted(trigger_fields._LEGACY_TRIGGER_ATTRIBUTES)


def test_a_legacy_block_never_pastes_into_a_de_document():
    loaded = _parsed()
    manager = scenario_io.parse_triggers(loaded)
    block = trigger_clipboard.copy_block(manager, [0], scenario_version=loaded.vocabulary_version)
    assert block.scenario_version == "legacy-1.22"
    de = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    target = scenario_io.parse_triggers(de)
    for version in (de.vocabulary_version, ""):
        with pytest.raises(ValueError, match="older-format"):
            trigger_clipboard.paste_into(target, block, scenario_version=version)
    assert trigger_clipboard.legacy_paste_refusal(block.scenario_version, de.vocabulary_version)
    assert not trigger_clipboard.legacy_paste_refusal(de.vocabulary_version, de.vocabulary_version)


@_GUI[0]
@_GUI[1]
def test_the_window_lists_legacy_triggers_read_only():
    from descape import scenario_format

    window = conftest.blank_window(load=False)
    try:
        loaded = _parsed()
        window.load_scenario(Path("old.scx"), loaded=loaded)
        panel = window.trigger_panel
        panel.show_scenario(loaded)
        assert scenario_io.parse_triggers(loaded) is not None
        assert panel.tree.topLevelItemCount() >= 2
        text = panel.status.text()
        assert text.startswith("2 triggers — The Conquerors, trigger format 1.6")
        assert scenario_format.read_only_reason(loaded.source_format).rstrip(".") + ". " in text
        assert ".." not in text
        assert not panel._editable
        window._trigger_clipboard = trigger_clipboard.copy_block(
            scenario_io.parse_triggers(loaded), [0], scenario_version=loaded.vocabulary_version
        )
        block = window._trigger_clipboard
        window.load_scenario(Path("new.aoe2scenario"), loaded=scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH))
        assert "older-format" in window._paste_refusal(block)
    finally:
        conftest.close_window(window)


def test_selected_ids_follow_only_a_count_past_4_and_a_set_field_4():
    effects = [
        _effect(14, [-1, -1, -1, -1]),  # count 4: field 4 absent, so no ids follow
        _effect(14, [-1, -1, -1, -1, 1], selected=(42,)),
        _effect(14, _effect_fields(23, selected=-1)),  # an unset count reads as none
    ]
    tail = _p("d", 1.6) + _p("b", 0) + _p("i", 1) + _trigger(b"Kill", b"", effects, []) + _p("i", 0) + _p("II", 0, 0)
    manager = scenario_io.parse_triggers(_parsed(tail))
    assert [list(e.selected_object_ids) for e in manager.triggers[0].effects] == [[], [42], []]
    hits = find_triggers.find_triggers(_parsed(), find_triggers.TriggerFindCriteria(text="greet"))
    assert any(h.kind == "trigger" and h.trigger_name == "Greet" for h in hits)


def test_an_entry_wider_than_de_is_refused():
    wide = _p("d", 1.6) + _p("b", 0) + _p("i", 1) + _trigger(b"x", b"", [], [_condition(3, [0] * 22)])
    with pytest.raises(legacy_triggers.LegacyTriggerFormatError, match="more than"):
        legacy_triggers.normalize(wide + _p("i", 0) + _p("II", 0, 0))


def test_legacy_entries_never_paste_into_a_de_document():
    loaded = _parsed()
    manager = scenario_io.parse_triggers(loaded)
    refs = [("effect", 0)]
    block = trigger_clipboard.copy_entries(manager, 0, refs, scenario_version=loaded.vocabulary_version)
    de = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    target = scenario_io.parse_triggers(de)
    with pytest.raises(ValueError, match="older-format"):
        trigger_clipboard.paste_entries(target, 0, block, scenario_version=de.vocabulary_version)
    assert "read-only" in trigger_clipboard.legacy_paste_refusal("1.59", loaded.vocabulary_version)
