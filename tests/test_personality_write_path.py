"""GH #126 Step 4: the AI personality write path. OptionsEditModel's
personality entries plus scenario_write's two new splices: PlayerDataTwo's
ai_names/ai_files region (between the disables and Messages splices) and the
Files AI library (first after _assemble_body(), addressed from the body's end).

Custom choices here are synthetic, already-resolved AiChoices, so the default
tier needs no install. The Step 0 oracle at the bottom replays the user's own
in-game edit against the editor's saves when those are present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from descape import ai_scripts, content_roots, scenario_write
from descape import player_fields as pf
from descape.ai_scripts import AiChoice
from descape.options_model import OptionsEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import TriggerEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
_GAIA_AND_FILLER = [8, *range(9, 16)]


def _custom(name: str, text: bytes, *library: tuple[bytes, bytes]) -> AiChoice:
    stem = name.rsplit(".", 1)[0].encode()
    lib = ((stem + b".per\x00", text), *library)
    return AiChoice(f"test:{name}", name, ai_scripts.AI_TYPE_CUSTOM, "test", text=text, library=lib, resolved=True)


def _save(loaded, out: Path, **kwargs):
    write_scenario(loaded, out, backup=False, **kwargs)
    reloaded = load_map_and_units(out)
    parse_triggers(reloaded)
    return reloaded


def _rows(loaded) -> list[tuple[str, int, bytes]]:
    layout = pf.personality_layout(loaded)
    assert layout is not None
    names = loaded._scenario.sections["PlayerDataTwo"].retriever_map["ai_names"].data
    body = loaded.decompressed_body
    return [
        (names[i], body[layout.ai_type.offset + i], pf.ai_file_text(body, layout.ai_files[i]))
        for i in range(16)
    ]


def _raw_slots(loaded) -> list[bytes]:
    layout = pf.personality_layout(loaded)
    body = loaded.decompressed_body
    return [body[s.offset : s.offset + s.length] for s in layout.ai_names + layout.ai_files]


def _library(loaded) -> list[tuple[bytes, bytes]]:
    parse_triggers(loaded)
    layout = pf.ai_library_layout(loaded)
    assert layout is not None
    return [(e.name, pf.ai_library_entry_text(loaded.decompressed_body, e)) for e in layout.entries]


# -- model -------------------------------------------------------------------


def test_the_model_seeds_p1_to_p8_from_the_file() -> None:
    model = OptionsEditModel(load_map_and_units(BLANK_TEMPLATE_PATH))
    assert model.personality_supported and not model.has_edits
    for pid in range(1, 9):
        original = model.original_personality(pid)
        assert (original.key, original.resolved) == (f"stored:{pid}", True)
        assert model.current_personality(pid) is original
    with pytest.raises(ValueError):
        pf.personality_field_id(0)  # GAIA is never a personality entry


def test_selecting_the_row_the_file_displays_as_does_not_dirty(tmp_path: Path) -> None:
    """A file whose P1 holds an older-version PromiDE stub displays as
    Standard, so picking Standard restores its own bytes, not the 1.59 stub."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    old_standard = AiChoice("test:old", "PromiDE", 1, "test", text=b"(load \"old stub\")", resolved=True)
    model.set_personality(1, old_standard)
    reloaded = _save(loaded, tmp_path / "old.aoe2scenario", options=model)

    model = OptionsEditModel(reloaded)
    model.set_personality(1, ai_scripts.STANDARD)
    assert not model.has_edits
    model.set_personality(1, ai_scripts.NONE)
    model.set_personality(1, ai_scripts.STANDARD)
    assert not model.has_edits
    assert model.current_personality(1).text == b'(load "old stub")'


def test_undo_replays_a_cached_key_without_disk(tmp_path: Path) -> None:
    model = OptionsEditModel(load_map_and_units(BLANK_TEMPLATE_PATH))
    field_id = pf.personality_field_id(3)
    before = model.current_value(field_id)
    model.set_personality(3, _custom("Foo.ai", b"(defrule)"))
    after = model.current_value(field_id)
    model.set_value(field_id, before)  # OptionsDiffRecord.undo()
    assert not model.has_edits
    model.set_value(field_id, after)  # redo
    assert model.current_personality(3).stored_name == "Foo.ai"
    with pytest.raises(KeyError):
        model.set_value(field_id, "never-cached")


def test_an_unresolved_choice_is_refused() -> None:
    model = OptionsEditModel(load_map_and_units(BLANK_TEMPLATE_PATH))
    with pytest.raises(ValueError):
        model.set_personality(2, AiChoice("x", "X.ai", 0, "test"))


def test_a_custom_choice_is_refused_when_the_library_does_not_verify(monkeypatch) -> None:
    model = OptionsEditModel(load_map_and_units(BLANK_TEMPLATE_PATH))
    monkeypatch.setattr(OptionsEditModel, "personality_custom_supported", property(lambda self: False))
    with pytest.raises(ValueError, match="Files library"):
        model.set_personality(2, _custom("Foo.ai", b"x"))
    model.set_personality(2, ai_scripts.NONE)  # built-ins still write
    assert model.has_personality_edits


def test_a_personality_gate_failure_leaves_other_player_rows_writable() -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    layout = pf.personality_layout(loaded)
    body = bytearray(loaded.decompressed_body)
    body[layout.ai_names[2].offset] ^= 0x7F
    loaded.decompressed_body = bytes(body)
    model = OptionsEditModel(loaded)
    assert not model.personality_supported
    assert model.has_player_edits is False and model._player_field_ids
    with pytest.raises(KeyError):
        model.set_personality(3, ai_scripts.NONE)
    food = next(f for f in model._player_field_ids if f.startswith("player:food:"))
    model.set_value(food, 1234)
    assert model.has_player_edits


# -- PlayerDataTwo splice ----------------------------------------------------


def test_standard_and_none_read_back_and_nothing_else_moves(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    before = _raw_slots(loaded)
    model = OptionsEditModel(loaded)
    model.set_personality(3, ai_scripts.NONE)
    model.set_personality(5, ai_scripts.STANDARD)
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    rows = _rows(reloaded)
    assert rows[2] == ("NoneAi", 2, ai_scripts.NONE_STUB)
    assert rows[4] == ("PromiDE", 1, ai_scripts.STANDARD_STUB)
    after = _raw_slots(reloaded)
    for i in [0, 1, 3, 5, 6, 7, *_GAIA_AND_FILLER]:
        assert after[i] == before[i] and after[16 + i] == before[16 + i], i
    layout = pf.personality_layout(reloaded)
    assert reloaded.decompressed_body[layout.ai_files[2].offset : layout.ai_files[2].offset + 8] == bytes(8)
    assert _library(reloaded) == []  # built-ins never add library entries


def test_grow_same_length_and_shrink(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(3, ai_scripts.NONE)
    grown = _save(loaded, tmp_path / "grow.aoe2scenario", options=model)
    assert len(grown.decompressed_body) > len(loaded.decompressed_body)

    model = OptionsEditModel(grown)
    same = AiChoice(
        "test:same", "NoneAX", 0, "test", text=b"y" * 54, library=((b"NoneAX.per\x00", b"y" * 54),), resolved=True
    )
    model.set_personality(3, same)
    layout = pf.personality_layout(grown)
    region = layout.region
    same_len = _save(grown, tmp_path / "same.aoe2scenario", options=model)
    assert _rows(same_len)[2] == ("NoneAX", 0, b"y" * 54)
    assert pf.personality_layout(same_len).region == region
    # Before the Files library, only P3's name/text and its ai_type byte differ.
    allowed = set(range(layout.ai_names[2].offset, layout.ai_names[2].offset + layout.ai_names[2].length))
    allowed |= set(range(layout.ai_files[2].offset, layout.ai_files[2].offset + layout.ai_files[2].length))
    allowed.add(layout.ai_type.offset + 2)
    old, new = grown.decompressed_body, same_len.decompressed_body
    assert {i for i in range(grown.files_section_start) if old[i] != new[i]} <= allowed

    model = OptionsEditModel(grown)
    model.set_personality(3, _custom("X.ai", b""))
    shrunk = _save(grown, tmp_path / "shrink.aoe2scenario", options=model)
    assert _rows(shrunk)[2] == ("X.ai", 0, b"")
    end = pf.personality_layout(shrunk).region[1]
    assert end < region[1]


def test_a_splice_outside_playerdatatwo_is_refused_at_both_bounds(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(2, ai_scripts.NONE)
    start, end = model.personality_layout.region
    section_start = loaded.player_data_two_section_end - loaded._scenario.sections["PlayerDataTwo"].byte_length
    ai_type = model.personality_layout.ai_type.offset
    for bad in ((section_start - 1, end), (start, ai_type + 1)):
        monkeypatch.setattr(model, "serialize_personality_resize", lambda body, bad=bad: (*bad, b""))
        with pytest.raises(WriteBlockedError, match="PlayerDataTwo"):
            write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)
    assert not (tmp_path / "out.aoe2scenario").exists()


# -- Files library -----------------------------------------------------------


def test_a_custom_ai_adds_its_closure_sorted(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(2, _custom("Zed.ai", b'(load "Lib\\b")', (b"Lib\\b.per\x00", b"(defrule)")))
    model.set_personality(6, _custom("Alpha.ai", b"(defconst a 1)"))
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _rows(reloaded)[1] == ("Zed.ai", 0, b'(load "Lib\\b")')
    assert _library(reloaded) == [
        (b"Alpha.per\x00", b"(defconst a 1)"),
        (b"Lib\\b.per\x00", b"(defrule)"),
        (b"Zed.per\x00", b'(load "Lib\\b")'),
    ]


def test_two_players_on_one_custom_ai_add_one_entry(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    foo = _custom("Foo.ai", b"(defrule)")
    model.set_personality(2, foo)
    model.set_personality(7, foo)
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _library(reloaded) == [(b"Foo.per\x00", b"(defrule)")]


def test_existing_entries_are_kept_byte_for_byte_and_deduplicated_by_stem(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(2, _custom("Foo.ai", b'(load "shared")', (b"shared.per2\x00", b"old shared")))
    first = _save(loaded, tmp_path / "first.aoe2scenario", options=model)

    model = OptionsEditModel(first)
    model.set_personality(4, _custom("Bar.ai", b'(load "Shared")', (b"Shared.per\x00", b"new shared")))
    second = _save(first, tmp_path / "second.aoe2scenario", options=model)
    assert _library(second) == [
        (b"Bar.per\x00", b'(load "Shared")'),
        (b"Foo.per\x00", b'(load "shared")'),
        (b"shared.per2\x00", b"old shared"),
    ]


def test_switching_away_prunes_only_what_nothing_still_loads(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(2, _custom("Foo.ai", b'(load "common")', (b"common.per\x00", b"c")))
    model.set_personality(3, _custom("Bar.ai", b'(load "common")', (b"common.per\x00", b"c")))
    both = _save(loaded, tmp_path / "both.aoe2scenario", options=model)

    model = OptionsEditModel(both)
    model.set_personality(2, ai_scripts.STANDARD)
    one = _save(both, tmp_path / "one.aoe2scenario", options=model)
    assert [k for k, _ in _library(one)] == [b"Bar.per\x00", b"common.per\x00"]

    model = OptionsEditModel(one)
    model.set_personality(3, ai_scripts.NONE)
    none = _save(one, tmp_path / "none.aoe2scenario", options=model)
    assert _library(none) == []
    # An emptied library is written the way a blank file stores one: present=0, no count.
    tail_len = len(loaded.decompressed_body) - loaded.files_section_start
    assert none.decompressed_body[-tail_len:] == loaded.decompressed_body[-tail_len:]


def test_inactive_custom_rows_keep_their_library_entries(tmp_path: Path) -> None:
    """R4_LeLoi_4's inactive P7 keeps its closure; the blank template's P5..P8 are inactive."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(7, _custom("Idle.ai", b"x"))
    model.set_personality(2, _custom("Busy.ai", b"y"))
    both = _save(loaded, tmp_path / "both.aoe2scenario", options=model)
    model = OptionsEditModel(both)
    model.set_personality(2, ai_scripts.NONE)
    after = _save(both, tmp_path / "after.aoe2scenario", options=model)
    assert [k for k, _ in _library(after)] == [b"Idle.per\x00"]


def test_the_library_splice_is_end_relative_after_a_trigger_resize(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)
    triggers = TriggerEditModel(loaded)
    triggers.manager().triggers[0].name = "a much longer trigger name than before, to shift Files"
    triggers.mark_dirty(0)
    model = OptionsEditModel(loaded)
    model.set_personality(3, _custom("Foo.ai", b"(defrule)"))
    model.set_personality(4, ai_scripts.NONE)
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model, triggers=triggers)
    assert parse_triggers(reloaded).triggers[0].name.startswith("a much longer")
    assert _rows(reloaded)[2][:2] == ("Foo.ai", 0) and _rows(reloaded)[3][:2] == ("NoneAi", 2)
    assert (b"Foo.per\x00", b"(defrule)") in _library(reloaded)


def test_every_splice_in_one_save_reads_back(tmp_path: Path) -> None:
    """Five resizing splices plus fixed patches in one save: triggers, the
    Files library, disables, personality, Messages and player_data_1. If the
    personality or library splice sat on the wrong side of a neighbour, one
    of these reads back as garbage."""
    from descape import disables_fields
    from descape.messages_model import MessagesEditModel

    loaded = load_map_and_units(FIXTURE_PATH)
    parse_triggers(loaded)
    triggers = TriggerEditModel(loaded)
    triggers.manager().triggers[0].name = "ordering probe trigger"
    triggers.mark_dirty(0)
    options = OptionsEditModel(loaded)
    messages = MessagesEditModel(loaded)
    options.set_value(disables_fields.disables_field_id("buildings", 2), (72, 621))
    civ_field = pf.player_field_id("civilization", 3)
    civ_after = next(v for v, _ in pf.civilization_choices(loaded) if v != options.current_value(civ_field))
    options.set_value(civ_field, civ_after)
    options.set_personality(3, _custom("Foo.ai", b'(load "inc")', (b"inc.per\x00", b"(defrule)")))
    options.set_personality(6, ai_scripts.NONE)
    messages.set_value("instructions", "Personality ordering probe.")

    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=options, messages=messages, triggers=triggers)
    assert parse_triggers(reloaded).triggers[0].name == "ordering probe trigger"
    assert disables_fields.current_ids(reloaded, "buildings", 2) == (72, 621)
    assert MessagesEditModel(reloaded).current_value("instructions") == "Personality ordering probe."
    assert OptionsEditModel(reloaded).current_value(civ_field) == civ_after
    rows = _rows(reloaded)
    assert rows[2] == ("Foo.ai", 0, b'(load "inc")') and rows[5][:2] == ("NoneAi", 2)
    assert (b"inc.per\x00", b"(defrule)") in _library(reloaded)


def test_a_library_splice_outside_files_is_refused(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(3, _custom("Foo.ai", b"x"))
    parse_triggers(loaded)
    bad = (loaded.files_section_start - 1, len(loaded.decompressed_body), b"")
    monkeypatch.setattr(model, "serialize_ai_library_resize", lambda: bad)
    with pytest.raises(WriteBlockedError, match="Files region"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


def test_the_write_path_re_gates_the_personality_block(tmp_path: Path, monkeypatch) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(3, ai_scripts.NONE)
    monkeypatch.setattr(scenario_write, "personality_write_supported", lambda _loaded: False)
    with pytest.raises(WriteBlockedError, match="personality"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model, backup=False)


def test_a_clean_model_writes_nothing_new(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = OptionsEditModel(loaded)
    model.set_personality(3, ai_scripts.NONE)
    model.set_personality(3, model.original_personality(3))
    assert model.serialize_personality_resize(loaded.decompressed_body) is None
    assert model.serialize_ai_library_resize() is None
    reloaded = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert reloaded.decompressed_body == loaded.decompressed_body


# -- Step 0 oracle -----------------------------------------------------------

_STEP0 = Path(os.environ.get("DESCAPE_GH126_STEP0_DIR", Path(__file__).resolve().parents[1] / "build" / "gh126_step0"))


def _regions(loaded) -> tuple[bytes, bytes, bytes]:
    parse_triggers(loaded)
    layout = pf.personality_layout(loaded)
    library = pf.ai_library_layout(loaded)
    body = loaded.decompressed_body
    start, end = layout.region
    return body[start:end], body[layout.ai_type.offset : layout.ai_type.offset + 16], body[library.ai_files_present.offset :]


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("source", "target", "pick"),
    [("gh126_b_personalities", "gh126_c_p3_standard", "standard"), ("gh126_c_p3_standard", "gh126_b_personalities", "e3")],
)
def test_step0_edit_matches_the_editors_own_save(source, target, pick, tmp_path: Path) -> None:
    """The user's in-game P3 switch, replayed through DEscape, gives the
    editor's own bytes for ai_names/ai_files, ai_type and the whole library."""
    paths = [_STEP0 / f"{name}.aoe2scenario" for name in (source, target)]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"GH #126 Step 0 saves not in {_STEP0} (set DESCAPE_GH126_STEP0_DIR)")
    if pick == "standard":
        choice = ai_scripts.STANDARD
    else:
        roots = content_roots.content_roots()
        e3 = [c for c in ai_scripts.scan_ai_choices(roots) if c.stored_name == "E3-p2.ai" and c.root.kind == "install"]
        if not e3:
            pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
        choice = ai_scripts.resolve(e3[0], roots)
    loaded = load_map_and_units(paths[0])
    model = OptionsEditModel(loaded)
    model.set_personality(3, choice)
    written = _save(loaded, tmp_path / "out.aoe2scenario", options=model)
    assert _regions(written) == _regions(load_map_and_units(paths[1]))
