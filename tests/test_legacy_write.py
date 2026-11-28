"""TASK-520: saving a pre-DE scenario back to its own format, gated per edit family."""

import struct
from pathlib import Path

import pytest

from descape import diplomacy_fields, legacy_write, scenario_io
from descape.messages_model import MessagesEditModel
from descape.scenario_write import (
    LegacyEditNotAllowed,
    LegacyFormatReadOnly,
    _patch_header_trigger_count,
    build_patched,
    write_scenario,
)

import conftest


def test_allowed_is_pinned_until_the_oracle_pass():
    """Each family joins a band only with its oracle pass, as a deliberate one-line diff."""
    assert {"1.20": frozenset(), "1.22": frozenset(), "1.26": frozenset()} == legacy_write.ALLOWED


def test_codec_is_a_bijection_over_every_byte():
    raw = bytes(range(256))
    assert legacy_write.encode(legacy_write.decode(raw.replace(b"\x00", b""))) == raw.replace(b"\x00", b"")
    assert legacy_write.encode("Résumé – “test” €") == "Résumé – “test” €".encode("cp1252")


@pytest.mark.parametrize("text", ["中", "\x00", "\x80"])
def test_codec_refuses_what_the_format_cannot_store(text):
    with pytest.raises(ValueError, match="can't be stored"):
        legacy_write.encode(text)


def test_header_length_counts_what_follows_it():
    header = b"1.22" + b"\0\0\0\0" + bytes(61)
    assert struct.unpack_from("<I", legacy_write.fix_header_length(header), 4)[0] == 61


def _legacy_header(instructions: bytes) -> bytes:
    """A hand-packed header-2 FileHeader: version, header_length, savable, timestamp, str32, ivu, player_count."""
    body = struct.pack("<II", 2, 0) + struct.pack("<I", len(instructions)) + instructions + struct.pack("<II", 0, 2)
    return b"1.22" + struct.pack("<I", len(body)) + body


def test_legacy_header_walk_finds_instructions_and_player_count():
    loaded = conftest.legacy_loaded()
    de = loaded._scenario.sections["FileHeader"].retriever_map
    names = ("version", "header_length", "savable", "timestamp_of_last_save", "scenario_instructions", "player_count")
    retrievers = {name: de[name] for name in names}
    retrievers["individual_victories_used"] = de["player_count"]  # a u32 stand-in
    assert scenario_io._header_field_spans(_legacy_header(b"Hello\x00"), retrievers) == {}  # the DE walk
    header = _legacy_header(b"Hello\x00")
    spans = scenario_io._header_field_spans(header, retrievers, legacy=True)
    assert spans["scenario_instructions"] == (20, 26)
    assert header[slice(*spans["player_count"])] == struct.pack("<I", 2)


def test_legacy_diplomacy_section_is_four_bytes_shorter():
    loaded = conftest.legacy_loaded()
    retrievers = dict(loaded._scenario.sections["Diplomacy"].retriever_map)
    assert diplomacy_fields.expected_section_length(retrievers) == 12616
    for name in diplomacy_fields._TEAM_SCALARS:
        retrievers.pop(name)
    assert diplomacy_fields.expected_section_length(retrievers) == 12612


def test_gate_refuses_every_kind_until_allowed_and_non_families_always(tmp_path):
    loaded = conftest.legacy_loaded()
    messages = MessagesEditModel(loaded)
    messages.set_value("hints", "Résumé")
    with pytest.raises(LegacyEditNotAllowed, match="messages"):
        build_patched(loaded, messages=messages)
    with legacy_write._allow_for_oracle("1.22"):
        patched = build_patched(loaded, messages=messages)
        assert struct.unpack_from("<I", patched.header, 4)[0] == struct.unpack_from("<I", loaded.header_bytes, 4)[0]
        messages.set_value("hints_id", 5)
        with pytest.raises(LegacyEditNotAllowed, match="string ids"):
            build_patched(loaded, messages=messages)
        messages.set_value("hints_id", messages.original_value("hints_id"))
        tile = loaded.map_manager.terrain[0]
        tile.terrain_id = 45  # not a Conquerors terrain
        with pytest.raises(LegacyEditNotAllowed, match="terrain id 45"):
            build_patched(loaded, messages=messages)
        tile.terrain_id = 0
        tile.elevation = legacy_write.BAND_MAX_ELEVATION["1.22"] + 1
        with pytest.raises(LegacyEditNotAllowed, match="elevation"):
            build_patched(loaded)
        tile.elevation = 0
        assert write_scenario(loaded, tmp_path / "slot.scx.autosave", messages=messages, backup=False).wrote
    assert legacy_write.ALLOWED["1.22"] == frozenset()


def test_an_instructions_edit_splices_the_header_copy_and_keeps_each_nul(monkeypatch):
    """The real header-2 path: a hand-packed header's Instructions copy is found, spliced in cp1252
    with its trailing NUL, and header_length is rewritten; an edited Messages field keeps its NUL too."""
    from descape import messages_model

    loaded = conftest.legacy_loaded()
    header = _legacy_header(b"Old\x00")
    section = loaded._scenario.sections["FileHeader"]
    names = ("version", "header_length", "savable", "timestamp_of_last_save", "scenario_instructions", "player_count")
    retrievers = {name: section.retriever_map[name] for name in names}
    retrievers["individual_victories_used"] = retrievers["player_count"]  # a u32 stand-in
    monkeypatch.setattr(section, "retriever_map", retrievers)  # header 2's own fields only
    loaded.header_bytes = header
    loaded.header_instructions_span = scenario_io._header_instructions_span(header, retrievers, legacy=True)
    assert loaded.header_instructions_span == (20, 24)
    stored = {"instructions": b"Old\x00", "hints": b"Hint\x00", "victory": b"", "loss": b"", "history": b"", "scouts": b""}
    monkeypatch.setattr(messages_model, "raw_payloads", lambda _loaded: dict(stored))
    messages = MessagesEditModel(loaded)
    assert messages.original_value("instructions") == "Old"
    messages.set_value("instructions", "Résumé €")
    messages.set_value("hints", "Hint – two")
    with legacy_write._allow_for_oracle("1.22"):
        patched = build_patched(loaded, messages=messages)
    text = "Résumé €".encode("cp1252")
    assert patched.header[16:20 + len(text) + 1] == struct.pack("<I", len(text) + 1) + text + b"\x00"
    assert struct.unpack_from("<I", patched.header, 4)[0] == len(patched.header) - 8
    assert patched.header[-8:] == header[-8:], "individual_victories_used and player_count follow untouched"
    assert messages._encoded("hints") == "Hint – two".encode("cp1252") + b"\x00"
    assert messages._encoded("victory") == b"", "an untouched field writes its stored bytes"


def _stance_edit(options):
    field_id = next(iter(diplomacy_fields.stance_offsets(options.loaded)))
    options.set_value(field_id, 3 if options.original_value(field_id) != 3 else 0)


@pytest.mark.parametrize("kind", ["units", "map options", "allied victory", "triggers"])
def test_kinds_outside_the_four_families_raise_even_when_all_four_are_allowed(kind):
    from types import SimpleNamespace

    from descape import option_fields
    from descape.options_model import OptionsEditModel
    from descape.unit_model import UnitEditModel

    loaded = conftest.legacy_loaded(Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario")
    kwargs = {}
    if kind == "units":
        units = UnitEditModel(loaded)
        units.remove(next(u for row in loaded.unit_manager.units for u in row))
        kwargs["units"] = units
    elif kind == "triggers":
        kwargs["triggers"] = SimpleNamespace(has_edits=True)  # model_kinds() reads only this
    else:
        options = kwargs["options"] = OptionsEditModel(loaded)
        if kind == "map options":
            spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == "lock_teams")
            options.set_value(spec.field_id, 1 - option_fields.current_value(loaded, spec))
        else:
            field_id = next(iter(diplomacy_fields.allied_victory_offsets(loaded)))
            options.set_value(field_id, 1 - options.original_value(field_id))
    with legacy_write._allow_for_oracle("1.22"), pytest.raises(LegacyEditNotAllowed, match=f"save {kind} edits"):
        build_patched(loaded, **kwargs)


def test_a_stance_edit_saves_only_once_its_band_allows_diplomacy():
    from descape.options_model import OptionsEditModel

    loaded = conftest.legacy_loaded()
    options = OptionsEditModel(loaded)
    _stance_edit(options)
    assert options.pending_kinds() == {legacy_write.DIPLOMACY}
    with pytest.raises(LegacyEditNotAllowed, match="diplomacy"):
        build_patched(loaded, options=options)
    with legacy_write._allow_for_oracle("1.22", {legacy_write.TERRAIN}), pytest.raises(LegacyEditNotAllowed):
        build_patched(loaded, options=options)
    with legacy_write._allow_for_oracle("1.22", {legacy_write.DIPLOMACY}):
        assert build_patched(loaded, options=options).body != loaded.decompressed_body


def test_a_script_on_a_legacy_document_keeps_only_tiles_its_band_saves():
    """A run's illegal or unallowed tile fails and rolls back; a unit write refuses up front."""
    from descape import scripting
    from descape.edit_history import EditHistory

    loaded = conftest.legacy_loaded(Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario")
    start = loaded.map_manager.terrain[0].terrain_id
    legal = 1 if start != 1 else 2

    def run(script):
        history = EditHistory()
        return scripting.call(script, scripting.ScriptRun(loaded, history), title="t", label="t"), history

    outcome, _ = run(lambda editor: editor.set_terrain(0, 0, legal))
    assert not outcome.ok and "can't save terrain edits" in " ".join(outcome.findings)
    with legacy_write._allow_for_oracle("1.22", {legacy_write.TERRAIN}):
        for script, finding in (
            (lambda editor: editor.set_terrain(0, 0, 45), "terrain_id 45 is not a The Conquerors terrain"),
            (lambda editor: editor.set_elevation(0, 0, 1), "can't save elevation edits"),
        ):
            outcome, _ = run(script)
            assert not outcome.ok and finding in " ".join(outcome.findings), outcome.findings
            assert loaded.map_manager.terrain[0].terrain_id == start and loaded.map_manager.terrain[0].elevation == 0
        unit = loaded.unit_manager.units[1][0]
        outcome, _ = run(lambda editor: editor.move_unit(unit, unit.x + 1, unit.y))
        assert not outcome.ok and "can't save" not in outcome.error and "Older-format" in outcome.error
        outcome, history = run(lambda editor: editor.set_terrain(0, 0, legal))
        assert outcome.ok and outcome.changed == ("terrain",) and len(history.records) == 1
    assert loaded.map_manager.terrain[0].terrain_id == legal


def test_header_trigger_count_patch_refuses_a_legacy_header():
    loaded = conftest.legacy_loaded()
    with pytest.raises(LegacyFormatReadOnly):
        _patch_header_trigger_count(loaded.header_bytes, loaded, 1)


def _messages_body(retriever_map, prefixes: dict[str, bytes]) -> bytes:
    """A Messages section in walk order: zero ids, then each text field's prefix (default an empty str16)."""
    out = b""
    for name in scenario_io._MESSAGE_FIELD_ORDER:
        if (retriever := retriever_map.get(name)) is None:
            continue
        if name in scenario_io._MESSAGE_TEXT_FIELDS:
            out += prefixes.get(name, b"\x00\x00")
        else:
            out += bytes(scenario_io.retriever_length(retriever))
    return out


def test_a_negative_messages_length_fails_closed_and_payloads_stay_aligned():
    """A -1 str16 prefix would be rewritten as 0, so the legacy walk refuses it; the payload
    reader still steps over it by the prefix alone, so later fields keep their own bytes."""
    from types import SimpleNamespace

    from descape import messages_fields

    loaded = conftest.legacy_loaded()
    retrievers = loaded._scenario.sections["Messages"].retriever_map
    first = next(n for n in scenario_io._MESSAGE_FIELD_ORDER if n in scenario_io._MESSAGE_TEXT_FIELDS)
    clean = _messages_body(retrievers, {})
    assert scenario_io._verify_messages_block(clean, 0, len(clean), retrievers, legacy=True)
    negative = _messages_body(retrievers, {first: struct.pack("<h", -1)})
    assert not scenario_io._verify_messages_block(negative, 0, len(negative), retrievers, legacy=True)

    fields = messages_fields.present_fields(retrievers)
    body = bytes(4 * len(fields)) + struct.pack("<h", -1) + b"".join(
        struct.pack("<h", 2) + b"x\x00" for _ in fields[1:]
    )
    fake = SimpleNamespace(_scenario=loaded._scenario, decompressed_body=body, messages_section_start=0)
    payloads = messages_fields.raw_payloads(fake)
    assert payloads[fields[0].field_id] == b"" and all(payloads[f.field_id] == b"x\x00" for f in fields[1:])


def test_only_a_changed_value_has_to_be_band_legal():
    """A tile whose stored terrain is outside the corpus set still takes an elevation edit."""
    loaded = conftest.legacy_loaded()
    o = loaded.terrain_block_offset
    stored = bytearray(loaded.decompressed_body)
    stored[o] = 45  # not a Conquerors terrain, as if the file held it
    loaded.decompressed_body = bytes(stored)
    raised = bytearray(stored)
    raised[o + 1] += 1
    assert legacy_write.illegal_tile(loaded, bytes(raised), o) is None
    raised[o] = 46
    assert "terrain id 46" in legacy_write.illegal_tile(loaded, bytes(raised), o)
    raised[o], raised[o + 1] = 45, legacy_write.BAND_MAX_ELEVATION["1.22"] + 1
    assert "elevation" in legacy_write.illegal_tile(loaded, bytes(raised), o)


def test_a_legacy_field_with_a_nul_reserves_its_byte(monkeypatch):
    from descape import messages_model
    from descape.messages_fields import MAX_FIELD_BYTES

    loaded = conftest.legacy_loaded()
    stored = {"instructions": b"", "hints": b"Hint\x00", "victory": b"", "loss": b"", "history": b"", "scouts": b""}
    monkeypatch.setattr(messages_model, "raw_payloads", lambda _loaded: dict(stored))
    messages = MessagesEditModel(loaded)
    with pytest.raises(ValueError, match="limit"):
        messages.set_value("hints", "a" * MAX_FIELD_BYTES)
    messages.set_value("hints", "a" * (MAX_FIELD_BYTES - 1))
    messages.set_value("victory", "a" * MAX_FIELD_BYTES)  # no NUL to keep
    assert len(messages._encoded("hints")) == MAX_FIELD_BYTES


def test_convert_to_version_refuses_a_legacy_document():
    from descape.scenario_convert import convert_scenario

    with pytest.raises(LegacyFormatReadOnly):
        convert_scenario(conftest.legacy_loaded())
