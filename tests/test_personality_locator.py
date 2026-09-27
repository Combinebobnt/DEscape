"""GH #126 Step 3: the AI personality locator and its two gates, kept apart
from write_targets()/verify_player_block() so a personality-only failure
never greys out the rest of the Players tab.

Default tier runs against the 1.58 blank template (empty Files library);
the corpus tests cover real libraries and the 1.37/1.54 refusals."""

from __future__ import annotations

import struct
from itertools import pairwise

import pytest

from descape import options_model as om
from descape import player_fields as pf
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _loaded(path=BLANK_TEMPLATE_PATH):
    return load_map_and_units(path)


def _corrupt(loaded, offset: int) -> None:
    body = bytearray(loaded.decompressed_body)
    body[offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)


# -- PlayerDataTwo locator ---------------------------------------------------


def test_personality_layout_on_a_clean_load() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    assert layout is not None
    assert len(layout.ai_names) == len(layout.ai_files) == layout.ai_type.length == 16
    start, end = layout.region
    assert start == layout.ai_names[0].offset
    for before, after in pairwise(layout.ai_names + layout.ai_files):
        assert before.offset + before.length == after.offset
    assert end == layout.ai_type.offset  # ai_type directly follows ai_files
    assert end < loaded.player_data_two_section_end


def test_ai_names_spans_hold_the_parsed_names() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    names = loaded._scenario.sections["PlayerDataTwo"].retriever_map["ai_names"].data
    body = loaded.decompressed_body
    for span, value in zip(layout.ai_names, names, strict=True):
        assert body[span.offset : span.offset + span.length] == pf.encode_name_str16(value)


def test_ai_type_target_is_p1_to_p8_only() -> None:
    layout = pf.personality_layout(_loaded())
    assert layout.ai_type_target(1).offset == layout.ai_type.offset
    assert layout.ai_type_target(8) == pf.PlayerWriteTarget(layout.ai_type.offset + 7, 1, "u8")
    for bad in (0, 9):
        with pytest.raises(ValueError):
            layout.ai_type_target(bad)


def test_corrupting_an_ai_names_length_prefix_fails_the_gate() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    _corrupt(loaded, layout.ai_names[2].offset)
    assert not pf.verify_personality_block(loaded)


def test_corrupting_an_ai_files_text_prefix_fails_the_gate() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    _corrupt(loaded, layout.ai_files[4].offset + 8)  # past the 8-byte `unknown`
    assert not pf.verify_personality_block(loaded)


def test_corrupting_ai_type_fails_the_gate() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    _corrupt(loaded, layout.ai_type.offset + 8)  # GAIA's slot
    assert not pf.verify_personality_block(loaded)


def test_a_shifted_anchor_fails_the_gate() -> None:
    loaded = _loaded()
    loaded.player_data_two_section_end += 1
    assert not pf.verify_personality_block(loaded)


def test_a_personality_failure_leaves_the_players_gate_alone() -> None:
    loaded = _loaded()
    layout = pf.personality_layout(loaded)
    _corrupt(loaded, layout.ai_names[0].offset)
    assert not om.personality_write_supported(loaded)
    assert not om.personality_custom_supported(loaded)
    assert om.players_write_supported(loaded)
    assert "player:personality:1" not in pf.write_targets(loaded)


# -- Files library locator ---------------------------------------------------


def test_empty_library_on_the_blank_template() -> None:
    loaded = _loaded()
    library = pf.ai_library_layout(loaded)
    assert library is not None
    assert library.entries == ()
    assert library.number_of_ai_files.length == 0
    assert library.ai_files.length == 0
    assert loaded.files_section_end == len(loaded.decompressed_body)


def test_corrupting_ai_files_present_fails_the_library_gate() -> None:
    loaded = _loaded()
    library = pf.ai_library_layout(loaded)
    _corrupt(loaded, library.ai_files_present.offset)
    assert not pf.verify_ai_library_block(loaded)
    assert pf.verify_personality_block(loaded)


def test_a_shifted_files_anchor_fails_the_library_gate() -> None:
    loaded = _loaded()
    pf.ai_library_layout(loaded)
    loaded.files_section_start += 1
    assert not pf.verify_ai_library_block(loaded)


def test_gates_pass_on_a_clean_load() -> None:
    loaded = _loaded()
    assert om.personality_write_supported(loaded)
    assert om.personality_custom_supported(loaded)


# -- encoders ----------------------------------------------------------------


def test_encode_name_str16() -> None:
    assert pf.encode_name_str16("PromiDE") == struct.pack("<h", 7) + b"PromiDE"
    assert pf.encode_name_str16("") == b"\x00\x00"
    assert pf.encode_name_str16("é.ai") == struct.pack("<h", 5) + "é.ai".encode()
    for bad in (b"PromiDE", "a\x00", 3, "x" * 0x8000):
        with pytest.raises(ValueError):
            pf.encode_name_str16(bad)


def test_encode_str32_is_verbatim_bytes() -> None:
    text = b"(defrule (true) => (chat-to-all \"hi\"))\r\n\x00"
    assert pf.encode_str32(text) == struct.pack("<i", len(text)) + text
    assert pf.encode_str32(b"") == b"\x00\x00\x00\x00"
    with pytest.raises(TypeError):
        pf.encode_str32("text")


# -- corpus ------------------------------------------------------------------


@pytest.mark.corpus
def test_personality_gate_across_the_corpus(scenario_path) -> None:
    loaded = load_map_and_units(scenario_path)
    assert pf.verify_personality_block(loaded), scenario_path.name


@pytest.mark.corpus
def test_ai_library_layout_across_the_corpus(scenario_path) -> None:
    """Every file whose trigger parse succeeds and that has a Files section
    gets a library layout whose entries tile the ai_files span exactly and
    whose keys are the parsed names."""
    loaded = load_map_and_units(scenario_path)
    library = pf.ai_library_layout(loaded)
    if loaded.trigger_read_supported is False or "Files" not in loaded._scenario.sections:
        assert library is None
        assert not om.personality_custom_supported(loaded)
        return
    assert library is not None, scenario_path.name
    pos = library.ai_files.offset
    for entry in library.entries:
        assert entry.span.offset == pos
        pos += entry.span.length
    assert pos == library.ai_files.offset + library.ai_files.length
    parsed = [e.retriever_map["ai_file_name"].data for e in loaded._scenario.sections["Files"].retriever_map["ai_files"].data or []]
    assert [e.name.removesuffix(b"\x00").decode("utf-8") for e in library.entries] == parsed


@pytest.mark.corpus
def test_custom_gate_refuses_the_no_files_structure(corpus_files) -> None:
    """Not vacuous: the corpus has both passing files and ones the Files
    gate refuses (1.37 has no Files section; the 1.54 set fails its trigger
    parse), and the PlayerDataTwo gate passes on both kinds."""
    passed, refused = 0, 0
    for path in corpus_files:
        loaded = load_map_and_units(path)
        assert om.personality_write_supported(loaded), path.name
        if om.personality_custom_supported(loaded):
            passed += 1
        else:
            refused += 1
    assert passed, "no corpus file passed the custom-personality gate"
    assert refused, "the custom-personality gate never refused a corpus file"
