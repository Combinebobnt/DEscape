"""Guards tests/fixtures/units_120x120.aoe2scenario, the default tier's only
fixture with a non-trivial Units section.

Everything the phase 3.5a write path will claim is about what happens to
units that *were not* edited, and a unit-free file passes every one of those
claims trivially. Until this fixture existed, every file this repo ships had
zero units in descape/templates/blank_120x120.aoe2scenario itself, so the
byte-blob write path could only be exercised against the untracked examples/
corpus.

These tests are about the fixture itself: that its generator is deterministic,
that the file on disk still matches what the generator produces, and that its
raw on-disk bytes carry the specific caption encoding the eventual normalizer
(descape/unit_model.py, not yet written) will depend on. Anything testing the
write path *through* the fixture belongs in a later tests/test_units_write_path.py.

AoE2FileSection.get_data_as_bytes() recomputes live from current retriever
values -- it is not a cache of the bytes a unit was parsed from. So "the raw
bytes a unit was written with" has to come from slicing loaded.decompressed_body
directly, walked with each unit's own byte_length starting at
loaded.units_block_offset. _raw_unit_slices() below does that walk once; every
test here builds on it rather than re-deriving the offset arithmetic.
"""

from __future__ import annotations

import struct
from pathlib import Path

import conftest

from descape.scenario_io import load_map_and_units

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _generator():
    return conftest.load_verify_module("gen_units_fixture")


def _raw_unit_slices(loaded) -> list[tuple[int, object, bytes]]:
    """(player, entry, raw_bytes) for every unit, in players_units order.

    Each PlayerUnitsStruct's own byte_length covers its unit_count u32 *plus*
    every one of its units (scenario_io._verify_units_block reads unit_count
    this same way, by offset, before skipping the whole struct) -- so the walk
    steps over 4 bytes for unit_count before iterating a player's own units,
    not just from one unit's byte_length to the next.
    """
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    offset = loaded.units_block_offset
    slices = []
    for player, player_units in enumerate(players_units):
        offset += 4  # unit_count
        for entry in player_units.retriever_map["units"].data:
            raw = loaded.decompressed_body[offset : offset + entry.byte_length]
            offset += entry.byte_length
            slices.append((player, entry, raw))
    assert offset == loaded.units_section_end, "the walk did not land on units_section_end"
    return slices


def test_fixture_is_tracked_on_disk() -> None:
    assert FIXTURE_PATH.is_file(), (
        f"{FIXTURE_PATH} is missing -- regenerate it with `.venv/bin/python3 tools/gen_units_fixture.py`"
    )


def test_generator_is_deterministic() -> None:
    """Two runs produce identical bytes. Not a given: the generator goes
    through the library's own serializer, and anything order-dependent in
    there would make the committed fixture drift on every regeneration."""
    gen = _generator()
    assert gen.build_fixture_bytes() == gen.build_fixture_bytes()


def test_committed_fixture_matches_its_generator() -> None:
    """The file in git is what the generator produces now. A failure here
    means either the generator changed or the fixture was edited by hand;
    either way, regenerate rather than adjusting this assertion."""
    assert FIXTURE_PATH.read_bytes() == _generator().build_fixture_bytes()


def test_fixture_passes_its_own_verification() -> None:
    """The generator's reload-through-the-real-loader check, run against the
    committed file rather than against a freshly written one."""
    _generator().verify_fixture(FIXTURE_PATH)


def test_fixture_carries_the_content_the_write_path_tests_need() -> None:
    """Each assertion corresponds to a case a future write-path test would
    silently stop covering if the fixture's contents were trimmed."""
    gen = _generator()
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = loaded.unit_manager
    assert loaded.units_write_supported
    assert loaded.number_of_unit_sections == 9

    all_units = [u for units in manager.units for u in units]
    assert len(all_units) == gen.UNIT_COUNT

    # Both rotation encodings AGENTS.md names for the Wall family must survive
    # untouched: a plain doodad-variant integer (trees) and the k*2pi/5
    # radian form. Neither is an angle -- a write path must never touch them.
    # rotation is stored as f32, so the round trip loses precision relative to
    # the python float the generator started from -- compare against that same
    # f32 rounding, not the original double.
    wall = next(u for u in all_units if u.reference_id == gen._REF_WALL)
    assert wall.rotation == struct.unpack("<f", struct.pack("<f", gen._WALL_ROTATION_RADIANS))[0]
    trees = [u for u in all_units if u.unit_const in (gen._UNIT_CONST_TREE_OAK, gen._UNIT_CONST_TREE_PINE)]
    assert {t.rotation for t in trees} == {gen._TREE_OAK_ROTATION, gen._TREE_PINE_ROTATION}

    # A real, on-map building -- exercises batch_api.is_building() against a
    # fixture the default tier actually runs, not only the corpus tier.
    house = next(u for u in all_units if u.reference_id == gen._REF_HOUSE)
    assert house.unit_const == gen._UNIT_CONST_HOUSE

    # The garrison and non-empty-caption hooks stage 2's model tests will need.
    villager = next(u for u in all_units if u.reference_id == gen._REF_VILLAGER_P1)
    assert villager.garrisoned_in_id == gen._REF_HOUSE
    captioned = next(u for u in all_units if u.reference_id == gen._REF_ARCHER_P2)
    assert captioned.caption_string == gen.NON_EMPTY_CAPTION

    # The reference_id gap and the next_unit_id_to_place headroom (plan
    # finding 8): a naive len(units)-based add path would collide with these.
    reference_ids = {u.reference_id for u in all_units}
    assert 202 not in reference_ids
    assert min(reference_ids) < 202 < max(reference_ids)


def test_every_unit_raw_slice_matches_its_game_style_reserialization() -> None:
    """The structural check the fixture exists to pass: the byte range each
    unit actually occupies on disk, sliced independently of any retriever
    re-serialization, equals tools/_fixture_bytes.py's own game-style
    reserialization of that same unit.

    Not circular despite both sides tracing back to the same generator: this
    verifies the *splice landed at the right offset* (units_block_offset and
    each entry's byte_length correctly bound the players_units array after a
    real save/compress/reload round trip) and that byte_length bookkeeping
    matches what was actually written. A generator bug that spliced at the
    wrong offset, or a byte_length miscount, would fail this even though both
    sides "agree" about caption encoding.
    """
    from _fixture_bytes import _game_style_bytes

    loaded = load_map_and_units(FIXTURE_PATH)
    for player, entry, raw in _raw_unit_slices(loaded):
        assert raw == _game_style_bytes(entry), f"player {player} reference_id {entry.retriever_map['reference_id'].data}"


def test_empty_captions_are_length_zero_on_disk() -> None:
    """The measured fact (158,394 units across the real 20-file corpus, plan
    finding 2): the game writes an empty unit caption as a bare length-0 str32
    field, never the library's length-1-plus-NUL form. Read via the retriever
    (caption_string's own 4-byte length prefix), not by pattern-matching
    trailing bytes -- a caption whose *content* happened to end the same way
    would make a blind byte match lie.
    """
    gen = _generator()
    loaded = load_map_and_units(FIXTURE_PATH)
    for player, entry, raw in _raw_unit_slices(loaded):
        if entry.retriever_map["reference_id"].data == gen._REF_ARCHER_P2:
            continue  # the one deliberately non-empty caption; see the test below
        offset_in_entry = sum(
            r.get_data_as_bytes().__len__() for name, r in entry.retriever_map.items() if name != "caption_string"
        )
        length_prefix = int.from_bytes(raw[offset_in_entry : offset_in_entry + 4], "little", signed=True)
        assert length_prefix == 0, f"reference_id {entry.retriever_map['reference_id'].data}: caption length prefix {length_prefix}, expected 0"
        assert len(raw) == offset_in_entry + 4, "a length-0 str32 field is exactly its 4-byte prefix, no trailing byte"


def test_non_empty_caption_raw_encoding_is_an_assumption_not_a_measurement() -> None:
    """Settles nothing about what AoE2:DE itself writes for a non-empty unit
    caption -- that is genuinely unmeasured; no corpus unit has one. This fixture's bytes
    are produced by tools/_fixture_bytes.py's _game_style_bytes(), which only
    special-cases the *empty* string; a non-empty one passes through in the
    library's own form unchanged. So this assertion is the generator checking
    itself, not independent evidence.

    It exists anyway so the assumption is visible and singular: if real
    evidence about the game's non-empty-caption encoding ever surfaces, this
    is the one place to update, and descape/unit_model.py's construction gate
    (stage 2.2) is the actual safety net -- a file where the assumption is
    wrong fails closed (UnitEditsUnavailableError) rather than silently
    corrupting captions.
    """
    gen = _generator()
    loaded = load_map_and_units(FIXTURE_PATH)
    text = gen.NON_EMPTY_CAPTION
    # Library form (add_str_trail): a 4-byte length prefix of len(text) + 1,
    # covering the text plus its trailing NUL.
    expected_tail = (len(text) + 1).to_bytes(4, "little", signed=True) + text.encode("utf-8") + b"\x00"
    for player, entry, raw in _raw_unit_slices(loaded):
        if entry.retriever_map["reference_id"].data != gen._REF_ARCHER_P2:
            continue
        assert raw.endswith(expected_tail), f"raw tail {raw[-len(expected_tail):]!r} != assumed {expected_tail!r}"
        return
    raise AssertionError("the non-empty-caption fixture unit was not found")
