"""Phase 3.5a's write path: descape/unit_model.py's serialize() plus the units
branch in descape/scenario_write.py (_assemble_body, _patch_next_unit_id).

The claims under test, mirroring tests/test_trigger_write_path.py's own
ordering:

1. **Containment.** A document with no unit edits writes exactly the bytes it
   wrote before phase 3.5a existed.
2. **Locality.** Moving one unit changes that unit's bytes and nothing else.
3. **Gates.** units_write_supported is re-checked at write time, and the
   tightened Map Options boundary refuses (never silently drops) a patch
   landing inside the Units region.
4. **Semantic round-trip.** Write, reload through the real loader, and
   confirm every edit landed -- including next_unit_id_to_place bookkeeping
   and rotation's verbatim pass-through (AGENTS.md's hard rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from conftest import differing_ranges
from descape.option_fields import specs_for
from descape.options_model import field_offsets
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_TREE_OAK = 100
_REF_TREE_PINE = 101
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301

_WALL_ROTATION_RADIANS = 2 * (2 * 3.141592653589793 / 5)


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded)


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _unit_byte_range(loaded, reference_id: int) -> tuple[int, int]:
    """(start, end) of one unit's own slice within the *assembled* Units
    section (relative to units_block_offset), by re-walking the same
    unit_count + per-unit byte_length structure serialize() produces --
    independent of unit_model.py's own internals."""
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    offset = 0
    for pu in players_units:
        offset += 4  # unit_count
        for entry in pu.retriever_map["units"].data:
            if entry.retriever_map["reference_id"].data == reference_id:
                return offset, offset + entry.byte_length
            offset += entry.byte_length
    raise AssertionError(f"reference_id {reference_id} not found")


# -- 1. containment -----------------------------------------------------------


def test_a_model_with_no_edits_writes_the_same_file_as_no_model_at_all(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)

    model = UnitEditModel(loaded)
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, units=model)

    assert with_model.read_bytes() == without.read_bytes()


def test_zero_edit_save_reproduces_the_fixture_byte_for_byte(tmp_path: Path) -> None:
    loaded, model = _open()
    out = tmp_path / "roundtrip.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_the_zero_unit_template_is_unaffected(tmp_path: Path) -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = UnitEditModel(loaded)
    assert not model.has_edits

    out = tmp_path / "blank.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == BLANK_TEMPLATE_PATH.read_bytes()


@pytest.mark.corpus
def test_zero_edit_save_is_byte_identical_across_the_corpus(scenario_path: Path, tmp_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units are not editable")
    model = UnitEditModel(loaded)
    out = tmp_path / f"{scenario_path.stem}.roundtrip{scenario_path.suffix}"
    write_scenario(loaded, out, units=model)
    assert _written_body(out) == loaded.decompressed_body


# -- 2. locality ----------------------------------------------------------


def test_moving_one_unit_touches_only_that_units_bytes(tmp_path: Path) -> None:
    loaded, model = _open()
    base_out = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base_out, units=model)

    villager = _unit(loaded, _REF_VILLAGER_P1)
    start, end = _unit_byte_range(loaded, _REF_VILLAGER_P1)
    model.set_position(villager, 40.5, 40.5, 0.0)

    edited_out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited_out, units=model)

    before = _written_body(base_out)
    after = _written_body(edited_out)
    ranges = differing_ranges(before, after)
    assert ranges, "the move produced no byte difference at all"
    window_start = loaded.units_block_offset + start
    window_end = loaded.units_block_offset + end
    for diff_start, diff_end in ranges:
        assert window_start <= diff_start and diff_end <= window_end, (
            f"diff range ({diff_start}, {diff_end}) falls outside the moved unit's own "
            f"byte window ({window_start}, {window_end})"
        )


def test_reassign_re_serializes_nothing_end_to_end(tmp_path: Path) -> None:
    """The provenance test the plan calls for (verification item 7), run
    through the real write path rather than just UnitEditModel.serialize()
    directly."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    model.reassign(wall, 1)

    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    for pu in players_units:
        for entry in pu.retriever_map["units"].data:
            entry.get_data_as_bytes = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a clean unit's blob must never re-enter the library")
            )

    out = tmp_path / "reassigned.aoe2scenario"
    write_scenario(loaded, out, units=model)  # must not raise

    reloaded = load_map_and_units(out)
    assert any(u.reference_id == _REF_WALL for u in reloaded.unit_manager.units[1])
    assert not any(u.reference_id == _REF_WALL for u in reloaded.unit_manager.units[0])


# -- 3. gates -----------------------------------------------------------------


def test_write_is_blocked_when_edits_exist_on_a_ungated_file(tmp_path: Path) -> None:
    loaded, model = _open()
    unit = _unit(loaded, _REF_VILLAGER_P1)
    model.set_position(unit, 1.5, 1.5, 0.0)
    loaded.units_write_supported = False

    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", units=model)


def test_every_mapped_option_offset_sits_before_units_block_offset() -> None:
    """Plan verification item 10: a positive assertion, not just the absence
    of a failure -- a guard nothing exercises proves nothing."""
    loaded = load_map_and_units(FIXTURE_PATH)
    offsets, all_available = field_offsets(loaded, specs_for(loaded))
    assert all_available
    assert offsets
    for spec, fo in offsets.items():
        assert fo.offset + fo.length <= loaded.units_block_offset, (
            f"{spec.field_id}: offset {fo.offset}+{fo.length} reaches into the Units region "
            f"(starts at {loaded.units_block_offset})"
        )


@pytest.mark.corpus
def test_every_mapped_option_offset_sits_before_units_block_offset_across_the_corpus(
    scenario_path: Path,
) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units_block_offset is not trustworthy")
    offsets, all_available = field_offsets(loaded, specs_for(loaded))
    if not all_available:
        pytest.skip(f"{scenario_path.name}: not every option anchor verified")
    for spec, fo in offsets.items():
        assert fo.offset + fo.length <= loaded.units_block_offset, (
            f"{scenario_path.name}: {spec.field_id} reaches into the Units region"
        )


def test_a_patch_landing_inside_the_units_region_is_refused_not_dropped(tmp_path: Path) -> None:
    """The tightened _patch_options() boundary (stage 3): a hypothetical
    option patch inside [units_block_offset, units_section_end) must raise,
    not silently land where _assemble_body()'s units branch would overwrite
    it. No real option field reaches here today (test_every_mapped_option_
    offset_sits_before_units_block_offset asserts that positively), so a real
    OptionsEditModel is used with serialize_patches() monkeypatched to
    return an out-of-bounds patch -- everything else about the model (its
    gates, its specs) stays real."""
    from descape.options_model import OptionsEditModel

    loaded = load_map_and_units(FIXTURE_PATH)
    model = OptionsEditModel(loaded)
    model.serialize_patches = lambda: [(loaded.units_block_offset, b"\x00")]

    class _DirtyOptions:
        has_edits = True
        has_diplomacy_edits = False
        has_player_edits = False
        has_player_count_edit = False
        specs = model.specs

        def serialize_patches(self):
            return model.serialize_patches()

        def header_patch(self):
            return None

    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=_DirtyOptions())


def test_assemble_body_handles_an_unparsed_triggers_section() -> None:
    """Plan verification item 11: a units-only edit on a document that has
    never called parse_triggers() must succeed even though
    triggers_section_end is still -1."""
    loaded, model = _open()
    assert loaded.triggers_section_end == -1
    unit = _unit(loaded, _REF_VILLAGER_P1)
    model.set_position(unit, 1.5, 1.5, 0.0)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        write_scenario(loaded, Path(d) / "out.aoe2scenario", units=model)  # must not raise


# -- 4. semantic round-trip ----------------------------------------------------


def test_placed_moved_and_reassigned_units_all_read_back(tmp_path: Path) -> None:
    loaded, model = _open()

    moved = _unit(loaded, _REF_VILLAGER_P1)
    model.set_position(moved, 60.5, 61.5, 3.0)

    wall = _unit(loaded, _REF_WALL)
    model.reassign(wall, 1)

    new_unit = model.add(player=2, unit_const=83, x=15.5, y=16.5, z=0.0, rotation=2.5)
    new_ref = new_unit.reference_id

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, units=model)
    reloaded = load_map_and_units(out)

    got_moved = _unit(reloaded, _REF_VILLAGER_P1)
    assert (got_moved.x, got_moved.y, got_moved.z) == (60.5, 61.5, 3.0)

    assert any(u.reference_id == _REF_WALL for u in reloaded.unit_manager.units[1])
    assert not any(u.reference_id == _REF_WALL for u in reloaded.unit_manager.units[0])

    got_new = next(u for u in reloaded.unit_manager.units[2] if u.reference_id == new_ref)
    assert (got_new.x, got_new.y, got_new.unit_const) == (15.5, 16.5, 83)

    next_id = int.from_bytes(reloaded.decompressed_body[0:4], "little")
    assert next_id > new_ref


def test_next_unit_id_is_untouched_when_nothing_was_added(tmp_path: Path) -> None:
    loaded, model = _open()
    original_next_id = int.from_bytes(loaded.decompressed_body[0:4], "little")
    unit = _unit(loaded, _REF_VILLAGER_P1)
    model.set_position(unit, 1.5, 1.5, 0.0)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, units=model)
    reloaded = load_map_and_units(out)
    assert int.from_bytes(reloaded.decompressed_body[0:4], "little") == original_next_id


def test_rotation_survives_both_variant_index_encodings_untouched(tmp_path: Path) -> None:
    """AGENTS.md's hard rule: rotation is a doodad-variant index for the
    trees, and a k*2pi/5-radian variant selector for the wall -- a write
    path must round-trip the raw f32 bits, not floats, for units it never
    touched. Compares against a moved *unrelated* unit's save, per plan
    verification item 13."""
    loaded, model = _open()
    unrelated = _unit(loaded, _REF_ARCHER_P1)
    model.set_position(unrelated, 5.5, 5.5, 0.0)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, units=model)
    reloaded = load_map_and_units(out)

    import struct

    oak = _unit(reloaded, _REF_TREE_OAK)
    pine = _unit(reloaded, _REF_TREE_PINE)
    wall = _unit(reloaded, _REF_WALL)
    assert struct.pack("<f", oak.rotation) == struct.pack("<f", 7.0)
    assert struct.pack("<f", pine.rotation) == struct.pack("<f", 41.0)
    assert struct.pack("<f", wall.rotation) == struct.pack("<f", _WALL_ROTATION_RADIANS)


def test_untouched_fields_survive_a_write_reload_cycle_field_for_field(tmp_path: Path) -> None:
    """z, status, initial_animation_frame, garrisoned_in_id, caption_string_id
    on units the edit never touched -- plan verification item 14."""
    loaded, model = _open()
    unrelated = _unit(loaded, _REF_ARCHER_P1)
    model.set_position(unrelated, 5.5, 5.5, 0.0)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, units=model)
    reloaded = load_map_and_units(out)

    original_villager = _unit(loaded, _REF_VILLAGER_P1)
    got_villager = _unit(reloaded, _REF_VILLAGER_P1)
    for field in ("z", "status", "initial_animation_frame", "garrisoned_in_id", "caption_string_id"):
        assert getattr(got_villager, field) == getattr(original_villager, field), field

    original_captioned = _unit(loaded, _REF_ARCHER_P2)
    got_captioned = _unit(reloaded, _REF_ARCHER_P2)
    assert got_captioned.caption_string == original_captioned.caption_string
