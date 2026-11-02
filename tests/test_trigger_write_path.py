"""Phase 4b's write path: descape/trigger_model.py plus the triggers branch in
descape/scenario_write.py.

The claims under test, in the order the risk runs:

1. **Containment.** A document with no trigger edits writes exactly the bytes
   it wrote before phase 4b existed. This is what keeps every pre-existing
   write-path guarantee intact, and it's the one that would fail silently.
2. **Locality.** Editing one trigger changes that trigger's bytes and the two
   trigger counters, and nothing else. The whole byte-blob model exists for
   this; whole-section re-serialization fails it on every corpus file that has
   triggers (see tests/test_trigger_fixture.py's drift test).
3. **Structural correctness.** A reorder or delete remaps trigger-to-trigger
   references across the list, so the triggers that go stale are not only the
   one the user named. Splicing a stale blob back writes references that point
   at the wrong trigger, which no byte-identity check would catch.
4. **Semantic round-trip.** Write, reload through the real loader, and compare
   the trigger data field by field.

The default tier runs all of these against tests/fixtures/
triggers_120x120.aoe2scenario; the corpus tier repeats the ones that benefit
from real-world scale and version spread.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

# Bound at import, so a test that monkeypatches the library's name still reads the real links.
from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import (
    TriggerEditModel,
    TriggerEditsUnavailableError,
    block_room,
    display_order_moved_to_slot,
    display_order_with_block_inserted,
    display_order_with_copies_inserted,
    display_order_with_copy_inserted,
    entry_display_order,
    entry_order_after_insert,
    entry_order_after_removal,
    move_entries,
    moved_display_order,
    moved_display_order_block,
    moved_section_display_order,
    moved_to_section_edge,
    paste_anchor,
    remove_entries,
    section_edge_refusal,
    set_entry_order,
    would_move_entries,
)
from descape.trigger_organize import parse_tag, sections, tag_chain

import conftest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_REFERENCE_EFFECT_TYPES = (8, 9)  # ACTIVATE_TRIGGER, DEACTIVATE_TRIGGER


def _section_bytes(loaded) -> bytes:
    return loaded.decompressed_body[loaded.units_section_end : loaded.triggers_section_end]


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _reference_map(manager) -> list[list[int]]:
    return [
        [
            effect.trigger_id
            for effect in trigger.effects
            if effect.effect_type in _REFERENCE_EFFECT_TYPES
        ]
        for trigger in manager.triggers
    ]


def _differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """Maximal [start, end) spans where two equal-length buffers differ."""
    assert len(before) == len(after)
    ranges: list[tuple[int, int]] = []
    start = None
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if a != b and start is None:
            start = i
        elif a == b and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(before)))
    return ranges


# -- 1. containment ---------------------------------------------------------


def test_clean_model_serializes_the_original_section_verbatim() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    assert not model.has_edits
    assert model.serialize() == _section_bytes(loaded)


def test_a_model_with_no_edits_writes_the_same_file_as_no_model_at_all(tmp_path: Path) -> None:
    """The strongest containment check: constructing a TriggerEditModel, which
    parses and commits nothing, must not change a single byte of the output."""
    loaded = load_map_and_units(FIXTURE_PATH)
    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)

    model = TriggerEditModel(loaded)
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, triggers=model)

    assert with_model.read_bytes() == without.read_bytes()
    assert _written_body(without) == loaded.decompressed_body


def test_zero_edit_save_reproduces_the_fixture_byte_for_byte(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    out = tmp_path / "roundtrip.aoe2scenario"
    write_scenario(loaded, out, triggers=TriggerEditModel(loaded))
    assert out.read_bytes() == FIXTURE_PATH.read_bytes()


def test_the_zero_trigger_template_is_unaffected(tmp_path: Path) -> None:
    """The shipped donor has no triggers at all. Its model is constructible and
    its save is unchanged -- the degenerate case the fixture exists because of."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = TriggerEditModel(loaded)
    assert model.trigger_count == 0
    assert not model.has_edits

    out = tmp_path / "blank.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert out.read_bytes() == BLANK_TEMPLATE_PATH.read_bytes()


# -- 2. locality ------------------------------------------------------------


def test_editing_one_trigger_touches_only_that_trigger_and_the_counters(tmp_path: Path) -> None:
    """Plan verification item 5. The counters are unchanged in value here (the
    trigger count didn't change), so the *only* differing bytes should fall
    inside the edited trigger's own slice."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    original_section = _section_bytes(loaded)

    manager = model.manager()
    manager.triggers[1].name = "Fixture: armour split (edited)"
    model.mark_dirty(1)

    out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    body = _written_body(out)

    # Everything before the Triggers section is untouched.
    assert body[: loaded.units_section_end] == loaded.decompressed_body[: loaded.units_section_end]
    # So is everything after it.
    assert body[len(body) - (len(loaded.decompressed_body) - loaded.triggers_section_end) :] == (
        loaded.decompressed_body[loaded.triggers_section_end :]
    )

    new_section = body[loaded.units_section_end : len(body) - (len(loaded.decompressed_body) - loaded.triggers_section_end)]
    prefix_end = model.regions.triggers_start + len(model_blob_lengths(original_section, model)[0])
    # Trigger 0's blob is untouched, so the sections agree byte-for-byte up to
    # where trigger 1 starts.
    assert new_section[:prefix_end] == original_section[:prefix_end]
    # And the tail after trigger_data agrees too, allowing for trigger 1's own
    # length change.
    assert new_section[len(new_section) - (len(original_section) - model.regions.triggers_end) :] == (
        original_section[model.regions.triggers_end :]
    )


def model_blob_lengths(section_bytes: bytes, model: TriggerEditModel) -> list[bytes]:
    """Per-trigger slices of a section, cut at the model's own region map."""
    loaded = model.loaded
    entries = loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data or []
    slices = []
    offset = model.regions.triggers_start
    for entry in entries:
        slices.append(section_bytes[offset : offset + entry.byte_length])
        offset += entry.byte_length
    return slices


def test_editing_a_trigger_does_not_normalize_its_neighbours(tmp_path: Path) -> None:
    """The point of the blobs, stated as a test the whole-section serializer
    would fail: after editing one trigger, every other trigger's bytes are
    still exactly what was parsed, drift and all."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    original_slices = model_blob_lengths(_section_bytes(loaded), model)

    manager = model.manager()
    manager.triggers[0].name = "edited"
    model.mark_dirty(0)
    new_section = model.serialize()

    # Rebuild the new per-trigger boundaries: only trigger 0 changed length.
    offset = model.regions.triggers_start
    new_first_length = len(new_section) - len(_section_bytes(loaded)) + len(original_slices[0])
    offset += new_first_length
    for original in original_slices[1:]:
        assert new_section[offset : offset + len(original)] == original
        offset += len(original)


def test_the_trigger_header_fields_round_trip_inside_their_own_trigger(tmp_path: Path) -> None:
    """GH #141's four header fields. Re-serializing a trigger can change its
    length on its own, so the diff is against the same trigger marked dirty
    with no edit: every differing byte must fall inside its blob. Its
    untouched short_description_stid of -1 is re-serialized and must come
    back as -1."""

    def write(name: str, edit: bool) -> tuple[Path, TriggerEditModel]:
        loaded = load_map_and_units(FIXTURE_PATH)
        model = TriggerEditModel(loaded)
        trigger = model.manager().triggers[0]
        assert (trigger.short_description_stid, trigger.execute_on_load) == (-1, 1)
        if edit:
            trigger.description_stid = 43998
            trigger.description_order = 100
            trigger.execute_on_load = 0
        model.mark_dirty(0)
        out = tmp_path / name
        write_scenario(loaded, out, triggers=model)
        return out, model

    control, _ = write("control.aoe2scenario", edit=False)
    edited, model = write("edited.aoe2scenario", edit=True)

    reloaded = parse_triggers(load_map_and_units(edited)).triggers[0]
    assert (
        reloaded.description_stid,
        reloaded.short_description_stid,
        reloaded.description_order,
        reloaded.execute_on_load,
    ) == (43998, -1, 100, 0)
    control_body = _written_body(control)
    blob_start = model.loaded.units_section_end + model.regions.triggers_start
    # Trigger 0's blob as the control file itself carries it.
    control_loaded = load_map_and_units(control)
    blob_end = blob_start + len(model_blob_lengths(_section_bytes(control_loaded), TriggerEditModel(control_loaded))[0])
    ranges = _differing_ranges(control_body, _written_body(edited))
    assert ranges
    assert all(blob_start <= start and end <= blob_end for start, end in ranges), (ranges, blob_start, blob_end)


# -- 3. structural correctness ----------------------------------------------


def test_removing_a_trigger_dirties_every_trigger_whose_references_moved() -> None:
    """The trap this test exists for: remove_trigger() renumbers ids across the
    whole list and rewrites (de)activate-trigger references to match, so a
    trigger the caller never named comes back changed. Splicing its original
    bytes would write references pointing at the wrong triggers."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    before = _reference_map(model.manager())

    dirtied = model.structural_edit(lambda manager: manager.remove_trigger(0))
    after = _reference_map(model.manager())

    assert model.trigger_count == len(before) - 1
    # The references trigger is now at index 1 and its ids were remapped.
    assert after != before[1:]
    remapped = [index for index, refs in enumerate(after) if refs != before[index + 1]]
    assert remapped, "the fixture no longer exercises a reference remap"
    assert set(remapped) <= set(dirtied)
    for index in remapped:
        assert model.is_dirty(index)


def test_reordering_dirties_the_remapped_trigger_but_not_the_ones_that_only_moved() -> None:
    """TriggerStruct carries no trigger_id field of its own, so a trigger that
    merely changed position is still described by its original bytes. Only the
    ones whose references were rewritten go stale."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    before = _reference_map(model.manager())
    moved_names = [trigger.name for trigger in model.manager().triggers]

    dirtied = model.structural_edit(lambda manager: manager.reorder_triggers([1, 0, 2, 3]))

    assert [trigger.name for trigger in model.manager().triggers] == [
        moved_names[1],
        moved_names[0],
        moved_names[2],
        moved_names[3],
    ]
    referencing = [index for index, refs in enumerate(before) if refs]
    assert dirtied, "the fixture no longer exercises a reference remap"
    assert set(dirtied) <= set(referencing)
    # The two triggers that swapped places carry no references, so they keep
    # their blobs.
    assert not model.is_dirty(0)
    assert not model.is_dirty(1)


def test_moved_display_order_swaps_adjacent_slots() -> None:
    assert moved_display_order([0, 1, 2, 3], trigger_index=1, delta=1) == [0, 2, 1, 3]
    assert moved_display_order([0, 1, 2, 3], trigger_index=1, delta=-1) == [1, 0, 2, 3]
    # trigger_index is the stable list index, not the display slot -- moving
    # a display order that is already non-identity has to find it first.
    assert moved_display_order([2, 0, 3, 1], trigger_index=3, delta=-1) == [2, 3, 0, 1]


def test_moved_display_order_refuses_a_move_past_either_end() -> None:
    with pytest.raises(IndexError):
        moved_display_order([0, 1, 2, 3], trigger_index=0, delta=-1)
    with pytest.raises(IndexError):
        moved_display_order([0, 1, 2, 3], trigger_index=3, delta=1)


# GH #133: a section move. Two sections A[a1] and B[b1] with no leading run,
# then the same names behind a leading trigger.
_AB = ["--- A ---", "a1", "--- B ---", "b1"]
_LEAD_AB = ["lead", "--- A ---", "a1", "--- B ---", "b1"]


def _partition(names: list[str], order: list[int]) -> list[tuple[int | None, tuple[int, ...]]]:
    return [(s.header_index, s.member_indices) for s in sections(names, order)]


def test_a_section_move_swaps_two_equal_sections_whole() -> None:
    assert moved_section_display_order([0, 1, 2, 3], _AB, 0, 1) == [2, 3, 0, 1]
    assert moved_section_display_order([0, 1, 2, 3], _AB, 2, -1) == [2, 3, 0, 1]


def test_a_section_move_rotates_sections_of_unequal_length() -> None:
    names = ["--- A ---", "a1", "a2", "a3", "--- B ---", "b1"]
    assert moved_section_display_order(list(range(6)), names, 4, -1) == [4, 5, 0, 1, 2, 3]
    assert moved_section_display_order(list(range(6)), names, 0, 1) == [4, 5, 0, 1, 2, 3]


def test_an_empty_section_moves_as_its_header_alone() -> None:
    names = ["--- A ---", "--- B ---", "b1", "--- C ---"]
    assert moved_section_display_order([0, 1, 2, 3], names, 0, 1) == [1, 2, 0, 3]
    assert moved_section_display_order([0, 1, 2, 3], names, 3, -1) == [0, 3, 1, 2]


def test_a_member_moves_its_section_exactly_as_its_header_does() -> None:
    names = ["--- A ---", "a1", "a2", "--- B ---", "b1", "--- C ---", "c1"]
    order = [0, 2, 1, 5, 6, 3, 4]  # non-identity: C sits between A and B
    for delta in (-1, 1):
        by_header = moved_section_display_order(order, names, 5, delta)
        assert moved_section_display_order(order, names, 6, delta) == by_header
    assert moved_section_display_order(order, names, 6, 1) == [0, 2, 1, 3, 4, 5, 6]


def test_a_section_move_keeps_every_section_and_only_reorders_them() -> None:
    names = ["lead", "--- A ---", "a1", "a2", "--- B ---", "--- C ---", "c1", "c2", "c3"]
    order = [0, 5, 6, 1, 2, 3, 4, 7, 8]
    before = _partition(names, order)
    for index in (5, 6, 1, 2, 3, 4):
        for delta in (-1, 1):
            try:
                moved = moved_section_display_order(order, names, index, delta)
            except IndexError:
                continue
            assert sorted(moved) == sorted(order), "not a permutation of the input"
            after = _partition(names, moved)
            assert sorted(after, key=str) == sorted(before, key=str), "a section changed membership"
            assert after[0] == before[0], "the leading run never moves"


def test_a_section_move_refuses_the_leading_run_and_both_ends() -> None:
    order = [0, 1, 2, 3, 4]
    with pytest.raises(IndexError):
        moved_section_display_order(order, _LEAD_AB, 0, 1)  # the leading run
    with pytest.raises(IndexError):
        moved_section_display_order(order, _LEAD_AB, 2, -1)  # up into the leading run
    with pytest.raises(IndexError):
        moved_section_display_order(order, _LEAD_AB, 4, 1)  # past the end
    with pytest.raises(IndexError):
        moved_section_display_order([0, 1, 2, 3], _AB, 1, -1)  # past the start
    with pytest.raises(IndexError):
        moved_section_display_order([0, 1, 2, 3], _AB, 9, 1)  # not in the order
    with pytest.raises(ValueError):
        moved_section_display_order([0, 1, 2, 3], _AB, 0, 2)
    assert moved_section_display_order(order, _LEAD_AB, 1, 1) == [0, 3, 4, 1, 2]


def test_a_display_order_only_edit_dirties_the_model() -> None:
    """The has_edits hole this plan's Context section measured directly: a
    pure trigger_display_order permutation dirties no blob (a trigger that
    only moves keeps its original bytes) and changes no trigger identity, so
    without this term has_edits reads False for an edit that demonstrably
    changed the regenerated display-order array -- and scenario_write.py
    gates the whole splice on has_edits, so the reorder would be dropped on
    save with no error anywhere. Same defect class as the unreferenced-delete
    finding above.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    assert not model.has_edits

    model.structural_edit(lambda manager: setattr(
        manager, "trigger_display_order", moved_display_order(
            list(manager.trigger_display_order), trigger_index=0, delta=1
        )
    ))

    assert model.has_edits, "a display-order-only edit left the model reading as clean"


def test_flipping_only_display_order_changes_exactly_one_range(tmp_path: Path) -> None:
    """Mirrors test_options_write_path.py's
    test_flipping_only_exec_order_changes_exactly_one_byte for the other axis
    this reorder work touches -- locality, not the exec-order flag's specific
    single-byte shape. A Move Down on the first trigger swaps display slots 0
    and 1, each a little-endian u32; since both values are under 256, only
    each slot's low byte actually differs, and the two slots are 3 bytes apart
    in the packed array, so this is two 1-byte ranges rather than one
    contiguous span (measured, not assumed -- an earlier draft of this test
    asserted a single 8-byte range and failed). The claim that matters is
    locality: every differing byte falls inside the display-order region and
    nothing outside it moved.
    """
    base = tmp_path / "base.aoe2scenario"
    write_scenario(load_map_and_units(FIXTURE_PATH), base, triggers=TriggerEditModel(load_map_and_units(FIXTURE_PATH)))

    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.structural_edit(lambda manager: setattr(
        manager,
        "trigger_display_order",
        moved_display_order(list(manager.trigger_display_order), trigger_index=0, delta=1),
    ))
    assert model.has_edits

    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, triggers=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges, "expected the display-order swap to change some bytes"
    region_start = loaded.units_section_end + model.regions.triggers_end
    region_end = loaded.units_section_end + model.regions.display_order_end
    for start, end in ranges:
        assert region_start <= start and end <= region_end, (
            f"range {(start, end)} falls outside the display-order region "
            f"[{region_start}, {region_end})"
        )
    assert sum(end - start for start, end in ranges) == 2, (
        "expected exactly the two low bytes of the swapped u32 slots to differ"
    )


def test_a_section_move_changes_only_the_display_order_region(tmp_path: Path) -> None:
    """GH #133's gap closer: the test above pins an adjacent swap only. A
    section move is a non-adjacent rotation, so this pins that one through
    structural_edit() + write_scenario() too. The base is the fixture saved
    as --- A --- 0: 1, 2 | --- B --- 3, and Section Up on B gives
    [3, 0, 1, 2]: all four slots change, each u32 only in its low byte, so
    four 1-byte ranges (measured)."""
    seed = load_map_and_units(FIXTURE_PATH)
    seed_model = TriggerEditModel(seed)
    for index, name in ((0, "--- A ---"), (3, "--- B ---")):
        seed_model.manager().triggers[index].name = name
        seed_model.mark_dirty(index)
    base = tmp_path / "base.aoe2scenario"
    write_scenario(seed, base, triggers=seed_model)

    loaded = load_map_and_units(base)
    model = TriggerEditModel(loaded)
    names = [t.name or "" for t in model.manager().triggers]
    assert list(model.manager().trigger_display_order) == [0, 1, 2, 3], "fixture assumption"
    model.structural_edit(lambda manager: setattr(
        manager,
        "trigger_display_order",
        moved_section_display_order(list(manager.trigger_display_order), names, 3, -1),
    ))
    assert model.has_edits
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, triggers=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    region_start = loaded.units_section_end + model.regions.triggers_end
    region_end = loaded.units_section_end + model.regions.display_order_end
    assert ranges, "expected the section move to change some bytes"
    for start, end in ranges:
        assert region_start <= start and end <= region_end, (
            f"range {(start, end)} falls outside the display-order region [{region_start}, {region_end})"
        )
    assert [end - start for start, end in ranges] == [1, 1, 1, 1]
    reloaded = parse_triggers(load_map_and_units(edited))
    assert list(reloaded.trigger_display_order) == [3, 0, 1, 2]
    assert [t.name for t in reloaded.triggers] == [t.name for t in parse_triggers(load_map_and_units(base)).triggers]


class _StubTrigger:
    """A bare identity carrier -- display_order_with_copy_inserted() only ever
    compares triggers by id(), never reads a field off them."""


def test_display_order_with_copy_inserted_translates_stale_ids_by_identity() -> None:
    """copy_trigger()'s append_after_source path renumbers every trigger_id
    (reorder_triggers -> the triggers setter), so a pre-edit display order is
    stale by the time this runs and has to be translated through object
    identity rather than trusted as indices into the new list."""
    before = [_StubTrigger() for _ in range(4)]
    before_order = [2, 0, 3, 1]  # a real permutation, not identity
    new_trigger = _StubTrigger()
    # Renumbered and shuffled, as reorder_triggers() would leave them, plus
    # the new trigger inserted right after its source (index 1 below).
    after = [before[3], before[1], new_trigger, before[0], before[2]]

    result = display_order_with_copy_inserted(before, before_order, after, source_index=1)

    # before_order named before[2], before[0], before[3], before[1] in that
    # order; their new positions are 4, 3, 0, 1 respectively, with the copy
    # inserted directly after the source's (before[1]'s) new position.
    assert result == [4, 3, 0, 1, 2]
    assert sorted(result) == list(range(len(after))), "not a permutation of the new list"


class _NamedStub:
    """An identity carrier with the one field the GH #134 helper reads."""

    def __init__(self, name: str) -> None:
        self.name = name


def _copies_inserted(names: list[str], order: list[int], sources: list[int]) -> list[str]:
    """Run display_order_with_copies_inserted() over a renumbered `after`
    list (reversed, copies appended) and return the names in display order."""
    before = [_NamedStub(n) for n in names]
    copy_of = {id(before[i]): _NamedStub(names[i] + "'") for i in sources}
    after = list(reversed(before)) + list(copy_of.values())
    result = display_order_with_copies_inserted(before, order, after, copy_of)
    assert sorted(result) == list(range(len(after))), "not a permutation of the new list"
    return [after[i].name for i in result]


def test_copies_inserted_with_one_source_equals_the_single_copy_helper() -> None:
    before = [_NamedStub(n) for n in "abcd"]
    before_order = [2, 0, 3, 1]
    copy = _NamedStub("b'")
    after = [before[3], before[1], copy, before[0], before[2]]
    single = display_order_with_copy_inserted(before, before_order, after, source_index=1)
    assert display_order_with_copies_inserted(before, before_order, after, {id(before[1]): copy}) == single


def test_copies_of_a_contiguous_run_land_as_one_block_after_it() -> None:
    """GH #134: `a, b` duplicates to `a, b, a', b'`, not `a, a', b, b'`.
    Contiguous means adjacent in display order, not in list index."""
    assert _copies_inserted(["a", "b", "c", "d"], [0, 1, 2, 3], [1, 2]) == ["a", "b", "c", "b'", "c'", "d"]
    assert _copies_inserted(["a", "b", "c", "d"], [3, 1, 0, 2], [0, 1]) == ["d", "b", "a", "b'", "a'", "c"]


def test_scattered_copies_each_land_after_their_own_source() -> None:
    assert _copies_inserted(["a", "b", "c", "d"], [0, 1, 2, 3], [0, 2]) == ["a", "a'", "b", "c", "c'", "d"]


def test_a_divider_run_lands_after_its_section_end() -> None:
    """A lone header's copy goes past its members, so it heads an empty
    section of its own rather than taking the original's members."""
    names = ["--- A ---", "a1", "a2", "--- B ---", "b1"]
    assert _copies_inserted(names, [0, 1, 2, 3, 4], [0]) == ["--- A ---", "a1", "a2", "--- A ---'", "--- B ---", "b1"]
    assert _copies_inserted(names, [0, 1, 2, 3, 4], [0, 1, 2]) == [
        "--- A ---", "a1", "a2", "--- A ---'", "a1'", "a2'", "--- B ---", "b1",
    ]


def test_a_plain_run_at_the_same_section_end_keeps_its_copies_first() -> None:
    names = ["--- A ---", "a1", "a2", "--- B ---"]
    assert _copies_inserted(names, [0, 1, 2, 3], [0, 2]) == ["--- A ---", "a1", "a2", "a2'", "--- A ---'", "--- B ---"]


def test_paste_anchor_moves_only_a_divider_block_to_the_section_end() -> None:
    names = ["lead", "--- A ---", "a1", "a2", "--- B ---"]
    order = [0, 1, 2, 3, 4]
    assert paste_anchor(names, order, 2, False) == 2
    assert paste_anchor(names, order, None, True) is None
    assert paste_anchor(names, order, 2, True) == 3
    assert paste_anchor(names, order, 1, True) == 3, "a header anchors past its own members"
    assert paste_anchor(names, order, 0, True) == 0, "the leading section ends at its last trigger"
    assert paste_anchor(names, order, 4, True) == 4, "an empty section's end is its header"
    assert paste_anchor(names, [4, 0, 3, 1, 2], 0, True) == 3, "sections come from the display order"


def test_a_structural_edit_that_dirties_no_blob_still_reaches_the_splice(
    tmp_path: Path,
) -> None:
    """The gap 4b.6b's Delete button found, and the only shape that hits it.

    Removing a trigger nothing references dirties *nothing*: every survivor's
    bytes are genuinely unchanged, and TriggerStruct carries no trigger_id of
    its own, so position-independence holds. has_edits used to be a scan over
    the blobs alone, which made it False here -- and scenario_write.py gates the
    whole splice on has_edits, so the section wrote back verbatim and the
    deleted trigger reappeared on the next load. Nothing else in the suite could
    reach this: every other structural operation dirties at least one blob.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    names = [trigger.name for trigger in model.manager().triggers]
    # The references trigger points at two others; deleting *it* leaves nobody's
    # references to remap, which is what makes every remaining blob still valid.
    doomed = names.index("Fixture: references")

    dirtied = model.structural_edit(lambda manager: manager.remove_trigger(doomed))

    assert dirtied == [], "the fixture changed -- this trigger is now referenced by another"
    assert not any(model.is_dirty(i) for i in range(model.trigger_count))
    assert model.has_edits, "a deletion that dirties no blob is still an edit"

    out = tmp_path / "deleted.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    manager = parse_triggers(load_map_and_units(out))
    assert manager is not None
    assert [trigger.name for trigger in manager.triggers] == [
        name for name in names if name != "Fixture: references"
    ]


def test_undoing_a_blobless_structural_edit_puts_the_document_back_on_the_verbatim_branch(
    tmp_path: Path,
) -> None:
    """The other half of the flag: left True through an undo it would keep the
    next save re-serializing a section the document no longer edits, exactly the
    way a stranded variables_dirty does."""
    from descape.edit_history import EditHistory

    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    history = EditHistory()
    doomed = [t.name for t in model.manager().triggers].index("Fixture: references")

    model.begin_trigger_edit()
    model.structural_edit(lambda manager: manager.remove_trigger(doomed))
    record = model.commit_trigger_edit("Delete trigger", history)
    assert model.has_edits

    model.restore(record.before)
    assert not model.has_edits, "an undone deletion must return the document to verbatim"
    assert model.serialize() == _section_bytes(loaded)


def test_adding_a_trigger_marks_only_the_new_one_dirty(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    before_count = model.trigger_count

    dirtied = model.structural_edit(lambda manager: manager.add_trigger("Added by a test"))

    assert model.trigger_count == before_count + 1
    assert dirtied == [before_count]
    assert not any(model.is_dirty(i) for i in range(before_count))

    out = tmp_path / "added.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)
    manager = parse_triggers(reloaded)
    assert manager is not None
    assert [trigger.name for trigger in manager.triggers][-1] == "Added by a test"


@pytest.mark.parametrize("effect_type", [77, 78])
def test_a_new_create_object_attack_or_armor_effect_round_trips(tmp_path: Path, effect_type: int) -> None:
    """TASK-395: the library ships 77/78's armour/attack pair defaulting to [],
    which packed into quantity wrote a Triggers section that no longer parsed.
    The New effect path, saved with its defaults untouched."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.structural_edit(lambda manager: manager.add_trigger("Create object attack/armor")._add_effect(effect_type))

    out = tmp_path / "create_object_aa.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    manager = parse_triggers(load_map_and_units(out))
    assert manager is not None
    effect = manager.triggers[-1].effects[-1]
    assert (effect.effect_type, effect.armour_attack_quantity, effect.armour_attack_class) == (effect_type, 1, 0)


def test_both_trigger_counters_follow_the_trigger_count(tmp_path: Path) -> None:
    """Options.number_of_triggers and FileHeader.trigger_count, the two
    counters that live outside the Triggers section."""
    import struct

    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.structural_edit(lambda manager: manager.add_trigger("Counter check"))
    expected = model.trigger_count

    out = tmp_path / "counters.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)

    assert (
        struct.unpack_from("<I", reloaded.decompressed_body, reloaded.options_section_end - 4)[0]
        == expected
    )
    assert (
        struct.unpack_from("<I", reloaded.header_bytes, len(reloaded.header_bytes) - 4)[0] == expected
    )


def test_a_bypassed_add_is_refused_rather_than_mis_spliced() -> None:
    """Calling TriggerManager's structural API without going through
    structural_edit() leaves the blob list out of step with the trigger list.
    That must fail loudly at serialize() rather than splice a trigger's bytes
    into the wrong slot."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().add_trigger("Bypassed")

    with pytest.raises(RuntimeError, match="bypassed structural_edit"):
        model.serialize()


def test_a_structural_edit_that_raises_leaves_nothing_spliced_from_stale_bytes(
    tmp_path: Path,
) -> None:
    """A failed structural edit is not a no-op: reorder_triggers() reassigns
    trigger_id as it walks and raises partway through on a bad id, so the list
    can be left half-remapped. Nothing can tell which blobs are still valid, so
    none of them may be trusted."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)

    with pytest.raises(ValueError):
        model.structural_edit(lambda manager: manager.reorder_triggers([0, 1, 2, 99]))

    assert model.dirty_indices() == list(range(model.trigger_count))
    # And it can still save: re-serializing everything gives up the minimal
    # diff but is always correct.
    out = tmp_path / "after_failure.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)
    assert parse_triggers(reloaded) is not None


def test_a_bypassed_reorder_is_refused_rather_than_mis_spliced() -> None:
    """The case a length check alone would miss, and the worst one available:
    a bypassed reorder keeps the trigger count identical while every blob now
    points at a different trigger, so each splices real bytes into the wrong
    slot and the file loads fine and is wrong."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().reorder_triggers([1, 0, 2, 3])

    assert model.trigger_count == 4  # the length check cannot see this
    with pytest.raises(RuntimeError, match="bypassed structural_edit"):
        model.serialize()


# -- write-time coherence validator ------------------------------------------
#
# One test per _validate_coherence() guard, each building the violating state
# directly. The negative corpus test at the bottom is the guard against a
# "legacy implies identity display order" rule, which real files disprove.


def test_a_bypassed_reorder_is_refused_by_the_coherence_validator(monkeypatch) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().reorder_triggers([1, 0, 2, 3])
    calls = []
    original = TriggerEditModel._validate_coherence
    monkeypatch.setattr(
        TriggerEditModel,
        "_validate_coherence",
        lambda self, manager: (calls.append(True), original(self, manager)),
    )
    with pytest.raises(RuntimeError, match="bypassed structural_edit"):
        model.serialize()
    assert calls, "serialize() must route through _validate_coherence()"


def test_a_display_order_of_the_wrong_length_is_refused_by_count() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().trigger_display_order = [0, 1, 2]
    with pytest.raises(RuntimeError, match="has 3 entries for 4 triggers"):
        model.serialize()


def test_a_display_order_that_is_not_a_permutation_names_the_bad_indices() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().trigger_display_order = [0, 0, 1, 2]
    with pytest.raises(RuntimeError, match=r"duplicated: \[0\], missing: \[3\]"):
        model.serialize()


def test_a_pending_exec_order_on_a_file_storing_no_flag_is_refused() -> None:
    """Unreachable through set_exec_order() (its gate implies an offset), so
    the state is built by hand: no stored flag, and a pending value anyway."""
    import dataclasses

    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.regions = dataclasses.replace(model.regions, exec_order_offset=None)
    model._exec_order = 1
    with pytest.raises(TriggerEditsUnavailableError, match="stores no such flag"):
        model.serialize()


# -- 3b. content-mutating structural operations ------------------------------
#
# A 4b.1 regression, found by the 4b.3 review and fixed with the
# `content_touched=` argument. structural_edit()'s automatic dirty detection is
# a reference-signature diff, so it sees the id-remap class of change and
# nothing else. Two library methods rewrite a *pre-existing* trigger's content
# some other way, which that diff cannot see -- so without a declaration the
# edit is silently spliced away on save, the quietest possible way to lose a
# user's work.


def test_replace_player_is_lost_without_a_content_declaration() -> None:
    """Documents the trap rather than the fix. If this ever starts failing,
    automatic detection got better and the declaration became optional -- check
    that before deleting it."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    original = _section_bytes(loaded)

    assert model.structural_edit(lambda m: m.replace_player(0, 3)) == []
    assert not model.has_edits
    assert model.serialize() == original, "the edit is silently spliced away"


def test_replace_player_round_trips_when_declared(tmp_path: Path) -> None:
    """Plan verification item 8."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    original = _section_bytes(loaded)
    before = [effect.source_player for effect in model.manager().triggers[0].effects]

    assert model.structural_edit(lambda m: m.replace_player(0, 3), content_touched=[0]) == [0]
    assert model.has_edits
    assert model.is_dirty(0)
    assert model.serialize() != original

    out = tmp_path / "replaced.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = parse_triggers(load_map_and_units(out))
    after = [effect.source_player for effect in reloaded.triggers[0].effects]

    assert after != before, "the player change must survive the write"
    assert all(player == 3 for player in after if player is not None and player > 0)
    # Every other trigger still spliced verbatim.
    assert _reference_map(reloaded) == _reference_map(parse_triggers(loaded))


def test_a_content_declaration_survives_the_trigger_moving() -> None:
    """content_touched is pre-edit numbering followed by object identity, so an
    operation that both rewrites and moves a trigger still marks the right
    slot. Marking by raw index would dirty whichever trigger landed there."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)

    dirty = model.structural_edit(
        lambda m: (
            m.replace_player(0, 3),
            m.reorder_triggers([3, 2, 1, 0]),
        ),
        content_touched=[0],
    )
    moved_to = next(i for i, t in enumerate(model.manager().triggers) if t.trigger_id == 3)
    assert moved_to in dirty
    assert model.is_dirty(moved_to)


def test_an_out_of_range_content_declaration_raises() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    with pytest.raises(IndexError):
        model.structural_edit(lambda m: None, content_touched=[model.trigger_count])


# -- 4. semantic round-trip -------------------------------------------------


def _object_fields(obj) -> tuple:
    """Every field an object's own _link_list declares, plus the raw quantity
    triple for effects.

    Driven off _link_list rather than a hand-listed set of attributes so a
    library version that adds a field is compared too, instead of quietly
    falling outside the check. UnsupportedAttributeError is caught the same way
    AoE2Object.__repr__ catches it: after a depoison the classes are pristine
    rather than gated, so a field absent in this scenario version reads as
    None, but a gated one can still raise.
    """
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    # _uuid and _instance_number_history are per-load bookkeeping, not content.
    # conditions/effects are compared as their own objects below; their repr
    # embeds the per-load uuid, so including them here would compare noise.
    skip = {"_uuid", "_instance_number_history", "conditions", "effects"}
    names = [name for name in obj._get_object_attrs() if name not in skip]
    # The bit-split slots behind Effect.quantity. Finding 8's drift is an int
    # -1 sentinel re-typed to f32 -1.0 in exactly one of these, which the
    # public property hides.
    names += [name for name in ("_quantity_int", "_quantity_float", "variable") if hasattr(obj, name)]

    values = []
    for name in sorted(set(names)):
        try:
            values.append((name, repr(getattr(obj, name, None))))
        except UnsupportedAttributeError:
            values.append((name, "<unsupported>"))
    return tuple(values)


def _trigger_snapshot(manager) -> list[tuple]:
    """Every _link_list-derived field of every trigger, condition and effect."""
    return [
        (
            _object_fields(trigger),
            tuple(_object_fields(condition) for condition in trigger.conditions),
            tuple(_object_fields(effect) for effect in trigger.effects),
        )
        for trigger in manager.triggers
    ]


def test_untouched_triggers_survive_a_write_reload_cycle_field_for_field(tmp_path: Path) -> None:
    """Plan verification item 6, default tier. Under the byte-blob model
    finding 8's _quantity_float retype should not happen at all for an
    untouched trigger, so this asserts the raw
    _quantity_int/_quantity_float/variable triple, not just the public
    quantity property."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    before = _trigger_snapshot(model.manager())

    manager = model.manager()
    manager.triggers[0].name = "Fixture: setup (edited)"
    model.mark_dirty(0)

    out = tmp_path / "semantic.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded = load_map_and_units(out)
    reloaded_manager = parse_triggers(reloaded)
    assert reloaded_manager is not None
    after = _trigger_snapshot(reloaded_manager)

    assert len(after) == len(before)
    # Every untouched trigger comes back field-for-field identical, raw
    # quantity triple included.
    assert after[1:] == before[1:]
    # The edited one differs in its own fields but not in its conditions or
    # effects, which the edit never went near.
    assert after[0][0] != before[0][0]
    assert after[0][1:] == before[0][1:]
    assert reloaded_manager.triggers[0].name == "Fixture: setup (edited)"


def test_variables_splice_verbatim_until_marked_dirty() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    assert not model.variables_dirty

    model.mark_dirty(0)
    section = model.serialize()
    original = _section_bytes(loaded)
    assert section[len(section) - (len(original) - model.regions.unknown_bytes_end) :] == (
        original[model.regions.unknown_bytes_end :]
    )


def test_adding_a_variable_is_noticed_even_though_no_trigger_changed(tmp_path: Path) -> None:
    """Variables live outside the trigger list, so the blob reconciliation
    cannot see them. Without an explicit check, adding one leaves has_edits
    False, the write takes the verbatim-tail branch, and the variable silently
    vanishes."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)

    dirtied = model.structural_edit(lambda manager: manager.add_variable("second_var", 1))

    assert dirtied == [], "adding a variable should not dirty any trigger"
    assert model.variables_dirty
    assert model.has_edits

    out = tmp_path / "variables.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)
    manager = parse_triggers(reloaded)
    assert manager is not None
    assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var"), (1, "second_var")]

    # Everything before the variable block is still spliced verbatim: no
    # trigger was edited, so no trigger's bytes may have moved through the
    # library.
    original = _section_bytes(loaded)
    rewritten = reloaded.decompressed_body[
        reloaded.units_section_end : reloaded.triggers_section_end
    ]
    assert rewritten[: model.regions.unknown_bytes_end] == original[: model.regions.unknown_bytes_end]


def test_removing_a_variable_shrinks_the_block_it_writes(tmp_path: Path) -> None:
    """The other half of 4b.6c, and the half with no library method behind it.

    TriggerManager has add_variable() but no remove, so removal is a plain
    manager.variables list mutation inside structural_edit(). That leaves the
    rewritten block depending on the library's own link push shrinking both
    number_of_variables and variable_data together. Nothing checks the two
    against each other at write time, and there is no variables equivalent of
    _check_alignment(), so a count that stopped following the list would be
    caught here or not at all.

    Removal renumbers nothing, so the ids that survive keep theirs and the
    removed one becomes a gap. The middle variable is the one removed for
    exactly that reason: dropping the last would not tell a shrunk list apart
    from a truncated one.
    """
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.structural_edit(lambda manager: manager.add_variable("second_var", 1))
    model.structural_edit(lambda manager: manager.add_variable("third_var", 2))

    dirtied = model.structural_edit(lambda manager: manager.variables.remove(manager.variables[1]))

    assert dirtied == [], "removing a variable should not dirty any trigger"
    assert model.variables_dirty
    assert model.has_edits

    out = tmp_path / "variable_removed.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)
    manager = parse_triggers(reloaded)
    assert manager is not None
    assert [(v.variable_id, v.name) for v in manager.variables] == [(0, "fixture_var"), (2, "third_var")]


def test_removing_every_variable_writes_an_empty_block(tmp_path: Path) -> None:
    """The edge the shrink path is most likely to get wrong: a zero-length
    variable_data with a zero count, and the tail after it still spliced at the
    right offset. A block that kept its old count here would have the reload
    read the tail's first bytes as a variable name."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)

    model.structural_edit(lambda manager: manager.variables.clear())

    assert model.variables_dirty
    out = tmp_path / "no_variables.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = load_map_and_units(out)
    manager = parse_triggers(reloaded)
    assert manager is not None
    assert list(manager.variables) == []
    # The tail still lands where it should: a misplaced variable block would
    # desynchronize everything after it, and the triggers are the readable
    # part of that.
    assert len(manager.triggers) == len(parse_triggers(load_map_and_units(FIXTURE_PATH)).triggers)


# -- gates ------------------------------------------------------------------


def test_a_file_whose_triggers_do_not_parse_refuses_a_model(tmp_path: Path) -> None:
    """Stand-in for the 1.54/trigger-3.9 set, which isn't in the default tier:
    a scenario whose parse was already recorded as unsupported."""
    loaded = load_map_and_units(FIXTURE_PATH)
    loaded.trigger_read_supported = False
    loaded._trigger_manager = None

    with pytest.raises(TriggerEditsUnavailableError):
        TriggerEditModel(loaded)


def test_a_triggers_section_that_runs_past_its_end_reads_none() -> None:
    """The 1.54/trigger-3.9 set's failure shape: the parse asks for more
    bytes than remain (ValueError on the 0.9.3 pin, EndOfFileError on 0.9.4).
    A trigger count of 0x10000000 reproduces it; parse_triggers() must degrade, not raise."""
    import struct

    loaded = load_map_and_units(FIXTURE_PATH)
    tail = bytearray(loaded.trigger_tail)
    struct.pack_into("<i", tail, 9, 0x10000000)  # after f64 version + s8 instruction start
    loaded.trigger_tail = bytes(tail)
    assert parse_triggers(loaded) is None
    assert loaded.trigger_read_supported is False


def test_a_file_that_fails_the_alignment_gate_refuses_a_model() -> None:
    """Stand-in for v1.36/1.37, where the library models no Files section and
    the walk leaves a large remainder unconsumed."""
    loaded = load_map_and_units(FIXTURE_PATH)
    assert parse_triggers(loaded) is not None
    loaded.trigger_write_supported = False

    with pytest.raises(TriggerEditsUnavailableError, match="alignment gate"):
        TriggerEditModel(loaded)


def test_write_is_blocked_when_edits_exist_on_an_ungated_file(tmp_path: Path) -> None:
    """The gate has to hold at the write boundary too, not only at model
    construction -- trigger_write_supported can only be trusted after a parse,
    and a UI that mis-gates would otherwise splice at an untrusted offset."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.mark_dirty(0)
    loaded.trigger_write_supported = False

    with pytest.raises(WriteBlockedError, match="alignment gate"):
        write_scenario(loaded, tmp_path / "blocked.aoe2scenario", triggers=model)


# -- corpus tier ------------------------------------------------------------


def test_clean_serialize_is_verbatim_across_the_corpus(scenario_path: Path) -> None:
    """Plan verification items 1, 2 and 4 against real files: the alignment
    gate decides which files are editable, per-trigger byte lengths account for
    exactly the trigger_data region (TriggerEditModel refuses to build
    otherwise), and an unedited model reproduces the section verbatim."""
    loaded = load_map_and_units(scenario_path)
    if parse_triggers(loaded) is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    if not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: fails the alignment gate, editing is correctly blocked")

    model = TriggerEditModel(loaded)
    assert model.serialize() == _section_bytes(loaded)


@pytest.mark.corpus
@pytest.mark.parametrize(
    "name", ["atilla_1_scn_resaved", "R4_LeLoi_4", "F7_2_Dos Pilas (648)"]
)
def test_a_legacy_file_with_a_custom_display_order_serializes_cleanly(name: str) -> None:
    """The measurement that ruled out a "legacy implies identity display
    order" coherence rule. If this starts failing, someone added it."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this corpus")
    loaded = load_map_and_units(path)
    if parse_triggers(loaded) is None or not loaded.trigger_write_supported:
        pytest.skip(f"{path.name}: triggers are not writable")
    _check_legacy_custom_order_serializes_cleanly(loaded)


def _check_legacy_custom_order_serializes_cleanly(loaded) -> None:
    from descape.trigger_model import exec_order_value

    model = TriggerEditModel(loaded)
    order = list(model.manager().trigger_display_order)
    assert exec_order_value(loaded) == 1, "assumption: legacy mode"
    assert order != sorted(order), "assumption: non-identity display order"
    assert model.serialize() == _section_bytes(loaded)


@pytest.mark.test_map
def test_the_test_maps_legacy_custom_display_order_serializes_cleanly() -> None:
    loaded = load_map_and_units(conftest.TEST_MAP_PATH)
    assert parse_triggers(loaded) is not None and loaded.trigger_write_supported, "test map assumption: writable"
    _check_legacy_custom_order_serializes_cleanly(loaded)


def test_display_order_survives_an_unrelated_trigger_save_across_the_corpus(
    scenario_path: Path, tmp_path: Path
) -> None:
    """Mirrors test_options_write_path.py's
    test_exec_order_survives_an_unrelated_trigger_edit_across_the_corpus for
    the other axis this reorder work touches: constructs a real
    TriggerEditModel, makes a genuinely unrelated field edit, and saves
    through it -- not a bare passthrough save, which proves nothing about
    whether an edit disturbs display order.

    Baseline this rests on: zero-edit serialize() is byte-verbatim on all
    six non-identity corpus
    files, so display order was never at risk from an *unedited* model. The
    risk is a *trigger* edit rewriting it as an unintended side effect.
    """
    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: triggers are not editable")
    if not manager.triggers:
        pytest.skip(f"{scenario_path.name}: no triggers to edit")
    before_order = list(manager.trigger_display_order)

    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[0]
    with_edit = trigger.name + " (edited)"
    trigger.name = with_edit
    model.mark_dirty(0)

    out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded = parse_triggers(load_map_and_units(out))
    assert reloaded is not None
    assert reloaded.triggers[0].name == with_edit, f"{scenario_path.name}: the edit itself was lost"
    assert list(reloaded.trigger_display_order) == before_order, (
        f"{scenario_path.name}: an unrelated field edit disturbed display order"
    )


def test_a_new_condition_and_effect_match_each_files_own_vocabulary(
    scenario_path: Path, tmp_path: Path
) -> None:
    """4b.6b's picker adds an entry by type id, and the library builds it from
    its *module-level* default_attributes, which _initialise_version_dependencies
    rewrites on every load. That makes correctness here a property of the
    version spread, not of one fixture: this asserts the new entry matches the
    vocabulary JSON for the file's own scenario version, across 1.37 through
    1.58, and that it then survives the splice.
    """
    from descape import library_compat

    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: triggers are not editable")
    if not manager.triggers:
        pytest.skip(f"{scenario_path.name}: no triggers to edit")
    if not library_compat.vocabulary_is_available(loaded.scenario_version):
        pytest.skip(f"{scenario_path.name}: no shipped vocabulary for this version")

    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[0]

    # "timer" and "send_chat": present in every shipped version, and neither is
    # one of the armour/attack effects whose list slots the library normalises.
    condition_id = next(e.id for e in vocabulary.conditions.values() if e.name == "timer")
    effect_id = next(e.id for e in vocabulary.effects.values() if e.name == "send_chat")
    trigger._add_condition(condition_id)
    trigger._add_effect(effect_id)
    model.mark_dirty(0)

    for entries, entry_id, created in (
        (vocabulary.conditions, condition_id, trigger.conditions[-1]),
        (vocabulary.effects, effect_id, trigger.effects[-1]),
    ):
        definition = entries[entry_id]
        for attribute in definition.attributes:
            assert getattr(created, attribute) == definition.default_attributes[attribute], (
                f"{scenario_path.name} (v{loaded.scenario_version}): new {definition.name}'s "
                f"{attribute} is not this version's vocabulary default"
            )

    out = tmp_path / "added.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = parse_triggers(load_map_and_units(out))
    assert reloaded is not None
    assert len(reloaded.triggers) == len(manager.triggers)
    assert reloaded.triggers[0].conditions[-1].condition_type == condition_id
    assert reloaded.triggers[0].effects[-1].effect_type == effect_id


# A real 1.54 file with trigger version 4.1 and no triggers. No examples/ file
# is 1.54 with trigger 4.x, so pass --scenario-dir at a folder holding it.
_V154_TRIGGER_41 = "Arabia_chickens.aoe2scenario"


@pytest.mark.corpus
def test_new_entries_on_a_1_54_trigger_4_1_file_reread(request, tmp_path: Path) -> None:
    """AoE2ScenarioParser 0.8.4's "writing new effects/conditions to 1.54"
    fix, measured on DEscape's write path: 0.8.3 built a new 1.54 condition
    or effect with its conditional filler s32 present, so the saved Triggers
    section no longer parsed. 0.9.3 sizes the filler off static_value_30/62."""
    import conftest

    path = conftest._scenario_dir(request.config) / _V154_TRIGGER_41
    if not path.is_file():
        pytest.skip(f"{_V154_TRIGGER_41} is not in {path.parent}")
    loaded = load_map_and_units(path)
    manager = parse_triggers(loaded)
    assert (loaded.scenario_version, loaded.trigger_version) == ("1.54", 4.1)
    assert manager is not None and loaded.trigger_write_supported and not manager.triggers

    model = TriggerEditModel(loaded)
    model.structural_edit(lambda m: m.add_trigger("New entries"))
    trigger = model.manager().triggers[0]
    trigger._add_condition(10)  # timer
    trigger._add_effect(3)  # send_chat
    trigger.conditions[0].timer = 37
    trigger.effects[0].message = "1.54 filler"
    model.mark_dirty(0)

    out = tmp_path / "added.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded_loaded = load_map_and_units(out)
    reloaded = parse_triggers(reloaded_loaded)
    assert reloaded is not None and reloaded_loaded.trigger_write_supported
    assert [(c.condition_type, c.timer) for c in reloaded.triggers[0].conditions] == [(10, 37)]
    assert [(e.effect_type, e.message) for e in reloaded.triggers[0].effects] == [(3, "1.54 filler")]


def test_retyping_an_effect_keeps_its_place_and_survives_the_write_path(
    scenario_path: Path, tmp_path: Path
) -> None:
    """GH #37's retype, across the version spread its predecessor above covers.

    Same reason that one is corpus-marked: retype_entry() builds through the
    library's module-level default_attributes, which are rewritten per load, so
    "the fresh entry is this version's send_chat" is a property of the spread
    rather than of one fixture. The entry's index and the trigger's effect
    display order both have to come out unchanged, since the whole point of
    retyping rather than Delete + New is keeping the position.
    """
    from descape import library_compat, trigger_fields

    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: triggers are not editable")
    if not library_compat.vocabulary_is_available(loaded.scenario_version):
        pytest.skip(f"{scenario_path.name}: no shipped vocabulary for this version")
    target = next((i for i, t in enumerate(manager.triggers) if t.effects), None)
    if target is None:
        pytest.skip(f"{scenario_path.name}: no trigger carries an effect")

    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    effect_id = next(e.id for e in vocabulary.effects.values() if e.name == "send_chat")
    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[target]
    before_types = [e.effect_type for e in trigger.effects]
    before_order = list(trigger.effect_order)
    if before_types[0] == effect_id:
        pytest.skip(f"{scenario_path.name}: the first effect is already a send_chat")

    carried = trigger_fields.retype_carryover(
        trigger.effects[0],
        vocabulary.effects.get(before_types[0]),
        vocabulary.effects[effect_id],
        "effect_type",
    )[0]
    dropped = trigger_fields.retype_entry(
        trigger, "effect", 0, effect_id, vocabulary.effects, "effect_type"
    )
    model.mark_dirty(target)

    fresh = trigger.effects[0]
    definition = vocabulary.effects[effect_id]
    for attribute in definition.attributes:
        if attribute in carried:
            assert getattr(fresh, attribute) == carried[attribute], (
                f"{scenario_path.name}: a carried {attribute} did not reach the fresh entry"
            )
        else:
            assert getattr(fresh, attribute) == definition.default_attributes[attribute], (
                f"{scenario_path.name} (v{loaded.scenario_version}): retyped send_chat's "
                f"{attribute} is not this version's vocabulary default"
            )
    assert not set(dropped) & set(carried), "a field was reported both carried and dropped"
    assert len(trigger.effects) == len(before_types), "a retype changed the list length"
    assert list(trigger.effect_order) == before_order, "a retype rewrote the display order"

    out = tmp_path / "retyped.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded = parse_triggers(load_map_and_units(out))
    assert reloaded is not None
    reloaded_types = [e.effect_type for e in reloaded.triggers[target].effects]
    assert reloaded_types == [effect_id, *before_types[1:]], (
        f"{scenario_path.name}: the retype did not survive the splice in place"
    )


def test_copying_a_trigger_preserves_a_custom_display_order_across_the_corpus(
    scenario_path: Path,
) -> None:
    """The already-shipped defect the reorder item's plan turned up: Copy is
    the one structural op whose library call (copy_trigger's
    append_after_source path) ends at reorder_triggers(), whose
    triggers-setter resets trigger_display_order to identity. On a file that
    ships a genuinely custom order -- 6 of the 14 parseable corpus files --
    copying a trigger through the panel would silently flatten it, and on a
    legacy-exec-order file (three of those six) silently rewrite what order
    the whole scenario executes in. Confirmed by direct measurement against
    atilla_1_scn_resaved before this fix existed: display_order collapsed
    from a genuine shuffle to plain identity.

    This asserts the fix's actual invariant: with the inserted copy's slot
    removed, the display order is *exactly* the original sequence of
    trigger objects, untouched -- not merely "still a valid permutation".
    """
    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: triggers are not editable")
    before_order = list(manager.trigger_display_order)
    if before_order == list(range(len(before_order))):
        pytest.skip(f"{scenario_path.name}: display order is already identity")

    before_triggers = list(manager.triggers)
    before_ids_in_order = [id(before_triggers[old_index]) for old_index in before_order]
    source_index = 0
    model = TriggerEditModel(loaded)

    def mutate(m):
        new_trigger = m.copy_trigger(source_index)
        m.trigger_display_order = display_order_with_copy_inserted(
            before_triggers, before_order, m.triggers, source_index
        )
        return new_trigger

    model.structural_edit(mutate)

    after_triggers = model.manager().triggers
    after_order = list(model.manager().trigger_display_order)
    before_id_set = {id(t) for t in before_triggers}
    ids_by_display_slot = [id(after_triggers[i]) for i in after_order]
    without_copy = [i for i in ids_by_display_slot if i in before_id_set]

    assert without_copy == before_ids_in_order, (
        f"{scenario_path.name}: copying a trigger disturbed the display order "
        f"of triggers the user never touched"
    )
    copy_id = next(i for i in ids_by_display_slot if i not in before_id_set)
    source_id = id(before_triggers[source_index])
    assert ids_by_display_slot[ids_by_display_slot.index(source_id) + 1] == copy_id, (
        f"{scenario_path.name}: the copy did not land directly after its source "
        f"in display order"
    )


def test_single_trigger_edit_stays_local_across_the_corpus(
    scenario_path: Path, tmp_path: Path
) -> None:
    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: triggers are not editable")
    if not manager.triggers:
        pytest.skip(f"{scenario_path.name}: no triggers to edit")

    model = TriggerEditModel(loaded)
    original_section = _section_bytes(loaded)
    slices = model_blob_lengths(original_section, model)

    manager.triggers[0].name = (manager.triggers[0].name or "") + " (edited)"
    model.mark_dirty(0)

    out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    body = _written_body(out)

    assert body[: loaded.units_section_end] == loaded.decompressed_body[: loaded.units_section_end]
    tail_length = len(loaded.decompressed_body) - loaded.triggers_section_end
    assert body[len(body) - tail_length :] == loaded.decompressed_body[loaded.triggers_section_end :]

    new_section = body[loaded.units_section_end : len(body) - tail_length]
    delta = len(new_section) - len(original_section)
    offset = model.regions.triggers_start + len(slices[0]) + delta
    original_offset = model.regions.triggers_start + len(slices[0])
    assert new_section[offset:] == original_section[original_offset:]


def _check_section_moves_round_trip(path: Path, tmp_path: Path) -> int:
    """GH #133 on a real file: move the first real section down and the last
    one up, each through structural_edit() + write_scenario(), and re-derive
    the partition independently from the reloaded file. Every section keeps
    its title and exact membership; only the two blocks trade places. Returns
    how many moves ran (0 = skipped). A leading run must refuse Section Up on
    the first real section."""
    loaded = load_map_and_units(path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{path.name}: triggers are not editable")
    names = [t.name or "" for t in manager.triggers]
    order = list(manager.trigger_display_order)
    before = sections(names, order)
    real = [p for p, s in enumerate(before) if s.header_index is not None]
    if len(real) < 2:
        pytest.skip(f"{path.name}: fewer than two sections")
    if before[0].header_index is None:
        with pytest.raises(IndexError):
            moved_section_display_order(order, names, before[real[0]].header_index, -1)

    moves = [(real[0], 1), (real[-1], -1)]
    for n, (position, delta) in enumerate(moves):
        fresh = load_map_and_units(path)
        model = TriggerEditModel(fresh)
        header = before[position].header_index
        model.structural_edit(lambda m, h=header, d=delta: setattr(
            m, "trigger_display_order", moved_section_display_order(list(m.trigger_display_order), names, h, d)
        ))
        out = tmp_path / f"section_move_{n}.aoe2scenario"
        write_scenario(fresh, out, triggers=model)
        reloaded = parse_triggers(load_map_and_units(out))
        assert [t.name or "" for t in reloaded.triggers] == names, f"{path.name}: a name changed"
        after = sections(names, list(reloaded.trigger_display_order))
        expected = list(before)
        expected[position], expected[position + delta] = expected[position + delta], expected[position]
        assert after == expected, f"{path.name}: moving section {before[position].title!r} by {delta} broke the partition"
    return len(moves)


def test_a_section_move_keeps_every_section_whole_across_the_corpus(scenario_path: Path, tmp_path: Path) -> None:
    _check_section_moves_round_trip(scenario_path, tmp_path)


@pytest.mark.corpus
@pytest.mark.parametrize(
    "name, has_leading_run",
    [("F7_3_York (865)", False), ("old-allies-final-v2", True)],
)
def test_a_section_move_on_both_leading_run_branches(name: str, has_leading_run: bool, tmp_path: Path) -> None:
    """The plan's named pair: York has 18 sections and no leading run,
    old-allies 29 and a leading run, so both refusal branches run."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this corpus")
    manager = parse_triggers(load_map_and_units(path))
    assert manager is not None
    names = [t.name or "" for t in manager.triggers]
    parts = sections(names, list(manager.trigger_display_order))
    assert (parts[0].header_index is None) == has_leading_run, "corpus assumption"
    assert _check_section_moves_round_trip(path, tmp_path) == 2


@pytest.mark.test_map
def test_a_section_move_on_the_test_maps_leading_run_branch() -> None:
    """The named pair's leading-run branch on the test map, in memory only:
    the write round trip is the map240 item of
    test_a_section_move_keeps_every_section_whole_across_the_corpus."""
    manager = parse_triggers(load_map_and_units(conftest.TEST_MAP_PATH))
    assert manager is not None
    names = [t.name or "" for t in manager.triggers]
    order = list(manager.trigger_display_order)
    before = sections(names, order)
    assert before[0].header_index is None, "test map assumption: a leading run"
    real = [p for p, s in enumerate(before) if s.header_index is not None]
    assert len(real) >= 2, "test map assumption: two real sections"
    with pytest.raises(IndexError):
        moved_section_display_order(order, names, before[real[0]].header_index, -1)
    for position, delta in ((real[0], 1), (real[-1], -1)):
        moved = moved_section_display_order(order, names, before[position].header_index, delta)
        expected = list(before)
        expected[position], expected[position + delta] = expected[position + delta], expected[position]
        assert sections(names, moved) == expected, f"moving section {before[position].title!r} by {delta}"


@pytest.mark.corpus
@pytest.mark.parametrize("name", ["F7_3_York (865)", "old-allies-final-v2"])
def test_an_edge_move_never_changes_any_sections_member_set(name: str) -> None:
    """GH #211 on the same named pair: from every section, its first, middle
    and last member and every other member go to the top and to the bottom.
    The block lands at that edge and no section gains or loses a trigger."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this corpus")
    _check_edge_moves_keep_member_sets(path)


@pytest.mark.test_map
def test_an_edge_move_never_changes_any_sections_member_set_on_the_test_map() -> None:
    # The leading run and 3 real sections, each of 4+ members: 6 unrefused moves apiece.
    assert _check_edge_moves_keep_member_sets(conftest.TEST_MAP_PATH) >= 24, "test map assumption: 4 sections"


def _check_edge_moves_keep_member_sets(path: Path) -> int:
    """Returns how many edge moves ran."""
    name = path.stem
    manager = parse_triggers(load_map_and_units(path))
    assert manager is not None
    names = [t.name or "" for t in manager.triggers]
    order = list(manager.trigger_display_order)
    parts = sections(names, order)
    before = _section_members(names, order)
    moves = 0
    for position, section in enumerate(parts):
        members = section.member_indices
        if section.header_index is not None:
            assert section_edge_refusal(order, parts, [section.header_index], True).startswith("a section header")
        if not members:
            continue
        for indices in ([members[0]], [members[len(members) // 2]], [members[-1]], list(members[::2])):
            for top in (True, False):
                reason = section_edge_refusal(order, parts, indices, top)
                if reason:
                    assert reason == "already there", (name, section.title, indices, top)
                    continue
                moved = moved_to_section_edge(order, parts, indices, top)
                assert sorted(moved) == sorted(order), "not a permutation of the input"
                after = sections(names, moved)
                assert [s.header_index for s in after] == [s.header_index for s in parts]
                assert _section_members(names, moved) == before, (name, section.title, indices, top)
                landed = after[position].member_indices
                edge = landed[: len(indices)] if top else landed[len(landed) - len(indices) :]
                assert list(edge) == [i for i in order if i in indices], (name, section.title, indices, top)
                moves += 1
    assert moves, f"{name}: no edge move ran"
    return moves


# -- the quantity cluster: an object_attributes switch must still save --------
#
# Trigger 1 is "Fixture: armour split": effect 0 is Modify Attribute on ARMOR
# (armour_attack_quantity 2, class 3), effect 1 on HIT_POINTS (quantity 45).
# Driven through the viewer's funnel, since that is where the fixup lives.

_WORK_RATE = 13


def _cluster_spec(name: str):
    from descape import trigger_fields

    if name == "quantity_float":
        return trigger_fields.FieldSpec(name, trigger_fields.FLOAT, attribute="quantity")
    return trigger_fields.FieldSpec(name, trigger_fields.INT)


class _Kept(list):
    """A list that also holds a reference keeping its source alive."""

    def __init__(self, items, keep):
        super().__init__(items)
        self.keep = keep


def _edit_and_reload(tmp_path: Path, edits, setup=None):
    """Apply (effect index, field, value) edits to trigger 1, save, reparse."""
    import conftest

    window = conftest.shown_window()
    try:
        window.load_scenario(FIXTURE_PATH)
        window.mode_combo.setCurrentText("Triggers")
        if setup is not None:
            setup(window)
        for effect_index, name, value in edits:
            window.set_entry_field(1, "effect", effect_index, _cluster_spec(name), value)
        status = window.status_log.toPlainText()
        out = tmp_path / "cluster.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    # The LoadedScenario rides along: the library's scenario store is weak, and
    # an effect whose scenario is gone raises on its first version lookup.
    reloaded_scenario = load_map_and_units(out)
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    return _Kept(reloaded.triggers[1].effects, reloaded_scenario), status


@pytest.mark.parametrize(
    "edits,effect_index,expected",
    [
        ([(0, "object_attributes", 0)], 0, {"quantity": 2}),
        ([(0, "object_attributes", _WORK_RATE)], 0, {"quantity": 2.0}),
        ([(0, "object_attributes", 9)], 0, {"armour_attack_quantity": 2, "armour_attack_class": 3}),
        ([(1, "object_attributes", 8)], 1, {"armour_attack_quantity": 45, "armour_attack_class": 0}),
        ([(1, "object_attributes", 9)], 1, {"armour_attack_quantity": 45, "armour_attack_class": 0}),
        (
            [(1, "object_attributes", _WORK_RATE), (1, "quantity_float", 2.5), (1, "object_attributes", 0)],
            1,
            {"quantity": 2},
        ),
    ],
    ids=["armor->hp", "armor->work rate", "armor->attack", "hp->armor", "hp->attack", "float 2.5->hp"],
)
def test_an_attribute_switch_saves_and_carries_the_amount(tmp_path: Path, edits, effect_index, expected) -> None:
    effects, _ = _edit_and_reload(tmp_path, edits)
    effect = effects[effect_index]
    for name, value in expected.items():
        got = getattr(effect, name)
        assert got == value and type(got) is type(value), f"{name}: {got!r}, expected {value!r}"


def test_a_by_variable_switch_onto_armor_saves(tmp_path: Path) -> None:
    def add_by_variable(window):
        window.entry_structural_edit("new", 1, "effect", -1, 79)  # MODIFY_ATTRIBUTE_BY_VARIABLE

    effects, _ = _edit_and_reload(
        tmp_path,
        [(2, "object_attributes", 0), (2, "variable", 5), (2, "object_attributes", 8)],
        setup=add_by_variable,
    )
    assert (effects[2].armour_attack_class, effects[2].variable) == (0, 5)


def test_an_attribute_switch_says_what_it_rewrote(tmp_path: Path) -> None:
    _, status = _edit_and_reload(tmp_path, [(0, "object_attributes", 0)])
    assert "set quantity to 2" in status and "cleared armour attack class" in status, status


def test_retyping_onto_a_switching_type_keeps_a_carried_armor_attribute_saveable(tmp_path: Path) -> None:
    """GH #37's retype carries object_attributes onto a fresh entry whose own
    construction populated the int slot, not the pair ARMOR makes live."""

    def retype(window):
        window.entry_structural_edit("retype", 1, "effect", 0, 104)  # MODIFY_ATTRIBUTE_FOR_CLASS

    effects, _ = _edit_and_reload(tmp_path, [], setup=retype)
    assert (effects[0].effect_type, effects[0].object_attributes) == (104, 8)


def test_a_bypassed_cluster_write_is_refused_legibly_at_save() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().triggers[1].effects[0].object_attributes = 0
    model.mark_dirty(1)
    with pytest.raises(RuntimeError, match="trigger 1 effect 0 cannot be saved"):
        model.serialize()


# -- block helpers (GH #27) ---------------------------------------------------


@pytest.mark.parametrize(
    "order,indices,delta,expected",
    [
        ([0, 1, 2, 3, 4], [2], -1, [0, 2, 1, 3, 4]),
        ([0, 1, 2, 3, 4], [1, 2], -1, [1, 2, 0, 3, 4]),
        ([0, 1, 2, 3, 4], [1, 3], -1, [1, 0, 3, 2, 4]),
        ([0, 1, 2, 3, 4], [1, 3], 1, [0, 2, 1, 4, 3]),
        # Pressed against an end: the blocked member stays, the rest close up.
        ([0, 1, 2, 3, 4], [0, 2], -1, [0, 2, 1, 3, 4]),
        ([0, 1, 2, 3, 4], [0, 1], -1, [0, 1, 2, 3, 4]),
        ([0, 1, 2, 3, 4], [3, 4], 1, [0, 1, 2, 3, 4]),
        # By list index, through a custom order.
        ([3, 2, 1, 0], [1, 0], -1, [3, 1, 0, 2]),
    ],
)
def test_moved_display_order_block(order, indices, delta, expected) -> None:
    assert moved_display_order_block(order, indices, delta) == expected


def test_moved_display_order_block_refuses_an_unknown_index() -> None:
    with pytest.raises(IndexError):
        moved_display_order_block([0, 1, 2], [5], -1)


# GH #211: several slots in one call, each pass exactly the one-slot move above.
@pytest.mark.parametrize(
    "indices,delta,expected",
    [
        ([6], -5, [0, 6, 1, 2, 3, 4, 5, 7]),
        ([1], 5, [0, 2, 3, 4, 5, 6, 1, 7]),
        # Clamped at each end.
        ([2], -5, [2, 0, 1, 3, 4, 5, 6, 7]),
        ([6], 5, [0, 1, 2, 3, 4, 5, 7, 6]),
        # A split block with room keeps its spacing.
        ([5, 7], -5, [5, 0, 7, 1, 2, 3, 4, 6]),
        # Pressed against an end it closes up, then stops.
        ([1, 3], -5, [1, 3, 0, 2, 4, 5, 6, 7]),
        ([4, 6], 5, [0, 1, 2, 3, 5, 7, 4, 6]),
        ([0, 1], -5, [0, 1, 2, 3, 4, 5, 6, 7]),
        ([3], 0, [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_moved_display_order_block_moves_several_slots(indices, delta, expected) -> None:
    order = list(range(8))
    assert moved_display_order_block(order, indices, delta) == expected
    assert order == list(range(8)), "the input order is never written"
    # The same as that many single presses.
    pressed = order
    for _ in range(abs(delta)):
        pressed = moved_display_order_block(pressed, indices, 1 if delta > 0 else -1)
    assert pressed == expected


def test_moved_display_order_block_stops_at_the_first_pass_that_changes_nothing() -> None:
    """A huge distance must cost the passes that move something, not one per slot asked for."""
    assert moved_display_order_block([3, 2, 1, 0], [0], -(10**9)) == [0, 3, 2, 1]
    assert moved_display_order_block([3, 2, 1, 0], [3, 1], 10**9) == [2, 0, 3, 1]


def test_block_room_is_the_leading_members_distance_to_that_end() -> None:
    order = [3, 2, 1, 0, 4, 5]
    assert (block_room(order, [1], -1), block_room(order, [1], 1)) == (2, 3)
    assert (block_room(order, [3], -1), block_room(order, [5], 1)) == (0, 0)
    # A contiguous block (slots 1 and 2), then a split one (slots 1 and 4).
    assert (block_room(order, [1, 2], -5), block_room(order, [1, 2], 5)) == (1, 3)
    assert (block_room(order, [4, 2], -5), block_room(order, [4, 2], 5)) == (1, 1)
    # Capped by its room, a split block moves without closing up.
    assert moved_display_order_block(order, [4, 2], -min(5, block_room(order, [4, 2], -5))) == [2, 3, 1, 4, 0, 5]
    with pytest.raises(IndexError):
        block_room(order, [1, 9], -1)
    with pytest.raises(IndexError):
        block_room(order, [], 1)


@pytest.mark.parametrize(
    ("order", "indices", "target", "expected"),
    [
        # Source below the target: the pre-move slot shifts left by one.
        ([0, 1, 2, 3, 4], [1], 4, [0, 2, 3, 1, 4]),
        # Source above the target: no shift.
        ([0, 1, 2, 3, 4], [3], 1, [0, 3, 1, 2, 4]),
        # The end of the list.
        ([0, 1, 2, 3, 4], [0], 5, [1, 2, 3, 4, 0]),
        ([0, 1, 2, 3, 4], [2], 0, [2, 0, 1, 3, 4]),
        # A block keeps display order, and straddling members shift only the
        # ones below the target.
        ([0, 1, 2, 3, 4], [3, 0], 3, [1, 2, 0, 3, 4]),
        ([0, 1, 2, 3, 4], [4, 1], 3, [0, 2, 1, 4, 3]),
        # By list index, through a custom order.
        ([3, 2, 1, 0], [2], 4, [3, 1, 0, 2]),
    ],
)
def test_display_order_moved_to_slot(order, indices, target, expected) -> None:
    assert display_order_moved_to_slot(order, indices, target) == expected


def test_display_order_moved_to_slot_is_a_no_op_onto_itself() -> None:
    assert display_order_moved_to_slot([0, 1, 2], [1], 1) == [0, 1, 2]
    assert display_order_moved_to_slot([0, 1, 2], [1], 2) == [0, 1, 2]


def test_display_order_moved_to_slot_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        display_order_moved_to_slot([0, 1, 2], [5], 0)
    with pytest.raises(IndexError):
        display_order_moved_to_slot([0, 1, 2], [1], 4)
    with pytest.raises(IndexError):
        display_order_moved_to_slot([0, 1, 2], [1], -1)


# GH #211: the top and the bottom of a trigger's own section. A leading run of
# three, section A with three members, section B with one.
_EDGE_NAMES = ["lead0", "lead1", "lead2", "--- A ---", "a0", "a1", "a2", "--- B ---", "b0"]
_EDGE_ORDER = list(range(9))


def _section_members(names: list[str], order: list[int]) -> dict[int | None, frozenset[int]]:
    return {s.header_index: frozenset(s.member_indices) for s in sections(names, order)}


@pytest.mark.parametrize(
    "indices,top,expected",
    [
        # Inside its own section, just below the divider or just above the next one.
        ([6], True, [0, 1, 2, 3, 6, 4, 5, 7, 8]),
        ([4], False, [0, 1, 2, 3, 5, 6, 4, 7, 8]),
        # The leading run: slot 0, and its bottom stops before the first divider.
        ([2], True, [2, 0, 1, 3, 4, 5, 6, 7, 8]),
        ([0], False, [1, 2, 0, 3, 4, 5, 6, 7, 8]),
        # A split selection gathers into one block, in display order.
        ([6, 4], True, [0, 1, 2, 3, 4, 6, 5, 7, 8]),
        ([6, 4], False, [0, 1, 2, 3, 5, 4, 6, 7, 8]),
        ([0, 2], False, [1, 0, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_moved_to_section_edge_stays_inside_the_section(indices, top, expected) -> None:
    parts = sections(_EDGE_NAMES, _EDGE_ORDER)
    assert section_edge_refusal(_EDGE_ORDER, parts, indices, top) == ""
    moved = moved_to_section_edge(_EDGE_ORDER, parts, indices, top)
    assert moved == expected
    assert _section_members(_EDGE_NAMES, moved) == _section_members(_EDGE_NAMES, _EDGE_ORDER)


def test_a_file_with_no_sections_is_one_section_for_an_edge_move() -> None:
    names = ["t0", "t1", "t2", "t3", "t4"]
    order = [4, 3, 2, 1, 0]
    parts = sections(names, order)
    assert moved_to_section_edge(order, parts, [1], True) == [1, 4, 3, 2, 0]
    assert moved_to_section_edge(order, parts, [3], False) == [4, 2, 1, 0, 3]
    assert moved_to_section_edge(order, parts, [0, 3], True) == [3, 0, 4, 2, 1]


@pytest.mark.parametrize(
    "indices,top,reason",
    [
        ([3], True, "a section header moves with Section Up/Down"),
        ([4, 3], False, "a section header moves with Section Up/Down"),
        ([2, 4], True, "the selection spans sections"),
        ([5, 8], False, "the selection spans sections"),
        ([4], True, "already there"),
        ([6], False, "already there"),
        # A contiguous block already at the edge, and a one-member section either way.
        ([4, 5], True, "already there"),
        ([8], True, "already there"),
        ([8], False, "already there"),
    ],
)
def test_an_edge_move_is_refused_for_a_header_a_spanning_selection_or_no_change(indices, top, reason) -> None:
    parts = sections(_EDGE_NAMES, _EDGE_ORDER)
    assert section_edge_refusal(_EDGE_ORDER, parts, indices, top) == reason
    with pytest.raises(ValueError, match=reason):
        moved_to_section_edge(_EDGE_ORDER, parts, indices, top)
    with pytest.raises(ValueError):
        section_edge_refusal(_EDGE_ORDER, parts, [], top)


def test_an_edge_move_reads_positions_from_the_order_not_from_a_stale_partition() -> None:
    """The panel's cached partition keeps its old member order after a
    one-row move patches the tree, so it may only supply membership."""
    names = ["t0", "t1", "t2", "t3", "t4"]
    stale = sections(names, [0, 1, 2, 3, 4])
    order = [1, 2, 3, 4, 0]  # trigger 0 has since moved to the bottom
    assert section_edge_refusal(order, stale, [0], False) == "already there"
    assert section_edge_refusal(order, stale, [1], True) == "already there"
    assert section_edge_refusal(order, stale, [0], True) == ""
    assert moved_to_section_edge(order, stale, [0], True) == [0, 1, 2, 3, 4]
    assert moved_to_section_edge(order, stale, [1], False) == [2, 3, 4, 0, 1]

    # Sectioned: a member moved inside A, which the stale partition still lists first.
    parts = sections(_EDGE_NAMES, _EDGE_ORDER)
    moved = [0, 1, 2, 3, 5, 6, 4, 7, 8]
    assert section_edge_refusal(moved, parts, [4], False) == "already there"
    assert moved_to_section_edge(moved, parts, [4], True) == _EDGE_ORDER


def test_display_order_with_block_inserted_lands_after_the_anchor() -> None:
    assert display_order_with_block_inserted([3, 2, 1, 0], [4, 5], 2) == [3, 2, 4, 5, 1, 0]
    assert display_order_with_block_inserted([3, 2, 1, 0], [4, 5], None) == [3, 2, 1, 0, 4, 5]


def test_a_pasted_block_round_trips_field_for_field_and_keeps_a_custom_display_order(tmp_path: Path) -> None:
    """GH #27's paste, as the funnel runs it: one import, then display order
    repaired from the pre-import value. The pasted triggers' conditions and
    effects come back identical to their sources', and the custom order with
    the block below its anchor survives the write."""
    from descape import trigger_clipboard

    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().trigger_display_order = [3, 2, 1, 0]
    block = trigger_clipboard.copy_block(model.manager(), [0, 1])
    sources = _trigger_snapshot(model.manager())

    def mutate(m):
        before_order = list(m.trigger_display_order)
        pasted = trigger_clipboard.paste_into(m, block).triggers
        new = list(range(4, 4 + len(pasted)))
        m.trigger_display_order = display_order_with_block_inserted(before_order, new, 2)

    model.structural_edit(mutate)
    out = tmp_path / "pasted.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded_scenario = load_map_and_units(out)  # kept alive: the library's store is weak
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    # Block order is display order: trigger 1 sits before 0 in [3, 2, 1, 0].
    assert list(reloaded.trigger_display_order) == [3, 2, 4, 5, 1, 0]
    after = _trigger_snapshot(reloaded)
    assert after[:4] == sources
    assert after[4][1:] == sources[1][1:] and after[5][1:] == sources[0][1:]
    assert [t.name for t in reloaded.triggers[4:]] == ["Fixture: armour split (copy)", "Fixture: setup (copy)"]


# -- GH #213: condition and effect display order -------------------------------

# The library's own heal, measured on this order: removing list index 0 gives
# [0, 3, 1, 2], which shows the four survivors in a different order.
_ORDER = [0, 3, 1, 4, 2]

# Trigger 3 is the fixture's only one with several conditions (4) and effects (7).
ENTRY_ORDER_TRIGGER = 3
ENTRY_ORDERS = {"condition": [2, 0, 3, 1], "effect": [4, 0, 6, 2, 5, 1, 3]}


def write_entry_order_fixture(tmp_path: Path) -> Path:
    """The fixture with trigger 3 saved under ENTRY_ORDERS, as a real file: the
    shipped one holds identity orders only, and an order assigned in memory
    never reaches a clean trigger's spliced blob."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[ENTRY_ORDER_TRIGGER]
    for kind, order in ENTRY_ORDERS.items():
        set_entry_order(trigger, kind, order)
    model.mark_dirty(ENTRY_ORDER_TRIGGER)
    out = tmp_path / "entry_order.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    return out


def _ordered_trigger():
    """Fixture trigger 3 under ENTRY_ORDERS, in memory, and the scenario that
    keeps it alive (the library's scenario store is weak)."""
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = parse_triggers(loaded)
    assert manager is not None
    trigger = manager.triggers[ENTRY_ORDER_TRIGGER]
    for kind, order in ENTRY_ORDERS.items():
        set_entry_order(trigger, kind, order)
    return trigger, loaded


def _shown(entries, order) -> list:
    return [entries[index] for index in order]


def _shown_ids(trigger, kind: str) -> list[int]:
    """id() of each entry in display order, read off the raw order array."""
    return [id(entry) for entry in _shown(getattr(trigger, f"{kind}s"), getattr(trigger, f"{kind}_order"))]


def test_entry_order_after_removal_keeps_the_survivors_display_order() -> None:
    entries = list("abcde")
    assert entry_order_after_removal(_ORDER, [0]) == [2, 0, 3, 1]
    for size in range(6):
        for removed in itertools.combinations(range(5), size):
            order = entry_order_after_removal(_ORDER, removed)
            kept = [entry for index, entry in enumerate(entries) if index not in removed]
            assert sorted(order) == list(range(len(kept))), removed
            assert _shown(kept, order) == [entry for entry in _shown(entries, _ORDER) if entry in kept], removed


def test_entry_order_after_insert_keeps_the_others_and_lands_the_block_at_its_slot() -> None:
    entries = list("abcde")
    for at, slot, count in itertools.product(range(6), range(6), (1, 2)):
        new = ["X", "Y"][:count]
        order = entry_order_after_insert(_ORDER, at, count, slot)
        after = entries[:at] + new + entries[at:]
        shown = _shown(after, order)
        assert sorted(order) == list(range(5 + count)), (at, slot, count)
        assert shown[slot : slot + count] == new, (at, slot, count)
        assert [entry for entry in shown if entry not in new] == _shown(entries, _ORDER), (at, slot, count)


def test_an_identity_entry_order_stays_identity() -> None:
    """No churn in a file that never used the feature."""
    identity = list(range(5))
    for size in range(6):
        for removed in itertools.combinations(range(5), size):
            assert entry_order_after_removal(identity, removed) == list(range(5 - size))
    # A paste below a row: on an identity order the list index is the display slot.
    for at in range(6):
        assert entry_order_after_insert(identity, at, 2, at) == list(range(7))


def test_the_entry_order_helpers_refuse_bad_input() -> None:
    with pytest.raises(IndexError):
        entry_order_after_insert([0, 1, 2], 4, 1, 0)
    with pytest.raises(IndexError):
        entry_order_after_insert([0, 1, 2], 0, 1, 4)
    trigger, _keep = _ordered_trigger()
    for bad in ([0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4], [0, 1, 2, 3, 4]):
        with pytest.raises(ValueError, match="not a permutation"):
            set_entry_order(trigger, "condition", bad)
    assert list(trigger.condition_order) == ENTRY_ORDERS["condition"], "a refused order must not be assigned"


def test_entry_display_order_falls_back_to_identity() -> None:
    """A stand-in with no order array (tests, the census tool), and an array
    that is not a permutation, both read as list order."""
    assert entry_display_order(SimpleNamespace(conditions=[7, 8, 9], effects=[]), "condition") == [0, 1, 2]
    assert entry_display_order(SimpleNamespace(conditions=[7, 8, 9], condition_order=[0, 0, 1]), "condition") == [0, 1, 2]
    assert entry_display_order(SimpleNamespace(conditions=[7, 8, 9], condition_order=[0, 1]), "condition") == [0, 1, 2]
    assert entry_display_order(SimpleNamespace(conditions=[7, 8, 9], condition_order=[2, 0, 1]), "condition") == [2, 0, 1]
    assert entry_display_order(SimpleNamespace(), "effect") == []


def test_the_entry_order_read_and_the_one_assigned_are_never_the_live_list() -> None:
    """The library heals its array in place (restore()'s Trap 6), so a "before"
    order that aliased it would be rewritten by the very removal it describes."""
    trigger, _keep = _ordered_trigger()
    before = entry_display_order(trigger, "condition")
    assert before == ENTRY_ORDERS["condition"] and before is not trigger.condition_order
    trigger.remove_condition(condition_index=0)
    assert list(trigger.condition_order) != before, "fixture assumption: the heal rewrote the live array"
    assert before == ENTRY_ORDERS["condition"]

    mine = [1, 0, 2]
    set_entry_order(trigger, "condition", mine)
    trigger.conditions.append(copy.deepcopy(trigger.conditions[0]))
    assert list(trigger.condition_order) == [1, 0, 2, 3], "an append lands last in display"
    assert mine == [1, 0, 2]


def test_the_librarys_own_heal_reorders_the_survivors() -> None:
    """The premise of remove_entries(). If this fails the library changed, and
    the order helpers may no longer be needed."""
    trigger, _keep = _ordered_trigger()
    before = _shown_ids(trigger, "condition")
    victim = id(trigger.conditions[0])
    trigger.remove_condition(condition_index=0)
    assert _shown_ids(trigger, "condition") != [entry for entry in before if entry != victim]


@pytest.mark.parametrize("kind", ["condition", "effect"])
def test_remove_entries_keeps_the_survivors_display_order_for_every_subset(kind: str) -> None:
    base, _keep = _ordered_trigger()
    other = "effect" if kind == "condition" else "condition"
    count = len(ENTRY_ORDERS[kind])
    for size in range(1, count + 1):
        for removed in itertools.combinations(range(count), size):
            trigger = copy.deepcopy(base)
            entries = getattr(trigger, f"{kind}s")
            before = _shown_ids(trigger, kind)
            gone = {id(entries[index]) for index in removed}
            other_before = _shown_ids(trigger, other)

            landing = remove_entries(trigger, [(kind, index) for index in removed])

            after = _shown_ids(trigger, kind)
            assert after == [entry for entry in before if entry not in gone], removed
            assert sorted(getattr(trigger, f"{kind}_order")) == list(range(count - size)), removed
            assert _shown_ids(trigger, other) == other_before, removed
            # The row that followed the first removed one, else the last row left.
            first_gone = min(slot for slot, entry in enumerate(before) if entry in gone)
            following = [entry for entry in before[first_gone:] if entry not in gone]
            expected = following[0] if following else (after[-1] if after else None)
            landed = None if landing is None else id(getattr(trigger, f"{landing[0]}s")[landing[1]])
            assert (landing is None or landing[0] == kind) and landed == expected, removed


def test_remove_entries_takes_both_kinds_and_lands_on_the_first_refs_kind() -> None:
    trigger, _keep = _ordered_trigger()
    conditions, effects = _shown_ids(trigger, "condition"), _shown_ids(trigger, "effect")
    gone = {id(trigger.conditions[3]), id(trigger.effects[0]), id(trigger.effects[6])}
    # Effect order [4, 0, 6, 2, 5, 1, 3]: 0 and 6 sit at slots 1 and 2, so slot 1 now holds old 2 (new 1).
    assert remove_entries(trigger, [("effect", 6), ("condition", 3), ("effect", 0)]) == ("effect", 1)
    assert _shown_ids(trigger, "condition") == [entry for entry in conditions if entry not in gone]
    assert _shown_ids(trigger, "effect") == [entry for entry in effects if entry not in gone]
    assert remove_entries(trigger, []) is None


def test_remove_entries_refuses_a_bad_ref_before_removing_anything() -> None:
    trigger, _keep = _ordered_trigger()
    with pytest.raises(IndexError):
        remove_entries(trigger, [("condition", 0), ("effect", 7)])
    assert (len(trigger.conditions), len(trigger.effects)) == (4, 7)
    assert {kind: list(getattr(trigger, f"{kind}_order")) for kind in ENTRY_ORDERS} == ENTRY_ORDERS


def test_move_entries_permutes_the_order_and_never_the_list() -> None:
    trigger, _keep = _ordered_trigger()
    listed = [id(entry) for entry in (*trigger.conditions, *trigger.effects)]

    # Condition order [2, 0, 3, 1]: list index 0 sits at slot 1.
    assert would_move_entries(trigger, [("condition", 0)], -1)
    assert list(trigger.condition_order) == ENTRY_ORDERS["condition"], "the gate must not write"
    assert move_entries(trigger, [("condition", 0)], -1)
    assert list(trigger.condition_order) == [0, 2, 3, 1]
    assert list(trigger.effect_order) == ENTRY_ORDERS["effect"]

    # At the top: nothing to do, and nothing written.
    assert not would_move_entries(trigger, [("condition", 0)], -1)
    assert not move_entries(trigger, [("condition", 0)], -1)
    assert list(trigger.condition_order) == [0, 2, 3, 1]

    # Both kinds, each inside its own list. Effect 4 is already first and stays.
    assert move_entries(trigger, [("condition", 3), ("effect", 4)], -1)
    assert list(trigger.condition_order) == [0, 3, 2, 1]
    assert list(trigger.effect_order) == ENTRY_ORDERS["effect"]
    assert move_entries(trigger, [("condition", 2), ("effect", 0)], 1)
    assert list(trigger.condition_order) == [0, 3, 1, 2]
    assert list(trigger.effect_order) == [4, 6, 0, 2, 5, 1, 3]

    # A block pressed against the end closes up and then stops.
    assert move_entries(trigger, [("effect", 5), ("effect", 3)], 1)
    assert list(trigger.effect_order) == [4, 6, 0, 2, 1, 5, 3]
    assert not would_move_entries(trigger, [("effect", 5), ("effect", 3)], 1)
    assert not would_move_entries(trigger, [("condition", 2), ("effect", 3)], 1)

    assert [id(entry) for entry in (*trigger.conditions, *trigger.effects)] == listed, "a move reordered a list"


def test_a_stale_entry_ref_moves_nothing() -> None:
    trigger, _keep = _ordered_trigger()
    assert not would_move_entries(trigger, [("condition", 9)], 1)
    assert not would_move_entries(trigger, [], 1)
    with pytest.raises(IndexError):
        move_entries(trigger, [("condition", 9)], 1)
    assert list(trigger.condition_order) == ENTRY_ORDERS["condition"]


def test_move_entries_five_places_and_to_the_edge_stay_inside_each_list() -> None:
    """GH #211. Conditions show as 2, 0, 3, 1 and effects as 4, 0, 6, 2, 5, 1, 3."""
    trigger, _keep = _ordered_trigger()
    listed = [id(entry) for entry in (*trigger.conditions, *trigger.effects)]

    # Five down: effect 0 goes from slot 1 to slot 6, condition 2 has three slots below it.
    refs = [("effect", 0), ("condition", 2)]
    assert would_move_entries(trigger, refs, 5)
    assert _entry_orders(trigger) == ENTRY_ORDERS, "the gate must not write"
    assert move_entries(trigger, refs, 5)
    assert _entry_orders(trigger) == {"condition": [0, 3, 1, 2], "effect": [4, 6, 2, 5, 1, 3, 0]}

    # Five up: a split block closes up against the top and stops there.
    assert move_entries(trigger, [("effect", 2), ("effect", 1)], -5)
    assert _entry_orders(trigger) == {"condition": [0, 3, 1, 2], "effect": [2, 1, 4, 6, 5, 3, 0]}
    assert not would_move_entries(trigger, [("effect", 2), ("effect", 1)], -5)

    # To the edge: each kind's refs gather at the end of their own list.
    refs = [("effect", 4), ("effect", 5), ("condition", 3)]
    assert would_move_entries(trigger, refs, 1, to_edge=True)
    assert move_entries(trigger, refs, 1, to_edge=True)
    assert _entry_orders(trigger) == {"condition": [0, 1, 2, 3], "effect": [2, 1, 6, 3, 0, 4, 5]}
    assert not would_move_entries(trigger, refs, 1, to_edge=True)
    assert not move_entries(trigger, refs, 1, to_edge=True)
    # And at the top. Effect 2 is already first, so only the conditions change.
    refs = [("condition", 2), ("effect", 2)]
    assert move_entries(trigger, refs, -1, to_edge=True)
    assert _entry_orders(trigger) == {"condition": [2, 0, 1, 3], "effect": [2, 1, 6, 3, 0, 4, 5]}
    # The one-step gate and the edge gate answer alike, both ways.
    for check, delta in itertools.product([[("effect", 2), ("effect", 1)], [("effect", 5)], [("condition", 0)]], (-1, 1)):
        assert would_move_entries(trigger, check, delta) == would_move_entries(trigger, check, delta, to_edge=True)

    assert [id(entry) for entry in (*trigger.conditions, *trigger.effects)] == listed, "a move reordered a list"


def test_a_stale_entry_ref_moves_nothing_to_the_edge_either() -> None:
    trigger, _keep = _ordered_trigger()
    assert not would_move_entries(trigger, [("condition", 9)], 1, to_edge=True)
    assert not would_move_entries(trigger, [("effect", 0), ("condition", 9)], -1, to_edge=True)
    assert not would_move_entries(trigger, [("effect", 0), ("effect", 7)], -5)
    with pytest.raises(IndexError):
        move_entries(trigger, [("condition", 9)], 1, to_edge=True)
    with pytest.raises(IndexError):
        move_entries(trigger, [("effect", 0), ("effect", 7)], -1, to_edge=True)
    assert _entry_orders(trigger) == ENTRY_ORDERS


def test_a_non_identity_entry_order_survives_the_write_path(tmp_path: Path) -> None:
    base = write_entry_order_fixture(tmp_path)
    loaded = load_map_and_units(base)
    manager = parse_triggers(loaded)
    assert manager is not None
    for index, trigger in enumerate(manager.triggers):
        for kind, custom in ENTRY_ORDERS.items():
            order = list(getattr(trigger, f"{kind}_order"))
            assert order == (custom if index == ENTRY_ORDER_TRIGGER else sorted(order)), (index, kind)

    out = tmp_path / "resaved.aoe2scenario"
    write_scenario(loaded, out, triggers=TriggerEditModel(loaded))
    assert out.read_bytes() == base.read_bytes()


def test_a_move_and_a_removal_reload_and_leave_the_other_triggers_bytes_alone(tmp_path: Path) -> None:
    base = write_entry_order_fixture(tmp_path)
    loaded = load_map_and_units(base)
    model = TriggerEditModel(loaded)
    trigger = model.manager().triggers[ENTRY_ORDER_TRIGGER]
    condition_types = [c.condition_type for c in trigger.conditions]
    effect_types = [e.effect_type for e in trigger.effects]

    assert move_entries(trigger, [("effect", 0)], 1)
    assert remove_entries(trigger, [("condition", 0)]) == ("condition", 2)
    model.mark_dirty(ENTRY_ORDER_TRIGGER)
    out = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded_scenario = load_map_and_units(out)  # kept alive: the library's store is weak
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    after = reloaded.triggers[ENTRY_ORDER_TRIGGER]
    assert list(after.effect_order) == [4, 6, 0, 2, 5, 1, 3]
    assert [e.effect_type for e in after.effects] == effect_types
    # Shown as 2, 0, 3, 1; without 0 that is 2, 3, 1, renumbered 1, 2, 0.
    assert list(after.condition_order) == [1, 2, 0]
    assert [c.condition_type for c in after.conditions] == condition_types[1:]
    assert _saved_blobs(out)[:ENTRY_ORDER_TRIGGER] == _saved_blobs(base)[:ENTRY_ORDER_TRIGGER]


# The same, through the viewer's funnels, since that is what the buttons call.


def _entry_types(trigger) -> dict[str, list[int]]:
    return {kind: [getattr(entry, f"{kind}_type") for entry in getattr(trigger, f"{kind}s")] for kind in ENTRY_ORDERS}


def _entry_orders(trigger) -> dict[str, list[int]]:
    return {kind: list(getattr(trigger, f"{kind}_order")) for kind in ENTRY_ORDERS}


def _shown_fields(trigger) -> dict[str, list[tuple]]:
    """Each kind's entries as their fields, in display order."""
    return {
        kind: [_object_fields(getattr(trigger, f"{kind}s")[index]) for index in getattr(trigger, f"{kind}_order")]
        for kind in ENTRY_ORDERS
    }


def _funnel_edit_and_reload(tmp_path: Path, operate):
    """Run `operate(window)` with the ENTRY_ORDERS fixture's trigger 3 current,
    save and reparse. Returns that trigger's _shown_fields() before and after,
    and its reloaded types and orders, checked against the live ones."""
    import conftest

    base = write_entry_order_fixture(tmp_path)
    window = conftest.shown_window()
    try:
        window.load_scenario(base)
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_panel.select_trigger(ENTRY_ORDER_TRIGGER)
        before = _shown_fields(window.trigger_panel._manager().triggers[ENTRY_ORDER_TRIGGER])
        operate(window)
        assert window.trigger_edits is not None, "the edit never reached the model"
        live = window.trigger_edits.manager().triggers[ENTRY_ORDER_TRIGGER]
        after, types, orders = _shown_fields(live), _entry_types(live), _entry_orders(live)
        out = tmp_path / "funnel.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded_scenario = load_map_and_units(out)  # kept alive: the library's store is weak
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    trigger = reloaded.triggers[ENTRY_ORDER_TRIGGER]
    assert (_entry_types(trigger), _entry_orders(trigger)) == (types, orders), "the edit did not survive the write path"
    assert _saved_blobs(out)[:ENTRY_ORDER_TRIGGER] == _saved_blobs(base)[:ENTRY_ORDER_TRIGGER]
    return before, after, types, orders


@pytest.mark.parametrize("op", ["delete", "cut"])
def test_a_delete_or_cut_through_the_funnel_keeps_the_survivors_display_order(tmp_path: Path, op: str) -> None:
    refs = [("effect", 0), ("condition", 3), ("effect", 4)]
    blocks = []

    def operate(window) -> None:
        window.entry_structural_edit(op, ENTRY_ORDER_TRIGGER, "effect", refs, -1)
        blocks.append(window._entry_clipboard)

    before, after, _types, orders = _funnel_edit_and_reload(tmp_path, operate)
    for kind, order in ENTRY_ORDERS.items():
        gone = {index for ref_kind, index in refs if ref_kind == kind}
        assert after[kind] == [fields for fields, index in zip(before[kind], order, strict=True) if index not in gone]
    # Effects showed as 4, 0, 6, 2, 5, 1, 3; the heal alone gives [4, 0, 2, 1, 3].
    assert orders == {"condition": [2, 0, 1], "effect": [4, 1, 3, 0, 2]}
    if op == "cut":
        # The clipboard holds them as they were shown: effect 4 above effect 0.
        assert [e.effect_type for e in blocks[0].effects] == [15, 56]
    else:
        assert blocks[0] is None


def test_a_move_through_the_funnel_reloads_and_leaves_the_lists_alone(tmp_path: Path) -> None:
    """Move Up/Down's funnel. The helper also checks that triggers 0 to 2,
    which the move never names, keep their bytes."""
    refs = [("effect", 0), ("effect", 6), ("condition", 3)]
    before, after, types, orders = _funnel_edit_and_reload(
        tmp_path, lambda w: w.entry_move(ENTRY_ORDER_TRIGGER, refs, 1)
    )
    assert orders == {"condition": [2, 0, 1, 3], "effect": [4, 2, 0, 6, 5, 1, 3]}
    assert types == {"condition": [10, 5, 6, 6], "effect": [56, 19, 11, 14, 15, 12, 19]}, "the lists never move"
    for kind, order in ENTRY_ORDERS.items():
        assert after[kind] == [before[kind][order.index(index)] for index in orders[kind]]


@pytest.mark.parametrize(
    "delta,to_edge,expected",
    [
        (-1, True, {"condition": [3, 2, 0, 1], "effect": [0, 6, 4, 2, 5, 1, 3]}),
        (1, True, {"condition": [2, 0, 1, 3], "effect": [4, 2, 5, 1, 3, 0, 6]}),
        (5, False, {"condition": [2, 0, 1, 3], "effect": [4, 2, 5, 1, 3, 0, 6]}),
        (-5, False, {"condition": [3, 2, 0, 1], "effect": [0, 6, 4, 2, 5, 1, 3]}),
    ],
    ids=["top", "bottom", "five down", "five up"],
)
def test_a_far_move_through_the_funnel_reloads_with_its_order(tmp_path: Path, delta, to_edge, expected) -> None:
    """GH #211's edge and five-place moves, saved and reparsed. The helper also
    checks that triggers 0 to 2 keep their bytes."""
    refs = [("effect", 0), ("effect", 6), ("condition", 3)]
    before, after, types, orders = _funnel_edit_and_reload(
        tmp_path, lambda w: w.entry_move(ENTRY_ORDER_TRIGGER, refs, delta, to_edge)
    )
    assert orders == expected
    assert types == {"condition": [10, 5, 6, 6], "effect": [56, 19, 11, 14, 15, 12, 19]}, "the lists never move"
    for kind, order in ENTRY_ORDERS.items():
        assert after[kind] == [before[kind][order.index(index)] for index in orders[kind]]


def test_a_paste_through_the_funnel_lands_below_the_clicked_row(tmp_path: Path) -> None:
    from descape import trigger_clipboard

    def operate(window) -> None:
        block = trigger_clipboard.copy_entries(
            window.trigger_panel._manager(),
            2,
            [("effect", 0), ("effect", 1)],
            doc_id=window._doc_id,
            scenario_version=window.scenario.scenario_version,
        )
        window.entry_structural_edit("paste", ENTRY_ORDER_TRIGGER, "effect", [], -1, block=block, anchor=("effect", 0))

    before, after, types, orders = _funnel_edit_and_reload(tmp_path, operate)
    assert after["condition"] == before["condition"]
    # Effect 0 is the second row, so the two pasted ones are rows 3 and 4.
    assert after["effect"][:2] == before["effect"][:2] and after["effect"][4:] == before["effect"][2:]
    assert types["effect"] == [56, 8, 9, 19, 11, 14, 15, 12, 19], "and below index 0 in the list"
    assert orders["effect"] == [6, 0, 1, 2, 8, 4, 7, 3, 5]
    assert [types["effect"][index] for index in orders["effect"][2:4]] == [8, 9]


def _stamp_create_objects(window) -> None:
    from descape import brush

    window.trigger_panel.select_entry("effect", 2)  # the fixture's Create Object
    window.brush_size_spin.setValue(3)
    window.brush_shape_combo.setCurrentIndex(window.brush_shape_combo.findData(brush.BRUSH_SHAPE_SQUARE))
    window.stamp_create_objects(50, 50)


_APPENDS = [
    ("new condition", lambda w: w.entry_structural_edit("new", ENTRY_ORDER_TRIGGER, "condition", -1, 3), (1, 0)),
    ("new effect", lambda w: w.entry_structural_edit("new", ENTRY_ORDER_TRIGGER, "effect", -1, 55), (0, 1)),
    (
        "duplicate",
        lambda w: w.entry_structural_edit(
            "copy", ENTRY_ORDER_TRIGGER, "effect", [("condition", 2), ("effect", 4), ("effect", 0)], -1
        ),
        (1, 2),
    ),
    ("create objects stamp", _stamp_create_objects, (0, 9)),
]


@pytest.mark.parametrize("label,operate,added", _APPENDS, ids=[label for label, _, _ in _APPENDS])
def test_an_append_through_the_funnel_lands_last_in_display(tmp_path: Path, label, operate, added) -> None:
    """New, Duplicate and the Create Objects stamp only append, which the
    library's own heal gets right: the rows above keep their order."""
    before, after, _types, orders = _funnel_edit_and_reload(tmp_path, operate)
    for (kind, order), count in zip(ENTRY_ORDERS.items(), added, strict=True):
        assert orders[kind] == [*order, *range(len(order), len(order) + count)], label
        assert after[kind][: len(order)] == before[kind], label


def test_a_duplicate_lands_in_the_order_the_rows_were_given(tmp_path: Path) -> None:
    refs = [("effect", 4), ("effect", 0)]
    before, after, _types, _orders = _funnel_edit_and_reload(
        tmp_path, lambda w: w.entry_structural_edit("copy", ENTRY_ORDER_TRIGGER, "effect", refs, -1)
    )
    # Effect 4 is the first row and effect 0 the second.
    assert after["effect"][-2:] == before["effect"][:2]


def test_a_retype_through_the_funnel_keeps_its_row(tmp_path: Path) -> None:
    """Type… appends, pops and replaces at the same index, so the order stands."""
    before, after, types, orders = _funnel_edit_and_reload(
        tmp_path, lambda w: w.entry_structural_edit("retype", ENTRY_ORDER_TRIGGER, "effect", 0, 3)
    )
    assert orders == ENTRY_ORDERS
    assert types["effect"] == [3, 19, 11, 14, 15, 12, 19]
    # Effect 0 is the second row: only that one changed.
    assert [a == b for a, b in zip(after["effect"], before["effect"], strict=True)] == [
        True, False, True, True, True, True, True,
    ]


def _check_removals_keep_display_order(scenario_path: Path) -> int:
    """Every trigger of one file with a non-identity condition or effect order,
    each of its entries removed in turn. Returns how many orders were checked."""
    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    checked = 0
    for index, live in enumerate(manager.triggers):
        for kind in ENTRY_ORDERS:
            order = list(getattr(live, f"{kind}_order"))
            assert sorted(order) == list(range(len(getattr(live, f"{kind}s")))), (
                f"{scenario_path.name}: trigger {index}'s {kind} order is not a permutation"
            )
            if order == sorted(order):
                continue
            trigger = copy.deepcopy(live)
            entries = getattr(trigger, f"{kind}s")
            for victim in range(len(entries)):
                before = _shown_ids(trigger, kind)
                entry = entries[victim]
                remove_entries(trigger, [(kind, victim)])
                assert _shown_ids(trigger, kind) == [shown for shown in before if shown != id(entry)], (
                    f"{scenario_path.name}: trigger {index}, removing {kind} {victim}"
                )
                # Put it back for the next round, cheaper than a deepcopy per entry.
                entries.insert(victim, entry)
                set_entry_order(trigger, kind, order)
            checked += 1
    return checked


def test_removing_any_entry_keeps_the_survivors_display_order_across_the_corpus(scenario_path: Path) -> None:
    if not _check_removals_keep_display_order(scenario_path):
        pytest.skip(f"{scenario_path.name}: every condition and effect order is identity")


@pytest.mark.corpus
@pytest.mark.parametrize(
    "name", ["8tp3w9j", "C2_ElCid_coop_3_v0_12", "atilla_1_scn_resaved", "old-allies-final-v2"]
)
def test_removing_any_entry_keeps_the_survivors_display_order_on_a_non_identity_file(name: str) -> None:
    """The quick corpus subset holds identity orders only, so the four
    examples/ files that do not are named here."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this corpus")
    assert _check_removals_keep_display_order(path), "corpus assumption: a non-identity entry order"


def _quantity_fields(loaded) -> list[list[tuple[bytes, ...]]]:
    """Each trigger's effects' `quantity` and, where the file stores it, `quantity_float` bytes."""
    names = ("quantity", "quantity_float")
    return [
        [
            tuple(entry.retriever_map[n].get_data_as_bytes() for n in names if n in entry.retriever_map)
            for entry in struct.retriever_map["effect_data"].data or []
        ]
        for struct in _raw_structs(loaded)
    ]


@pytest.mark.corpus
@pytest.mark.parametrize(
    "name", ["8tp3w9j", "old-allies-final-v2", "2_Joan_coop_2_v0_15", "C2_ElCid_coop_1_v0_16"]
)
def test_resave_keeps_every_effect_quantity_pair(name: str, tmp_path: Path) -> None:
    """Every trigger dirty, both stored halves byte-equal. QUICK_CORPUS_NAMES
    holds none of the files with a disagreeing pair, and the last has no float
    field. Other fields differ by design on a dirty trigger."""
    path = Path(__file__).resolve().parent.parent / "examples" / f"{name}.aoe2scenario"
    if not path.exists():
        pytest.skip(f"{path.name} is not in this corpus")
    loaded = load_map_and_units(path)
    model = TriggerEditModel(loaded)
    before = _quantity_fields(loaded)
    for index in range(len(model.manager().triggers)):
        model.mark_dirty(index)
    out = tmp_path / "resaved.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded = load_map_and_units(out)
    assert parse_triggers(reloaded) is not None
    after = _quantity_fields(reloaded)
    assert any(before), "corpus assumption: the file has effects"
    assert after == before


# -- tag management: rename and remove a tag across its carriers ---------------

# Trigger 1 is the case variant, trigger 2 carries D1 only as a chained second
# tag, so renaming "D1" must leave both alone.
_TAGGED_NAMES = ["[D1] setup", "[d1]armour split", "[P1][D1] references", "[D1]: variable"]


def _tagged_base(tmp_path: Path) -> Path:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    for index, name in enumerate(_TAGGED_NAMES):
        model.manager().triggers[index].name = name
        model.mark_dirty(index)
    out = tmp_path / "tagged.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    return out


def _tag_window(base: Path):
    import conftest

    window = conftest.shown_window()
    window.load_scenario(base)
    window.mode_combo.setCurrentText("Triggers")
    return window


def _live_names(window) -> list[str]:
    return [t.name for t in window.trigger_edits.manager().triggers]


def _saved_blobs(path: Path) -> list[bytes]:
    loaded = load_map_and_units(path)
    return model_blob_lengths(_section_bytes(loaded), TriggerEditModel(loaded))


def _save_and_reload_names(window, tmp_path: Path) -> tuple[Path, list[str]]:
    out = tmp_path / "retagged.aoe2scenario"
    write_scenario(window.scenario, out, triggers=window.trigger_edits)
    reloaded = parse_triggers(load_map_and_units(out))
    assert reloaded is not None
    return out, [t.name for t in reloaded.triggers]


def test_renaming_a_tag_rewrites_every_carrier_and_nothing_else(tmp_path: Path) -> None:
    base = _tagged_base(tmp_path)
    window = _tag_window(base)
    try:
        window.rename_trigger_tag("D1", "Intro")
        expected = ["[Intro] setup", "[d1]armour split", "[P1][Intro] references", "[Intro]: variable"]
        assert _live_names(window) == expected
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].label == 'Rename tag "D1" to "Intro" (3 triggers)'
        assert window.edit_history.records[0].touched == [0, 2, 3]
        out, names = _save_and_reload_names(window, tmp_path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    assert names == expected
    # Names change length, so no byte-range confinement: every non-carrier's
    # own blob is byte-identical instead.
    before, after = _saved_blobs(base), _saved_blobs(out)
    assert [after[i] == before[i] for i in range(4)] == [False, True, False, False]


def test_a_rename_without_its_content_declaration_is_spliced_away(tmp_path: Path, monkeypatch) -> None:
    """The control for the test above: content_touched is what saves it."""
    base = _tagged_base(tmp_path)
    window = _tag_window(base)
    original = type(window)._trigger_edit
    monkeypatch.setattr(
        type(window),
        "_trigger_edit",
        lambda self, model, label, content_touched=(), **kw: original(self, model, label, (), **kw),
    )
    try:
        window.rename_trigger_tag("D1", "Intro")
        assert _live_names(window)[0] == "[Intro] setup"
        _, names = _save_and_reload_names(window, tmp_path)
    finally:
        window.edit_history.mark_saved()
        window.close()
    assert names == _TAGGED_NAMES


def test_one_undo_reverses_a_tag_rename_and_redo_reapplies_it(tmp_path: Path) -> None:
    window = _tag_window(_tagged_base(tmp_path))
    try:
        window.rename_trigger_tag("D1", "Intro")
        renamed = _live_names(window)
        window.undo()
        assert _live_names(window) == _TAGGED_NAMES
        assert not window.edit_history.is_dirty
        window.redo()
        assert _live_names(window) == renamed
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renaming_onto_an_existing_tag_merges_the_facets(tmp_path: Path) -> None:
    window = _tag_window(_tagged_base(tmp_path))
    try:
        window.rename_trigger_tag("d1", "D1")
        names = _live_names(window)
        assert [parse_tag(n) for n in names] == ["D1", "D1", "P1", "D1"]
        assert names[1] == "[D1]armour split"
        assert len(window.edit_history.records) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize(
    "old,new", [("D1", ""), ("D1", "   "), ("D1", "D1"), ("D1", " D1 "), ("Nope", "X")],
    ids=["empty", "whitespace", "unchanged", "unchanged-padded", "no-carriers"],
)
def test_a_no_op_tag_rename_records_nothing(tmp_path: Path, old, new) -> None:
    window = _tag_window(_tagged_base(tmp_path))
    try:
        window.rename_trigger_tag(old, new)
        assert window.edit_history.records == []
        assert window.trigger_edits is None or not window.trigger_edits.has_edits
    finally:
        window.close()


def test_a_new_tag_holding_the_closer_is_refused_before_anything_opens(tmp_path: Path, monkeypatch) -> None:
    from descape import viewer

    warnings = []
    monkeypatch.setattr(viewer.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window = _tag_window(_tagged_base(tmp_path))
    try:
        window.rename_trigger_tag("D1", "a]b")
        assert window.edit_history.records == []
        assert not window.trigger_edits.has_edits
        assert _live_names(window) == _TAGGED_NAMES
    finally:
        window.close()
    assert len(warnings) == 1 and '"]"' in warnings[0]


def test_removing_a_tag_strips_every_carrier_and_saves(tmp_path: Path) -> None:
    base = _tagged_base(tmp_path)
    window = _tag_window(base)
    try:
        window.remove_trigger_tag("D1")
        expected = ["setup", "[d1]armour split", "[P1] references", ": variable"]
        assert _live_names(window) == expected
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[0].label == 'Remove tag "D1" (3 triggers)'
        assert window.edit_history.records[0].touched == [0, 2, 3]
        out, names = _save_and_reload_names(window, tmp_path)
        window.undo()
        assert _live_names(window) == _TAGGED_NAMES
    finally:
        window.edit_history.mark_saved()
        window.close()
    assert names == expected
    before, after = _saved_blobs(base), _saved_blobs(out)
    assert [after[i] == before[i] for i in range(4)] == [False, True, False, False]


def test_removing_a_tag_nobody_carries_records_nothing(tmp_path: Path) -> None:
    window = _tag_window(_tagged_base(tmp_path))
    try:
        window.remove_trigger_tag("Nope")
        assert window.edit_history.records == []
    finally:
        window.close()


def test_renaming_each_files_commonest_tag_survives_a_reload(scenario_path: Path, tmp_path: Path) -> None:
    from collections import Counter

    import conftest

    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    if not loaded.trigger_write_supported:
        pytest.skip(f"{scenario_path.name}: fails the alignment gate")
    before = [t.name or "" for t in manager.triggers]
    tags = Counter(tag for n in before for tag in set(tag_chain(n)))
    if not tags:
        pytest.skip(f"{scenario_path.name}: no tagged trigger")
    old = tags.most_common(1)[0][0]
    new = "DEscapeRetag"
    assert new not in tags

    window = conftest.shown_window()
    try:
        window.load_scenario(scenario_path)
        window.mode_combo.setCurrentText("Triggers")
        window.rename_trigger_tag(old, new)
        out = tmp_path / "retagged.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
    finally:
        window.edit_history.mark_saved()
        window.close()
    reloaded = parse_triggers(load_map_and_units(out))
    after = [t.name or "" for t in reloaded.triggers]
    assert {t for n in after for t in tag_chain(n)} == ({t for n in before for t in tag_chain(n)} - {old}) | {new}
    for name_before, name_after in zip(before, after, strict=True):
        if old not in tag_chain(name_before):
            assert name_after == name_before
        else:
            assert tag_chain(name_after) == [new if t == old else t for t in tag_chain(name_before)]


# -- GH #2: renumber IDs to display order --------------------------------------

# Where each entry kind stores its type, and the types that link to a trigger.
_LINK_TYPES = {"effect_data": ("effect_type", {"8", "9"}), "condition_data": ("condition_type", {"79"})}


def _links_by_trigger(manager) -> list[list[int]]:
    """Every trigger's links (Activate/Deactivate Trigger, Trigger Active), as IDs."""
    return [[ce.trigger_id for ce in get_trigger_referencing_ce(t)] for t in manager.triggers]


def _raw_structs(loaded) -> list:
    return loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data or []


def _raw_fields(struct) -> dict:
    """A parsed struct's stored values, type-strict (repr), entry lists recursed."""
    out: dict = {}
    for name, retriever in struct.retriever_map.items():
        data = retriever.data
        if name in _LINK_TYPES or (isinstance(data, list) and data and hasattr(data[0], "retriever_map")):
            out[name] = [_raw_fields(item) for item in data or []]
        else:
            out[name] = repr(data)
    return out


def _entry_diffs(kind: str, old: dict, new: dict, slot_of: dict[int, int]) -> list[str]:
    """What changed in one condition or effect beyond a renumber's own remap
    and the library's normalizations of any re-serialized trigger, each
    allowed only in its exact value shape: `item_id` -1 filled from the field
    its getter derives it from, and a legacy location reference moved into an
    unset or equal one."""
    old, new = dict(old), dict(new)
    type_field, link_types = _LINK_TYPES[kind]
    problems = []
    if old.get(type_field) in link_types:
        raw = int(old["trigger_id"])
        expected = repr(slot_of.get(raw, raw))
        if new["trigger_id"] != expected:
            problems.append(f"trigger_id {raw} became {new['trigger_id']}, expected {expected}")
        old["trigger_id"] = new["trigger_id"]
    if kind == "effect_data":
        if old.get("item_id") == "-1" and new.get("item_id") != "-1":
            derived = next(
                (old[f] for f in ("object_list_unit_id", "technology", "tribute_list") if old.get(f) not in (None, "-1", "None")),
                None,
            )
            if new["item_id"] == derived:
                old["item_id"] = new["item_id"]
        location, legacy = "location_object_reference", "legacy_location_object_reference"
        if legacy in old and old[legacy] not in ("-1", "None") and (
            old[location] in ("-1", old[legacy]) and new[location] == old[legacy] and new[legacy] == "-1"
        ):
            old[location], old[legacy] = new[location], new[legacy]
    problems += [f"{k}: {old.get(k)} -> {new.get(k)}" for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)]
    return problems


def _trigger_diffs(old: dict, new: dict, slot_of: dict[int, int]) -> list[str]:
    problems = []
    for key in sorted(old.keys() | new.keys()):
        if key in _LINK_TYPES:
            olds, news = old.get(key, []), new.get(key, [])
            if len(olds) != len(news):
                problems.append(f"{key}: {len(olds)} entries became {len(news)}")
                continue
            for i, (o, n) in enumerate(zip(olds, news, strict=True)):
                problems += [f"{key}[{i}] {p}" for p in _entry_diffs(key, o, n, slot_of)]
        elif old.get(key) != new.get(key):
            problems.append(f"{key}: {old.get(key)} -> {new.get(key)}")
    return problems


def _check_renumber_round_trip(path: Path, tmp_path: Path) -> int:
    """Renumber `path` through structural_edit() + write_scenario(), reload:
    names in the old display order, every link re-pointed by display
    position, the execution-order byte unchanged, every trigger the renumber
    left clean byte-identical, and every dirty one equal field for field but
    for _entry_diffs()' allowances. Returns how many triggers went dirty."""
    from descape.trigger_model import exec_order_value, ids_follow_display_order, renumber_to_display_order

    loaded = load_map_and_units(path)
    manager = parse_triggers(loaded)
    if manager is None or not loaded.trigger_write_supported:
        pytest.skip(f"{path.name}: triggers are not editable")
    order = list(manager.trigger_display_order)
    if ids_follow_display_order(order):
        pytest.skip(f"{path.name}: IDs already follow display order")
    names = [t.name or "" for t in manager.triggers]
    links = _links_by_trigger(manager)
    exec_byte = exec_order_value(loaded)
    model = TriggerEditModel(loaded)
    slices = model_blob_lengths(_section_bytes(loaded), model)
    # Only a trigger holding a link can go dirty; dumped before the save can rewrite the structs.
    fields = {i: _raw_fields(s) for i, s in enumerate(_raw_structs(loaded)) if links[i]}
    slot_of = {index: slot for slot, index in enumerate(order)}

    model.structural_edit(renumber_to_display_order)
    dirty = set(model.dirty_indices())
    out = tmp_path / "renumbered.aoe2scenario"
    write_scenario(loaded, out, triggers=model)

    reloaded_loaded = load_map_and_units(out)
    reloaded = parse_triggers(reloaded_loaded)
    assert [t.name or "" for t in reloaded.triggers] == [names[i] for i in order], f"{path.name}: names"
    assert list(reloaded.trigger_display_order) == list(range(len(order)))
    assert _links_by_trigger(reloaded) == [[slot_of.get(raw, raw) for raw in links[i]] for i in order], (
        f"{path.name}: a link no longer names its trigger"
    )
    assert exec_order_value(reloaded_loaded) == exec_byte
    after_slices = model_blob_lengths(_section_bytes(reloaded_loaded), TriggerEditModel(reloaded_loaded))
    after_structs = _raw_structs(reloaded_loaded)
    problems = []
    for position, index in enumerate(order):
        if position not in dirty:
            if after_slices[position] != slices[index]:
                problems.append(f"clean trigger {names[index]!r} changed bytes")
            continue
        assert index in fields, f"{path.name}: trigger {names[index]!r} went dirty holding no link"
        problems += [
            f"{names[index]!r} {p}" for p in _trigger_diffs(fields[index], _raw_fields(after_structs[position]), slot_of)
        ]
    assert not problems, f"{path.name}: {problems[:10]}"
    return len(dirty)


def _renumber_fixture() -> tuple:
    """The fixture with a Trigger Active link added on trigger 3 (onto 1) and
    display order [2, 0, 3, 1], so every link of all three types changes ID."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    model.manager().triggers[3].new_condition.trigger_active(trigger_id=1)
    model.mark_dirty(3)
    model.structural_edit(lambda m: setattr(m, "trigger_display_order", [2, 0, 3, 1]))
    return loaded, model


def _link_state(manager) -> tuple:
    """List order by identity, IDs, display order and every link."""
    return (
        [id(t) for t in manager.triggers],
        [t.trigger_id for t in manager.triggers],
        list(manager.trigger_display_order),
        _links_by_trigger(manager),
    )


def test_renumbering_gives_ids_in_display_order_and_keeps_every_link(tmp_path: Path) -> None:
    from descape.trigger_model import exec_order_value, renumber_to_display_order

    loaded, model = _renumber_fixture()
    manager = model.manager()
    order = list(manager.trigger_display_order)
    old = list(manager.triggers)
    targets = [(ce, ce.trigger_id, old[ce.trigger_id]) for t in old for ce in get_trigger_referencing_ce(t)]
    kinds = sorted({(type(ce).__name__, getattr(ce, "effect_type", getattr(ce, "condition_type", None))) for ce, _, _ in targets})
    assert kinds == [("Condition", 79), ("Effect", 8), ("Effect", 9)], "fixture assumption: all three link types"
    names = [t.name for t in old]
    links = _links_by_trigger(manager)
    slot_of = {index: slot for slot, index in enumerate(order)}
    exec_byte = exec_order_value(loaded)
    slices = model_blob_lengths(_section_bytes(loaded), model)

    model.structural_edit(renumber_to_display_order)
    assert list(manager.trigger_display_order) == [0, 1, 2, 3]
    assert [t.trigger_id for t in manager.triggers] == [0, 1, 2, 3]
    assert all(live is old[index] for live, index in zip(manager.triggers, order, strict=True))
    assert all(manager.triggers[ce.trigger_id] is target for ce, _, target in targets)
    assert all(ce.trigger_id != raw for ce, raw, _ in targets), "fixture assumption: every link moves"
    dirty = model.dirty_indices()

    out = tmp_path / "renumbered.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    reloaded_loaded = load_map_and_units(out)
    reloaded = parse_triggers(reloaded_loaded)
    assert [t.name for t in reloaded.triggers] == [names[i] for i in order]
    # By position, not name: dividers repeat names.
    assert _links_by_trigger(reloaded) == [[slot_of[raw] for raw in links[i]] for i in order]
    assert exec_order_value(reloaded_loaded) == exec_byte
    after = model_blob_lengths(_section_bytes(reloaded_loaded), TriggerEditModel(reloaded_loaded))
    clean = [position for position in range(4) if position not in dirty]
    assert clean, "fixture assumption: a trigger holding no link"
    assert all(after[position] == slices[order[position]] for position in clean)


def test_a_renumber_whose_remap_misses_a_link_puts_the_manager_back(monkeypatch) -> None:
    """D8: the helper restores the original list before raising, so
    structural_edit()'s except branch writes the old display order onto it."""
    import AoE2ScenarioParser.objects.managers.trigger_manager as library

    from descape.trigger_model import renumber_to_display_order

    _loaded, model = _renumber_fixture()
    before = _link_state(model.manager())
    monkeypatch.setattr(library, "get_trigger_referencing_ce", lambda trigger: [])

    with pytest.raises(RuntimeError, match="links to ID"):
        renumber_to_display_order(model.manager())
    assert _link_state(model.manager()) == before
    with pytest.raises(RuntimeError, match="links to ID"):
        model.structural_edit(renumber_to_display_order)
    assert _link_state(model.manager()) == before


def test_renumber_refusal_names_a_bad_order_and_a_stray_id() -> None:
    from descape.trigger_model import ids_follow_display_order, renumber_refusal, renumber_to_display_order

    assert ids_follow_display_order([0, 1, 2]) and ids_follow_display_order([])
    assert not ids_follow_display_order([1, 0, 2])
    _loaded, model = _renumber_fixture()
    manager = model.manager()
    assert renumber_refusal(manager) == ""
    manager.trigger_display_order = [0, 0, 1, 2]
    assert "not a permutation" in renumber_refusal(manager)
    before = _link_state(manager)
    with pytest.raises(ValueError, match="not a permutation"):
        renumber_to_display_order(manager)
    assert _link_state(manager) == before, "a refusal changes nothing"
    manager.trigger_display_order = [0, 1, 2, 3]
    assert renumber_refusal(manager) == "IDs already follow display order"  # no empty undo step
    manager.trigger_display_order = [2, 0, 3, 1]
    manager.triggers[1].trigger_id = 7
    assert renumber_refusal(manager) == "trigger 1 carries ID 7"


def test_a_renumber_that_fails_in_the_viewer_warns_once_and_changes_nothing(monkeypatch) -> None:
    import AoE2ScenarioParser.objects.managers.trigger_manager as library

    window = conftest.shown_window()
    warned: list[str] = []
    window._warn = lambda title, text: warned.append(title)
    try:
        window.load_scenario(FIXTURE_PATH)
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_structural_edit("move_down", [0])
        manager = window.trigger_edits.manager()
        before = _link_state(manager)
        monkeypatch.setattr(library, "get_trigger_referencing_ce", lambda trigger: [])
        window.trigger_structural_edit("renumber", [])
        assert warned == ["Trigger edit failed"]
        assert _link_state(window.trigger_edits.manager()) == before
        assert "Renumber trigger IDs to display order failed" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_refused_renumber_records_nothing() -> None:
    window = conftest.shown_window()
    warned: list[str] = []
    window._warn = lambda title, text: warned.append(title)  # a real box hangs a headless worker
    try:
        window.load_scenario(FIXTURE_PATH)
        window.mode_combo.setCurrentText("Triggers")
        window.trigger_structural_edit("move_down", [0])
        model = window.trigger_edits
        records = len(window.edit_history.records)
        good = list(model.manager().trigger_display_order)
        model.manager().trigger_display_order = [0, 0, 1, 2]
        window.trigger_structural_edit("renumber", [])
        assert warned == [] and len(window.edit_history.records) == records
        assert "Cannot renumber trigger IDs: the display order is not a permutation" in window.status_log.toPlainText()
        model.manager().trigger_display_order = good
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_renumbering_round_trips_across_the_corpus(scenario_path: Path, tmp_path: Path) -> None:
    _check_renumber_round_trip(scenario_path, tmp_path)


def test_no_entry_outside_the_three_link_types_stores_a_trigger_id(scenario_path: Path) -> None:
    """Fact 2, re-measured per file: a renumber remaps only types 8/9/79."""
    loaded = load_map_and_units(scenario_path)
    if parse_triggers(loaded) is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    stray = []
    for t, trigger in enumerate(_raw_structs(loaded)):
        for kind, (type_field, link_types) in _LINK_TYPES.items():
            for i, entry in enumerate(trigger.retriever_map[kind].data or []):
                stored = entry.retriever_map.get("trigger_id")
                if stored is None or repr(entry.retriever_map[type_field].data) in link_types:
                    continue
                if stored.data not in (-1, None):
                    stray.append((t, kind, i, stored.data))
    assert not stray, f"{scenario_path.name}: {stray[:10]}"
