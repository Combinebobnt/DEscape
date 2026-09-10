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

from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.trigger_model import (
    TriggerEditModel,
    TriggerEditsUnavailableError,
    display_order_with_copy_inserted,
    moved_display_order,
)

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
    for i, (a, b) in enumerate(zip(before, after)):
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


@pytest.mark.corpus
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


@pytest.mark.corpus
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


@pytest.mark.corpus
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


@pytest.mark.corpus
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
