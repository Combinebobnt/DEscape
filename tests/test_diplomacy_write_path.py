"""Diplomacy mode step 2: the write path. descape/options_model.py's
OptionsEditModel extended with the Diplomacy grid's cells (an additive entry
set under synthetic ids, "stance:i:j" and "allied_victory:i"),
diplomacy_write_supported() as its own gate, and the
descape/scenario_write.py branch that re-checks it. Still no panel -- every
edit here goes through OptionsEditModel.set_value() directly, the same way
tests/test_options_write_path.py exercises the scalar rows.

The claims, in the order this feature's acceptance gates list them:

1. **Containment.** A model with no diplomacy edits writes exactly the bytes
   it wrote before this work existed.
2. **Verbatim preservation.** One stance edit leaves every other cell, the
   filler rows/columns 8-15, individual_victories and separator untouched.
3. **Locality.** One stance edit changes exactly 4 bytes, at the cell's own
   offset.
4. **Round trip.** Write, reload through the real loader, read back.
5. **Directionality.** Editing stance[i][j] leaves stance[j][i] untouched.
6. **The gates.** diplomacy_write_supported() is independent of
   options_write_supported() -- corrupting one must not flip the other.
7. **The allied-victory mirror (TASK-062).** An allied-victory edit writes
   Units.player_data_3's u8 copy too; a mirror failure takes out only those
   eight cells, and the save re-gates it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import option_fields
from descape.diplomacy_fields import (
    allied_victory_cell_id,
    allied_victory_offsets,
    allied_victory_value,
    stance_cell_id,
    stance_offsets,
    verify_diplomacy_block,
)
from descape.options_model import (
    OptionsEditModel,
    allied_victory_write_supported,
    diplomacy_write_supported,
    options_write_supported,
    players_write_supported,
)
from descape.player_fields import aok_allied_victory_mirror_targets
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario

_CELL = struct.Struct("<I")


def _loaded(path: Path = BLANK_TEMPLATE_PATH):
    return load_map_and_units(path)


def _written_body(path: Path) -> bytes:
    raw = path.read_bytes()
    loaded = load_map_and_units(path)
    return _decompress_bytes(raw[len(loaded.header_bytes) :])


def _differing_ranges(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """Maximal [start, end) spans where two equal-length buffers differ. Same
    helper tests/test_options_write_path.py uses, kept local for the same
    reason that file gives: neither file's imports should depend on the
    other's layout."""
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


# -- construction -------------------------------------------------------------


def test_diplomacy_cells_are_present_in_a_writable_model() -> None:
    loaded = _loaded()
    assert diplomacy_write_supported(loaded)
    model = OptionsEditModel(loaded)
    for cell_id in stance_offsets(loaded):
        assert model.original_value(cell_id) in (0, 1, 3), cell_id
    for cell_id in allied_victory_offsets(loaded):
        assert model.original_value(cell_id) in (0, 1), cell_id


def test_diplomacy_cells_are_absent_when_the_grid_gate_fails() -> None:
    """An independent gate, not folded into options_write_supported(): a
    Diplomacy-only failure (here, one corrupted cell byte -- not the shared
    diplomacy_section_end anchor, which the 4 Diplomacy scalar specs still
    key off of until step 5 moves them out of this section's byte-mismatch
    check) leaves the scalar rows constructible, and simply omits the grid's
    cells rather than raising -- see options_model.diplomacy_write_supported's
    docstring."""
    loaded = _loaded()
    target = stance_offsets(loaded)[stance_cell_id(3, 5)]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)

    assert not diplomacy_write_supported(loaded)
    assert options_write_supported(loaded, option_fields.specs_for(loaded))

    model = OptionsEditModel(loaded)
    assert not model.has_diplomacy_edits
    with pytest.raises(KeyError):
        model.set_value(stance_cell_id(1, 2), 3)
    # A scalar row unaffected by the grid corruption is still writable.
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    assert model.has_edits


# -- 1. containment -------------------------------------------------------


def test_a_model_with_no_diplomacy_edits_writes_the_same_file_as_no_model_at_all(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)

    model = OptionsEditModel(loaded)
    assert not model.has_edits
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)

    assert with_model.read_bytes() == without.read_bytes()


def test_a_stance_set_back_to_its_stored_value_leaves_the_document_clean() -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(1, 2))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(1, 2), other)
    assert model.has_edits
    assert model.has_diplomacy_edits
    model.set_value(stance_cell_id(1, 2), stored)
    assert not model.has_edits
    assert not model.has_diplomacy_edits


# -- 2. verbatim preservation, 3. locality ---------------------------------


def test_one_stance_edit_changes_only_bytes_within_that_cells_own_span(
    tmp_path: Path,
) -> None:
    """Containment, not equality, per test_options_write_path.py's own
    victory_condition precedent: {ALLY, NEUTRAL, ENEMY} are all < 256, so a
    little-endian u32 cell only ever moves its low byte here -- a real diff
    range narrower than the cell's full 4-byte span, not a bug in the walk."""
    loaded = _loaded()
    offsets = stance_offsets(loaded)
    target = offsets[stance_cell_id(3, 5)]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(3, 5))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(3, 5), other)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges, "no bytes changed"
    for start, end in ranges:
        assert target.offset <= start and end <= target.offset + target.length, (
            f"changed range ({start}, {end}) escapes the cell's own span "
            f"({target.offset}, {target.offset + target.length})"
        )


def test_one_allied_victory_edit_changes_both_stored_copies_and_nothing_else(
    tmp_path: Path,
) -> None:
    """TASK-062: the flag has two stored copies, Diplomacy's u32 and
    Units.player_data_3's u8 aok_allied_victory, and an edit writes both.
    {0, 1} flags are < 256, so each copy moves only its low byte. Every other
    player's copies and the whole stance grid stay byte-identical, since
    every changed range must sit inside one of the two spans."""
    loaded = _loaded()
    primary = allied_victory_offsets(loaded)[allied_victory_cell_id(4)]
    mirror = aok_allied_victory_mirror_targets(loaded)[4]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    stored = model.original_value(allied_victory_cell_id(4))
    model.set_value(allied_victory_cell_id(4), 1 - stored)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    spans = [(primary.offset, primary.offset + primary.length), (mirror.offset, mirror.offset + mirror.length)]
    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert len(ranges) == 2, f"expected one changed range per stored copy, got {ranges}"
    for start, end in ranges:
        assert any(lo <= start and end <= hi for lo, hi in spans), (
            f"changed range ({start}, {end}) escapes both copies' spans {spans}"
        )


def test_an_allied_victory_edit_emits_one_patch_per_stored_copy() -> None:
    loaded = _loaded()
    primary = allied_victory_offsets(loaded)[allied_victory_cell_id(3)]
    mirror = aok_allied_victory_mirror_targets(loaded)[3]
    assert (primary.length, mirror.length, mirror.codec) == (4, 1, "u8")

    model = OptionsEditModel(loaded)
    stored = model.original_value(allied_victory_cell_id(3))
    model.set_value(allied_victory_cell_id(3), 1 - stored)
    assert model.has_allied_victory_edits and model.has_diplomacy_edits
    assert not model.has_player_edits, "the mirror is not a Players field"
    assert sorted(model.serialize_patches()) == sorted(
        [(primary.offset, struct.pack("<I", 1 - stored)), (mirror.offset, bytes([1 - stored]))]
    )


def test_a_stance_edit_is_not_an_allied_victory_edit() -> None:
    model = OptionsEditModel(_loaded())
    stored = model.original_value(stance_cell_id(1, 2))
    model.set_value(stance_cell_id(1, 2), next(v for v in (0, 1, 3) if v != stored))
    assert model.has_diplomacy_edits
    assert not model.has_allied_victory_edits


def test_an_allied_victory_value_the_u8_copy_cannot_hold_is_refused() -> None:
    """The Diplomacy copy is a u32, but the player_data_3 copy is one byte, so
    256 must be refused at set time rather than desync the pair."""
    model = OptionsEditModel(_loaded())
    with pytest.raises(ValueError):
        model.set_value(allied_victory_cell_id(2), 256)
    assert not model.has_edits


def test_a_stance_edit_leaves_every_other_stance_and_allied_cell_untouched(
    tmp_path: Path,
) -> None:
    """Verbatim preservation across the whole section, not just the one
    changed cell -- includes the filler rows/columns 8-15, which
    stance_offsets()/allied_victory_offsets() never expose (see
    diplomacy_fields.py's module docstring), and individual_victories /
    separator, which sit between the grid and the allied-victory array."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(3, 5))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(3, 5), other)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)
    written = _written_body(out)

    reloaded = load_map_and_units(out)
    assert verify_diplomacy_block(reloaded)
    for cell_id, fo in stance_offsets(loaded).items():
        (value,) = _CELL.unpack_from(written, fo.offset)
        expected = other if cell_id == stance_cell_id(3, 5) else model.original_value(cell_id)
        assert value == expected, cell_id
    for cell_id, fo in allied_victory_offsets(loaded).items():
        (value,) = _CELL.unpack_from(written, fo.offset)
        assert value == model.original_value(cell_id), cell_id


# -- 4. round trip ----------------------------------------------------------


def test_a_stance_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(1, 2))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(1, 2), other)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    (value,) = _CELL.unpack_from(reloaded.decompressed_body, stance_offsets(reloaded)[stance_cell_id(1, 2)].offset)
    assert value == other
    assert diplomacy_write_supported(reloaded)


def test_an_allied_victory_edit_reads_back_after_a_reload(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(allied_victory_cell_id(2))
    model.set_value(allied_victory_cell_id(2), 1 - stored)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    (value,) = _CELL.unpack_from(
        reloaded.decompressed_body, allied_victory_offsets(reloaded)[allied_victory_cell_id(2)].offset
    )
    assert value == 1 - stored
    pd3 = reloaded._scenario.sections["Units"].retriever_map["player_data_3"].data
    assert [e.retriever_map["aok_allied_victory"].data for e in pd3] == [
        allied_victory_value(reloaded, p) for p in range(1, 9)
    ], "the player_data_3 copy reads back in sync with the Diplomacy copy"
    assert allied_victory_write_supported(reloaded)


# -- 5. directionality --------------------------------------------------------


def test_editing_one_stance_direction_leaves_the_reverse_pair_untouched(
    tmp_path: Path,
) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored_fwd = model.original_value(stance_cell_id(1, 2))
    stored_rev = model.original_value(stance_cell_id(2, 1))
    other = next(v for v in (0, 1, 3) if v != stored_fwd)
    model.set_value(stance_cell_id(1, 2), other)
    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    body = reloaded.decompressed_body
    offsets = stance_offsets(reloaded)
    (fwd,) = _CELL.unpack_from(body, offsets[stance_cell_id(1, 2)].offset)
    (rev,) = _CELL.unpack_from(body, offsets[stance_cell_id(2, 1)].offset)
    assert fwd == other
    assert rev == stored_rev


# -- 6. the gates ------------------------------------------------------------


def test_set_value_refuses_an_unrecognized_diplomacy_cell() -> None:
    model = OptionsEditModel(_loaded())
    with pytest.raises(KeyError):
        model.set_value(stance_cell_id(9, 1), 0)


def test_the_write_path_re_gates_diplomacy_rather_than_trusting_construction(
    tmp_path: Path,
) -> None:
    """Re-verified at save time rather than trusted from construction, same
    shape test_options_write_path.py's own re-gate test uses. Corrupts a
    *different* cell's stored byte (3,5), not the anchor -- corrupting the
    anchor would also fail options_write_supported() first (lock_teams and
    friends still key off diplomacy_section_end until step 5), which would
    raise WriteBlockedError for the wrong reason and leave the new
    has_diplomacy_edits-gated check in scenario_write.py unexercised."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(1, 2))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(1, 2), other)

    target = stance_offsets(loaded)[stance_cell_id(3, 5)]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert options_write_supported(loaded, model.specs)
    assert not diplomacy_write_supported(loaded)

    with pytest.raises(WriteBlockedError):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


def test_a_diplomacy_only_failure_does_not_block_an_unrelated_scalar_save(
    tmp_path: Path,
) -> None:
    """The reverse of the re-gate test above: a Diplomacy-grid failure must
    not block writing a scalar option edit that has nothing to do with it --
    the two gates stay independent all the way through the write path, not
    just at construction. The grid is broken by corrupting one cell byte
    (not the shared diplomacy_section_end anchor -- see the "gate fails"
    test above for why that would take lock_teams down too)."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value("lock_teams")
    model.set_value("lock_teams", 1 - stored)

    target = stance_offsets(loaded)[stance_cell_id(3, 5)]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)
    assert not diplomacy_write_supported(loaded)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)

    reloaded = load_map_and_units(out)
    spec = next(s for s in option_fields.specs_for(reloaded) if s.field_id == "lock_teams")
    assert option_fields.current_value(reloaded, spec) == 1 - stored


def _corrupt_mirror(loaded, player: int) -> None:
    target = aok_allied_victory_mirror_targets(loaded)[player]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)


def test_a_mirror_failure_takes_out_only_the_allied_victory_cells() -> None:
    """Gate independence (TASK-062): every corpus file passes all three gates,
    so a broken player_data_3 copy is manufactured by corrupting one stored
    mirror byte. The allied-victory ids drop out of the model; the stance grid
    and the Teams scalars stay writable."""
    loaded = _loaded()
    _corrupt_mirror(loaded, 6)
    assert aok_allied_victory_mirror_targets(loaded) is None
    assert diplomacy_write_supported(loaded)
    assert not allied_victory_write_supported(loaded)

    model = OptionsEditModel(loaded)
    for player in range(1, 9):
        assert not model.can_set(allied_victory_cell_id(player))
        with pytest.raises(KeyError):
            model.set_value(allied_victory_cell_id(player), 1)
    stored = model.original_value(stance_cell_id(1, 2))
    model.set_value(stance_cell_id(1, 2), next(v for v in (0, 1, 3) if v != stored))
    model.set_value("lock_teams", 1 - model.original_value("lock_teams"))
    assert model.has_diplomacy_edits and not model.has_allied_victory_edits


def test_the_write_path_re_gates_the_allied_victory_mirror(tmp_path: Path) -> None:
    """has_diplomacy_edits re-verifies only the Diplomacy section the primary
    lives in, so a mirror that stops verifying after construction must be
    caught by its own re-gate. Corrupts a different player's mirror byte, so
    diplomacy_write_supported() still passes and only the new check can fire."""
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    model.set_value(allied_victory_cell_id(3), 1 - model.original_value(allied_victory_cell_id(3)))
    _corrupt_mirror(loaded, 6)
    assert diplomacy_write_supported(loaded)
    assert players_write_supported(loaded), "the mirror is not part of the Players gate"

    with pytest.raises(WriteBlockedError, match="allied-victory"):
        write_scenario(loaded, tmp_path / "out.aoe2scenario", options=model)


def test_a_broken_mirror_does_not_block_a_stance_only_save(tmp_path: Path) -> None:
    loaded = _loaded()
    model = OptionsEditModel(loaded)
    stored = model.original_value(stance_cell_id(1, 2))
    other = next(v for v in (0, 1, 3) if v != stored)
    model.set_value(stance_cell_id(1, 2), other)
    _corrupt_mirror(loaded, 6)

    out = tmp_path / "out.aoe2scenario"
    write_scenario(loaded, out, options=model)
    reloaded = load_map_and_units(out)
    (value,) = _CELL.unpack_from(reloaded.decompressed_body, stance_offsets(reloaded)[stance_cell_id(1, 2)].offset)
    assert value == other


# -- 7. corpus ----------------------------------------------------------------


def test_the_allied_victory_copies_agree_across_the_corpus(scenario_path) -> None:
    """TASK-062's Stage 0 (20/20 in sync, 2026-09-02) recorded in code. The
    copies differ in width (u32 and u8), so compare decoded ints."""
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip("Units block fails closed on this file, so there is no mirror to reach")
    targets = aok_allied_victory_mirror_targets(loaded)
    assert targets is not None
    body = loaded.decompressed_body
    for player, target in targets.items():
        assert body[target.offset] == allied_victory_value(loaded, player), player


def test_a_browsed_diplomacy_model_saves_byte_identically_across_the_corpus(
    scenario_path, tmp_path: Path
) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.terrain_write_supported:
        pytest.skip("terrain block failed verification, so no save path at all")
    if not diplomacy_write_supported(loaded):
        pytest.skip("diplomacy grid fails closed on this file, as intended")

    without = tmp_path / "without.aoe2scenario"
    write_scenario(loaded, without)
    model = OptionsEditModel(loaded)
    with_model = tmp_path / "with.aoe2scenario"
    write_scenario(loaded, with_model, options=model)
    assert with_model.read_bytes() == without.read_bytes()
