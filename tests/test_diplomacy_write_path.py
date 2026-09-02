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
    stance_cell_id,
    stance_offsets,
    verify_diplomacy_block,
)
from descape.options_model import (
    OptionsEditModel,
    diplomacy_write_supported,
    options_write_supported,
)
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
    for i, (a, b) in enumerate(zip(before, after)):
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


def test_one_allied_victory_edit_changes_only_bytes_within_that_cells_own_span(
    tmp_path: Path,
) -> None:
    """{0, 1} flags are < 256 too -- same containment reasoning as the stance
    test above."""
    loaded = _loaded()
    offsets = allied_victory_offsets(loaded)
    target = offsets[allied_victory_cell_id(4)]

    base = tmp_path / "base.aoe2scenario"
    write_scenario(loaded, base)

    model = OptionsEditModel(loaded)
    stored = model.original_value(allied_victory_cell_id(4))
    model.set_value(allied_victory_cell_id(4), 1 - stored)
    edited = tmp_path / "edited.aoe2scenario"
    write_scenario(loaded, edited, options=model)

    ranges = _differing_ranges(_written_body(base), _written_body(edited))
    assert ranges, "no bytes changed"
    for start, end in ranges:
        assert target.offset <= start and end <= target.offset + target.length, (
            f"changed range ({start}, {end}) escapes the cell's own span "
            f"({target.offset}, {target.offset + target.length})"
        )


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


# -- 7. corpus ----------------------------------------------------------------


@pytest.mark.corpus
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
