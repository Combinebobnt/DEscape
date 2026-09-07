"""Covers descape/diplomacy_fields.py: the backward offset walk and the
load-time verification gate for Diplomacy mode's per-player stance grid and
allied-victory flags.

test_a_single_byte_corruption_flips_verify_to_false is the load-bearing test
here, mirroring test_options_model.py's own: "a gate that never fires is not
a gate." The corpus tests guard the two non-obvious shapes measured against
the real scenario corpus -- a directional (asymmetric) grid, and a player
count that genuinely varies rather than defaulting to 8 -- so neither passes
vacuously against a corpus that happened not to exercise it.
"""

from __future__ import annotations

import struct

import pytest

from descape import scenario_io
from descape.diplomacy_fields import (
    NUM_PLAYERS,
    allied_victory_cell_id,
    allied_victory_offsets,
    parse_cell_id,
    stance_cell_id,
    stance_offsets,
    verify_diplomacy_block,
)


def _loaded():
    return scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)


# -- cell ids -------------------------------------------------------------


def test_stance_cell_id_round_trips_through_parse_cell_id() -> None:
    assert parse_cell_id(stance_cell_id(3, 5)) == ("stance", 3, 5)


def test_allied_victory_cell_id_round_trips_through_parse_cell_id() -> None:
    assert parse_cell_id(allied_victory_cell_id(3)) == ("allied_victory", 3)


def test_parse_cell_id_rejects_an_unrecognized_id() -> None:
    with pytest.raises(ValueError):
        parse_cell_id("not-a-cell-id")


# -- offsets, against a real file ------------------------------------------


def test_stance_offsets_cover_the_full_grid_including_the_diagonal() -> None:
    """8x8, not 7 opponent rows -- the diagonal is real per-file data (see
    module docstring) and must round-trip even though a panel would render
    it read-only."""
    loaded = _loaded()
    offsets = stance_offsets(loaded)
    expected = {stance_cell_id(r, c) for r in range(1, NUM_PLAYERS + 1) for c in range(1, NUM_PLAYERS + 1)}
    assert set(offsets) == expected
    assert len(offsets) == NUM_PLAYERS * NUM_PLAYERS


def test_allied_victory_offsets_cover_every_player() -> None:
    loaded = _loaded()
    offsets = allied_victory_offsets(loaded)
    assert set(offsets) == {allied_victory_cell_id(p) for p in range(1, NUM_PLAYERS + 1)}


def test_stance_row_stride_and_cell_size_match_the_measured_constants() -> None:
    """Pins the derived stride/cell-size against the measured table (64
    bytes/row, 4 bytes/cell) -- a regression signal, even though the module
    itself never hardcodes either number."""
    loaded = _loaded()
    offsets = stance_offsets(loaded)
    row1_col1 = offsets[stance_cell_id(1, 1)]
    row1_col2 = offsets[stance_cell_id(1, 2)]
    row2_col1 = offsets[stance_cell_id(2, 1)]
    assert row1_col1.length == 4
    assert row1_col2.offset - row1_col1.offset == 4
    assert row2_col1.offset - row1_col1.offset == 64


def test_offsets_are_distinct_and_dont_overlap() -> None:
    loaded = _loaded()
    spans = [(fo.offset, fo.length) for fo in stance_offsets(loaded).values()]
    spans += [(fo.offset, fo.length) for fo in allied_victory_offsets(loaded).values()]
    spans.sort()
    for (off_a, len_a), (off_b, _len_b) in zip(spans, spans[1:]):
        assert off_a + len_a <= off_b


def test_stance_and_allied_victory_offsets_are_empty_without_a_trusted_anchor() -> None:
    loaded = _loaded()
    loaded.diplomacy_section_end = -1
    assert stance_offsets(loaded) == {}
    assert allied_victory_offsets(loaded) == {}


# -- verify_diplomacy_block(), mutation-tested -----------------------------


def test_verify_diplomacy_block_true_on_the_blank_template() -> None:
    assert verify_diplomacy_block(_loaded())


def test_a_single_byte_corruption_flips_verify_to_false() -> None:
    """Proof the gate is load-bearing, not vacuous: an unmutated load
    verifies True, and flipping exactly one byte inside one cell's own
    4-byte span flips it False."""
    loaded = _loaded()
    assert verify_diplomacy_block(loaded)

    target = stance_offsets(loaded)[stance_cell_id(3, 5)]
    body = bytearray(loaded.decompressed_body)
    body[target.offset] ^= 0xFF
    loaded.decompressed_body = bytes(body)

    assert not verify_diplomacy_block(loaded)


def test_an_unavailable_anchor_fails_the_gate_closed() -> None:
    loaded = _loaded()
    loaded.diplomacy_section_end = -1
    assert not verify_diplomacy_block(loaded)


# defined_player_count()/defined_player_ids() moved to
# descape/player_fields.py in step 3e -- their tests moved to
# tests/test_player_fields.py with them.


# -- against the corpus -----------------------------------------------------


@pytest.mark.corpus
def test_verify_diplomacy_block_true_across_the_corpus(scenario_path) -> None:
    loaded = scenario_io.load_map_and_units(scenario_path)
    assert verify_diplomacy_block(loaded), scenario_path.name


@pytest.mark.corpus
def test_the_stance_grid_is_directional_across_the_corpus(corpus_files) -> None:
    """Not-vacuous guard: at least one corpus file must have
    stance[i][j] != stance[j][i] for some pair, or this would pass just as
    happily against a bug that silently mirrored the grid."""
    cell = struct.Struct("<I")
    found_asymmetric = False
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(str(path))
        body = loaded.decompressed_body
        offsets = stance_offsets(loaded)
        for row in range(1, NUM_PLAYERS + 1):
            for col in range(row + 1, NUM_PLAYERS + 1):
                (fwd,) = cell.unpack_from(body, offsets[stance_cell_id(row, col)].offset)
                (rev,) = cell.unpack_from(body, offsets[stance_cell_id(col, row)].offset)
                if fwd != rev:
                    found_asymmetric = True
    assert found_asymmetric, "no corpus file had an asymmetric stance pair"
