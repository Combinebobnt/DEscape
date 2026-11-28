"""bridge_piece_frames' measured table of DE bridge piece frames (TASK-705).

The table is hand-measured from rendered frames; these tests pin its shape
against the committed tables (frame counts, spans, file names) and its
loud-refusal contract. The frame-by-frame oracle under every element is in
tests/test_object_orientation.py.
"""

from __future__ import annotations

import pytest

from descape import bridge_piece_frames, game_dataset, render_units, unit_sprites, unit_variant
from descape.terrain_palette import tile_span

TWELVE = {1840, 1841, 1844, 1845, 1993, 1994, 1996, 1997, 1999, 2000, 2039, 2041}
PAIRS = [(1840, 1841), (1844, 1845), (1993, 1994), (1996, 1997), (1999, 2000), (2039, 2041)]


def _identity(dx, dy):
    return (dx, dy)


def test_the_table_holds_exactly_the_twelve_and_return_of_rome_matches():
    assert set(bridge_piece_frames._TABLE["aoe2"]) == TWELVE
    assert bridge_piece_frames._TABLE["ror"] == bridge_piece_frames._TABLE["aoe2"]
    assert bridge_piece_frames._TABLE["aoc"] == {}


@pytest.mark.parametrize("dataset", [game_dataset.AOE2, game_dataset.ROR])
def test_every_row_matches_its_committed_graphic_and_frame_count(dataset):
    with game_dataset.activated(dataset):
        for const in TWELVE:
            assert bridge_piece_frames.covers(const), const
            assert len(bridge_piece_frames.frames(const)) == unit_variant.variant_count_for(const)


def test_every_end_row_holds_two_opposite_inward_vectors_along_its_long_axis():
    for const in TWELVE:
        row = bridge_piece_frames.frames(const)
        if row[0][0] != bridge_piece_frames.END:
            assert all(entry[0] != bridge_piece_frames.END for entry in row)
            continue
        span_x, span_y = tile_span(const, render_units.NON_BUILDING_SPAN)
        long_axis = (0, 1) if span_y > span_x else (1, 0)
        (_k0, v0), (_k1, v1) = row
        assert v1 == (-v0[0], -v0[1]), const
        assert abs(v0[0]) == long_axis[0] and abs(v0[1]) == long_axis[1], const


def test_each_pair_holds_the_same_kinds():
    for b, c in PAIRS:
        kinds = [sorted(entry[0] for entry in bridge_piece_frames.frames(const)) for const in (b, c)]
        assert kinds[0] == kinds[1], (b, c)


def test_the_measured_end_frames():
    """Typed again from the rendered frames (tall open end = inward), as the oracle."""
    end = bridge_piece_frames.END
    assert bridge_piece_frames.frames(1840) == ((end, (0, -1)), (end, (0, 1)))
    assert bridge_piece_frames.frames(1841) == ((end, (1, 0)), (end, (-1, 0)))
    assert bridge_piece_frames.frames(2039) == ((end, (0, 1)), (end, (0, -1)))


def test_image_frame_refuses_loudly():
    with pytest.raises(ValueError):
        bridge_piece_frames.image_frame(4, 0, 4, _identity)
    with pytest.raises(ValueError):
        bridge_piece_frames.image_frame(1840, 2, 1840, _identity)
    with pytest.raises(ValueError):
        bridge_piece_frames.image_frame(1840, 0, 1996, _identity)  # an end has no end frame to land on


def test_covers_is_false_when_the_art_differs(monkeypatch):
    assert bridge_piece_frames.covers(1840)
    monkeypatch.setitem(unit_variant._variant_counts(), 1840, 3)
    assert not bridge_piece_frames.covers(1840)
    monkeypatch.setitem(unit_variant._variant_counts(), 1840, 2)
    graphic = dict(unit_sprites.graphic_map()[1840], file_name="s_bridge_piece_other_x1")
    monkeypatch.setitem(unit_sprites.graphic_map(), 1840, graphic)
    assert not bridge_piece_frames.covers(1840)
    assert not bridge_piece_frames.covers(4)
