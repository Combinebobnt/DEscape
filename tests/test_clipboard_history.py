"""clipboard_history.py's pure half -- no Qt, no QApplication, RegionBlocks
built by hand. The dialog and window wiring are covered by
tests/test_clipboard_dialog.py, which needs a real ViewerWindow."""

from __future__ import annotations

import numpy as np

from descape import clipboard_history, terrain_palette
from descape.region_clipboard import RegionBlock


def _block(width: int = 2, height: int = 2, units: tuple = (), terrain_ids=None) -> RegionBlock:
    count = width * height
    ids = tuple(terrain_ids) if terrain_ids is not None else tuple(range(count))
    return RegionBlock(
        width=width,
        height=height,
        terrain_ids=ids,
        elevations=(0,) * count,
        layers=(-1,) * count,
        units=units,
    )


def _history_with(count: int) -> clipboard_history.ClipboardHistory:
    history = clipboard_history.ClipboardHistory()
    for index in range(count):
        block = _block()
        history.push(block, clipboard_history.thumbnail_rgb(block), f"entry {index}")
    return history


# -- push / eviction ---------------------------------------------------------


def test_push_puts_the_new_entry_first_and_activates_it() -> None:
    history = clipboard_history.ClipboardHistory()
    first = history.push(_block(), np.zeros((1, 1, 3), np.uint8), "a")
    second = history.push(_block(3, 3), np.zeros((1, 1, 3), np.uint8), "b")
    assert [e.entry_id for e in history.entries] == [second.entry_id, first.entry_id]
    assert history.active_id == second.entry_id
    assert history.active_block is second.block


def test_pushing_past_the_cap_evicts_the_tail_and_keeps_the_newest_active() -> None:
    history = _history_with(clipboard_history.MAX_ENTRIES + 3)
    assert len(history.entries) == clipboard_history.MAX_ENTRIES
    assert history.active_id == history.entries[0].entry_id
    assert history.entries[0].label == f"entry {clipboard_history.MAX_ENTRIES + 2}"


def test_entry_ids_are_monotonic_and_never_reused() -> None:
    """The property an index-based design cannot offer, and the reason every
    dialog callback passes an entry_id rather than a row."""
    history = _history_with(3)
    doomed = history.entries[1].entry_id
    assert history.remove(doomed)
    fresh = history.push(_block(), np.zeros((1, 1, 3), np.uint8), "new")
    assert fresh.entry_id != doomed
    assert fresh.entry_id > max(e.entry_id for e in history.entries if e is not fresh)


# -- active / delete ---------------------------------------------------------


def test_set_active_on_a_middle_entry_does_not_reorder() -> None:
    history = _history_with(3)
    order = [e.entry_id for e in history.entries]
    assert history.set_active(order[1])
    assert [e.entry_id for e in history.entries] == order
    assert history.active_block is history.entries[1].block


def test_set_active_and_rename_and_remove_reject_an_unknown_id() -> None:
    history = _history_with(2)
    assert not history.set_active(999)
    assert not history.rename(999, "x")
    assert not history.remove(999)


def test_deleting_the_active_entry_promotes_its_successor() -> None:
    history = _history_with(3)
    middle = history.entries[1].entry_id
    expected = history.entries[2].entry_id
    history.set_active(middle)
    assert history.remove(middle)
    assert history.active_id == expected


def test_deleting_the_active_last_entry_promotes_the_new_last() -> None:
    history = _history_with(3)
    last = history.entries[2].entry_id
    expected = history.entries[1].entry_id
    history.set_active(last)
    assert history.remove(last)
    assert history.active_id == expected


def test_deleting_the_only_entry_leaves_nothing_active() -> None:
    history = _history_with(1)
    assert history.remove(history.entries[0].entry_id)
    assert history.active_id is None
    assert history.active is None
    assert history.active_block is None


def test_deleting_a_non_active_entry_leaves_active_on_the_same_entry() -> None:
    """The discriminating test for id-vs-index: removing a row above the
    active one shifts its index but must not shift what is active."""
    history = _history_with(3)
    history.set_active(history.entries[2].entry_id)
    active_entry = history.active
    assert history.remove(history.entries[0].entry_id)
    assert history.active is active_entry


def test_clear_empties_and_deactivates() -> None:
    history = _history_with(3)
    history.clear()
    assert history.entries == []
    assert history.active_id is None
    assert history.active_block is None


def test_rename_changes_only_the_label() -> None:
    history = _history_with(2)
    target = history.entries[1]
    assert history.rename(target.entry_id, "shoreline")
    renamed = history.entries[1]
    assert renamed.label == "shoreline"
    assert renamed.entry_id == target.entry_id
    assert renamed.block is target.block


# -- labels and thumbnails ---------------------------------------------------


def test_default_label_reports_the_blocks_own_size_and_unit_count() -> None:
    assert clipboard_history.default_label(_block(8, 4)) == "8x4 (0 units)"
    assert clipboard_history.default_label(_block(2, 2, units=(object(), object()))) == (
        "2x2 (2 units)"
    )


def test_thumbnail_of_a_small_block_is_one_pixel_per_tile() -> None:
    """Compared against color_for_terrain_id itself, never literal RGB: it
    returns real texture averages when an AoE2:DE install is configured, so
    hardcoded numbers would pass on CI and fail on a dev box with the game
    installed."""
    block = _block(3, 2, terrain_ids=(0, 1, 2, 3, 4, 5))
    thumb = clipboard_history.thumbnail_rgb(block)
    assert thumb.dtype == np.uint8
    assert thumb.shape == (2, 3, 3)
    for y in range(2):
        for x in range(3):
            expected = terrain_palette.color_for_terrain_id(block.terrain_ids[y * 3 + x])
            assert tuple(thumb[y, x].tolist()) == tuple(expected)


def test_thumbnail_of_an_oversized_block_subsamples_without_averaging() -> None:
    size = clipboard_history.THUMBNAIL_MAX_TILES * 2
    block = _block(size, size, terrain_ids=tuple(range(size * size)))
    thumb = clipboard_history.thumbnail_rgb(block)
    assert thumb.shape == (
        clipboard_history.THUMBNAIL_MAX_TILES,
        clipboard_history.THUMBNAIL_MAX_TILES,
        3,
    )
    # Nearest-neighbour: output (y, x) reads source tile (y*2, x*2) exactly,
    # so its color is a real tile's color rather than a blend of two.
    for out_y, out_x in [(0, 0), (3, 7), (clipboard_history.THUMBNAIL_MAX_TILES - 1, 5)]:
        src = (out_y * 2) * size + (out_x * 2)
        expected = terrain_palette.color_for_terrain_id(block.terrain_ids[src])
        assert tuple(thumb[out_y, out_x].tolist()) == tuple(expected)


def test_thumbnail_preserves_a_non_square_aspect() -> None:
    cap = clipboard_history.THUMBNAIL_MAX_TILES
    block = _block(cap * 2, 4, terrain_ids=tuple(range(cap * 2 * 4)))
    assert clipboard_history.thumbnail_rgb(block).shape == (4, cap, 3)


def test_entries_compare_without_tripping_over_the_numpy_thumbnail() -> None:
    """The thumbnail field is compare=False precisely so this does not raise
    ValueError on an ambiguous array truth value."""
    history = _history_with(1)
    entry = history.entries[0]
    assert history.active == entry
    assert entry == entry
