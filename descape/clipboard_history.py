"""The Select tool's clipboard HISTORY -- the last few Copy Region snapshots,
rather than the single slot phase 2.8 shipped.

Qt-free, in the same sense region_clipboard.py is: plain data plus pure
functions, importable and fully testable with no QApplication. That file is
copy/paste *semantics*; this one is the session-level *collection*, which is
also why the thumbnail builder lives here -- a thumbnail is presentation, not
payload, and RegionBlock's own shape must not change to carry one.

Session-only by design. Nothing here is written to disk: settings.py rewrites
the whole YAML on every settings change, and a Select All on a 480x480 map is
~230k tiles, which has no business in a config file. The paste-filter
checkboxes are deliberately session-only for the same class of reason.

Rules pinned here so they are not re-litigated:

- entry_id, never a row index, is the dialog's currency. Every callback passes
  an entry_id and this module resolves it, which is what lets a Ctrl+C on the
  main window repopulate an open dialog without losing the user's selected row.
- push() inserts at index 0 and activates it, then truncates to MAX_ENTRIES, so
  eviction can never orphan the active entry: the just-pushed one is active by
  construction.
- Delete-active successor rule, one rule: the entry that now occupies the
  deleted row becomes active; if the deleted row was last, the new last does;
  None once the list is empty.
- No dedupe. RegionBlock is a frozen dataclass of tuples so == would work, but
  re-copying something to bring it back to the top is a legitimate gesture, and
  collapsing it would make the list lie about what the user did.
- MAX_ENTRIES is a memory bound, not a UI one -- ten whole-map copies is the
  ceiling it chooses. If it ever bites, lower it rather than adding disk
  persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from descape import terrain_palette
from descape.region_clipboard import RegionBlock

MAX_ENTRIES = 10
THUMBNAIL_MAX_TILES = 64


@dataclass(frozen=True)
class ClipboardEntry:
    entry_id: int
    block: RegionBlock
    label: str
    # compare=False, not decoration: the generated __eq__ would compare the
    # numpy thumbnail, and array == array returns an array rather than a bool,
    # raising ValueError the first time two entries are compared at all.
    thumbnail: np.ndarray = field(compare=False)


def default_label(block: RegionBlock) -> str:
    return f"{block.width}x{block.height} ({len(block.units)} units)"


def thumbnail_rgb(block: RegionBlock, max_tiles: int = THUMBNAIL_MAX_TILES) -> np.ndarray:
    """A flat-color terrain preview of `block` as an (h, w, 3) uint8 array --
    one pixel per tile, pixel (y, x) being color_for_terrain_id() of that
    tile's terrain. No isometric projection, no unit sprites, no elevation
    shading: this exists to tell one copied region from another in a list.

    numpy rather than a QImage so the whole thumbnail stays assertable
    headlessly; the dialog converts it at display time.

    A block larger than max_tiles on either axis is NEAREST-NEIGHBOUR
    subsampled (integer index mapping, no averaging) so flat terrain colors
    stay flat -- averaging two adjacent terrains would invent a third color
    that is on neither tile. Smaller blocks emit one pixel per tile and are
    scaled up by Qt. Cost is bounded at max_tiles^2 lookups regardless of
    region size, and color_for_terrain_id delegates to an lru_cache'd
    asset_source.get_terrain_average_color(), so repeats are dict hits."""
    src_w, src_h = block.width, block.height
    out_w = min(src_w, max_tiles)
    out_h = min(src_h, max_tiles)
    if out_w <= 0 or out_h <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    xs = (np.arange(out_w) * src_w) // out_w
    ys = (np.arange(out_h) * src_h) // out_h
    thumb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for out_y, src_y in enumerate(ys.tolist()):
        row = src_y * src_w
        for out_x, src_x in enumerate(xs.tolist()):
            thumb[out_y, out_x] = terrain_palette.color_for_terrain_id(
                block.terrain_ids[row + src_x]
            )
    return thumb


class ClipboardHistory:
    """Newest first: entries[0] is the most recent copy."""

    def __init__(self) -> None:
        self.entries: list[ClipboardEntry] = []
        self.active_id: int | None = None
        self._next_id = 1

    @property
    def active(self) -> ClipboardEntry | None:
        if self.active_id is None:
            return None
        for entry in self.entries:
            if entry.entry_id == self.active_id:
                return entry
        return None

    @property
    def active_block(self) -> RegionBlock | None:
        entry = self.active
        return None if entry is None else entry.block

    def _index_of(self, entry_id: int) -> int | None:
        for index, entry in enumerate(self.entries):
            if entry.entry_id == entry_id:
                return index
        return None

    def push(self, block: RegionBlock, thumbnail: np.ndarray, label: str) -> ClipboardEntry:
        entry = ClipboardEntry(self._next_id, block, label, thumbnail)
        self._next_id += 1
        self.entries.insert(0, entry)
        self.active_id = entry.entry_id
        del self.entries[MAX_ENTRIES:]
        return entry

    def set_active(self, entry_id: int) -> bool:
        if self._index_of(entry_id) is None:
            return False
        self.active_id = entry_id
        return True

    def remove(self, entry_id: int) -> bool:
        index = self._index_of(entry_id)
        if index is None:
            return False
        was_active = entry_id == self.active_id
        del self.entries[index]
        if was_active:
            # The successor rule: whoever now occupies that row, else the new
            # last row, else nothing left to be active.
            if not self.entries:
                self.active_id = None
            else:
                self.active_id = self.entries[min(index, len(self.entries) - 1)].entry_id
        return True

    def rename(self, entry_id: int, label: str) -> bool:
        index = self._index_of(entry_id)
        if index is None:
            return False
        self.entries[index] = replace(self.entries[index], label=label)
        return True

    def clear(self) -> None:
        self.entries.clear()
        self.active_id = None
