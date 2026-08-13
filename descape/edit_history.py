"""Undo/redo for terrain-tile edits (v2's write path).

Deliberately Qt-free -- this is the single choke point every terrain/elevation
edit tool goes through, and keeping it importable
without a running QApplication is what makes it possible to test the undo/redo
invariants headlessly (see tests/test_write_path.py's test_history_invariants,
and tests/test_edit_history.py for the fake-tile unit-level invariants).

The core rule this module exists to enforce: nothing may mutate a TerrainTile's
terrain_id/elevation/layer except through this module's begin_stroke/
commit_stroke pair (or their apply() convenience wrapper). A tool that writes
to a tile directly produces an edit that's silently un-undoable and leaves the
on-screen render stale -- both invisible until a user hits Ctrl+Z.

Two entry points, for two callers:
- apply(label, tiles, mutate_fn) -- one-shot: snapshot, run mutate_fn(), diff,
  record. Used by tests and any non-interactive caller.
- begin_stroke()/stroke_dirty_indices()/commit_stroke() -- split apart so
  viewer.py's drag-painting can apply a brush per newly-touched tile *live*
  (querying stroke_dirty_indices() after each touch to know what to
  incrementally re-render) while still recording the whole drag as a single
  undo step at release. apply() is implemented on top of these three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

# Proportional to tiles actually changed, not map size, so ordinary edits stay
# small -- but MapManager.set_elevation's propagation can touch a large region
# from one click, so an explicit cap keeps a long editing session bounded.
# An *extensive* buffer is called for ("enough history that undoing all
# the way back is realistic"), hence hundreds of operations, not tens.
DEFAULT_MAX_RECORDS = 500

# (terrain_id, elevation, layer) -- the three writable TerrainTile fields.
TileState = tuple[int, int, int]


def tile_state(tile) -> TileState:
    return (tile.terrain_id, tile.elevation, tile.layer)


@dataclass
class DiffRecord:
    """One undo step: everything one brush stroke (or other atomic edit)
    changed. changes is a list of (tile_index, old_state, new_state) --
    only tiles that actually changed, not the whole map."""

    label: str
    changes: list[tuple[int, TileState, TileState]]

    def touched_indices(self) -> list[int]:
        return [i for i, _old, _new in self.changes]


class EditHistory:
    """A cursor into a list of DiffRecords, not a pop-stack -- undo/redo move
    the cursor rather than destroying data, which is what keeps "jump to
    history entry N" (a later, explicitly out-of-v2-scope UI) just a matter of
    replaying records between the current cursor and N, instead of a redesign.
    """

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS):
        self.max_records = max_records
        self.records: list[DiffRecord] = []
        self.cursor = 0
        # Cursor position at the last successful save. None means "the saved
        # state has fallen off the front of a capped history" -- see apply()
        # -- i.e. no cursor position can currently reconstruct it, so treat
        # the file as dirty regardless of cursor until the next save.
        self.saved_at_cursor: int | None = 0
        # Set only between begin_stroke() and commit_stroke()/abort_stroke().
        self._stroke_before: list[TileState] | None = None

    def reset(self) -> None:
        """Back to a freshly-loaded, clean state -- called by load_scenario()
        and close_scenario() so history never leaks across files."""
        self.records.clear()
        self.cursor = 0
        self.saved_at_cursor = 0
        self._stroke_before = None

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.records)

    @property
    def is_dirty(self) -> bool:
        return self.saved_at_cursor != self.cursor

    def mark_saved(self) -> None:
        """Call after a successful Save As. Never call this after a failed
        save -- the on-disk file didn't change, so the dirty state shouldn't
        either."""
        self.saved_at_cursor = self.cursor

    def begin_stroke(self, tiles: Sequence) -> None:
        """Snapshots current tile state. Must be paired with exactly one of
        commit_stroke() or abort_stroke() before begin_stroke() is called
        again."""
        if self._stroke_before is not None:
            raise RuntimeError("begin_stroke() called while a stroke was already in progress")
        self._stroke_before = [tile_state(t) for t in tiles]

    def stroke_dirty_indices(self, tiles: Sequence) -> list[int]:
        """Cumulative set of tile indices changed since begin_stroke() --
        for live incremental re-render mid-drag (see viewer.py's stroke
        handling: it calls this after every newly-touched tile and diffs
        against what it saw last time, to redraw only what's newly dirty).
        A plain linear scan is cheap enough at this project's map sizes
        (tens of thousands of tiles) to call once per touched tile."""
        if self._stroke_before is None:
            raise RuntimeError("stroke_dirty_indices() called with no stroke in progress")
        before = self._stroke_before
        return [i for i, t in enumerate(tiles) if tile_state(t) != before[i]]

    def abort_stroke(self) -> None:
        """Discards the in-progress snapshot without recording anything.
        Not used by v2's UI (which always commits on release or defensively
        on leave -- see viewer.py) but kept as the correct, explicit way to
        bail out of an in-progress stroke if a future caller needs to."""
        self._stroke_before = None

    def commit_stroke(self, label: str, tiles: Sequence) -> list[int]:
        """Diffs current tile state against begin_stroke()'s snapshot and
        pushes one DiffRecord covering everything that changed, ending the
        in-progress stroke.

        Returns the list of changed tile indices -- empty if nothing actually
        changed, in which case *no* record is pushed. That matters: a stroke
        that repaints a tile with the terrain it already has must not create
        a phantom undo step, or Ctrl+Z appears to do nothing.

        Committing while the cursor isn't at the tip (i.e. after some
        undo(s)) discards every record after the cursor first -- standard
        redo invalidation, easy to forget, produces a corrupt timeline if
        missed.
        """
        if self._stroke_before is None:
            raise RuntimeError("commit_stroke() called with no stroke in progress")
        before = self._stroke_before
        self._stroke_before = None

        changes: list[tuple[int, TileState, TileState]] = []
        for i, t in enumerate(tiles):
            new = tile_state(t)
            if new != before[i]:
                changes.append((i, before[i], new))

        if not changes:
            return []

        del self.records[self.cursor :]
        self.records.append(DiffRecord(label, changes))
        self.cursor += 1

        overflow = len(self.records) - self.max_records
        if overflow > 0:
            del self.records[:overflow]
            self.cursor -= overflow
            if self.saved_at_cursor is not None:
                self.saved_at_cursor -= overflow
                if self.saved_at_cursor < 0:
                    # The saved state was among the dropped records -- no
                    # cursor position can reconstruct it anymore, so the file
                    # must read as dirty until the next save regardless of
                    # where the cursor lands.
                    self.saved_at_cursor = None

        return [i for i, _old, _new in changes]

    def apply(self, label: str, tiles: Sequence, mutate_fn: Callable[[], None]) -> list[int]:
        """Convenience one-shot wrapper around begin_stroke()/commit_stroke():
        snapshot, run mutate_fn() (expected to mutate some of `tiles` in place
        -- e.g. via MapManager.set_elevation or a direct terrain_id
        assignment), diff, record. Used by tests and any non-interactive
        caller; viewer.py's drag-painting uses begin_stroke/
        stroke_dirty_indices/commit_stroke directly instead, for live
        per-tile feedback during a drag rather than only at its end."""
        self.begin_stroke(tiles)
        mutate_fn()
        return self.commit_stroke(label, tiles)

    def undo(self, tiles: Sequence) -> list[int]:
        if not self.can_undo:
            return []
        self.cursor -= 1
        record = self.records[self.cursor]
        for i, old, _new in record.changes:
            tiles[i].terrain_id, tiles[i].elevation, tiles[i].layer = old
        return record.touched_indices()

    def redo(self, tiles: Sequence) -> list[int]:
        if not self.can_redo:
            return []
        record = self.records[self.cursor]
        for i, _old, new in record.changes:
            tiles[i].terrain_id, tiles[i].elevation, tiles[i].layer = new
        self.cursor += 1
        return record.touched_indices()
