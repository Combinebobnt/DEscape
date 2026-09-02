"""Undo/redo for terrain-tile, trigger, map-option, unit and Messages edits
(v2's, phase 4b's, the Map Options panel's, phase 3.5a's, and Messages
mode's write paths).

Deliberately Qt-free *and* library-free -- this is the single choke point every
terrain/elevation edit tool goes through, and keeping it importable
without a running QApplication is what makes it possible to test the undo/redo
invariants headlessly (see tests/test_write_path.py's test_history_invariants,
and tests/test_edit_history.py for the fake-tile unit-level invariants).
Trigger and unit records hold opaque snapshot objects and hand them straight
back to TriggerEditModel.restore()/UnitEditModel.restore(); no
AoE2ScenarioParser type is named here at runtime, only under TYPE_CHECKING.

One history, not two. It is the sole source of truth for is_dirty,
mark_saved(), can_undo/can_redo and viewer.py's _confirm_discard_changes, so a
second instance for triggers, units, or Messages would leave the document
closable without a save prompt. The contract that keeps this true: **any
mutation that sets trigger, map-option, unit, or Messages model dirtiness
must push a record here** (see tests/test_trigger_undo.py,
tests/test_options_undo.py, tests/test_units_undo.py, and
tests/test_messages_undo.py).

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Sequence

if TYPE_CHECKING:  # keeps this module library-free at runtime -- see docstring
    from descape.messages_model import MessagesEditModel
    from descape.options_model import OptionsEditModel
    from descape.trigger_model import TriggerEditModel, TriggerSnapshot
    from descape.unit_model import UnitEditModel, UnitSnapshot

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
    """One undo step, in whichever domain the subclass covers.

    Records own their own restore logic rather than EditHistory switching on
    them, which is what lets EditHistory stay a plain cursor-mover and keeps
    this module free of both Qt and AoE2ScenarioParser.

    `kind` is the discriminator viewer.py branches on via peek_undo()/
    peek_redo() -- it has to decide whether a repaint or a trigger-panel
    refresh is called for *before* the cursor moves.
    """

    label: str
    kind: ClassVar[str] = "base"

    def touched_indices(self) -> list[int]:
        """Tile indices this record changed. Empty for domains that touch no
        tiles -- viewer.py's _apply_dirty() no-ops on an empty list."""
        return []

    def require_target(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> None:
        """Raises if this record cannot be applied against what the caller
        passed. Called before the cursor moves, so a refused undo leaves the
        history exactly where it was rather than silently desynced."""

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        raise NotImplementedError

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        raise NotImplementedError


@dataclass
class TileDiffRecord(DiffRecord):
    """Everything one brush stroke (or other atomic tile edit) changed.
    changes is a list of (tile_index, old_state, new_state) -- only tiles that
    actually changed, not the whole map."""

    changes: list[tuple[int, TileState, TileState]]
    kind: ClassVar[str] = "tile"

    def touched_indices(self) -> list[int]:
        return [i for i, _old, _new in self.changes]

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        for i, old, _new in self.changes:
            tiles[i].terrain_id, tiles[i].elevation, tiles[i].layer = old
        return self.touched_indices()

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        for i, _old, new in self.changes:
            tiles[i].terrain_id, tiles[i].elevation, tiles[i].layer = new
        return self.touched_indices()


@dataclass
class TriggerDiffRecord(DiffRecord):
    """One trigger edit, as a pair of TriggerEditModel snapshots.

    Byte snapshots of the Triggers section were the original design and cannot
    work: bytes restore neither the live object graph the trigger panel reads
    from nor the blob dirty state a save branches on, so an undone edit would
    save through the re-serializing branch and stop being byte-identical.

    `touched` is trigger indices for the panel refresh, deliberately *not*
    returned from undo()/redo() -- those always return tile indices, so that
    viewer.py's existing _apply_dirty() pipe and the five other callers that
    read the return value as a plain tile-index list keep working unchanged.
    """

    before: TriggerSnapshot
    after: TriggerSnapshot
    touched: list[int] = field(default_factory=list)
    kind: ClassVar[str] = "trigger"

    def require_target(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> None:
        if triggers is None:
            raise RuntimeError(
                f"Undo record {self.label!r} is a trigger edit, but no TriggerEditModel "
                f"was passed -- refusing to move the cursor past an edit that cannot be "
                f"restored"
            )

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        triggers.restore(self.before)
        return []

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        triggers.restore(self.after)
        return []


@dataclass
class OptionsDiffRecord(DiffRecord):
    """One map-option scalar edit, as the raw before/after values of one field.

    Values rather than a snapshot, unlike TriggerDiffRecord: an option field is
    one fixed-width integer with no object graph behind it, and
    OptionsEditModel.set_value() is idempotent, so replaying the "before" value
    is a complete restore.

    Covers only the byte-patched scalars. The trigger execution-order row looks
    identical in the panel but is *not* one of these -- it lives inside the
    Triggers region and rides on TriggerEditModel, so its record is a
    TriggerDiffRecord. That is also what makes it restore together with
    trigger_display_order, which item 2's execution-order invariant couples it
    to.
    """

    field_id: str
    before: int
    after: int
    kind: ClassVar[str] = "options"

    def require_target(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> None:
        if options is None:
            raise RuntimeError(
                f"Undo record {self.label!r} is a map-option edit, but no OptionsEditModel "
                f"was passed -- refusing to move the cursor past an edit that cannot be "
                f"restored"
            )

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        options.set_value(self.field_id, self.before)
        return []

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        options.set_value(self.field_id, self.after)
        return []


@dataclass
class UnitDiffRecord(DiffRecord):
    """One unit edit (place/remove/move/reassign), as a pair of
    UnitEditModel snapshots -- same shape as TriggerDiffRecord, for the same
    reason: a byte-snapshot record cannot restore the live object graph the
    model reads from or the blob dirty state a save branches on."""

    before: UnitSnapshot
    after: UnitSnapshot
    kind: ClassVar[str] = "unit"

    def require_target(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> None:
        if units is None:
            raise RuntimeError(
                f"Undo record {self.label!r} is a unit edit, but no UnitEditModel "
                f"was passed -- refusing to move the cursor past an edit that cannot be "
                f"restored"
            )

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        units.restore(self.before)
        return []

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        units.restore(self.after)
        return []


@dataclass
class MessagesDiffRecord(DiffRecord):
    """One Messages mode field edit (a text field or a string-table id), as
    the raw before/after values of one field -- modelled directly on
    OptionsDiffRecord: MessagesEditModel.set_value() is idempotent, so
    replaying "before" is a complete restore, and there is no object graph
    to snapshot the way trigger/unit edits need.

    `before`/`after` are `str | int` since one record type covers both the
    text fields and their string-table id counterparts (field ids
    "instructions".."scouts" vs. "instructions_id".."scouts_id" -- see
    descape/messages_model.py).
    """

    field_id: str
    before: str | int
    after: str | int
    kind: ClassVar[str] = "messages"

    def require_target(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> None:
        if messages is None:
            raise RuntimeError(
                f"Undo record {self.label!r} is a Messages mode edit, but no MessagesEditModel "
                f"was passed -- refusing to move the cursor past an edit that cannot be "
                f"restored"
            )

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        messages.set_value(self.field_id, self.before)
        return []

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        messages.set_value(self.field_id, self.after)
        return []


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

        self._push(TileDiffRecord(label, changes))
        return [i for i, _old, _new in changes]

    def _push(self, record: DiffRecord) -> None:
        """Redo truncation, append, cursor advance, and overflow bookkeeping --
        shared by every push path. Extracted rather than duplicated because a
        second push path that reimplements the saved_at_cursor arithmetic will
        get it subtly wrong, and the symptom (a file that silently stops
        reading as dirty) shows up nowhere near the bug."""
        del self.records[self.cursor :]
        self.records.append(record)
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

    def push_trigger_record(self, record: TriggerDiffRecord) -> None:
        """The trigger side's way in. Deliberately a plain push rather than a
        begin/commit pair here: TriggerEditModel owns the snapshot timing (it
        has to capture the "before" side *before* the mutation runs), so by the
        time a record exists there is nothing left for this module to diff."""
        self._push(record)

    def push_options_record(self, record: OptionsDiffRecord) -> None:
        """The map-options side's way in. A plain push for a simpler reason
        than the trigger side's: the caller already knows both values, so there
        is nothing to snapshot or diff."""
        self._push(record)

    def push_unit_record(self, record: UnitDiffRecord) -> None:
        """The unit side's way in. A plain push, same reason as the trigger
        side's: UnitEditModel owns the snapshot timing (begin_unit_edit()
        captures the "before" side before the mutation runs)."""
        self._push(record)

    def push_messages_record(self, record: MessagesDiffRecord) -> None:
        """The Messages side's way in. A plain push, same reason as the
        map-options side's: the caller already knows both values, so there
        is nothing to snapshot or diff."""
        self._push(record)

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

    def peek_undo(self) -> DiffRecord | None:
        """The record undo() would apply, without moving the cursor -- so
        viewer.py can branch on record.kind (repaint vs. trigger-panel
        refresh) *before* committing to the undo."""
        return self.records[self.cursor - 1] if self.can_undo else None

    def peek_redo(self) -> DiffRecord | None:
        return self.records[self.cursor] if self.can_redo else None

    def undo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None = None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        """Always returns *tile* indices -- empty for a trigger, option, or
        unit record.

        Deliberately not a discriminated return type: five call sites outside
        viewer.py already consume this as a plain tile-index list
        (tests/test_flat_chunks.py, tools/verify_iso_incremental.py,
        tools/verify_iso_units.py), and _apply_dirty([]) is already a safe
        no-op. The trigger/unit side rides on the record instead, via
        peek_undo()'s .kind and .touched.
        """
        if not self.can_undo:
            return []
        record = self.records[self.cursor - 1]
        record.require_target(tiles, triggers, options, units, messages)  # must raise before the cursor moves
        self.cursor -= 1
        return record.undo(tiles, triggers, options, units, messages)

    def redo(
        self,
        tiles: Sequence,
        triggers: TriggerEditModel | None = None,
        options: OptionsEditModel | None = None,
        units: UnitEditModel | None = None,
        messages: MessagesEditModel | None = None,
    ) -> list[int]:
        if not self.can_redo:
            return []
        record = self.records[self.cursor]
        record.require_target(tiles, triggers, options, units, messages)  # must raise before the cursor moves
        self.cursor += 1
        return record.redo(tiles, triggers, options, units, messages)
