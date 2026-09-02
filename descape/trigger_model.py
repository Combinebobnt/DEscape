"""Phase 4b's byte-blob-per-trigger model: what makes editing one trigger
produce a diff of exactly that trigger.

Why not just re-serialize the Triggers section. Measured against the real
examples/ corpus: AoE2ScenarioParser's whole-section get_data_as_bytes()
is not byte-identical to
the bytes it parsed, on every corpus file that has triggers at all. Deltas run
from -41 to +5310 bytes, and on one file a MODIFY_OBJECT_ATTRIBUTE effect's int
-1 sentinel is silently re-typed to f32 -1.0, a semantic write to a field the
user never touched.

Most of the volume is one small thing: the game writes an empty str32 as a bare
length-0 field, while the library appends a NUL trail and writes length 1. One
byte per empty string, and empty strings are everywhere (145 of 167 triggers
drift in F7_3_York, 560 of 590 in old-allies-final-v2, 103 of 103 in
atilla_1_scn_resaved). The shipped trigger fixture reproduces that on all four
of its triggers, which is the only reason the default tier can tell this design
apart from whole-section re-serialization -- see tools/gen_trigger_fixture.py's
_game_style_bytes().

So this module keeps, per trigger, both the object graph and the original byte
slice it parsed from. On save an untouched trigger splices back verbatim and an
edited one goes through the library. Every normalization artifact is confined
to triggers the user actually edited, and a save with no trigger edits at all
is byte-identical to what descape/scenario_write.py already produced before
phase 4b existed.

Two invariants worth stating outright, because breaking either is silent:

1. **Never hold a TriggerManager across calls.** The library's field gating is
   class-level and process-global, so parsing a second scenario poisons the
   classes a manager for the first one reads through. Every method here goes
   back through scenario_io.parse_triggers(), which is memoized (no re-parse)
   and depoisons on the way. See that function's docstring.
2. **A reference remap dirties more triggers than the one that moved.**
   remove_triggers()/reorder_triggers() rewrite (de)activate-trigger references
   across the whole list. A trigger whose references were rewritten is no
   longer described by its original bytes even though the user never selected
   it, so it has to be marked dirty or the splice writes stale ids back.
"""

from __future__ import annotations

import copy
import struct
from dataclasses import dataclass, field
from typing import Sequence

from AoE2ScenarioParser.objects.managers.trigger_manager import (
    TriggerManager,
    get_trigger_referencing_ce,
)

from descape.edit_history import EditHistory, TriggerDiffRecord
from descape.scenario_io import LoadedScenario, parse_triggers, retriever_length

# trigger_version f64 + trigger_instruction_start s8 + number_of_triggers s32,
# in that order, in all 19 DE structure versions.
SECTION_HEADER_SIZE = 13
_TRIGGER_COUNT_STRUCT = struct.Struct("<i")  # s32, per versions/DE/*/structure.json
_DISPLAY_ORDER_STRUCT = struct.Struct("<I")  # u32, same source


class TriggerEditsUnavailableError(Exception):
    """Raised by TriggerEditModel() for a file whose triggers cannot be edited:
    the section did not parse (the 1.54/trigger-3.9 set), or it parsed but
    failed the alignment gate (v1.36/1.37, where the library models no Files
    section and a large remainder goes unconsumed)."""


@dataclass(frozen=True)
class SectionRegions:
    """Byte offsets within the Triggers section, as parsed.

    Built once from the retriever map before anything commits, since a commit
    rewrites the retrievers this is measured from. Everything from
    `variables_end` on is the version-dependent tail (useless_trigger_data,
    unknown_bytes2, redacted, legacy_exec_order on 1.55+), which always splices
    verbatim -- nothing in this phase edits it.
    """

    triggers_start: int  # == SECTION_HEADER_SIZE
    triggers_end: int
    display_order_end: int
    unknown_bytes_end: int
    variables_end: int
    section_end: int
    # Start offset of legacy_exec_order, the one byte of that tail this module
    # does edit (the Map Options panel's trigger execution-order row). None
    # when the file does not store it: its SET_REPEAT eval is
    # `1 if trigger_version >= 4.5 else 0`, so it consumes zero bytes on
    # F7_3_York despite that file being scenario version 1.55. Found by a
    # forward walk rather than assumed to be the section's last byte -- on such
    # a file the last byte belongs to `redacted`, and patching it would corrupt
    # a field nothing here understands.
    exec_order_offset: int | None


def _map_regions(section) -> SectionRegions:
    lengths = {name: retriever_length(r) for name, r in section.retriever_map.items()}
    offsets = {}
    pos = 0
    for name, length in lengths.items():
        pos += length
        offsets[name] = pos
    if offsets["number_of_triggers"] != SECTION_HEADER_SIZE:
        raise TriggerEditsUnavailableError(
            f"Triggers section header is {offsets['number_of_triggers']} bytes, "
            f"expected {SECTION_HEADER_SIZE} -- structure version changed"
        )
    exec_order_length = lengths.get("legacy_exec_order", 0)
    return SectionRegions(
        triggers_start=SECTION_HEADER_SIZE,
        triggers_end=offsets["trigger_data"],
        display_order_end=offsets["trigger_display_order_array"],
        unknown_bytes_end=offsets["unknown_bytes"],
        variables_end=offsets["variable_data"],
        section_end=pos,
        exec_order_offset=(
            offsets["legacy_exec_order"] - exec_order_length if exec_order_length == 1 else None
        ),
    )


def exec_order_write_supported(loaded: LoadedScenario) -> bool:
    """Whether legacy_exec_order can be written back for `loaded`, answerable
    without building a TriggerEditModel.

    Its own gate, separate from options_model's: that module explicitly skips
    the Triggers section, so verify_options_block() says nothing about this
    byte, and trigger_write_supported says nothing about the byte-patched rows.
    A file whose Map anchor failed can still legitimately edit this flag, and a
    file that failed the trigger alignment gate must not while its other rows
    are fine. The Map Options panel needs the answer at populate time, before
    any model exists.

    Fails closed on every step: no parse, a header the walk does not recognise,
    a version that stores no such byte, or a byte that does not match what the
    retriever parsed.
    """
    if not loaded.trigger_write_supported:
        return False
    section = loaded._scenario.sections.get("Triggers")
    if section is None:  # parse_triggers() has not run
        return False
    try:
        regions = _map_regions(section)
    except TriggerEditsUnavailableError:
        return False
    if regions.exec_order_offset is None:
        return False
    original = loaded.decompressed_body[loaded.units_section_end : loaded.triggers_section_end]
    if len(original) != regions.section_end:
        return False
    offset = regions.exec_order_offset
    if offset < regions.variables_end:
        # The flag is meant to live in the verbatim-spliced tail. Anywhere else
        # means the walk found a layout this module was not written for.
        return False
    return (
        original[offset : offset + 1]
        == section.retriever_map["legacy_exec_order"].get_data_as_bytes()
    )


def exec_order_value(loaded: LoadedScenario) -> int | None:
    """The file's own legacy_exec_order byte, read directly off the parsed
    section retriever -- module-level and answerable without building a
    TriggerEditModel, for the same "before any model exists" reason
    exec_order_write_supported() is.

    The Triggers-mode status line uses this for its read-only execution-order
    readout, which must not build an edit model just to show it --
    tests/test_trigger_panel.py's test_browsing_every_row_saves_byte_identically
    pins "browsing must not build an edit model", and the window's own
    _pending_option_values() only reaches `exec_order` through the lazily
    constructed self.trigger_edits, which is the wrong path for a passive
    readout.

    None when the file stores no such byte at all (trigger_version < 4.5, see
    exec_order_write_supported()'s docstring) -- distinct from the flag being
    a real 0 or 1.
    """
    section = loaded._scenario.sections.get("Triggers")
    if section is None:
        return None
    retriever = section.retriever_map.get("legacy_exec_order")
    if retriever is None or retriever_length(retriever) == 0:
        return None
    return retriever.data


def _variable_signature(manager: TriggerManager) -> tuple[tuple[int, str], ...]:
    """The variable block's content, for spotting a change to it.

    Variables live outside the trigger list, so nothing about the trigger
    reconciliation notices them: an add_variable() that isn't detected here
    leaves the old variable region spliced back verbatim, old count included,
    and the new variable silently vanishes on save.
    """
    return tuple((variable.variable_id, variable.name) for variable in manager.variables)


@dataclass
class TriggerSnapshot:
    """One side of a trigger undo record: everything needed to put the
    document back, and nothing more.

    The expensive-looking option was measured and rejected. On the worst corpus
    file (old-allies-final-v2: 590 triggers, 1.16 MB Triggers section) a
    whole-graph deepcopy costs 2.97 s per record and a section re-parse 3.53 s,
    against 1.7 ms for deepcopying only the triggers whose *content* changed.
    So `triggers` holds plain references (membership and order, ~8 bytes each)
    and `states` holds deepcopies in the one or two slots that need them.

    Why `ref_ids` exists separately: remove_triggers()/reorder_triggers()
    rewrite `trigger_id` on surviving triggers and `ce.trigger_id` on every
    trigger referencing them, in place. Auditing those two methods, that pair
    is the *only* thing they write to a pre-existing trigger -- so the whole
    id-remap class restores from these two small lists with no deepcopy at all.

    `variables_dirty` is not optional bookkeeping: without it, undoing a
    variable add leaves the model's flag True, so the next save re-serializes a
    variable block the document no longer edits and reintroduces NUL-trail
    drift across every variable name in an otherwise clean document.

    Unlike `triggers`, `variables` is always a full deep copy rather than a
    list of live references: the list is capped at 256 entries (see
    viewer.py's `_MAX_VARIABLES`), so there is no equivalent of the
    per-trigger deepcopy budget to worry about, and a rename mutates a
    `Variable` object's `name` in place rather than replacing it in the list.
    A snapshot holding references to those same objects would have its
    "before" name overwritten by the very edit it exists to undo.
    """

    triggers: list = field(default_factory=list)
    ref_ids: list[tuple[int, ...]] = field(default_factory=list)
    states: list = field(default_factory=list)
    trigger_ids: list[int] = field(default_factory=list)
    display_order: list[int] = field(default_factory=list)
    blobs: list[bytes | None] = field(default_factory=list)
    variables: list | None = None
    variables_dirty: bool = False
    # Restored for the same reason `variables_dirty` is: left True through an
    # undo, it keeps the next save on the re-serializing branch for a document
    # that is no longer edited at all.
    structure_dirty: bool = False
    # The pending legacy_exec_order override, or None for "the file's own
    # byte". Both the value and the fact that it is set have to travel in the
    # snapshot: this is the whole of that edit's state, so an undo that did not
    # restore it would leave the flag flipped with no record left to undo it
    # by, and a redo would have nothing to reapply.
    exec_order: int | None = None


def _reference_signature(trigger) -> tuple[int, ...]:
    """The trigger ids this trigger points at, in order.

    This is what a remap rewrites, and comparing it before and after a
    structural operation is how the model finds every trigger that went stale
    without having to reimplement the library's remap rules.
    """
    return tuple(ce.trigger_id for ce in get_trigger_referencing_ce(trigger))


def display_order_with_copy_inserted(
    before_triggers: Sequence,
    before_order: Sequence[int],
    after_triggers: Sequence,
    source_index: int,
) -> list[int]:
    """Recompute trigger_display_order after copy_trigger(source_index), which
    otherwise silently discards a custom order.

    copy_trigger()'s append_after_source path calls move_triggers(), which
    calls reorder_triggers(), which assigns `manager.triggers = [...]` --
    resetting trigger_display_order to identity and renumbering every
    trigger_id in the process (see structural_edit()'s docstring on the
    id-remap class). So every entry in `before_order` is a *stale* id by the
    time this runs; it has to be translated through object identity, the same
    reconciliation structural_edit() itself uses.

    `before_triggers`/`after_triggers` are the pre- and post-edit
    `manager.triggers` lists (order does not matter here, only membership and
    identity). The new trigger -- the one object in `after_triggers` with no
    match in `before_triggers` -- is inserted directly after the source's
    display slot, matching where copy_trigger() places it in the physical
    list (`viewer.py`'s trigger_structural_edit(): "the copy lands at
    index + 1, not at the end").
    """
    new_index_of = {id(t): i for i, t in enumerate(after_triggers)}
    before_ids = {id(t) for t in before_triggers}
    remapped = [new_index_of[id(before_triggers[old_index])] for old_index in before_order]
    copy_new_index = next(i for i, t in enumerate(after_triggers) if id(t) not in before_ids)
    source_new_index = new_index_of[id(before_triggers[source_index])]
    remapped.insert(remapped.index(source_new_index) + 1, copy_new_index)
    return remapped


def moved_display_order(order: Sequence[int], trigger_index: int, delta: int) -> list[int]:
    """A new trigger_display_order with the trigger at list index
    `trigger_index` moved by `delta` display slots (+1 = Move Down, -1 = Move
    Up).

    Reordering means permuting this array only -- ids are never renumbered,
    so `trigger_index` is the trigger's stable list index, not its current
    display slot; `order.index(trigger_index)` finds the slot to move.
    Raises IndexError if `trigger_index` is not in `order`, or if `delta`
    would move it past either end -- the caller (button enablement) is
    expected to prevent that, not this function.
    """
    moved = list(order)
    slot = moved.index(trigger_index)
    target = slot + delta
    if not 0 <= target < len(moved):
        raise IndexError(
            f"cannot move display slot {slot} by {delta}: only {len(moved)} slots exist"
        )
    moved[slot], moved[target] = moved[target], moved[slot]
    return moved


class TriggerEditModel:
    """Per-document trigger edit state: which triggers are dirty, and how the
    Triggers section serializes given that.

    Construct one per open scenario, after parse_triggers() has succeeded.
    Holding it across a document switch is safe (it stores the LoadedScenario,
    never a manager) but pointless -- each document gets its own.
    """

    def __init__(self, loaded: LoadedScenario):
        manager = parse_triggers(loaded)
        if manager is None:
            raise TriggerEditsUnavailableError(
                "This file's Triggers section cannot be parsed, so its triggers cannot be edited."
            )
        if not loaded.trigger_write_supported:
            raise TriggerEditsUnavailableError(
                "This file's Triggers section parsed but failed the alignment gate, so "
                "writing triggers back would splice at an offset that cannot be trusted."
            )

        self.loaded = loaded
        section = loaded._scenario.sections["Triggers"]
        self.regions = _map_regions(section)
        self._original_section = loaded.decompressed_body[
            loaded.units_section_end : loaded.triggers_section_end
        ]
        if len(self._original_section) != self.regions.section_end:
            raise TriggerEditsUnavailableError(
                f"Triggers section walks to {self.regions.section_end} bytes but spans "
                f"{len(self._original_section)} in the file -- refusing to splice"
            )

        self._blobs: list[bytes | None] = self._slice_blobs(section)
        self._variables_dirty = False
        # Pending legacy_exec_order value, or None for "unchanged, splice the
        # file's own byte". Normalised back to None when set to the value the
        # file already holds, so flipping the flag and flipping it back leaves
        # the document on the verbatim-splice branch rather than merely
        # producing identical bytes through the rewrite branch.
        self._exec_order: int | None = None
        self._exec_order_supported = exec_order_write_supported(loaded)
        # True once the trigger list's membership or order has changed. Not
        # derivable from the blobs: serialize() always regenerates the section's
        # trigger counter and its display-order array, so a structural edit
        # changes the section even when every surviving blob is still valid.
        # See has_edits.
        self._structure_dirty = False
        # Strong references to the Trigger objects each blob belongs to. Used
        # by serialize() to prove the blob list still lines up with the live
        # trigger list, and strong so no id() can be recycled underneath it.
        self._tracked: list = list(manager.triggers)
        # Set only between begin_trigger_edit() and commit/abort -- the
        # pre-mutation snapshot plus the ids of the objects the caller declared
        # it may edit by content.
        self._pending: tuple[TriggerSnapshot, set[int]] | None = None

    # -- construction helpers ------------------------------------------------

    def _slice_blobs(self, section) -> list[bytes | None]:
        """One verbatim byte slice per trigger, cut at the parse-time
        byte_lengths. Their sum is exactly the trigger_data region, which
        _map_regions() derived the same way -- so this is a consistency check
        of the whole layout, not just a slicing convenience."""
        blobs: list[bytes | None] = []
        offset = self.regions.triggers_start
        for entry in section.retriever_map["trigger_data"].data or []:
            blobs.append(self._original_section[offset : offset + entry.byte_length])
            offset += entry.byte_length
        if offset != self.regions.triggers_end:
            raise TriggerEditsUnavailableError(
                f"Per-trigger byte lengths sum to {offset}, but trigger_data ends at "
                f"{self.regions.triggers_end}"
            )
        return blobs

    # -- state ---------------------------------------------------------------

    def manager(self) -> TriggerManager:
        """The live TriggerManager. Re-fetched every call on purpose -- see
        this module's docstring, invariant 1. Never cache what this returns."""
        manager = parse_triggers(self.loaded)
        if manager is None:  # pragma: no cover -- __init__ already refused this file
            raise TriggerEditsUnavailableError("Triggers section stopped parsing")
        return manager

    @property
    def trigger_count(self) -> int:
        return len(self._blobs)

    @property
    def has_edits(self) -> bool:
        """True if saving would take the re-serializing branch. False means a
        save is byte-identical to one from a document that never opened the
        trigger panel.

        The `_structure_dirty` term is not redundant with the blob scan, and
        leaving it out loses whole edits silently. Deleting a trigger nothing
        else references dirties no blob at all: every survivor's bytes are
        genuinely unchanged and position-independent. But the section still
        differs, because serialize() regenerates the trigger counter and the
        display-order array from the live list -- so without this term
        scenario_write.py takes the verbatim-splice branch and the deleted
        trigger comes back. Found by 4b.6b's delete operation; the same applies
        to a reorder, where the count does not change either.

        `_exec_order` is the third term for exactly that reason. It dirties no
        blob and changes no structure -- it is one byte of the verbatim-spliced
        tail -- so without it a save whose only edit is the trigger
        execution-order flag takes the splice branch and writes the original
        byte back, with no error anywhere. See tests/test_trigger_write_path.py's
        single-byte exec-order test.
        """
        return (
            self._structure_dirty
            or self._variables_dirty
            or self._exec_order is not None
            or any(blob is None for blob in self._blobs)
        )

    @property
    def variables_dirty(self) -> bool:
        return self._variables_dirty

    # -- trigger execution order ---------------------------------------------
    # One byte of the section's verbatim-spliced tail, owned here rather than
    # by descape/options_model.py even though the Map Options panel is where it
    # is shown. Two reasons, and getting either wrong drops the edit silently:
    # it lives inside the Triggers region, which the write path splices whole
    # rather than byte-patches; and item 2's invariant -- a save collapses back
    # to one coherent execution-order mode -- couples it to
    # trigger_display_order, which lives here too.

    @property
    def exec_order_supported(self) -> bool:
        """Whether set_exec_order() will be accepted for this file."""
        return self._exec_order_supported

    @property
    def exec_order(self) -> int:
        """The flag's current effective value: the pending one if set,
        otherwise the byte the file holds."""
        if self._exec_order is not None:
            return self._exec_order
        return self._original_exec_order()

    def _original_exec_order(self) -> int:
        offset = self.regions.exec_order_offset
        if offset is None:
            raise TriggerEditsUnavailableError(
                "This file stores no trigger execution-order flag."
            )
        return self._original_section[offset]

    def set_exec_order(self, value: int) -> None:
        """Set the pending execution-order flag. Must be wrapped in a
        begin_trigger_edit()/commit_trigger_edit() pair like every other edit
        here, so it gets an undo record -- a model that is dirty while the
        history is not closes the document with no save prompt.

        Setting the value the file already holds clears the pending edit
        instead of recording an identical one, which is what keeps has_edits
        False for a flag flipped and flipped back.
        """
        if not self._exec_order_supported:
            raise TriggerEditsUnavailableError(
                "This file's trigger execution-order flag cannot be written back."
            )
        if value not in (0, 1):
            raise ValueError(f"legacy_exec_order is a u8 flag, not {value!r}")
        self._exec_order = None if value == self._original_exec_order() else value

    def is_dirty(self, index: int) -> bool:
        return self._blobs[index] is None

    def dirty_indices(self) -> list[int]:
        return [i for i, blob in enumerate(self._blobs) if blob is None]

    def mark_dirty(self, index: int) -> None:
        """Call after mutating trigger `index`'s own fields, conditions, or
        effects. Anything that changes a trigger and does not go through this
        (or through the structural methods below) silently splices the
        pre-edit bytes back on save."""
        if not 0 <= index < len(self._blobs):
            raise IndexError(f"No trigger at index {index} (have {len(self._blobs)})")
        self._blobs[index] = None

    def mark_variables_dirty(self) -> None:
        """Call after adding, removing, or renaming a variable. The variable
        block sits after unknown_bytes and splices verbatim until this is
        called, so an unedited file's variable names keep their exact bytes."""
        self._variables_dirty = True

    # -- undo/redo support ---------------------------------------------------

    def snapshot(self, content_touched: Sequence[int] = ()) -> TriggerSnapshot:
        """Capture restorable state. `content_touched` names the indices whose
        *content* may differ across this edit; only those get a deepcopy.

        Must be called *before* the mutation for the "before" side. There is no
        way to recover it afterwards, which is why begin/commit exists at all:
        `structural_edit`'s own dirty set is computed after `mutate_fn` runs by
        diffing reference signatures, and `mark_dirty()`'s contract is to be
        called after mutating. A record built at either point would capture
        post-edit state as its "before" and make undo a silent no-op.
        """
        manager = self.manager()
        triggers = list(manager.triggers)
        touched = set(content_touched)
        for index in touched:
            if not 0 <= index < len(triggers):
                raise IndexError(f"No trigger at index {index} (have {len(triggers)})")
        return TriggerSnapshot(
            triggers=triggers,
            ref_ids=[_reference_signature(t) for t in triggers],
            states=[copy.deepcopy(t) if i in touched else None for i, t in enumerate(triggers)],
            trigger_ids=[t.trigger_id for t in triggers],
            display_order=list(manager.trigger_display_order),
            blobs=list(self._blobs),
            variables=copy.deepcopy(manager.variables),
            variables_dirty=self._variables_dirty,
            structure_dirty=self._structure_dirty,
            exec_order=self._exec_order,
        )

    def restore(self, snapshot: TriggerSnapshot) -> None:
        """Put the document back to `snapshot`. The single place any undo/redo
        writes to the model, so edit_history.py stays Qt- and library-free.

        Every step below is load-bearing and was verified by direct experiment;
        each silently produces a wrong result if skipped or reordered.
        """
        manager = self.manager()
        if len(snapshot.triggers) != len(snapshot.blobs):
            raise RuntimeError(
                f"Snapshot holds {len(snapshot.triggers)} triggers but {len(snapshot.blobs)} "
                f"blobs -- refusing to restore a record that cannot align"
            )

        # Trap 5: never hand the record's own object to the live list, or the
        # next in-place edit mutates the snapshot and undoing twice replays
        # already-edited state. Replace the entry outright rather than copying
        # fields across, which would walk into the Effect.quantity bit-split.
        restored = [
            copy.deepcopy(state) if state is not None else trigger
            for trigger, state in zip(snapshot.triggers, snapshot.states)
        ]

        # Trap 1: `manager.triggers = [...]` resets trigger_display_order to
        # identity unconditionally in its setter. Slice-assign preserves the
        # machinery. Trap 6: the library mutates lists handed to it in place,
        # so every one of these is a fresh copy of the record's, never the
        # record's own -- a snapshot list assigned to trigger_display_order was
        # observed being rewritten from [3, 2, 1, 0] to [3, 2, 1, 0, 4].
        manager.triggers[:] = restored
        # Trap 7: trigger_display_order's *getter* has side effects
        # (list_changed then update_order_array), so the order must go back
        # through the setter before anything reads the getter.
        manager.trigger_display_order = list(snapshot.display_order)

        # Trap 2: restoring list membership does not restore trigger_id -- after
        # a remove_trigger(0) and a slice-restore, ids came back [0, 0, 1, 2].
        for trigger, trigger_id in zip(manager.triggers, snapshot.trigger_ids):
            trigger.trigger_id = trigger_id

        # Trap 3: remove_triggers()/reorder_triggers() rewrite ce.trigger_id on
        # *surviving* triggers in place, so restoring membership does not undo
        # them. This is also the restore checksum: a length mismatch means the
        # live graph is not the one this snapshot was taken from.
        for trigger, ref_ids in zip(manager.triggers, snapshot.ref_ids):
            referencing = get_trigger_referencing_ce(trigger)
            if len(referencing) != len(ref_ids):
                raise RuntimeError(
                    f"Trigger {trigger.trigger_id} has {len(referencing)} trigger references "
                    f"but the snapshot recorded {len(ref_ids)} -- refusing to restore"
                )
            for ce, trigger_id in zip(referencing, ref_ids):
                ce.trigger_id = trigger_id

        if snapshot.variables is not None:
            # Trap 5 again: hand the live list fresh copies, never the
            # snapshot's own objects, or the next rename mutates the record
            # in place and a second undo replays already-edited state.
            manager.variables = copy.deepcopy(snapshot.variables)

        # Trap 4: _tracked and _blobs must be restored in the same step, or the
        # next serialize() trips _check_alignment(). Trap 7 again: _tracked
        # comes from the *live* list, not the record -- its content-touched
        # entries were just replaced by fresh deepcopies, so setting it from
        # the record makes _check_alignment raise on the next save.
        self._blobs = list(snapshot.blobs)
        self._tracked = list(manager.triggers)
        self._variables_dirty = snapshot.variables_dirty
        self._structure_dirty = snapshot.structure_dirty
        self._exec_order = snapshot.exec_order

    def begin_trigger_edit(self, content_touched: Sequence[int] = ()) -> None:
        """Snapshot before mutating. Pairs with exactly one commit_trigger_edit()
        or abort_trigger_edit(), mirroring begin_stroke/commit_stroke.

        `content_touched` is the caller's declaration of which triggers this
        edit may change by content -- field, condition, or effect edits. Those
        get the deepcopy and an unconditional mark_dirty() at commit. The
        id-remap class needs no declaration; it is detected automatically.
        """
        if self._pending is not None:
            raise RuntimeError("begin_trigger_edit() called while an edit was already in progress")
        touched = list(content_touched)
        before = self.snapshot(touched)
        self._pending = (before, {id(before.triggers[i]) for i in touched})

    def abort_trigger_edit(self) -> None:
        """Discards the in-progress snapshot without recording. Does not roll
        the document back -- call restore() for that.

        That includes the dirty flags: an aborted edit that already ran a
        structural_edit() leaves `_structure_dirty` and the blob list as the
        mutation left them, with no record to undo them by. Deliberate, since
        the document really is in that state, but it means an abort path has to
        pair with restore() to be a true no-op. No caller aborts today --
        viewer.py's _trigger_edit() always commits, on purpose.
        """
        self._pending = None

    def commit_trigger_edit(self, label: str, history: EditHistory) -> TriggerDiffRecord:
        """Close the pair opened by begin_trigger_edit(), mark the declared
        triggers dirty, and push one record onto `history`.

        Takes the history rather than returning the record for the caller to
        push, so the contract that keeps a single EditHistory honest -- any
        mutation that sets model dirtiness must push a record -- cannot be
        forgotten at a call site. A model that is dirty while the history is
        not closes the document without a save prompt.

        Call this even if the wrapped structural_edit() raised: a failed
        structural edit is not a no-op (it marks every blob dirty), so the
        state change still needs a record to undo.

        Pushes unconditionally rather than diffing for a no-op. Detecting "no
        real change" would need a content comparison of the touched triggers,
        and getting it wrong in the false-negative direction makes a genuine
        second edit to an already-dirty trigger silently un-undoable. The UI's
        job is not to open an edit it did not make.
        """
        if self._pending is None:
            raise RuntimeError("commit_trigger_edit() called with no edit in progress")
        before, touched_ids = self._pending
        self._pending = None

        manager = self.manager()
        # Post-edit positions of the objects the caller declared, found by
        # identity: a structural edit in the same pair may have moved them, and
        # a removed one simply drops out (the before side restores it).
        post_touched = [i for i, t in enumerate(manager.triggers) if id(t) in touched_ids]
        for index in post_touched:
            self.mark_dirty(index)
        after = self.snapshot(post_touched)

        record = TriggerDiffRecord(label, before, after, touched=post_touched)
        history.push_trigger_record(record)
        return record

    # -- structural edits ----------------------------------------------------

    def structural_edit(self, mutate_fn, content_touched: Sequence[int] = ()) -> list[int]:
        """Runs `mutate_fn(manager)` and reconciles the blob list with whatever
        it did to the trigger list.

        This is the only correct way to call TriggerManager's own structural
        API (add_trigger, remove_triggers, reorder_triggers, move_triggers,
        copy_trigger, import_triggers, add_variable). Those methods renumber
        trigger ids across the whole list and remap (de)activate-trigger
        references to match, so triggers the caller never named come back
        changed.

        **Content-mutating operations must declare themselves via
        `content_touched`.** Automatic dirty detection here is a
        reference-signature diff, so it sees only the id-remap class of change.
        An operation that rewrites a pre-existing trigger's *content* some
        other way is invisible to it, and without a declaration its edit is
        silently spliced away on save. Two library methods do exactly that:

        - `replace_player()` rewrites `source_player` in place on the named
          trigger (verified: the effect changed player, `structural_edit`
          returned [], and serialize() reproduced the original bytes).
        - `copy_trigger_per_player()` / `copy_trigger_tree_per_player()` rename
          the *source* trigger after copying it; the copies are detected, the
          rename is not.

        For both, pass the pre-edit index of every trigger the call may rewrite
        in place. Indices are given in pre-edit numbering and followed through
        the edit by object identity, so a declared trigger that also moves is
        still marked at its new position.

        Reconciliation is by object identity, not by index: the library moves
        the same Trigger objects around rather than rebuilding them, so a blob
        follows its trigger through a reorder. A trigger whose reference
        signature changed is marked dirty; one that only moved is not, because
        TriggerStruct carries no trigger_id field of its own (verified against
        versions/DE/*/structure.json) and its bytes are position-independent.

        Returns the indices marked dirty, in the post-edit numbering.
        """
        manager = self.manager()
        # `before` keeps every pre-edit Trigger alive for the duration. Without
        # it, an operation that deletes a trigger and creates another could see
        # CPython reuse the freed object's id() and match the wrong blob.
        before = list(manager.triggers)
        before_blobs = {id(t): blob for t, blob in zip(before, self._blobs)}
        before_signatures = {id(t): _reference_signature(t) for t in before}
        before_variables = _variable_signature(manager)
        before_display_order = list(manager.trigger_display_order)
        for index in content_touched:
            if not 0 <= index < len(before):
                raise IndexError(f"No trigger at index {index} (have {len(before)})")
        content_ids = {id(before[index]) for index in content_touched}

        try:
            mutate_fn(manager)
        except Exception:
            # A failed structural edit is not a no-op. reorder_triggers()
            # assigns the new display order *before* validating it and
            # reassigns trigger_id as it walks, so a bad id leaves the list
            # half-remapped and the display order holding the invalid input.
            #
            # There is no way to tell which blobs still describe their trigger,
            # so none of them are trusted: everything re-serializes from the
            # object graph, which is always correct even though it gives up
            # this document's minimal-diff guarantee. The display order is
            # rolled back to its pre-edit value so the document stays saveable
            # rather than tripping _display_order_bytes()'s permutation check
            # on every later save.
            self._blobs = [None] * len(manager.triggers)
            self._tracked = list(manager.triggers)
            self._variables_dirty = True
            self._structure_dirty = True
            if len(before_display_order) == len(manager.triggers):
                manager.trigger_display_order = before_display_order
            else:
                manager.trigger_display_order = list(range(len(manager.triggers)))
            raise

        # Membership or order, by identity. Any difference means the section's
        # regenerated counter or display-order array will differ, whether or not
        # a single blob went stale. See has_edits.
        #
        # trigger_display_order is checked separately from membership/order of
        # `manager.triggers` itself: a pure display-order permutation (the
        # trigger reordering feature) changes neither list's identities, so it
        # would otherwise leave _structure_dirty False -- dirtying no blob,
        # changing no structure by this method's own definition of structure
        # -- and has_edits would be False for an edit that demonstrably changed
        # the regenerated display-order array. Same defect class 4b.6b's
        # unreferenced-delete finding was: the section differs, but nothing
        # here noticed.
        if [id(t) for t in manager.triggers] != [id(t) for t in before] or (
            list(manager.trigger_display_order) != before_display_order
        ):
            self._structure_dirty = True

        blobs: list[bytes | None] = []
        newly_dirty: list[int] = []
        for index, trigger in enumerate(manager.triggers):
            key = id(trigger)
            blob = before_blobs.get(key)
            if (
                blob is None
                or key in content_ids
                or before_signatures.get(key) != _reference_signature(trigger)
            ):
                blobs.append(None)
                newly_dirty.append(index)
            else:
                blobs.append(blob)
        self._blobs = blobs
        self._tracked = list(manager.triggers)
        # Variables live outside the trigger list, so nothing above sees a
        # change to them.
        if _variable_signature(manager) != before_variables:
            self._variables_dirty = True
        return newly_dirty

    # -- serialization -------------------------------------------------------

    def serialize(self) -> bytes:
        """The whole Triggers section, ready to splice into the decompressed
        body at units_section_end.

        Clean triggers contribute their original bytes; dirty ones are taken
        from the library's per-struct serialization after a commit. The
        section header's trigger counter and the display-order array are always
        regenerated (both are pure counters over the current list), and
        everything after the variable block splices verbatim.
        """
        manager = self.manager()
        self._check_alignment(manager)
        # A whole-manager commit is what rebuilds trigger_data when triggers
        # were added or removed. Its output for clean triggers is ignored --
        # only the dirty ones are read back out of it.
        manager.commit()

        section = self.loaded._scenario.sections["Triggers"]
        entries = section.retriever_map["trigger_data"].data or []
        if len(entries) != len(self._blobs):
            raise RuntimeError(
                f"Committed {len(entries)} trigger structs but the model tracks "
                f"{len(self._blobs)} blobs -- a structural edit bypassed structural_edit()"
            )

        parts: list[bytes] = [
            self._original_section[: SECTION_HEADER_SIZE - _TRIGGER_COUNT_STRUCT.size],
            _TRIGGER_COUNT_STRUCT.pack(len(self._blobs)),
        ]
        for blob, entry in zip(self._blobs, entries):
            parts.append(entry.get_data_as_bytes() if blob is None else blob)
        parts.append(self._display_order_bytes(manager))
        parts.append(
            self._original_section[self.regions.display_order_end : self.regions.unknown_bytes_end]
        )
        parts.append(self._variable_block_bytes(section))
        parts.append(self._tail_bytes())
        return b"".join(parts)

    def _tail_bytes(self) -> bytes:
        """Everything after the variable block: useless_trigger_data,
        unknown_bytes2, redacted, and legacy_exec_order on the versions that
        store it.

        Spliced verbatim unless set_exec_order() left a pending value, in which
        case exactly that one byte is overwritten. Never the tail's last byte:
        on a file whose trigger version is below 4.5 the flag consumes zero
        bytes and the last byte belongs to `redacted`, so the offset comes from
        the forward walk in _map_regions() -- see SectionRegions.
        """
        tail = self._original_section[self.regions.variables_end :]
        if self._exec_order is None:
            return tail
        offset = self.regions.exec_order_offset
        if offset is None:  # pragma: no cover -- set_exec_order() refuses first
            raise TriggerEditsUnavailableError(
                "A trigger execution-order edit is pending on a file that stores no such flag."
            )
        local = offset - self.regions.variables_end
        patched = bytearray(tail)
        patched[local] = self._exec_order
        return bytes(patched)

    def _check_alignment(self, manager: TriggerManager) -> None:
        """Refuses to serialize if the blob list no longer describes the live
        trigger list.

        Comparing object identity rather than length: a bypassed add or remove
        changes the count and would be caught either way, but a bypassed
        reorder or move leaves the count identical while every blob now points
        at the wrong trigger. That case splices real bytes into the wrong slot
        and produces a file that loads fine and is wrong, which is the worst
        outcome available here.
        """
        live = list(manager.triggers)
        aligned = len(live) == len(self._tracked) and all(
            a is b for a, b in zip(live, self._tracked)
        )
        if not aligned:
            raise RuntimeError(
                f"The trigger list changed behind the model's back "
                f"({len(self._tracked)} tracked, {len(live)} live) -- a structural "
                f"edit bypassed structural_edit(), so every blob's index mapping "
                f"is untrustworthy"
            )

    def _display_order_bytes(self, manager: TriggerManager) -> bytes:
        """trigger_display_order_array, regenerated rather than spliced: it is
        one u32 per trigger and its length changes with the trigger count, so
        there is nothing to preserve."""
        order = list(manager.trigger_display_order)
        if sorted(order) != list(range(len(self._blobs))):
            raise RuntimeError(
                f"trigger_display_order is not a permutation of 0..{len(self._blobs) - 1}: {order}"
            )
        return b"".join(_DISPLAY_ORDER_STRUCT.pack(index) for index in order)

    def _variable_block_bytes(self, section) -> bytes:
        """number_of_variables + variable_data, spliced verbatim unless
        mark_variables_dirty() was called. Variable names are str16s and drift
        the same way trigger names do, so an untouched variable block must not
        go through the library."""
        if not self._variables_dirty:
            return self._original_section[self.regions.unknown_bytes_end : self.regions.variables_end]
        return section.retriever_map["number_of_variables"].get_data_as_bytes() + b"".join(
            entry.get_data_as_bytes() for entry in (section.retriever_map["variable_data"].data or [])
        )
