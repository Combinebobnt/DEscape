"""v2's write path: patch the terrain struct array directly in the original
decompressed body and recompress, rather than going through
AoE2ScenarioParser's own commit()/write_to_file().

Why not the library's own write path -- confirmed this way, not assumed:
re-serializing Map/Units through AoE2ScenarioParser's managers and writing via
write_to_file() was tested directly against every example file in this repo.
It fails on 10 of 12: two versions crash entirely (a class-level property-
disabling bug in the library that persists across scenario instances loaded in
the same process -- this tool routinely loads several versions back-to-back,
so that's not an edge case here), and the other eight "succeed" but produce a
body that isn't byte-identical to the original even with zero edits (e.g. a
str16 field silently drops a trailing NUL). None of that is a bug worth
chasing upstream -- it's the general shape of "reserialize through an object
model that wasn't written to be byte-for-byte."

What works, confirmed on the full example corpus, is two separate guarantees.
The *decompressed content* is byte-identical for a zero-edit round trip on
every file: never touch the parsed object graph's serialization at all. Take
the *original* decompressed body (kept verbatim since load, see
scenario_io.LoadedScenario), overwrite only the terrain struct array's bytes
from the current (possibly edited) in-memory tile values, recompress with the
same zlib call AoE2ScenarioParser itself uses, and concatenate after the
verbatim original header. Everything else in the file -- every other Map
retriever, Units, the never-parsed trigger tail -- passes through as pure
untouched bytes. The *raw compressed file* is byte-identical for a zero-edit
round trip too, but not via recompression: re-deflating identical
decompressed bytes does not reliably reproduce the original compressed
stream (a different encoder/level/version upstream produces a
semantically-identical but bit-different DEFLATE stream for the same input).
Instead, when nothing patched anything, the write path reuses
scenario.original_compressed_body verbatim instead of recompressing.

Phase 4b extends that same shape one level down rather than replacing it.
Triggers still splice verbatim by default; a document with actual trigger edits
swaps in descape/trigger_model.py's serialization of the Triggers section,
which itself splices back the original bytes of every trigger the user did not
touch. So the "never re-serialize what wasn't edited" rule above holds at
per-trigger granularity too, and a save with no trigger edits is byte-for-byte
what this module produced before phase 4b existed.

The Map Options panel's scalars are the same rule again, one level smaller:
every one is fixed-width, so each is an in-place byte patch that shifts no
offset, and a document whose options were only browsed patches nothing at all.

Messages mode is the first exception to "every mapped field is fixed-width,
so no patch shifts an offset": a message field is a length-prefixed string,
so _patch_messages() is a splice, not an in-place overwrite. It is applied
*last*, to the fully-assembled body -- Messages sits upstream (section 3 of
13) of every offset the terrain/options/units/triggers patches above use, so
running it last means all of those still land at their original,
pre-splice offsets; only the Messages splice itself has to move bodily.

Phase 3.5a adds a third level, the same shape as Triggers: Units still splice
verbatim by default; a document with actual unit edits swaps in
descape/unit_model.py's serialization, which splices back the original bytes
of every unit the user did not touch. The one place library output actually
reaches the file is that module's normalizer (stripping the trailing NUL the
library always appends to a unit's caption_string, which the game does not) --
everything else here still deals in raw bytes only. Both the Units and
Triggers sections can change length in the same save, so _assemble_body() is
the one place that concatenates the (possibly resized) Units section, the
(possibly resized) Triggers section, and everything around them in a single
step, cut at original load-time offsets rather than offsets derived from each
other.

The civ/architecture maintainer plan's Step B adds a second resizing splice,
_patch_player_data_1() (civilization/architecture_set on a scenario version
1.56+ file, a length-prefixed str16 like a message field): DataHeader sits
*before* Messages in the body, not after, so it needs the opposite ordering
rule -- run everything above it while it is still at its original length,
then apply Messages, then apply this one last of all. The general rule this
generalizes to, for any future resizing step: fixed-offset patches run
first, then resizing splices in *descending* offset order (Messages, the
higher offset, before player_data_1, offset 0) -- each splice needs every
offset used by an earlier step to still be valid, which only holds if
nothing upstream of that step has shifted yet.

See tests/test_write_path.py, tests/test_trigger_write_path.py,
tests/test_units_write_path.py, and tests/test_player_write_path.py for the
evidence.
"""

from __future__ import annotations

import os
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from descape.scenario_io import (
    TERRAIN_STRUCT_SIZE,
    FORBIDDEN_WRITE_MARKER,
    TEMPLATE_DIR,
    LoadedScenario,
    _LAYER_STRUCT,
)
from descape.messages_model import MessagesEditModel
from descape.options_model import (
    OptionsEditModel,
    diplomacy_write_supported,
    options_write_supported,
    player_count_write_supported,
    players_write_supported,
)
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

# Options.number_of_triggers and FileHeader.trigger_count. Both are their own
# section's *last* retriever in all 19 DE structure versions, so each is
# addressed as "the 4 bytes ending where that section ends".
_TRIGGER_COUNT_STRUCT = struct.Struct("<I")

# DataHeader.next_unit_id_to_place, the leading u32 of decompressed_body
# (plan finding 8) -- the same fact descape/unit_model.py's UnitEditModel
# reads at construction, duplicated here as a plain offset-0 patch the same
# way _TRIGGER_COUNT_STRUCT is duplicated per module rather than shared.
_NEXT_UNIT_ID_STRUCT = struct.Struct("<I")


class WriteBlockedError(Exception):
    """Raised instead of writing, e.g. for the compatdata guard or an
    unsupported (terrain_write_supported=False) file."""


@dataclass(frozen=True)
class WriteResult:
    """wrote is False for the no-op short circuit (out_path already holds
    these exact bytes) -- backups is then always []. Otherwise backups lists
    whichever of .bak/.orig were actually written, per descape.backup."""

    wrote: bool
    backups: list[Path]


def _compress_bytes(data: bytes) -> bytes:
    # Same call AoE2ScenarioParser.scenarios.aoe2_scenario._compress_bytes()
    # uses -- kept independent (not imported) so this module has zero
    # dependency on the library's write path, which is the entire point.
    deflate_obj = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return deflate_obj.compress(data) + deflate_obj.flush()


def _patch_terrain_block(scenario: LoadedScenario) -> bytes:
    """Returns a new decompressed body: scenario.decompressed_body with the
    terrain struct array overwritten from the *current* MapManager.terrain
    tile values. Only terrain_id, elevation, and layer are touched -- the
    per-tile 'unused' 3 bytes are carried through untouched, exactly as
    loaded.

    Elevation is written as-is, with no legality check of its own here:
    elevation_tools.set_tile_elevation() is the one path in this codebase
    responsible for keeping every pair of adjacent tiles within AoE2:DE's
    tolerance, and this function trusts that whatever set the in-memory
    value already went through it. A future write path that assigns
    elevation any other way must re-establish that itself first -- an
    illegal jump between neighbours renders as a degenerate stretched seam
    in the editor and crashes the game outright on load, not a cosmetic
    glitch."""
    body = bytearray(scenario.decompressed_body)
    offset = scenario.terrain_block_offset
    for i, tile in enumerate(scenario.map_manager.terrain):
        o = offset + TERRAIN_STRUCT_SIZE * i
        body[o] = tile.terrain_id
        body[o + 1] = tile.elevation
        _LAYER_STRUCT.pack_into(body, o + 5, tile.layer)
    return bytes(body)


def _patch_options(body: bytes, scenario: LoadedScenario, options: OptionsEditModel) -> bytes:
    """Returns `body` with every pending map-option scalar patched in place.

    Every mapped field is fixed-width, so no patch shifts an offset and the
    order among them does not matter. What does matter is that they all run
    *before* _assemble_body(), which changes the Units and/or Triggers
    sections' length: each of these sits before units_block_offset, so
    patching after would still land correctly today but only by accident of
    the length-changing step being downstream. Asserted rather than assumed,
    per patch.

    The boundary is units_block_offset, not units_section_end: once a units
    edit exists, a patch landing inside [units_block_offset, units_section_end)
    would be silently discarded by _assemble_body()'s units branch rather than
    refused. Falls back to units_section_end when units_block_offset is -1
    (units_write_supported is False for this file, so nothing splices there
    and the wider bound is harmless) -- every mapped option field sits well
    before either boundary today, so this cannot fire yet; the point is that a
    future field addition inside that range gets refused, not silently
    dropped.

    The four bytes ending at options_section_end are Options.number_of_triggers,
    which the caller patches from the live trigger count before calling this.
    No mapped field reaches them -- the backward walk subtracts that retriever
    first -- and this refuses rather than letting the two writers race on the
    same bytes.
    """
    patched = bytearray(body)
    counter_start = scenario.options_section_end - _TRIGGER_COUNT_STRUCT.size
    limit = scenario.units_block_offset if scenario.units_block_offset >= 0 else scenario.units_section_end
    for offset, data in options.serialize_patches():
        end = offset + len(data)
        if offset < 0 or end > limit:
            raise WriteBlockedError(
                f"A Map Options/Diplomacy/Players patch at {offset}..{end} falls outside "
                f"the pre-Units region this write path owns (ends at {limit})."
            )
        if offset < scenario.options_section_end and end > counter_start:
            raise WriteBlockedError(
                f"A Map Options/Diplomacy/Players patch at {offset}..{end} overlaps "
                f"Options.number_of_triggers ({counter_start}..{scenario.options_section_end}), "
                f"which the trigger splice writes."
            )
        assert len(data) == end - offset, (
            "a serialize_patches() entry changed length -- str16 fields must ride "
            "serialize_resizes() instead, never this fixed-offset/fixed-length path"
        )
        patched[offset:end] = data
    return bytes(patched)


def _patch_trigger_count(body: bytes, scenario: LoadedScenario, count: int) -> bytes:
    """Options.number_of_triggers, patched from the live trigger count.
    Lifted out of the old _splice_triggers() (phase 3.5a absorbed the actual
    splice into _assemble_body(), which has to run after both the Units and
    Triggers sections' lengths are known)."""
    patched = bytearray(body)
    _TRIGGER_COUNT_STRUCT.pack_into(patched, scenario.options_section_end - _TRIGGER_COUNT_STRUCT.size, count)
    return bytes(patched)


def _patch_next_unit_id(body: bytes, units: UnitEditModel) -> bytes:
    """DataHeader.next_unit_id_to_place, patched only when at least one unit
    was added this session -- never lowered, and a document with no adds
    must not touch this byte at all (plan finding 8, stage 3.2)."""
    patched = bytearray(body)
    _NEXT_UNIT_ID_STRUCT.pack_into(patched, 0, units.next_unit_id)
    return bytes(patched)


def _assemble_body(
    scenario: LoadedScenario,
    base: bytes,
    units: UnitEditModel | None,
    triggers: TriggerEditModel | None,
) -> bytes:
    """Returns `base` with the Units and/or Triggers sections replaced by
    their edit models' serializations, cutting both halves at
    *original load-time* offsets and concatenating once -- so no offset here
    is ever derived from another.

    This is the one and only length-changing step; every other patch in
    write_scenario() runs before it, in place, at offsets that stay valid
    because none of them shift anything. A units-only edit must never read
    triggers_section_end: it is -1 until parse_triggers() has run, so the
    clean (no trigger edits) branch takes one slice spanning Triggers and
    everything after, and never mentions it.
    """
    assert len(base) == len(scenario.decompressed_body), (
        "an in-place patch changed the body's length -- every offset below is stale"
    )

    if units is not None and units.has_edits:
        assert 0 <= scenario.units_block_offset < scenario.units_section_end <= len(base)
        head = base[: scenario.units_block_offset] + units.serialize()
    else:
        head = base[: scenario.units_section_end]

    if triggers is not None and triggers.has_edits:
        assert scenario.units_section_end < scenario.triggers_section_end <= len(base)
        tail = triggers.serialize() + base[scenario.triggers_section_end :]
    else:
        tail = base[scenario.units_section_end :]

    return head + tail


def _patch_messages(body: bytes, scenario: LoadedScenario, messages: MessagesEditModel) -> bytes:
    """Returns `body` with the whole Messages section replaced by
    messages.serialize(). A length-changing step -- see the module
    docstring -- so it must run *after* every fixed-width patch and the
    Units/Triggers splice above: those all address offsets that stay valid
    exactly because nothing before them has shifted yet at the point they
    run. No longer strictly last: _patch_player_data_1() (below) is a
    second, later-inserted resizing step, ordered after this one for the
    reason its own docstring gives."""
    start, end = messages.section_span()
    return body[:start] + messages.serialize() + body[end:]


def _patch_player_data_1(body: bytes, scenario: LoadedScenario, options: OptionsEditModel) -> bytes:
    """Returns `body` with DataHeader.player_data_1 spliced to reflect
    every pending str16-coded Players mode edit (civilization/architecture
    on a scenario version 1.56+ file) -- the civ/architecture maintainer
    plan's Step B. A no-op (returns `body` unchanged) unless
    options.serialize_resizes() has something to apply, so a save with no
    such edit is byte-identical to one from before this step existed.

    Must run *last*, after _patch_messages() -- the load-bearing ordering
    decision. DataHeader sits at body offset 0, so a length change here
    shifts every anchor downstream of it (messages_section_start/_end,
    player_data_two_section_end, options_section_end, terrain_block_offset,
    units_block_offset, units_section_end) and, within DataHeader itself,
    per_player_lock_civilization/_personality, which sit after
    player_data_1. Running this splice after everything else means every
    other patch above ran against a buffer still at its original length,
    so every offset it used was valid at the moment it ran; the general
    rule (see the module docstring) is fixed-offset patches first, then
    resizing splices in *descending* offset order -- Messages (higher
    offset) before player_data_1 (offset 0).

    Reads the *partially patched* body, not scenario.decompressed_body:
    Number of Players patches eight `active` flags inside this very array
    through _patch_options(), and rebuilding it from the original bytes
    would silently discard a count edit made in the same save. Every offset
    involved is a load-time one and still valid here, since both
    length-changing steps that ran before this (the Units/Triggers assembly
    and the Messages splice) sit after player_data_1 in the body.

    Refuses (WriteBlockedError) a resize region reaching past
    messages_section_start: this write path's own established boundary for
    "definitely still inside DataHeader, not into territory a different
    patch already owns" -- player_data_1_splice()'s own span check already
    makes this unreachable in practice, but a locator bug landing on the
    wrong side of that boundary must be refused, not silently written
    somewhere real content wasn't.
    """
    resizes = options.serialize_resizes(body)
    if not resizes:
        return body
    assert len(resizes) == 1, "OptionsEditModel.serialize_resizes() returns at most one region today"
    start, end, replacement = resizes[0]
    if not (0 <= start <= end <= scenario.messages_section_start):
        raise WriteBlockedError(
            f"A Players mode resize at {start}..{end} falls outside the DataHeader "
            f"region this write path owns (must end at or before "
            f"messages_section_start={scenario.messages_section_start})."
        )
    new_body = body[:start] + replacement + body[end:]
    assert len(new_body) == len(body) + (len(replacement) - (end - start)), (
        "player_data_1 splice length mismatch"
    )
    return new_body


def _patch_header_instructions(header_bytes: bytes, messages: MessagesEditModel) -> bytes:
    """Splices FileHeader.scenario_instructions to mirror the Instructions
    field, if messages.header_patch() reports a change to make -- see that
    method's docstring for when it does not (unchanged, or a header
    span/sync check that failed at load time). Applied before
    _patch_header_trigger_count() so that function's own "last 4 bytes of
    the header" patch lands correctly regardless of how this splice moved
    the header's length -- it addresses len(header) - 4, not a fixed
    offset, so the two are actually order-independent, but this keeps the
    same before/after shape the plan called for."""
    patch = messages.header_patch()
    if patch is None:
        return header_bytes
    start, end, payload = patch
    return header_bytes[:start] + payload + header_bytes[end:]


def _patch_header_trigger_count(header_bytes: bytes, count: int) -> bytes:
    """FileHeader.trigger_count, the header's last 4 bytes. The header lives
    outside the compressed body entirely, so it is patched separately."""
    header = bytearray(header_bytes)
    _TRIGGER_COUNT_STRUCT.pack_into(header, len(header) - 4, count)
    return bytes(header)


def _patch_header_player_count(header_bytes: bytes, options: OptionsEditModel) -> bytes:
    """FileHeader.player_count, Number of Players' second buffer (the other
    eight locations are `active` flags inside decompressed_body, patched by
    _patch_options()). A no-op unless the count is actually pending.

    Ordering, the load-bearing part: this runs *before*
    _patch_header_instructions(), because player_count sits after
    scenario_instructions in the header's on-disk order, so an instructions
    splice that changed that field's length would invalidate the load-time
    span this patch addresses. _patch_header_trigger_count() has no such
    problem -- it addresses len(header) - 4, not a fixed offset -- which is
    why it can stay where it is.
    """
    patch = options.header_patch()
    if patch is None:
        return header_bytes
    start, end, payload = patch
    if not (0 <= start <= end <= len(header_bytes)) or len(payload) != end - start:
        raise WriteBlockedError(
            f"A Number of Players header patch at {start}..{end} does not fit this "
            f"file's {len(header_bytes)}-byte header."
        )
    header = bytearray(header_bytes)
    header[start:end] = payload
    return bytes(header)


def write_scenario(
    scenario: LoadedScenario,
    out_path: str | Path,
    triggers: TriggerEditModel | None = None,
    options: OptionsEditModel | None = None,
    backup: bool = True,
    units: UnitEditModel | None = None,
    messages: MessagesEditModel | None = None,
) -> WriteResult:
    """Writes `scenario`'s current Map terrain state (edited or not) to
    out_path as a full .aoe2scenario file.

    `triggers` is the document's TriggerEditModel, or None for a document that
    never opened the trigger panel. **The trigger tail still splices verbatim
    unless that model reports actual edits**, so a save from a document whose
    triggers were only browsed is byte-identical to one from before phase 4b
    existed. That containment is what lets every pre-existing write-path test
    keep passing unchanged, and tests/test_trigger_write_path.py pins it.

    `options` is the document's OptionsEditModel, or None for a document whose
    Map Options panel was never edited. Same containment as `triggers`: a model
    reporting no edits patches nothing, so a browse-only save is byte-identical.
    The trigger execution-order row is deliberately *not* handled here -- it
    lives inside the Triggers region and rides on `triggers`' own serialization.

    `units` is the document's UnitEditModel, or None for a document that never
    opened a unit-editing tool -- same containment again: units are unmodified
    regardless of what's in memory unless that model reports actual edits.
    See descape/unit_model.py's module docstring for why the write path is a
    per-unit byte splice rather than a whole-section re-serialization.

    `messages` is the document's MessagesEditModel, or None for a document
    that never opened Messages mode -- same containment again: the Messages
    section (and FileHeader.scenario_instructions) are unmodified unless
    that model reports actual edits. Unlike every other model here, an edit
    can change the Messages section's byte length -- see this module's
    docstring for why that patch has to run last, against the fully
    assembled body.

    Reads from MapManager.terrain directly, so this is automatically correct
    at any edit_history cursor position: saving after an undo writes the
    undone state, with no separate bookkeeping needed.

    Returns a WriteResult. If out_path already holds exactly the bytes this
    call would write, that's a full no-op: WriteResult(wrote=False,
    backups=[]), no write, no `.bak` refresh, no `.orig`, no mtime churn --
    a habitual Ctrl+S with nothing changed must never replace a genuinely
    useful `.bak` with a copy of the current file. Otherwise, when `backup`
    is True (the default) and out_path already exists, descape.backup.
    make_backups() refreshes `.bak` and (on first overwrite) `.orig` before
    the real write; WriteResult.backups lists whichever were written.

    Raises:
        WriteBlockedError: if out_path's path contains FORBIDDEN_WRITE_MARKER
            (never write into a Workshop/Proton compatdata folder -- the
            user's only copy of these files), if out_path's parent is
            TEMPLATE_DIR (never overwrite a shipped File > New template --
            covers the whole directory, so future sibling templates are
            protected without another edit here), or if this scenario failed
            its load-time terrain-block verification (terrain_write_supported
            is False) and so has no reliable offset to patch, or if `triggers`
            has edits on a file whose Triggers section failed its alignment
            gate (trigger_write_supported is False), or if `units` has edits
            on a file whose Units section failed its alignment gate
            (units_write_supported is False), or if `options` has edits
            on a file whose map-option block does not re-verify, or has
            Diplomacy grid edits on a file whose grid does not re-verify, or
            has Players mode edits on a file whose per-player block does not
            re-verify (diplomacy_write_supported/players_write_supported are
            False; each a separate gate from the map-option one, per
            options_model.diplomacy_write_supported's docstring), or if
            `messages` has edits on a file whose Messages section no longer
            passes messages_write_supported. All of the above are raised
            before any backup is written or out_path is touched.
        BackupFailedError (a WriteBlockedError subclass, descape.backup):
            if a `.bak`/`.orig` copy fails partway through. Aborts the save
            entirely rather than proceeding without backups -- on a full
            disk, a failed backup followed by a failed write would destroy
            both copies of a file the user has no other copy of. Every
            pre-existing backup is left intact.
    """
    out_path = Path(out_path)
    if FORBIDDEN_WRITE_MARKER in str(out_path):
        raise WriteBlockedError(
            f"Refusing to write to a path containing {FORBIDDEN_WRITE_MARKER!r}: {out_path}"
        )
    if out_path.resolve().parent == TEMPLATE_DIR.resolve():
        raise WriteBlockedError(
            f"Refusing to write into the shipped template directory ({TEMPLATE_DIR}): {out_path}"
        )
    if not scenario.terrain_write_supported:
        raise WriteBlockedError(
            "This file's terrain block failed load-time verification -- "
            "writing would silently patch the wrong bytes. Terrain mode should "
            "already be disabled for this file; write_scenario() should "
            "never be reached here."
        )

    patched_body = _patch_terrain_block(scenario)
    header_bytes = scenario.header_bytes

    # Ahead of the Messages header splice below, not after it: Number of
    # Players patches FileHeader.player_count at a load-time offset that
    # sits *after* scenario_instructions, so an instructions splice must
    # not run first -- see _patch_header_player_count(). Both blocks raise
    # before anything is written either way, so moving this one earlier
    # changes only which gate reports first when two are broken at once.
    if options is not None and options.has_edits:
        # Re-gated here rather than trusted from construction, the same shape
        # the trigger path uses: OptionsEditModel already refused to exist for a
        # file that failed this, so reaching it means something changed under
        # the model between then and now.
        if not options_write_supported(scenario, options.specs):
            raise WriteBlockedError(
                "This file's map-option block no longer verifies -- patching would "
                "land at an offset that can't be trusted."
            )
        if options.has_diplomacy_edits and not diplomacy_write_supported(scenario):
            raise WriteBlockedError(
                "This file's Diplomacy grid no longer verifies -- patching would "
                "land at an offset that can't be trusted."
            )
        if options.has_player_edits and not players_write_supported(scenario):
            raise WriteBlockedError(
                "This file's Players mode fields no longer verify -- patching would "
                "land at an offset that can't be trusted."
            )
        if options.has_player_count_edit and not player_count_write_supported(scenario):
            raise WriteBlockedError(
                "This file's Number of Players write surface no longer verifies -- "
                "patching would land at an offset that can't be trusted."
            )
        header_bytes = _patch_header_player_count(header_bytes, options)
        patched_body = _patch_options(patched_body, scenario, options)

    if messages is not None and messages.has_edits:
        if not scenario.messages_write_supported:
            raise WriteBlockedError(
                "This file's Messages section failed its load-time verification -- "
                "splicing would land at an offset that can't be trusted. Messages "
                "mode should already be disabled for this file."
            )
        header_bytes = _patch_header_instructions(header_bytes, messages)

    if units is not None and units.has_edits:
        # Re-gated here rather than trusted from construction, the same shape
        # the trigger/options paths use.
        if not scenario.units_write_supported:
            raise WriteBlockedError(
                "This file's Units section failed its alignment gate -- splicing "
                "would land at an offset that can't be trusted. Unit editing should "
                "already be disabled for this file."
            )
        if units.has_added_units:
            patched_body = _patch_next_unit_id(patched_body, units)

    if triggers is not None and triggers.has_edits:
        if not scenario.trigger_write_supported:
            raise WriteBlockedError(
                "This file's Triggers section failed its alignment gate -- splicing "
                "would land at an offset that can't be trusted. Trigger editing should "
                "already be disabled for this file."
            )
        patched_body = _patch_trigger_count(patched_body, scenario, triggers.trigger_count)
        header_bytes = _patch_header_trigger_count(header_bytes, triggers.trigger_count)

    patched_body = _assemble_body(scenario, patched_body, units, triggers)

    if messages is not None and messages.has_edits:
        patched_body = _patch_messages(patched_body, scenario, messages)

    if options is not None:
        # Runs unconditionally (not gated on options.has_edits the way
        # _patch_options() above is): serialize_resizes() already no-ops
        # cleanly for a model with no str16-coded pending edit, and this
        # keeps the ordering rule -- Messages before player_data_1 -- in
        # one place rather than duplicating the has_edits gate here.
        patched_body = _patch_player_data_1(patched_body, scenario, options)

    if header_bytes == scenario.header_bytes and patched_body == scenario.decompressed_body:
        # Nothing patched anything -- reuse the original compressed bytes
        # verbatim rather than recompressing, since re-deflating identical
        # decompressed bytes does not reliably reproduce the original
        # compressed stream (different encoder/level/version upstream).
        compressed = scenario.original_compressed_body
    else:
        compressed = _compress_bytes(patched_body)
    final_bytes = header_bytes + compressed

    existed = out_path.exists()
    if existed and out_path.stat().st_size == len(final_bytes) and out_path.read_bytes() == final_bytes:
        return WriteResult(wrote=False, backups=[])

    backups: list[Path] = []
    if backup:
        from descape.backup import make_backups  # local: descape.backup imports this module

        backups = make_backups(out_path)

    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(final_bytes)
        if existed:
            shutil.copymode(out_path, tmp)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)

    return WriteResult(wrote=True, backups=backups)
