"""Byte offsets for the Map Options panel's fixed-width scalar fields, and
the load-time verification gate for writing them.

Every field this module handles lives in a section (GlobalVictory,
Diplomacy, Map, Options) that DEscape's ordinary load walk already parses in
full -- see scenario_io.load_map_and_units(). What that walk does *not*
capture is the byte offset within decompressed_body where a given field
sits, because most of these sections have leading or interspersed
variable-length retrievers (Map's leading str16s, Options' per-player
disabled-id arrays, Diplomacy's per_player_diplomacy struct array) that make
a forward walk from the section's start untrustworthy without simulating
their repeat logic.

The fix, mirroring descape/trigger_model.py's SectionRegions: walk
*backward* from a byte offset scenario_io.py already trusts (a section's
end, or -- for Map -- terrain_block_offset, which is exactly "the end of
every Map field except terrain_data"), subtracting each retriever's parsed
byte length in reverse declaration order. Every retriever passed on the way
back is fixed-width by construction here (see descape/option_fields.py's
field list for why), so the backward walk never has to resolve a repeat
count itself -- it just asks
the already-parsed retriever how many bytes it actually occupied in this
file, via scenario_io.retriever_length().

legacy_exec_order (Triggers section) is deliberately excluded here --  its
read/write path is descape/trigger_model.py's TriggerEditModel, not this
module: that field lives inside the Triggers region and is emitted by
TriggerEditModel.serialize()'s tail rather than byte-patched here.

OptionsEditModel, at the bottom of this module, is the write half built on
those offsets: one in-place byte patch per changed scalar, no offset shift,
and nothing at all for a document that was only browsed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

from descape import player_fields
from descape.diplomacy_fields import (
    allied_victory_offsets,
    stance_offsets,
    verify_diplomacy_block,
)
from descape.option_fields import OptionFieldSpec, current_value, specs_for
from descape.scenario_io import LoadedScenario, retriever_length

# The LoadedScenario attribute holding the trusted byte offset, within
# decompressed_body, where each section *ends*. Walking backward from here
# is what lets every field below skip every variable-length retriever ahead
# of it in the section -- see this module's docstring.
_ANCHOR_ATTR = {
    "GlobalVictory": "global_victory_section_end",
    "Diplomacy": "diplomacy_section_end",
    "Map": "terrain_block_offset",
    "Options": "options_section_end",
}

# Map's anchor is not the section's true end: terrain_block_offset is where
# terrain_data *starts* (map_section_end - TERRAIN_STRUCT_SIZE * w * h, per
# scenario_io.py), because terrain_data gets its own, stricter verification
# path (scenario_io._verify_terrain_block()). Every other anchored section's
# stored offset really is one past its own last retriever, so only Map needs
# its trailing retriever excluded before the reversed walk starts -- treating
# every anchor as "one past the section's last retriever" would silently
# subtract terrain_data's 100800+ bytes here and land at a wildly negative
# offset for everything before it.
_ANCHOR_EXCLUDES_TAIL = {
    "Map": ("terrain_data",),
}


@dataclass(frozen=True)
class FieldOffset:
    offset: int  # byte offset within decompressed_body
    length: int  # byte length of this retriever's parsed data


def _anchor(loaded: LoadedScenario, section: str) -> int:
    """-1 if `section`'s anchor cannot be trusted. Map's anchor
    (terrain_block_offset) is already -1 whenever terrain_write_supported is
    False -- scenario_io.py sets both together -- so no separate check is
    needed here; the other three anchors are set unconditionally by the load
    walk and have no comparable failure mode of their own."""
    return getattr(loaded, _ANCHOR_ATTR[section])


def field_offsets(
    loaded: LoadedScenario, specs: Sequence[OptionFieldSpec]
) -> tuple[dict[OptionFieldSpec, FieldOffset], bool]:
    """Byte offset+length, within decompressed_body, of every spec's
    retriever.

    Returns (offsets, all_available). `all_available` is False if any spec
    in `specs` sits in a section whose anchor could not be trusted -- e.g.
    Map's terrain block failed its own load-time verification -- and is the
    signal callers use to fail write support closed rather than silently
    offering a partial write surface for just the sections that did verify.
    Specs whose section is "Triggers" (legacy_exec_order) are silently
    skipped: that field's offset is trigger_model.py's concern, not this
    module's.
    """
    by_section: dict[str, list[OptionFieldSpec]] = {}
    for spec in specs:
        if spec.section not in _ANCHOR_ATTR:
            continue
        by_section.setdefault(spec.section, []).append(spec)

    offsets: dict[OptionFieldSpec, FieldOffset] = {}
    all_available = True
    for section, section_specs in by_section.items():
        anchor = _anchor(loaded, section)
        if anchor < 0:
            all_available = False
            continue
        retriever_map = loaded._scenario.sections[section].retriever_map
        excluded = _ANCHOR_EXCLUDES_TAIL.get(section, ())
        names = [name for name in retriever_map if name not in excluded]
        wanted = {spec.retriever: spec for spec in section_specs}
        pos = anchor
        for name in reversed(names):
            length = retriever_length(retriever_map[name])
            pos -= length
            spec = wanted.pop(name, None)
            if spec is not None:
                offsets[spec] = FieldOffset(pos, length)
            if not wanted:
                break
    return offsets, all_available


def verify_options_block(loaded: LoadedScenario, specs: Sequence[OptionFieldSpec]) -> bool:
    """True iff every spec's computed offset reproduces exactly the bytes
    its retriever already parsed -- the load-time trust check mirroring
    scenario_io._verify_terrain_block()/_verify_units_block(). Fails closed
    (False) if any section's anchor was unavailable or `specs` is empty,
    rather than reporting success for a subset it never checked.
    """
    offsets, all_available = field_offsets(loaded, specs)
    if not all_available or not offsets:
        return False
    body = loaded.decompressed_body
    for spec, fo in offsets.items():
        if fo.offset < 0 or fo.offset + fo.length > len(body):
            return False
        retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
        if body[fo.offset : fo.offset + fo.length] != retriever.get_data_as_bytes():
            return False
    return True


# Little-endian struct per AoE2ScenarioParser datatype string. The format comes
# from the retriever the file itself was parsed with, never from a hand-written
# per-spec declaration: signedness is real here (ai_map_type is s32,
# no_waves_on_shore is s8) and a spec that drifted from the structure it
# describes would write a value the game reads back as something else. The
# proof that this mapping is right is not the table -- it is
# test_every_writable_field_repacks_its_own_stored_bytes, which round-trips the
# value already in the file through it and compares against the file's bytes.
_PACK_STRUCT = {
    "u8": struct.Struct("<B"),
    "s8": struct.Struct("<b"),
    "u16": struct.Struct("<H"),
    "s16": struct.Struct("<h"),
    "u32": struct.Struct("<I"),
    "s32": struct.Struct("<i"),
}


class OptionEditsUnavailableError(Exception):
    """Raised by OptionsEditModel() for a file whose map options cannot be
    written: an anchor that could not be trusted, a field whose computed
    offset does not reproduce its own parsed bytes, or a retriever whose
    datatype this module cannot pack."""


def pack_struct_for(retriever) -> struct.Struct | None:
    """The struct this retriever's value packs through, or None if it is not
    a fixed-width integer this module can write.

    The width check is not redundant with the lookup: `datatype.var` and the
    bytes the retriever actually consumed are independently sourced (the
    structure definition vs. the parse), and a disagreement between them means
    the offset walk and the packer are describing different fields.
    """
    packer = _PACK_STRUCT.get(getattr(retriever.datatype, "var", None))
    if packer is None or packer.size != retriever_length(retriever):
        return None
    return packer


def options_write_supported(loaded: LoadedScenario, specs: Sequence[OptionFieldSpec]) -> bool:
    """Whether every byte-patched map-option row can be written back.

    All-or-nothing on purpose, matching verify_options_block(): a partial write
    surface would offer some rows and silently drop others on a file where the
    walk half-worked, and there is no way for a user to tell those apart. The
    trigger execution-order row has its own separate gate
    (trigger_model.exec_order_write_supported) because this module does not
    walk the Triggers section at all.
    """
    if not verify_options_block(loaded, specs):
        return False
    offsets, _ = field_offsets(loaded, specs)
    for spec in offsets:
        retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
        if pack_struct_for(retriever) is None:
            return False
    return True


def diplomacy_write_supported(loaded: LoadedScenario) -> bool:
    """Whether the Diplomacy grid (per-player stance cells and allied-victory
    flags) can be written back.

    A gate independent of options_write_supported() on purpose: folding this
    into that all-or-nothing scalar gate would grey out every Map Options row
    whenever the grid failed, and this file's Diplomacy section anchor
    failing must not do that. Every grid cell is a raw u32
    (descape/diplomacy_fields.py's module docstring), not a per-file
    retriever datatype, so unlike pack_struct_for() there is no datatype
    check to make here -- verify_diplomacy_block() already fails closed on
    both the anchor and the section's measured length.
    """
    return verify_diplomacy_block(loaded)


def players_write_supported(loaded: LoadedScenario) -> bool:
    """Whether Players mode's per-player fields can be written back.

    A gate independent of options_write_supported() and
    diplomacy_write_supported(), for the same reason those two are
    independent of each other: a PlayerDataTwo/Units failure must not grey
    out a single Map Options row or the Diplomacy grid, and vice versa.
    player_fields.verify_player_block() already fails closed on every
    anchor, on units_write_supported, and on an empty target set.
    """
    return player_fields.verify_player_block(loaded)


def player_count_write_supported(loaded: LoadedScenario) -> bool:
    """Whether Number of Players can be written back.

    A *fourth* gate, independent of players_write_supported() rather than
    folded into it, because the two come apart on a real corpus file: this
    one additionally needs FileHeader.player_count's own offset, which
    comes from a header walk that does not reconcile on a scenario version
    1.37 file. Folding it in would grey out every other per-player row on
    that file to protect one row that is genuinely unwritable there.
    See player_fields.verify_player_count_block().
    """
    return player_fields.verify_player_count_block(loaded)


class OptionsEditModel:
    """Per-document map-option edit state: which scalars the user changed, and
    the byte patches that writes them back.

    Construct one per open scenario, on the first real edit rather than at
    panel populate -- the same lazy contract viewer.py's _ensure_trigger_edits()
    has, and for the same reason: a browse-only session must save
    byte-identically.

    Every field here is fixed-width, so every write is an in-place patch that
    shifts no offset -- the same shape as the existing 4-byte
    Options.number_of_triggers patch. Nothing in this model mutates the parsed
    retrievers, which is why the panel has to be told the pending values
    explicitly on a repopulate: re-reading the file would show the original.
    """

    def __init__(self, loaded: LoadedScenario, specs: Sequence[OptionFieldSpec] | None = None):
        if specs is None:
            specs = specs_for(loaded)
        offsets, all_available = field_offsets(loaded, specs)
        if not all_available or not offsets:
            raise OptionEditsUnavailableError(
                "This file's map-option offsets could not all be resolved, so patching "
                "would land at an offset that cannot be trusted."
            )
        if not verify_options_block(loaded, specs):
            raise OptionEditsUnavailableError(
                "This file's map-option block failed its verification -- at least one "
                "computed offset does not hold the bytes its retriever parsed."
            )

        self.loaded = loaded
        self._offsets: dict[str, FieldOffset] = {}
        self._packers: dict[str, struct.Struct] = {}
        self._specs: dict[str, OptionFieldSpec] = {}
        self._original: dict[str, int | str] = {}
        for spec, fo in offsets.items():
            retriever = loaded._scenario.sections[spec.section].retriever_map[spec.retriever]
            packer = pack_struct_for(retriever)
            if packer is None:
                raise OptionEditsUnavailableError(
                    f"{spec.field_id!r} is a {getattr(retriever.datatype, 'var', '?')} "
                    f"retriever, which this module cannot pack."
                )
            self._offsets[spec.field_id] = fo
            self._packers[spec.field_id] = packer
            self._specs[spec.field_id] = spec
            self._original[spec.field_id] = current_value(loaded, spec)

        # Diplomacy grid cells: an additive entry set under synthetic ids
        # ("stance:3:5", "allied_victory:3"), gated by diplomacy_write_supported()
        # rather than the check above -- see that function's docstring. A grid
        # failure here does not raise: it just leaves the grid unwritable
        # through this model, the same outcome set_value() already gives any
        # field_id this model was never given an offset for. Every cell is a
        # raw u32 (diplomacy_fields.py's module docstring), so the packer is
        # fixed rather than looked up per retriever.
        self._diplomacy_field_ids: frozenset[str] = frozenset()
        if diplomacy_write_supported(loaded):
            cell_packer = _PACK_STRUCT["u32"]
            diplomacy_offsets = {**stance_offsets(loaded), **allied_victory_offsets(loaded)}
            body = loaded.decompressed_body
            for field_id, fo in diplomacy_offsets.items():
                self._offsets[field_id] = FieldOffset(fo.offset, fo.length)
                self._packers[field_id] = cell_packer
                (self._original[field_id],) = cell_packer.unpack_from(body, fo.offset)
            self._diplomacy_field_ids = frozenset(diplomacy_offsets)

        # Players mode fields: an additive entry set under synthetic ids
        # ("player:food:3"), gated by players_write_supported() rather than
        # the map-option check above -- same reasoning as the Diplomacy
        # block. A gate failure here does not raise: it just leaves every
        # per-player row unwritable through this model. Stores the full
        # target tuple per id, not one offset -- a mirrored field
        # (player_fields._MIRRORS, plus color/pop_limit's own optional
        # mirrors) patches two or three locations from one edit, which the
        # single-offset _offsets/_packers dicts above have no room for.
        # tribe_name's "c256" codec is a string -- _original/_pending are
        # int | str (step 3d) so it rides the same additive set as every
        # other Tier-1 player field, with encode_target() (called from
        # set_value()/serialize_patches() below) dispatching on the target's
        # own codec rather than a struct.Struct the way the int-only fields
        # above do.
        self._player_field_ids: frozenset[str] = frozenset()
        self._player_targets: dict[str, tuple[player_fields.PlayerWriteTarget, ...]] = {}
        if players_write_supported(loaded):
            specs_by_id = {s.field_id: s for s in player_fields.specs_for(loaded)}
            for key, targets in player_fields.write_targets(loaded).items():
                field_id, player_id = player_fields.parse_player_field_id(key)
                spec = specs_by_id[field_id]
                self._player_targets[key] = targets
                self._original[key] = player_fields.current_value(loaded, spec, player_id)
            self._player_field_ids = frozenset(self._player_targets)

        # Number of Players: one scenario-wide scalar, not a per-player row,
        # and the only field in this model whose write reaches two buffers
        # (eight `active` flags in decompressed_body, plus
        # FileHeader.player_count in header_bytes -- see header_patch()).
        # Kept out of _player_targets because its stored value is a *count*,
        # not a value to encode into each target: set_value() would otherwise
        # try to pack 5 into each of the eight flags. It is still counted in
        # _player_field_ids so has_player_edits (and therefore
        # write_scenario()'s re-gate) covers a count-only edit.
        self._player_count_targets: tuple[player_fields.PlayerWriteTarget, ...] = ()
        if player_count_write_supported(loaded):
            targets = player_fields.player_count_targets(loaded)
            if targets is not None:
                self._player_count_targets = targets
                self._original[player_fields.PLAYER_COUNT_FIELD_ID] = (
                    player_fields.defined_player_count(loaded)
                )
                self._player_field_ids |= {player_fields.PLAYER_COUNT_FIELD_ID}

        # field_id -> the value the user set, present only while it differs
        # from what the file holds. Setting a field back to its original value
        # removes it, so a change made and undone leaves has_edits False rather
        # than merely producing identical bytes through the patch path.
        self._pending: dict[str, int | str] = {}

    # -- state ---------------------------------------------------------------

    @property
    def specs(self) -> tuple[OptionFieldSpec, ...]:
        """The specs this model resolved offsets for, for a caller that needs
        to re-run the gate against the same field set -- scenario_write.py
        re-verifies rather than trusting construction. Diplomacy grid cells
        carry no spec (see has_diplomacy_edits for their equivalent)."""
        return tuple(self._specs.values())

    @property
    def has_edits(self) -> bool:
        return bool(self._pending)

    @property
    def has_diplomacy_edits(self) -> bool:
        """Whether any pending edit is a Diplomacy grid cell, the signal
        scenario_write.py needs to re-run diplomacy_write_supported() on save
        -- options_write_supported()'s re-gate does not cover these, since
        they carry no OptionFieldSpec (see the `specs` property)."""
        return any(field_id in self._diplomacy_field_ids for field_id in self._pending)

    @property
    def has_player_edits(self) -> bool:
        """Whether any pending edit is a Players mode field, the signal
        scenario_write.py needs to re-run players_write_supported() on save
        -- options_write_supported()'s re-gate does not cover these, for the
        same reason has_diplomacy_edits' docstring gives."""
        return any(field_id in self._player_field_ids for field_id in self._pending)

    @property
    def has_player_count_edit(self) -> bool:
        """Whether Number of Players is pending -- the signal
        scenario_write.py needs to re-run player_count_write_supported() on
        save, and to know it must patch header_bytes as well as the body."""
        return player_fields.PLAYER_COUNT_FIELD_ID in self._pending

    def original_value(self, field_id: str) -> int | str:
        return self._original[field_id]

    def current_value(self, field_id: str) -> int | str:
        return self._pending.get(field_id, self._original[field_id])

    def pending_values(self) -> dict[str, int | str]:
        """Only the fields that differ from the file, for the panel's `values`
        override. Empty for a document nothing has been changed in."""
        return dict(self._pending)

    def set_value(self, field_id: str, value: int | str) -> None:
        """Record `field_id` as set to raw `value`. Must be wrapped in an undo
        record by the caller -- a model that is dirty while the history is not
        closes the document with no save prompt.

        Raises rather than clamping an out-of-range value: every caller here
        comes from a widget whose range is already the field's own, so a value
        this cannot pack means the spec and the file's datatype disagree, which
        is a bug to surface and not a number to round. A player field's own
        codec decides the check -- encode_target() raises for tribe_name's
        c256 overflow the same way a struct.Struct.pack() raises for an
        out-of-range int.
        """
        if field_id == player_fields.PLAYER_COUNT_FIELD_ID:
            if not self._player_count_targets:
                raise KeyError("Number of players is not writable on this file")
            if not isinstance(value, int) or not 1 <= value <= player_fields.NUM_PLAYERS:
                raise ValueError(
                    f"{value!r} is not a player count this file can hold "
                    f"(1..{player_fields.NUM_PLAYERS})"
                )
            if value == self._original[field_id]:
                self._pending.pop(field_id, None)
            else:
                self._pending[field_id] = value
            return

        if field_id in self._player_targets:
            for target in self._player_targets[field_id]:
                try:
                    player_fields.encode_target(target, value)
                except ValueError as e:
                    raise ValueError(f"{value!r} does not fit {field_id!r}: {e}") from e
            if value == self._original[field_id]:
                self._pending.pop(field_id, None)
            else:
                self._pending[field_id] = value
            return

        if field_id not in self._offsets:
            raise KeyError(f"{field_id!r} is not a writable map option on this file")
        packer = self._packers[field_id]
        try:
            packer.pack(value)
        except struct.error as e:
            raise ValueError(f"{value!r} does not fit {field_id!r} ({packer.format}): {e}") from e
        if value == self._original[field_id]:
            self._pending.pop(field_id, None)
        else:
            self._pending[field_id] = value

    # -- serialization -------------------------------------------------------

    def serialize_patches(self) -> list[tuple[int, bytes]]:
        """(offset within decompressed_body, replacement bytes) for every
        pending edit, in ascending offset order. Empty for a clean model, which
        is what keeps a browse-only save byte-identical. A Players mode field
        emits one patch per target -- two or three for a mirrored field --
        rather than the single patch every other field here produces.

        A str16-coded Players mode field (civilization/architecture on a
        1.56+ file) is excluded here -- routed through serialize_resizes()
        instead, since it can change the byte length of its containing
        entry, which this fixed-offset/fixed-length patch shape cannot
        express. Filtered by the target's own codec, not the field id: a
        pre-1.56 civilization/architecture edit is a plain u32 and keeps
        riding this method unchanged."""
        patches: list[tuple[int, bytes]] = []
        for field_id, value in self._pending.items():
            if field_id == player_fields.PLAYER_COUNT_FIELD_ID:
                # Nine locations from one edit: the eight `active` flags
                # here, plus FileHeader.player_count via header_patch(),
                # which is a different buffer entirely.
                patches.extend(
                    zip(
                        (t.offset for t in self._player_count_targets),
                        player_fields.encode_player_count(self._player_count_targets, value),
                    )
                )
            elif field_id in self._player_targets:
                targets = self._player_targets[field_id]
                if targets[0].codec == "str16":
                    continue
                patches.extend(
                    (target.offset, player_fields.encode_target(target, value))
                    for target in targets
                )
            else:
                patches.append((self._offsets[field_id].offset, self._packers[field_id].pack(value)))
        patches.sort()
        return patches

    def header_patch(self) -> tuple[int, int, bytes] | None:
        """(start, end, replacement) for FileHeader.player_count within
        header_bytes, or None when Number of Players is not pending.

        The only edit in this model that reaches a second buffer: every
        other one is a decompressed_body offset, which is why this is a
        separate method rather than another serialize_patches() entry.
        Shaped like MessagesEditModel.header_patch() so scenario_write.py
        applies both the same way, but this one never changes the header's
        length -- player_count is a fixed-width field.
        """
        if not self.has_player_count_edit:
            return None
        start, end = self.loaded.header_player_count_span
        count = int(self._pending[player_fields.PLAYER_COUNT_FIELD_ID])
        return start, end, count.to_bytes(end - start, "little")

    def serialize_resizes(self, body: bytes | None = None) -> list[tuple[int, int, bytes]]:
        """(start, end, replacement) for every region this save resizes --
        currently at most one entry, the whole player_data_1 array, emitted
        only when at least one pending edit is a str16-coded Players mode
        field (civilization/architecture on a 1.56+ file). Empty for a
        clean model or one with only serialize_patches()-shaped edits, the
        same containment every other model in this codebase gives a
        browse-only save.

        One splice covers every str16 edit in this save, not one per field:
        player_fields.player_data_1_splice() rebuilds the whole array from
        the bytes already there with every edited entry's span substituted,
        so a civilization edit on P3 and an architecture edit on P5 in the
        same save produce one region, not two overlapping ones.

        `body` is the caller's partially-patched body, and the write path
        must pass it: Number of Players patches `active` *inside*
        player_data_1 through serialize_patches(), so rebuilding the array
        from the original bytes would silently discard a count edit made in
        the same save. Defaults to the original body for a caller with no
        patches to preserve.
        """
        edits: list[tuple[int, str, str]] = []
        for field_id, value in self._pending.items():
            targets = self._player_targets.get(field_id)
            if targets is None or targets[0].codec != "str16":
                continue
            spec_field_id, player_id = player_fields.parse_player_field_id(field_id)
            edits.append((player_id, spec_field_id, value))
        if not edits:
            return []
        result = player_fields.player_data_1_splice(self.loaded, edits, body)
        return [] if result is None else [result]
