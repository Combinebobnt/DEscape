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
        self._original: dict[str, int] = {}
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
        # tribe_name is skipped: its "c256" codec is a string, and
        # _original/_pending stay int-only until step 3d widens them.
        self._player_field_ids: frozenset[str] = frozenset()
        self._player_targets: dict[str, tuple[player_fields.PlayerWriteTarget, ...]] = {}
        if players_write_supported(loaded):
            specs_by_id = {s.field_id: s for s in player_fields.specs_for(loaded)}
            for key, targets in player_fields.write_targets(loaded).items():
                if targets[0].codec == "c256":
                    continue
                field_id, player_id = player_fields.parse_player_field_id(key)
                spec = specs_by_id[field_id]
                self._player_targets[key] = targets
                self._original[key] = player_fields.current_value(loaded, spec, player_id)
            self._player_field_ids = frozenset(self._player_targets)

        # field_id -> the value the user set, present only while it differs
        # from what the file holds. Setting a field back to its original value
        # removes it, so a change made and undone leaves has_edits False rather
        # than merely producing identical bytes through the patch path.
        self._pending: dict[str, int] = {}

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

    def original_value(self, field_id: str) -> int:
        return self._original[field_id]

    def current_value(self, field_id: str) -> int:
        return self._pending.get(field_id, self._original[field_id])

    def pending_values(self) -> dict[str, int]:
        """Only the fields that differ from the file, for the panel's `values`
        override. Empty for a document nothing has been changed in."""
        return dict(self._pending)

    def set_value(self, field_id: str, value: int) -> None:
        """Record `field_id` as set to raw `value`. Must be wrapped in an undo
        record by the caller -- a model that is dirty while the history is not
        closes the document with no save prompt.

        Raises rather than clamping an out-of-range value: every caller here
        comes from a widget whose range is already the field's own, so a value
        this cannot pack means the spec and the file's datatype disagree, which
        is a bug to surface and not a number to round.
        """
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
        rather than the single patch every other field here produces."""
        patches: list[tuple[int, bytes]] = []
        for field_id, value in self._pending.items():
            if field_id in self._player_targets:
                patches.extend(
                    (target.offset, player_fields.encode_target(target, value))
                    for target in self._player_targets[field_id]
                )
            else:
                patches.append((self._offsets[field_id].offset, self._packers[field_id].pack(value)))
        patches.sort()
        return patches
