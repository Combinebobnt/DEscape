"""Phase 3.5a's byte-blob-per-unit model: the Units-section write path.

Same shape as descape/trigger_model.py (a byte blob per record, spliced back
verbatim unless the record was actually edited) but two structural
simplifications apply here that don't apply there:

1. **No remap class.** A unit's `reference_id` is stable and file-wide unique
   (measured against the real 20-file corpus), and UnitStruct carries no
   positional identity of its own. Nothing here renumbers ids or rewrites
   other units' references the way remove_triggers()/reorder_triggers() do,
   so no operation below ever dirties a unit the caller did not name.
2. **Player is positional.** A unit's owner is which of the 9 per-player
   lists it lives in, not a stored field, so reassign() is a pure list-to-list
   move with zero re-serialization -- the moved unit's blob travels with it.

Why the blob splice, stated honestly (measured, not assumed): on every
corpus file, reassembling players_units from each unit's own original byte
slice is byte-identical to the source file. Re-serializing the *whole*
section through the library drifts by exactly one byte per unit on
v1.55+ files (an empty/short `caption_string` str32 the game writes as
length 0 or N+1, and the library always writes as length N+2 -- one NUL more
than either): 100% of units drift on v1.55+, 0% on v1.54-and-earlier. Because
that drift disappears after normalizing (see `_serialize_unit` below), a
whole-section re-serialization and this module's blob splice produce
*identical* output on the measured corpus -- byte-identity alone cannot
tell them apart. The blob splice is kept anyway for three reasons the plan
records: doctrine consistency with the terrain/options/trigger write paths,
insurance against a future drift source affecting only re-serialized units
(this module breaks one unit's bytes where whole-section reserialization
would break every unit in the file), and cost (no commit-readback for units
nobody touched).

**Rotate is narrowly scoped, not out of scope.** For walls, gates and most
GAIA doodads, `rotation` is a shape-variant index rather than an angle
(AGENTS.md's hard rule), and transforming one would write a value the game
re-derives or a frame the author never picked. set_rotation() is the sole
operation here that transforms an existing rotation, and it refuses any const
descape/unit_rotation.py does not classify ANGLE -- that guard is what carries
the old blanket rule forward as "never transform a non-angle rotation".
add()'s `rotation` parameter stays a verbatim pass-through, never validated or
normalized as an angle: storing a caller-supplied value on a newly placed unit
transforms nothing.

Units are parsed eagerly at load time (scenario_io._load_map_and_units()
depoisons and parses them unconditionally, before any lazy step), unlike
Triggers -- so, unlike TriggerEditModel, this module holds no "never cache
the manager, always re-fetch" invariant: loaded.unit_manager is a live
reference good for the document's whole lifetime. The cost of that
simplicity: there is no parse_units()-style re-depoison hook the way
parse_triggers() re-depoisons on every call. Reading a unit's
caption_string_id/caption_string on document A *after* opening document B in
the same process can raise UnsupportedAttributeError if B's scenario version
doesn't support those fields (see descape/library_compat.py's docstring on
class-level poisoning). Documented and accepted as an open limitation for
this headless slice -- nothing here reads captions back after the fact, and
a future UI session inherits the same caveat parse_triggers() already lives
with for the trigger side.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

from AoE2ScenarioParser.objects.data_objects.unit import Unit
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager

from descape import gate_orientation, render, terrain_palette, unit_rotation
from descape.edit_history import EditHistory, UnitDiffRecord
from descape.scenario_io import LoadedScenario

# str32's length prefix is a fixed 4-byte signed int in every DE structure
# version that defines caption_string (v1.55+; see versions/DE/*/structure.json)
# -- confirmed directly, not derived from retriever.datatype, since the field
# simply doesn't exist in entry.retriever_map on older versions.
_CAPTION_FIELD = "caption_string"
_CAPTION_PREFIX_SIZE = 4

_UNIT_COUNT_STRUCT = struct.Struct("<I")  # PlayerUnitsStruct.unit_count, per
# versions/DE/*/structure.json -- always u32, every version measured.

# DataHeader.next_unit_id_to_place is a u32 at decompressed_body[0:4] in
# every corpus file (plan finding 8) -- read here once at construction,
# never through UnitManager.next_unit_id, which is a side-effecting
# generator property a full manager.commit() advances and does not reliably
# write back (plan fact 11).
_NEXT_UNIT_ID_STRUCT = struct.Struct("<I")

# UnitManager._link_list[0]'s own `.name` -- selected by name (fact 11)
# rather than passed as a literal object, so a commit(link_list=[...]) never
# touches _link_list[1] ("next_unit_id"), whose push is itself a same-value
# no-op but whose *read* (getattr(manager, "next_unit_id")) is the
# side-effecting generator call this module exists to avoid.
_PLAYER_UNITS_LINK_NAME = "_player_units"


class UnitEditsUnavailableError(Exception):
    """Raised by UnitEditModel() for a file whose units cannot be edited: the
    Units section failed load-time byte-alignment verification
    (units_write_supported is False), the file has other than 9 unit
    sections, or the per-unit construction gate found a unit whose bytes
    the normalizer cannot reproduce (see _serialize_unit's docstring)."""


def _strip_one_trailing_nul(data: bytes) -> bytes:
    """One trailing NUL removed from a length-prefixed str32 field's raw
    bytes, with its length prefix decremented to match.

    Uses `assert`, not `if`: a caller reaching this with a library form that
    doesn't end in a NUL trail means the assumption this normalizer rests on
    has already broken, and this must fail loudly rather than silently strip
    the wrong byte.
    """
    assert len(data) >= _CAPTION_PREFIX_SIZE, f"caption_string field too short to hold its own length prefix: {data!r}"
    length = int.from_bytes(data[:_CAPTION_PREFIX_SIZE], "little", signed=True)
    body = data[_CAPTION_PREFIX_SIZE:]
    assert body[-1:] == b"\x00", f"library caption_string form does not end in a NUL trail: {data!r}"
    return (length - 1).to_bytes(_CAPTION_PREFIX_SIZE, "little", signed=True) + body[:-1]


def _serialize_unit(entry) -> bytes:
    """The game-shaped byte form of one committed UnitStruct entry.

    Concatenates every retriever's own get_data_as_bytes() in declaration
    order (matching AoE2FileSection's on-disk struct layout, the same
    assumption tools/_fixture_bytes.py's _game_style_bytes() makes), except
    caption_string, which additionally has one trailing NUL stripped
    whenever the caption is the *empty* string. Fields this scenario
    version's UnitStruct doesn't define (caption_string_id/caption_string
    before 1.54/1.55) are simply absent from entry.retriever_map, so nothing
    else needs to know about scenario version here.

    Special-cased on emptiness rather than applied unconditionally, despite
    plan fact 12 finding the library appends a NUL to *every* caption it
    writes, not just empty ones: measured directly against
    tests/fixtures/units_120x120.aoe2scenario's own deliberately non-empty
    caption unit (reference_id 300, plan stage 1.2's "the only coverage
    anywhere for fact 12's unmeasured branch"), an unconditional strip does
    NOT reproduce that unit's raw bytes -- its generator
    (tools/_fixture_bytes.py's _game_style_bytes()) only special-cases the
    empty string too, leaving a non-empty caption in the library's own form.
    Plan open question 1 pre-authorized exactly this revision ("if [stage]
    1.3 falsifies it, revise before stage 2"): the true game encoding for a
    non-empty caption is still genuinely unmeasured (no real corpus unit has
    one), so this only ever touches the one case that IS measured (finding
    2/3, 158,394 real corpus units), and never guesses at the unmeasured
    one. The per-file construction gate below is still what fails closed if
    even the empty-caption case turns out wrong for some future file.
    """
    parts = []
    for name, retriever in entry.retriever_map.items():
        data = retriever.get_data_as_bytes()
        if name == _CAPTION_FIELD and retriever.data == "":
            data = _strip_one_trailing_nul(data)
        parts.append(data)
    return b"".join(parts)


def _raw_unit_blobs(loaded: LoadedScenario, players_units) -> list[list[bytes]]:
    """Per-player lists of each unit's original on-disk byte slice, walked
    from units_block_offset. Mirrors scenario_io._verify_units_block()'s and
    tests/test_units_fixture.py's _raw_unit_slices()'s own walk: each
    PlayerUnitsStruct's own byte_length covers its leading unit_count u32
    *and* every one of its units, so the walk steps over 4 bytes for
    unit_count before iterating that player's own units.
    """
    offset = loaded.units_block_offset
    blobs: list[list[bytes]] = []
    for player_units in players_units:
        offset += 4  # unit_count
        player_blobs = []
        for entry in player_units.retriever_map["units"].data:
            player_blobs.append(loaded.decompressed_body[offset : offset + entry.byte_length])
            offset += entry.byte_length
        blobs.append(player_blobs)
    if offset != loaded.units_section_end:
        raise UnitEditsUnavailableError(
            f"Units section walk landed at {offset}, expected units_section_end "
            f"{loaded.units_section_end} -- refusing to construct an edit model"
        )
    return blobs


def _check_all_units_reproduce(blobs_by_player, entries_by_player) -> None:
    """The per-file construction gate (plan stage 2.2): re-derives finding 3
    on this exact file, at this exact load, rather than trusting a one-time
    corpus probe. Finding 4 (field edits don't reach the bytes without a
    commit) is what makes comparing against still-unedited retrievers safe
    here. Fails closed per unit -- see _serialize_unit's docstring for why
    there is no file-wide caption "style" to decide in advance."""
    for player_blobs, player_entries in zip(blobs_by_player, entries_by_player):
        for raw, entry in zip(player_blobs, player_entries):
            produced = _serialize_unit(entry)
            if produced != raw:
                ref_id = entry.retriever_map["reference_id"].data
                raise UnitEditsUnavailableError(
                    f"Unit reference_id {ref_id} does not reproduce its original bytes "
                    f"({len(produced)} vs {len(raw)} bytes) -- refusing to construct an "
                    f"edit model for this file"
                )


def _highest_reference_id(manager: UnitManager) -> int:
    return max((u.reference_id for units in manager.units for u in units), default=0)


def _player_units_link(manager: UnitManager):
    return next(link for link in manager._link_list if getattr(link, "name", None) == _PLAYER_UNITS_LINK_NAME)


def span_low_corner(unit: Unit) -> tuple[int, int]:
    """The (tile_x, tile_y) low corner of `unit`'s footprint, the anchor
    set_unit_const() preserves across a swap.

    Deliberately render._span_start() per axis and never unit_tile_bounds(),
    which clamps to the map: a clamped corner would re-anchor a map-edge gate
    onto a different tile than it started on. Public because the viewer's
    map-edge fit check has to measure the same corner the model re-anchors
    from, rather than re-deriving the parity rule on its own side.
    """
    span_x, span_y = terrain_palette.tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
    return render._span_start(unit.x, span_x), render._span_start(unit.y, span_y)


# The only fields any in-place operation mutates -- add/remove/reassign move
# whole Unit objects between/within lists rather than editing fields, so
# PlayerListSnapshot's `units` list (membership + order, by identity) already
# covers them. Unit carries no nested mutable graph (unlike Trigger), so a
# plain value tuple restored by field assignment is enough -- no deepcopy
# budget to worry about the way TriggerSnapshot's `states` has.
#
# `rotation` is in here because set_rotation() edits it in place, and
# `unit_const` because set_unit_const() does. Leaving either out would make
# undo restore the BYTES (the blob comes back) while the live Unit object
# stayed rotated or stayed cycled, so the inspector and the render would
# disagree with what saving would actually write.
UnitState = tuple[float, float, float, float, int]


def unit_state(unit: Unit) -> UnitState:
    return (unit.x, unit.y, unit.z, unit.rotation, unit.unit_const)


@dataclass
class PlayerListSnapshot:
    """One player's unit list, membership/order/field-state, captured whole.

    Whole-list rather than a positional delta, on purpose: it is what makes
    reassign's undo restore the unit to its *exact* original index rather
    than merely its original owner (the insertion-position rule -- see
    UnitEditModel.reassign's docstring), and unlike triggers there is no
    O(bytes) cost to worry about capturing more than strictly needed, since
    a UnitState is four floats and a const.
    """

    units: list = field(default_factory=list)
    blobs: list = field(default_factory=list)
    states: list = field(default_factory=list)


@dataclass
class UnitSnapshot:
    """One side of a unit undo record. `players` holds only the player
    indices the edit actually touched -- 1 for move/add/remove, 2 for
    reassign -- never all 9."""

    players: dict[int, PlayerListSnapshot] = field(default_factory=dict)
    next_unit_id: int = 0
    dirty: bool = False


class UnitEditModel:
    """Per-document unit edit state: which units are dirty, and how the
    Units section serializes given that.

    Construct lazily, on a document's first unit *edit* -- never at load or
    panel-populate time. The construction gate below (_check_all_units_
    reproduce) scans every unit in the file, and the largest corpus file has
    10,871 of them; a browse-only session must not pay that cost.
    """

    def __init__(self, loaded: LoadedScenario):
        if not loaded.units_write_supported:
            raise UnitEditsUnavailableError(
                "This file's Units section failed load-time verification, so writing "
                "units back would splice at an offset that cannot be trusted."
            )
        if loaded.number_of_unit_sections != 9:
            raise UnitEditsUnavailableError(
                f"This file has {loaded.number_of_unit_sections} unit sections, expected "
                f"9 (GAIA + 8 players) -- refusing to construct an edit model for an "
                f"unfamiliar layout"
            )

        self.loaded = loaded
        manager = loaded.unit_manager
        players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
        entries_by_player = [pu.retriever_map["units"].data for pu in players_units]
        blobs_by_player = _raw_unit_blobs(loaded, players_units)

        for player, (units, entries) in enumerate(zip(manager.units, entries_by_player)):
            if len(units) != len(entries):
                raise UnitEditsUnavailableError(
                    f"Player {player} has {len(units)} parsed units but {len(entries)} raw "
                    f"struct entries -- refusing to construct an edit model"
                )

        _check_all_units_reproduce(blobs_by_player, entries_by_player)

        self._blobs: list[list] = blobs_by_player
        # Strong references to the Unit objects each blob belongs to, one
        # list per player -- mirrors TriggerEditModel._tracked, and enforces
        # the banned `unit.player =` setter the same way: a unit moved by it
        # surfaces as a loud _check_alignment() error on the next
        # serialize()/restore() instead of silently splicing a blob into the
        # wrong player's section.
        self._tracked: list[list] = [list(units) for units in manager.units]
        self._next_unit_id = _NEXT_UNIT_ID_STRUCT.unpack_from(loaded.decompressed_body, 0)[0]
        self._dirty = False
        self._has_added_units = False
        # Set only between begin_unit_edit() and commit/abort -- the
        # pre-mutation snapshot of the player list(s) the caller declared
        # this edit may touch.
        self._pending: UnitSnapshot | None = None

    # -- state -----------------------------------------------------------

    @property
    def has_edits(self) -> bool:
        """True if saving would take the re-serializing branch."""
        return self._dirty

    @property
    def has_added_units(self) -> bool:
        """True once add() has been called at least once this session --
        scenario_write.py only patches next_unit_id_to_place when this is
        True, and never lowers it otherwise."""
        return self._has_added_units

    @property
    def next_unit_id(self) -> int:
        """The value scenario_write.py should patch DataHeader.
        next_unit_id_to_place to, once has_added_units is True."""
        return self._next_unit_id

    def _locate(self, unit: Unit) -> tuple[int, int]:
        for player, tracked in enumerate(self._tracked):
            for index, candidate in enumerate(tracked):
                if candidate is unit:
                    return player, index
        raise ValueError("unit is not tracked by this model -- was it added through it?")

    def _reserve_reference_id(self) -> int:
        """Findings 8 and 11: never read UnitManager.next_unit_id (a
        side-effecting generator a full commit() does not reliably
        synchronize with the file's own counter). `highest + 1` is a
        defensive floor in case units were added by some other path in the
        same session; `self._next_unit_id` (seeded from the file's own
        next_unit_id_to_place at construction) is what normally dominates.
        """
        highest = _highest_reference_id(self.loaded.unit_manager)
        candidate = max(self._next_unit_id, highest + 1)
        self._next_unit_id = candidate + 1
        return candidate

    # -- operations --------------------------------------------------------
    # Each mutates manager.units[p] and self._blobs[p]/self._tracked[p] in
    # the same step and sets self._dirty. Callers wrap a call to exactly one
    # of these in a begin_unit_edit()/commit_unit_edit() pair -- see those
    # methods' own docstrings.

    def set_position(self, unit: Unit, x: float, y: float, z: float) -> None:
        """Assigns x/y/z and marks the unit's blob dirty. `z` passes through
        verbatim -- never derived from terrain elevation."""
        player, index = self._locate(unit)
        unit.x, unit.y, unit.z = x, y, z
        self._blobs[player][index] = None
        self._dirty = True

    def set_rotation(self, unit: Unit, rotation: float) -> None:
        """Assigns `rotation` (radians) and marks the unit's blob dirty.

        Refuses -- loudly, never a silent no-op -- any const whose
        unit_rotation.semantics_for() is not ANGLE. A silent refusal here
        would leave a caller believing it had rotated a wall, and the whole
        point of the guard is that the module's own hard rule ("never
        transform a non-angle rotation") is enforced rather than merely
        documented.

        The value is passed through as given: wrapping into [0, 2*pi) is
        unit_rotation.rotate_step()'s job, and every caller in the app already
        goes through it. That keeps this method's contract the same as
        set_position()'s -- assign what you were handed, normalize nothing.
        """
        if not unit_rotation.rotation_is_angle(unit.unit_const):
            raise ValueError(
                f"unit_const {unit.unit_const} stores a graphic-variant index in `rotation`, "
                f"not an angle ({unit_rotation.semantics_for(unit.unit_const)}) -- refusing to "
                f"transform it"
            )
        player, index = self._locate(unit)
        unit.rotation = rotation
        self._blobs[player][index] = None
        self._dirty = True

    def set_unit_const(self, unit: Unit, new_const: int) -> None:
        """Swaps a gate's `unit_const` for one of its orientation siblings and
        re-anchors x/y so the footprint keeps the low corner it had.

        The only code anywhere that may change a placed unit's const, and the
        guard below is what keeps AGENTS.md's gate rule enforced rather than
        merely documented: a const that is not one of
        gate_orientation.orientation_siblings()' four is refused loudly, the
        same contract set_rotation() has for a non-ANGLE const.

        **Re-anchoring is not optional.** A gate's four orientations have four
        different spans ((4, 1), (1, 4) and two 4x4 diagonals), and every
        corpus placement sits at `tile + span/2` per axis, so passing x/y
        through verbatim would leave the gate half a footprint off its own
        tiles. span_low_corner() forward and render.span_anchor() back is that
        rule and its exact inverse, which is what makes four steps land on the
        original coordinates rather than drifting.

        `rotation` and `z` pass through verbatim. A gate's stored rotation is
        0.0 or the junk sentinel 7.0 and every sibling has angle_count == 1,
        so normalizing it here would be exactly the violation the hard rule
        names.
        """
        siblings = gate_orientation.orientation_siblings(unit.unit_const)
        if siblings is None or new_const not in siblings:
            raise ValueError(
                f"unit_const {unit.unit_const} cannot become {new_const}: a placed unit's const "
                f"may only change among a gate's own orientation siblings "
                f"({siblings if siblings is not None else 'this const is not a gate'})"
            )
        player, index = self._locate(unit)
        low_x, low_y = span_low_corner(unit)
        new_span = terrain_palette.tile_span(new_const, render.NON_BUILDING_SPAN)
        unit.unit_const = new_const
        unit.x, unit.y = render.span_anchor(low_x, low_y, *new_span)
        self._blobs[player][index] = None
        self._dirty = True

    def reassign(self, unit: Unit, new_player: int) -> None:
        """Moves `unit` to `new_player`'s list. Zero re-serialization: the
        unit's blob moves with it and stays non-None, since UnitStruct
        carries no player field of its own (finding 7) -- reassignment is a
        pure blob-list move.

        Deliberately not `unit.player = ...` (banned -- see this module's
        docstring on _check_alignment) or UnitManager.change_ownership(),
        both of which go through the same setter: it calls
        actions.unit_change_ownership(), which appends to the destination
        list and discards the unit's source position. That matters for
        undo: PlayerListSnapshot captures *both* affected player lists in
        full, so restoring a reassign puts the unit back at its exact
        original index, not merely its original owner. A naive
        remove-then-append inverse would restore ownership but produce a
        different byte layout than the original file.
        """
        if not 0 <= new_player <= 8:
            raise ValueError(f"new_player must be 0 (GAIA)..8, got {new_player}")
        player, index = self._locate(unit)
        if new_player == player:
            return
        blob = self._blobs[player][index]
        del self.loaded.unit_manager.units[player][index]
        del self._blobs[player][index]
        del self._tracked[player][index]
        self.loaded.unit_manager.units[new_player].append(unit)
        self._blobs[new_player].append(blob)
        self._tracked[new_player].append(unit)
        # Resyncs the cached _player that render.py/unit_filter.py read for
        # colour -- reassign never touches it otherwise, since ownership here
        # is purely which list the unit lives in.
        self.loaded.unit_manager.update_unit_player_values()
        self._dirty = True

    def add(
        self,
        player: int,
        unit_const: int,
        x: float,
        y: float,
        z: float = 0.0,
        rotation: float = 0.0,
        status: int = 2,
        initial_animation_frame: int = 0,
        garrisoned_in_id: int = -1,
        caption_string_id: int = -1,
        caption_string: str = "",
    ) -> Unit:
        """Places a new unit for `player`. `rotation` is a pass-through
        parameter, not an edit operation -- storing a caller-supplied value
        on a newly placed unit is fine (this module's hard rule is about
        never *transforming* an existing unit's rotation, which nothing here
        does).

        Deliberately constructs Unit directly rather than calling
        UnitManager.add_unit()/clone_unit(): both use the same unreliable
        id generator this method replaces (see _reserve_reference_id), and
        clone_unit() delegates to add_unit() either way. Support-gating
        caption_string_id/caption_string for scenario versions that don't
        have them is unnecessary here: commit()'s own push_to_link() already
        skips writing a field whose Support range excludes this scenario's
        version, and a version whose UnitStruct doesn't define the retriever
        at all simply has no such key in entry.retriever_map -- see
        _serialize_unit.
        """
        if not 0 <= player <= 8:
            raise ValueError(f"player must be 0 (GAIA)..8, got {player}")
        reference_id = self._reserve_reference_id()
        unit = Unit(
            player=player,
            x=x,
            y=y,
            z=z,
            reference_id=reference_id,
            unit_const=unit_const,
            status=status,
            rotation=rotation,
            initial_animation_frame=initial_animation_frame,
            garrisoned_in_id=garrisoned_in_id,
            caption_string_id=caption_string_id,
            caption_string=caption_string,
            uuid=self.loaded._scenario.uuid,
        )
        self.loaded.unit_manager.units[player].append(unit)
        self._blobs[player].append(None)
        self._tracked[player].append(unit)
        self._dirty = True
        self._has_added_units = True
        return unit

    def add_many(self, player: int, specs: Sequence) -> list[Unit]:
        """Batch counterpart to add(): resolves _reserve_reference_id()'s
        cost (a walk of all nine player lists) once for the whole batch
        instead of once per unit. Used only by descape/terrain_units.py's
        bulk tree/doodad placement, where a large Paint Can fill can add
        thousands of units in one gesture.

        Each spec supplies exactly the fields terrain_units.UnitAddSpec
        carries (x, y, unit_const, rotation, initial_animation_frame); every
        other Unit field takes add()'s own default (z=0.0, status=2,
        garrisoned_in_id=-1, no caption) since nothing in this feature needs
        them to vary.
        """
        if not 0 <= player <= 8:
            raise ValueError(f"player must be 0 (GAIA)..8, got {player}")
        next_id = max(self._next_unit_id, _highest_reference_id(self.loaded.unit_manager) + 1)
        units = []
        for spec in specs:
            units.append(
                Unit(
                    player=player,
                    x=spec.x,
                    y=spec.y,
                    z=0.0,
                    reference_id=next_id,
                    unit_const=spec.unit_const,
                    status=2,
                    rotation=spec.rotation,
                    initial_animation_frame=spec.initial_animation_frame,
                    garrisoned_in_id=-1,
                    caption_string_id=-1,
                    caption_string="",
                    uuid=self.loaded._scenario.uuid,
                )
            )
            next_id += 1
        self.loaded.unit_manager.units[player].extend(units)
        self._blobs[player].extend([None] * len(units))
        self._tracked[player].extend(units)
        self._next_unit_id = next_id
        self._dirty = True
        self._has_added_units = True
        return units

    def remove_many(self, units: Sequence[Unit]) -> None:
        """Batch counterpart to remove(): builds the whole batch's garrison-
        reference set once and rebuilds each touched player's three parallel
        lists in a single filtering pass, instead of remove()'s n x
        (_locate() scan + referencing() scan + del-with-tail-shift).

        Raises the same UnitEditsUnavailableError as remove() if ANY unit in
        the batch is referenced by another unit's garrisoned_in_id, checked
        for the whole batch before anything is removed so a refusal never
        leaves it partially applied.
        """
        if not units:
            return
        to_remove = {id(u) for u in units}
        reference_ids = {u.reference_id for u in units}
        referencing = [
            u
            for player_units in self.loaded.unit_manager.units
            for u in player_units
            if id(u) not in to_remove and u.garrisoned_in_id in reference_ids
        ]
        if referencing:
            raise UnitEditsUnavailableError(
                f"{len(units)} unit(s) in this batch are referenced by garrisoned_in_id on "
                f"{len(referencing)} other unit(s) -- refusing to remove any of them"
            )

        for player, tracked in enumerate(self._tracked):
            keep = [i for i, u in enumerate(tracked) if id(u) not in to_remove]
            if len(keep) == len(tracked):
                continue
            manager_units = self.loaded.unit_manager.units[player]
            blobs = self._blobs[player]
            manager_units[:] = [manager_units[i] for i in keep]
            blobs[:] = [blobs[i] for i in keep]
            tracked[:] = [tracked[i] for i in keep]
        self._dirty = True

    def referencing(self, unit: Unit) -> list[Unit]:
        """Every OTHER unit whose garrisoned_in_id points at `unit`'s
        reference_id -- the dangling-reference guard remove() enforces,
        factored out (phase 3.5b's b2.4) so a bulk delete's pre-flight over
        a whole selection and remove()'s own single-unit guard read the same
        check rather than two copies drifting apart. Trigger effects may
        also reference a unit's id, but Triggers may not even be parsed, so
        that case stays documented as unhandled (plan open question 3), not
        covered here.
        """
        return [
            u
            for units in self.loaded.unit_manager.units
            for u in units
            if u is not unit and u.garrisoned_in_id == unit.reference_id
        ]

    def remove(self, unit: Unit) -> None:
        """Deletes `unit`. Deliberately not UnitManager.remove_unit(), which
        scans all 9 lists and reads unit.player -- this already knows the
        list from _locate()."""
        player, index = self._locate(unit)
        referencing = self.referencing(unit)
        if referencing:
            raise UnitEditsUnavailableError(
                f"Unit reference_id {unit.reference_id} is referenced by garrisoned_in_id "
                f"on {len(referencing)} other unit(s) -- refusing to remove it"
            )
        del self.loaded.unit_manager.units[player][index]
        del self._blobs[player][index]
        del self._tracked[player][index]
        self._dirty = True

    # -- undo/redo support ---------------------------------------------------

    def _capture(self, players: Sequence[int]) -> UnitSnapshot:
        manager = self.loaded.unit_manager
        snapshots = {}
        for p in players:
            snapshots[p] = PlayerListSnapshot(
                units=list(manager.units[p]),
                blobs=list(self._blobs[p]),
                states=[unit_state(u) for u in manager.units[p]],
            )
        return UnitSnapshot(players=snapshots, next_unit_id=self._next_unit_id, dirty=self._dirty)

    def restore(self, snapshot: UnitSnapshot) -> None:
        """Put the document back to `snapshot`. The single place any undo/
        redo writes to the model, mirroring TriggerEditModel.restore()."""
        self._check_alignment()
        manager = self.loaded.unit_manager
        for player, pls in snapshot.players.items():
            # Trap 2 (plan): manager.units[p][:] = ..., never manager.units =
            # [...], whose setter re-wraps in UuidList and pads to 9.
            manager.units[player][:] = list(pls.units)
            self._blobs[player][:] = list(pls.blobs)
            for unit, state in zip(pls.units, pls.states):
                unit.x, unit.y, unit.z, unit.rotation, unit.unit_const = state
            self._tracked[player][:] = list(pls.units)
        # Trap 1 (plan): restore _player by direct assignment or
        # update_unit_player_values(), never the banned `player` property.
        manager.update_unit_player_values()
        self._next_unit_id = snapshot.next_unit_id
        self._dirty = snapshot.dirty

    def begin_unit_edit(self, players: Sequence[int]) -> None:
        """Snapshot before mutating. `players` is the caller's declaration of
        which player list(s) the upcoming edit will touch: one of
        set_position/set_rotation/set_unit_const/add/remove, or both of
        reassign's source and destination. Pairs with exactly one
        commit_unit_edit() or abort_unit_edit().

        Must be called *before* the mutation -- there is no way to recover
        the "before" state afterwards.
        """
        if self._pending is not None:
            raise RuntimeError("begin_unit_edit() called while an edit was already in progress")
        self._pending = self._capture(players)

    def abort_unit_edit(self) -> None:
        """Discards the in-progress snapshot without recording. Does not roll
        the document back -- call restore() for that."""
        self._pending = None

    def commit_unit_edit(self, label: str, history: EditHistory, push: bool = True) -> UnitDiffRecord:
        """Closes the pair opened by begin_unit_edit() and, by default,
        pushes one record onto `history`. Takes the history rather than
        returning the record for the caller to push, so the single-history
        contract (any mutation that sets model dirtiness must push a record)
        cannot be forgotten at a call site.

        `push=False` is for exactly one caller: phase 2.8's region paste,
        which folds this record into a CompositeDiffRecord alongside a tile
        record so one Ctrl+V is one Ctrl+Z. That caller must push the
        composite itself (via push_composite_record()) -- passing push=False
        and then never pushing anything is the single-history contract's
        hole reopened.
        """
        if self._pending is None:
            raise RuntimeError("commit_unit_edit() called with no edit in progress")
        before = self._pending
        self._pending = None
        after = self._capture(list(before.players))
        record = UnitDiffRecord(label, before, after)
        if push:
            history.push_unit_record(record)
        return record

    # -- serialization -------------------------------------------------------

    def _check_alignment(self) -> None:
        """Refuses to serialize/restore if a player's live unit list no
        longer matches what this model tracked -- the same alignment gate
        TriggerEditModel._check_alignment() enforces, and for the same
        reason: a bypassed mutation (e.g. the banned `unit.player =` setter)
        would otherwise splice a real blob into the wrong player's section
        and produce a file that loads fine and is wrong."""
        live = self.loaded.unit_manager.units
        aligned = len(live) == len(self._tracked) and all(
            len(a) == len(b) and all(x is y for x, y in zip(a, b)) for a, b in zip(live, self._tracked)
        )
        if not aligned:
            raise RuntimeError(
                "The unit lists changed behind the model's back (e.g. via the banned "
                "`unit.player = ...` setter, or UnitManager.add_unit/remove_unit/"
                "change_ownership) -- every blob's index mapping is untrustworthy"
            )

    def serialize(self) -> bytes:
        """The whole players_units array, ready to splice into the
        decompressed body at units_block_offset.

        Commits only if at least one blob is dirty -- a reassign-only or
        remove-only edit needs no commit at all, since neither changes any
        unit's own bytes. Counts are always regenerated from len(blobs),
        never spliced or read back from a stale pre-commit retriever.
        """
        self._check_alignment()
        manager = self.loaded.unit_manager
        needs_commit = any(blob is None for blobs in self._blobs for blob in blobs)

        entries_by_player = None
        if needs_commit:
            # A whole-manager-scoped, but link-name-narrowed, commit is what
            # rebuilds each player's unit structs when units were added or
            # removed (fact 11: selecting the link by name is what keeps this
            # from also reading/advancing next_unit_id). Its output for clean
            # units is ignored -- only the dirty ones are read back out of it.
            manager.commit(link_list=[_player_units_link(manager)])
            players_units = self.loaded._scenario.sections["Units"].retriever_map["players_units"].data
            entries_by_player = [pu.retriever_map["units"].data for pu in players_units]
            for player, (entries, blobs) in enumerate(zip(entries_by_player, self._blobs)):
                if len(entries) != len(blobs):
                    raise RuntimeError(
                        f"Player {player}: committed {len(entries)} unit structs but the "
                        f"model tracks {len(blobs)} blobs -- an operation bypassed the "
                        f"model's own API"
                    )

        parts: list[bytes] = []
        for player, blobs in enumerate(self._blobs):
            parts.append(_UNIT_COUNT_STRUCT.pack(len(blobs)))
            for index, blob in enumerate(blobs):
                if blob is not None:
                    parts.append(blob)
                else:
                    parts.append(_serialize_unit(entries_by_player[player][index]))
        return b"".join(parts)
