"""Phase 3.5a's edit model: descape/unit_model.py's UnitEditModel.

The claims under test, in the order the risk runs:

1. **Construction gate.** Every unit in a supported file reproduces its
   original bytes through _serialize_unit(); a corrupted unit refuses
   construction rather than silently editing with a bad normalizer.
2. **Containment.** A model with no edits serializes the section verbatim.
3. **The four operations.** set_position/add/remove/reassign each mutate
   exactly what they claim to and nothing else -- in particular, reassign
   moves a blob without re-serializing it (finding 7: player is positional).
4. **Guards.** The dangling-reference guard on remove, the alignment gate
   against a bypassed mutation, and id bookkeeping (finding 8, fact 11).

See tests/test_units_write_path.py for the write-path integration (byte
locality, write-blocked gates) and tests/test_units_undo.py for undo/redo.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import library_compat, unit_model
from descape.edit_history import EditHistory
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, load_map_and_units_from_bytes
from descape.terrain_units import UnitAddSpec
from descape.unit_model import (
    UnitEditModel,
    UnitEditsUnavailableError,
    UnitFieldSnapshot,
    UnitSnapshot,
    _serialize_unit,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_REF_TREE_OAK = 100
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded)


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _poison_unit_caption_fields() -> None:
    """Reproduces, without a second document, exactly what loading a
    pre-1.55 file does to the Unit class: replaces caption_string_id/
    caption_string with a property whose getter and setter both raise
    UnsupportedAttributeError -- the same shape
    RetrieverObjectLink.overwrite_unsupported_properties() installs. Callers
    must depoison() in teardown or this poisons every later test in the
    process (see library_compat.depoison()'s own docstring)."""

    def _raise(self_, val=None):
        raise UnsupportedAttributeError("synthetic poisoning for a test")

    Unit.caption_string_id = property(_raise, _raise)
    Unit.caption_string = property(_raise, _raise)


# -- 1. construction gate ----------------------------------------------------


def _non_canonical_caption_bytes() -> bytes:
    """The fixture with one empty caption stored as a length-1 lone NUL
    rather than the game's length 0: it loads fine, but the library
    serializer (and so the encoder) writes it back as length 0."""
    from descape.scenario_write import _compress_bytes

    loaded = load_map_and_units(FIXTURE_PATH)
    start, length = loaded.unit_spans[1][0]  # player 1's house, empty caption
    end = start + length
    body = loaded.decompressed_body
    assert body[end - 4 : end] == b"\x00\x00\x00\x00"
    return loaded.header_bytes + _compress_bytes(body[: end - 4] + b"\x01\x00\x00\x00\x00" + body[end:])


@pytest.mark.parametrize("fast_units", [True, False], ids=["encoder-gate", "entry-gate"])
def test_construction_gate_fires_on_a_unit_that_does_not_reproduce(fast_units: bool, monkeypatch) -> None:
    """A gate that never fires is not a gate (plan verification item 9): the
    encoder gate on a fast load, and the entry gate a load whose model
    scenario_io.unit_codec() refuses falls back to."""
    from descape import scenario_io

    if not fast_units:
        monkeypatch.setattr(scenario_io, "unit_codec", lambda loaded: None)
    loaded = load_map_and_units_from_bytes(_non_canonical_caption_bytes(), "odd.aoe2scenario", fast_units=fast_units)
    assert loaded.units_write_supported
    with pytest.raises(UnitEditsUnavailableError, match="reference_id 200"):
        UnitEditModel(loaded)


def test_a_unit_changed_before_the_model_exists_still_constructs() -> None:
    """The gate checks the file's bytes, not the live Units: a caller that
    changed a Unit before the lazy first-edit construction (as viewer tests
    do) still gets a model."""
    loaded = load_map_and_units(FIXTURE_PATH)
    _unit(loaded, _REF_HOUSE).rotation += 1.0
    UnitEditModel(loaded)


def test_entry_gate_fires_on_a_corrupted_entry_when_there_is_no_codec(monkeypatch) -> None:
    """A structure unit_codec() refuses loads through the library walk, and the
    model falls back to the entry gate, which must still fire."""
    from descape import scenario_io

    monkeypatch.setattr(scenario_io, "unit_codec", lambda loaded: None)
    loaded = load_map_and_units(FIXTURE_PATH, fast_units=False)
    UnitEditModel(loaded)  # the clean file passes the entry gate
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    entry = players_units[1].retriever_map["units"].data[0]  # player 1's house
    entry.retriever_map["rotation"].set_data(entry.retriever_map["rotation"].data + 1.0)

    with pytest.raises(UnitEditsUnavailableError):
        UnitEditModel(loaded)


def test_a_clean_file_constructs_without_raising() -> None:
    _open()  # must not raise


def test_the_zero_unit_template_constructs_with_no_edits() -> None:
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = UnitEditModel(loaded)
    assert not model.has_edits


def test_a_file_that_fails_units_write_supported_refuses_a_model() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    loaded.units_write_supported = False
    with pytest.raises(UnitEditsUnavailableError):
        UnitEditModel(loaded)


def test_a_file_with_an_unfamiliar_section_count_refuses_a_model() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    loaded.number_of_unit_sections = 8
    with pytest.raises(UnitEditsUnavailableError):
        UnitEditModel(loaded)


# -- 2. containment -----------------------------------------------------------


def test_clean_model_serializes_the_original_section_verbatim() -> None:
    loaded, model = _open()
    assert not model.has_edits
    section = loaded.decompressed_body[loaded.units_block_offset : loaded.units_section_end]
    assert model.serialize() == section


def test_the_non_empty_caption_is_left_untouched_by_the_normalizer() -> None:
    """The one measured falsification of the plan's original "strip
    unconditionally" design (see _serialize_unit's own docstring): a
    non-empty caption's library form matches its raw on-disk bytes exactly,
    with no stripping."""
    loaded = load_map_and_units(FIXTURE_PATH, fast_units=False)
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    entry = next(
        e
        for pu in players_units
        for e in pu.retriever_map["units"].data
        if e.retriever_map["reference_id"].data == _REF_ARCHER_P2
    )
    assert entry.retriever_map["caption_string"].data == "Fixture caption"
    assert _serialize_unit(entry) == entry.get_data_as_bytes()


# -- 3. operations -------------------------------------------------------------


def test_set_position_dirties_only_the_moved_unit() -> None:
    loaded, model = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.set_position(villager, 50.5, 51.5, 2.0)
    assert (villager.x, villager.y, villager.z) == (50.5, 51.5, 2.0)
    assert model.has_edits
    dirty = [blob is None for blobs in model._blobs for blob in blobs]
    assert sum(dirty) == 1


def test_set_rotation_dirties_only_the_rotated_unit() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    model.set_rotation(archer, 1.5)
    assert archer.rotation == 1.5
    assert model.has_edits
    dirty = [blob is None for blobs in model._blobs for blob in blobs]
    assert sum(dirty) == 1


@pytest.mark.parametrize("reference_id,unit_const", [(_REF_WALL, None), (_REF_ARCHER_P1, 64)])
def test_set_rotation_refuses_a_non_angle_const(reference_id, unit_const) -> None:
    """A wall (shape variants) and a gate (angle_count 1, orientation lives in
    the const) each raise rather than silently no-op'ing -- and the section
    stays byte-identical, so a refused rotate leaves nothing behind.

    The gate case swaps a const onto a real fixture unit rather than needing a
    gate in the fixture: the guard reads unit.unit_const, and a field edit
    doesn't reach the bytes without a commit (finding 4).
    """
    loaded, model = _open()
    before = model.serialize()
    unit = _unit(loaded, reference_id)
    if unit_const is not None:
        unit.unit_const = unit_const
    with pytest.raises(ValueError):
        model.set_rotation(unit, 1.0)
    assert not model.has_edits
    assert model.serialize() == before


def test_set_variant_dirties_only_the_cycled_tree() -> None:
    loaded, model = _open()
    oak = _unit(loaded, _REF_TREE_OAK)
    model.set_variant(oak, 8.0)
    assert oak.rotation == 8.0
    assert model.has_edits
    dirty = [blob is None for blobs in model._blobs for blob in blobs]
    assert sum(dirty) == 1


@pytest.mark.parametrize("reference_id", [_REF_WALL, _REF_ARCHER_P1, _REF_HOUSE])
def test_set_variant_refuses_a_non_cyclable_const(reference_id) -> None:
    """A wall (neighbour-derived variant), an ANGLE archer and an INERT house."""
    loaded, model = _open()
    before = model.serialize()
    with pytest.raises(ValueError):
        model.set_variant(_unit(loaded, reference_id), 1.0)
    assert not model.has_edits
    assert model.serialize() == before


# -- set_unit_const: the gate orientation cycle ------------------------------
#
# A stone closed gate's four siblings, in cycle order, with the span each one
# occupies: 64 ne (4, 1), 659 e (4, 4 sparse), 88 se (1, 4), 667 n (4, 4
# sparse). Anchored on low corner (8, 5) throughout, which is the low corner
# of a gate placed at (10.0, 5.5): a span-4 axis anchors two tiles below the
# unit's own tile, not at int(x).
_GATE_NE, _GATE_E, _GATE_SE, _GATE_N = 64, 659, 88, 667
_GATE_ANCHORS = {
    _GATE_NE: (10.0, 5.5),
    _GATE_E: (10.0, 7.0),
    _GATE_SE: (8.5, 7.0),
    _GATE_N: (10.0, 7.0),
}


def _gate(loaded, unit_const: int = _GATE_NE) -> object:
    """A stone gate, made by re-pointing the fixture's wall at a gate const
    and its own anchor. Same trick as test_set_rotation_refuses_a_non_angle
    _const's gate case, since the fixture holds no gate and every guard here
    reads the live unit's fields."""
    unit = _unit(loaded, _REF_WALL)
    unit.unit_const = unit_const
    unit.x, unit.y = _GATE_ANCHORS[unit_const]
    unit.z, unit.rotation = 3.0, 7.0
    return unit


def test_set_unit_const_re_anchors_the_gate_onto_its_preserved_low_corner() -> None:
    """The measured parity table: ne sits at fractional (0.0, 0.5), se at
    (0.5, 0.0), and both diagonals at (0.0, 0.0). Preserving the low corner is
    what produces all three; passing x/y through verbatim would leave the
    gate half a footprint off its own tiles."""
    loaded, model = _open()
    gate = _gate(loaded)
    model.set_unit_const(gate, _GATE_E)
    assert (gate.x, gate.y) == (10.0, 7.0)
    model.set_unit_const(gate, _GATE_SE)
    assert (gate.x, gate.y) == (8.5, 7.0)
    assert model.has_edits


def test_four_cycle_steps_return_the_exact_original_const_and_position() -> None:
    loaded, model = _open()
    gate = _gate(loaded)
    original = (gate.unit_const, gate.x, gate.y)
    for const in (_GATE_E, _GATE_SE, _GATE_N, _GATE_NE):
        model.set_unit_const(gate, const)
    assert (gate.unit_const, gate.x, gate.y) == original


def test_a_cycle_passes_rotation_and_z_through_verbatim() -> None:
    """A gate's stored rotation is 0.0 or the junk sentinel 7.0, and every
    sibling has angle_count == 1, so normalizing it here would be AGENTS.md's
    verbatim violation, not a tidy-up."""
    loaded, model = _open()
    gate = _gate(loaded)
    model.set_unit_const(gate, _GATE_E)
    assert gate.rotation == 7.0
    assert gate.z == 3.0


def test_set_unit_const_dirties_only_the_cycled_unit() -> None:
    loaded, model = _open()
    gate = _gate(loaded)
    model.set_unit_const(gate, _GATE_E)
    dirty = [blob is None for blobs in model._blobs for blob in blobs]
    assert sum(dirty) == 1


@pytest.mark.parametrize(
    "new_const",
    [
        789,  # a palisade gate: right orientation, wrong family
        4,  # not a gate at all
        1192,  # the class-39 const whose code collides, deliberately grouped nowhere
    ],
)
def test_set_unit_const_refuses_anything_but_an_orientation_sibling(new_const) -> None:
    loaded, model = _open()
    before = model.serialize()
    gate = _gate(loaded)
    with pytest.raises(ValueError):
        model.set_unit_const(gate, new_const)
    assert not model.has_edits
    assert model.serialize() == before


def test_set_unit_const_refuses_a_unit_that_is_not_a_gate() -> None:
    """The guard is what enforces AGENTS.md's rule that a placed unit's const
    is otherwise never changed, so a wall must raise rather than no-op."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    with pytest.raises(ValueError):
        model.set_unit_const(wall, _GATE_NE)
    assert not model.has_edits


def test_reassign_moves_the_unit_and_marks_no_blob_dirty() -> None:
    """finding 7: player is positional, so reassignment is a pure blob-list
    move with zero re-serialization -- plan verification item 7."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    assert wall in loaded.unit_manager.units[0]

    model.reassign(wall, 1)

    assert wall not in loaded.unit_manager.units[0]
    assert wall in loaded.unit_manager.units[1]
    assert wall._player == 1, "update_unit_player_values() must resync the cached _player"
    assert model.has_edits
    assert all(blob is not None for blobs in model._blobs for blob in blobs), (
        "reassign must not dirty any blob"
    )


def test_reassign_to_the_same_player_is_a_no_op() -> None:
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    model.reassign(wall, 0)
    assert not model.has_edits


def test_reassign_rejects_an_out_of_range_player() -> None:
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    with pytest.raises(ValueError):
        model.reassign(wall, 9)


def test_reassign_serializes_with_no_per_unit_reserialization() -> None:
    """The provenance test the plan calls for (verification item 7): if
    reassign's blob truly never re-enters the library, serialize() must
    succeed even when every entry's own get_data_as_bytes() would raise. On a
    fast_units=False load, so there are parsed entries to booby-trap."""
    loaded = load_map_and_units(FIXTURE_PATH, fast_units=False)
    model = UnitEditModel(loaded)
    wall = _unit(loaded, _REF_WALL)
    model.reassign(wall, 1)

    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    for pu in players_units:
        for entry in pu.retriever_map["units"].data:
            entry.get_data_as_bytes = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a clean unit's blob must never re-enter the library")
            )

    model.serialize()  # must not raise


def test_reassign_on_a_fast_load_serializes_without_a_commit() -> None:
    """The fast-load form of the provenance test: a reassign-only serialize()
    never commits, so the empty unit slots the fast walk left stay empty."""
    loaded, model = _open()
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    assert all(pu.retriever_map["units"].data == [] for pu in players_units)
    model.reassign(_unit(loaded, _REF_WALL), 1)

    section = model.serialize()
    assert all(pu.retriever_map["units"].data == [] for pu in players_units)
    assert len(section) == loaded.players_units_end - loaded.units_block_offset


def test_add_places_a_new_unit_with_a_reserved_reference_id() -> None:
    loaded, model = _open()
    before_next = model.next_unit_id
    unit = model.add(player=2, unit_const=83, x=15.5, y=15.5, z=0.0, rotation=1.5)

    assert unit.reference_id == before_next
    assert unit in loaded.unit_manager.units[2]
    assert model.has_added_units
    assert model.next_unit_id == before_next + 1
    assert model.has_edits


def test_add_reserves_ids_above_every_existing_reference_id() -> None:
    loaded, model = _open()
    highest = max(u.reference_id for u in loaded.unit_manager.get_all_units())
    unit = model.add(player=0, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)
    assert unit.reference_id > highest


def test_add_rotation_is_a_verbatim_pass_through() -> None:
    """AGENTS.md's hard rule: rotation is a variant index for many GAIA
    objects, not an angle. add() must never validate or normalize it."""
    _loaded, model = _open()
    unit = model.add(player=0, unit_const=349, x=1.5, y=1.5, z=0.0, rotation=41.0)
    assert unit.rotation == 41.0


def test_add_rejects_an_out_of_range_player() -> None:
    _loaded, model = _open()
    with pytest.raises(ValueError):
        model.add(player=9, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)


def test_add_depoisons_before_constructing_unit() -> None:
    """Plan 2026-09-12: Unit.__init__ unconditionally assigns
    caption_string_id/caption_string, so a class left poisoned by an earlier
    document's load raises inside add()'s own Unit(...) call, before
    commit-time version gating ever gets a chance to matter. Regression for
    the library_compat.depoison() call at the top of add(). The fixture is
    1.58, so the real value must land after the fix, not a None sentinel --
    that's what discriminates this fix from the rejected None-sentinel
    variant."""
    _loaded, model = _open()
    _poison_unit_caption_fields()
    try:
        unit = model.add(player=0, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)
        assert unit.caption_string_id == -1
        assert unit.caption_string == ""
    finally:
        library_compat.depoison()


# -- batch operations (descape/terrain_units.py's own callers) --------------


def test_add_many_places_units_with_contiguous_reference_ids() -> None:
    loaded, model = _open()
    before_next = model.next_unit_id
    specs = [
        UnitAddSpec(x=1.5, y=1.5, unit_const=349, rotation=3.0, initial_animation_frame=3),
        UnitAddSpec(x=2.5, y=2.5, unit_const=350, rotation=5.0, initial_animation_frame=5),
    ]
    units = model.add_many(player=0, specs=specs)

    assert [u.reference_id for u in units] == [before_next, before_next + 1]
    for unit in units:
        assert unit in loaded.unit_manager.units[0]
    assert model.has_added_units
    assert model.next_unit_id == before_next + 2
    assert model.has_edits


def test_add_many_rotation_and_frame_pass_through_verbatim() -> None:
    """Same hard rule as add()'s own verbatim-pass-through test: rotation is
    a variant index here, never validated or normalized."""
    _loaded, model = _open()
    spec = UnitAddSpec(x=1.5, y=1.5, unit_const=349, rotation=41.0, initial_animation_frame=41)
    unit = model.add_many(player=0, specs=[spec])[0]
    assert unit.rotation == 41.0
    assert unit.initial_animation_frame == 41


def test_add_many_matches_add_for_equivalent_placements() -> None:
    """The batch path must place the same shape of unit add() does -- this
    guards against add_many() drifting onto a different set of Unit
    defaults (z, status, garrisoned_in_id, caption) than add()'s own."""
    _loaded_a, model_a = _open()
    single = model_a.add(player=1, unit_const=349, x=3.5, y=3.5, z=0.0, rotation=2.0, initial_animation_frame=2)

    _loaded_b, model_b = _open()
    spec = UnitAddSpec(x=3.5, y=3.5, unit_const=349, rotation=2.0, initial_animation_frame=2)
    batched = model_b.add_many(player=1, specs=[spec])[0]

    assert (batched.x, batched.y, batched.z) == (single.x, single.y, single.z)
    assert (batched.unit_const, batched.status) == (single.unit_const, single.status)
    assert (batched.rotation, batched.initial_animation_frame) == (single.rotation, single.initial_animation_frame)
    assert batched.garrisoned_in_id == single.garrisoned_in_id


def test_add_many_rejects_an_out_of_range_player() -> None:
    _loaded, model = _open()
    spec = UnitAddSpec(x=1.5, y=1.5, unit_const=83, rotation=0.0, initial_animation_frame=0)
    with pytest.raises(ValueError):
        model.add_many(player=9, specs=[spec])


def test_add_many_depoisons_before_constructing_units() -> None:
    """Same regression as test_add_depoisons_before_constructing_unit: the
    identical caption-field assignment sits in add_many()'s own Unit(...)
    construction (unit_model.py's Paint Can-only batch path)."""
    _loaded, model = _open()
    _poison_unit_caption_fields()
    try:
        spec = UnitAddSpec(x=1.5, y=1.5, unit_const=83, rotation=0.0, initial_animation_frame=0)
        unit = model.add_many(player=0, specs=[spec])[0]
        assert unit.caption_string_id == -1
        assert unit.caption_string == ""
    finally:
        library_compat.depoison()


def _count_depoisons(monkeypatch) -> list:
    calls = []
    real = library_compat.depoison
    monkeypatch.setattr(library_compat, "depoison", lambda: (calls.append(1), real())[1])
    return calls


def test_batch_adds_depoisons_once_for_a_run_of_adds(monkeypatch) -> None:
    """Paste Region's ~700 add() calls each walked five classes; inside
    batch_adds() the walk runs once, and add() outside it still pays its own."""
    _loaded, model = _open()
    calls = _count_depoisons(monkeypatch)
    with model.batch_adds():
        for i in range(5):
            model.add(player=1, unit_const=83, x=1.5 + i, y=1.5)
    assert len(calls) == 1
    model.add(player=1, unit_const=83, x=9.5, y=1.5)
    assert len(calls) == 2, "add() after the block must depoison again"
    model._check_alignment()


def test_batch_adds_depoisons_a_poisoned_class_before_the_first_add() -> None:
    """The same regression as test_add_depoisons_before_constructing_unit,
    through the batch: the block's own depoison() must run before any Unit(...)."""
    _loaded, model = _open()
    _poison_unit_caption_fields()
    try:
        with model.batch_adds():
            unit = model.add(player=0, unit_const=83, x=1.5, y=1.5)
        assert unit.caption_string_id == -1
        assert unit.caption_string == ""
    finally:
        library_compat.depoison()


def test_add_depoisons_again_after_a_batch_that_raised() -> None:
    _loaded, model = _open()
    with pytest.raises(ValueError), model.batch_adds():
        model.add(player=9, unit_const=83, x=1.5, y=1.5)
    _poison_unit_caption_fields()
    try:
        unit = model.add(player=0, unit_const=83, x=1.5, y=1.5)
        assert unit.caption_string == ""
    finally:
        library_compat.depoison()


def test_batch_adds_is_refused_inside_a_fields_only_edit() -> None:
    _loaded, model = _open()
    model.begin_unit_edit([1], fields_only=True)
    with pytest.raises(RuntimeError, match="fields_only"), model.batch_adds():
        pass
    model.abort_unit_edit()


def test_serialize_depoisons_before_encoding_a_dirty_unit() -> None:
    """Plan 2026-09-12's save-side half: a poisoned class's property getter
    is a data descriptor that shadows the real instance attribute, so both
    the encoder's getattr and the commit's push_to_link raised serializing
    *any* dirty unit, not just a newly added one. Regression for the
    depoison() at the top of each writer; tests/test_fast_units.py has the
    real older-document form (corpus)."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    model.set_position(wall, 10.5, 10.5, 0.0)
    try:
        _poison_unit_caption_fields()
        encoded = model.serialize()  # must not raise
        _poison_unit_caption_fields()
        assert encoded == model._serialize_via_commit()
    finally:
        library_compat.depoison()


@pytest.mark.parametrize("reference_id", [101, _REF_WALL], ids=["doodad-variant-41", "radian-wall"])
def test_a_moved_unit_saves_its_rotation_bytes_verbatim(reference_id: int, tmp_path: Path) -> None:
    """AGENTS.md's verbatim rule on the encoder: a GAIA doodad whose rotation
    is variant index 41.0, and a wall storing its index as 2*2pi/5, moved and
    saved, keep the rotation field's four bytes exactly."""
    import struct

    from descape import scenario_io
    from descape.scenario_write import write_scenario

    loaded, model = _open()
    unit = _unit(loaded, reference_id)
    player, index = next((p, i) for p, units in enumerate(loaded.unit_manager.units) for i, u in enumerate(units) if u is unit)
    fmt = scenario_io.unit_codec(loaded).unit_format
    offset = struct.calcsize(fmt.fixed.format[: 1 + fmt.fields.index("rotation")])
    start, _ = loaded.unit_spans[player][index]
    before = loaded.decompressed_body[start + offset : start + offset + 4]

    model.set_position(unit, unit.x + 1.0, unit.y, unit.z)
    out = tmp_path / "moved.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=model)
    saved = load_map_and_units(out)
    start, _ = saved.unit_spans[player][index]
    assert saved.decompressed_body[start + offset : start + offset + 4] == before
    assert saved.unit_manager.units[player][index].x == unit.x


def test_remove_many_deletes_every_unit() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    villager = _unit(loaded, _REF_VILLAGER_P2)
    model.remove_many([archer, villager])
    assert archer not in loaded.unit_manager.units[1]
    assert villager not in loaded.unit_manager.units[2]
    assert model.has_edits


def test_remove_many_of_an_empty_list_is_a_no_op() -> None:
    _loaded, model = _open()
    model.remove_many([])
    assert not model.has_edits


def test_remove_many_refuses_the_whole_batch_if_any_unit_is_referenced() -> None:
    """Same dangling-reference guard as remove(), checked for the whole
    batch up front -- a refusal must leave every unit in the batch alone,
    including the ones that were individually fine to remove."""
    loaded, model = _open()
    house = _unit(loaded, _REF_HOUSE)
    archer = _unit(loaded, _REF_ARCHER_P1)
    with pytest.raises(UnitEditsUnavailableError):
        model.remove_many([archer, house])
    assert archer in loaded.unit_manager.units[1]
    assert house in loaded.unit_manager.units[1]


def test_remove_many_matches_remove_for_the_same_units() -> None:
    loaded_a, model_a = _open()
    tree_a = _unit(loaded_a, _REF_TREE_OAK)
    model_a.remove(tree_a)

    loaded_b, model_b = _open()
    tree_b = _unit(loaded_b, _REF_TREE_OAK)
    model_b.remove_many([tree_b])

    assert tree_a not in loaded_a.unit_manager.units[0]
    assert tree_b not in loaded_b.unit_manager.units[0]


def _mixed_batch(loaded, model) -> list:
    """Units from three players, listed out of index order: loaded ones (real
    blobs, each list's first unit among them) and appended ones (no blob)."""
    extra = model.add_many(0, [UnitAddSpec(x=5.5 + i, y=5.5, unit_const=349, rotation=0.0, initial_animation_frame=0) for i in range(6)])
    return [
        extra[3], _unit(loaded, _REF_ARCHER_P1), extra[0], _unit(loaded, _REF_ARCHER_P2),
        extra[5], _unit(loaded, _REF_TREE_OAK), _unit(loaded, _REF_VILLAGER_P2),
    ]


def _model_state(loaded, model) -> tuple:
    model._check_alignment()
    lists = [[u.reference_id for u in units] for units in loaded.unit_manager.units]
    return lists, [list(blobs) for blobs in model._blobs], model.serialize()


def test_remove_many_by_index_matches_the_whole_list_rewrite(monkeypatch) -> None:
    """The del-by-index path (a small batch) leaves the three parallel lists,
    _pos and the serialized section exactly as the whole-list rewrite does."""
    loaded_a, model_a = _open()
    model_a.remove_many(_mixed_batch(loaded_a, model_a))

    loaded_b, model_b = _open()
    monkeypatch.setattr(unit_model, "_REMOVE_BY_INDEX_MAX", 0)
    model_b.remove_many(_mixed_batch(loaded_b, model_b))

    assert _model_state(loaded_a, model_a) == _model_state(loaded_b, model_b)
    assert [u.reference_id for u in loaded_a.unit_manager.units[2]] == []


def test_a_small_remove_many_never_rewrites_a_whole_list(monkeypatch) -> None:
    """Non-vacuity for the fast path: a Draw-sized batch must not go through
    UuidList's whole-list `[:] =`, which re-wraps every unit in the list."""
    loaded, model = _open()
    batch = _mixed_batch(loaded, model)
    rewrites = []
    uuid_list = type(loaded.unit_manager.units[0])
    real_setitem = uuid_list.__setitem__
    monkeypatch.setattr(uuid_list, "__setitem__", lambda self, i, o: (rewrites.append(i), real_setitem(self, i, o))[1])

    model.remove_many(batch)
    assert rewrites == [], "the small batch rewrote a whole unit list"

    monkeypatch.setattr(unit_model, "_REMOVE_BY_INDEX_MAX", 1)
    model.remove_many([_unit(loaded, _REF_WALL), _unit(loaded, _REF_VILLAGER_P1)])
    assert rewrites, "a batch past _REMOVE_BY_INDEX_MAX should take the whole-list rewrite"
    model._check_alignment()


def test_remove_many_skips_an_untracked_unit_as_before() -> None:
    """A unit the model doesn't track sends the batch down the rewrite path,
    which removes the tracked ones and leaves the rest alone."""
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    stray = copy.copy(_unit(loaded, _REF_ARCHER_P2))
    model.remove_many([archer, stray])
    assert archer not in loaded.unit_manager.units[1]
    assert [u.reference_id for u in loaded.unit_manager.units[2]] == [_REF_ARCHER_P2, _REF_VILLAGER_P2]
    model._check_alignment()


def test_remove_deletes_the_unit() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    model.remove(archer)
    assert archer not in loaded.unit_manager.units[2]
    assert model.has_edits


def test_remove_refuses_a_unit_referenced_by_garrisoned_in_id() -> None:
    """The dangling-reference guard: the villager (203) is garrisoned in the
    house (200), so removing the house must be refused rather than leaving a
    dangling reference."""
    loaded, model = _open()
    house = _unit(loaded, _REF_HOUSE)
    with pytest.raises(UnitEditsUnavailableError):
        model.remove(house)
    assert house in loaded.unit_manager.units[1], "a refused remove must not have removed anything"


def test_remove_of_the_referencing_unit_itself_is_unaffected() -> None:
    """Removing the villager (the one *holding* the garrison reference, not
    the one referenced) must not trip the guard."""
    loaded, model = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model.remove(villager)
    assert villager not in loaded.unit_manager.units[1]


def test_an_operation_on_an_untracked_unit_raises() -> None:
    """A Unit object this model never tracked (e.g. constructed by hand,
    bypassing add()) must not be silently accepted."""
    loaded, model = _open()
    # new_unit() depoisons first: on this pre-1.59 file a bare Unit(...) raises
    # on its own capture_flag default.
    stray = unit_model.new_unit(loaded._scenario.uuid, 0, 4, 0.0, 0.0, 99999)
    with pytest.raises(ValueError):
        model.set_position(stray, 1.0, 1.0, 0.0)


# -- Batch D's D1a: unit_gen counter -------------------------------------------


def _mutate_set_position(loaded, model) -> None:
    model.set_position(_unit(loaded, _REF_VILLAGER_P1), 50.5, 51.5, 2.0)


def _mutate_set_rotation(loaded, model) -> None:
    model.set_rotation(_unit(loaded, _REF_ARCHER_P1), 1.5)


def _mutate_set_variant(loaded, model) -> None:
    model.set_variant(_unit(loaded, _REF_TREE_OAK), 8.0)


def _mutate_set_unit_const(loaded, model) -> None:
    model.set_unit_const(_gate(loaded), _GATE_E)


def _mutate_reassign(loaded, model) -> None:
    model.reassign(_unit(loaded, _REF_WALL), 1)


def _mutate_add(loaded, model) -> None:
    model.add(player=0, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)


def _mutate_add_many(loaded, model) -> None:
    spec = UnitAddSpec(x=1.5, y=1.5, unit_const=349, rotation=0.0, initial_animation_frame=0)
    model.add_many(player=0, specs=[spec])


def _mutate_remove_many(loaded, model) -> None:
    model.remove_many([_unit(loaded, _REF_TREE_OAK)])


def _mutate_remove(loaded, model) -> None:
    model.remove(_unit(loaded, _REF_TREE_OAK))


def _mutate_set_garrisoned_in(loaded, model) -> None:
    model.set_garrisoned_in(_unit(loaded, _REF_VILLAGER_P1), -1)


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_set_position,
        _mutate_set_rotation,
        _mutate_set_variant,
        _mutate_set_unit_const,
        _mutate_reassign,
        _mutate_add,
        _mutate_add_many,
        _mutate_remove_many,
        _mutate_remove,
        _mutate_set_garrisoned_in,
    ],
    ids=[
        "set_position",
        "set_rotation",
        "set_variant",
        "set_unit_const",
        "reassign",
        "add",
        "add_many",
        "remove_many",
        "remove",
        "set_garrisoned_in",
    ],
)
def test_every_public_mutator_bumps_unit_gen(mutate) -> None:
    loaded, model = _open()
    gen0 = loaded.unit_gen
    mutate(loaded, model)
    assert loaded.unit_gen == gen0 + 1


def test_restore_bumps_unit_gen() -> None:
    """restore() is the ninth site (unit_model.py's own D1a list): undo/redo
    can move x/y/z/rotation/unit_const without going through any of the
    other eight, so a memo built before an undo must not survive it either."""
    loaded, model = _open()
    snapshot = model._capture([0])
    gen0 = loaded.unit_gen
    model.restore(snapshot)
    assert loaded.unit_gen == gen0 + 1


def test_a_direct_list_append_does_not_bump_unit_gen() -> None:
    """The counter's documented contract (LoadedScenario.unit_gen's own
    docstring): only UnitEditModel mutations bump it. A test fixture (or any
    other code) that appends straight to unit_manager.units must bump
    scenario.unit_gen itself if it wants a memo built afterward to be
    invalidated."""
    loaded, _model = _open()
    gen0 = loaded.unit_gen
    loaded.unit_manager.units[0].append(unit_model.new_unit(loaded._scenario.uuid, 0, 4, 0.0, 0.0, 88888))
    assert loaded.unit_gen == gen0


# -- 4. alignment guard --------------------------------------------------------


def test_check_alignment_catches_a_bypassed_mutation() -> None:
    """A unit moved by the banned `unit.player = ...` setter (or any direct
    manager.units mutation bypassing this model) must surface as a loud
    error, not a silently mis-spliced save."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    loaded.unit_manager.units[0].remove(wall)
    loaded.unit_manager.units[1].append(wall)

    with pytest.raises(RuntimeError):
        model.serialize()


# -- 5. the derived structures (_pos, _garrison, _highest_ref_id) -------------


def test_check_alignment_catches_a_stale_position_entry() -> None:
    """The identity check inside _locate() cannot see a stale index that
    still resolves to a real unit. The alignment gate is what does, and
    without the _pos invariant this mutation would serialize silently."""
    loaded, model = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    model._pos[id(villager)] = (1, 0)  # the house's slot

    with pytest.raises(RuntimeError):
        model.serialize()


def test_check_alignment_catches_a_leaked_position_entry() -> None:
    """The id() reuse hazard: an entry left behind for a removed unit."""
    _loaded, model = _open()
    stray = object()
    model._pos[id(stray)] = (0, 0)

    with pytest.raises(RuntimeError):
        model.serialize()


def test_locate_self_heals_from_a_stale_position_entry() -> None:
    """A miss or a mismatch costs one full walk and a retry, then lands on
    the right unit. Both halves are exercised: the eager reindex a mid-list
    remove does, and the rebuild a deliberately poisoned entry forces."""
    loaded, model = _open()
    house = _unit(loaded, _REF_HOUSE)
    archer = _unit(loaded, _REF_ARCHER_P1)
    villager = _unit(loaded, _REF_VILLAGER_P1)
    house_pos = (house.x, house.y, house.z)

    model.remove(archer)  # player 1's middle entry: the villager shifts to 1
    assert model._pos[id(villager)] == (1, 1)

    model._pos[id(villager)] = (1, 0)  # now the house's slot
    model.set_position(villager, 60.5, 61.5, 2.0)

    assert (villager.x, villager.y, villager.z) == (60.5, 61.5, 2.0)
    assert (house.x, house.y, house.z) == house_pos, "the write must not have landed on the house"
    assert model._pos[id(villager)] == (1, 1)


def test_referencing_matches_the_flat_scan_it_replaces() -> None:
    """Parity for the reverse-map: same units, same order, for every unit in
    the file, including a unit garrisoned in itself, which the self-exclusion
    must keep out of its own answer."""
    loaded, model = _open()
    self_ref = _unit(loaded, _REF_ARCHER_P2)
    self_ref.garrisoned_in_id = self_ref.reference_id

    def flat_scan(unit):
        return [
            u
            for units in loaded.unit_manager.units
            for u in units
            if u is not unit and u.garrisoned_in_id == unit.reference_id
        ]

    all_units = list(loaded.unit_manager.get_all_units())
    assert any(flat_scan(u) for u in all_units), "a parity check over an all-empty answer proves nothing"
    for unit in all_units:
        expected = flat_scan(unit)
        actual = model.referencing(unit)
        assert len(actual) == len(expected)
        assert all(a is b for a, b in zip(actual, expected, strict=True))

    assert model.referencing(self_ref) == []
    assert [u.reference_id for u in model.referencing(_unit(loaded, _REF_HOUSE))] == [_REF_VILLAGER_P1]


def test_a_warm_garrison_map_survives_adds_and_removes() -> None:
    """The reverse-map is spliced rather than dropped, so it stays warm
    across a whole group-delete loop. serialize()'s alignment gate is what
    proves each splice still matches a fresh walk."""
    loaded, model = _open()
    model.warm_garrison_map()

    added = model.add(player=0, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)
    spec = UnitAddSpec(x=2.5, y=2.5, unit_const=349, rotation=0.0, initial_animation_frame=0)
    batched = model.add_many(player=1, specs=[spec])[0]
    model.remove(_unit(loaded, _REF_TREE_OAK))
    model.remove_many([_unit(loaded, _REF_ARCHER_P2)])
    model.reassign(added, 2)

    model.serialize()  # the alignment gate must accept every splice above
    assert model.referencing(added) == []
    assert model.referencing(batched) == []


def test_removing_a_garrison_holder_keeps_the_map_answering() -> None:
    """Removing the unit that HOLDS a reference must drop it from its
    holder's bucket, or the house would stay undeletable forever."""
    loaded, model = _open()
    house = _unit(loaded, _REF_HOUSE)
    villager = _unit(loaded, _REF_VILLAGER_P1)
    assert model.referencing(house)  # warm, and non-empty

    model.remove(villager)

    assert model.referencing(house) == []
    model.remove(house)  # no longer refused
    assert house not in loaded.unit_manager.units[1]


def test_check_alignment_catches_a_drifted_highest_reference_id() -> None:
    """A cache claiming to be fresh while sitting above (or below) the real
    maximum would hand the next add() an id a full rescan never would."""
    _loaded, model = _open()
    model._highest_ref_id += 5
    model._highest_ref_id_stale = False

    with pytest.raises(RuntimeError):
        model.serialize()


def test_a_remove_marks_the_highest_reference_id_cache_stale() -> None:
    """remove() lowers the true maximum, so a monotone cache would be wrong
    from that point on. The flag is what forces the next add() to rescan."""
    loaded, model = _open()
    assert model._highest_ref_id == _REF_VILLAGER_P2

    model.remove(_unit(loaded, _REF_VILLAGER_P2))

    assert model._highest_ref_id_stale
    assert model._highest_ref_id_now() == _REF_ARCHER_P2
    model.serialize()  # invariant 4 would fire here if the flag were missed


# -- Batch D's D6: delta snapshots for set_* ops -----------------------------


def _units_section(loaded) -> bytes:
    return loaded.decompressed_body[loaded.units_block_offset : loaded.units_section_end]


def test_set_position_delta_round_trip_is_byte_clean() -> None:
    loaded, model = _open()
    history = EditHistory()
    original = _units_section(loaded)
    villager = _unit(loaded, _REF_VILLAGER_P1)

    model.begin_unit_edit([1], fields_only=True)
    model.set_position(villager, 40.5, 40.5, 5.0)
    model.commit_unit_edit("Move villager", history)
    assert model.has_edits

    history.undo([], None, None, model)
    assert not model.has_edits, "undo must restore blob cleanliness, not just content"
    assert model.serialize() == original

    history.redo([], None, None, model)
    assert model.has_edits
    assert (villager.x, villager.y, villager.z) == (40.5, 40.5, 5.0)


def test_set_rotation_delta_round_trip_is_byte_clean() -> None:
    loaded, model = _open()
    history = EditHistory()
    original = _units_section(loaded)
    archer = _unit(loaded, _REF_ARCHER_P1)
    before_rotation = archer.rotation

    model.begin_unit_edit([1], fields_only=True)
    model.set_rotation(archer, 1.5)
    model.commit_unit_edit("Rotate archer", history)

    history.undo([], None, None, model)
    assert not model.has_edits
    assert archer.rotation == before_rotation
    assert model.serialize() == original

    history.redo([], None, None, model)
    assert archer.rotation == 1.5


def test_set_unit_const_delta_round_trip_is_byte_clean() -> None:
    loaded, model = _open()
    history = EditHistory()
    gate = _gate(loaded)
    original = _units_section(loaded)
    before = (gate.unit_const, gate.x, gate.y)

    model.begin_unit_edit([1], fields_only=True)
    model.set_unit_const(gate, _GATE_E)
    model.commit_unit_edit("Cycle gate", history)

    history.undo([], None, None, model)
    assert (gate.unit_const, gate.x, gate.y) == before
    assert model.serialize() == original

    history.redo([], None, None, model)
    assert gate.unit_const == _GATE_E


@pytest.mark.parametrize(
    "mutate",
    [_mutate_reassign, _mutate_add, _mutate_add_many, _mutate_remove_many, _mutate_remove],
    ids=["reassign", "add", "add_many", "remove_many", "remove"],
)
def test_a_membership_mutator_raises_inside_a_fields_only_edit(mutate) -> None:
    """A fields_only edit's precondition (Batch D's D6): only set_position/
    set_rotation/set_unit_const may run inside one, enforced rather than
    left as a caller convention that could be told a lie by accident."""
    loaded, model = _open()
    model.begin_unit_edit([0, 1, 2], fields_only=True)
    with pytest.raises(RuntimeError):
        mutate(loaded, model)
    model.abort_unit_edit()


def test_a_delta_records_memory_is_proportional_to_touched_units_not_the_list() -> None:
    """The whole point of D6: a fields_only Nudge of two units out of
    player 1's much larger list must not capture that whole list twice."""
    loaded, model = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    archer = _unit(loaded, _REF_ARCHER_P1)
    assert len(loaded.unit_manager.units[1]) > 2

    model.begin_unit_edit([1], fields_only=True)
    model.set_position(villager, villager.x + 1, villager.y, villager.z)
    model.set_position(archer, archer.x + 1, archer.y, archer.z)
    pending = model._pending
    assert isinstance(pending, UnitFieldSnapshot)
    assert len(pending.entries) == 2
    model.commit_unit_edit("Nudge two units", EditHistory())


def test_every_membership_changing_op_still_produces_a_whole_list_record() -> None:
    """A non-fields_only edit (the default) must still snapshot whole player
    lists -- fields_only is opt-in per edit, not a global behaviour change."""
    loaded, model = _open()
    history = EditHistory()
    model.begin_unit_edit([1])
    model.remove(_unit(loaded, _REF_VILLAGER_P1))
    record = model.commit_unit_edit("Remove villager", history)
    assert isinstance(record.before, UnitSnapshot)
    assert record.unit_field_entries is None


# -- GH #115: set_garrisoned_in() ----------------------------------------------


def _tower(loaded, model, player: int = 1, x: float = 40.5, y: float = 40.5):
    """A Watch Tower placed through the model: the fixture's own host is a House."""
    return model.add(player=player, unit_const=79, x=x, y=y)


def test_set_garrisoned_in_garrisons_unloads_and_moves_between_hosts() -> None:
    """Each step splices the warm reverse-map; serialize() runs
    _check_alignment() -> _check_derived(), which raises on any drift."""
    loaded, model = _open()
    tower = _tower(loaded, model)
    other = _tower(loaded, model, player=2, x=50.5, y=50.5)
    archer = _unit(loaded, _REF_ARCHER_P1)
    model.warm_garrison_map()

    model.set_garrisoned_in(archer, tower.reference_id)
    assert archer.garrisoned_in_id == tower.reference_id
    assert model.referencing(tower) == [archer]
    model.serialize()

    model.set_garrisoned_in(archer, other.reference_id)
    assert model.referencing(tower) == []
    assert model.referencing(other) == [archer]
    model.serialize()

    model.set_garrisoned_in(archer, -1)
    assert archer.garrisoned_in_id == -1
    assert model.referencing(other) == []
    model.serialize()


def test_set_garrisoned_in_dirties_only_that_unit() -> None:
    loaded, model = _open()
    tower = _tower(loaded, model)
    archer = _unit(loaded, _REF_ARCHER_P1)
    player, index = model._locate(archer)
    clean = [i for i, blob in enumerate(model._blobs[2]) if blob is not None]

    model.set_garrisoned_in(archer, tower.reference_id)

    assert model._blobs[player][index] is None
    assert [i for i, blob in enumerate(model._blobs[2]) if blob is not None] == clean


def test_set_garrisoned_in_to_the_current_value_is_a_no_op() -> None:
    loaded, model = _open()
    villager = _unit(loaded, _REF_VILLAGER_P1)
    gen0 = loaded.unit_gen
    model.set_garrisoned_in(villager, _REF_HOUSE)
    assert loaded.unit_gen == gen0
    assert not model.has_edits


def test_set_garrisoned_in_refuses_a_unit_inside_itself() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    with pytest.raises(ValueError, match="itself"):
        model.set_garrisoned_in(archer, archer.reference_id)
    assert archer.garrisoned_in_id == -1


def test_set_garrisoned_in_refuses_a_host_with_no_unit() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    with pytest.raises(ValueError, match="no unit"):
        model.set_garrisoned_in(archer, 999_999)
    assert archer.garrisoned_in_id == -1


def test_set_garrisoned_in_refuses_a_cycle() -> None:
    """The House holds the villager, so the House going inside the villager
    would close a loop."""
    loaded, model = _open()
    house = _unit(loaded, _REF_HOUSE)
    with pytest.raises(ValueError, match="cycle"):
        model.set_garrisoned_in(house, _REF_VILLAGER_P1)
    assert house.garrisoned_in_id == -1


def test_set_garrisoned_in_refuses_a_cycle_through_a_unit_holding_nothing() -> None:
    """A two-step loop the no-nesting check alone would not see: the tower is
    inside the archer (a file fact), and the archer holds only the tower."""
    loaded, model = _open()
    tower = _tower(loaded, model)
    archer = _unit(loaded, _REF_ARCHER_P1)
    tower.garrisoned_in_id = archer.reference_id
    model._garrison = None
    with pytest.raises(ValueError, match="cycle"):
        model.set_garrisoned_in(archer, tower.reference_id)


def test_set_garrisoned_in_refuses_a_unit_that_holds_a_garrison() -> None:
    loaded, model = _open()
    tower = _tower(loaded, model)
    house = _unit(loaded, _REF_HOUSE)
    with pytest.raises(ValueError, match="holds"):
        model.set_garrisoned_in(house, tower.reference_id)
    assert house.garrisoned_in_id == -1


def test_set_garrisoned_in_unloads_even_a_unit_that_holds_a_garrison() -> None:
    """Unload only ever removes a level of nesting, so the scope does not
    block it."""
    loaded, model = _open()
    tower = _tower(loaded, model)
    house = _unit(loaded, _REF_HOUSE)
    house.garrisoned_in_id = tower.reference_id
    model._garrison = None
    model.set_garrisoned_in(house, -1)
    assert house.garrisoned_in_id == -1


def test_a_unit_whose_own_reference_id_is_minus_one_is_not_read_as_a_holder() -> None:
    """-1's reverse-map bucket is every free unit, which is not a garrison."""
    loaded, model = _open()
    tower = _tower(loaded, model)
    archer = _unit(loaded, _REF_ARCHER_P1)
    archer.reference_id = -1
    model.set_garrisoned_in(archer, tower.reference_id)
    assert archer.garrisoned_in_id == tower.reference_id


def test_set_garrisoned_in_is_refused_inside_a_fields_only_edit() -> None:
    loaded, model = _open()
    tower = _tower(loaded, model)
    archer = _unit(loaded, _REF_ARCHER_P1)
    model.begin_unit_edit([1], fields_only=True)
    with pytest.raises(RuntimeError, match="fields_only"):
        model.set_garrisoned_in(archer, tower.reference_id)
    model.abort_unit_edit()
    assert archer.garrisoned_in_id == -1


def test_undo_and_redo_restore_the_live_garrison_link() -> None:
    """The live Unit's link, not only its blob: UnitState carries
    garrisoned_in_id, or the inspector and the save would disagree."""
    loaded, model = _open()
    history = EditHistory()
    tower = _tower(loaded, model)
    archer = _unit(loaded, _REF_ARCHER_P1)
    model.warm_garrison_map()

    model.begin_unit_edit([1])
    model.set_garrisoned_in(archer, tower.reference_id)
    model.set_position(archer, tower.x, tower.y, tower.z)
    model.commit_unit_edit("Garrison units", history)

    history.undo([], None, None, model)
    assert archer.garrisoned_in_id == -1
    assert model.referencing(tower) == []
    model.serialize()

    history.redo([], None, None, model)
    assert archer.garrisoned_in_id == tower.reference_id
    assert model.referencing(tower) == [archer]
    model.serialize()


# -- corpus ---------------------------------------------------------------


@pytest.mark.corpus
def test_construction_gate_passes_across_the_corpus(scenario_path: Path) -> None:
    """units_write_supported=True files should all pass the per-file gate --
    this is what actually re-derives finding 3 on real data, per load."""
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units are not editable")
    UnitEditModel(loaded)  # must not raise


@pytest.mark.corpus
def test_clean_serialize_is_verbatim_across_the_corpus(scenario_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units are not editable")
    model = UnitEditModel(loaded)
    section = loaded.decompressed_body[loaded.units_block_offset : loaded.units_section_end]
    assert model.serialize() == section
