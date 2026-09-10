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

from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_units import UnitAddSpec
from descape.unit_model import UnitEditModel, UnitEditsUnavailableError, _serialize_unit

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


# -- 1. construction gate ----------------------------------------------------


def test_construction_gate_fires_on_a_corrupted_unit() -> None:
    """A gate that never fires is not a gate (plan verification item 9)."""
    loaded = load_map_and_units(FIXTURE_PATH)
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
    loaded = load_map_and_units(FIXTURE_PATH)
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
    succeed even when every entry's own get_data_as_bytes() would raise."""
    loaded, model = _open()
    wall = _unit(loaded, _REF_WALL)
    model.reassign(wall, 1)

    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    for pu in players_units:
        for entry in pu.retriever_map["units"].data:
            entry.get_data_as_bytes = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a clean unit's blob must never re-enter the library")
            )

    model.serialize()  # must not raise


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
    loaded, model = _open()
    unit = model.add(player=0, unit_const=349, x=1.5, y=1.5, z=0.0, rotation=41.0)
    assert unit.rotation == 41.0


def test_add_rejects_an_out_of_range_player() -> None:
    loaded, model = _open()
    with pytest.raises(ValueError):
        model.add(player=9, unit_const=83, x=1.5, y=1.5, z=0.0, rotation=0.0)


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
    loaded, model = _open()
    spec = UnitAddSpec(x=1.5, y=1.5, unit_const=349, rotation=41.0, initial_animation_frame=41)
    unit = model.add_many(player=0, specs=[spec])[0]
    assert unit.rotation == 41.0
    assert unit.initial_animation_frame == 41


def test_add_many_matches_add_for_equivalent_placements() -> None:
    """The batch path must place the same shape of unit add() does -- this
    guards against add_many() drifting onto a different set of Unit
    defaults (z, status, garrisoned_in_id, caption) than add()'s own."""
    loaded_a, model_a = _open()
    single = model_a.add(player=1, unit_const=349, x=3.5, y=3.5, z=0.0, rotation=2.0, initial_animation_frame=2)

    loaded_b, model_b = _open()
    spec = UnitAddSpec(x=3.5, y=3.5, unit_const=349, rotation=2.0, initial_animation_frame=2)
    batched = model_b.add_many(player=1, specs=[spec])[0]

    assert (batched.x, batched.y, batched.z) == (single.x, single.y, single.z)
    assert (batched.unit_const, batched.status) == (single.unit_const, single.status)
    assert (batched.rotation, batched.initial_animation_frame) == (single.rotation, single.initial_animation_frame)
    assert batched.garrisoned_in_id == single.garrisoned_in_id


def test_add_many_rejects_an_out_of_range_player() -> None:
    loaded, model = _open()
    spec = UnitAddSpec(x=1.5, y=1.5, unit_const=83, rotation=0.0, initial_animation_frame=0)
    with pytest.raises(ValueError):
        model.add_many(player=9, specs=[spec])


def test_remove_many_deletes_every_unit() -> None:
    loaded, model = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    villager = _unit(loaded, _REF_VILLAGER_P2)
    model.remove_many([archer, villager])
    assert archer not in loaded.unit_manager.units[1]
    assert villager not in loaded.unit_manager.units[2]
    assert model.has_edits


def test_remove_many_of_an_empty_list_is_a_no_op() -> None:
    loaded, model = _open()
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
    from AoE2ScenarioParser.objects.data_objects.unit import Unit

    loaded, model = _open()
    stray = Unit(
        player=0, x=0, y=0, z=0, reference_id=99999, unit_const=4, status=2,
        rotation=0, initial_animation_frame=0,
    )
    with pytest.raises(ValueError):
        model.set_position(stray, 1.0, 1.0, 0.0)


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
