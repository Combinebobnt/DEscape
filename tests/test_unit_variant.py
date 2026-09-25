"""unit_variant's cyclable predicate and cycle arithmetic.

The pure cases run in the default tier: unit_variant reads only committed JSON
tables. The corpus-marked tests at the bottom re-measure the two facts the
design rests on (the committed variant_count matches the real .sld, and every
cyclable placement stores a literal index) rather than trusting the plan.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from descape import gate_orientation, unit_rotation, unit_sprites, unit_variant

OAK, PINE, RAINFOREST, AQUEDUCT, GRANARY, MOLE = 349, 350, 1146, 231, 1089, 2421
BARRELS = 1330
WALL, CLIFF, GATE, ARCHER, HOUSE = 117, 264, 64, 4, 70


@pytest.mark.parametrize("const", [OAK, PINE, RAINFOREST, AQUEDUCT, GRANARY, BARRELS])
def test_trees_and_scenery_are_cyclable(const):
    assert unit_variant.is_cyclable(const)


def test_barrels_cycle_through_six_variants():
    """GH #52's Barrels step, DEscape half: the count Next/Previous wraps at."""
    assert unit_variant.variant_count_for(BARRELS) == 6


@pytest.mark.parametrize("const", [WALL, CLIFF, GATE, ARCHER, HOUSE])
def test_walls_cliffs_gates_angles_and_inert_consts_are_not(const):
    assert not unit_variant.is_cyclable(const)


@pytest.mark.parametrize("const", [AQUEDUCT, MOLE])
def test_class_27_non_walls_stay_cyclable(const):
    """Regression guard: both are class 27, the wall class, so an exclusion
    keyed on class_ rather than on the wall const set would silently drop them."""
    assert unit_variant.is_cyclable(const)


def test_a_const_whose_file_holds_one_variant_is_not_cyclable():
    """1280 (s_statue_b_x1) declares angle_count 2 but its .sld holds 1 frame."""
    assert unit_rotation.angle_count_for(1280) == 2
    assert unit_variant.variant_count_for(1280) == 1
    assert not unit_variant.is_cyclable(1280)


def test_excluded_consts_are_exactly_the_walls_cliffs_and_gates():
    gates = {c for group in gate_orientation.groups().values() for c in group}
    assert len(unit_sprites._ROTATION_VARIANT_CONSTS) == 8
    assert len(unit_sprites._CLIFF_VARIANT_CONSTS) == 96
    assert len(gates) == 96
    assert unit_variant.excluded_consts() == (
        unit_sprites._ROTATION_VARIANT_CONSTS | unit_sprites._CLIFF_VARIANT_CONSTS | gates
    )
    assert len(unit_variant.excluded_consts()) == 200


def test_cycle_step_wraps_and_round_trips():
    assert unit_variant.cycle_step(41.0, 42, 42, 1) == 0.0
    assert unit_variant.cycle_step(0.0, 42, 42, -1) == 41.0
    forward = unit_variant.cycle_step(7.0, 42, 42, 1)
    assert forward == 8.0
    assert unit_variant.cycle_step(forward, 42, 42, -1) == 7.0


def test_cycle_step_always_writes_a_literal_integer():
    """Even from a past-the-end literal or a radian-encoded value."""
    for rotation in (7.0, 50.0, 2.513274, 0.0):
        value = unit_variant.cycle_step(rotation, 42, 42, 3)
        assert value == int(value)
        assert 0 <= value < 42


@pytest.mark.parametrize("count", [0, 1])
def test_cycle_and_randomize_refuse_a_single_variant(count):
    with pytest.raises(ValueError):
        unit_variant.cycle_step(0.0, 2, count, 1)
    with pytest.raises(ValueError):
        unit_variant.random_variant(0.0, 2, count, random.Random(0))


def test_random_variant_never_returns_the_current_one_and_covers_the_rest():
    rng = random.Random(1234)
    seen = set()
    for _ in range(2000):
        value = unit_variant.random_variant(3.0, 5, 5, rng)
        assert value != 3.0
        assert value == int(value)
        seen.add(value)
    assert seen == {0.0, 1.0, 2.0, 4.0}


@pytest.mark.corpus
def test_committed_variant_count_matches_the_real_sld():
    """Render trusts the file; the edit path trusts the committed table. This
    is what pins the two together."""
    from descape import asset_source

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")
    mismatches = []
    for const, entry in unit_sprites.graphic_map().items():
        if not unit_variant.is_cyclable(const):
            continue
        frames = unit_sprites.sld_frame_count(entry["file_name"])
        if frames is None:
            continue
        real = frames // int(entry["frame_count"])
        if real != unit_variant.variant_count_for(const):
            mismatches.append((const, real, unit_variant.variant_count_for(const)))
    assert not mismatches, mismatches[:10]


@pytest.mark.corpus
def test_every_cyclable_placement_stores_a_literal_index(corpus_files):
    """What licenses cycle_step()'s always-literal write: no tree or doodad in
    the corpus uses the radian encoding walls do."""
    from descape import scenario_io

    radian = []
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        for unit in loaded.unit_manager.get_all_units():
            if not unit_variant.is_cyclable(unit.unit_const):
                continue
            if abs(unit.rotation - round(unit.rotation)) >= 1e-6:
                radian.append((path.name, unit.unit_const, unit.rotation))
    assert not radian, f"{len(radian)} non-literal: {radian[:10]}"


_UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _sprite_hashes(loaded):
    """One digest per sprite anchor, over every draw's rgba bytes."""
    import hashlib

    from descape import render

    _img, elevations, proj = render.render_terrain_iso_with_proj(loaded, with_sprites=True)
    layer = render.sprite_draws_by_anchor(loaded, proj, elevations)
    hashes = {}
    for anchor, draws in layer.by_anchor.items():
        digest = hashlib.sha1(usedforsecurity=False)
        for item in draws:
            draw = item[0] if isinstance(item, tuple) else item
            digest.update(draw.rgba.tobytes())
        hashes[anchor] = digest.hexdigest()
    return hashes


@pytest.mark.corpus
def test_cycling_changes_the_rendered_sprite_and_leaves_its_neighbour_alone():
    """The drawn frame must differ after each cycle, and the adjacent untouched
    pine must stay byte-identical, so a global repaint cannot pass."""
    from descape import asset_source
    from descape.scenario_io import load_map_and_units
    from descape.unit_model import UnitEditModel

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")

    loaded = load_map_and_units(_UNITS_FIXTURE)
    model = UnitEditModel(loaded)
    units = {u.reference_id: u for u in loaded.unit_manager.get_all_units()}
    oak, pine = units[100], units[101]

    oak_anchor, pine_anchor = (int(oak.x), int(oak.y)), (int(pine.x), int(pine.y))
    seen = [_sprite_hashes(loaded)]
    assert oak_anchor in seen[0] and pine_anchor in seen[0], "no sprite resolved, so this proves nothing"
    angle_count = unit_rotation.angle_count_for(oak.unit_const)
    variant_count = unit_variant.variant_count_for(oak.unit_const)
    for _ in range(3):
        model.set_variant(oak, unit_variant.cycle_step(oak.rotation, angle_count, variant_count, 5))
        seen.append(_sprite_hashes(loaded))

    assert len({h[oak_anchor] for h in seen}) == 4, "a cycled variant drew a repeated frame"
    assert len({h[pine_anchor] for h in seen}) == 1, "an untouched neighbour changed too"


@pytest.mark.corpus
def test_cycling_barrels_changes_its_sprite_and_leaves_its_twin_alone():
    """GH #52's Barrels Next/Previous step: each Next draws a new frame,
    Previous draws the prior one again, and an identical Barrels on the next
    tile stays byte-identical throughout."""
    from descape import asset_source, render
    from descape.scenario_io import load_map_and_units
    from descape.unit_model import UnitEditModel

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")

    loaded = load_map_and_units(_UNITS_FIXTURE)
    model = UnitEditModel(loaded)
    # GAIA doodads on empty tiles, both starting at literal variant 0.
    barrels = model.add(0, BARRELS, *render.span_anchor(8, 8, 1, 1), rotation=0.0)
    twin = model.add(0, BARRELS, *render.span_anchor(9, 8, 1, 1), rotation=0.0)

    barrels_anchor, twin_anchor = (int(barrels.x), int(barrels.y)), (int(twin.x), int(twin.y))
    seen = [_sprite_hashes(loaded)]
    assert barrels_anchor in seen[0] and twin_anchor in seen[0], "no sprite resolved, so this proves nothing"
    assert seen[0][barrels_anchor] == seen[0][twin_anchor], "twins at one variant drew different frames"
    angle_count = unit_rotation.angle_count_for(BARRELS)
    variant_count = unit_variant.variant_count_for(BARRELS)
    for step in (1, 1, -1):
        model.set_variant(barrels, unit_variant.cycle_step(barrels.rotation, angle_count, variant_count, step))
        seen.append(_sprite_hashes(loaded))

    assert [u.rotation for u in (barrels, twin)] == [1.0, 0.0]
    drawn = [h[barrels_anchor] for h in seen]
    assert len(set(drawn[:3])) == 3, "a Next drew a repeated frame"
    assert drawn[3] == drawn[1], "Previous did not redraw the prior variant"
    assert len({h[twin_anchor] for h in seen}) == 1, "the untouched twin changed too"
