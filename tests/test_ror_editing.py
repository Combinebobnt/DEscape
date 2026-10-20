"""GH #67 PR 2: Return of Rome scenarios are edited with Return of Rome rules.

Every const set the AGENTS.md write-path exceptions name follows the active
game dataset, measured on the RoR .dat and 8 RoR samples (2026-10-17): walls
lose WALL5 2678, cliffs lose the Short Marble family, Farm 50 becomes
cyclable, and the trebuchets stay ANGLE. RoR civ tokens re-encode, so a 1.56+
RoR file's Players rows are writable, and its buildings draw their civ's and
age's art. Qt-free."""

from __future__ import annotations

import pytest

from descape import (
    civ_art,
    cliff_catalog,
    command_buttons,
    game_dataset,
    gate_orientation,
    player_fields,
    render,
    scenario_io,
    trigger_fields,
    unit_kind,
    unit_rotation,
    unit_sprites,
    unit_variant,
)
from descape.options_model import players_write_supported
from descape.unit_model import UnitEditModel
from testkit.ror_fixture import ror_copy

ROR_WALLS = (72, 117, 119, 155, 231, 370, 788, 1062)
SHORT_MARBLE = range(2190, 2226)
TREBUCHETS = (42, 331, 1690, 1691)
FARM, RICE_FARM, PASTURE = 50, 1187, 1897
CLIFF, HOPLITE = 264, 93
BARRACKS = 12


def _ror() -> None:
    game_dataset.set_active(game_dataset.ROR)


def _gates() -> frozenset[int]:
    return frozenset(c for group in gate_orientation.groups().values() for c in group)


# -- const sets ------------------------------------------------------------------


def test_the_ror_wall_family_drops_wall5():
    assert 2678 in unit_sprites.wall_family_consts()
    _ror()
    assert unit_sprites.wall_family_consts() == ROR_WALLS
    assert 2678 not in unit_sprites.wall_connector_consts()
    assert not unit_sprites.rotation_variant_eligible(2678)


def test_the_cliff_sets_agree_with_the_dat_in_both_datasets():
    assert len(unit_sprites.cliff_consts()) == 96
    assert unit_sprites.cliff_consts() == unit_kind.cliff_consts()
    _ror()
    assert len(unit_sprites.cliff_consts()) == 60
    assert unit_sprites.cliff_consts() == unit_kind.cliff_consts()
    assert not set(SHORT_MARBLE) & unit_sprites.cliff_consts()


@pytest.mark.parametrize("dataset", game_dataset.DATASETS)
def test_each_wall_literal_equals_its_dat_derived_class_set(dataset):
    game_dataset.set_active(dataset)
    assert unit_kind.wall_consts() - _gates() == unit_sprites.rotation_variant_const_set()


def test_the_literals_stay_the_aoe2_sets_under_ror():
    """The module names stay AoE2's (the tools read them); the accessors, which unit_variant reads, switch."""
    _ror()
    assert 2678 in unit_sprites._ROTATION_VARIANT_CONSTS
    assert len(unit_sprites._CLIFF_VARIANT_CONSTS) == 96
    expected = frozenset(ROR_WALLS) | (unit_sprites._CLIFF_VARIANT_CONSTS - frozenset(SHORT_MARBLE)) | _gates()
    assert unit_variant.excluded_consts() == expected
    assert len(unit_variant.excluded_consts()) == 8 + 60 + len(_gates())


@pytest.mark.parametrize("dataset", game_dataset.DATASETS)
def test_the_trebuchets_are_angle_and_not_variant(dataset):
    game_dataset.set_active(dataset)
    for const in TREBUCHETS:
        assert unit_rotation.rotation_is_angle(const), const
        assert not unit_sprites.rotation_is_variant(const), const


def test_the_ror_fortified_palisade_draws_its_body():
    _ror()
    assert unit_sprites.graphic_map()[119]["graphic_id"] == 605


def test_ror_heroes_glow():
    from descape.terrain_palette import HERO_GLOW_CONSTS

    _ror()
    assert HERO_GLOW_CONSTS


def test_cliff_families_follow_the_dataset():
    marble = {piece.unit_const for pieces in cliff_catalog.families().values() for piece in pieces}
    assert set(SHORT_MARBLE) <= marble
    _ror()
    families = cliff_catalog.families()
    consts = {piece.unit_const for pieces in families.values() for piece in pieces}
    assert consts == unit_sprites.cliff_consts()
    assert not any(name.startswith("const_") for name in families)


def test_command_button_tables_follow_the_dataset():
    from descape import object_catalog

    aoe2_code = command_buttons._rows()[command_buttons.UNIT][HOPLITE]["code"]
    _ror()
    assert command_buttons._rows()[command_buttons.UNIT] == object_catalog.dat_objects()
    assert command_buttons._rows()[command_buttons.UNIT][HOPLITE]["code"] != aoe2_code


# -- model guards ----------------------------------------------------------------


@pytest.fixture()
def ror_model(tmp_path):
    loaded = scenario_io.load_map_and_units(ror_copy(tmp_path))
    _ror()
    return UnitEditModel(loaded)


def test_set_cliff_frame_refuses_short_marble_and_accepts_a_ror_cliff(ror_model):
    marble = ror_model.add(0, SHORT_MARBLE[0], 10.5, 10.5)
    with pytest.raises(ValueError):
        ror_model.set_cliff_frame(marble, 1)
    cliff = ror_model.add(0, CLIFF, 20.5, 20.5)
    ror_model.set_cliff_frame(cliff, 2)
    assert (cliff.rotation, cliff.initial_animation_frame) == (2.0, 2)


def test_set_variant_cycles_a_ror_farm_only_under_ror(tmp_path):
    loaded = scenario_io.load_map_and_units(ror_copy(tmp_path))
    model = UnitEditModel(loaded)
    farm = model.add(1, FARM, 30.5, 30.5)
    with pytest.raises(ValueError):
        model.set_variant(farm, 1.0)
    _ror()
    assert unit_variant.is_cyclable(FARM)
    model.set_variant(farm, 1.0)
    assert farm.rotation == 1.0


def test_set_rotation_turns_a_hoplite(ror_model):
    hoplite = ror_model.add(1, HOPLITE, 40.5, 40.5)
    ror_model.set_rotation(hoplite, 1.0)
    assert hoplite.rotation == 1.0


def test_set_wall_variant_refuses_wall5_under_ror(ror_model):
    wall5 = ror_model.add(1, 2678, 50.5, 50.5)
    with pytest.raises(ValueError):
        ror_model.set_wall_variant(wall5, 1)
    wall = ror_model.add(1, 117, 51.5, 50.5)
    ror_model.set_wall_variant(wall, 1)
    assert wall.rotation == 1.0


# -- players ---------------------------------------------------------------------


def test_a_ror_civ_token_is_encodable_and_keeps_players_writable(tmp_path):
    from descape import option_fields
    from descape.options_model import OptionsEditModel
    from descape.scenario_write import write_scenario

    _ror()
    path = ror_copy(tmp_path)
    loaded = scenario_io.load_map_and_units(path)
    model = OptionsEditModel(loaded, option_fields.specs_for(loaded))
    # EGYPTIAN-CIV is RoR-only; PERSIAN-CIV is also AoE2's token, so it would prove nothing.
    model.set_value(player_fields.player_field_id("civilization", 1), "EGYPTIAN-CIV")
    model.set_value(player_fields.player_field_id("architecture", 1), "LAC-VIET-CIV")
    out = tmp_path / "saved.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)
    reloaded = scenario_io.load_map_and_units(out)
    spec = next(s for s in player_fields.specs_for(reloaded) if s.field_id == "civilization")
    assert player_fields.current_value(reloaded, spec, 1) == "EGYPTIAN-CIV"
    assert players_write_supported(reloaded), "a stored RoR token must re-encode"
    assert reloaded.game_variant == game_dataset.ROR_VARIANT


def test_ror_starting_ages_are_its_four_ages():
    assert len(player_fields.starting_age_choices()) == 5
    _ror()
    assert player_fields.starting_age_choices() == (
        (2, "Stone Age"), (3, "Tool Age"), (4, "Bronze Age"), (5, "Iron Age"), (6, "Post-Iron Age"),
    )


def test_ror_civs_index_their_own_dat():
    assert civ_art.civ_index("PERSIAN-CIV") not in (None, 9)
    assert civ_art.civ_index("PERSIAN-CIV", ror=True) == 9
    assert civ_art.civ_index(17, ror=True) == 17
    assert civ_art.civ_index(18, ror=True) is None
    assert civ_art.civ_index("RANDOM", ror=True) is None


def test_a_ror_players_barracks_draws_its_civ_and_age_art(tmp_path):
    loaded = scenario_io.load_map_and_units(ror_copy(tmp_path))
    art = civ_art.player_art(loaded, {"civilization": {1: "EGYPTIAN-CIV"}, "architecture": {1: "EGYPTIAN-CIV"}, "starting_age": {1: 4}})
    assert art[1] == (1, 4)
    _ror()
    assert unit_sprites.resolve_entry(BARRACKS, 1, 4)["file_name"] == "b_egy_barracks_age3_x1"
    assert unit_sprites.resolve_entry(BARRACKS, 13, 4)["file_name"] == "b_rom_barracks_age3_x1"


def test_a_ror_file_gets_player_art_and_a_mismatched_display_none(tmp_path):
    loaded = scenario_io.load_map_and_units(ror_copy(tmp_path))
    assert loaded.player_art is not None
    assert render.player_art_for(loaded, 1) is None, "AoE2 data shown on a RoR file"
    _ror()
    assert render.player_art_for(loaded, 1) == loaded.player_art[1]


# -- farms, triggers -------------------------------------------------------------


def test_ror_farms_drape_their_foundation_terrain_under_the_sprite():
    assert render._terrain_overlay_for(FARM) == 7
    _ror()
    assert FARM in unit_sprites.graphic_map(), "RoR farms carry a sprite (the vases)"
    assert render._terrain_overlay_for(FARM) == 7
    assert render._terrain_overlay_for(RICE_FARM) == 63
    assert render._terrain_overlay_for(PASTURE) == 117


def test_a_ror_tech_reference_shows_its_ror_name():
    assert trigger_fields.resolve_reference("TechInfo", 101) == "FEUDAL AGE"
    _ror()
    assert trigger_fields.resolve_reference("TechInfo", 101) == "Tool Age"
    assert trigger_fields.resolve_reference("TechInfo", 99999) == ""


def test_ror_pastures_draw_their_hut_and_posts():
    """Same annex tree as AoE2 in the RoR .dat (hut 1890, posts 1888); the hut shows in game."""
    _ror()
    pieces = unit_sprites.graphic_map()[PASTURE].get("pieces", ())
    assert {piece["unit_id"] for piece in pieces} >= {1888}


def test_ror_lists_the_objects_it_shares_with_aoe2_as_hidden():
    """GH #67: the Castle and the other objects RoR's editor hides but its .dat
    keeps as the same object (same code and class) show under "Show
    editor-hidden objects"; an emptied slot and a reused one do not."""
    import json

    from descape import object_catalog

    aoe2 = json.loads(object_catalog._CATALOG_JSON_PATH.read_text())["objects"]
    _ror()
    listed = {entry.id: entry for entry in object_catalog.objects()}
    assert listed[82].hidden and listed[82].category == "Buildings"
    assert 2678 not in listed and 1808 not in listed
    assert not listed[HOPLITE].hidden
    hidden = [entry for entry in listed.values() if entry.hidden]
    assert len(hidden) > 500
    ror = object_catalog.dat_objects()
    for entry in hidden:
        old, new = aoe2[str(entry.id)], ror[entry.id]
        assert (old["code"], old["class"]) == (new["code"], new["class"]), entry.id


# -- terrain texture scale on zoomed-out levels ------------------------------------


def test_ror_zoomed_out_levels_shrink_textures_and_aoe2_does_not():
    from descape import asset_source

    assert [asset_source.terrain_texture_size(t) for t in (16, 32, 64, 128)] == [512] * 4
    _ror()
    assert [asset_source.terrain_texture_size(t) for t in (16, 32, 64, 128)] == [128, 256, 512, 512]
    assert all(asset_source.terrain_texture_size(t) % t == 0 for t in (8, 16, 24, 32, 48))


def test_a_ror_zoomed_out_tile_crops_the_shrunk_texture(tmp_path, monkeypatch):
    """A tile at 32 px keeps the share of the texture it has at 64 px: a
    quarter-size crop of the half-size texture, not of the full one."""
    import functools
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    from descape import asset_source

    rng = np.random.default_rng(67)
    path = tmp_path / "g_test.png"
    Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(path)
    # Cached, as the real one is: a dataset switch calls its cache_clear().
    monkeypatch.setattr(asset_source, "get_terrain_texture_path", functools.cache(lambda _tid: path))
    full = asset_source.get_terrain_texture_array(7)
    tile = SimpleNamespace(x=1, y=0, terrain_id=7)
    assert np.array_equal(render._top_block(tile, 32), full[0:32, 32:64]), "AoE2: one size at every level"
    _ror()
    half = asset_source.terrain_texture_at(7, 256)
    assert half.shape == (256, 256, 3)
    assert np.array_equal(render._top_block(tile, 32), half[0:32, 32:64])
    assert np.array_equal(render._top_block(tile, 64), full[0:64, 64:128]), "the reference level is unchanged"


def test_ror_shared_hidden_relics_and_wild_animals_stay_gaia_only():
    _ror()
    gaia_only = unit_kind.gaia_only_consts()
    assert 285 in gaia_only, "RELIC"
    assert unit_kind.placement_owner(285, 1) == 0
