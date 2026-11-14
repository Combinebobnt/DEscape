"""TASK-478: a data-mod switch keeps the decoded art and drops what the tables key.

A data mod ships no sprite or texture art, only tables that can point a const
or a terrain id at another file (its portraits, GH #363, drop on every clear). So a switch keeps the file-keyed caches (decoded .sld frames,
loaded textures) and drops the id-keyed ones (a terrain id's texture path, a
blend block), and a remapped id resolves to the new file's art. A dataset
switch (AoE2 to Return of Rome) still drops everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, game_dataset, terrain_blend, unit_sprites
from testkit import data_mod_fixture
from testkit.data_mod_fixture import SPEARMAN

from test_unit_sprites import build_sld

TERRAIN = 0  # mod_tables() points it at sentinel.dds
SIZE = asset_source.LOADED_TEXTURE_SIZE


@pytest.fixture
def switch_install(tmp_path):
    """An install holding Spearman's and terrain 0's files under both the
    committed tables' names and mod_tables()' sentinel names, each its own colour."""
    from PIL import Image

    root = tmp_path / "install"
    graphics = root / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    entry = unit_sprites.graphic_map()[SPEARMAN]
    frames = entry["frame_count"] * entry["angle_count"]
    (graphics / f"{entry['file_name']}.sld").write_bytes(build_sld(frames, playercolor=False))
    (graphics / "sentinel_x1.sld").write_bytes(build_sld(frames, playercolor=False, colour_base=11))
    textures = root / asset_source.TERRAIN_TEXTURE_SUBPATH
    textures.mkdir(parents=True)
    install_name = asset_source._terrain_texture_map()[TERRAIN]
    Image.new("RGB", (16, 16), (30, 140, 30)).save(textures / install_name, format="PNG")
    Image.new("RGB", (16, 16), (200, 40, 160)).save(textures / "sentinel.dds", format="PNG")
    mod_dir = tmp_path / "data_mod"
    mod_dir.mkdir()
    mod = data_mod_fixture.mod_tables(mod_dir)
    asset_source.set_install_path_override(root)
    asset_source.clear_install_caches()
    yield mod
    asset_source.set_install_path_override(None)


def _art():
    """(Spearman's sprite pixels, terrain 0's texture, terrain 0's path) under the tables in effect."""
    sprite = unit_sprites.sprite_for(SPEARMAN, 0.0, 1, 32)
    texture = asset_source.terrain_texture_at(TERRAIN, SIZE)
    assert sprite is not None and texture is not None
    return sprite.rgba.copy(), np.array(texture), asset_source.get_terrain_texture_path(TERRAIN)


def _texture_entries() -> int:
    return asset_source._terrain_texture_for_path.cache_info().currsize


def test_a_data_mod_switch_keeps_the_decoded_art_and_a_dataset_switch_does_not(switch_install):
    _art()
    counts, textures = unit_sprites.cache_counts(), _texture_entries()
    assert counts != (0, 0) and textures
    generation = asset_source.install_generation()
    assert game_dataset.set_data_mod(switch_install)
    assert unit_sprites.cache_counts() == counts
    assert _texture_entries() == textures
    assert asset_source.install_generation() == generation, "ThumbLoader pairs it with game_dataset.generation()"
    game_dataset.set_active(game_dataset.ROR)
    assert unit_sprites.cache_counts() == (0, 0)
    assert _texture_entries() == 0


@pytest.mark.parametrize(
    ("changed", "keeps"),
    [
        (game_dataset.DATA_MOD_SWITCH, True),
        (game_dataset.GAME_SWITCH, False),
        # A layer the kept block does not list, as a data mod moving art resolution would add (GH #272).
        (frozenset({game_dataset.TABLES, "art"}), False),
        (None, False),
    ],
)
def test_only_a_tables_change_keeps_the_file_keyed_caches(switch_install, changed, keeps):
    _art()
    counts, textures = unit_sprites.cache_counts(), _texture_entries()
    generation = asset_source.install_generation()
    asset_source.clear_install_caches(changed)
    if keeps:
        assert (unit_sprites.cache_counts(), _texture_entries(), asset_source.install_generation()) == (counts, textures, generation)
    else:
        assert (unit_sprites.cache_counts(), _texture_entries()) == ((0, 0), 0)
        assert asset_source.install_generation() > generation


def test_a_switch_follows_the_tables_remap_to_the_other_files_art(switch_install):
    before = _art()
    assert game_dataset.set_data_mod(switch_install)
    warm = _art()
    assert warm[2].name == "sentinel.dds"
    assert not np.array_equal(warm[0], before[0]) and not np.array_equal(warm[1], before[1])
    asset_source.clear_install_caches()
    cold = _art()
    for got, want in zip(warm, cold, strict=True):
        assert np.array_equal(got, want) if isinstance(want, np.ndarray) else got == want
    assert game_dataset.set_data_mod(None)
    back = _art()
    for got, want in zip(back, before, strict=True):
        assert np.array_equal(got, want) if isinstance(want, np.ndarray) else got == want


def test_a_blend_block_composed_before_a_switch_is_not_served_after_it(switch_install):
    key = (TERRAIN, -1, ())
    before = terrain_blend.blended_block(key, 0, 0, 32)
    assert before is not None
    assert game_dataset.set_data_mod(switch_install)
    after = terrain_blend.blended_block(key, 0, 0, 32)
    assert after is not None and not np.array_equal(after, before)
    asset_source.clear_install_caches()
    assert np.array_equal(terrain_blend.blended_block(key, 0, 0, 32), after)
