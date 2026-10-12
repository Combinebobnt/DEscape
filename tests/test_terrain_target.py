"""descape/terrain_target.py and batch_api.set_layer: which of a tile's two
terrains a terrain tool writes (its base terrain_id, or the layered terrain
in `layer`). Fake tiles throughout, plus one real save and reload on the
blank template. The viewer half is tests/test_terrain_target_viewer.py.
"""

from __future__ import annotations

import pytest

from descape import batch_api, settings, terrain_target
from descape.terrain_target import CLEAR, LAYER, TERRAIN


class FakeTile:
    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


def test_set_layer_writes_the_layer_and_leaves_the_base() -> None:
    tile = FakeTile(terrain_id=4, elevation=3)
    batch_api.set_layer(tile, 9)
    assert (tile.terrain_id, tile.elevation, tile.layer) == (4, 3, 9)


def test_set_layer_none_clears() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    batch_api.set_layer(tile, None)
    assert (tile.terrain_id, tile.layer) == (4, -1)


def test_set_layer_equal_to_the_base_writes_none() -> None:
    tile = FakeTile(terrain_id=4)
    batch_api.set_layer(tile, 4)
    assert (tile.terrain_id, tile.layer) == (4, -1)


def test_set_layer_equal_to_the_base_clears_an_existing_layer() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    batch_api.set_layer(tile, 4)
    assert (tile.terrain_id, tile.layer) == (4, -1)


@pytest.mark.parametrize("bad_id", [-1, 256, 40000])
def test_set_layer_rejects_an_out_of_range_id_and_writes_nothing(bad_id: int) -> None:
    tile = FakeTile(terrain_id=4, elevation=2, layer=9)
    with pytest.raises(ValueError):
        batch_api.set_layer(tile, bad_id)
    assert (tile.terrain_id, tile.elevation, tile.layer) == (4, 2, 9)


@pytest.mark.parametrize("edge_id", [0, 255])
def test_set_layer_accepts_the_range_ends(edge_id: int) -> None:
    tile = FakeTile(terrain_id=4)
    batch_api.set_layer(tile, edge_id)
    assert tile.layer == edge_id


def test_set_terrain_still_clears_the_layer() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    batch_api.set_terrain(tile, 6)
    assert (tile.terrain_id, tile.layer) == (6, -1)


def test_write_tile_terrain_sets_the_base_and_clears_the_layer() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    terrain_target.write_tile(tile, TERRAIN, 6)
    assert (tile.terrain_id, tile.layer) == (6, -1)


def test_write_tile_layer_sets_the_layer_only() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    terrain_target.write_tile(tile, LAYER, 6)
    assert (tile.terrain_id, tile.layer) == (4, 6)


def test_write_tile_layer_equal_to_the_base_writes_none() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    terrain_target.write_tile(tile, LAYER, 4)
    assert (tile.terrain_id, tile.layer) == (4, -1)


def test_write_tile_clear_ignores_the_pick() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    terrain_target.write_tile(tile, CLEAR, 6)
    assert (tile.terrain_id, tile.layer) == (4, -1)


def test_write_tile_rejects_an_unknown_target() -> None:
    tile = FakeTile(terrain_id=4, layer=9)
    with pytest.raises(ValueError):
        terrain_target.write_tile(tile, "overlay", 6)
    assert (tile.terrain_id, tile.layer) == (4, 9)


def test_targets_are_three_distinct_names() -> None:
    assert terrain_target.TARGETS == (TERRAIN, LAYER, CLEAR)
    assert len(set(terrain_target.TARGETS)) == 3


def test_label_table_covers_every_terrain_tool() -> None:
    terrain_tools = {t.tool_id for t in settings.TOOLS if t.param_widget == "terrain"}
    assert terrain_tools  # the table below would otherwise pass on an empty set
    assert set(terrain_target.LAYER_STROKE_LABELS) == {
        (tool_id, target) for tool_id in terrain_tools for target in (LAYER, CLEAR)
    }
    # One History label per gesture kind: no two (tool, target) pairs share one.
    labels = list(terrain_target.LAYER_STROKE_LABELS.values())
    assert len(set(labels)) == len(labels)
    assert not set(labels) & {t.stroke_label for t in settings.TOOLS}


def test_stroke_label_per_target() -> None:
    assert terrain_target.stroke_label("draw", TERRAIN, "Paint terrain") == "Paint terrain"
    assert terrain_target.stroke_label("draw", LAYER, "Paint terrain") == "Paint layer"
    assert terrain_target.stroke_label("draw", CLEAR, "Paint terrain") == "Clear layer"
    assert terrain_target.stroke_label("fill", LAYER, "Fill terrain") == "Fill layer"
    assert terrain_target.stroke_label("fill", CLEAR, "Fill terrain") == "Clear layer (fill)"
    assert terrain_target.stroke_label("draw_line", LAYER, "Draw line") == "Draw layer line"
    assert terrain_target.stroke_label("draw_line", CLEAR, "Draw line") == "Clear layer (line)"
    assert terrain_target.stroke_label("draw_rect", LAYER, "Draw rectangle") == "Draw layer rectangle"
    assert terrain_target.stroke_label("draw_rect", CLEAR, "Draw rectangle") == "Clear layer (rectangle)"
    # A tool with no per-target label keeps its own.
    assert terrain_target.stroke_label("elevation", LAYER, "Elevate") == "Elevate"


def test_set_layer_round_trips_through_a_real_save(tmp_path) -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario

    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    assert scenario.terrain_has_layer
    width = mm.map_width
    base_ids = [t.terrain_id for t in mm.terrain]
    # Asymmetric on purpose: an L, two layer ids, nothing mirrored across either axis.
    wanted = {(3, 5): 9, (4, 5): 9, (5, 5): 9, (3, 6): 11, (3, 7): 11}
    for (x, y), layer_id in wanted.items():
        batch_api.set_layer(mm.terrain[y * width + x], layer_id)
    # A layer equal to the base must reload unlayered.
    batch_api.set_layer(mm.terrain[20 * width + 10], base_ids[20 * width + 10])

    out = tmp_path / "layered.aoe2scenario"
    write_scenario(scenario, out, backup=False)
    reloaded = load_map_and_units(out).map_manager

    got = {(i % width, i // width): t.layer for i, t in enumerate(reloaded.terrain) if t.layer != -1}
    assert got == wanted
    assert [t.terrain_id for t in reloaded.terrain] == base_ids
