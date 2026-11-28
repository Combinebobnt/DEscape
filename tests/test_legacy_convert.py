"""GH #315's converter (descape/legacy_convert.py) on hand-built values: each
import rule and value remap. The DE-import pairs are corpus-tier
(test_legacy_convert_oracle.py); a real legacy load-and-convert test joins
here once small AoK/AoC fixtures exist.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import game_dataset, games, legacy_convert, scenario_format, unit_sprites
from descape.legacy_convert import plan_unit_image, unit_rule


@pytest.fixture(autouse=True)
def _aoe2():
    with game_dataset.activated(games.default().id):
        yield


def test_each_measured_family_has_its_rule() -> None:
    assert unit_rule(349) == "trees"  # oak
    assert unit_rule(72) == "walls"  # palisade
    assert unit_rule(69) == "fish"  # shore fish
    assert unit_rule(70) == "buildings"  # house, 3 frames
    assert unit_rule(95) == "gates"  # stone gate corner
    assert unit_rule(4) is None  # archer: an ANGLE facing
    assert unit_rule(next(iter(unit_sprites.cliff_const_set()))) is None


def test_tree_variants_land_in_the_importers_band_with_frame_equal() -> None:
    rng = random.Random(3)
    for v in range(14):
        rotation, frame = plan_unit_image("trees", 349, float(v), v, rng)
        assert 3 * v <= rotation <= 3 * v + 2 and frame == rotation
    assert plan_unit_image("trees", 349, 0.7, 5, rng) == (0.7, 5)  # not a legacy variant: as stored


def test_fish_buildings_gates_and_walls_take_their_rule() -> None:
    rng = random.Random(4)
    for _ in range(50):
        rotation, frame = plan_unit_image("fish", 69, 7.0, 33, rng)
        assert rotation == frame and 0 <= frame < legacy_convert._frame_count(69)
        rotation, frame = plan_unit_image("buildings", 70, 0.785, 9, rng)
        assert rotation == 0.0 and 0 <= frame < legacy_convert._frame_count(70)
    one = next(c for c in (45, 95, 103, 209) if legacy_convert._frame_count(c) == 1)
    assert plan_unit_image("buildings", one, 0.0, 34, rng) == (0.0, 0)  # one frame
    assert plan_unit_image("gates", 95, 1.25, 7, rng) == (1.25, 0)  # rotation unmeasured: kept
    assert plan_unit_image("walls", 72, 2.513, 0, rng, wall_shape=1) == (1.0, 0)
    assert plan_unit_image(None, 4, 3.14159, 2, rng) == (3.14159, 2)


def test_wall_shapes_come_from_neighbours_and_a_lone_wall_is_a_tower() -> None:
    def wall(x):
        return SimpleNamespace(unit_const=72, x=x + 0.5, y=10.5, rotation=0.0)

    run = [wall(4), wall(5), wall(6)]
    lone = SimpleNamespace(unit_const=72, x=20.5, y=20.5, rotation=0.0)
    shapes = legacy_convert.wall_shapes([*run, lone], 40, 40)
    assert shapes[id(run[1])] == 0  # inside an x run
    assert shapes[id(lone)] == unit_sprites.WALL_TOWER


@pytest.mark.parametrize(
    ("field_id", "legacy", "de"),
    [
        ("civilization", 1, "BRITON-CIV"),
        ("civilization", 17, "HUN-CIV"),
        ("civilization", 13, "CELTIC-CIV"),  # through the library's names, unmeasured
        ("starting_age", 0xFFFFFFFF, 0),
        ("starting_age", 1, 3),
        ("pop_limit", 75, legacy_convert.DE_POP_LIMIT),
        ("player_name_string_id", -1, -2),
        ("player_name_string_id", 9001, 9001),
        ("tribe_name", "Bl\xe9da", "Bléda"),
    ],
)
def test_player_values_remap_as_the_importer_does(field_id, legacy, de) -> None:
    assert legacy_convert.player_value_to_de(field_id, legacy) == de


def test_terrain_remap_writes_the_measured_layer() -> None:
    assert legacy_convert.terrain_to_de(33, -1) == (6, 32)
    assert legacy_convert.terrain_to_de(34, -1) == (0, 32)
    assert legacy_convert.terrain_to_de(38, -1) == (38, -1)  # the game's own save normalizes it


def test_legacy_text_re_decodes_cp1252_and_keeps_crlf() -> None:
    raw = "It\x92s caf\xe9\r\nnext\x00".encode("latin-1")
    # The library decodes non-utf-8 bytes as latin-1, which misreads 0x92.
    assert legacy_convert.legacy_text(raw.decode("latin-1")) == "It’s café\r\nnext"
    assert legacy_convert.legacy_text("plain") == "plain"
    assert legacy_convert.legacy_text("café ’") == "café ’"  # already a utf-8 read


def test_elevation_steps_count_neighbour_jumps_over_one() -> None:
    assert legacy_convert.elevation_steps(3, 2, [0, 1, 2, 0, 1, 2]) == 0
    assert legacy_convert.elevation_steps(3, 2, [0, 2, 2, 0, 0, 2]) == 3  # two along x, one along y


def test_trigger_count_reads_the_tails_s32() -> None:
    tail = bytes(9) + (103).to_bytes(4, "little", signed=True) + bytes(4)
    assert legacy_convert.legacy_trigger_count(SimpleNamespace(trigger_tail=tail)) == 103
    assert legacy_convert.legacy_trigger_count(SimpleNamespace(trigger_tail=b"")) == 0


def test_convert_refuses_a_de_document() -> None:
    with pytest.raises(ValueError, match="pre-DE"):
        legacy_convert.convert(SimpleNamespace(source_format=SimpleNamespace(legacy=False)))


def test_the_report_names_what_was_left_out() -> None:
    report = legacy_convert.ConversionReport(triggers_left_out=4)
    assert report.left_out and any("4 triggers" in line for line in report.lines())
    clean = legacy_convert.ConversionReport()
    clean.rewritten["trees"] = 9
    assert not clean.left_out and clean.lines() == ["Unit images set as AoE2:DE's import sets them: 9 trees"]


# -- plan_units() repairs and the whole-document plumbing ----------------------


def _unit(rid, const=4, host=-1, x=5.5):
    return SimpleNamespace(
        unit_const=const, reference_id=rid, garrisoned_in_id=host, x=x, y=6.5, z=0.0, status=2,
        rotation=1.5, initial_animation_frame=0,
    )  # fmt: skip


def _legacy_units(players: dict[int, list], *, game: str = scenario_format.AOC, counter: int = 0):
    units = [players.get(p, []) for p in range(9)]
    return SimpleNamespace(
        map_manager=SimpleNamespace(map_width=40, map_height=40),
        unit_manager=SimpleNamespace(units=units),
        decompressed_body=counter.to_bytes(4, "little", signed=True),
        source_format=SimpleNamespace(game=game),
    )


def _planned(legacy):
    report = legacy_convert.ConversionReport()
    lists, counter = legacy_convert.plan_units(legacy, random.Random(0), report)
    return lists, counter, report


def test_plan_units_drops_unknown_consts_and_renumbers_bad_ids() -> None:
    legacy = _legacy_units({1: [_unit(10), _unit(11, const=99999), _unit(-3), _unit(10, x=7.5)], 2: [_unit(12)]}, counter=5)
    lists, counter, report = _planned(legacy)
    assert report.dropped_consts == {99999: 1}
    assert report.renumbered == 2
    assert [s["reference_id"] for s in lists[1]] == [10, 13, 14], "fresh ids above the highest, in list order"
    assert [s["x"] for s in lists[1]] == [5.5, 5.5, 7.5]
    assert counter == 15
    assert [s["reference_id"] for s in lists[2]] == [12]


def test_plan_units_repairs_garrison_links_and_keeps_cross_player_ones() -> None:
    legacy = _legacy_units({1: [_unit(1), _unit(2, host=1), _unit(3, host=99), _unit(4, host=4)], 2: [_unit(5, host=1)]})
    lists, _counter, report = _planned(legacy)
    assert [s["garrisoned_in_id"] for s in lists[1]] == [-1, 1, -1, -1]
    assert lists[2][0]["garrisoned_in_id"] == 1, "a cross-player link is kept"
    assert report.links_cut == 2 and report.aok_links_cleared == 0


def test_plan_units_reads_an_aok_zero_link_as_none() -> None:
    lists, _counter, report = _planned(_legacy_units({0: [_unit(0)], 1: [_unit(1, host=0), _unit(2, host=1)]}, game=scenario_format.AOK))
    assert [s["garrisoned_in_id"] for s in lists[1]] == [-1, 1]
    assert report.aok_links_cleared == 1 and report.links_cut == 0
    lists, _counter, report = _planned(_legacy_units({0: [_unit(0)], 1: [_unit(1, host=0)]}))
    assert lists[1][0]["garrisoned_in_id"] == 0 and not report.aok_links_cleared, "outside AoK, 0 is a real host"


@pytest.mark.test_map
def test_a_converted_test_map_keeps_units_counter_and_cameras_and_round_trips(tmp_path) -> None:
    from descape import scenario_io
    from descape.scenario_write import write_scenario

    import conftest

    path = Path(__file__).resolve().parent / "fixtures" / "test_map_240x240.aoe2scenario"
    legacy = conftest.legacy_loaded(path)
    new, report = legacy_convert.load_converted(legacy, rng=random.Random(1))
    assert not new.source_format.legacy and new.terrain_write_supported
    assert report.unmeasured_band is None and not report.dropped_consts

    def placed(doc):
        return [[(u.reference_id, u.unit_const, u.x, u.y, u.garrisoned_in_id) for u in us] for us in doc.unit_manager.units]

    assert placed(new) == placed(legacy)
    counter = int.from_bytes(new.decompressed_body[:4], "little", signed=True)
    highest = max(u.reference_id for us in new.unit_manager.units for u in us)
    assert counter > highest and counter >= int.from_bytes(legacy.decompressed_body[:4], "little", signed=True)

    def cameras(doc):
        entries = doc._scenario.sections["Units"].retriever_map["player_data_3"].data
        return [[e.retriever_map[n].data for n in legacy_convert._PLAYER_DATA_3_CAMERAS] for e in entries]

    bound = new.map_manager.map_width - 1
    assert cameras(new) == [[min(v, bound) if v >= 0 else v for v in row] for row in cameras(legacy)]
    out = tmp_path / "converted.aoe2scenario"
    write_scenario(new, out, backup=False)
    assert placed(scenario_io.load_map_and_units(out)) == placed(legacy)


def test_an_unmeasured_band_is_reported() -> None:
    report = legacy_convert.ConversionReport(unmeasured_band="Age of Kings")
    assert not report.left_out and any("Age of Kings file" in line for line in report.lines())
