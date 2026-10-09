"""descape.find_objects: Edit > Find and Replace's Qt-free object search (GH #144).

Default tier, no QApplication. Synthetic SimpleNamespace units for the
facets, plus tests/fixtures/units_120x120.aoe2scenario for a real load."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import find_objects, object_catalog
from descape.find_objects import FindCriteria, FindPatternError
from descape.scenario_io import load_map_and_units
from descape.scenario_write import WriteBlockedError
from descape.terrain_palette import TREE_UNIT_IDS

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_ARCHER = 4
_VILLAGER = 83
_HOUSE = 70
_OAK = 349
_STONE_WALL = 117


def _unit(ref, const, x, y, *, host=-1, caption="", z=0.0):
    return SimpleNamespace(
        reference_id=ref, unit_const=const, x=x, y=y, z=z, garrisoned_in_id=host, caption_string=caption
    )


def _loaded(players, width=40, height=40):
    lists = [list(players.get(p, ())) for p in range(9)]
    return SimpleNamespace(
        unit_manager=SimpleNamespace(units=lists),
        map_manager=SimpleNamespace(map_width=width, map_height=height),
    )


def _scene():
    return _loaded(
        {
            0: [_unit(1, _OAK, 2.5, 2.5), _unit(2, _STONE_WALL, 3.5, 2.5)],
            1: [
                _unit(10, _HOUSE, 10.0, 10.0),
                _unit(11, _ARCHER, 12.5, 12.5, caption="Hero archer"),
                _unit(12, _VILLAGER, 10.0, 10.0, host=10),
                # Self-reference: legal on disk, reads as not garrisoned.
                _unit(13, _VILLAGER, 15.5, 15.5, host=13),
            ],
            2: [_unit(20, _ARCHER, 50.5, 5.5), _unit(21, _ARCHER, 30.5, 30.5)],
            3: [_unit(21, _VILLAGER, 33.5, 33.5)],
        }
    )


def _refs(rows):
    return [r.reference_id for r in rows]


def _find(loaded, **kwargs):
    mm = loaded.map_manager
    return find_objects.find(loaded, FindCriteria(**kwargs), mm.map_width, mm.map_height)


def test_empty_criteria_returns_every_unit_in_file_order() -> None:
    assert _refs(_find(_scene())) == [1, 2, 10, 11, 12, 13, 20, 21, 21]


def test_substring_matches_the_display_name_case_insensitively_by_default() -> None:
    archer = object_catalog.display_name(_ARCHER)
    rows = _find(_scene(), text=archer.lower())
    assert {r.unit_const for r in rows} == {_ARCHER}
    assert archer != archer.lower()
    assert _find(_scene(), text=archer.lower(), case_sensitive=True) == []
    assert {r.unit_const for r in _find(_scene(), text=archer, case_sensitive=True)} == {_ARCHER}


def test_substring_mode_escapes_regex_metacharacters() -> None:
    loaded = _scene()
    assert _find(loaded, text=".*") == []
    assert len(_find(loaded, text=".*", regex=True)) == 9


def test_regex_mode_and_invalid_regex() -> None:
    archer = object_catalog.display_name(_ARCHER)
    rows = _find(_scene(), text=f"^{archer[:3]}", regex=True)
    assert _ARCHER in {r.unit_const for r in rows}
    with pytest.raises(FindPatternError):
        _find(_scene(), text="(unclosed", regex=True)
    # The same text in substring mode is just a literal.
    assert _find(_scene(), text="(unclosed") == []


def test_hash_id_matches_unit_const_exactly() -> None:
    assert {r.unit_const for r in _find(_scene(), text="#83")} == {_VILLAGER}
    assert {r.unit_const for r in _find(_scene(), text="83")} == {_VILLAGER}
    # Exact, not a prefix: #8 names no const here.
    assert _find(_scene(), text="#8") == []


def test_captions_match_only_when_asked() -> None:
    assert _find(_scene(), text="hero archer") == []
    assert _refs(_find(_scene(), text="hero archer", match_captions=True)) == [11]


def test_player_category_kind_and_const_facets() -> None:
    loaded = _scene()
    assert _refs(_find(loaded, players=frozenset({0}))) == [1, 2]
    assert _refs(_find(loaded, players=frozenset({2, 3}))) == [20, 21, 21]
    assert _refs(_find(loaded, consts=frozenset({_VILLAGER}))) == [12, 13, 21]
    assert _refs(_find(loaded, kinds=frozenset({"trees"}))) == [1]
    assert _refs(_find(loaded, kinds=frozenset({"walls"}))) == [2]
    assert _refs(_find(loaded, kinds=frozenset({"trees", "walls"}))) == [1, 2]
    assert _find(loaded, kinds=frozenset()) == []
    assert find_objects.category_of(_HOUSE) == "Buildings"
    assert find_objects.category_of(_ARCHER) == "Units"
    buildings = _find(loaded, categories=frozenset({"Buildings"}))
    assert all(find_objects.category_of(r.unit_const) == "Buildings" for r in buildings)
    assert 10 in _refs(buildings) and 11 not in _refs(buildings)
    assert _OAK in TREE_UNIT_IDS


def test_area_matches_by_clamped_footprint_intersection() -> None:
    loaded = _scene()
    # A House (2x2 at 10.0, 10.0) covers tiles 9..10: a rect touching only one of them catches it.
    assert 10 in _refs(_find(loaded, area=(10, 10, 11, 11)))
    assert 10 in _refs(_find(loaded, area=(9, 9, 10, 10)))
    # Half-open: (8, 8, 9, 9) is tile 8 alone, and (11, 11, 12, 12) tile 11 alone.
    assert 10 not in _refs(_find(loaded, area=(8, 8, 9, 9)))
    assert 10 not in _refs(_find(loaded, area=(11, 11, 12, 12)))
    # The archer at 12.5 is on tile 12 only.
    assert 11 in _refs(_find(loaded, area=(12, 12, 13, 13)))
    assert 11 not in _refs(_find(loaded, area=(0, 0, 12, 12)))
    # Off-map units never match an area.
    assert 20 not in _refs(_find(loaded, area=(0, 0, 40, 40)))


def test_garrisoned_rule_and_self_reference() -> None:
    loaded = _scene()
    rows = {r.reference_id: r for r in _find(loaded)}
    assert rows[12].garrisoned_in == 10
    assert rows[13].garrisoned_in == -1
    assert rows[10].occupants == 1
    assert rows[13].occupants == 0
    without = _refs(_find(loaded, include_garrisoned=False))
    assert 12 not in without and 13 in without


def test_off_map_rows_are_flagged_and_can_be_excluded() -> None:
    loaded = _scene()
    rows = {r.reference_id: r for r in _find(loaded)}
    assert rows[20].off_map is True
    assert rows[11].off_map is False
    assert 20 not in _refs(_find(loaded, include_off_map=False))


def test_duplicate_reference_ids_are_flagged() -> None:
    rows = _find(_scene())
    dupes = [(r.player, r.reference_id) for r in rows if r.duplicate_ref]
    assert dupes == [(2, 21), (3, 21)]
    assert all(r.trigger_refs is None for r in rows)


def test_trigger_ref_counts_fill_the_column() -> None:
    loaded = _scene()
    mm = loaded.map_manager
    rows = find_objects.find(loaded, FindCriteria(), mm.map_width, mm.map_height, trigger_ref_counts={11: 3})
    by_ref = {r.key: r.trigger_refs for r in rows}
    assert by_ref[(1, 11)] == 3
    assert by_ref[(1, 10)] == 0


def test_resolve_keys_reaches_garrisoned_and_off_map_units_and_drops_ambiguous() -> None:
    loaded = _scene()
    loaded.unit_manager.units[4].extend([_unit(40, _ARCHER, 1.5, 1.5), _unit(40, _ARCHER, 2.5, 1.5)])
    resolved = find_objects.resolve_keys(loaded, [(1, 12), (2, 20), (9, 9), (4, 40), (3, 21)])
    assert [(p, u.reference_id) for p, u in resolved] == [(1, 12), (2, 20), (3, 21)]


def test_real_fixture_load() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    mm = loaded.map_manager
    rows = find_objects.find(loaded, FindCriteria(), mm.map_width, mm.map_height)
    assert len(rows) == 8
    villager = next(r for r in rows if r.reference_id == 203)
    assert villager.garrisoned_in == 200
    assert next(r for r in rows if r.reference_id == 200).occupants == 1
    captioned = find_objects.find(
        loaded, FindCriteria(text="fixture caption", match_captions=True), mm.map_width, mm.map_height
    )
    assert [r.reference_id for r in captioned] == [300]


def test_csv_round_trip(tmp_path) -> None:
    rows = _find(_scene())
    out = tmp_path / "found.csv"
    assert find_objects.rows_to_csv(rows, out) == len(rows)
    with out.open(newline="", encoding="utf-8") as handle:
        read = list(csv.reader(handle))
    assert tuple(read[0]) == find_objects.CSV_COLUMNS
    assert len(read) == len(rows) + 1
    archer = next(line for line in read[1:] if line[1] == "11")
    assert archer[0] == object_catalog.display_name(_ARCHER)
    assert archer[7] == "Hero archer"
    assert archer[12] == ""


def test_csv_refuses_compatdata(tmp_path) -> None:
    target = tmp_path / "compatdata" / "found.csv"
    target.parent.mkdir()
    with pytest.raises(WriteBlockedError):
        find_objects.rows_to_csv(_find(_scene()), target)
    assert not target.exists()


def test_csv_refuses_a_symlink_into_compatdata(tmp_path) -> None:
    real = tmp_path / "compatdata"
    real.mkdir()
    link = tmp_path / "exports"
    link.symlink_to(real)
    with pytest.raises(WriteBlockedError):
        find_objects.rows_to_csv(_find(_scene()), link / "found.csv")
    assert not (real / "found.csv").exists()
