"""GH #372: tools/gen_object_catalog.py's sparse `civ_icons` key, from
stand-ins shaped like genieutils' units. Install-free."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


@pytest.fixture()
def gen(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module("gen_object_catalog")


def unit(icon: int, enabled: bool = True) -> NS:
    return NS(
        name="MONKX", language_dll_name=5000, class_=18, type=70, hide_in_editor=0, icon_id=icon,
        creatable=None, standing_graphic=(10, -1), resource_storages=(), obstruction_type=0,
        combat_level=0, terrain_restriction=0, clearance_size=(0, 0), placement_side_terrain=(),
        enabled=enabled,
    )


def civs(*lists) -> list[NS]:
    return [NS(units=units) for units in lists]


def test_only_a_civ_whose_icon_differs_gets_an_override(gen):
    gaia = [unit(33), unit(17)]
    table = civs(gaia, [unit(33), unit(17)], [unit(218, enabled=False), unit(17)], [unit(292), unit(613)])
    entries = gen._object_entries(gaia, frozenset(), table)
    assert entries["0"]["civ_icons"] == {"2": 218, "3": 292}, "a disabled copy counts"
    assert entries["1"]["civ_icons"] == {"3": 613}
    assert entries["0"]["icon"] == 33, "GAIA's icon stays the row's own"


def test_no_override_leaves_the_key_out(gen):
    gaia = [unit(33)]
    entries = gen._object_entries(gaia, frozenset(), civs(gaia, [unit(33)]))
    assert "civ_icons" not in entries["0"]
    assert "civ_icons" not in gen._object_entries(gaia, frozenset())["0"], "no civs passed"


def test_a_missing_or_short_civ_copy_is_skipped(gen):
    gaia = [unit(33), unit(17)]
    entries = gen._object_entries(gaia, frozenset(), civs(gaia, [None, unit(18)], [unit(34)]))
    assert entries["0"]["civ_icons"] == {"2": 34}
    assert entries["1"]["civ_icons"] == {"1": 18}, "civ 2's list stops before const 1"


def test_an_override_of_minus_one_is_kept(gen):
    gaia = [unit(33)]
    entries = gen._object_entries(gaia, frozenset(), civs(gaia, [unit(-1)]))
    assert entries["0"]["civ_icons"] == {"1": -1}
