"""GH #262: the lenient derive's helpers (tools/_game_root.py) and the small
generators' fallbacks. A data mod's .dat may break an invariant measured on
the stock .dat; lenient, the guard degrades only what it guards and records a
warning, strict it fails as before. Install-free: stand-ins shaped like
genieutils' objects, called at function level."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


@pytest.fixture()
def tools(monkeypatch):
    """Imports a tools/ module by name."""
    monkeypatch.syspath_prepend(str(TOOLS))
    return importlib.import_module


@pytest.fixture()
def lenient():
    """Makes a module's _game_root lenient for the test, returning its WARNINGS; strict and empty again after."""
    roots = []

    def on(module):
        root = getattr(module, "_game_root", module)
        root.WARNINGS.clear()
        root.set_lenient(True)
        roots.append(root)
        return root.WARNINGS

    yield on
    for root in roots:
        root.set_lenient(False)
        root.WARNINGS.clear()


def test_refuse_raises_when_strict_and_records_once_when_lenient(tools, lenient):
    root = tools("_game_root")
    root.set_lenient(False)
    with pytest.raises(SystemExit) as raised:
        root.refuse("const 45 broke a rule", "drawn alone")
    assert raised.value.code == "const 45 broke a rule", "the strict message is unchanged"
    warnings = lenient(root)
    assert root.refuse("const 45 broke a rule", "drawn alone") is None
    root.refuse("const 45 broke a rule", "drawn alone")
    root.refuse("const 46 broke a rule", "left out")
    assert warnings == ["const 45 broke a rule (drawn alone)", "const 46 broke a rule (left out)"]


def test_a_strict_run_after_a_lenient_one_raises_again(tools, lenient):
    root = tools("_game_root")
    lenient(root)
    root.refuse("a", "b")
    root.set_lenient(False)
    assert not root.lenient()
    with pytest.raises(SystemExit):
        root.refuse("a", "b")


def test_the_dataset_arguments_carry_the_lenient_flag(tools):
    root = tools("_game_root")
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    root.add_dataset_argument(parser)
    assert parser.parse_args(["install"]).lenient is False
    assert parser.parse_args(["install", "--lenient"]).lenient is True


def _terrain(priority: int, blend_type: int, name: str, enabled: bool = True):
    return NS(enabled=enabled, blend_priority=priority, blend_type=blend_type, name=name, overlay_mask_name=None)


TERRAINS = [
    _terrain(5, 1, "A"),
    _terrain(5, 2, "B"),  # A's priority
    _terrain(7, 99, "C"),  # no such blend type
    _terrain(9, 0, "off", enabled=False),
    _terrain(8, 0, "D"),
]


def test_the_blend_table_keeps_every_terrain_and_reports_the_violations_when_strict(tools):
    gen = tools("gen_terrain_blend_map")
    gen._game_root.set_lenient(False)
    table, violations = gen._blend_table(TERRAINS)
    assert sorted(table) == ["0", "1", "2", "4"]
    assert [line.split(":")[0].strip() for line in violations] == ["id=1 name='B'", "id=2 name='C'"]


def test_a_lenient_blend_table_drops_each_offending_terrain(tools, lenient):
    gen = tools("gen_terrain_blend_map")
    warnings = lenient(gen)
    table, violations = gen._blend_table(TERRAINS)
    assert violations == []
    assert sorted(table) == ["0", "4"], "the later of two equal priorities is dropped"
    assert len(warnings) == 2 and "id=1 name='B'" in warnings[0] and "already used by id=0" in warnings[0]


def test_a_restriction_without_a_dat_row_is_left_out_when_lenient(tools, lenient):
    gen = tools("gen_object_catalog")
    restrictions = [NS(passable_buildable_dmg_multiplier=[1, 0]) for _ in range(3)]
    objects = {"5": {"restriction": 1}, "6": {"restriction": 30}}
    terrains = {"0": {}, "1": {}}
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match=r"Terrain restriction 30 has no \.dat row"):
        gen._terrain_restriction_entries(restrictions, objects, terrains)
    warnings = lenient(gen)
    assert gen._terrain_restriction_entries(restrictions, objects, terrains) == {"1": [0], "2": [0]}
    assert len(warnings) == 1 and warnings[0].startswith("Terrain restriction 30 has no .dat row")


@pytest.mark.parametrize(
    "span, unit",
    [
        ("_fish_span", NS(outline_size_x=1.0, outline_size_y=1.0, clearance_size=(2.0, 2.0))),
        ("_doodad_span", NS(collision_size_x=3.0, collision_size_y=3.0, outline_size_x=1.5, outline_size_y=1.5)),
    ],
    ids=["fish", "doodad"],
)
def test_a_span_that_breaks_its_rule_gets_no_row_when_lenient(tools, lenient, span, unit):
    gen = tools("gen_unit_render_data")
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="re-"):
        getattr(gen, span)(3118, unit)
    warnings = lenient(gen)
    assert getattr(gen, span)(3118, unit) is None
    assert len(warnings) == 1 and "3118" in warnings[0]
