"""tools/_game_root.py's classic layer (TASK-515): the 1990s installs' .dat and
DRS archives found case-insensitively along ordered art folders, the VER 3.7
.dat routed to tools/_dat37.py, the aoe1 terrain texture fallback, and the
still-unregistered games.AOE1 descriptor agreeing with the generators.

The default tier packs its own archives; the corpus tier reruns every
generator against the Age of Empires install `--aoe1-dir` names and compares
each table with descape/aoe1/ byte for byte."""

from __future__ import annotations

import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_drs import pack_drs

from descape import dat_derive, games
from tools import _game_root

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _tools_on_path(monkeypatch):
    """_game_root imports its sibling readers (_drs, _dat37) lazily, as the generators run it."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))


def test_find_path_matches_each_part_case_insensitively_and_an_exact_one_first(tmp_path):
    (tmp_path / "Data2").mkdir()
    (tmp_path / "Data2" / "Empires.dat").write_bytes(b"")
    assert _game_root.find_path(tmp_path, "data2/empires.dat") == tmp_path / "Data2" / "Empires.dat"
    (tmp_path / "Data2" / "empires.dat").write_bytes(b"")
    assert _game_root.find_path(tmp_path, "data2/empires.dat") == tmp_path / "Data2" / "empires.dat", "exact wins"
    assert _game_root.find_path(tmp_path, "data/empires.dat") is None
    assert _game_root.find_path(tmp_path, "data2/empires.dat/more") is None, "a file is no folder"


def test_the_aoe1_dat_is_found_case_insensitively_else_its_canonical_path(tmp_path):
    canonical = tmp_path / "data2" / "empires.dat"
    assert _game_root.dat_path(tmp_path, _game_root.AOE1) == canonical, "absent: the canonical path, for the error"
    (tmp_path / "DATA2").mkdir()
    (tmp_path / "DATA2" / "Empires.DAT").write_bytes(b"")
    assert _game_root.dat_path(tmp_path, _game_root.AOE1) == tmp_path / "DATA2" / "Empires.DAT"
    assert _game_root.dat_path(tmp_path, _game_root.AOE1, tmp_path / "x.dat") == tmp_path / "x.dat"


def test_classic_archives_layer_rise_of_romes_folder_over_the_base_games(tmp_path):
    (tmp_path / "Data").mkdir()
    (tmp_path / "data2").mkdir()
    pack_drs(tmp_path / "Data" / "Graphics.drs", {b" pls": {1: b"base", 2: b"base only"}})
    pack_drs(tmp_path / "data2" / "graphics.drs", {b" pls": {1: b"rise of rome"}})
    pack_drs(tmp_path / "Data" / "Interfac.drs", {b"anib": {50500: b"palette"}})
    graphics = _game_root.classic_graphics(tmp_path, _game_root.AOE1)
    assert (graphics.read("slp", 1), graphics.read("slp", 2)) == (b"rise of rome", b"base only")
    both = _game_root.classic_archives(tmp_path, _game_root.AOE1, "interfac.drs", "graphics.drs")
    assert [archive.path for archive in both.archives] == [
        tmp_path / "data2" / "graphics.drs", tmp_path / "Data" / "Interfac.drs", tmp_path / "Data" / "Graphics.drs",
    ], "folder order first, then name order within a folder"
    assert both.read("bina", 50500) == b"palette", "the palette lives only in data/"
    with pytest.raises(FileNotFoundError, match=r"terrain\.drs"):
        _game_root.classic_archives(tmp_path, _game_root.AOE1, "terrain.drs")


def test_parse_dat_sends_a_1997_dat_to_dat37(tmp_path, monkeypatch):
    packer = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    path = tmp_path / "empires.dat"
    path.write_bytes(packer.compress(b"VER 3.7\0" + b"\0" * 8000) + packer.flush())
    monkeypatch.setitem(sys.modules, "_dat37", NS(parse=lambda p: ("dat37", Path(p).name)))
    assert _game_root.dat_version(path) == _game_root.DAT37_VERSION
    assert _game_root.parse_dat(path) == ("dat37", "empires.dat")


def _tool(name: str):
    import importlib

    return importlib.import_module(name)


def test_an_aoe1_terrain_without_an_slp_draws_its_terrain_to_draws():
    gen = _tool("gen_terrain_texture_map")
    terrains = [
        NS(enabled=True, name="Grass", slp=15000, terrain_to_draw=-1),
        NS(enabled=True, name="Forest", slp=-1, terrain_to_draw=0),
        NS(enabled=True, name="Junk", slp=-1, terrain_to_draw=99),
        NS(enabled=False, name="Off", slp=-1, terrain_to_draw=0),
        NS(enabled=True, name="Chain", slp=-1, terrain_to_draw=1),
        NS(enabled=True, name="Lost", slp=15009, terrain_to_draw=-1),
    ]
    through = gen._classic_texture_mapping(terrains, {15000}, through=True)
    assert through == ({"0": "slp_15000.png", "1": "slp_15000.png"}, [(5, "Lost", "slp_15009")]), (
        "one step only: a terrain drawing through one that has no SLP of its own gets none"
    )
    assert gen._classic_texture_mapping(terrains, {15000}) == ({"0": "slp_15000.png"}, [(5, "Lost", "slp_15009")])
    assert set(gen._DRAWS_THROUGH) == {_game_root.AOE1}, "AoC's tables stay as they are until TASK-430.04"


def test_the_aoe1_ship_check_passes_no_beach_and_skips_water_impass():
    gen = _tool("gen_terrain_classes")
    families = {0: "land", 1: "deep_water", 2: "beach", 4: "shallows", 15: "deep_water"}
    classes = {str(terrain_id): {"family": family} for terrain_id, family in families.items()}

    def dat(passed: set[int]):
        row = [1.0 if terrain_id in passed else 0.0 for terrain_id in range(23)]
        return NS(terrain_restrictions=[None, None, None, NS(passable_buildable_dmg_multiplier=row)])

    gen._ship_check(_game_root.AOE1, dat({1, 4}), classes)
    for passed in ({1, 2, 4}, {1, 4, 15}, {1}):
        with pytest.raises(SystemExit, match="the aoe1 ship restriction passes"):
            gen._ship_check(_game_root.AOE1, dat(passed), classes)
    gen._ship_check(_game_root.AOC, dat({1, 2, 4, 15}), classes)


def test_the_aoe1_water_bytes_need_their_measured_names_at_their_ids():
    gen = _tool("gen_terrain_classes")
    terrains = [NS(enabled=True, name=name) for name in gen._AOE1_TERRAIN_NAMES]
    gen._aoe1_name_check([*terrains, NS(enabled=False, name="unused")])
    shifted = [NS(enabled=False, name=""), *terrains[:5], *terrains[6:]]
    with pytest.raises(SystemExit, match="not those _AOE1_IS_WATER was measured on"):
        gen._aoe1_name_check(shifted)


def test_the_1997_dats_missing_fields_read_as_every_civs_standard_and_no_hero():
    catalog, render = _tool("gen_object_catalog"), _tool("gen_unit_render_data")
    assert catalog._for_civ(NS(civ=None), 3) and catalog._for_civ(NS(civ=-1), 3) and not catalog._for_civ(NS(civ=2), 3)
    assert catalog._in_standard_game(NS(full_tech_mode=None))
    assert catalog._commands(NS(effects=[NS(effect_commands=["x"])]), None) == []
    assert not render._is_glowing_hero(NS(creatable=NS(hero_mode=None, hero_glow_graphic=12218)))


def test_the_unregistered_aoe1_descriptor_agrees_with_the_generators(tmp_path):
    """C registers games.AOE1: its tables, .dat and stand-in are already what the generators and registry expect."""
    game = games.AOE1
    assert game.id == _game_root.AOE1 and game not in games.GAMES and game.id in _game_root.PENDING
    assert tuple(map(Path, game.dat_subpaths)) == _game_root.dat_subpaths(game.id)
    assert _game_root.game_root(tmp_path, game.id) == game.root(tmp_path)
    assert [name for name in dat_derive.TABLE_NAMES if not (game.tables_dir / name).is_file()] == []
    assert game.stand_in == games.ROR.id and games.ROR.stand_in is None
    assert not game.writable and game.art in games.IMPLEMENTED_ART and game.reader in games.IMPLEMENTED_READERS
    assert game.install not in games.IMPLEMENTED_INSTALLS, "C implements its install kind, then registers it"
    assert game.stamp_key not in {other.stamp_key for other in games.GAMES}


@pytest.mark.corpus
def test_every_generator_reproduces_the_committed_aoe1_tables(request, tmp_path):
    install = request.config.getoption("--aoe1-dir")
    if not install:
        pytest.skip("no Age of Empires install: pass --aoe1-dir")
    for script, _names in dat_derive.GENERATORS:
        run = subprocess.run(
            [sys.executable, str(ROOT / "tools" / script), install, "--dataset", _game_root.AOE1, "--out-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=600, check=False, cwd=ROOT,
        )
        assert run.returncode == 0, (script, run.stderr[-2000:])
    differ = [name for name in dat_derive.TABLE_NAMES if (tmp_path / name).read_bytes() != (games.AOE1.tables_dir / name).read_bytes()]
    assert differ == []
