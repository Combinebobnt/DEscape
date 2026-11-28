"""tools/_game_root.py's HD Edition dataset (TASK-431): its texture stems by
name_2, its ship check, its loose SLP folders in their lookup order, and the
games.HD descriptor agreeing with the generators.

The corpus tier reruns every generator against the HD Edition install
HD_INSTALL_PATH names and compares each table with descape/hd/ byte for byte."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from descape import dat_derive, games
from tools import _game_root

ROOT = Path(__file__).resolve().parent.parent
# Read at import: conftest unsets HD_INSTALL_PATH inside every test.
HD_INSTALL = os.environ.get("HD_INSTALL_PATH")


@pytest.fixture(autouse=True)
def _tools_on_path(monkeypatch):
    """_game_root imports _drs lazily, as the generators run it."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))


def _tool(name: str):
    import importlib

    return importlib.import_module(name)


def test_an_hd_terrain_draws_its_name_2_else_its_terrain_to_draws_one_step():
    gen = _tool("gen_terrain_texture_map")
    terrains = [
        NS(enabled=True, name="Grass", name_2="g_grs", terrain_to_draw=-1),
        NS(enabled=True, name="Forest", name_2="None", terrain_to_draw=0),
        NS(enabled=False, name="Off", name_2="g_off", terrain_to_draw=-1),
        NS(enabled=True, name="Lost", name_2="g_lost", terrain_to_draw=-1),
        NS(enabled=True, name="Chain", name_2="None", terrain_to_draw=1),
        NS(enabled=True, name="Upper", name_2="G_UP", terrain_to_draw=-1),
        NS(enabled=True, name="Junk", name_2="None", terrain_to_draw=99),
    ]
    mapping, missing = gen._hd_texture_mapping(terrains, {"g_grs", "g_up", "g_off"})
    assert mapping == {"0": "g_grs_00_color.png", "1": "g_grs_00_color.png", "5": "G_UP_00_color.png"}, (
        "one step only, a disabled terrain skipped, the stem matched case-insensitively"
    )
    assert missing == [(3, "Lost", "g_lost")]


def test_the_hd_ship_check_passes_its_beaches():
    gen = _tool("gen_terrain_classes")
    families = {0: "land", 1: "deep_water", 2: "beach", 4: "shallows", 23: "medium_water"}
    classes = {str(terrain_id): {"family": family} for terrain_id, family in families.items()}

    def dat(passed: set[int]):
        row = [1.0 if terrain_id in passed else 0.0 for terrain_id in range(100)]
        return NS(terrain_restrictions=[None, None, None, NS(passable_buildable_dmg_multiplier=row)])

    gen._ship_check(_game_root.HD, dat({1, 2, 4, 23}), classes)
    for passed in ({1, 4, 23}, {0, 1, 2, 4, 23}):
        with pytest.raises(SystemExit, match="the hd ship restriction passes"):
            gen._ship_check(_game_root.HD, dat(passed), classes)


def test_hd_slp_art_takes_the_expansions_over_the_base_game_and_needs_one_slp(tmp_path):
    drs_dir = tmp_path / _game_root.HD_DRS_SUBPATH
    for folder, files in (("graphics", {"8.slp": b"base", "305.slp": b"base"}), ("gamedata_x1", {"8.slp": b"x1"}),
                          ("gamedata_x2", {"8.slp": b"x2", "6000.slp": b"x2"})):
        (drs_dir / folder).mkdir(parents=True)
        for name, data in files.items():
            (drs_dir / folder / name).write_bytes(data)
    art = _game_root.slp_art(tmp_path, _game_root.HD)
    assert (art.read("slp", 8), art.read("slp", 305), art.ids("slp")) == (b"x2", b"base", [8, 305, 6000])
    with pytest.raises(FileNotFoundError, match=r"no \.slp"):
        _game_root.slp_art(tmp_path / "empty", _game_root.HD)
    with pytest.raises(ValueError, match="no SLP art"):
        _game_root.slp_art(tmp_path, _game_root.AOE2)


def test_the_hd_palette_is_its_loose_jasc_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _game_root.slp_palette(tmp_path, _game_root.HD)
    path = tmp_path / _game_root.HD_PALETTE_SUBPATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"JASC-PAL\r\n0100\r\n2\r\n0 0 0\r\n255 128 1\r\n")
    assert _game_root.slp_palette(tmp_path, _game_root.HD)[:2] == [(0, 0, 0), (255, 128, 1)]


def test_the_hd_descriptor_agrees_with_the_generators(tmp_path):
    """games.HD's tables, .dat and stand-in are what the generators and registry expect."""
    game = games.HD
    assert game.id == _game_root.HD and game in games.GAMES and game.id in _game_root.DATASETS
    assert tuple(map(Path, game.dat_subpaths)) == _game_root.dat_subpaths(game.id)
    assert _game_root.game_root(tmp_path, game.id) == game.root(tmp_path)
    assert [name for name in dat_derive.TABLE_NAMES if not (game.tables_dir / name).is_file()] == []
    assert game.stand_in == games.AOE2.id and not game.writable
    assert game.art in games.IMPLEMENTED_ART and game.reader in games.IMPLEMENTED_READERS
    assert game.install in games.IMPLEMENTED_INSTALLS
    assert game.stamp_key not in {other.stamp_key for other in games.GAMES if other is not game}
    assert not game.formats & {fmt for other in games.GAMES if other is not game for fmt in other.formats}


@pytest.mark.corpus
def test_every_generator_reproduces_the_committed_hd_tables(tmp_path):
    if not HD_INSTALL:
        pytest.skip("no HD Edition install: set HD_INSTALL_PATH to its Age2HD folder")
    for script, _names in dat_derive.GENERATORS:
        run = subprocess.run(
            [sys.executable, str(ROOT / "tools" / script), HD_INSTALL, "--dataset", _game_root.HD, "--out-dir", str(tmp_path)],
            capture_output=True, text=True, timeout=600, check=False, cwd=ROOT,
        )
        assert run.returncode == 0, (script, run.stderr[-2000:])
    differ = [name for name in dat_derive.TABLE_NAMES if (tmp_path / name).read_bytes() != (games.HD.tables_dir / name).read_bytes()]
    assert differ == []
