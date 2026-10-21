"""The write guards on real Windows path strings: a drive letter, backslashes
and the case Windows ignores. test_write_guard_paths.py covers the same
predicates with PureWindowsPath on every OS; here the strings go through the
native Path, resolve() included, so this only runs on Windows."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH, is_in_game_profile, is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows path strings need the native Windows Path")

_PROFILE = r"C:\Users\x\Games\Age of Empires 2 DE\76561198000000000"


@pytest.mark.parametrize("path, expected", [
    (_PROFILE + r"\resources\_common\scenario\a.aoe2scenario", True),
    (r"C:\Users\x\games\age of empires 2 de\76561198000000000\resources\_common\scenario\a.aoe2scenario", True),
    ("C:/Users/x/Games/Age of Empires 2 DE/76561198000000000/resources/_common/scenario/a.aoe2scenario", True),
    (r"C:\Users\x\Games\Other\a.aoe2scenario", False),
    (r"C:\Users\x\Age of Empires 2 DE\a.aoe2scenario", False),  # without Games
    (r"D:\maps\a.aoe2scenario", False),
])
def test_is_in_game_profile(path, expected) -> None:
    assert Path(path).drive.upper() in {"C:", "D:"}  # parsed as a native Windows path
    assert is_in_game_profile(path) is expected


@pytest.mark.parametrize("path, expected", [
    (_PROFILE + r"\mods\subscribed\1234_map\a.aoe2scenario", True),
    (_PROFILE + r"\MODS\Subscribed\1234_map\a.aoe2scenario", True),
    ("C:/Users/x/Games/Age of Empires 2 DE/76561198000000000/mods/subscribed/1234_map/a.aoe2scenario", True),
    (r"C:\Users\x\subscribed\a.aoe2scenario", False),  # subscribed without mods
    (r"C:\Users\x\subscribed\mods\a.aoe2scenario", False),  # the pair, in order
    (_PROFILE + r"\mods\local\a.aoe2scenario", False),
])
def test_is_workshop_subscribed(path, expected) -> None:
    assert Path(path).drive.upper() == "C:"
    assert is_workshop_subscribed(path) is expected


@pytest.mark.parametrize("tail", [r"mods\subscribed", r"MODS\Subscribed", "mods/subscribed"])
def test_write_scenario_refuses_a_windows_subscribed_path(tmp_path, tail) -> None:
    subscribed = tmp_path / "mods" / "subscribed" / "1234_map"
    subscribed.mkdir(parents=True)
    target = str(tmp_path) + "\\" + tail + r"\1234_map\a.aoe2scenario"
    s = load_map_and_units(BLANK_TEMPLATE_PATH)
    tile = s.map_manager.terrain[0]
    tile.terrain_id = 15 if tile.terrain_id == 2 else 2  # an unrefused write would write
    with pytest.raises(WriteBlockedError, match="subscribed"):
        write_scenario(s, target)
    assert list(subscribed.iterdir()) == []
