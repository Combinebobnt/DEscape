"""scenario_io's two write-guard predicates on their own: the mods/subscribed/
refusal (AGENTS.md hard rule 1) and the game-profile test that forces Save's
backups. The write paths that call them have their own tests."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from descape.scenario_io import is_in_game_profile, is_workshop_subscribed, names_workshop_subscribed


@pytest.mark.parametrize("path, expected", [
    ("/p/Games/Age of Empires 2 DE/1/mods/subscribed/m/x.aoe2scenario", True),
    ("/p/1/Mods/SUBSCRIBED/m", True),
    (PureWindowsPath(r"C:\Users\x\GAMES\age of empires 2 de\1\Mods\Subscribed\m"), True),
    ("/p/mods/local/m", False),
    ("/p/subscribed/mods/m", False),  # the pair, in order
    ("/p/mods_subscribed/m", False),  # whole parts, not a substring
])
def test_names_workshop_subscribed(path, expected) -> None:
    assert names_workshop_subscribed(path) is expected


@pytest.mark.parametrize("path, expected", [
    ("/s/steamapps/compatdata/813780/pfx/x.aoe2scenario", True),
    ("/s/steamapps/CompatData/813780", True),
    (PureWindowsPath(r"C:\Users\x\GAMES\age of empires 2 de\1\resources\_common\scenario"), True),
    ("/p/Games/other game/x", False),
    ("/p/compatdata_backup/x", False),
    ("/p/maps/x.aoe2scenario", False),
])
def test_is_in_game_profile(path, expected) -> None:
    assert is_in_game_profile(Path(path)) is expected


def test_a_symlink_into_either_is_caught_once_resolved(tmp_path) -> None:
    subscribed = tmp_path / "mods" / "subscribed"
    profile = tmp_path / "compatdata" / "813780"
    subscribed.mkdir(parents=True)
    profile.mkdir(parents=True)
    (tmp_path / "a").symlink_to(subscribed, target_is_directory=True)
    (tmp_path / "b").symlink_to(profile, target_is_directory=True)

    assert not names_workshop_subscribed(tmp_path / "a" / "x")
    assert is_workshop_subscribed(tmp_path / "a" / "x")
    assert is_in_game_profile(tmp_path / "b" / "x")


def test_an_unresolvable_path_counts_as_in_the_profile(monkeypatch) -> None:
    """Errs toward backups; is_workshop_subscribed() instead lets the error out."""
    def _raise(self, strict=False):
        raise RuntimeError("Symlink loop")

    monkeypatch.setattr(Path, "resolve", _raise)
    assert is_in_game_profile(Path("/p/maps/x.aoe2scenario"))
    with pytest.raises(RuntimeError):
        is_workshop_subscribed(Path("/p/maps/x.aoe2scenario"))
