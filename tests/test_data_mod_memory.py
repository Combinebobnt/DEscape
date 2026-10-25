"""GH #58: what remembers a data mod, Qt-free. Settings keeps one choice per
saved scenario path, an autosave row keeps the choice its slot was written
under, and data_mods.find() resolves either back to a discovered mod."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import autosave, data_mods, settings


def _mod(name: str, workshop: str | None = None, folder: str = "subscribed") -> data_mods.DataMod:
    root = Path("/mods") / name
    return data_mods.DataMod(f"{folder}//{name}", name.title(), workshop, root, root / "empires2_x2_p1.dat")


def test_a_choice_round_trips_per_resolved_path_and_none_forgets_it(tmp_path):
    a = tmp_path / "a.aoe2scenario"
    alpha = data_mods.identity(_mod("alpha", "11"))
    assert settings.get_data_mod_for(a) is None
    settings.set_data_mod_for(a, alpha)
    assert settings.get_data_mod_for(tmp_path / "x" / ".." / "a.aoe2scenario") == alpha
    settings._data_mod_by_scenario = None  # re-read from disk
    assert settings.get_data_mod_for(a) == alpha
    settings.set_data_mod_for(a, None)
    assert settings.get_data_mod_for(a) is None
    assert "data_mod_by_scenario" not in settings._load_config()


def test_a_malformed_entry_drops_only_itself(tmp_path):
    good = data_mods.identity(_mod("alpha"))
    a, b, c = (tmp_path / name for name in ("a", "b", "c"))
    stored = {settings._scenario_key(a): good, settings._scenario_key(b): {"title": "no path"}, settings._scenario_key(c): "junk"}
    settings.CONFIG_PATH.write_text(yaml.safe_dump({"data_mod_by_scenario": stored}))
    assert settings.get_data_mod_for(a) == good
    assert settings.get_data_mod_for(b) is None and settings.get_data_mod_for(c) is None
    with pytest.raises(ValueError):
        settings.set_data_mod_for(tmp_path / "d", {"title": "no path"})


def test_a_malformed_autosave_row_mod_reads_as_none(tmp_path):
    slot = tmp_path / "a.aoe2scenario.1.autosave"
    slot.write_bytes(b"x")
    autosave.record("k", slot, "a.aoe2scenario", untitled=False, data_mod={"title": "no path"})
    assert autosave.entries()[0].data_mod is None


def test_an_autosave_row_carries_its_mod(tmp_path):
    slot = tmp_path / "a.aoe2scenario.1.autosave"
    slot.write_bytes(b"x")
    alpha = data_mods.identity(_mod("alpha"))
    autosave.record("k", slot, "a.aoe2scenario", untitled=False, data_mod=alpha)
    other = tmp_path / "b.aoe2scenario.1.autosave"
    other.write_bytes(b"y")
    autosave.record("j", other, "b.aoe2scenario", untitled=False)
    by_key = {entry.key: entry.data_mod for entry in autosave.entries()}
    assert by_key == {"k": alpha, "j": None}


def test_find_matches_the_status_path_then_the_workshop_id():
    subscribed, local, other = _mod("alpha", "11"), _mod("alpha", "11", folder="local"), _mod("beta", "22")
    assert data_mods.find(data_mods.identity(subscribed), [other, local, subscribed]) is subscribed
    assert data_mods.find(data_mods.identity(subscribed), [other, local]) is local, "moved between mod folders"
    assert data_mods.find(data_mods.identity(_mod("gone")), [other, local]) is None
