"""GH #195: settings.get/set_unit_favorites() (a list of object ids per game) and
get/set_units_favorites_mode(), the Favorite Objects window's view.

Game ids come from games.ids(), so no test depends on which game is listed first."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import games, settings, settings_file

_ONE, _OTHER = games.ids()[:2]


def _reread(monkeypatch) -> None:
    monkeypatch.setattr(settings, "_unit_favorites", None)


def _stored(tmp_path: Path) -> dict:
    return yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}


def test_favorites_are_empty_until_set_and_round_trip_per_game_in_order(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_unit_favorites(_ONE) == []
    settings.set_unit_favorites(_ONE, [83, 4, -1])
    settings.set_unit_favorites(_OTHER, [93])
    _reread(monkeypatch)
    assert settings.get_unit_favorites(_ONE) == [83, 4, -1]
    assert settings.get_unit_favorites(_OTHER) == [93]
    assert _stored(tmp_path)["units_favorites"] == {_ONE: [83, 4, -1], _OTHER: [93]}


@pytest.mark.parametrize(
    "given, kept",
    [
        ([4, "4", 4.0, None, [4], 83], [4, 83]),
        ([True, False, 4], [4]),
        ([-100, -99, 32767, 32768, 70000], [-99, 32767]),
        ([4, 83, 4, 83, 12], [4, 83, 12]),
    ],
    ids=["non-ints", "bools", "range", "duplicates"],
)
def test_the_setter_keeps_only_valid_ids_once(given, kept, monkeypatch) -> None:
    settings.set_unit_favorites(_ONE, given)
    assert settings.get_unit_favorites(_ONE) == kept
    _reread(monkeypatch)
    assert settings.get_unit_favorites(_ONE) == kept


def test_a_malformed_file_entry_drops_only_itself(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"units_favorites:\n  {_ONE}: [4, bogus, 83, 99999, true, 4]\n  {_OTHER}: not a list\n  7: [12]\n"
    )
    assert settings.get_unit_favorites(_ONE) == [4, 83]
    assert settings.get_unit_favorites(_OTHER) == []


@pytest.mark.parametrize("raw", ["units_favorites: [4, 83]\n", "units_favorites: 12\n", "units_favorites: nope\n"])
def test_a_favorites_key_that_is_not_a_mapping_reads_as_none(tmp_path: Path, raw) -> None:
    (tmp_path / "config.yaml").write_text(raw)
    assert settings.get_unit_favorites(_ONE) == []


def test_the_getter_returns_a_copy() -> None:
    settings.set_unit_favorites(_ONE, [4, 83])
    got = settings.get_unit_favorites(_ONE)
    got.append(12)
    assert settings.get_unit_favorites(_ONE) == [4, 83]


def test_setting_one_game_keeps_the_others(monkeypatch) -> None:
    settings.set_unit_favorites(_ONE, [4])
    settings.set_unit_favorites(_OTHER, [93])
    settings.set_unit_favorites(_ONE, [83, 4])
    _reread(monkeypatch)
    assert settings.get_unit_favorites(_OTHER) == [93]


def test_an_empty_list_drops_the_game_then_the_key(tmp_path: Path, monkeypatch) -> None:
    settings.set_unit_favorites(_ONE, [4])
    settings.set_unit_favorites(_OTHER, [93])
    settings.set_unit_favorites(_ONE, [])
    assert _stored(tmp_path)["units_favorites"] == {_OTHER: [93]}
    settings.set_unit_favorites(_OTHER, ["only junk"])
    assert "units_favorites" not in _stored(tmp_path)
    _reread(monkeypatch)
    assert settings.get_unit_favorites(_ONE) == settings.get_unit_favorites(_OTHER) == []


def test_clearing_a_game_with_no_favorites_writes_nothing(tmp_path: Path) -> None:
    settings.set_unit_favorites(_ONE, [])
    assert not (tmp_path / "config.yaml").exists()


def test_the_view_mode_defaults_round_trips_and_refuses_an_unknown_one(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_units_favorites_mode() == settings.UNITS_FAVORITES_MODE_DEFAULT
    assert settings.UNITS_FAVORITES_MODE_DEFAULT in settings.UNITS_CATALOG_MODES
    settings.set_units_favorites_mode("grid")
    monkeypatch.setattr(settings, "_units_favorites_mode", None)
    assert settings.get_units_favorites_mode() == "grid"
    with pytest.raises(ValueError):
        settings.set_units_favorites_mode("tiles")
    assert _stored(tmp_path)["units_favorites_mode"] == "grid"
    assert settings.get_units_catalog_mode() == settings.UNITS_CATALOG_MODE_DEFAULT, "the catalog keeps its own"


def test_a_stored_unknown_view_mode_reads_as_the_default(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("units_favorites_mode: tiles\n")
    assert settings.get_units_favorites_mode() == settings.UNITS_FAVORITES_MODE_DEFAULT


def test_an_import_summary_counts_the_games_whose_favorites_differ(tmp_path: Path) -> None:
    settings.set_unit_favorites(_ONE, [4, 83])
    doc = {
        "format": settings_file.FORMAT_ID,
        "version": settings_file.FORMAT_VERSION,
        "exported_by": "DEscape test",
        "known_keys": list(settings_file.PORTABLE_KEYS),
        "settings": {"units_favorites": {_ONE: [83, 4], _OTHER: [93]}},
    }
    path = tmp_path / "import" / "s.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    lines = settings_file.describe_changes(settings_file.parse_settings_file(path))
    assert lines == ["Will change:", "  Panels: Units favorites: 2 games' favorites differ"]


@pytest.mark.parametrize(
    "key, getter, setter, ids, default",
    [
        (
            "favorite_star_style",
            settings.get_favorite_star_style,
            settings.set_favorite_star_style,
            [sid for sid, _label in settings.FAVORITE_STAR_STYLES],
            settings.FAVORITE_STAR_STYLE_DEFAULT,
        ),
        (
            "units_catalog_right_click",
            settings.get_units_catalog_right_click,
            settings.set_units_catalog_right_click,
            [cid for cid, _label in settings.UNITS_CATALOG_RIGHT_CLICKS],
            settings.UNITS_CATALOG_RIGHT_CLICK_DEFAULT,
        ),
    ],
    ids=["star style", "right-click"],
)
def test_the_star_and_right_click_settings_default_round_trip_and_refuse(
    tmp_path: Path, monkeypatch, key, getter, setter, ids, default
) -> None:
    memo = f"_{key}"
    assert getter() == default == ids[0], "the default is listed first"
    for value in ids:
        setter(value)
        monkeypatch.setattr(settings, memo, None)
        assert getter() == value
    with pytest.raises(ValueError):
        setter("bogus")
    assert _stored(tmp_path)[key] == ids[-1], "a refused value writes nothing"
    (tmp_path / "config.yaml").write_text(f"{key}: bogus\n")
    monkeypatch.setattr(settings, memo, None)
    assert getter() == default


def test_a_theme_import_sets_the_star_style(tmp_path: Path, monkeypatch) -> None:
    settings.apply_appearance_values({"favorite_star_style": "mono"})
    assert settings.get_favorite_star_style() == "mono"
    monkeypatch.setattr(settings, "_favorite_star_style", None)
    assert settings.get_favorite_star_style() == "mono"
