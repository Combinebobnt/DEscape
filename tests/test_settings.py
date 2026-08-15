"""settings.py round-trip gaps: legacy-key migration branches, out-of-range
fallback, invalid-input rejection, and malformed-YAML tolerance -- none of
which the existing verify_*.py scripts touch at all (settings.py has no
verify script of its own).

Every test here relies on conftest.py's autouse _isolated_settings fixture
having already redirected descape.settings.CONFIG_PATH to this test's own
tmp_path/config.yaml -- writing to that same path (via the tmp_path
parameter, which pytest hands back the identical directory both fixtures
share within one test) is how each test controls what settings.py reads,
without ever touching the developer's real config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import iso_geometry, settings


def _write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "config.yaml").write_text(text)


def test_missing_config_falls_back_to_default_quality(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_graphics_quality() == settings.GRAPHICS_QUALITY_DEFAULT


def test_potato_mode_true_migrates_to_potato_quality(tmp_path: Path) -> None:
    _write_config(tmp_path, "potato_mode: true\n")
    assert settings.get_graphics_quality() == 2


def test_potato_mode_false_migrates_to_default_quality(tmp_path: Path) -> None:
    _write_config(tmp_path, "potato_mode: false\n")
    assert settings.get_graphics_quality() == settings.GRAPHICS_QUALITY_DEFAULT


def test_out_of_range_graphics_quality_falls_back_to_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "graphics_quality: 99\n")
    assert settings.get_graphics_quality() == settings.GRAPHICS_QUALITY_DEFAULT


def test_set_graphics_quality_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        settings.set_graphics_quality(0)
    with pytest.raises(ValueError):
        settings.set_graphics_quality(settings.GRAPHICS_QUALITY_MAX + 1)


def test_set_graphics_quality_migrates_away_legacy_potato_mode(tmp_path: Path) -> None:
    _write_config(tmp_path, "potato_mode: true\n")
    settings.set_graphics_quality(4)
    on_disk = (tmp_path / "config.yaml").read_text()
    assert "graphics_quality: 4" in on_disk
    assert "potato_mode" not in on_disk


def test_malformed_yaml_is_tolerated_not_raised(tmp_path: Path) -> None:
    _write_config(tmp_path, "not: valid: yaml: [unterminated\n")
    # _load_config() catches yaml.YAMLError/OSError and returns {} -- every
    # getter must fall back to its own default rather than propagate.
    assert settings.get_graphics_quality() == settings.GRAPHICS_QUALITY_DEFAULT
    assert settings.get_dark_mode() is False
    assert settings.get_window_size() == (settings.DEFAULT_WINDOW_WIDTH, settings.DEFAULT_WINDOW_HEIGHT)


def test_legacy_elev_step_divisor_migrates_to_pct(tmp_path: Path) -> None:
    _write_config(tmp_path, "elev_step_divisor: 4\n")
    assert settings.get_elev_step_pct() == 25


def test_out_of_range_elev_step_pct_falls_back_to_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "elev_step_pct: 9999\n")
    assert settings.get_elev_step_pct() == iso_geometry.ELEV_STEP_DEFAULT_PCT


def test_set_elev_step_pct_clamps_above_max() -> None:
    settings.set_elev_step_pct(9999)
    assert settings.get_elev_step_pct() == settings.ELEV_STEP_PCT_MAX


def test_set_elev_step_pct_clamps_below_min() -> None:
    settings.set_elev_step_pct(-5)
    assert settings.get_elev_step_pct() == settings.ELEV_STEP_PCT_MIN


def test_set_elev_step_pct_migrates_away_legacy_divisor(tmp_path: Path) -> None:
    _write_config(tmp_path, "elev_step_divisor: 4\n")
    settings.set_elev_step_pct(60)
    on_disk = (tmp_path / "config.yaml").read_text()
    # 50, not 60 -- set_elev_step_pct snaps, so a programmatic caller can't
    # persist an off-stop value.
    assert "elev_step_pct: 50" in on_disk
    assert "elev_step_divisor" not in on_disk


def test_elev_step_stops_span_min_to_max() -> None:
    assert settings.ELEV_STEP_PCT_STOPS[0] == settings.ELEV_STEP_PCT_MIN
    assert settings.ELEV_STEP_PCT_STOPS[-1] == settings.ELEV_STEP_PCT_MAX
    assert all(
        b - a == settings.ELEV_STEP_PCT_STEP
        for a, b in zip(settings.ELEV_STEP_PCT_STOPS, settings.ELEV_STEP_PCT_STOPS[1:])
    )


def test_the_default_pct_is_itself_a_stop() -> None:
    """Load-bearing twice over: the slider's index lookup can't represent an
    off-stop default, and _update_elev_step_label's "(Tall, default)" suffix
    only ever shows if some stop equals ELEV_STEP_DEFAULT_PCT."""
    assert iso_geometry.ELEV_STEP_DEFAULT_PCT in settings.ELEV_STEP_PCT_STOPS


def test_off_stop_config_value_snaps_on_read(tmp_path: Path) -> None:
    _write_config(tmp_path, "elev_step_pct: 48\n")
    assert settings.get_elev_step_pct() == 50


def test_config_value_below_the_new_floor_snaps_up_not_to_default(tmp_path: Path) -> None:
    """10 was the old ELEV_STEP_PCT_MIN, so real configs hold it. The read
    gate gates on 1 <= raw <= MAX, not MIN, precisely so this snaps to the
    floor rather than falling back to the default."""
    _write_config(tmp_path, "elev_step_pct: 10\n")
    assert settings.get_elev_step_pct() == settings.ELEV_STEP_PCT_MIN


def test_elev_step_index_round_trips_every_stop() -> None:
    for pct in settings.ELEV_STEP_PCT_STOPS:
        assert settings.elev_step_pct_for_index(settings.elev_step_index(pct)) == pct


def test_elev_step_index_snaps_an_off_stop_pct_instead_of_raising() -> None:
    # A bare ELEV_STEP_PCT_STOPS.index() would raise ValueError here, which
    # would crash SettingsDialog on construction rather than degrade.
    assert settings.elev_step_index(48) == settings.elev_step_index(50)


def test_a_snapped_write_survives_a_fresh_read_from_disk(tmp_path: Path, monkeypatch) -> None:
    """The other set_elev_step_pct tests read back through the module-level
    _elev_step_pct cache, so they'd pass even if the snap never reached the
    file. Clearing the cache forces the real between-sessions path."""
    settings.set_elev_step_pct(60)
    monkeypatch.setattr(settings, "_elev_step_pct", None)
    assert settings.get_elev_step_pct() == 50
    assert "elev_step_pct: 50" in (tmp_path / "config.yaml").read_text()


def test_get_window_size_falls_back_below_minimum(tmp_path: Path) -> None:
    _write_config(tmp_path, "window_size: [10, 10]\n")
    assert settings.get_window_size() == (settings.DEFAULT_WINDOW_WIDTH, settings.DEFAULT_WINDOW_HEIGHT)


def test_get_window_size_accepts_valid_saved_size(tmp_path: Path) -> None:
    _write_config(tmp_path, "window_size: [1234, 987]\n")
    assert settings.get_window_size() == (1234, 987)
