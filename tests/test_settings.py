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
    assert "elev_step_pct: 60" in on_disk
    assert "elev_step_divisor" not in on_disk


def test_get_window_size_falls_back_below_minimum(tmp_path: Path) -> None:
    _write_config(tmp_path, "window_size: [10, 10]\n")
    assert settings.get_window_size() == (settings.DEFAULT_WINDOW_WIDTH, settings.DEFAULT_WINDOW_HEIGHT)


def test_get_window_size_accepts_valid_saved_size(tmp_path: Path) -> None:
    _write_config(tmp_path, "window_size: [1234, 987]\n")
    assert settings.get_window_size() == (1234, 987)
