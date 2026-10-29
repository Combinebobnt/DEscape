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

import os
from pathlib import Path

import pytest

from descape import edge_ticks, grid_overlay, iso_geometry, settings, terrain_style, view_layers

import conftest


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
    assert settings.get_theme() == "editor_parchment_red"
    assert settings.get_window_size() == (settings.DEFAULT_WINDOW_WIDTH, settings.DEFAULT_WINDOW_HEIGHT)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("dark_mode: true\n", "dark"), ("dark_mode: false\n", "light"), ("", "editor_parchment_red")],
)
def test_legacy_dark_mode_migrates_to_a_theme(tmp_path: Path, text: str, expected: str) -> None:
    _write_config(tmp_path, text)
    assert settings.get_theme() == expected


def test_a_theme_key_wins_over_legacy_dark_mode(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: solarized_light\ndark_mode: true\n")
    assert settings.get_theme() == "solarized_light"


def test_set_theme_pops_legacy_dark_mode_and_clears_overrides(tmp_path: Path) -> None:
    import yaml

    _write_config(tmp_path, "dark_mode: true\ntheme_colors:\n  window: '#123456'\n")
    assert settings.get_theme_colors() == {"window": "#123456"}
    settings.set_theme("dim")
    on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert "dark_mode" not in on_disk
    assert on_disk["theme"] == "dim"
    assert on_disk["theme_colors"] == {}
    assert settings.get_theme_colors() == {}


def test_unknown_theme_falls_back_to_the_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: no_such_preset\n")
    assert settings.get_theme() == "editor_parchment_red"


def test_set_theme_rejects_an_unknown_id_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_theme("no_such_preset")
    assert not (tmp_path / "config.yaml").exists()


def test_a_bad_theme_override_is_skipped_per_entry(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "theme_colors:\n  window: '#ABCDEF'\n  text: 'red'\n  no_such_role: '#000000'\n  base: 12\n",
    )
    assert settings.get_theme_colors() == {"window": "#abcdef"}


def test_theme_colors_that_are_not_a_mapping_are_ignored(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme_colors: [1, 2]\n")
    assert settings.get_theme_colors() == {}


def test_set_and_clear_theme_color_round_trip(tmp_path: Path) -> None:
    settings.set_theme_color("highlight", "#FF8800")
    settings.set_theme_color("text", "#010203")
    settings.clear_theme_color("text")
    settings._theme_colors = None  # re-read from disk
    assert settings.get_theme_colors() == {"highlight": "#ff8800"}


def test_set_theme_color_rejects_bad_input_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_theme_color("no_such_role", "#000000")
    with pytest.raises(ValueError):
        settings.set_theme_color("window", "#12345")
    assert not (tmp_path / "config.yaml").exists()


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
    # strict=False: a list against its own tail is ragged by construction, and
    # the one-shorter right operand is what makes this the pairwise check.
    assert all(
        b - a == settings.ELEV_STEP_PCT_STEP
        for a, b in zip(settings.ELEV_STEP_PCT_STOPS, settings.ELEV_STEP_PCT_STOPS[1:], strict=False)
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


def test_get_log_height_is_none_until_first_saved() -> None:
    assert settings.get_log_height() is None


def test_log_height_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_log_height(200)
    monkeypatch.setattr(settings, "_log_height", None)
    assert settings.get_log_height() == 200
    assert "log_height: 200" in (tmp_path / "config.yaml").read_text()


def test_set_log_height_below_minimum_is_a_silent_no_op(tmp_path: Path) -> None:
    settings.set_log_height(settings.MIN_LOG_PANE - 1)
    assert settings.get_log_height() is None
    assert not (tmp_path / "config.yaml").exists()


def test_get_log_height_rejects_a_malformed_saved_value(tmp_path: Path) -> None:
    _write_config(tmp_path, "log_height: banana\n")
    assert settings.get_log_height() is None


# --- Settings > Appearance > Preload neighbouring zoom levels --------------


def test_preload_zoom_levels_is_on_until_a_config_says_otherwise(tmp_path: Path) -> None:
    """Default ON, unlike every other expensive toggle here: the warm it
    gates is sliced across idle ticks and never blocks the window, so there
    is no open-time cost to opt into (2026-09-04 plan, Step 5)."""
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_preload_zoom_levels() is True


def test_preload_zoom_levels_reads_false_from_config(tmp_path: Path) -> None:
    _write_config(tmp_path, "preload_zoom_levels: false\n")
    assert settings.get_preload_zoom_levels() is False


def test_preload_zoom_levels_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_preload_zoom_levels(False)
    monkeypatch.setattr(settings, "_preload_zoom_levels", None)
    assert settings.get_preload_zoom_levels() is False


# --- Settings > General > X11 compatibility on Wayland ---------------------


def test_xwayland_on_wayland_is_off_until_a_config_says_otherwise(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_xwayland_on_wayland() is False


def test_xwayland_on_wayland_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_xwayland_on_wayland(True)
    monkeypatch.setattr(settings, "_xwayland_on_wayland", None)
    assert settings.get_xwayland_on_wayland() is True
    settings.set_xwayland_on_wayland(False)
    monkeypatch.setattr(settings, "_xwayland_on_wayland", None)
    assert settings.get_xwayland_on_wayland() is False


# --- View > Distance Ticks -------------------------------------------------


def test_distance_ticks_is_off_until_a_config_says_otherwise(tmp_path: Path) -> None:
    """Default off is load-bearing, not a preference: every existing capture
    test would otherwise start seeing marks it never had to account for."""
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_distance_ticks() is False


def test_distance_ticks_reads_true_from_config(tmp_path: Path) -> None:
    _write_config(tmp_path, "distance_ticks: true\n")
    assert settings.get_distance_ticks() is True


def test_distance_ticks_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_distance_ticks(True)
    monkeypatch.setattr(settings, "_distance_ticks", None)
    assert settings.get_distance_ticks() is True


def test_grid_overlay_defaults_off_at_the_default_appearance(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_grid_overlay() is False
    assert settings.get_grid_blend() == grid_overlay.BLEND_DEFAULT
    assert settings.get_grid_thickness() == grid_overlay.THICKNESS_DEFAULT


def test_grid_keys_read_from_config(tmp_path: Path) -> None:
    _write_config(tmp_path, "grid_overlay: true\ngrid_blend: 60\ngrid_thickness: 3\n")
    assert settings.get_grid_overlay() is True
    assert settings.get_grid_blend() == 60
    assert settings.get_grid_thickness() == 3


def test_grid_appearance_clamps_or_snaps_an_out_of_range_value(tmp_path: Path) -> None:
    _write_config(tmp_path, "grid_blend: -500\ngrid_thickness: 0\n")
    assert settings.get_grid_blend() == grid_overlay.BLEND_MIN
    assert settings.get_grid_thickness() == grid_overlay.THICKNESS_STOPS[0]


@pytest.mark.parametrize("raw", ["banana", "true", "[1, 2]"])
def test_grid_appearance_falls_back_on_a_malformed_value(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"grid_blend: {raw}\ngrid_thickness: {raw}\n")
    assert settings.get_grid_blend() == grid_overlay.BLEND_DEFAULT
    assert settings.get_grid_thickness() == grid_overlay.THICKNESS_DEFAULT


def test_a_legacy_grid_lightness_is_read_as_a_blend(tmp_path: Path) -> None:
    """A config written before the blend slider keeps roughly its look
    instead of opening invisible."""
    _write_config(tmp_path, "grid_lightness: 30\n")
    assert settings.get_grid_blend() == grid_overlay.blend_for_lightness(30)


def test_grid_blend_wins_over_a_legacy_grid_lightness(tmp_path: Path) -> None:
    _write_config(tmp_path, "grid_blend: -40\ngrid_lightness: 240\n")
    assert settings.get_grid_blend() == -40


def test_setting_a_blend_does_not_write_the_legacy_key_back(tmp_path: Path) -> None:
    _write_config(tmp_path, "grid_lightness: 30\n")
    settings.set_grid_blend(-40)
    text = (tmp_path / "config.yaml").read_text()
    assert "grid_blend: -40" in text
    assert "grid_lightness" not in text


def test_grid_keys_round_trip_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_grid_overlay(True)
    settings.set_grid_blend(90)
    settings.set_grid_thickness(4)
    for name in ("_grid_overlay", "_grid_blend", "_grid_thickness"):
        monkeypatch.setattr(settings, name, None)
    assert settings.get_grid_overlay() is True
    assert settings.get_grid_blend() == 90
    assert settings.get_grid_thickness() == 4


def test_grid_major_every_defaults_to_four(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_grid_major_every() == grid_overlay.MAJOR_EVERY_DEFAULT == 4


@pytest.mark.parametrize("raw", ["true", "'8'", "banana", "3", "32", "0", "-4", "8.0", "[8]"])
def test_grid_major_every_reads_anything_but_a_stop_as_the_default(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"grid_major_every: {raw}\n")
    assert settings.get_grid_major_every() == 4


@pytest.mark.parametrize("value", grid_overlay.MAJOR_EVERY_STOPS)
def test_grid_major_every_round_trips_through_the_file(tmp_path: Path, monkeypatch, value: int) -> None:
    settings.set_grid_major_every(value)
    assert (tmp_path / "config.yaml").exists()
    monkeypatch.setattr(settings, "_grid_major_every", None)
    assert settings.get_grid_major_every() == value


@pytest.mark.parametrize("value", [3, 32, True, "8"])
def test_setting_a_grid_major_every_off_the_stops_raises_and_writes_nothing(tmp_path: Path, value) -> None:
    with pytest.raises(ValueError):
        settings.set_grid_major_every(value)
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_grid_major_every() == 4


def test_grid_keys_tolerate_malformed_yaml(tmp_path: Path) -> None:
    _write_config(tmp_path, "not: valid: yaml: [unterminated\n")
    assert settings.get_grid_overlay() is False
    assert settings.get_grid_blend() == grid_overlay.BLEND_DEFAULT


def test_stack_badges_default_on(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_stack_badges() is True


def test_stack_badges_reads_false_from_config(tmp_path: Path) -> None:
    _write_config(tmp_path, "stack_badges: false\n")
    assert settings.get_stack_badges() is False


def test_stack_badges_round_trips_independently_of_distance_ticks(tmp_path: Path, monkeypatch) -> None:
    settings.set_stack_badges(False)
    monkeypatch.setattr(settings, "_stack_badges", None)
    monkeypatch.setattr(settings, "_distance_ticks", None)
    assert settings.get_stack_badges() is False
    assert settings.get_distance_ticks() is False


# --- GH #100: stack badge position ---------------------------------------


def test_the_stack_badge_position_defaults_to_bottom_right(tmp_path: Path) -> None:
    assert settings.get_stack_badge_position() == "bottom_right"
    assert settings.STACK_BADGE_POSITION_DEFAULT == "bottom_right"
    assert [pid for pid, _label in settings.STACK_BADGE_POSITIONS] == [
        "bottom_right", "bottom_left", "top_right", "top_left", "above"
    ]


@pytest.mark.parametrize("raw", ["centre", "'BOTTOM_RIGHT'", "0", "true", "[above]"])
def test_an_off_list_stack_badge_position_falls_back(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"stack_badge_position: {raw}\n")
    assert settings.get_stack_badge_position() == "bottom_right"


def test_setting_an_unknown_stack_badge_position_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_stack_badge_position("centre")
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("position", [pid for pid, _label in settings.STACK_BADGE_POSITIONS])
def test_the_stack_badge_position_round_trips(tmp_path: Path, monkeypatch, position: str) -> None:
    settings.set_stack_badge_position(position)
    monkeypatch.setattr(settings, "_stack_badge_position", None)
    assert settings.get_stack_badge_position() == position


# --- GH #161: terrain picker mode and grid swatch size --------------------


def test_the_terrain_picker_defaults_to_list_and_48_px(tmp_path: Path) -> None:
    assert settings.get_terrain_picker_mode() == "list"
    assert [mid for mid, _label in settings.TERRAIN_PICKER_MODES] == ["list", "grid"]
    assert settings.get_terrain_grid_swatch_px() == 48
    assert settings.TERRAIN_GRID_SWATCH_SIZES == (32, 48, 64)


@pytest.mark.parametrize("raw", ["icons", "'GRID'", "0", "true", "[grid]"])
def test_an_off_list_terrain_picker_mode_falls_back(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"terrain_picker_mode: {raw}\n")
    assert settings.get_terrain_picker_mode() == "list"


@pytest.mark.parametrize("raw", ["40", "128", "'48'", "48.0", "true", "[48]"])
def test_an_off_list_terrain_grid_swatch_px_falls_back(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"terrain_grid_swatch_px: {raw}\n")
    assert settings.get_terrain_grid_swatch_px() == 48


@pytest.mark.parametrize(
    ("setter", "bad"),
    [
        (settings.set_terrain_picker_mode, "icons"),
        (settings.set_terrain_grid_swatch_px, 40),
        (settings.set_terrain_grid_swatch_px, "48"),
        (settings.set_terrain_grid_swatch_px, True),
    ],
)
def test_setting_an_unknown_terrain_picker_value_raises_and_writes_nothing(tmp_path: Path, setter, bad) -> None:
    with pytest.raises(ValueError):
        setter(bad)
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("mode", [mid for mid, _label in settings.TERRAIN_PICKER_MODES])
def test_the_terrain_picker_mode_round_trips(tmp_path: Path, monkeypatch, mode: str) -> None:
    settings.set_terrain_picker_mode(mode)
    monkeypatch.setattr(settings, "_terrain_picker_mode", None)
    assert settings.get_terrain_picker_mode() == mode


@pytest.mark.parametrize("px", settings.TERRAIN_GRID_SWATCH_SIZES)
def test_the_terrain_grid_swatch_px_round_trips(tmp_path: Path, monkeypatch, px: int) -> None:
    settings.set_terrain_grid_swatch_px(px)
    monkeypatch.setattr(settings, "_terrain_grid_swatch_px", None)
    assert settings.get_terrain_grid_swatch_px() == px


def test_the_stack_badge_background_colour_row_follows_the_text_row() -> None:
    ids = [cid for cid, _label, _default in settings.OVERLAY_COLORS]
    assert ids[ids.index("unit_stack") + 1] == "unit_stack_background"
    assert settings.get_default_overlay_color("unit_stack_background") == "#000000"


def test_a_missing_interval_falls_back_to_the_default(tmp_path: Path) -> None:
    assert settings.get_distance_tick_interval() == edge_ticks.TICK_INTERVAL_DEFAULT


@pytest.mark.parametrize("interval", edge_ticks.TICK_INTERVALS)
def test_every_legal_interval_reads_back_unchanged(tmp_path: Path, interval: int) -> None:
    _write_config(tmp_path, f"distance_tick_interval: {interval}\n")
    assert settings.get_distance_tick_interval() == interval


@pytest.mark.parametrize("raw", ["3", "7", "0", "-4", "'four'", "true", "[4]"])
def test_an_off_list_interval_falls_back_rather_than_snapping(tmp_path: Path, raw: str) -> None:
    """Gates on membership, not on a range. 3 and 7 both sit between or
    beside the legal pair, so a clamping implementation would hand back a
    legal-looking neighbour instead of the default and hide the bad key."""
    _write_config(tmp_path, f"distance_tick_interval: {raw}\n")
    assert settings.get_distance_tick_interval() == edge_ticks.TICK_INTERVAL_DEFAULT


def test_setting_an_off_list_interval_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_distance_tick_interval(3)


def test_a_rejected_interval_writes_nothing(tmp_path: Path) -> None:
    """The raise must come before the file write, or a rejected value still
    lands on disk for the next read to fall back from."""
    with pytest.raises(ValueError):
        settings.set_distance_tick_interval(99)
    assert not (tmp_path / "config.yaml").exists()


def test_the_interval_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
    settings.set_distance_tick_interval(other)
    monkeypatch.setattr(settings, "_distance_tick_interval", None)
    assert settings.get_distance_tick_interval() == other


def test_the_two_tick_keys_are_independent(tmp_path: Path, monkeypatch) -> None:
    """Writing one must not clear the other. Both go through _load_config,
    so a set() that rebuilt the dict instead of updating it would."""
    settings.set_distance_ticks(True)
    other = next(n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT)
    settings.set_distance_tick_interval(other)
    monkeypatch.setattr(settings, "_distance_ticks", None)
    monkeypatch.setattr(settings, "_distance_tick_interval", None)
    assert settings.get_distance_ticks() is True
    assert settings.get_distance_tick_interval() == other


def test_malformed_yaml_leaves_both_tick_keys_at_their_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "distance_ticks: [unclosed\n")
    assert settings.get_distance_ticks() is False
    assert settings.get_distance_tick_interval() == edge_ticks.TICK_INTERVAL_DEFAULT


# The memoized-globals list itself moved to testkit.settings_isolation, shared
# with the tools/ capture scripts; its drift test went with it, to
# tests/test_settings_isolation.py.


def test_a_config_predating_the_ruler_moves_elevate_off_r(tmp_path: Path) -> None:
    """The migration's whole reason to exist: without it this config and the
    new tool_ruler default would both hold R, and Qt fires neither."""
    _write_config(tmp_path, "keybinds:\n  tool_elevation: R\n  tool_pan: M\n")
    assert settings.get_keybind("tool_elevation") == "E"
    assert settings.get_keybind("tool_ruler") == "R"


def test_keybind_migration_does_not_fire_twice(tmp_path: Path) -> None:
    """The discriminating case. Asserting only that a pre-Ruler config comes
    back as E passes with the "tool_ruler not in persisted" guard removed;
    this one does not. A config that already names tool_ruler was written
    after the change, so an R on Elevate there is a deliberate user choice and
    must survive every launch."""
    _write_config(tmp_path, "keybinds:\n  tool_elevation: R\n  tool_ruler: ''\n")
    assert settings.get_keybind("tool_elevation") == "R"


def test_migration_leaves_a_non_default_elevate_binding_alone(tmp_path: Path) -> None:
    """Value-equals-old-default is the whole test. Anyone who had already
    moved Elevate somewhere else never had the collision to begin with."""
    _write_config(tmp_path, "keybinds:\n  tool_elevation: K\n")
    assert settings.get_keybind("tool_elevation") == "K"


def test_migration_does_not_write_to_disk(tmp_path: Path) -> None:
    """Translate on read, never write from a getter, matching the potato_mode
    migration above. The corrected value persists on its own the next time
    set_keybind rewrites the whole dict."""
    _write_config(tmp_path, "keybinds:\n  tool_elevation: R\n")
    assert settings.get_keybind("tool_elevation") == "E"
    assert (tmp_path / "config.yaml").read_text() == "keybinds:\n  tool_elevation: R\n"


def test_a_config_predating_gh207_gets_ctrl_0_on_show_gaia(tmp_path: Path) -> None:
    """set_keybind pinned filter_show_gaia: '' before GH #207 gave it Ctrl+0;
    without the migration only Ctrl+1..8 would reach this user."""
    _write_config(tmp_path, "keybinds:\n  filter_show_gaia: ''\n  tool_pan: M\n")
    assert settings.get_keybind("filter_show_gaia") == "Ctrl+0"
    assert settings.get_keybind("filter_player_1") == "Ctrl+1"


def test_the_show_gaia_migration_does_not_fire_twice(tmp_path: Path) -> None:
    """A config naming filter_player_1 was written after GH #207: a cleared Show GAIA there is deliberate."""
    _write_config(tmp_path, "keybinds:\n  filter_show_gaia: ''\n  filter_player_1: Ctrl+1\n")
    assert settings.get_keybind("filter_show_gaia") == ""


def test_the_show_gaia_migration_yields_to_a_users_own_ctrl_0(tmp_path: Path) -> None:
    """The migrated value counts as a default, so a pre-GH #207 custom Ctrl+0 elsewhere keeps it."""
    _write_config(tmp_path, "keybinds:\n  filter_show_gaia: ''\n  tool_pan: Ctrl+0\n")
    assert settings.get_keybind("tool_pan") == "Ctrl+0"
    assert settings.get_keybind("filter_show_gaia") == ""


def test_a_config_with_no_keybinds_block_gets_the_new_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: dark\n")
    assert settings.get_keybind("tool_elevation") == "E"
    assert settings.get_keybind("tool_ruler") == "R"


def test_default_keybinds_are_pairwise_conflict_free() -> None:
    """The defaults pass the same rule set_keybind() and the load-time
    reconciler apply (GH #267): no key on two actions live in a common mode.
    Nothing else checks the shipped table, and the Default button's refusal
    reasons from it. Empty means unbound, and any number of actions may share that."""
    bound = [(action_id, key) for action_id, key in settings._DEFAULT_KEYBINDS.items() if key]
    conflicts = [
        (a, b, key)
        for i, (a, key) in enumerate(bound)
        for b, other in bound[i + 1 :]
        if key == other and settings.keybind_scopes_overlap(a, b)
    ]
    assert not conflicts, conflicts


# -- GH #267: per-mode keybind scopes ---------------------------------------------


def test_tool_rows_take_their_tooldef_modes_and_every_scope_names_real_rows_and_modes() -> None:
    rebindable = {action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS}
    for tool in settings.TOOLS:
        assert settings.keybind_modes(f"tool_{tool.tool_id}") == frozenset(tool.modes), tool.tool_id
    assert set(settings._SCOPED_KEYBINDS) <= rebindable
    assert not any(action_id.startswith("tool_") for action_id in settings._SCOPED_KEYBINDS), "tools derive from TOOLS"
    named = {mode for action_id in rebindable for mode in settings.keybind_modes(action_id)}
    assert named and named <= set(settings.MODE_LABELS)
    assert settings.keybind_modes("tool_pan") == frozenset() and settings.keybind_modes("tool_ruler") == frozenset()
    assert settings.keybind_modes("unit_rotate_cw") == frozenset(), "Rotate also turns a region outside Units"
    assert settings.keybind_modes("some_removed_action") == frozenset(), "an unknown id stays global"


def test_the_mode_tag_lists_modes_in_combo_order_and_is_empty_for_a_global_action() -> None:
    assert settings.keybind_mode_tag("tool_place_unit") == " (Units)"
    assert settings.keybind_mode_tag("tool_select") == " (Terrain, Units)"
    assert settings.keybind_mode_tag("trigger_move_up") == " (Triggers)"
    assert settings.keybind_mode_tag("edit_undo") == ""
    assert settings.keybind_display_label("tool_set_level") == "Set Elevation Tool (Terrain)"
    assert settings.keybind_display_label("edit_undo") == "Undo"


def test_scope_overlap_is_symmetric_and_a_global_action_overlaps_everything() -> None:
    pairs = [
        ("tool_place_unit", "tool_set_level", False),
        ("tool_select", "tool_set_level", True),
        ("tool_select", "tool_place_unit", True),
        ("edit_undo", "tool_place_unit", True),
        ("tool_pan", "trigger_move_up", True),
        ("trigger_move_up", "unit_select_stack", False),
    ]
    for a, b, expected in pairs:
        assert settings.keybind_scopes_overlap(a, b) is expected, (a, b)
        assert settings.keybind_scopes_overlap(b, a) is expected, (b, a)


def test_two_disjoint_tools_share_a_key_and_a_tool_of_both_modes_clears_them_both(tmp_path: Path) -> None:
    assert settings.keybind_holder("U") is None, "test assumes nothing holds U"
    assert settings.set_keybind("tool_place_unit", "U") == []
    assert settings.set_keybind("tool_set_level", "U") == []
    assert settings.get_keybind("tool_place_unit") == "U" and settings.get_keybind("tool_set_level") == "U"
    assert settings.keybind_conflicts("U", "tool_select") == ["tool_set_level", "tool_place_unit"]

    assert settings.set_keybind("tool_select", "U") == ["tool_set_level", "tool_place_unit"]
    assert settings.get_keybind("tool_select") == "U"
    assert settings.get_keybind("tool_set_level") == "" and settings.get_keybind("tool_place_unit") == ""


def test_a_global_command_on_a_scoped_tools_key_clears_the_tool(tmp_path: Path) -> None:
    settings.set_keybind("tool_place_unit", "U")
    assert settings.keybind_conflicts("U", "edit_undo") == ["tool_place_unit"]
    assert settings.set_keybind("edit_undo", "U") == ["tool_place_unit"]
    assert settings.get_keybind("tool_place_unit") == ""


def test_keybind_conflicts_ignores_the_action_itself_and_an_empty_key(tmp_path: Path) -> None:
    settings.set_keybind("tool_place_unit", "U")
    assert settings.keybind_conflicts("U", "tool_place_unit") == []
    assert settings.keybind_conflicts("U", "tool_set_level") == []
    assert settings.keybind_conflicts("", "tool_select") == []


def test_load_time_reconciliation_keeps_two_disjoint_tools_on_one_key(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  tool_place_unit: U\n  tool_set_level: U\n")
    assert settings.get_keybind("tool_place_unit") == "U"
    assert settings.get_keybind("tool_set_level") == "U"


def test_load_time_reconciliation_clears_draw_under_a_customized_select(tmp_path: Path) -> None:
    """Draw (Terrain) is at its default D; Select (Terrain, Units) moved onto it overlaps it."""
    _write_config(tmp_path, "keybinds:\n  tool_select: D\n")
    assert settings.get_keybind("tool_select") == "D"
    assert settings.get_keybind("tool_draw") == ""


def test_load_time_reconciliation_keeps_draw_beside_a_units_tool_on_its_key(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  tool_place_unit: D\n")
    assert settings.get_keybind("tool_place_unit") == "D"
    assert settings.get_keybind("tool_draw") == "D"


def test_load_time_reconciliation_resolves_the_non_transitive_case_in_declared_order(tmp_path: Path) -> None:
    """Set Elevation and Place Unit are disjoint, Select overlaps both, all three
    customized onto U. TOOLS order: Set Elevation claims U first, Select collides
    with it and does not outrank a customized holder, so Select is cleared, and
    Place Unit, disjoint from the only claim, keeps U."""
    ids = [action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS]
    assert ids.index("tool_set_level") < ids.index("tool_select") < ids.index("tool_place_unit")
    _write_config(tmp_path, "keybinds:\n  tool_set_level: U\n  tool_select: U\n  tool_place_unit: U\n")
    assert settings.get_keybind("tool_set_level") == "U"
    assert settings.get_keybind("tool_select") == ""
    assert settings.get_keybind("tool_place_unit") == "U"


def test_load_time_reconciliation_lets_a_customized_action_clear_every_default_holder_it_collides_with(
    tmp_path: Path, monkeypatch
) -> None:
    """No two shipped defaults share a key, so two disjoint tools get U as their
    default here. A later-declared global action customized onto U collides with
    both and outranks both, so both are cleared."""
    defaults = dict(settings._DEFAULT_KEYBINDS, tool_set_level="U", tool_place_unit="U")
    monkeypatch.setattr(settings, "_DEFAULT_KEYBINDS", defaults)
    _write_config(tmp_path, "keybinds:\n  adjust_increment: U\n")
    assert settings.get_keybind("adjust_increment") == "U"
    assert settings.get_keybind("tool_set_level") == "" and settings.get_keybind("tool_place_unit") == ""


def test_load_time_reconciliation_clears_a_default_colliding_with_a_persisted_override(
    tmp_path: Path,
) -> None:
    """edit_redo's default is Ctrl+Y; persisting it onto edit_undo's default
    (Ctrl+Z) is a deliberate customization and must survive -- edit_undo, left
    at its default, is the one that loses the collision."""
    _write_config(tmp_path, "keybinds:\n  edit_redo: Ctrl+Z\n")
    assert settings.get_keybind("edit_redo") == "Ctrl+Z"
    assert settings.get_keybind("edit_undo") == ""


def test_load_time_reconciliation_ignores_a_persisted_value_equal_to_its_own_default(
    tmp_path: Path,
) -> None:
    """A full persisted dict that just mirrors current defaults (what
    set_keybind's whole-dict write produces for someone who has only ever
    touched an unrelated binding) must not misread as two collisions -- mere
    presence in `persisted` is not "customized", only a value that differs
    from _DEFAULT_KEYBINDS is."""
    _write_config(tmp_path, "keybinds:\n  edit_undo: Ctrl+Z\n  edit_redo: Ctrl+Y\n")
    assert settings.get_keybind("edit_undo") == "Ctrl+Z"
    assert settings.get_keybind("edit_redo") == "Ctrl+Y"


def test_load_time_reconciliation_is_deterministic_between_two_customized_values(
    tmp_path: Path,
) -> None:
    """Two persisted (both customized) values colliding is not resolvable by
    the "config beats defaults" priority rule alone -- REBINDABLE_ACTIONS
    order is the deterministic tie-break, and edit_undo is declared before
    edit_redo."""
    _write_config(tmp_path, "keybinds:\n  edit_undo: F5\n  edit_redo: F5\n")
    assert settings.get_keybind("edit_undo") == "F5"
    assert settings.get_keybind("edit_redo") == ""


def test_keybind_holder_finds_the_action_holding_a_taken_sequence(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  edit_undo: F5\n")
    assert settings.keybind_holder("F5") == "edit_undo"


def test_keybind_holder_returns_none_for_an_untaken_sequence(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  edit_undo: F5\n")
    assert settings.keybind_holder("F9") is None


def test_keybind_holder_returns_none_for_an_empty_sequence(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: dark\n")
    assert settings.keybind_holder("") is None


def test_keybind_holder_returns_none_when_the_only_holder_is_excluded(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  edit_undo: F5\n")
    assert settings.keybind_holder("F5", exclude="edit_undo") is None


def test_keybind_holder_ignores_a_stale_action_id_not_in_rebindable_actions(tmp_path: Path) -> None:
    """A removed action_id can still sit in persisted config -- _load_keybinds()
    does `_keybinds.update(persisted)`, so it survives into the in-memory dict.
    keybind_holder must iterate REBINDABLE_ACTIONS, not the dict, or a stale
    entry would produce a refusal with no UI row to resolve it."""
    _write_config(tmp_path, "keybinds:\n  some_removed_action: F5\n")
    assert settings.keybind_holder("F5") is None


def test_the_retired_wall_run_keybind_is_dropped_on_load(tmp_path: Path) -> None:
    """GH #98 folded Wall Run into Place Unit. Its saved key must not linger
    for set_keybind()'s collision loop to auto-clear and report."""
    _write_config(tmp_path, "keybinds:\n  tool_wall_run: W\n")
    assert "tool_wall_run" not in settings._load_keybinds()
    assert settings.get_keybind("tool_place_unit") == ""
    assert settings.set_keybind("tool_pan", "W") == []


def test_a_saved_edit_disables_binding_loads_as_map_disables_and_is_then_dropped(tmp_path: Path) -> None:
    """Disabled Objects… moved to the Map menu, renaming its keybind id."""
    import yaml

    _write_config(tmp_path, "keybinds:\n  edit_disables: F7\n")
    assert settings.get_keybind("map_disables") == "F7"
    assert "edit_disables" not in settings._load_keybinds()
    assert settings.set_keybind("tool_pan", "F8") == []
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())["keybinds"]
    assert saved["map_disables"] == "F7" and "edit_disables" not in saved


def test_a_migrated_disables_binding_counts_as_customized_in_a_load_time_collision(tmp_path: Path) -> None:
    """edit_undo is declared first, so it would win a tie; the migrated binding
    must instead read as the user's own choice and beat the default."""
    _write_config(tmp_path, "keybinds:\n  edit_disables: Ctrl+Z\n")
    assert settings.get_keybind("map_disables") == "Ctrl+Z"
    assert settings.get_keybind("edit_undo") == ""


def test_a_saved_file_resize_binding_loads_as_map_resize_and_is_then_dropped(tmp_path: Path) -> None:
    """Resize Map… moved from the File menu to the Map menu, renaming its keybind id."""
    import yaml

    _write_config(tmp_path, "keybinds:\n  file_resize: F7\n")
    assert settings.get_keybind("map_resize") == "F7"
    assert "file_resize" not in settings._load_keybinds()
    assert settings.set_keybind("tool_pan", "F8") == []
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())["keybinds"]
    assert saved["map_resize"] == "F7" and "file_resize" not in saved


def test_a_saved_map_disables_binding_beats_a_stale_edit_disables_one(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  edit_disables: F7\n  map_disables: F9\n")
    assert settings.get_keybind("map_disables") == "F9"


# The mouse wheel rows (Settings > Keybinds): one action per modifier slot.


def test_the_wheel_bindings_default_on_an_empty_config(tmp_path: Path) -> None:
    assert settings.get_wheel_bindings() == {"plain": "zoom", "ctrl": "tool", "shift": "zoom", "alt": "zoom"}
    assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS
    assert set(settings.DEFAULT_WHEEL_BINDINGS) == set(settings.WHEEL_SLOTS) == set(settings.WHEEL_SLOT_LABELS)
    assert set(settings.WHEEL_ACTIONS) == set(settings.WHEEL_ACTION_LABELS)
    assert set(settings.DEFAULT_WHEEL_BINDINGS.values()) <= set(settings.WHEEL_ACTIONS)
    assert not (tmp_path / "config.yaml").exists(), "a getter wrote the config"


def test_get_wheel_bindings_hands_out_a_copy() -> None:
    settings.get_wheel_bindings()["plain"] = "none"
    assert settings.get_wheel_bindings()["plain"] == "zoom"


@pytest.mark.parametrize("slot", settings.WHEEL_SLOTS)
@pytest.mark.parametrize("action", settings.WHEEL_ACTIONS)
def test_every_wheel_slot_and_action_round_trips_through_the_file(
    tmp_path: Path, monkeypatch, slot: str, action: str
) -> None:
    settings.set_wheel_binding(slot, action)
    monkeypatch.setattr(settings, "_wheel_bindings", None)
    assert settings.get_wheel_bindings() == {**settings.DEFAULT_WHEEL_BINDINGS, slot: action}


@pytest.mark.parametrize("slot, action", [("meta", "zoom"), ("", "zoom"), ("plain", "spin"), ("plain", "")])
def test_setting_an_unknown_wheel_slot_or_action_raises_and_writes_nothing(
    tmp_path: Path, slot: str, action: str
) -> None:
    with pytest.raises(ValueError):
        settings.set_wheel_binding(slot, action)
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS


@pytest.mark.parametrize("junk", ["spin", "Zoom", "3", "true", "[zoom]", "{}", "null"])
def test_a_junk_stored_wheel_action_keeps_only_its_own_default(tmp_path: Path, junk: str) -> None:
    _write_config(tmp_path, f"wheel_bindings:\n  ctrl: {junk}\n  plain: rotate\n  meta: zoom\n")
    assert settings.get_wheel_bindings() == {**settings.DEFAULT_WHEEL_BINDINGS, "plain": "rotate"}


@pytest.mark.parametrize("raw", ["zoom", "3", "[plain, tool]", "true", "null"])
def test_a_non_mapping_wheel_bindings_block_reads_as_defaults(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"wheel_bindings: {raw}\n")
    assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS


def test_the_wheel_bindings_file_holds_overrides_only(tmp_path: Path) -> None:
    """A slot back on its default loses its key, and the last one takes the
    mapping with it, so a later default change still reaches that user."""
    import yaml

    def stored() -> dict:
        return yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}

    settings.set_pan_speed(900)  # an unrelated key the pruning must not touch
    settings.set_wheel_binding("plain", "tool")
    settings.set_wheel_binding("ctrl", "zoom")
    assert stored()["wheel_bindings"] == {"plain": "tool", "ctrl": "zoom"}

    settings.set_wheel_binding("shift", settings.DEFAULT_WHEEL_BINDINGS["shift"])
    assert stored()["wheel_bindings"] == {"plain": "tool", "ctrl": "zoom"}, "a default was written as an override"

    settings.set_wheel_binding("plain", settings.DEFAULT_WHEEL_BINDINGS["plain"])
    assert stored()["wheel_bindings"] == {"ctrl": "zoom"}

    settings.set_wheel_binding("ctrl", settings.DEFAULT_WHEEL_BINDINGS["ctrl"])
    assert "wheel_bindings" not in stored()
    assert stored()["pan_speed"] == 900
    assert settings.get_wheel_bindings() == settings.DEFAULT_WHEEL_BINDINGS


# -- Settings > Saving ------------------------------------------------------


def test_the_saving_keys_default_when_the_config_is_empty(tmp_path: Path) -> None:
    assert settings.get_autosave_enabled() is True
    assert settings.get_autosave_interval_min() == settings.AUTOSAVE_INTERVAL_DEFAULT
    assert settings.get_autosave_retention() == settings.AUTOSAVE_RETENTION_DEFAULT
    assert settings.get_autosave_location() == settings.AUTOSAVE_LOCATION_DEFAULT
    assert settings.get_backups_enabled() is True


def test_each_saving_key_round_trips_through_disk(tmp_path: Path) -> None:
    settings.set_autosave_enabled(False)
    settings.set_autosave_interval_min(15)
    settings.set_autosave_retention(10)
    settings.set_autosave_location("sidecar")
    settings.set_backups_enabled(False)

    on_disk = (tmp_path / "config.yaml").read_text()
    assert "autosave_enabled: false" in on_disk
    assert "autosave_interval_min: 15" in on_disk
    assert "autosave_retention: 10" in on_disk
    assert "autosave_location: sidecar" in on_disk
    assert "backups_enabled: false" in on_disk


@pytest.mark.parametrize("key, getter", [
    ("autosave_interval_min: 7\n", "get_autosave_interval_min"),
    ("autosave_retention: 4\n", "get_autosave_retention"),
    ("autosave_location: elsewhere\n", "get_autosave_location"),
])
def test_an_off_list_saving_value_falls_back_on_read(tmp_path: Path, key: str, getter: str) -> None:
    _write_config(tmp_path, key)
    default = {
        "get_autosave_interval_min": settings.AUTOSAVE_INTERVAL_DEFAULT,
        "get_autosave_retention": settings.AUTOSAVE_RETENTION_DEFAULT,
        "get_autosave_location": settings.AUTOSAVE_LOCATION_DEFAULT,
    }[getter]
    assert getattr(settings, getter)() == default


def test_the_membership_gated_setters_refuse_an_off_list_value() -> None:
    """The setter raises rather than falling back: a value the read path
    would silently ignore must never reach disk in the first place."""
    with pytest.raises(ValueError):
        settings.set_autosave_interval_min(7)
    with pytest.raises(ValueError):
        settings.set_autosave_retention(4)
    with pytest.raises(ValueError):
        settings.set_autosave_location("elsewhere")


# GH #127: a custom folder for the central autosave slots.


def test_the_autosave_folder_defaults_to_empty_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_autosave_dir() == ""
    custom = tmp_path / "my autosaves"
    custom.mkdir()
    settings.set_autosave_dir(str(custom))
    assert settings.get_autosave_dir() == str(custom)
    monkeypatch.setattr(settings, "_autosave_dir", None)
    assert settings.get_autosave_dir() == str(custom)
    settings.set_autosave_dir("")
    monkeypatch.setattr(settings, "_autosave_dir", None)
    assert settings.get_autosave_dir() == ""


def test_picking_the_default_folder_itself_stores_empty(tmp_path: Path) -> None:
    from descape import autosave

    settings.set_autosave_dir(str(autosave.autosave_dir()))
    assert settings.get_autosave_dir() == ""


def test_a_hand_written_default_folder_reads_back_as_empty(tmp_path: Path) -> None:
    """Otherwise every tick logs a false "not found; wrote to the default folder"."""
    from descape import autosave

    _write_config(tmp_path, f"autosave_dir: '{autosave.autosave_dir()}'\n")
    assert settings.get_autosave_dir() == ""


@pytest.mark.parametrize("kind", ["subscribed", "template", "relative"])
def test_the_autosave_folder_setter_refuses_and_keeps_the_stored_value(tmp_path: Path, kind: str) -> None:
    from descape.scenario_io import TEMPLATE_DIR

    kept = tmp_path / "kept"
    kept.mkdir()
    settings.set_autosave_dir(str(kept))
    bad = {
        "subscribed": str(tmp_path / "mods" / "subscribed" / "12345"),
        "template": str(TEMPLATE_DIR),
        "relative": "autosaves/here",
    }[kind]
    before = (tmp_path / "config.yaml").read_text()
    with pytest.raises(ValueError):
        settings.set_autosave_dir(bad)
    assert settings.get_autosave_dir() == str(kept)
    assert (tmp_path / "config.yaml").read_text() == before


@pytest.mark.parametrize("raw", ["/x/mods/subscribed/y", "relative/dir", "42", "''", "null", "[/tmp]"])
def test_a_refused_or_malformed_autosave_folder_reads_back_as_default(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"autosave_dir: {raw}\n")
    assert settings.get_autosave_dir() == ""


def test_a_game_profile_autosave_folder_is_kept(tmp_path: Path) -> None:
    profile = tmp_path / "steamapps" / "compatdata" / "813780" / "pfx"
    settings.set_autosave_dir(str(profile))
    assert settings.get_autosave_dir() == str(profile)


def test_pan_speed_defaults_and_round_trips(tmp_path: Path) -> None:
    assert settings.get_pan_speed() == settings.PAN_SPEED_DEFAULT
    settings.set_pan_speed(1234)
    assert settings.get_pan_speed() == 1234
    assert "pan_speed: 1234" in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("raw, expected", [
    ("pan_speed: 10\n", settings.PAN_SPEED_MIN),
    ("pan_speed: 99999\n", settings.PAN_SPEED_MAX),
    ("pan_speed: fast\n", settings.PAN_SPEED_DEFAULT),
    ("pan_speed:\n", settings.PAN_SPEED_DEFAULT),
])
def test_an_out_of_range_or_malformed_pan_speed_falls_back_on_read(
    tmp_path: Path, raw: str, expected: int
) -> None:
    """Clamped rather than refused, unlike the membership-gated setters
    above: the range is continuous, so an out-of-range number has an obvious
    nearest legal value where an off-list enum does not."""
    _write_config(tmp_path, raw)
    assert settings.get_pan_speed() == expected


def test_the_pan_speed_setter_clamps_rather_than_writing_an_illegal_value() -> None:
    settings.set_pan_speed(settings.PAN_SPEED_MAX + 500)
    assert settings.get_pan_speed() == settings.PAN_SPEED_MAX


_NEW_MAP_DEFAULT = {
    "tiles": 120,
    "game_data": "aoe2",
    "terrain_id": 0,
    "elevation": 0,
    "trees": settings.NEW_MAP_DEFAULT_TREES,
    "eye_candy": settings.NEW_MAP_DEFAULT_EYE_CANDY,
}


def test_new_map_choice_defaults_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_new_map_choice() == _NEW_MAP_DEFAULT
    choice = {
        "tiles": 300,
        "game_data": "ror",
        "terrain_id": 35,
        "elevation": 7,
        "trees": not settings.NEW_MAP_DEFAULT_TREES,
        "eye_candy": not settings.NEW_MAP_DEFAULT_EYE_CANDY,
    }
    settings.set_new_map_choice(choice)
    monkeypatch.setattr(settings, "_new_map", None)
    assert settings.get_new_map_choice() == choice
    import yaml

    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["new_map"] == choice


def test_new_map_choice_returns_a_copy() -> None:
    settings.get_new_map_choice()["tiles"] = 200
    assert settings.get_new_map_choice()["tiles"] == 120


@pytest.mark.parametrize("raw, expected", [
    ("new_map: {tiles: 10}\n", {"tiles": 61}),
    ("new_map: {tiles: 9999}\n", {"tiles": 480}),
    ("new_map: {tiles: big}\n", {}),
    ("new_map: {tiles: true}\n", {}),
    ("new_map: {tiles: .inf}\n", {}),
    ("new_map: {game_data: aok}\n", {}),
    ("new_map: {terrain_id: 256}\n", {}),
    ("new_map: {terrain_id: -1}\n", {}),
    ("new_map: {terrain_id: 35, elevation: 99}\n", {"terrain_id": 35, "elevation": 15}),
    ("new_map: {elevation: -3}\n", {"elevation": 0}),
    ("new_map: {trees: 1, eye_candy: 'yes'}\n", {}),
    ("new_map: {trees: null, eye_candy: [true]}\n", {}),
    ("new_map: {trees: false, eye_candy: true}\n", {"trees": False, "eye_candy": True}),
    # A choice saved before Trees and Eye candy existed.
    ("new_map: {tiles: 144, game_data: aoe2, terrain_id: 10, elevation: 2}\n", {"tiles": 144, "terrain_id": 10, "elevation": 2}),
    ("new_map: [1, 2]\n", {}),
    ("new_map:\n", {}),
])
def test_a_bad_new_map_field_falls_back_on_its_own(tmp_path: Path, raw: str, expected: dict) -> None:
    _write_config(tmp_path, raw)
    assert settings.get_new_map_choice() == {**_NEW_MAP_DEFAULT, **expected}


def test_the_new_map_setter_cleans_before_writing(tmp_path: Path) -> None:
    settings.set_new_map_choice(
        {"tiles": 5000, "game_data": "x", "terrain_id": 2, "elevation": 3, "trees": "no", "eye_candy": True}
    )
    assert settings.get_new_map_choice() == {
        **_NEW_MAP_DEFAULT, "tiles": 480, "terrain_id": 2, "elevation": 3, "eye_candy": True
    }


def test_wall_opacity_defaults_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_wall_opacity_pct() == view_layers.WALL_OPACITY_DEFAULT_PCT
    settings.set_wall_opacity_pct(30)
    monkeypatch.setattr(settings, "_wall_opacity_pct", None)
    assert settings.get_wall_opacity_pct() == 30
    assert "wall_opacity_pct: 30" in (tmp_path / "config.yaml").read_text()
    assert settings.get_view_layers().wall_opacity_pct == 30


@pytest.mark.parametrize("raw, expected", [
    ("wall_opacity_pct: 0\n", view_layers.WALL_OPACITY_MIN_PCT),
    ("wall_opacity_pct: 100\n", view_layers.WALL_OPACITY_MAX_PCT),
    ("wall_opacity_pct: half\n", view_layers.WALL_OPACITY_DEFAULT_PCT),
    ("wall_opacity_pct:\n", view_layers.WALL_OPACITY_DEFAULT_PCT),
])
def test_an_out_of_range_or_malformed_wall_opacity_falls_back_on_read(
    tmp_path: Path, raw: str, expected: int
) -> None:
    _write_config(tmp_path, raw)
    assert settings.get_wall_opacity_pct() == expected


def test_the_wall_opacity_setter_clamps_rather_than_writing_an_illegal_value() -> None:
    settings.set_wall_opacity_pct(view_layers.WALL_OPACITY_MAX_PCT + 50)
    assert settings.get_wall_opacity_pct() == view_layers.WALL_OPACITY_MAX_PCT
    settings.set_wall_opacity_pct(-5)
    assert settings.get_wall_opacity_pct() == view_layers.WALL_OPACITY_MIN_PCT


# The refused saved value behind get_autosave_dir()'s "", so a restart can still say why.
# Absolute on this OS: a Windows path needs a drive to be one.
_ABS_SUBSCRIBED = "C:/x/mods/subscribed/y" if os.name == "nt" else "/x/mods/subscribed/y"


@pytest.mark.parametrize("raw, expected", [
    (_ABS_SUBSCRIBED, (_ABS_SUBSCRIBED, "(mods/subscribed/) folder")),
    ("relative/dir", ("relative/dir", "not an absolute path")),
    ("42", ("42", "not a folder path")),
    ("[/tmp]", ("['/tmp']", "not a folder path")),
])
def test_a_refused_autosave_folder_keeps_its_reason(tmp_path: Path, raw: str, expected: tuple[str, str]) -> None:
    _write_config(tmp_path, f"autosave_dir: {raw}\n")
    refusal = settings.get_autosave_dir_refusal()
    assert refusal is not None
    assert refusal[0] == expected[0]
    assert expected[1] in refusal[1]
    assert settings.get_autosave_dir() == ""


@pytest.mark.parametrize("raw", ["''", "null"])
def test_an_unset_autosave_folder_has_no_refusal(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"autosave_dir: {raw}\n")
    assert settings.get_autosave_dir_refusal() is None


def _symlink_loop(tmp_path: Path) -> Path:
    a, b = tmp_path / "loop-a", tmp_path / "loop-b"
    a.symlink_to(b)
    b.symlink_to(a)
    return a


@conftest.requires_symlinks
def test_a_looped_autosave_folder_reads_back_as_default_with_its_reason(tmp_path: Path) -> None:
    loop = _symlink_loop(tmp_path)
    _write_config(tmp_path, f"autosave_dir: '{loop}'\n")
    assert settings.get_autosave_dir() == ""
    refusal = settings.get_autosave_dir_refusal()
    assert refusal is not None
    assert refusal[0] == str(loop)
    assert "cannot be resolved" in refusal[1]


@conftest.requires_symlinks
def test_the_default_folder_check_survives_a_symlink_loop(tmp_path: Path) -> None:
    assert settings._is_default_autosave_dir(str(_symlink_loop(tmp_path))) is False


def test_setting_the_autosave_folder_clears_a_saved_refusal(tmp_path: Path) -> None:
    _write_config(tmp_path, "autosave_dir: relative/dir\n")
    assert settings.get_autosave_dir_refusal() is not None
    settings.set_autosave_dir("")
    assert settings.get_autosave_dir_refusal() is None


# -- GH #139: text box heights ------------------------------------------------


def test_text_box_lines_are_none_until_first_saved_and_round_trip(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_text_box_lines("trigger.description") is None
    settings.set_text_box_lines("trigger.description", 12)
    settings.set_text_box_lines("messages.hints", 9)
    monkeypatch.setattr(settings, "_text_box_lines", None)
    assert settings.get_text_box_lines("trigger.description") == 12
    assert settings.get_text_box_lines("messages.hints") == 9
    assert settings.get_text_box_lines("trigger.message") is None
    assert "trigger.description: 12" in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("raw, expected", [
    ("text_box_lines: {trigger.description: 1}\n", settings.TEXT_BOX_LINES_MIN),
    ("text_box_lines: {trigger.description: 400}\n", settings.TEXT_BOX_LINES_MAX),
    ("text_box_lines: {trigger.description: tall}\n", None),
    ("text_box_lines: {trigger.description: true}\n", None),
    ("text_box_lines: {trigger.description: 7.5}\n", None),
    ("text_box_lines: banana\n", None),
    ("text_box_lines:\n", None),
])
def test_a_malformed_or_out_of_range_text_box_height_is_dropped_or_clamped_on_read(
    tmp_path: Path, raw: str, expected
) -> None:
    _write_config(tmp_path, raw)
    assert settings.get_text_box_lines("trigger.description") == expected


def test_a_malformed_entry_drops_only_itself(tmp_path: Path) -> None:
    _write_config(tmp_path, "text_box_lines: {trigger.description: tall, messages.hints: 10}\n")
    assert settings.get_text_box_lines("trigger.description") is None
    assert settings.get_text_box_lines("messages.hints") == 10


def test_setting_none_deletes_the_key_and_the_last_one_drops_the_block(tmp_path: Path, monkeypatch) -> None:
    settings.set_text_box_lines("trigger.description", 12)
    settings.set_text_box_lines("messages.hints", 9)
    settings.set_text_box_lines("trigger.description", None)
    monkeypatch.setattr(settings, "_text_box_lines", None)
    assert settings.get_text_box_lines("trigger.description") is None
    assert settings.get_text_box_lines("messages.hints") == 9
    settings.set_text_box_lines("messages.hints", None)
    assert "text_box_lines" not in (tmp_path / "config.yaml").read_text()


def test_the_text_box_setter_clamps_and_refuses_a_non_int(tmp_path: Path) -> None:
    settings.set_text_box_lines("trigger.description", 999)
    assert settings.get_text_box_lines("trigger.description") == settings.TEXT_BOX_LINES_MAX
    with pytest.raises(ValueError):
        settings.set_text_box_lines("trigger.description", True)
    with pytest.raises(ValueError):
        settings.set_text_box_lines("trigger.description", "12")
    assert settings.get_text_box_lines("trigger.description") == settings.TEXT_BOX_LINES_MAX


# -- GH #166: trigger status marker --------------------------------------------


def test_the_trigger_status_settings_default_to_colour_with_ok_rows_coloured(tmp_path: Path) -> None:
    assert settings.get_trigger_status_marker() == "color"
    assert settings.get_trigger_status_color_ok_rows() is True
    assert settings.get_trigger_status_color("ok") == "#4caf50"
    assert settings.get_trigger_status_color("problem") == "#e05252"
    assert [mid for mid, _label in settings.TRIGGER_STATUS_MARKERS] == ["color", "icon", "both", "off"]


def test_the_trigger_status_colour_defaults_are_the_viewer_status_colours() -> None:
    pytest.importorskip("PyQt5")
    from descape import viewer

    assert settings.get_default_trigger_status_color("ok") == viewer.STATUS_OK_COLOR
    assert settings.get_default_trigger_status_color("problem") == viewer.STATUS_ERROR_COLOR


@pytest.mark.parametrize("raw", ["colour", "'COLOR'", "0", "true", "[icon]"])
def test_an_off_list_trigger_status_marker_falls_back(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"trigger_status_marker: {raw}\n")
    assert settings.get_trigger_status_marker() == "color"


def test_setting_an_unknown_trigger_status_marker_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_trigger_status_marker("colour")
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("marker", [mid for mid, _label in settings.TRIGGER_STATUS_MARKERS])
def test_the_trigger_status_marker_round_trips(tmp_path: Path, monkeypatch, marker: str) -> None:
    settings.set_trigger_status_marker(marker)
    monkeypatch.setattr(settings, "_trigger_status_marker", None)
    assert settings.get_trigger_status_marker() == marker


# -- TASK-271: toolbar button style --------------------------------------------


def test_the_toolbar_style_defaults_to_icons_and_text(tmp_path: Path) -> None:
    assert settings.get_toolbar_style() == "icons_text"
    assert [sid for sid, _label in settings.TOOLBAR_STYLES] == ["icons", "icons_text", "text"]


@pytest.mark.parametrize("raw", ["icon", "'ICONS'", "0", "true", "[text]"])
def test_an_off_list_toolbar_style_falls_back(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"toolbar_style: {raw}\n")
    assert settings.get_toolbar_style() == "icons_text"


def test_setting_an_unknown_toolbar_style_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_toolbar_style("icon")
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("style", [sid for sid, _label in settings.TOOLBAR_STYLES])
def test_the_toolbar_style_round_trips(tmp_path: Path, monkeypatch, style: str) -> None:
    settings.set_toolbar_style(style)
    monkeypatch.setattr(settings, "_toolbar_style", None)
    assert settings.get_toolbar_style() == style


@pytest.mark.parametrize("raw", ["0", "'false'", "[]", "null"])
def test_a_non_bool_ok_rows_value_reads_as_the_default(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"trigger_status_color_ok_rows: {raw}\n")
    assert settings.get_trigger_status_color_ok_rows() is True


def test_the_ok_rows_toggle_round_trips(tmp_path: Path, monkeypatch) -> None:
    settings.set_trigger_status_color_ok_rows(False)
    monkeypatch.setattr(settings, "_trigger_status_color_ok_rows", None)
    assert settings.get_trigger_status_color_ok_rows() is False


def test_section_rows_are_uncoloured_by_default(tmp_path: Path) -> None:
    """TASK-317: the 2026-10-07 "dividers are never coloured" stays the default."""
    assert settings.get_trigger_status_color_sections() is False


@pytest.mark.parametrize("raw", ["1", "'true'", "[]", "null"])
def test_a_non_bool_color_sections_value_reads_as_false(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"trigger_status_color_sections: {raw}\n")
    assert settings.get_trigger_status_color_sections() is False


@pytest.mark.parametrize("value", [True, False])
def test_the_color_sections_toggle_round_trips(tmp_path: Path, monkeypatch, value: bool) -> None:
    import yaml

    settings.set_trigger_status_color_sections(value)
    monkeypatch.setattr(settings, "_trigger_status_color_sections", None)
    assert settings.get_trigger_status_color_sections() is value
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["trigger_status_color_sections"] is value


def test_a_trigger_status_colour_round_trips_normalized(tmp_path: Path, monkeypatch) -> None:
    settings.set_trigger_status_color("problem", "#ABCDEF")
    monkeypatch.setattr(settings, "_trigger_status_colors", None)
    assert settings.get_trigger_status_color("problem") == "#abcdef"
    assert settings.get_trigger_status_color("ok") == "#4caf50"


def test_a_malformed_trigger_status_colour_keeps_only_its_own_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "trigger_status_colors:\n  ok: notacolour\n  problem: '#112233'\n  bogus: '#000000'\n")
    assert settings.get_trigger_status_color("ok") == "#4caf50"
    assert settings.get_trigger_status_color("problem") == "#112233"


def test_a_non_mapping_trigger_status_colours_block_reads_as_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "trigger_status_colors: '#ff0000'\n")
    assert settings.get_trigger_status_color("ok") == "#4caf50"


def test_setting_a_bad_trigger_status_colour_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_trigger_status_color("problem", "red-ish")
    with pytest.raises(ValueError):
        settings.set_trigger_status_color("warning", "#ff0000")
    assert not (tmp_path / "config.yaml").exists()


# GH #182: Elevation View and View > Layers persist across launches.


def test_terrain_style_and_view_layers_default_on_an_empty_config(tmp_path: Path) -> None:
    assert not (tmp_path / "config.yaml").exists()
    assert settings.get_terrain_style() == "stepped"
    assert settings.get_view_layers() == view_layers.LayerState()


@pytest.mark.parametrize("style", terrain_style.TERRAIN_STYLES)
def test_terrain_style_round_trips(tmp_path: Path, monkeypatch, style: str) -> None:
    settings.set_terrain_style(style)
    monkeypatch.setattr(settings, "_terrain_style", None)
    assert settings.get_terrain_style() == style


@pytest.mark.parametrize("value", ["isometric", "Sloped", 3, "[sloped]", "{}"])
def test_an_unknown_terrain_style_reads_as_the_default(tmp_path: Path, value) -> None:
    _write_config(tmp_path, f"terrain_style: {value}\n")
    assert settings.get_terrain_style() == "stepped"


def test_setting_an_unknown_terrain_style_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_terrain_style("Sloped")
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("spec", view_layers.LAYERS, ids=lambda spec: spec.layer_id)
def test_each_view_layer_round_trips(tmp_path: Path, monkeypatch, spec) -> None:
    settings.set_view_layer(spec.layer_id, not spec.default)
    monkeypatch.setattr(settings, "_view_layers", None)
    got = settings.get_view_layers()
    assert getattr(got, spec.layer_id) is (not spec.default)
    others = {s.layer_id for s in view_layers.LAYERS} - {spec.layer_id}
    assert all(getattr(got, lid) == getattr(view_layers.LayerState(), lid) for lid in others)


def test_view_layer_entries_fall_back_per_entry(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "view_layers:\n  small_trees: true\n  hero_glow: 'no'\n  farm_overlay: 0\n  bogus_layer: false\n",
    )
    assert settings.get_view_layers() == view_layers.LayerState(small_trees=True)


def test_a_non_mapping_view_layers_block_reads_as_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "view_layers: [small_trees]\n")
    assert settings.get_view_layers() == view_layers.LayerState()


def test_setting_an_unknown_view_layer_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        settings.set_view_layer("bogus_layer", True)
    assert not (tmp_path / "config.yaml").exists()


def test_mod_art_exclusions_round_trip_through_the_file(tmp_path: Path, monkeypatch) -> None:
    assert settings.get_mod_art_excluded() == frozenset()
    settings.set_mod_art_excluded({"subscribed//2_B", "local//1_A"})
    monkeypatch.setattr(settings, "_mod_art_excluded", None)
    assert settings.get_mod_art_excluded() == {"subscribed//2_B", "local//1_A"}
    assert "mod_art_excluded:\n- local//1_A\n- subscribed//2_B\n" in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("raw", ["mod_art_excluded: subscribed//x\n", "mod_art_excluded: [3, null]\n"])
def test_a_malformed_mod_art_exclusion_list_reads_as_nothing_excluded(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, raw)
    assert settings.get_mod_art_excluded() == frozenset()


def test_mod_art_inclusions_round_trip_through_the_file(tmp_path: Path, monkeypatch) -> None:
    """GH #264: the opt-ins of mods the game disables, a key of their own beside the opt-outs."""
    assert settings.get_mod_art_included() == frozenset()
    settings.set_mod_art_excluded({"subscribed//2_B"})
    settings.set_mod_art_included({"subscribed//3_Off", "local//1_A"})
    monkeypatch.setattr(settings, "_mod_art_included", None)
    monkeypatch.setattr(settings, "_mod_art_excluded", None)
    assert settings.get_mod_art_included() == {"subscribed//3_Off", "local//1_A"}
    assert settings.get_mod_art_excluded() == {"subscribed//2_B"}
    assert "mod_art_included:\n- local//1_A\n- subscribed//3_Off\n" in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("raw", ["mod_art_included: subscribed//x\n", "mod_art_included: [3, null]\n"])
def test_a_malformed_mod_art_inclusion_list_reads_as_nothing_included(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, raw)
    assert settings.get_mod_art_included() == frozenset()


# --- TASK-272: Settings > Tools' tool option defaults ------------------------


def _raw_config(tmp_path: Path) -> dict:
    import yaml

    path = tmp_path / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def test_an_empty_config_gives_every_schema_default() -> None:
    from descape import tool_options

    assert settings.get_tool_option_defaults() == tool_options.defaults()


def test_only_the_values_that_differ_from_the_schema_default_are_written(tmp_path: Path) -> None:
    from descape import tool_options

    # Load From Toolbar's shape: every option at once, one of them changed.
    settings.set_tool_option_defaults({**tool_options.defaults(), "brush_size": 5})
    assert _raw_config(tmp_path)["tool_options"] == {"brush_size": 5}
    assert settings.get_tool_option_default("brush_size") == 5


def test_setting_a_value_back_to_its_default_removes_the_entry_and_then_the_key(tmp_path: Path) -> None:
    settings.set_tool_option_default("brush_size", 5)
    settings.set_tool_option_default("free_placement", True)
    settings.set_tool_option_default("brush_size", 1)
    assert _raw_config(tmp_path)["tool_options"] == {"free_placement": True}
    settings.set_tool_option_default("free_placement", False)
    assert "tool_options" not in _raw_config(tmp_path)


def test_a_bulk_set_is_one_write(monkeypatch) -> None:
    writes = []
    real = settings._save_config
    monkeypatch.setattr(settings, "_save_config", lambda config: (writes.append(1), real(config)))
    settings.set_tool_option_defaults({"brush_size": 3, "paint_trees": False, "eraser_targets": ["trees"]})
    assert len(writes) == 1


def test_reset_clears_every_override(tmp_path: Path) -> None:
    from descape import tool_options

    settings.set_tool_option_defaults({"brush_size": 3, "paint_trees": False})
    settings.reset_tool_option_defaults()
    assert "tool_options" not in _raw_config(tmp_path)
    assert settings.get_tool_option_defaults() == tool_options.defaults()


@pytest.mark.parametrize(
    "values",
    [
        {"no_such_option": 1},
        {"brush_size": 10},
        {"brush_size": True},
        {"brush_size": 4, "paint_trees": "false"},
        {"eraser_targets": ["units", "bogus"]},
    ],
)
def test_a_bad_name_or_value_raises_and_writes_nothing(tmp_path: Path, monkeypatch, values: dict) -> None:
    settings.set_tool_option_default("off_grid", True)
    before = (tmp_path / "config.yaml").read_bytes()
    with pytest.raises(ValueError):
        settings.set_tool_option_defaults(values)
    assert (tmp_path / "config.yaml").read_bytes() == before
    assert settings.get_tool_option_default("brush_size") == 1, "the memo must not take the good half either"
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    assert settings.get_tool_option_default("off_grid") is True


@pytest.mark.parametrize("raw", ["tool_options: [1, 2]\n", "tool_options: 7\n", "tool_options:\n"])
def test_a_non_mapping_store_reads_as_no_overrides(tmp_path: Path, raw: str) -> None:
    from descape import tool_options

    _write_config(tmp_path, raw)
    assert settings.get_tool_option_defaults() == tool_options.defaults()


def test_junk_entries_degrade_one_at_a_time(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "tool_options:\n"
        "  brush_size: 99\n"
        "  brush_shape: 3\n"
        "  paint_trees: false\n"
        "  free_placement: yes_please\n"
        "  no_such_option: 1\n"
        "  eraser_targets: [units, bogus, trees, true]\n",
    )
    defaults = settings.get_tool_option_defaults()
    assert (defaults["brush_size"], defaults["brush_shape"], defaults["free_placement"]) == (1, "square", False)
    assert defaults["paint_trees"] is False
    assert defaults["eraser_targets"] == ["units", "trees"], "the loader drops a set's bad members, never the set"
    settings.set_tool_option_default("off_grid", True)
    assert _raw_config(tmp_path)["tool_options"] == {
        "paint_trees": False,
        "off_grid": True,
        "eraser_targets": ["units", "trees"],
    }, "an unknown name is dropped on the next write"


def test_a_changed_schema_default_reaches_a_user_who_saved_every_value(monkeypatch) -> None:
    """The point of storing overrides only: a later build's new default still
    reaches someone who once stored a value equal to the old one."""
    import dataclasses

    from descape import tool_options

    settings.set_tool_option_defaults(tool_options.defaults())
    monkeypatch.setattr(settings, "_tool_option_overrides", None)
    newer = dataclasses.replace(tool_options.option("brush_size"), default=3)
    monkeypatch.setitem(tool_options.BY_NAME, "brush_size", newer)
    assert settings.get_tool_option_default("brush_size") == 3


@pytest.mark.parametrize(
    ("legacy_key", "name"),
    [
        ("paint_trees", "paint_trees"),
        ("paint_eye_candy", "paint_eye_candy"),
        ("trigger_area_edit", "area_edit"),
        ("trigger_area_tile_borders", "area_tile_borders"),
    ],
)
def test_a_legacy_write_through_key_is_read_once_as_the_default(tmp_path: Path, legacy_key: str, name: str) -> None:
    from descape import tool_options

    flipped = not tool_options.default(name)
    _write_config(tmp_path, f"{legacy_key}: {str(flipped).lower()}\n")
    assert settings.get_tool_option_default(name) is flipped
    settings.set_tool_option_default("brush_size", 2)  # any write moves it under tool_options
    raw = _raw_config(tmp_path)
    assert legacy_key not in raw
    assert raw["tool_options"] == {"brush_size": 2, name: flipped}


def test_a_tool_options_entry_wins_over_its_legacy_key(tmp_path: Path) -> None:
    _write_config(tmp_path, "paint_trees: false\ntool_options:\n  paint_trees: true\n")
    assert settings.get_tool_option_default("paint_trees") is True
    settings.set_tool_option_default("brush_size", 2)
    assert _raw_config(tmp_path) == {"tool_options": {"brush_size": 2}}
