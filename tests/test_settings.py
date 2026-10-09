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

from descape import edge_ticks, grid_overlay, iso_geometry, settings, terrain_style, view_layers


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
    assert settings.get_theme() == "light"
    assert settings.get_window_size() == (settings.DEFAULT_WINDOW_WIDTH, settings.DEFAULT_WINDOW_HEIGHT)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("dark_mode: true\n", "dark"), ("dark_mode: false\n", "light"), ("", "light")],
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


def test_unknown_theme_falls_back_to_light(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: no_such_preset\n")
    assert settings.get_theme() == "light"


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


def test_a_config_with_no_keybinds_block_gets_the_new_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "theme: dark\n")
    assert settings.get_keybind("tool_elevation") == "E"
    assert settings.get_keybind("tool_ruler") == "R"


def test_default_keybinds_have_no_duplicate_sequences() -> None:
    """Nothing in descape/ checks this at runtime (see the open duplicate-
    binding item), so the hand-audited claim that the DEFAULTS are
    collision-free is worth pinning here. Empty means unbound, and any number
    of actions may share that."""
    bound = [key for key in settings._DEFAULT_KEYBINDS.values() if key]
    assert len(bound) == len(set(bound)), sorted(bound)


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
    assert settings.set_keybind("tool_pan", "W") is None


def test_a_saved_edit_disables_binding_loads_as_map_disables_and_is_then_dropped(tmp_path: Path) -> None:
    """Disabled Objects… moved to the Map menu, renaming its keybind id."""
    import yaml

    _write_config(tmp_path, "keybinds:\n  edit_disables: F7\n")
    assert settings.get_keybind("map_disables") == "F7"
    assert "edit_disables" not in settings._load_keybinds()
    assert settings.set_keybind("tool_pan", "F8") is None
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())["keybinds"]
    assert saved["map_disables"] == "F7" and "edit_disables" not in saved


def test_a_migrated_disables_binding_counts_as_customized_in_a_load_time_collision(tmp_path: Path) -> None:
    """edit_undo is declared first, so it would win a tie; the migrated binding
    must instead read as the user's own choice and beat the default."""
    _write_config(tmp_path, "keybinds:\n  edit_disables: Ctrl+Z\n")
    assert settings.get_keybind("map_disables") == "Ctrl+Z"
    assert settings.get_keybind("edit_undo") == ""


def test_a_saved_map_disables_binding_beats_a_stale_edit_disables_one(tmp_path: Path) -> None:
    _write_config(tmp_path, "keybinds:\n  edit_disables: F7\n  map_disables: F9\n")
    assert settings.get_keybind("map_disables") == "F9"


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


@pytest.mark.parametrize("kind", ["compatdata", "template", "relative"])
def test_the_autosave_folder_setter_refuses_and_keeps_the_stored_value(tmp_path: Path, kind: str) -> None:
    from descape.scenario_io import TEMPLATE_DIR

    kept = tmp_path / "kept"
    kept.mkdir()
    settings.set_autosave_dir(str(kept))
    bad = {
        "compatdata": str(tmp_path / "steamapps" / "compatdata" / "813780" / "pfx"),
        "template": str(TEMPLATE_DIR),
        "relative": "autosaves/here",
    }[kind]
    before = (tmp_path / "config.yaml").read_text()
    with pytest.raises(ValueError):
        settings.set_autosave_dir(bad)
    assert settings.get_autosave_dir() == str(kept)
    assert (tmp_path / "config.yaml").read_text() == before


@pytest.mark.parametrize("raw", ["/x/compatdata/y", "relative/dir", "42", "''", "null", "[/tmp]"])
def test_a_refused_or_malformed_autosave_folder_reads_back_as_default(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"autosave_dir: {raw}\n")
    assert settings.get_autosave_dir() == ""


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


# The refused saved value behind get_autosave_dir()'s "", so a restart can still say why.


@pytest.mark.parametrize("raw, expected", [
    ("/x/compatdata/y", ("/x/compatdata/y", "compatdata/ folder")),
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


def test_a_looped_autosave_folder_reads_back_as_default_with_its_reason(tmp_path: Path) -> None:
    loop = _symlink_loop(tmp_path)
    _write_config(tmp_path, f"autosave_dir: '{loop}'\n")
    assert settings.get_autosave_dir() == ""
    refusal = settings.get_autosave_dir_refusal()
    assert refusal is not None
    assert refusal[0] == str(loop)
    assert "cannot be resolved" in refusal[1]


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


@pytest.mark.parametrize("raw", ["0", "'false'", "[]", "null"])
def test_a_non_bool_ok_rows_value_reads_as_the_default(tmp_path: Path, raw: str) -> None:
    _write_config(tmp_path, f"trigger_status_color_ok_rows: {raw}\n")
    assert settings.get_trigger_status_color_ok_rows() is True


def test_the_ok_rows_toggle_round_trips(tmp_path: Path, monkeypatch) -> None:
    settings.set_trigger_status_color_ok_rows(False)
    monkeypatch.setattr(settings, "_trigger_status_color_ok_rows", None)
    assert settings.get_trigger_status_color_ok_rows() is False


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
