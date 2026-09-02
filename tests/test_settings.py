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

from descape import edge_ticks, iso_geometry, settings


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
    other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
    settings.set_distance_tick_interval(other)
    monkeypatch.setattr(settings, "_distance_tick_interval", None)
    assert settings.get_distance_tick_interval() == other


def test_the_two_tick_keys_are_independent(tmp_path: Path, monkeypatch) -> None:
    """Writing one must not clear the other. Both go through _load_config,
    so a set() that rebuilt the dict instead of updating it would."""
    settings.set_distance_ticks(True)
    other = [n for n in edge_ticks.TICK_INTERVALS if n != edge_ticks.TICK_INTERVAL_DEFAULT][0]
    settings.set_distance_tick_interval(other)
    monkeypatch.setattr(settings, "_distance_ticks", None)
    monkeypatch.setattr(settings, "_distance_tick_interval", None)
    assert settings.get_distance_ticks() is True
    assert settings.get_distance_tick_interval() == other


def test_malformed_yaml_leaves_both_tick_keys_at_their_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "distance_ticks: [unclosed\n")
    assert settings.get_distance_ticks() is False
    assert settings.get_distance_tick_interval() == edge_ticks.TICK_INTERVAL_DEFAULT


# --- the memoized-globals list itself --------------------------------------


def test_every_memoized_global_is_listed_in_conftest() -> None:
    """conftest._SETTINGS_MEMOIZED_GLOBALS is hand-maintained, and a global
    missing from it leaks one test's value into every later test in the same
    process: silent contamination, not a failure. Reflective rather than a
    second hand-written list, which would only move the same hazard.

    Reads annotations rather than live values: by the time this runs, an
    earlier test in the same process may already have populated a memo, so
    "its value is None" is not a test-order-independent question. Every
    memoized setting is written as `_name: T | None = None`, which IS
    order-independent, and the second assertion below stops that convention
    from quietly decaying. Deliberately not a fresh importlib exec of
    settings.py either: that would rebind the real CONFIG_PATH, outside the
    isolation conftest sets up.
    """
    import conftest

    listed = set(conftest._SETTINGS_MEMOIZED_GLOBALS)
    annotations = getattr(settings, "__annotations__", {})
    nullable = {
        name
        for name, annotation in annotations.items()
        if name.startswith("_") and str(annotation).replace(" ", "").endswith("|None")
    }
    assert nullable == listed, "a memoized settings global is missing from conftest's reset list"
    # _DEFAULT_KEYBINDS is the near miss this guards: underscore-prefixed and
    # module-level, but neither nullable nor memoized.
    assert "_DEFAULT_KEYBINDS" not in listed


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
    _write_config(tmp_path, "dark_mode: true\n")
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
