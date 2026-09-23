"""One source of truth for "point descape.settings at a throwaway config and
forget everything it already memoized".

Redirecting `CONFIG_PATH` before any window exists is the load-bearing part:
`ViewerWindow.closeEvent()` unconditionally calls `settings.set_window_size()`,
and `load_scenario()` records a recent file, so without this a capture tool
writes straight through to the developer's real config.yaml. `descape.settings`
is a bind-by-value importer of `asset_source.CONFIG_PATH`, so there are two
module attributes to redirect, not one.

The memoized globals matter just as much. Each `settings.get_*()` caches its
value in a module global on first read, so a tool that imported anything
touching settings before isolating inherits whatever that read -- and in a
tools/ script several captures share one process, so state also leaks from one
capture to the next. `MEMOIZED_GLOBALS` is the full set, kept honest by
`tests/test_settings_isolation.py`, which discovers them from
`descape/settings.py` itself and fails when this tuple has drifted. It used to
be hand-copied into eight `tools/gen_*_eyeball.py` scripts, in four
different states of staleness.

tests/ uses `MEMOIZED_GLOBALS` through `monkeypatch` (see `tests/conftest.py`)
so values are restored afterwards; tools/ calls `isolate_settings()`, which
sets them outright, since a one-shot script has nothing to restore to.
"""

from __future__ import annotations

from pathlib import Path

MEMOIZED_GLOBALS = (
    "_zoom_centered_on_cursor",
    "_graphics_quality",
    "_pan_speed",
    "_dark_mode",
    "_preload_zoom_levels",
    "_paint_trees",
    "_paint_eye_candy",
    "_distance_ticks",
    "_distance_tick_interval",
    "_stack_badges",
    "_grid_overlay",
    "_grid_follow_elevation",
    "_footprint_outlines",
    "_footprint_scope",
    "_grid_blend",
    "_grid_thickness",
    "_elev_step_pct",
    "_overlay_colors",
    "_ruler_label_font_px",
    "_distance_tick_font_px",
    "_ui_font_family",
    "_ui_font_size",
    "_window_size",
    "_split_sizes",
    "_log_height",
    "_recent_files",
    "_keybinds",
    "_autosave_enabled",
    "_autosave_interval_min",
    "_autosave_retention",
    "_autosave_location",
    "_backups_enabled",
    "_selection_by_owner",
    "_range_rings",
    "_camera_markers",
    "_trigger_overlay",
)


def pin_install_path() -> Path | None:
    """Resolve the AoE2:DE install through the real config (read only) and pin it
    into AOE2DE_INSTALL_PATH, returning it, or None if no install is visible.

    Call before `isolate_settings()`: afterwards `get_install_path()` reads the
    throwaway config and finds nothing. The env var outranks the config, so the
    pinned path survives the redirect while every write still goes to the fake.
    """
    import os

    import descape.asset_source as asset_source_module

    install = asset_source_module.get_install_path()
    if install is not None:
        os.environ["AOE2DE_INSTALL_PATH"] = str(install)
    return install


def isolate_settings(config_dir: Path, *, redirect_asset_source: bool = True) -> Path:
    """Redirect settings persistence into `config_dir/config.yaml` and clear
    every memoized value, returning that path.

    `redirect_asset_source=False` leaves `asset_source.CONFIG_PATH` alone: that
    global is what `get_install_path()` reads, so a tool whose whole point is to
    look at real game assets (tools/gen_terrain_browser_eyeball.py and its .dds
    swatches) must keep it pointing at the real config.
    """
    import descape.asset_source as asset_source_module
    import descape.settings as settings_module

    fake_config_path = config_dir / "config.yaml"
    settings_module.CONFIG_PATH = fake_config_path
    if redirect_asset_source:
        asset_source_module.CONFIG_PATH = fake_config_path
    for name in MEMOIZED_GLOBALS:
        setattr(settings_module, name, None)
    return fake_config_path
