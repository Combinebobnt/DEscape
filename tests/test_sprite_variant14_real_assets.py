"""A variant-14 building draws its real sprite in a live window.

The decoder-level checks for SLD layout variant 14 all run on file bytes: a
corpus walk that lands on EOF, a byte-accounting identity, and synthetic
round-trips in tests/test_sld_decoder.py. None of them can tell whether a
placed Stable actually *paints*, which is the thing the variant was fixed for.

Non-vacuity is the whole risk here. `sprite_for()` returning None is a silent,
supported outcome -- it just falls back to a coloured mark -- so a render
comparison that forgot to resolve the sprite would compare two mark-only
canvases and pass. Every assertion below is therefore staged: the sprite layer
must exist, it must be anchored on the Stable, and the sprites-on canvas must
differ from the sprites-off one.

Needs AOE2DE_INSTALL_PATH rather than the configured install, for the reason
tests/test_sprite_edit_real_assets.py's docstring spells out: conftest's
autouse _isolated_settings hides config.yaml from the suite on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import asset_source, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# The Stable. Its graphic, b_west_stable_age3_x1, is layout variant 14 -- one
# of the 226 files the decoder rejected outright until 2026-08, and the exact
# const used below.
STABLE_CONST = 86
STABLE_GRAPHIC = "b_west_stable_age3_x1"
BASE_ELEVATION = 7
ANCHOR_X = ANCHOR_Y = 60


class _Unit:
    """The four attributes the sprite/footprint paths read, duck-typed -- the
    posture the other real-asset sprite tests already take."""

    def __init__(self, x: float, y: float, unit_const: int) -> None:
        self.x, self.y, self.unit_const, self.rotation = x, y, unit_const, 0.0


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    entry = unit_sprites.graphic_map().get(STABLE_CONST)
    if entry is None or entry.get("file_name") != STABLE_GRAPHIC:
        pytest.skip(f"unit_const {STABLE_CONST} does not map to {STABLE_GRAPHIC} on this install")


def _place_stable(scenario) -> None:
    for tile in scenario.map_manager.terrain:
        tile.elevation = BASE_ELEVATION
    scenario.unit_manager.units[1].append(_Unit(float(ANCHOR_X), float(ANCHOR_Y), STABLE_CONST))


def test_the_variant_fourteen_decoder_reaches_sprite_for():
    """The decode path end to end, below the renderer.

    Guards the specific regression that matters: if `_advance` ever loses its
    header-relative alignment, this returns None and the Stable silently drops
    back to a coloured mark with no test failing anywhere else.
    """
    _require_install()
    sprite = unit_sprites.sprite_for(STABLE_CONST, rotation=0.0, team_index=1, half_w=48)
    assert sprite is not None, (
        f"{STABLE_GRAPHIC} is SLD layout variant 14; returning None here means the "
        "decoder stopped reading that variant and the Stable draws a mark again"
    )


def test_a_placed_stable_paints_its_sprite_not_a_coloured_mark():
    """A real window, real assets, the Show sprites toggle, one placed Stable."""
    _require_install()
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        _place_stable(window.scenario)
        window._render_current()
        window.show_sprites_action.setChecked(True)
        assert window._cache.sprites_enabled is True

        sprites = window._cache._level(0).sprites
        assert sprites is not None and sprites.by_anchor, (
            "no sprite layer built -- the Stable's .sld did not resolve, so the "
            "canvas comparison below would compare two coloured-mark renders"
        )

        with_sprites, _e, _p = render.render_terrain_iso_with_proj(window.scenario, with_sprites=True)
        without, _e, _p = render.render_terrain_iso_with_proj(window.scenario, with_sprites=False)
        assert not np.array_equal(with_sprites, without), (
            "enabling sprites changed nothing on a map whose only unit is a Stable"
        )

        # The sprite must be substantially larger than the mark it replaces: a
        # decode that produced a few stray blocks would differ from the
        # mark-only render too, and pass an equality-only check.
        changed = np.count_nonzero(np.any(with_sprites != without, axis=-1))
        assert changed > 5000, f"only {changed} pixels differ -- too few for a whole building"
    finally:
        window.edit_history.mark_saved()
        window.close()
