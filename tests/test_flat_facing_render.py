"""The render-level half of the 2026-09-08 flat-facing-offset checklist's
in-app pass, automating items 2.4 ("iso unchanged") and 2.5 ("walls, cliffs,
gates and trees look exactly as before") -- both same-session in-process A/Bs
against a real, shown ViewerWindow, matching this suite's standing posture (see
tests/test_seam_viewer.py's own module docstring: there is no baseline PNG
anywhere in this repo).

**Why this is a render-level check and not just tests/test_unit_sprites.py's
existing unit-level ones.** Those pin angle_index()/_frame_for() directly with
the offset handed in as an argument; they stay green even if some future
render.py change threaded FLAT_ANGLE_ZERO_OFFSET_DEG into a path that should
never see it (icon_for()'s own VARIANT branch, or the Stepped/Sloped sprite
path, which never takes an offset parameter for this constant at all). This
file instead drives the real GUI paint dispatch and diffs pixels, the same
technique tests/test_seam_viewer.py established for the iso <-> render <->
viewer link.

Every unit is a bare stand-in, not a real wall/cliff/gate/tree unit_const --
`unit_graphic_map.json`'s real entries are large and one already-fake const
buys the same coverage: `_frame_for()`'s VARIANT branch is reached the same
way for a "wall" as it is for a real one, since the dispatch only looks at
`rotation_is_variant()`'s boolean, never at which const set that name
happens to belong to. What matters is registering each stand-in with
`"rotation_is_variant": True`, the same generated field a real wall,
cliff, gate or tree entry carries (a gate's own angle_count == 1 would
short-circuit either branch to frame 0 regardless, so it adds no extra
coverage beyond the others -- included anyway since the checklist names it).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import conftest
from descape import asset_source, iso_geometry, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH
from test_unit_sprites import build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Fixed anchor, well inside the 120x120 blank template and clear of
# MapView.set_source()'s decorative map-extent outline -- same reasoning as
# tests/test_seam_viewer.py's own _ANCHOR.
_ANCHOR_TX, _ANCHOR_TY = 60, 60

_WALL_CONST, _CLIFF_CONST, _GATE_CONST, _TREE_CONST, _ANGLE_CONST = range(90001, 90006)
_VARIANT_CONSTS = (_WALL_CONST, _CLIFF_CONST, _GATE_CONST, _TREE_CONST)
_FILE_NAME = "t_facing_render_x1"

# A radian encoding of variant index 2 of 5 (unit_sprites.variant_index()'s
# own formula) -- deliberately NOT a literal index. A literal rotation would
# resolve via variant_index()'s early "already an integer" branch, which
# never even computes `steps`, so a future mistake that added the offset
# INSIDE the radian branch's arithmetic would slip past a literal-rotation
# fixture undetected. This exercises that arithmetic instead.
_RADIAN_INDEX_2 = 2 * 2 * math.pi / 5


class _Unit:
    """The bare attributes the sprite/footprint paths read, duck-typed --
    same posture as tests/test_sprite_edit_real_assets.py's own _Unit."""

    def __init__(self, x: float, y: float, unit_const: int, rotation: float) -> None:
        self.x, self.y, self.unit_const, self.rotation = x, y, unit_const, rotation


@pytest.fixture
def facing_render_install(tmp_path, monkeypatch):
    """One shared 8-frame synthetic graphic (build_sld() varies each frame's
    colour by index, so a wrong frame is a wrong pixel) registered under five
    stand-in consts: four VARIANT ("wall"/"cliff"/"gate"/"tree", angle_count
    5) and one ANGLE (angle_count 8, for the Stepped-side control)."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{_FILE_NAME}.sld").write_bytes(build_sld(8, canvas=unit_sprites.NATIVE_TILE_W))

    def variant_entry():
        return {
            "graphic_id": 1, "file_name": _FILE_NAME, "angle_count": 5,
            "mirroring_mode": 6, "frame_count": 1, "rotation_is_variant": True,
        }

    graphics_map = {const: variant_entry() for const in _VARIANT_CONSTS}
    graphics_map[_ANGLE_CONST] = {
        "graphic_id": 2, "file_name": _FILE_NAME, "angle_count": 8,
        "mirroring_mode": 6, "frame_count": 1, "rotation_is_variant": False,
    }
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: graphics_map)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _flat_capture_rect(window, tiles):
    """A QRectF covering `tiles` (plus a one-tile margin) in Flat's plain
    axis-aligned grid -- integer width/height, per testkit.qt_capture's own
    1:1-only constraint (tests/README.md's capture-constraint survey)."""
    from PyQt5.QtCore import QRectF

    tile_px = window.map_view._tile_pixels
    xs = [tx * tile_px for tx, _ty in tiles] + [(tx + 1) * tile_px for tx, _ty in tiles]
    ys = [ty * tile_px for _tx, ty in tiles] + [(ty + 1) * tile_px for _tx, ty in tiles]
    pad = tile_px
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = max(xs) + pad, max(ys) + pad
    return QRectF(x0, y0, x1 - x0, y1 - y0)


def _iso_capture_rect(window, tiles):
    """Stepped's own version of the same rect, in iso screen space -- the
    same construction tests/test_seam_viewer.py's own _bbox_for_tiles uses."""
    from PyQt5.QtCore import QRectF

    proj = window.map_view._iso_proj
    assert proj is not None, "Stepped mode produced no projection"
    xs, ys = [], []
    for tx, ty in tiles:
        sx, sy = iso_geometry.tile_screen_origin(tx, ty, 0, proj)
        xs += [sx, sx + 2 * proj.half_w]
        ys += [sy, sy + 2 * proj.half_h]
    pad = proj.tile_px
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    return QRectF(x0, y0, max(xs) + pad - x0, max(ys) + pad - y0)


def test_flat_variant_sprites_are_unaffected_by_the_flat_facing_offset(facing_render_install):
    """2.5: "walls, cliffs, gates and trees look exactly as before." All four
    resolve through `_frame_for()`'s VARIANT branch, which never reads its
    `angle_offset_deg` argument at all (unit_sprites.variant_index()'s own
    docstring: "NEVER applies ANGLE_ZERO_OFFSET_DEG, by construction") -- so
    an absurd FLAT_ANGLE_ZERO_OFFSET_DEG must produce a byte-identical Flat
    render.
    """
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        tiles = [(_ANCHOR_TX + i, _ANCHOR_TY) for i in range(len(_VARIANT_CONSTS))]
        units = window.scenario.unit_manager.units[1]
        for (tx, ty), const in zip(tiles, _VARIANT_CONSTS):
            units.append(_Unit(float(tx), float(ty), const, _RADIAN_INDEX_2))
        window.refresh_map()

        rect = _flat_capture_rect(window, tiles)
        no_sprites = None
        window.show_sprites_action.setChecked(False)
        window.refresh_map()
        no_sprites = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        window.show_sprites_action.setChecked(True)
        window.refresh_map()
        before = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        assert not np.array_equal(before, no_sprites), "no stand-in painted anything, so this proves nothing"
        assert window.map_view._canvas_item._last_mip == 0, "capture wasn't 1:1"

        original = unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG
        try:
            unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG = 999.0
            unit_sprites.clear_caches()
            window.refresh_map()
            after = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        finally:
            unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG = original
            unit_sprites.clear_caches()

        assert np.array_equal(before, after), "a variant-indexed sprite moved under an absurd flat offset"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_stepped_angle_sprites_are_unaffected_by_the_flat_facing_offset(facing_render_install):
    """2.4: "Ticking View > Isometric View still renders the previously-
    confirmed correct iso facings, i.e. this change did not leak into that
    path." Stepped's own sprite dispatch (sprite_pieces_for() ->
    _draw_for_entry() -> _frame_for()) never takes FLAT_ANGLE_ZERO_OFFSET_DEG
    as an argument at all -- it always resolves at the isometric zero point
    -- so mutating that constant must leave a Stepped capture byte-identical.
    """
    window = conftest.stepped_window(BLANK_TEMPLATE_PATH)
    try:
        tile = (_ANCHOR_TX, _ANCHOR_TY)
        window.scenario.unit_manager.units[1].append(
            _Unit(float(tile[0]), float(tile[1]), _ANGLE_CONST, math.pi / 3)
        )
        window.refresh_map()

        rect = _iso_capture_rect(window, [tile])
        window.show_sprites_action.setChecked(False)
        window.refresh_map()
        no_sprites = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        window.show_sprites_action.setChecked(True)
        window.refresh_map()
        before = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        assert not np.array_equal(before, no_sprites), "the stand-in painted nothing, so this proves nothing"
        assert window.map_view._canvas_item._last_mip == 0, "capture wasn't 1:1"

        original = unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG
        try:
            unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG = 999.0
            unit_sprites.clear_caches()
            window.refresh_map()
            after = conftest.scene_rect_to_array(window.map_view.scene(), rect)
        finally:
            unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG = original
            unit_sprites.clear_caches()

        assert np.array_equal(before, after), "the flat offset leaked into the Stepped sprite path"
    finally:
        window.edit_history.mark_saved()
        window.close()
