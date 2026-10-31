"""Map > Resize Map…'s live preview (TASK-547, GH #45): that the picture agrees
with the write path, follows the current view, and marks the right objects.

Pixel claims are measured from widget.grab(), against the module's own
transform and the source's own colours, never pinned values.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from descape.scenario_io import TERRAIN_STRUCT_SIZE, load_map_and_units
from descape.scenario_resize import Anchor, ResizePlan, origin_offset, removal_closure, resize_body, resize_plan
from descape.scenario_write import build_patched

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_SOURCE_TILE = 255  # a terrain byte the write path's filler never writes
_REFUSED = "refused for this test"
_MAP = (140, 180, 90)
_GROUND = (20, 60, 120)


def _plan(old, new, anchor, *, refusal=None, removed=()) -> ResizePlan:
    dx, dy = origin_offset(*old, *new, anchor)
    return ResizePlan(
        old[0], old[1], new[0], new[1], anchor, dx, dy, 0, 0, 0, len(removed), "none", "", refusal, 0, tuple(removed)
    )


def _unit(x, y, reference_id=1, garrisoned_in_id=-1):
    return SimpleNamespace(x=x, y=y, reference_id=reference_id, garrisoned_in_id=garrisoned_in_id)


def _source(w=100, h=100, *, isometric=True, gaia=()):
    from descape.resize_preview import PreviewSource

    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = _MAP
    return PreviewSource(rgb, _GROUND, isometric, frozenset(id(u) for u in gaia))


def _grab(widget) -> np.ndarray:
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    return qimage_rgb888_to_array(widget.grab().toImage().convertToFormat(QImage.Format_RGB888))


def _preview(source, plan=None):
    from descape.resize_preview import ResizePreview

    conftest.ensure_qapp()
    widget = ResizePreview(source)
    if plan is not None:
        widget.set_plan(plan)
    return widget


def _pixel(pixels, widget, plan, x, y) -> tuple[int, ...]:
    """The grabbed colour at new-map tile point (x, y)."""
    from PyQt5.QtCore import QPointF

    from descape.resize_preview import preview_extent, tile_transform

    extent = preview_extent(plan)
    at = tile_transform(extent, widget.source.isometric, widget.width(), widget.height()).map(QPointF(x, y))
    return tuple(int(v) for v in pixels[int(at.y()), int(at.x())])


@pytest.fixture(scope="module")
def scenario():
    return load_map_and_units(UNITS_FIXTURE)


# --- preview_extent() ---------------------------------------------------------


def test_the_extent_is_the_union_of_both_maps_for_every_anchor() -> None:
    from descape.resize_preview import old_rect, preview_extent

    sizes = (((100, 100), (140, 140)), ((100, 100), (71, 71)), ((100, 100), (137, 137)), ((120, 80), (90, 150)))
    bad = []
    for old, new in sizes:
        for anchor in Anchor:
            plan = _plan(old, new, anchor)
            x, y, w, h = old_rect(plan)
            assert (x, y, w, h) == (plan.dx, plan.dy, *old)
            want = (min(0, x), min(0, y), max(new[0], x + w), max(new[1], y + h))
            if preview_extent(plan) != want:
                bad.append((old, new, anchor, preview_extent(plan), want))
    assert bad == []


def test_a_refused_plan_extends_over_the_old_map_alone() -> None:
    from descape.resize_preview import old_rect, preview_extent

    plan = _plan((100, 100), (140, 90), Anchor.BOTTOM_RIGHT, refusal=_REFUSED)
    assert (plan.dx, plan.dy) != (0, 0)
    assert preview_extent(plan) == (0, 0, 100, 100)
    assert old_rect(plan) == (0, 0, 100, 100)


# --- agreement with the write path --------------------------------------------


def _source_marked(scenario, patched):
    """`patched` with every old tile carrying the marker and its own (x, y)."""
    mm = scenario.map_manager
    block = bytearray()
    for y in range(mm.map_height):
        for x in range(mm.map_width):
            block += bytes((_SOURCE_TILE, 0, x, y)) + bytes(TERRAIN_STRUCT_SIZE - 4)
    t = patched.terrain_offset
    return dataclasses.replace(patched, body=patched.body[:t] + bytes(block) + patched.body[t + len(block) :])


def test_the_preview_shows_what_resize_body_writes_for_all_nine_anchors(scenario) -> None:
    from descape.resize_preview import anchor_point, old_rect

    patched = _source_marked(scenario, build_patched(scenario))
    mm = scenario.map_manager
    old_w, old_h = mm.map_width, mm.map_height
    t = patched.terrain_offset
    bad = []
    for size in (133, 101):  # odd deltas, so CENTER's floor is in play
        for anchor in Anchor:
            plan = resize_plan(scenario, size, size, anchor)
            assert plan.refusal is None
            body = resize_body(scenario, size, size, anchor, patched)
            tiles = np.frombuffer(body, dtype=np.uint8, count=size * size * TERRAIN_STRUCT_SIZE, offset=t)
            tiles = tiles.reshape(size, size, TERRAIN_STRUCT_SIZE)

            kept = tiles[..., 0] == _SOURCE_TILE
            x, y, w, h = old_rect(plan)
            shown = np.zeros((size, size), dtype=bool)
            shown[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)] = True
            if not np.array_equal(kept, shown):
                bad.append((size, anchor, "rect"))
                continue
            ny, nx = np.nonzero(kept)
            if not ((tiles[ny, nx, 2] == nx - x).all() and (tiles[ny, nx, 3] == ny - y).all()):
                bad.append((size, anchor, "source tile"))

            # From the bytes alone: the ring sits on the spot the write path pinned.
            column, row = anchor.value
            written_dx = int(nx[0]) - int(tiles[ny[0], nx[0], 2])
            written_dy = int(ny[0]) - int(tiles[ny[0], nx[0], 3])
            ring_x, ring_y = anchor_point(plan)
            if abs(column / 2 * old_w + written_dx - ring_x) > 0.5 or abs(row / 2 * old_h + written_dy - ring_y) > 0.5:
                bad.append((size, anchor, "ring"))
    assert bad == []


# --- tile_transform() ---------------------------------------------------------


def test_the_isometric_corners_land_on_the_compass_tips() -> None:
    from PyQt5.QtCore import QPointF

    from descape.resize_preview import tile_transform

    w, h = 120, 80
    transform = tile_transform((0, 0, w, h), True, 300, 220)
    west, north, east, south = (transform.map(QPointF(x, y)) for x, y in ((0, 0), (w, 0), (w, h), (0, h)))
    assert west.x() == min(p.x() for p in (west, north, east, south))
    assert north.y() == min(p.y() for p in (west, north, east, south))
    assert east.x() == max(p.x() for p in (west, north, east, south))
    assert south.y() == max(p.y() for p in (west, north, east, south))
    assert (west.x() + east.x()) / 2 == pytest.approx(150)
    assert (north.y() + south.y()) / 2 == pytest.approx(110)


def test_flat_keeps_the_corners_as_corners_at_one_scale() -> None:
    from PyQt5.QtCore import QPointF

    from descape.resize_preview import tile_transform

    transform = tile_transform((0, 0, 120, 80), False, 300, 220)
    top_left, bottom_right = transform.map(QPointF(0, 0)), transform.map(QPointF(120, 80))
    corner = transform.map(QPointF(120, 0))
    assert corner.y() == pytest.approx(top_left.y()) and corner.x() > top_left.x()
    assert (bottom_right.x() - top_left.x()) / 120 == pytest.approx((bottom_right.y() - top_left.y()) / 80)


def test_every_anchor_stays_inside_the_widget_in_both_views() -> None:
    from PyQt5.QtCore import QPointF

    from descape.resize_preview import PREVIEW_HEIGHT, PREVIEW_WIDTH, preview_extent, tile_transform

    bad = []
    for old, new in (((240, 240), (480, 480)), ((240, 240), (61, 61)), ((120, 80), (90, 150))):
        for anchor in Anchor:
            for isometric in (True, False):
                x0, y0, x1, y1 = extent = preview_extent(_plan(old, new, anchor))
                transform = tile_transform(extent, isometric, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                    at = transform.map(QPointF(*corner))
                    if not (0 <= at.x() <= PREVIEW_WIDTH and 0 <= at.y() <= PREVIEW_HEIGHT):
                        bad.append((old, new, anchor, isometric, corner))
    assert bad == []


# --- terrain_rgb() ------------------------------------------------------------


def test_pixel_y_x_is_tile_x_y() -> None:
    from descape.resize_preview import terrain_rgb
    from descape.terrain_palette import color_for_terrain_id, terrain_ids

    w, h = 5, 3
    ids = [terrain_ids()[i % len(terrain_ids())] for i in range(w * h)]
    terrain = [SimpleNamespace(terrain_id=t) for t in ids]
    rgb = terrain_rgb(SimpleNamespace(map_manager=SimpleNamespace(map_width=w, map_height=h, terrain=terrain)))
    assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8 and rgb.flags["C_CONTIGUOUS"]
    for y in range(h):
        for x in range(w):
            assert tuple(rgb[y, x]) == color_for_terrain_id(ids[y * w + x])


def test_a_pending_terrain_edit_shows() -> None:
    from descape.resize_preview import terrain_rgb
    from descape.terrain_palette import color_for_terrain_id, terrain_ids

    loaded = load_map_and_units(UNITS_FIXTURE)
    mm = loaded.map_manager
    x, y = 7, 11
    tile = mm.terrain[y * mm.map_width + x]
    before = terrain_rgb(loaded)
    new_id = next(t for t in terrain_ids() if color_for_terrain_id(t) != tuple(before[y, x]))
    tile.terrain_id = new_id
    after = terrain_rgb(loaded)
    assert tuple(after[y, x]) == color_for_terrain_id(new_id)
    assert int(np.count_nonzero((after != before).any(axis=2))) == 1


# --- which deleted units get a mark -------------------------------------------


def test_the_marks_say_only_what_the_dark_zone_does_not() -> None:
    from descape.resize_preview import deleted_marks

    player_off = _unit(95.5, 95.5, 1)
    gaia_off = _unit(96.5, 96.5, 2)
    gaia_straddler = _unit(49.5, 49.5, 3)
    player_straddler = _unit(48.5, 48.5, 4)
    occupant = _unit(95.5, 40.5, 5, garrisoned_in_id=1)
    gaia_occupant = _unit(40.5, 95.5, 6, garrisoned_in_id=4)
    self_held = _unit(94.5, 94.5, 7, garrisoned_in_id=7)
    removed = (player_off, gaia_off, gaia_straddler, player_straddler, occupant, gaia_occupant, self_held)
    plan = _plan((100, 100), (50, 50), Anchor.TOP_LEFT, removed=removed)
    gaia_ids = frozenset(id(u) for u in (gaia_off, gaia_straddler, gaia_occupant))
    assert deleted_marks(plan, gaia_ids) == [(95.5, 95.5), (49.5, 49.5), (48.5, 48.5), (94.5, 94.5)]


def test_the_marks_are_in_new_map_tiles() -> None:
    from descape.resize_preview import deleted_marks

    plan = _plan((100, 100), (60, 60), Anchor.BOTTOM_RIGHT, removed=(_unit(10.5, 20.5), _unit(70.5, 45.5)))
    assert (plan.dx, plan.dy) == (-40, -40)
    assert deleted_marks(plan, frozenset()) == [(-29.5, -19.5), (30.5, 5.5)]


def test_the_plan_carries_exactly_the_units_it_counted(scenario) -> None:
    plan = resize_plan(scenario, 100, 100, Anchor.BOTTOM_RIGHT)
    assert plan.units_deleted > 0 and len(plan.removed) == plan.units_deleted
    closure = removal_closure(scenario.unit_manager.units, plan.dx, plan.dy, 100, 100, old_w=120, old_h=120)
    assert {id(u) for u in plan.removed} == {id(u) for u in closure}
    assert resize_plan(scenario, 121, 121, Anchor.TOP_LEFT).removed == ()


# --- pixels -------------------------------------------------------------------


def test_a_shrink_keeps_the_tip_it_anchors_and_darkens_the_opposite_one() -> None:
    from descape.resize_preview import PREVIEW_HEIGHT, PREVIEW_WIDTH

    # Fixed widget pixels, inward of each tip of the fitted diamond: the screen is the claim, not the transform.
    cx, cy = PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2
    west, east, north, south = (cy, cx - 105), (cy, cx + 105), (cy - 50, cx), (cy + 50, cx)
    table = (
        (Anchor.BOTTOM_RIGHT, east, west),
        (Anchor.TOP_LEFT, west, east),
        (Anchor.TOP_RIGHT, north, south),
        (Anchor.BOTTOM_LEFT, south, north),
    )
    bad = []
    for anchor, kept, cut in table:
        pixels = _grab(_preview(_source(), _plan((100, 100), (70, 70), anchor)))
        if tuple(pixels[kept]) != _MAP or int(pixels[cut].sum()) >= sum(_MAP) / 2:
            bad.append((anchor, tuple(pixels[kept]), tuple(pixels[cut])))
    assert bad == []


def test_a_grow_from_center_lightens_the_ring_and_leaves_the_middle() -> None:
    plan = _plan((100, 100), (140, 140), Anchor.CENTER)
    widget = _preview(_source(), plan)
    pixels = _grab(widget)
    assert _pixel(pixels, widget, plan, 45, 45) == _MAP
    pale = _pixel(pixels, widget, plan, 8, 70)
    assert all(lit > dim for lit, dim in zip(pale, _GROUND, strict=True))


def test_the_flat_view_is_a_square_with_the_same_zones() -> None:
    from descape.resize_preview import PREVIEW_HEIGHT, PREVIEW_WIDTH

    cx, cy = PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2
    pixels = _grab(_preview(_source(isometric=False), _plan((100, 100), (70, 70), Anchor.BOTTOM_RIGHT)))
    assert tuple(pixels[cy + 80, cx + 80]) == _MAP
    assert int(pixels[cy - 80, cx - 80].sum()) < sum(_MAP) / 2
    # A diamond would leave these corners background; the square reaches them.
    assert int(pixels[cy - 80, cx + 80].sum()) != int(pixels[0, 0].sum())
    assert int(pixels[cy + 80, cx - 80].sum()) != int(pixels[0, 0].sum())


def test_each_tile_of_the_old_map_is_drawn_at_its_own_place() -> None:
    plan = _plan((80, 60), (120, 90), Anchor.BOTTOM_RIGHT)
    rgb = np.empty((60, 80, 3), dtype=np.uint8)
    rgb[:] = _MAP
    corners = {(10, 10): (250, 0, 0), (70, 10): (0, 250, 0), (70, 50): (0, 0, 250), (10, 50): (250, 250, 0)}
    for (x, y), color in corners.items():
        rgb[y - 8 : y + 8, x - 8 : x + 8] = color
    for isometric in (True, False):
        source = dataclasses.replace(_source(), rgb=rgb, isometric=isometric)
        widget = _preview(source, plan)
        pixels = _grab(widget)
        for (x, y), color in corners.items():
            assert _pixel(pixels, widget, plan, x + plan.dx, y + plan.dy) == color, (isometric, x, y)


def test_a_mark_lands_on_its_unit() -> None:
    unit = _unit(70.5, 45.5)
    plan = _plan((100, 100), (60, 60), Anchor.BOTTOM_RIGHT, removed=(unit,))
    widget = _preview(_source(), plan)
    red, green, blue = _pixel(_grab(widget), widget, plan, unit.x + plan.dx, unit.y + plan.dy)
    assert red > 2 * green and red > 2 * blue


def test_a_refused_plan_grabs_the_same_as_the_bare_map() -> None:
    source = _source()
    bare = _grab(_preview(source))
    refused = _plan((100, 100), (140, 90), Anchor.BOTTOM_RIGHT, refusal=_REFUSED)
    assert np.array_equal(_grab(_preview(source, refused)), bare)
    assert not np.array_equal(_grab(_preview(source, _plan((100, 100), (140, 140), Anchor.BOTTOM_RIGHT))), bare)


# --- the dialog ---------------------------------------------------------------


def _dialog(*, preview=True, note=lambda _width: ""):
    """A dialog on a 100x100 fake document; `note(width)` extends the summary."""
    from descape.resize_dialog import ResizeDialog

    conftest.ensure_qapp()

    def plan_for(width, height, anchor):
        refusal = "The map is already 100×100." if (width, height) == (100, 100) else None
        return dataclasses.replace(_plan((100, 100), (width, height), anchor, refusal=refusal), trigger_note=note(width))

    return ResizeDialog(None, 100, 100, plan_for, preview=_source() if preview else None)


def test_a_spin_and_an_anchor_change_both_reach_the_widget() -> None:
    dialog = _dialog()
    try:
        assert dialog.preview.plan is dialog.plan and dialog.plan.refusal is not None
        dialog.width_spin.setValue(70)
        assert (dialog.preview.plan.new_w, dialog.preview.plan.new_h) == (70, 70)
        assert dialog.preview.plan is dialog.plan
        dialog.anchor_buttons[Anchor.BOTTOM_RIGHT].setChecked(True)
        assert dialog.preview.plan.anchor is Anchor.BOTTOM_RIGHT
        assert dialog.preview.plan is dialog.plan
    finally:
        dialog.deleteLater()


def test_no_source_no_widget() -> None:
    dialog = _dialog(preview=False)
    try:
        assert dialog.preview is None
        dialog.width_spin.setValue(70)  # a refresh with nothing to tell
        assert dialog.plan.new_w == 70
    finally:
        dialog.deleteLater()


def test_resize_enables_as_it_does_without_a_preview() -> None:
    from PyQt5.QtWidgets import QDialogButtonBox

    dialogs = _dialog(), _dialog(preview=False)
    try:
        for dialog in dialogs:
            ok = dialog.buttons.button(QDialogButtonBox.Ok)
            assert ok.text() == "Resize" and not ok.isEnabled()
            dialog.width_spin.setValue(70)
            assert ok.isEnabled()
            dialog.width_spin.setValue(100)
            assert not ok.isEnabled()
    finally:
        for dialog in dialogs:
            dialog.deleteLater()


def test_a_longer_summary_grows_the_shown_dialog_and_a_shorter_one_does_not_shrink_it() -> None:
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog(note=lambda width: "A much longer note, on and on. " * 40 if width == 70 else "")
    try:
        dialog.show()
        QApplication.processEvents()
        opened = dialog.height()
        dialog.width_spin.setValue(70)
        QApplication.processEvents()
        assert dialog.height() > opened
        label = dialog.summary_label
        assert label.height() >= label.heightForWidth(label.width())
        grown = dialog.height()
        dialog.width_spin.setValue(90)
        QApplication.processEvents()
        assert dialog.height() == grown
    finally:
        dialog.close()
        dialog.deleteLater()


def test_the_legend_is_not_clipped_as_it_opens() -> None:
    from PyQt5.QtWidgets import QApplication

    dialog = _dialog()
    try:
        dialog.show()
        QApplication.processEvents()
        for label in (dialog.legend_label, dialog.summary_label):
            assert label.height() >= label.heightForWidth(label.width())
        assert dialog.legend_label.width() == dialog.preview.width()
    finally:
        dialog.close()
        dialog.deleteLater()


# --- the window ---------------------------------------------------------------


@pytest.fixture
def window(tmp_path):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    path = tmp_path / "units.aoe2scenario"
    shutil.copyfile(UNITS_FIXTURE, path)
    w = ViewerWindow()
    w.load_scenario(path)
    yield w
    w.edit_history.mark_saved()
    w.close()


def _cancelled_dialogs(monkeypatch) -> list:
    import descape.viewer as viewer_module
    from descape.resize_dialog import ResizeDialog

    shown = []

    def exec_(self):
        shown.append(self)
        return viewer_module.QDialog.Rejected

    monkeypatch.setattr(ResizeDialog, "exec_", exec_)
    return shown


def test_the_source_follows_terrain_style_and_isometric_view(window, monkeypatch) -> None:
    from descape.terrain_style import ELEVATED_STYLES, label_for

    shown = _cancelled_dialogs(monkeypatch)

    def opened_isometric(style: str, iso_checked: bool) -> bool:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText(label_for(style))
        assert window._terrain_style == style
        window.iso_action.setChecked(iso_checked)
        shown.clear()
        window.resize_map()
        return shown[0].preview.source.isometric

    assert opened_isometric("flat", False) is False
    assert opened_isometric("flat", True) is True
    for style in ELEVATED_STYLES:
        assert opened_isometric(style, False) is True


def test_opening_and_cancelling_leaves_the_document_clean_and_its_terrain_unchanged(window, monkeypatch) -> None:
    from descape.resize_preview import terrain_rgb

    shown = _cancelled_dialogs(monkeypatch)
    before = window.scenario
    rgb = terrain_rgb(window.scenario)
    was_dirty = window.edit_history.is_dirty
    window.resize_map()
    assert len(shown) == 1
    assert window.scenario is before and window.edit_history.is_dirty == was_dirty and not was_dirty
    assert np.array_equal(terrain_rgb(window.scenario), rgb)
