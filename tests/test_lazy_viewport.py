"""Phase B-C's own verify script was never written (the plan called for
tools/verify_lazy_viewport.py). This is that check, written natively
against tests/README.md's now-stated convention ("New tools/verify_*.py
scripts are not the convention going forward") instead.

B-C replaced Stepped mode's single monolithic QPixmap with MapCanvasItem
(descape/viewer.py), a QGraphicsItem whose paint() composites only the
descape.render_cache.IsoChunkCache chunks its own exposedRect actually needs.
ItemUsesExtendedStyleOption is the flag that makes that lazy at all --
without it Qt always reports the item's FULL boundingRect as "exposed",
which silently degrades back to full-canvas cost with none of the
memory/latency upside a chunk cache exists for (see MapCanvasItem's own
docstring). tools/verify_iso_chunks.py proves the chunk cache is correct
in isolation; nothing before this file drove it through an actual Qt paint
cycle to prove the flag does its job on real geometry.

IMPORTANT, found while writing this: QGraphicsScene.render() -- the
technique tools/verify_iso_viewer_pick.py's own _scene_rect_to_array()
uses to grab pixels -- does NOT narrow option.exposedRect to the requested
source rect; it always hands every item its full boundingRect (confirmed
against a minimal reproduction, independent of this project's code). That
makes it fine for the byte-identity check below (check 3), but useless for
checks 1/2, which need a REAL QGraphicsView paint cycle -- checked here by
actually showing the view (offscreen platform), zooming in past fit (past
that point the whole map genuinely no longer fits in the viewport, unlike
right after the app's own fitInView-on-open, where near-full compositing
is correct, not a bug -- see IsoChunkCache's own docstring), and panning
via the scrollbars, exactly the technique descape/viewer.py's own
wheelEvent/mouseMoveEvent use (self.scale(...), horizontalScrollBar().
setValue(...)), each followed by QApplication.processEvents() the way
ViewerWindow._render_current's own wait-cursor comment explains Qt event
delivery requires.

Checks, per real example file that supports elevation editing (mirroring
tools/verify_iso_viewer_pick.py's own gate -- terrain_write_supported and
map_is_square; a fixture-tier trio also runs by default against the shipped
blank template descape/templates/blank_120x120.aoe2scenario, which satisfies
both):

1. exposedRect smaller than boundingRect -- once zoomed in past fit, a real
   paint cycle hands MapCanvasItem.paint() an option.exposedRect that is a
   strict, substantially smaller subset of its own boundingRect(), not the
   whole thing.
2. Panning composites proportionally to the viewport, not the map --
   zooming in and panning through the scrollbars touches only a fraction of
   the chunk grid, not every chunk on the map.
3. Byte identity -- a scene render over a fixed rect matches the
   corresponding crop of an independent full render_terrain_iso() call.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import conftest
from descape.render import render_terrain_iso
from testkit import qt_capture

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH

# PyQt5/descape.viewer are NOT imported at module level -- both pull in
# PyQt5, and an ImportError here would fail collection for every test in the
# suite, not just this module's, if PyQt5 is missing (see tests/conftest.py's
# own docstring on why verify_iso_viewer_pick.py/verify_copy_paste.py's real
# imports stay lazy, inside each check, for the same reason). pytestmark
# below skips this whole module up front when that's the case; the imports
# happen for real only once collection has already decided to run it.
# descape.scenario_io itself doesn't import PyQt5, so importing FIXTURE_PATH
# from it above is safe at module level.
pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")

# Sub-rect used by check 3 -- interior, well clear of the canvas edge, and
# small relative to any real map (smallest corpus file and the fixture are
# both 120x120 tiles) so it stays inside bounds regardless of tile_px.
PROBE_ORIGIN = (64, 64)
PROBE_SIZE = 96

# Zoom applied beyond fit-to-view before checks 1/2 start recording --
# mirrors the ~6x fit the B-C measurement pass itself used. At fit-to-view
# the ENTIRE map is visible,
# so near-full compositing there is correct, not a bug; only past this point
# does part of the canvas genuinely fall outside the viewport.
ZOOM_FACTOR = 6.0

# Scrollbar steps (in view pixels) for the scripted "pan" in checks 1/2.
PAN_STEPS = 4
PAN_STEP_PX = 150


# Item-agnostic (QGraphicsScene.render(), not reading a specific item's
# pixmap), so it exercises the real paint dispatch MapCanvasItem.paint() is
# fed through. Only used by check 3 here -- see the module docstring for why
# this specific API can't exercise checks 1/2.
_scene_rect_to_array = qt_capture.scene_rect_to_array


def _stepped_window(path: Path) -> "ViewerWindow":
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{path.name} failed to load")
    window.terrain_style_combo.setCurrentText("Stepped")
    return window


def _show_and_settle(window) -> None:
    """Shows the top-level WINDOW (not just map_view directly -- map_view is
    a child widget embedded in the window's layout, and calling show() on a
    child alone, confirmed empirically, never triggers a real paint cycle;
    only its top-level ancestor being shown does, offscreen platform or
    not) and pumps the event loop until the initial fitInView-on-open paint
    has actually happened.

    Callers that want to measure exposedRect/chunk-touch behavior must
    install their spies AFTER this returns, then call _zoom_and_pan() --
    the fit-to-view state legitimately exposes the whole canvas (the
    entire map IS on screen then; see IsoChunkCache's own docstring on why
    that's correct, not a bug), so a spy installed before this would have
    that call polluting what's supposed to be a "zoomed in past fit"
    measurement."""
    from PyQt5.QtWidgets import QApplication

    window.resize(300, 300)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()


def _zoom_and_pan(map_view) -> None:
    """Zooms in by ZOOM_FACTOR the same way wheelEvent does (self.scale(...)),
    then walks the scrollbars PAN_STEPS times the same way a middle-drag pan
    does (horizontalScrollBar()/verticalScrollBar().setValue()), each step
    followed by processEvents() so Qt actually delivers the resulting paint
    event before the next step. Call _show_and_settle() first."""
    from PyQt5.QtWidgets import QApplication

    map_view.scale(ZOOM_FACTOR, ZOOM_FACTOR)
    QApplication.processEvents()

    hbar, vbar = map_view.horizontalScrollBar(), map_view.verticalScrollBar()
    for _ in range(PAN_STEPS):
        hbar.setValue(hbar.value() + PAN_STEP_PX)
        vbar.setValue(vbar.value() + PAN_STEP_PX // 2)
        QApplication.processEvents()


def _check_exposed_rect_smaller(path: Path) -> tuple[bool, str]:
    conftest.ensure_qapp()
    window = _stepped_window(path)
    try:
        _show_and_settle(window)
        item = window.map_view._canvas_item
        if item is None:
            return False, "Stepped mode produced no MapCanvasItem"
        bounding = item.boundingRect()
        bounding_area = bounding.width() * bounding.height()

        exposed = []
        original_paint = item.paint

        def spy_paint(painter, option, widget=None):
            exposed.append((option.exposedRect.width(), option.exposedRect.height()))
            original_paint(painter, option, widget)

        item.paint = spy_paint
        _zoom_and_pan(window.map_view)

        if not exposed:
            return False, "MapCanvasItem.paint() was never called across the zoom+pan sequence"
        max_area = max(w * h for w, h in exposed)
        # Not just "< bounding_area": a near-full exposedRect at 6x zoom
        # would still technically be "smaller" while indicating the flag
        # isn't working. Require it well under 1/4 of the full canvas.
        if max_area >= bounding_area / 4:
            return False, (
                f"largest exposedRect ({max_area:.0f}px^2) is not substantially smaller than "
                f"boundingRect ({bounding_area:.0f}px^2) after {ZOOM_FACTOR}x zoom -- "
                f"ItemUsesExtendedStyleOption isn't doing its job"
            )
        return True, (
            f"OK ({len(exposed)} paint call(s), largest exposedRect {max_area:.0f}px^2 vs "
            f"full {bounding_area:.0f}px^2)"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def _check_panning_proportional(path: Path) -> tuple[bool, str]:
    conftest.ensure_qapp()
    window = _stepped_window(path)
    try:
        _show_and_settle(window)
        item = window.map_view._canvas_item
        if item is None:
            return False, "Stepped mode produced no MapCanvasItem"
        cache = item._cache  # descape.render_cache.IsoChunkCache
        canvas_w, canvas_h = cache.canvas_dims()
        chunk_px = cache.chunk_px
        total_grid_chunks = math.ceil(canvas_w / chunk_px) * math.ceil(canvas_h / chunk_px)
        if total_grid_chunks < 4:
            return None, f"{path.name}: canvas too small ({total_grid_chunks} chunks) for a meaningful pan probe"

        requested = set()
        original_get_chunk = cache.get_chunk

        def spy_get_chunk(mip, cx, cy):
            requested.add((mip, cx, cy))
            return original_get_chunk(mip, cx, cy)

        cache.get_chunk = spy_get_chunk
        _zoom_and_pan(window.map_view)

        if not requested:
            return False, "zoom+pan sequence never requested a single chunk"
        if len(requested) >= total_grid_chunks:
            return False, (
                f"zoom+pan sequence touched all {len(requested)}/{total_grid_chunks} chunks in the "
                f"grid -- looks like a full-canvas composite, not a lazy one"
            )
        return True, f"OK ({len(requested)}/{total_grid_chunks} chunks touched by zoom+{PAN_STEPS}-step pan)"
    finally:
        window.edit_history.mark_saved()
        window.close()


def _check_matches_full_render(path: Path) -> tuple[bool, str]:
    from PyQt5.QtCore import QRectF

    conftest.ensure_qapp()
    window = _stepped_window(path)
    try:
        item = window.map_view._canvas_item
        if item is None:
            return False, "Stepped mode produced no MapCanvasItem"
        bounding = item.boundingRect()
        x0, y0 = PROBE_ORIGIN
        probe = QRectF(x0, y0, PROBE_SIZE, PROBE_SIZE).intersected(bounding)
        if probe.isEmpty():
            return None, f"{path.name}: probe rect falls outside a canvas this small"

        got = _scene_rect_to_array(window.map_view.scene(), probe)
        full = render_terrain_iso(window.scenario)
        px0, py0 = int(probe.left()), int(probe.top())
        px1, py1 = px0 + got.shape[1], py0 + got.shape[0]
        expected = full[py0:py1, px0:px1]

        if got.shape != expected.shape:
            return False, f"crop shape mismatch: scene render {got.shape} vs full-render crop {expected.shape}"
        if not np.array_equal(got, expected):
            diff = int(np.count_nonzero(np.any(got != expected, axis=2)))
            return False, f"{diff}/{got.shape[0] * got.shape[1]} pixels differ from an independent full render"
        return True, "OK (scene render byte-identical to the full-render crop)"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_exposed_rect_smaller_than_bounding() -> None:
    conftest.run_check(_check_exposed_rect_smaller, FIXTURE_PATH)


@pytest.mark.gui
def test_panning_proportional_to_viewport() -> None:
    conftest.run_check(_check_panning_proportional, FIXTURE_PATH)


@pytest.mark.gui
def test_scene_render_matches_full_render() -> None:
    conftest.run_check(_check_matches_full_render, FIXTURE_PATH)


@pytest.mark.gui
@pytest.mark.corpus
def test_exposed_rect_smaller_than_bounding_corpus(scenario_path) -> None:
    conftest.run_check(_check_exposed_rect_smaller, scenario_path)


@pytest.mark.gui
@pytest.mark.corpus
def test_panning_proportional_to_viewport_corpus(scenario_path) -> None:
    conftest.run_check(_check_panning_proportional, scenario_path)


@pytest.mark.gui
@pytest.mark.corpus
def test_scene_render_matches_full_render_corpus(scenario_path) -> None:
    conftest.run_check(_check_matches_full_render, scenario_path)
