"""UnitGhostItem's refused look (TASK-327), painted straight into a QImage.

The item paints a refused sprite as a tinted copy and a refused mark as a
fill in the refusal colour under a device-space cross-hatch. These tests
call paint() with a plain QPainter, so the scene's GHOST_OPACITY does not
apply and a pixel reads back as the item drew it.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _refusal_color():
    from PyQt5.QtGui import QColor

    from descape import settings

    return QColor(settings.get_default_overlay_color("unit_ghost_refused"))


def _item():
    conftest.ensure_qapp()
    from descape.viewer_canvas import UnitGhostItem

    item = UnitGhostItem()
    item.set_refusal_color(_refusal_color())
    return item


def _draw(rgba: np.ndarray):
    """A sprite draw's shape as set_sprites() reads it."""
    return SimpleNamespace(rgba=rgba, hotspot_x=0, hotspot_y=0)


def _render(item, width: int, height: int, transform=None):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPainter, QTransform

    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setTransform(transform if transform is not None else QTransform())
    item.paint(painter, None)
    painter.end()
    return image


def _pixels(image) -> list[tuple[int, int, int, int]]:
    return [image.pixelColor(x, y).getRgb() for y in range(image.height()) for x in range(image.width())]


def _blend(channel: int, target: int, tint: float) -> int:
    return round(channel * (1.0 - tint) + target * tint)


# --- sprites -----------------------------------------------------------------------


def test_a_refused_sprite_pixel_blends_toward_the_refusal_colour_and_its_source_is_untouched() -> None:
    from descape.viewer_canvas import UnitGhostItem

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[1, 1] = (255, 255, 255, 255)
    rgba[2, 2] = (255, 255, 255, 128)
    before = rgba.tobytes()
    item = _item()

    item.set_sprites([(_draw(rgba), 0.0, 0.0)])
    assert _render(item, 4, 4).pixelColor(1, 1).getRgb() == (255, 255, 255, 255)
    assert not item.refused

    item.set_sprites([(_draw(rgba), 0.0, 0.0)], refused=True)
    assert item.refused
    color, tint = _refusal_color(), UnitGhostItem.REFUSED_TINT
    expected = [_blend(255, c, tint) for c in (color.red(), color.green(), color.blue())]
    got = _render(item, 4, 4).pixelColor(1, 1)
    assert all(abs(g - e) <= 1 for g, e in zip((got.red(), got.green(), got.blue()), expected, strict=True)), (
        got.getRgb(), expected,
    )
    assert got.alpha() == 255
    tinted = item._tint_cache[id(rgba)][1]
    assert tinted[2, 2, 3] == 128 and tinted[0, 0, 3] == 0, "alpha is the source's"

    # The item draws the sprite cache's own array: the tint is a copy, never written into it.
    assert item._arrays[0] is rgba
    assert rgba.tobytes() == before
    # _images keeps the untinted entry; the tint lives beside it.
    assert len(item._images) == 1


def test_one_source_is_blended_once_across_content_updates_and_again_on_a_colour_change() -> None:
    from PyQt5.QtGui import QColor

    rgba = np.full((3, 3, 4), 255, dtype=np.uint8)
    item = _item()
    item.set_sprites([(_draw(rgba), 0.0, 0.0)], refused=True)
    first = item._tinted[0]
    # A group drag re-sends the same members every mouse move.
    item.set_content([], [], refused_draws=[(_draw(rgba), 5.0, 5.0)])
    assert item._tinted[0] is first

    item.set_refusal_color(QColor(0, 0, 255))
    assert item._tinted[0] is not first
    assert _render(item, 8, 8).pixelColor(5, 5).blue() > _render(item, 8, 8).pixelColor(5, 5).red()


# --- marks -------------------------------------------------------------------------

_SQUARE = [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]]


def _owner_is_refusal_color() -> tuple[int, int, int]:
    color = _refusal_color()
    return color.red(), color.green(), color.blue()


def _hatch_ink() -> tuple[int, int, int, int]:
    from descape.viewer_canvas import _contrast_ink

    return _contrast_ink(_refusal_color()).getRgb()


def test_a_refused_mark_in_the_refusal_colour_still_differs_and_shows_hatch_ink() -> None:
    """A red player's mark tinted red would otherwise look unchanged."""
    from PyQt5.QtGui import QTransform

    owner = _owner_is_refusal_color()
    plain, refused = _item(), _item()
    plain.set_mark(_SQUARE, owner)
    refused.set_mark(_SQUARE, owner, refused=True)
    assert refused.refused and not plain.refused
    # Mark content stays the owner's: tests read _marks' brushes as the owner colour.
    assert {brush.color().getRgb()[:3] for _p, brush, _pen in refused._marks} == {owner}

    for scale, size in ((4.0, 44), (0.3, 6)):
        transform = QTransform.fromScale(scale, scale)
        plain_pixels = _pixels(_render(plain, size, size, transform))
        refused_pixels = _pixels(_render(refused, size, size, transform))
        assert plain_pixels != refused_pixels, scale
        assert _hatch_ink() in refused_pixels, scale
        assert _hatch_ink() not in plain_pixels, scale


def test_a_three_pixel_refused_mark_holds_hatch_ink_at_every_phase() -> None:
    """The hatch is anchored on the mark's device centre; an unanchored
    pattern misses a 3 px mark at some of the 64 phases."""
    from PyQt5.QtGui import QTransform

    item = _item()
    item.set_mark(_SQUARE, _owner_is_refusal_color(), refused=True)
    misses = []
    for ox in range(8):
        for oy in range(8):
            transform = QTransform.fromTranslate(ox + 2, oy + 2)
            transform.scale(0.3, 0.3)
            if _hatch_ink() not in _pixels(_render(item, 16, 16, transform)):
                misses.append((ox, oy))
    assert misses == []


def test_the_hatch_keeps_its_device_pixel_period_at_every_zoom() -> None:
    from PyQt5.QtGui import QTransform

    item = _item()
    item.set_mark(_SQUARE, _owner_is_refusal_color(), refused=True)
    ink = _hatch_ink()
    periods = set()
    for scale in (4.0, 12.0):
        image = _render(item, int(10 * scale), int(10 * scale), QTransform.fromScale(scale, scale))
        row = [x for x in range(image.width()) if image.pixelColor(x, 20).getRgb() == ink]
        assert len(row) >= 4, (scale, row)
        periods.add(tuple(b - a for a, b in pairwise(row[::2])))
    (gaps,) = {gaps[:3] for gaps in periods}
    assert set(gaps) == {8}
