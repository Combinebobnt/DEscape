"""The drawn icon set: every toolbar and menu icon as data on a 24x24 grid, plus one painter.

Each icon is a tuple of primitives, scaled to whatever size is asked for, so the
set ships no image files and follows the theme. A `Role.INK` part takes the
palette's `ButtonText`, the colour tool buttons and menus draw their text in.

The accents are named hues rather than fixed colours. No single mid-tone clears
3:1 on both the Wood presets' window (relative luminance about 0.39) and the
dark presets' (0.06 or less), so `adapted_accent()` blends each one toward black
or white until it clears 3:1 on the backdrops it is drawn over, in the same 5%
steps `themes.legible()` takes.

Antialiasing is on here, unlike the map renderers: an icon is chrome, and a
24 px glyph with hard edges reads as broken.
"""

from __future__ import annotations

from enum import Enum, auto

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QApplication

from descape import themes

GRID = 24.0
# Alpha of `Role.INK_SOFT`, out of 255.
SOFT_ALPHA = 102
# The contrast an accent must reach on every backdrop it is drawn over.
ACCENT_CONTRAST = 3.0
# The ink's floor: Wood's own ButtonText is only 5.4:1, which left the outlines soft (user, 2026-10-17).
INK_CONTRAST = 8.0


class Role(Enum):
    INK = auto()
    INK_SOFT = auto()


TAN = QColor("#a97b4c")
BLUE = QColor("#4687c8")
GREEN = QColor("#3c954a")
RED = QColor("#dc594d")
VIOLET = QColor("#a669d2")
TEAL = QColor("#3c8f97")
AMBER = QColor("#9e8028")
GREY = QColor("#848484")
PINK = QColor("#c66394")

# Every literal colour the artwork may use; a test refuses any other.
ACCENTS = (TAN, BLUE, GREEN, RED, VIOLET, TEAL, AMBER, GREY, PINK)

# (kind, paint, points), paint a Role or one of ACCENTS:
#   "poly" (x1, y1, x2, y2, ...), "rect" (x, y, w, h), "ellipse" (x, y, w, h),
#   "ring" (x, y, w, h, width) unfilled, "line" (x1, y1, x2, y2, width) round-capped.
Shape = tuple[str, "Role | QColor", tuple[float, ...]]


def ink_colour() -> QColor:
    """The palette's ButtonText, deepened to INK_CONTRAST where it is weaker (the mid-tone Wood presets)."""
    app = QApplication.instance()
    if app is None:
        return QColor(Qt.black)
    palette = app.palette()
    return adapted_colour(QColor(palette.color(QPalette.ButtonText)), backdrops_of(palette), INK_CONTRAST)


def backdrops_of(palette: QPalette) -> tuple[QColor, ...]:
    """What an icon sits on: the toolbar's Window, Button behind a checked tool, and the menu panel."""
    # Fusion paints a QMenu panel in Base.lighter(108), measured on a shown menu (PyQt 5.15, offscreen).
    menu = QColor(palette.color(QPalette.Base)).lighter(108)
    return (QColor(palette.color(QPalette.Window)), QColor(palette.color(QPalette.Button)), menu)


def backdrop_colours() -> tuple[QColor, ...]:
    app = QApplication.instance()
    if app is None:
        return ()
    return backdrops_of(app.palette())


def adapted_colour(colour: QColor, backdrops: tuple[QColor, ...], floor: float) -> QColor:
    """`colour` unchanged when it clears `floor` on every backdrop, else the first 5% blend
    toward black or white (whichever contrasts more with them) that does, or that pole."""
    if not backdrops:
        return QColor(colour)
    hexes = [b.name() for b in backdrops]

    def worst(candidate: str) -> float:
        return min(themes.contrast_ratio(candidate, b) for b in hexes)

    if worst(colour.name()) >= floor:
        return QColor(colour)
    pole = max(("#000000", "#ffffff"), key=worst)
    for step in range(1, 20):
        shade = themes.blend(colour.name(), pole, step / 20)
        if worst(shade) >= floor:
            return QColor(shade)
    return QColor(pole)


def adapted_accent(accent: QColor, backdrops: tuple[QColor, ...]) -> QColor:
    return adapted_colour(accent, backdrops, ACCENT_CONTRAST)


def _resolve(paint: Role | QColor, ink: QColor, accents: dict[int, QColor]) -> QColor:
    if paint is Role.INK:
        return QColor(ink)
    if paint is Role.INK_SOFT:
        soft = QColor(ink)
        soft.setAlpha(SOFT_ALPHA)
        return soft
    return QColor(accents.get(paint.rgb(), paint))


def _draw(painter: QPainter, shapes: tuple[Shape, ...], ink: QColor, accents: dict[int, QColor]) -> None:
    for kind, paint, points in shapes:
        colour = _resolve(paint, ink, accents)
        if kind in ("line", "ring"):
            *geometry, width = points
            pen = QPen(colour, width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            if kind == "line":
                x1, y1, x2, y2 = geometry
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            else:
                painter.drawEllipse(QRectF(*geometry))
            continue
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(colour))
        if kind == "rect":
            painter.drawRect(QRectF(*points))
        elif kind == "ellipse":
            painter.drawEllipse(QRectF(*points))
        elif kind == "poly":
            painter.drawPolygon(QPolygonF([QPointF(points[i], points[i + 1]) for i in range(0, len(points), 2)]))
        else:
            raise ValueError(
                f"{kind!r} is not a known icon primitive; use one of 'poly', 'rect', 'ellipse', 'ring' or 'line'"
            )


def paint_shapes(
    shapes: tuple[Shape, ...], size: int, ink: QColor, dpr: float = 1.0, backdrops: tuple[QColor, ...] = ()
) -> QPixmap:
    """`shapes` from the 24x24 grid on a transparent `size` px pixmap, accents adapted to `backdrops`."""
    accents = {accent.rgb(): adapted_accent(accent, backdrops) for accent in ACCENTS}
    pixmap = QPixmap(round(size * dpr), round(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # Qt already scales a painter by its pixmap's devicePixelRatio; scaling by it again clips a 2x icon.
    painter.scale(size / GRID, size / GRID)
    try:
        _draw(painter, shapes, ink, accents)
    finally:
        painter.end()
    return pixmap


def paint_icon(name: str, size: int, ink: QColor, dpr: float = 1.0, backdrops: tuple[QColor, ...] = ()) -> QPixmap:
    try:
        shapes = ICON_SHAPES[name]
    except KeyError:
        raise KeyError(f"{name!r} has no artwork in icon_art.ICON_SHAPES") from None
    return paint_shapes(shapes, size, ink, dpr, backdrops)


# The rotate arcs: half-annuli with an ink head, the CCW one the CW one mirrored.
_ARC_CW = (
    3.14, 14.56, 3.31, 10.67, 5.11, 7.21, 8.2, 4.84, 12, 4, 15.8, 4.84, 18.89, 7.21, 20.69, 10.67,
    17.8, 11.45, 16.6, 9.14, 14.54, 7.56, 12, 7, 9.46, 7.56, 7.4, 9.14, 6.2, 11.45, 6.09, 14.04,
)
_HEAD_CW = (20.41, 15.41, 22.72, 10.13, 15.77, 11.99)
_ARC_CCW = (
    20.86, 14.56, 20.69, 10.67, 18.89, 7.21, 15.8, 4.84, 12, 4, 8.2, 4.84, 5.11, 7.21, 3.31, 10.67,
    6.2, 11.45, 7.4, 9.14, 9.46, 7.56, 12, 7, 14.54, 7.56, 16.6, 9.14, 17.8, 11.45, 17.91, 14.04,
)
_HEAD_CCW = (3.59, 15.41, 1.28, 10.13, 8.23, 11.99)
_UNDO_ARC = (
    3.5, 13.5, 4.64, 9.25, 7.75, 6.14, 12, 5, 16.25, 6.14, 19.36, 9.25, 20.5, 13.5, 17.5, 13.5,
    16.76, 10.75, 14.75, 8.74, 12, 8, 9.25, 8.74, 7.24, 10.75, 6.5, 13.5,
)
_PAGE = (5, 2, 14, 2, 19, 7, 19, 22, 5, 22)
_PAGE_FOLD = (14, 2, 19, 7, 14, 7)
# The dashed selection frame shared by Select All, Deselect and Scatter.
_FRAME_DASHES = (
    (3, 4, 6.5, 4), (9.5, 4, 14.5, 4), (17.5, 4, 21, 4),
    (3, 20, 6.5, 20), (9.5, 20, 14.5, 20), (17.5, 20, 21, 20),
    (3, 5.6, 3, 7), (3, 10, 3, 14), (3, 17, 3, 18.4),
    (21, 5.6, 21, 7), (21, 10, 21, 14), (21, 17, 21, 18.4),
)
_EYE = (2, 12, 7, 6.5, 12, 5, 17, 6.5, 22, 12, 17, 17.5, 12, 19, 7, 17.5)
_TREE = (("ellipse", GREEN, (4, 1, 12, 11)), ("rect", TAN, (8.5, 11, 3, 6)))


def _frame(paint: Role) -> tuple[Shape, ...]:
    return tuple(("line", paint, (*dash, 1.6)) for dash in _FRAME_DASHES)


def _figure(paint: Role | QColor, x: float, y: float, scale: float = 1.0) -> tuple[Shape, ...]:
    """A person: head over a trapezoid body, its top-left at (x, y), 10 x 18 at scale 1."""
    s = scale
    return (
        ("ellipse", paint, (x + 2 * s, y, 6 * s, 6 * s)),
        ("poly", paint, (x, y + 18 * s, x + 1.5 * s, y + 7.5 * s, x + 8.5 * s, y + 7.5 * s, x + 10 * s, y + 18 * s)),
    )


ICON_SHAPES: dict[str, tuple[Shape, ...]] = {
    # --- tools: settings.TOOLS order, shown icon-only on the toolbar ---
    # A hand, the wrist cuffed in ink.
    "tool-pan": (
        ("line", TAN, (8, 12, 8, 5.5, 3.2)),
        ("line", TAN, (11.5, 11, 11.5, 3.5, 3.2)),
        ("line", TAN, (15, 11, 15, 4.5, 3.2)),
        ("line", TAN, (18.5, 12.5, 18.5, 7, 3.2)),
        ("line", TAN, (7, 15.5, 3.8, 11.8, 3.2)),
        ("poly", TAN, (6.4, 11, 20.1, 11, 20.1, 15, 17, 20.5, 9, 20.5, 6.4, 16)),
        ("rect", Role.INK, (8.5, 20, 9, 2.8)),
    ),
    # A brush over a terrain tile.
    "tool-draw": (
        ("poly", GREEN, (12, 13, 21.5, 17.5, 12, 22, 2.5, 17.5)),
        ("line", TAN, (20.5, 2, 14, 8.5, 3.0)),
        ("line", GREY, (14, 8.5, 12, 10.5, 4.2)),
        ("poly", Role.INK, (13.2, 11.3, 11.2, 9.3, 8, 15)),
    ),
    # A paint bucket, painted back to front so the paint runs out of it.
    "tool-fill": (
        ("ellipse", GREEN, (3, 18.3, 18, 4.4)),
        ("poly", GREEN, (15, 5.5, 17.5, 5.5, 18.6, 12.5, 16.1, 12.5)),
        ("ellipse", GREEN, (15.6, 11, 3.6, 4.6)),
        ("line", Role.INK, (4.5, 5.5, 10, 1.2, 1.5)),
        ("line", Role.INK, (10, 1.2, 15.5, 5.5, 1.5)),
        ("ellipse", GREY, (3.6, 2.8, 12.8, 3.2)),
        ("poly", BLUE, (4, 4.4, 16, 4.4, 14, 17.5, 6, 17.5)),
        ("ellipse", GREEN, (5, 3.4, 10, 2.0)),
    ),
    # A hill with an arrow up.
    "tool-elevation": (
        ("poly", GREEN, (2, 21, 8, 13, 12, 16, 16.5, 9.5, 22, 21)),
        ("line", Role.INK, (6, 10.5, 6, 4, 2.2)),
        ("poly", Role.INK, (6, 1.5, 9.8, 6, 2.2, 6)),
    ),
    # A flat-topped hill cut by a level line.
    "tool-set_level": (
        ("poly", GREEN, (2, 21, 7.5, 12, 16.5, 12, 22, 21)),
        ("line", Role.INK, (4.5, 12, 19.5, 12, 1.8)),
        ("poly", AMBER, (1, 8.5, 5, 12, 1, 15.5)),
        ("poly", AMBER, (23, 8.5, 19, 12, 23, 15.5)),
    ),
    # A polyline: two ends and a bend.
    "tool-draw_line": (
        ("line", Role.INK, (4.5, 19, 11, 6, 1.8)),
        ("line", Role.INK, (11, 6, 19.5, 12.5, 1.8)),
        ("ellipse", BLUE, (2, 16.5, 5, 5)),
        ("ellipse", BLUE, (17, 10, 5, 5)),
        ("ellipse", RED, (8.5, 3.5, 5, 5)),
    ),
    # A rectangle of terrain with its drag handle.
    "tool-draw_rect": (
        ("rect", GREEN, (4.5, 6.5, 15, 11)),
        ("rect", Role.INK, (3, 5, 18, 1.6)),
        ("rect", Role.INK, (3, 17.4, 18, 1.6)),
        ("rect", Role.INK, (3, 6.6, 1.6, 10.8)),
        ("rect", Role.INK, (19.4, 6.6, 1.6, 10.8)),
        ("ellipse", BLUE, (17, 15.5, 6, 6)),
    ),
    # A grassy ridge over a jagged rock face.
    "tool-cliff": (
        ("poly", TAN, (2, 8, 12, 12, 22, 8, 22, 15, 18.5, 20, 15, 16.5, 11, 21, 6.5, 16.5, 2, 19)),
        ("poly", GREEN, (2, 8, 12, 3.5, 22, 8, 12, 12)),
        ("line", Role.INK, (2, 8, 12, 12, 1.5)),
        ("line", Role.INK, (12, 12, 22, 8, 1.5)),
    ),
    # A dashed marquee.
    "tool-select": (
        ("rect", Role.INK_SOFT, (4, 5, 16, 14)),
        ("line", Role.INK, (4, 5, 8, 5, 1.6)),
        ("line", Role.INK, (11, 5, 15, 5, 1.6)),
        ("line", Role.INK, (18, 5, 20, 5, 1.6)),
        ("line", Role.INK, (4, 19, 6, 19, 1.6)),
        ("line", Role.INK, (9, 19, 13, 19, 1.6)),
        ("line", Role.INK, (16, 19, 20, 19, 1.6)),
        ("line", Role.INK, (4, 5, 4, 8, 1.6)),
        ("line", Role.INK, (4, 11, 4, 15, 1.6)),
        ("line", Role.INK, (20, 5, 20, 9, 1.6)),
        ("line", Role.INK, (20, 12, 20, 16, 1.6)),
    ),
    # A four-way arrow.
    "tool-move": (
        (
            "poly",
            Role.INK,
            (
                12, 2, 16, 6, 13.5, 6, 13.5, 10.5, 18, 10.5, 18, 8, 22, 12, 18, 16, 18, 13.5, 13.5, 13.5,
                13.5, 18, 16, 18, 12, 22, 8, 18, 10.5, 18, 10.5, 13.5, 6, 13.5, 6, 16, 2, 12, 6, 8, 6, 10.5,
                10.5, 10.5, 10.5, 6, 8, 6,
            ),
        ),
        ("rect", BLUE, (10.5, 10.5, 3, 3)),
    ),
    "tool-eyedropper": (
        ("poly", GREY, (19.27, 7.27, 10.27, 16.27, 7.73, 13.73, 16.73, 4.73)),
        ("ellipse", BLUE, (16.5, 1.5, 6.5, 6.5)),
        ("poly", Role.INK, (10.27, 16.27, 7.73, 13.73, 6, 18)),
        ("ellipse", RED, (2, 18, 4.5, 4.5)),
    ),
    # A diagonal ruler, ticks along its upper edge.
    "tool-ruler": (
        ("poly", AMBER, (2, 16, 16, 2, 22, 8, 8, 22)),
        ("line", Role.INK, (4.8, 13.2, 7.3, 15.7, 1.3)),
        ("line", Role.INK, (7.6, 10.4, 9.2, 12, 1.3)),
        ("line", Role.INK, (10.4, 7.6, 12.9, 10.1, 1.3)),
        ("line", Role.INK, (13.2, 4.8, 14.8, 6.4, 1.3)),
    ),
    # A figure and a plus.
    "tool-place_unit": (
        *_figure(BLUE, 2, 3),
        ("line", Role.INK, (18, 9.5, 18, 18.5, 2.4)),
        ("line", Role.INK, (13.5, 14, 22.5, 14, 2.4)),
    ),
    # A ring of stone walls, mortar in ink.
    "tool-wall_rect": (
        ("rect", GREY, (3, 4, 18, 4)),
        ("rect", GREY, (3, 16, 18, 4)),
        ("rect", GREY, (3, 8, 4, 8)),
        ("rect", GREY, (17, 8, 4, 8)),
        ("line", Role.INK, (9, 4, 9, 8, 1.1)),
        ("line", Role.INK, (15, 4, 15, 8, 1.1)),
        ("line", Role.INK, (6, 16, 6, 20, 1.1)),
        ("line", Role.INK, (12, 16, 12, 20, 1.1)),
        ("line", Role.INK, (18, 16, 18, 20, 1.1)),
        ("line", Role.INK, (3, 12, 7, 12, 1.1)),
        ("line", Role.INK, (17, 12, 21, 12, 1.1)),
    ),
    # A blue figure, an arrow, a red figure: the owner changes.
    "tool-convert": (
        *_figure(BLUE, 1.5, 1, 0.7),
        *_figure(RED, 15.5, 1, 0.7),
        ("line", Role.INK, (3, 19.5, 17, 19.5, 2.0)),
        ("poly", Role.INK, (22, 19.5, 17, 16, 17, 23)),
    ),
    # A figure struck by a trigger's spark.
    "tool-create_objects": (
        *_figure(Role.INK, 2, 3),
        ("poly", AMBER, (17.5, 1.5, 13, 12, 17, 12, 14, 22.5, 22.5, 8.5, 18.5, 8.5, 21.5, 1.5)),
    ),
    "tool-eraser": (
        ("poly", PINK, (19.47, 8.47, 10.47, 17.47, 5.53, 12.53, 14.53, 3.53)),
        ("poly", GREY, (14.97, 12.97, 10.47, 17.47, 5.53, 12.53, 10.03, 8.03)),
        ("line", Role.INK_SOFT, (4, 21, 14, 21, 1.6)),
    ),
    # --- toolbar buttons that are not tools ---
    "filters": (
        ("poly", Role.INK, (2, 3, 22, 3, 14, 12, 14, 20, 10, 22, 10, 12)),
        ("poly", TEAL, (2, 3, 22, 3, 19.3, 6, 4.7, 6)),
    ),
    # A toolbox.
    "more-tools": (
        ("line", Role.INK, (8, 10, 8, 5.5, 1.8)),
        ("line", Role.INK, (8, 5.5, 16, 5.5, 1.8)),
        ("line", Role.INK, (16, 5.5, 16, 10, 1.8)),
        ("rect", TAN, (2, 9.5, 20, 11.5)),
        ("rect", GREY, (10, 13, 4, 3.5)),
    ),
    # --- File ---
    "file-new": (
        ("poly", Role.INK_SOFT, _PAGE),
        ("poly", Role.INK, _PAGE_FOLD),
        ("line", GREEN, (12, 11, 12, 19, 2.2)),
        ("line", GREEN, (8, 15, 16, 15, 2.2)),
    ),
    "file-open": (
        ("poly", AMBER, (2, 5, 9, 5, 11, 8, 21, 8, 21, 20, 2, 20)),
        ("poly", TAN, (2, 20, 5.5, 11, 23, 11, 19.5, 20)),
        ("line", Role.INK_SOFT, (2, 20.8, 20, 20.8, 1.6)),
    ),
    "file-close": (
        ("poly", Role.INK_SOFT, _PAGE),
        ("poly", Role.INK, _PAGE_FOLD),
        ("line", RED, (8.5, 11.5, 15.5, 18.5, 2.2)),
        ("line", RED, (15.5, 11.5, 8.5, 18.5, 2.2)),
    ),
    "file-save": (
        ("rect", BLUE, (3, 3, 18, 18)),
        ("rect", GREY, (8, 3, 8, 6)),
        ("rect", Role.INK_SOFT, (6, 13, 12, 8)),
    ),
    "file-save-as": (
        ("rect", BLUE, (2, 3, 15, 15)),
        ("rect", GREY, (6, 3, 7, 5)),
        ("rect", Role.INK_SOFT, (4, 11, 11, 7)),
        ("poly", AMBER, (16, 16, 21, 11, 23, 13, 18, 18)),
        ("poly", Role.INK, (16, 16, 18, 18, 14.8, 19.2)),
    ),
    # A page with a clock on it: the automatic copies.
    "file-recover": (
        ("poly", Role.INK_SOFT, _PAGE),
        ("poly", Role.INK, _PAGE_FOLD),
        ("ellipse", BLUE, (7, 10, 10, 10)),
        ("line", Role.INK, (12, 15, 12, 11.8, 1.5)),
        ("line", Role.INK, (12, 15, 14.6, 16.4, 1.5)),
    ),
    # A list of lines struck out.
    "file-clear-recent": (
        ("line", Role.INK_SOFT, (3, 5, 15, 5, 2.0)),
        ("line", Role.INK_SOFT, (3, 10, 15, 10, 2.0)),
        ("line", Role.INK_SOFT, (3, 15, 11, 15, 2.0)),
        ("line", Role.INK, (3, 20, 9, 20, 2.0)),
        ("line", RED, (13.5, 13.5, 21, 21, 2.4)),
        ("line", RED, (21, 13.5, 13.5, 21, 2.4)),
    ),
    # An open door and an arrow out of it.
    "file-exit": (
        ("rect", Role.INK_SOFT, (3, 2, 11, 20)),
        ("poly", TAN, (3, 2, 10.5, 4.5, 10.5, 22.5, 3, 22)),
        ("ellipse", Role.INK, (7.5, 12, 1.8, 1.8)),
        ("line", RED, (12.5, 12, 19, 12, 2.2)),
        ("poly", RED, (23, 12, 18.5, 7.8, 18.5, 16.2)),
    ),
    # --- Edit ---
    "edit-undo": (("poly", Role.INK, _UNDO_ARC), ("poly", BLUE, (5, 19.5, 1.5, 13.5, 8.5, 13.5))),
    "edit-redo": (("poly", Role.INK, _UNDO_ARC), ("poly", GREEN, (19, 19.5, 15.5, 13.5, 22.5, 13.5))),
    "edit-cut": (
        ("line", Role.INK, (8.5, 15, 16, 2.5, 2.0)),
        ("line", Role.INK, (15.5, 15, 8, 2.5, 2.0)),
        ("ring", RED, (3.5, 15, 6, 6, 2.0)),
        ("ring", RED, (14.5, 15, 6, 6, 2.0)),
    ),
    "edit-copy": (
        ("rect", Role.INK_SOFT, (3, 2, 12, 15)),
        ("poly", BLUE, (9, 7, 17, 7, 21, 11, 21, 22, 9, 22)),
        ("poly", Role.INK, (17, 7, 21, 11, 17, 11)),
    ),
    "edit-paste": (
        ("rect", TAN, (4, 4, 16, 18.5)),
        ("rect", GREY, (8, 2, 8, 4.5)),
        ("rect", Role.INK_SOFT, (7, 9, 10, 10.5)),
    ),
    # A clipboard with earlier sheets stacked behind it.
    "edit-clipboard-history": (
        ("rect", Role.INK_SOFT, (9, 1.5, 13, 16)),
        ("rect", Role.INK, (6, 3.5, 13, 16)),
        ("rect", TAN, (2.5, 6, 13, 16.5)),
        ("rect", GREY, (5.5, 4.5, 7, 4)),
        ("rect", Role.INK_SOFT, (5, 11, 8, 8.5)),
    ),
    # A clock with a backward arrow.
    "edit-history": (
        ("ring", Role.INK, (3.5, 3.5, 17, 17, 2.0)),
        ("line", Role.INK, (12, 12, 12, 7, 2.0)),
        ("line", Role.INK, (12, 12, 15.5, 14.5, 2.0)),
        ("poly", BLUE, (0.8, 5.5, 7.5, 5.5, 2.8, 11)),
        ("ellipse", BLUE, (10.5, 10.5, 3, 3)),
    ),
    "select-all": (
        ("line", Role.INK, (7, 3, 10, 3, 1.6)),
        ("line", Role.INK, (14, 3, 17, 3, 1.6)),
        ("line", Role.INK, (7, 21, 10, 21, 1.6)),
        ("line", Role.INK, (14, 21, 17, 21, 1.6)),
        ("line", Role.INK, (3, 7, 3, 10, 1.6)),
        ("line", Role.INK, (3, 14, 3, 17, 1.6)),
        ("line", Role.INK, (21, 7, 21, 10, 1.6)),
        ("line", Role.INK, (21, 14, 21, 17, 1.6)),
        ("rect", BLUE, (1, 1, 4, 4)),
        ("rect", BLUE, (19, 1, 4, 4)),
        ("rect", BLUE, (1, 19, 4, 4)),
        ("rect", BLUE, (19, 19, 4, 4)),
    ),
    "select-none": (*_frame(Role.INK_SOFT), ("line", RED, (4, 20.5, 20, 3.5, 2.4))),
    # Coloured units strewn across a selection.
    "edit-scatter": (
        *_frame(Role.INK),
        ("ellipse", BLUE, (6, 7, 4, 4)),
        ("ellipse", GREEN, (13.5, 6, 4, 4)),
        ("ellipse", RED, (9.5, 12.5, 4, 4)),
        ("ellipse", AMBER, (15, 13, 4, 4)),
    ),
    # A magnifier with a blue lens.
    "edit-find": (
        ("ellipse", BLUE, (5, 5, 9, 9)),
        ("ring", Role.INK, (3.5, 3.5, 12, 12, 2.2)),
        ("line", Role.INK, (14.5, 14.5, 21, 21, 2.8)),
    ),
    # Fine turns carry a square, quarter turns a square already turned.
    "rotate-ccw": (("poly", Role.INK, _ARC_CCW), ("poly", Role.INK, _HEAD_CCW), ("rect", BLUE, (9, 10, 6, 6))),
    "rotate-cw": (("poly", Role.INK, _ARC_CW), ("poly", Role.INK, _HEAD_CW), ("rect", BLUE, (9, 10, 6, 6))),
    "rotate-ccw-quarter": (
        ("poly", Role.INK, _ARC_CCW),
        ("poly", Role.INK, _HEAD_CCW),
        ("poly", AMBER, (12, 9, 16, 13, 12, 17, 8, 13)),
    ),
    "rotate-cw-quarter": (
        ("poly", Role.INK, _ARC_CW),
        ("poly", Role.INK, _HEAD_CW),
        ("poly", AMBER, (12, 9, 16, 13, 12, 17, 8, 13)),
    ),
    # A tree and which way through its variants.
    "variant-prev": (
        *_TREE,
        ("line", Role.INK, (7, 20.5, 19, 20.5, 2.0)),
        ("poly", Role.INK, (2.5, 20.5, 7, 17, 7, 24)),
    ),
    "variant-next": (
        *_TREE,
        ("line", Role.INK, (2, 20.5, 14, 20.5, 2.0)),
        ("poly", Role.INK, (18.5, 20.5, 14, 17, 14, 24)),
    ),
    "variant-random": (
        *_TREE,
        ("rect", AMBER, (13, 12, 10, 10)),
        ("ellipse", Role.INK, (14.5, 13.5, 2.4, 2.4)),
        ("ellipse", Role.INK, (16.8, 15.8, 2.4, 2.4)),
        ("ellipse", Role.INK, (19.1, 18.1, 2.4, 2.4)),
    ),
    "flip-horizontal": (
        ("line", Role.INK_SOFT, (12, 2, 12, 4.5, 1.6)),
        ("line", Role.INK_SOFT, (12, 8, 12, 10, 1.6)),
        ("line", Role.INK_SOFT, (12, 14, 12, 16, 1.6)),
        ("line", Role.INK_SOFT, (12, 19.5, 12, 22, 1.6)),
        ("poly", Role.INK, (2, 5, 10, 12, 2, 19)),
        ("poly", BLUE, (22, 5, 14, 12, 22, 19)),
    ),
    "flip-vertical": (
        ("line", Role.INK_SOFT, (2, 12, 4.5, 12, 1.6)),
        ("line", Role.INK_SOFT, (8, 12, 10, 12, 1.6)),
        ("line", Role.INK_SOFT, (14, 12, 16, 12, 1.6)),
        ("line", Role.INK_SOFT, (19.5, 12, 22, 12, 1.6)),
        ("poly", Role.INK, (5, 2, 19, 2, 12, 10)),
        ("poly", BLUE, (5, 22, 12, 14, 19, 22)),
    ),
    # Three tiles stacked one on another.
    "select-stack": (
        ("poly", BLUE, (12, 13.5, 21, 17.5, 12, 21.5, 3, 17.5)),
        ("poly", TEAL, (12, 8.5, 21, 12.5, 12, 16.5, 3, 12.5)),
        ("poly", Role.INK, (12, 3.5, 21, 7.5, 12, 11.5, 3, 7.5)),
    ),
    # Three sliders.
    "settings": (
        ("line", Role.INK, (3, 6, 21, 6, 1.6)),
        ("line", Role.INK, (3, 12, 21, 12, 1.6)),
        ("line", Role.INK, (3, 18, 21, 18, 1.6)),
        ("ellipse", BLUE, (13, 3, 6, 6)),
        ("ellipse", AMBER, (4.5, 9, 6, 6)),
        ("ellipse", GREEN, (10, 15, 6, 6)),
    ),
    # --- Map ---
    "map-resize": (
        ("rect", Role.INK_SOFT, (2, 2, 20, 1.6)),
        ("rect", Role.INK_SOFT, (2, 20.4, 20, 1.6)),
        ("rect", Role.INK_SOFT, (2, 3.6, 1.6, 16.8)),
        ("rect", Role.INK_SOFT, (20.4, 3.6, 1.6, 16.8)),
        ("rect", BLUE, (5, 11, 8, 8)),
        ("line", Role.INK, (11.5, 12.5, 16.5, 7.5, 2.0)),
        ("poly", Role.INK, (19.5, 4.5, 18.3, 9.66, 14.34, 5.7)),
    ),
    # A stroke and its reflection across the axis.
    "map-mirror": (
        ("line", Role.INK_SOFT, (12, 2, 12, 4.5, 1.6)),
        ("line", Role.INK_SOFT, (12, 8, 12, 10, 1.6)),
        ("line", Role.INK_SOFT, (12, 14, 12, 16, 1.6)),
        ("line", Role.INK_SOFT, (12, 19.5, 12, 22, 1.6)),
        ("line", Role.INK, (9.5, 4.5, 6, 3.5, 2.2)),
        ("line", Role.INK, (6, 3.5, 3.8, 6.5, 2.2)),
        ("line", Role.INK, (3.8, 6.5, 5, 10.5, 2.2)),
        ("line", Role.INK, (5, 10.5, 8, 13.5, 2.2)),
        ("line", Role.INK, (8, 13.5, 9, 17.5, 2.2)),
        ("line", Role.INK, (9, 17.5, 6.5, 20.5, 2.2)),
        ("line", Role.INK, (6.5, 20.5, 3.5, 19.5, 2.2)),
        ("line", TEAL, (14.5, 4.5, 18, 3.5, 2.2)),
        ("line", TEAL, (18, 3.5, 20.2, 6.5, 2.2)),
        ("line", TEAL, (20.2, 6.5, 19, 10.5, 2.2)),
        ("line", TEAL, (19, 10.5, 16, 13.5, 2.2)),
        ("line", TEAL, (16, 13.5, 15, 17.5, 2.2)),
        ("line", TEAL, (15, 17.5, 17.5, 20.5, 2.2)),
        ("line", TEAL, (17.5, 20.5, 20.5, 19.5, 2.2)),
    ),
    # A building under a no-entry sign.
    "map-disables": (
        ("poly", Role.INK, (7, 12, 12, 7, 17, 12, 17, 17, 7, 17)),
        ("ring", RED, (3, 3, 18, 18, 2.4)),
        ("line", RED, (6.2, 17.8, 17.8, 6.2, 2.4)),
    ),
    # A map tile under a magnifier.
    "map-analysis": (
        ("poly", GREEN, (10, 2, 19, 7, 10, 12, 1, 7)),
        ("ring", Role.INK, (8.5, 8.5, 10, 10, 2.0)),
        ("line", Role.INK, (17, 17, 22, 22, 2.6)),
    ),
    # --- Help ---
    "help-about": (
        ("ellipse", BLUE, (2, 2, 20, 20)),
        ("ellipse", Role.INK, (10.6, 5.5, 2.8, 2.8)),
        ("line", Role.INK, (12, 10.5, 12, 17.5, 2.8)),
    ),
    # A log page with a bug on it.
    "help-debug-log": (
        ("rect", Role.INK_SOFT, (3, 2, 15, 20)),
        ("line", Role.INK, (6, 6.5, 15, 6.5, 1.4)),
        ("line", Role.INK, (6, 10.5, 12, 10.5, 1.4)),
        ("line", Role.INK, (6, 14.5, 10, 14.5, 1.4)),
        ("ellipse", RED, (13, 12.5, 9, 10)),
        ("ellipse", Role.INK, (15.8, 10.2, 3.4, 3.4)),
        ("line", Role.INK, (17.5, 13, 17.5, 22, 1.2)),
    ),
    # --- the Filters popup's bulk commands ---
    "filter-all-players": (
        *_figure(BLUE, 1, 4, 0.75),
        *_figure(RED, 8.5, 4, 0.75),
        *_figure(GREEN, 16, 4, 0.75),
        ("line", Role.INK, (2, 21.5, 22, 21.5, 1.6)),
    ),
    "filter-no-players": (
        *_figure(Role.INK_SOFT, 1, 4, 0.75),
        *_figure(Role.INK_SOFT, 8.5, 4, 0.75),
        *_figure(Role.INK_SOFT, 16, 4, 0.75),
        ("line", RED, (3, 21, 21, 3, 2.4)),
    ),
    "filter-show-all": (
        ("poly", Role.INK_SOFT, _EYE),
        ("ellipse", BLUE, (8, 8, 8, 8)),
        ("ellipse", Role.INK, (10.5, 10.5, 3, 3)),
    ),
    "filter-hide-all": (
        ("poly", Role.INK_SOFT, _EYE),
        ("ellipse", GREY, (8, 8, 8, 8)),
        ("line", RED, (4, 20, 20, 4, 2.4)),
    ),
}
