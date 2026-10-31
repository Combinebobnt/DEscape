"""descape/swatch_grid.py: SwatchGridView, the terrain picker's grid (GH #161).
Offscreen Qt; geometry claims are offscreen PyQt5 5.15 with the pinned test font."""

from __future__ import annotations

import itertools

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

WIDTHS = (300, 440, 613)


def _items():
    from descape.value_picker import PickerItem

    return [
        PickerItem("Zeta", 10, "Water"),
        PickerItem("Alpha", 11, "Water"),
        PickerItem("Water Yellow Shallow", 12, "Water"),
        PickerItem("Mediterranean", 13, "Water"),
        PickerItem("Grass", 20, "Grass"),
        PickerItem("Grass Dry", 21, "Grass"),
        PickerItem("Grass Other", 22, "Grass"),
        PickerItem("Rock", 30, "Rock"),
        PickerItem("Beach Non Navigable Wet Gravel", 31, "Rock"),
        PickerItem("Snow", 40, "Snow"),
        PickerItem("Snow Light", 41, "Snow"),
        PickerItem("Ice", 42, "Snow"),
        PickerItem("Ice Beach", 43, "Snow"),
        PickerItem("Dirt", 50, "Dirt"),
    ]


def _terrain_items():
    from descape.terrain_browser import items_for

    return [item for item in items_for() if not item.hidden]


class _Hook:
    """A swatch hook that records its calls and can advance a fake clock."""

    def __init__(self, clock=None, cost: float = 0.0):
        self.calls: list[int] = []
        self._clock = clock
        self._cost = cost

    def __call__(self, value: int):
        from PyQt5.QtGui import QColor, QPixmap

        self.calls.append(value)
        if self._clock is not None:
            self._clock.now += self._cost
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(40, 120, 200))
        return pixmap


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _pump(n: int = 4) -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def _grid(items=None, *, width: int = 440, height: int = 400, show: bool = True, hook=None, **kwargs):
    from descape.swatch_grid import SwatchGridView

    conftest.ensure_qapp()
    hook = hook or _Hook()
    grid = SwatchGridView(_items() if items is None else items, swatch=hook, **kwargs)
    grid.resize(width, height)
    if show:
        grid.show()
        _pump()
    return grid, hook


def _rect(grid, value):
    return grid.visualItemRect(grid.cell_item(value))


def _center(grid, value):
    return _rect(grid, value).center()


def _requeue(grid, hook) -> None:
    """Drops every swatch with the timer stopped, so a test drives the ticks itself."""
    grid.refresh_swatches()
    grid._timer.stop()
    hook.calls.clear()


def _shown_cells(grid):
    return [v for v in grid.values() if not grid.cell_item(v).isHidden()]


def _per_row(grid) -> int:
    tops = [_rect(grid, v).top() for v in _shown_cells(grid)]
    return max(tops.count(t) for t in set(tops))


# -- population and layout ------------------------------------------------------


def test_one_heading_per_group_then_its_cells_in_the_trees_order() -> None:
    from descape.swatch_grid import _HEADING_ROLE
    from descape.value_picker import ValuePickerView

    tree_view = ValuePickerView(_items())
    tree = tree_view.tree
    expected = []
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        expected.append(("h", group.text(0)))
        expected.extend(("c", group.child(c).data(0, 0x0100)) for c in range(group.childCount()))
    grid, _hook = _grid(show=False)
    try:
        got = []
        for row in range(grid.count()):
            item = grid.item(row)
            got.append(("h", item.text()) if item.data(_HEADING_ROLE) else ("c", item.data(0x0100)))
        assert got == expected
        assert grid.cell_item(11).toolTip() == "Alpha (11)"
    finally:
        grid.close()
        tree_view.deleteLater()


@pytest.mark.parametrize("width", WIDTHS)
def test_headings_span_the_viewport_and_never_overlap_a_cell(width: int) -> None:
    grid, _hook = _grid(_terrain_items(), width=width, height=500)
    try:
        viewport = grid.viewport().width()
        cells = [_rect(grid, v) for v in grid.values()]
        for heading in grid.heading_items():
            rect = grid.visualItemRect(heading)
            assert rect.width() >= viewport - 2 * grid.spacing(), (heading.text(), rect, viewport)
            assert not any(rect.intersects(cell) for cell in cells), heading.text()
        for a, b in itertools.combinations(cells, 2):
            assert not a.intersects(b)
        assert grid.horizontalScrollBar().maximum() == 0
    finally:
        grid.close()


def test_columns_follow_the_width() -> None:
    counts = []
    for width in WIDTHS:
        grid, _hook = _grid(_terrain_items(), width=width, height=500)
        try:
            counts.append(_per_row(grid))
        finally:
            grid.close()
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts


@pytest.mark.font_sensitive
@pytest.mark.parametrize("width", WIDTHS)
def test_every_caption_fits_its_cell(width: int) -> None:
    """The wrapped caption's own text rect, in the caption font, inside the cell; no column count assumed."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QFontMetrics

    from descape.swatch_grid import _TEXT_FLAGS

    grid, _hook = _grid(_terrain_items(), width=width, height=500)
    try:
        metrics = QFontMetrics(grid.caption_font())
        cells = [_rect(grid, v) for v in grid.values()]
        for value in grid.values():
            cell = _rect(grid, value)
            text = grid.cell_item(value).text()
            box = grid.caption_rect(cell, text)
            used = metrics.boundingRect(QRect(box.left(), box.top(), box.width(), 100_000), _TEXT_FLAGS, text)
            assert cell.contains(used), (text, cell, used)
            assert cell.contains(grid.swatch_rect(cell)), text
        for heading in grid.heading_items():
            assert not any(grid.visualItemRect(heading).intersects(cell) for cell in cells)
    finally:
        grid.close()


# -- the armed value ----------------------------------------------------------


def test_a_click_at_a_cells_centre_emits_picked_once() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    grid, _hook = _grid()
    try:
        picked = []
        grid.picked.connect(picked.append)
        QTest.mouseClick(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _center(grid, 21))
        assert picked == [21]
        assert grid.current_value() == 21
        QTest.mouseClick(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _center(grid, 21))
        assert picked == [21]
    finally:
        grid.close()


def test_set_current_value_arms_without_emitting() -> None:
    grid, _hook = _grid()
    try:
        picked = []
        grid.picked.connect(picked.append)
        grid.set_current_value(31)
        assert grid.current_value() == 31
        assert grid.currentItem() is grid.cell_item(31)
        assert picked == []
    finally:
        grid.close()


def test_arming_a_first_row_cell_brings_its_heading_into_view_too() -> None:
    grid, _hook = _grid(_terrain_items(), height=250)
    try:
        grid.scrollToBottom()
        _pump()
        heading, first = grid.heading_items()[1], next(
            grid.item(row) for row in range(grid.row(grid.heading_items()[1]) + 1, grid.count())
        )
        grid.set_current_value(first.data(0x0100))
        _pump()
        viewport = grid.viewport().rect()
        assert viewport.contains(grid.visualItemRect(heading))
        assert viewport.contains(grid.visualItemRect(first))
    finally:
        grid.close()


def test_arrow_keys_cross_sections_and_never_land_on_a_heading() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    from descape.swatch_grid import _HEADING_ROLE

    grid, _hook = _grid(width=300)
    try:
        grid.setFocus()
        grid.set_current_value(10)
        groups = {v: v // 10 for v in grid.values()}
        seen = set()
        for key in [Qt.Key_Down] * 8 + [Qt.Key_Right] * 4 + [Qt.Key_Up] * 8 + [Qt.Key_Left] * 4:
            QTest.keyClick(grid, key)
            current = grid.currentItem()
            assert current is not None and not current.data(_HEADING_ROLE)
            assert grid.current_value() == current.data(Qt.UserRole)
            seen.add(groups[grid.current_value()])
        assert len(seen) >= 3, seen
    finally:
        grid.close()


def test_a_heading_click_keeps_the_armed_cell() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    grid, _hook = _grid()
    try:
        grid.set_current_value(20)
        picked = []
        grid.picked.connect(picked.append)
        heading = grid.heading_items()[2]
        QTest.mouseClick(grid.viewport(), Qt.LeftButton, Qt.NoModifier, grid.visualItemRect(heading).center())
        assert grid.current_value() == 20
        assert grid.currentItem() is grid.cell_item(20)
        assert picked == []
    finally:
        grid.close()


def _frame_sides(grid, value) -> list[bool]:
    """Whether each side of the cell's border carries the Highlight colour: left, top, right, bottom."""
    from PyQt5.QtGui import QPalette

    image = grid.viewport().grab().toImage()
    rect = _rect(grid, value)
    highlight = grid.palette().color(QPalette.Highlight).rgb()
    mid = rect.center()
    points = [(rect.left(), mid.y()), (mid.x(), rect.top()), (rect.right(), mid.y()), (mid.x(), rect.bottom())]
    return [image.pixel(x, y) == highlight for x, y in points]


def test_the_frame_survives_a_ctrl_click_and_a_drag_onto_a_heading() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    grid, _hook = _grid()
    try:
        grid.set_current_value(21)
        _pump()
        assert _frame_sides(grid, 21) == [True] * 4
        assert not any(_frame_sides(grid, 20))

        QTest.mouseClick(grid.viewport(), Qt.LeftButton, Qt.ControlModifier, _center(grid, 21))
        _pump()
        assert grid.selectedItems() == []  # Qt's selection went empty...
        assert grid.current_value() == 21  # ...the armed value did not
        assert _frame_sides(grid, 21) == [True] * 4

        heading = grid.visualItemRect(grid.heading_items()[2]).center()
        QTest.mousePress(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _center(grid, 21))
        QTest.mouseMove(grid.viewport(), heading)
        QTest.mouseRelease(grid.viewport(), Qt.LeftButton, Qt.NoModifier, heading)
        _pump()
        assert grid.current_value() == 21
        assert _frame_sides(grid, 21) == [True] * 4
        assert not any(_frame_sides(grid, 30))
    finally:
        grid.close()


def test_the_swatch_is_drawn_untinted_on_the_armed_cell() -> None:
    grid, hook = _grid()
    try:
        grid.flush_swatches()
        grid.set_current_value(21)
        _pump()
        image = grid.viewport().grab().toImage()
        centre = grid.swatch_rect(_rect(grid, 21)).center()
        assert image.pixel(centre.x(), centre.y()) == hook(21).toImage().pixel(0, 0)
    finally:
        grid.close()


def test_set_visible_values_hides_cells_and_emptied_headings_and_restores_the_current() -> None:
    grid, _hook = _grid()
    try:
        grid.set_current_value(31)
        before = _rect(grid, 40)
        grid.set_visible_values({10, 11, 40, 41, 42, 43, 50})
        _pump()
        assert [h.isHidden() for h in grid.heading_items()] == [False, True, True, False, False]
        assert grid.cell_item(31).isHidden() and grid.currentItem() is None
        assert grid.current_value() == 31  # armed while hidden
        assert _rect(grid, 40).top() < before.top()  # reflowed up
        grid.set_visible_values(set(grid.values()))
        _pump()
        assert grid.currentItem() is grid.cell_item(31)
    finally:
        grid.close()


# -- collapsible sections (TASK-552) --------------------------------------------


def _heading(grid, group: str):
    return next(heading for heading in grid.heading_items() if heading.text().startswith(f"{group} ("))


def _heading_point(grid, group: str):
    return grid.visualItemRect(_heading(grid, group)).center()


def _click_heading(grid, group: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    QTest.mouseClick(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _heading_point(grid, group))
    _pump()


def _hidden(grid, *values) -> list[bool]:
    return [grid.cell_item(v).isHidden() for v in values]


def test_a_heading_click_collapses_its_section_and_a_second_restores_it() -> None:
    grid, _hook = _grid()
    try:
        picked = []
        grid.picked.connect(picked.append)
        below = _rect(grid, 30).top()
        _click_heading(grid, "Grass")
        assert not grid.section_open("Grass")
        assert _hidden(grid, 20, 21, 22) == [True] * 3
        assert not _heading(grid, "Grass").isHidden()
        assert _rect(grid, 30).top() < below  # the next section moved up
        assert grid.section_open("Water") and _hidden(grid, 10) == [False]
        _click_heading(grid, "Grass")
        assert grid.section_open("Grass")
        assert _hidden(grid, 20, 21, 22) == [False] * 3
        assert _rect(grid, 30).top() == below
        assert picked == []
    finally:
        grid.close()


def test_a_click_on_the_armed_sections_own_heading_keeps_the_armed_value() -> None:
    grid, _hook = _grid()
    try:
        grid.set_current_value(20)
        picked = []
        grid.picked.connect(picked.append)
        _click_heading(grid, "Grass")
        assert grid.current_value() == 20
        assert grid.currentItem() is None
        _click_heading(grid, "Grass")
        assert grid.current_value() == 20
        assert grid.currentItem() is grid.cell_item(20)
        assert picked == []
    finally:
        grid.close()


def test_a_double_click_on_a_heading_is_two_toggles() -> None:
    """The whole sequence: QTest.mouseDClick alone sends only the double-click event, which emits nothing."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    grid, _hook = _grid()
    try:
        seen = []
        grid.itemClicked.connect(lambda _item: seen.append(grid.section_open("Grass")))
        point = _heading_point(grid, "Grass")
        QTest.mousePress(grid.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        QTest.mouseRelease(grid.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        QTest.mouseDClick(grid.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        QTest.mouseRelease(grid.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        assert seen == [False, True]
        assert grid.section_open("Grass")
    finally:
        grid.close()


def test_a_drag_from_a_heading_onto_a_cell_toggles_nothing() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    grid, _hook = _grid()
    try:
        QTest.mousePress(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _heading_point(grid, "Grass"))
        QTest.mouseMove(grid.viewport(), _center(grid, 21))
        QTest.mouseRelease(grid.viewport(), Qt.LeftButton, Qt.NoModifier, _center(grid, 21))
        assert grid.section_open("Grass")
    finally:
        grid.close()


@pytest.mark.parametrize("width", WIDTHS)
def test_closed_sections_stack_their_headings_without_overlap(width: int) -> None:
    items = _terrain_items()
    groups = list(dict.fromkeys(item.group for item in items))
    grid, _hook = _grid(items, width=width, height=500)
    try:
        for closing in (groups[::2], groups):  # the second leaves no cell, so the vertical scrollbar drops
            for group in closing:
                grid.set_section_open(group, False)
            _pump()
            viewport = grid.viewport().width()
            headings = [grid.visualItemRect(heading) for heading in grid.heading_items()]
            cells = [_rect(grid, v) for v in _shown_cells(grid)]
            for rect in headings:
                assert rect.width() >= viewport - 2 * grid.spacing(), (rect, viewport)
                assert not any(rect.intersects(cell) for cell in cells), rect
            for a, b in itertools.combinations(headings, 2):
                assert not a.intersects(b)
            assert grid.horizontalScrollBar().maximum() == 0
    finally:
        grid.close()


def test_arrow_keys_skip_a_closed_section_and_never_land_on_a_heading() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    from descape.swatch_grid import _HEADING_ROLE

    grid, _hook = _grid(width=300)
    try:
        grid.setFocus()
        grid.set_section_open("Grass", False)
        grid.set_current_value(10)
        seen = set()
        for key in [Qt.Key_Down] * 8 + [Qt.Key_Right] * 4 + [Qt.Key_Up] * 8 + [Qt.Key_Left] * 4:
            QTest.keyClick(grid, key)
            current = grid.currentItem()
            assert current is not None and not current.data(_HEADING_ROLE)
            assert not current.isHidden()
            seen.add(grid.current_value() // 10)
        assert 2 not in seen
        assert len(seen) >= 3, seen
    finally:
        grid.close()


def test_reopening_a_section_leaves_the_view_where_it_was() -> None:
    """With the armed cell last in a long section, Qt's auto-scroll would take the heading from under the cursor."""
    from descape.value_picker import PickerItem

    last = 127
    items = [PickerItem(f"Top {i}", i, "Top") for i in range(3)]
    items += [PickerItem(f"Long {i:02}", 100 + i, "Long") for i in range(28)]
    grid, _hook = _grid(items, width=300, height=200)
    try:
        grid.set_current_value(last)
        _pump()
        _click_heading(grid, "Long")
        grid.scrollToTop()
        _pump()
        top = grid.visualItemRect(_heading(grid, "Long")).top()
        _click_heading(grid, "Long")
        assert grid.currentItem() is grid.cell_item(last)
        assert grid.visualItemRect(_heading(grid, "Long")).top() == top
        assert not _rect(grid, last).intersects(grid.viewport().rect()), "fixture: the armed cell must be off screen"
        grid.scroll_to_value(last)  # an explicit scroll still works
        _pump()
        assert _rect(grid, last).intersects(grid.viewport().rect())
    finally:
        grid.close()


def test_reveal_opens_only_the_sections_holding_a_wanted_value() -> None:
    grid, _hook = _grid()
    try:
        for group in ("Water", "Grass", "Rock"):
            grid.set_section_open(group, False)
        grid.set_visible_values({10, 20, 40})
        assert not grid.section_open("Water") and not grid.section_open("Grass")
        assert _hidden(grid, 10, 20) == [True, True]
        grid.set_visible_values({10, 40}, reveal=True)
        assert grid.section_open("Water")
        assert _hidden(grid, 10) == [False]
        assert not grid.section_open("Grass") and not grid.section_open("Rock")
        assert _heading(grid, "Grass").isHidden(), "no wanted cell: no heading, closed or not"
        assert _heading(grid, "Rock").isHidden()
        grid.set_visible_values({30})
        assert not _heading(grid, "Rock").isHidden(), "a closed section with a wanted cell keeps its heading"
        assert _hidden(grid, 30) == [True]
    finally:
        grid.close()


def test_arming_a_value_in_a_closed_section_opens_and_frames_it() -> None:
    grid, _hook = _grid()
    try:
        grid.set_section_open("Snow", False)
        grid.set_current_value(42)
        _pump()
        assert grid.section_open("Snow")
        assert grid.currentItem() is grid.cell_item(42)
        assert _frame_sides(grid, 42) == [True] * 4
    finally:
        grid.close()


@pytest.mark.parametrize("value", [None, 999])
def test_arming_nothing_or_an_unknown_value_leaves_the_sections_alone(value) -> None:
    grid, _hook = _grid()
    try:
        grid.set_section_open("Snow", False)
        grid.set_current_value(value)
        assert not grid.section_open("Snow")
    finally:
        grid.close()


def test_a_closed_section_decodes_nothing_until_it_opens() -> None:
    grid, hook = _grid(show=False)
    try:
        grid.set_section_open("Grass", False)
        grid.show()
        grid.flush_swatches()
        assert not {20, 21, 22} & set(hook.calls)
        assert not grid.fill_pending()
        grid.set_section_open("Grass", True)
        assert grid.fill_pending()
        grid.flush_swatches()
        assert {20, 21, 22} <= set(hook.calls)
    finally:
        grid.close()


def test_the_heading_arrow_shows_the_state_left_of_the_heading_text() -> None:
    from PyQt5.QtGui import QPalette

    from descape.swatch_grid import PAD

    grid, _hook = _grid()
    try:
        heading = grid.visualItemRect(_heading(grid, "Grass"))
        arrow = grid.arrow_rect(heading)
        opened = grid.viewport().grab().toImage()
        grid.set_section_open("Grass", False)
        _pump()
        closed = grid.viewport().grab().toImage()
        assert grid.visualItemRect(_heading(grid, "Grass")) == heading
        assert opened.copy(arrow) != closed.copy(arrow)

        base = grid.palette().color(QPalette.Base).rgb()
        for image in (opened, closed):
            ink = [
                x for x in range(heading.left(), heading.right())
                if any(image.pixel(x, y) != base for y in range(heading.top(), heading.bottom()))
            ]
            arrow_ink = [x for x in ink if x <= arrow.right()]
            text_ink = [x for x in ink if x > arrow.right()]
            assert arrow_ink and text_ink
            assert min(text_ink) >= arrow.right() + PAD
    finally:
        grid.close()


# -- swatch fill --------------------------------------------------------------


def test_no_swatch_is_asked_for_before_the_first_show() -> None:
    grid, hook = _grid(show=False)
    try:
        grid.set_visible_values({10, 11})
        grid.refresh_swatches()
        _pump()
        assert hook.calls == []
        assert not grid.fill_pending()
        grid.show()
        assert grid.fill_pending()
    finally:
        grid.close()


def test_a_flush_fills_every_shown_cell_and_stops_the_timer() -> None:
    grid, hook = _grid(show=False)
    try:
        grid.set_visible_values({10, 11, 12, 20})
        grid.show()
        grid.flush_swatches()
        assert not grid.fill_pending()
        assert sorted(hook.calls) == [10, 11, 12, 20]
        for value in (10, 11, 12, 20):
            assert not grid.cell_item(value).icon().isNull()
        assert grid.cell_item(30).icon().isNull()

        grid.set_visible_values(set(grid.values()))
        assert grid.fill_pending()  # unhiding iconless cells re-arms
        grid.flush_swatches()
        assert len(hook.calls) == len(grid.values())

        grid.refresh_swatches()
        assert grid.fill_pending()
        assert all(grid.cell_item(v).icon().isNull() for v in grid.values())
        grid.hide()
        assert not grid.fill_pending()
    finally:
        grid.close()


def test_the_timer_fills_everything_on_its_own_then_stops() -> None:
    from PyQt5.QtWidgets import QApplication

    grid, hook = _grid()
    try:
        for _ in range(200):
            if not grid.fill_pending():
                break
            QApplication.processEvents()
        assert not grid.fill_pending()
        assert sorted(hook.calls) == sorted(grid.values())
    finally:
        grid.close()


def test_cells_in_the_viewport_fill_first() -> None:
    grid, hook = _grid(_terrain_items(), height=200, clock=_Clock())
    try:
        grid.set_current_value(_terrain_items()[-1].value)  # scrolled to the end
        _pump(1)
        _requeue(grid, hook)
        shown = [v for v in grid.values() if _rect(grid, v).intersects(grid.viewport().rect())]
        grid._tick()  # a still clock: the budget never runs out
        assert hook.calls[: len(shown)] == shown
    finally:
        grid.close()


@pytest.mark.parametrize(("cost", "per_tick"), [(0.025, 1), (0.001, 10)])
def test_the_tick_budget_is_time_not_a_count(cost: float, per_tick: int) -> None:
    clock = _Clock()
    grid, hook = _grid(hook=_Hook(clock, cost), clock=clock)
    try:
        _requeue(grid, hook)
        grid._tick()
        assert len(hook.calls) == per_tick
        assert grid.fill_pending()
    finally:
        grid.close()


def test_a_tick_backs_off_while_a_mouse_button_is_held(monkeypatch) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from descape import swatch_grid

    grid, hook = _grid()
    try:
        _requeue(grid, hook)
        held = [Qt.LeftButton]
        # Not monkeypatch.undo(): that would also revert conftest's autouse config redirect.
        monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: held[0]))
        grid._tick()
        assert hook.calls == []
        assert grid._timer.interval() == swatch_grid.HELD_INTERVAL_MS and grid.fill_pending()
        held[0] = Qt.NoButton
        grid._tick()
        assert hook.calls and grid._timer.interval() == 0
    finally:
        grid.close()


def test_set_swatch_px_resizes_cells_without_a_new_decode() -> None:
    grid, hook = _grid()
    try:
        grid.flush_swatches()
        calls = len(hook.calls)
        before = _rect(grid, 20)
        grid.set_swatch_px(64)
        _pump()
        after = _rect(grid, 20)
        assert after.height() == before.height() + 16
        assert grid.iconSize().width() == 64
        assert len(hook.calls) == calls
        assert not grid.fill_pending()
    finally:
        grid.close()
