"""Trigger panel: tree column widths and the detail form's layout and scrolling."""

from __future__ import annotations

import os
from itertools import pairwise

import pytest
from trigger_panel_helpers import (
    _assert_form_rows_do_not_overlap,
    _group,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _triggers_window,
)

from testkit.qt_window import set_panel_width

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- column and form fitting -------------------------------------------------
#
# Every number here was measured on the shipped fixture at MIN_USEFUL_WIDTH
# before the fitting pass, and each one was a defect the panel shipped with.


def _select_an_effect(panel):
    """Select an effect with long field labels -- the worst case for the form."""
    from PyQt5.QtWidgets import QApplication

    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        QApplication.processEvents()
        effects = _group(panel, "effect")
        if effects.childCount():
            panel.entry_tree.setCurrentItem(effects.child(0))
            QApplication.processEvents()
            return
    raise AssertionError("the fixture carries no effects")


def test_the_trigger_column_fits_its_names_instead_of_the_whole_pane() -> None:
    """It was pinned at MIN_USEFUL_WIDTH (340) while the longest name needed
    246, so a list of short names always carried a horizontal scrollbar with
    nothing to scroll to. Three columns now: position and ID are asserted
    with >=, not ==, since ResizeToContents sizes a section to the header's
    own hint too, and both headers are wider than a one-digit fixture value --
    sizeHintForColumn() looks at row content only. Trigger keeps strict
    equality, unchanged from before the numeric columns existed. The status
    column (TASK-317) joins the >= group only while shown: hidden, it is 0
    wide with a positive size hint, and the default marker hides it."""
    from descape import settings

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        tree = panel.tree
        numeric = (panel._COL_POS, panel._COL_ID)
        assert tree.isColumnHidden(panel._COL_STATUS) and tree.columnWidth(panel._COL_STATUS) == 0
        settings.set_trigger_status_marker("both")
        panel.apply_status_style()
        assert not tree.isColumnHidden(panel._COL_STATUS)
        numeric = (*numeric, panel._COL_STATUS)
        for column in numeric:
            assert tree.columnWidth(column) >= tree.sizeHintForColumn(column)
        assert tree.columnWidth(panel._COL_NAME) == tree.sizeHintForColumn(panel._COL_NAME)
        if sum(tree.columnWidth(c) for c in numeric) + tree.sizeHintForColumn(
            panel._COL_NAME
        ) <= tree.viewport().width():
            assert not tree.horizontalScrollBar().isVisible(), (
                "names that fit must not produce a scrollbar"
            )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_id_column_shows_each_row_s_stable_list_index() -> None:
    """Column 0 renders the same index the row's Qt.UserRole already carries
    -- no new plumbing, just a second visible copy of an existing value. Also
    pins the deliberate choice that the free-text filter stays name-only: a
    needle that is a real row's id but appears in no name must match nothing,
    not fall back to an id match."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        tree = window.trigger_panel.tree
        assert tree.topLevelItemCount() > 0, "the fixture changed"
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            index = item.data(0, Qt.UserRole)
            assert index is not None, "the shipped fixture is flat -- no synthetic header expected"
            assert int(item.text(window.trigger_panel._COL_ID)) == index

        panel = window.trigger_panel
        needle = tree.topLevelItem(0).text(panel._COL_ID)
        panel.filter_edit.setText(needle)
        assert all(tree.topLevelItem(i).isHidden() for i in range(tree.topLevelItemCount())), (
            "an id-shaped needle must not match by id -- the filter only reads the name column"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_detail_column_is_reachable_and_keeps_a_minimum_share() -> None:
    """Two separate defects in one tree.

    Item was fixed at 240 of a 340 px pane, leaving Detail 98 px. Worse, Detail
    itself was left at Qt's 100 px default with stretchLastSection off, so a
    1263 px detail string was cut at 100 px *at every scroll position* rather
    than merely off-screen.
    """
    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        tree = panel.entry_tree

        assert tree.columnWidth(1) == tree.sizeHintForColumn(1), (
            "the detail column must be content-sized, or scrolling cannot reach its text"
        )
        visible_detail = tree.viewport().width() - tree.columnWidth(0)
        assert visible_detail >= TriggerPanel._MIN_DETAIL_WIDTH, (
            f"only {visible_detail} px of detail is visible before scrolling"
        )
        assert tree.columnWidth(0) <= tree.sizeHintForColumn(0), (
            "the item column must never exceed what its content needs"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_columns_re_fit_when_the_pane_is_resized() -> None:
    """The cap is a function of the viewport, so computing it only at the width
    the panel happened to be built at makes it meaningless."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
        tree = panel.entry_tree

        # The narrowest reachable pane: Triggers' drag floor lifts this (TASK-560).
        set_panel_width(window, settings.MIN_SPLIT_PANE)
        QApplication.processEvents()
        narrow = tree.viewport().width() - tree.columnWidth(0)
        assert narrow >= TriggerPanel._MIN_DETAIL_WIDTH, (
            f"narrowing the pane left only {narrow} px of detail visible"
        )

        set_panel_width(window, 600)
        QApplication.processEvents()
        assert tree.columnWidth(0) == tree.sizeHintForColumn(0), (
            "a pane with room to spare must give the item column its full content width"
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
@pytest.mark.parametrize("what", ["effect", "trigger"])
def test_a_long_form_scrolls_instead_of_crushing_its_rows(what: str) -> None:
    """Found by screenshot, and invisible to every other measurement: host
    width, scrollbar state and field widths all read correct while each row was
    squeezed to 6 px and drew over the next one.

    The cause is that a wrapped row's height depends on its width, and a
    QScrollArea sizes its widget from sizeHint(), computed as if nothing
    wrapped -- so the form was handed the unwrapped height and overflowed
    inside it. Asking the layout for heightForWidth, *after* activating it, is
    what makes the vertical scrollbar appear instead.

    The trigger form adds full-width caption rows and GH #141's long-label
    rows, which WrapLongRows puts on two lines (GH #139).
    """
    from PyQt5.QtWidgets import QApplication, QFormLayout

    from testkit.qt_window import FONT_DPI_OVERRIDE_ENV

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        if what == "effect":
            _select_an_effect(panel)
        else:
            panel.tree.setCurrentItem(panel.tree.topLevelItem(0))
            QApplication.processEvents()
        form = panel.property_form
        assert len(panel._rows) >= 8, "this fixture no longer exercises a long form"
        walked, wrapped = _assert_form_rows_do_not_overlap(form)
        assert walked >= len(panel._rows)
        if what == "trigger":
            spanning = [r for r in range(form.rowCount()) if form.itemAt(r, QFormLayout.SpanningRole)]
            assert len(spanning) >= 4, "the two prose captions and boxes are full-width rows"
            if not os.environ.get(FONT_DPI_OVERRIDE_ENV):
                # At the pinned baseline font both string table id rows wrap; a smaller DPI fits them beside.
                assert wrapped >= 2, "the wrapped-row check walked nothing"

        assert panel.property_host.height() > panel.property_area.viewport().height(), (
            "a form taller than its pane must make the host taller, not compress the rows"
        )
        assert panel.property_area.verticalScrollBar().isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_resizing_the_panel_refits_the_form_height() -> None:
    """A panel resize reaches the form only through TriggerPanel.resizeEvent.
    The form's height is a function of its width (wrapped rows), so the host's
    must follow it."""
    from PyQt5.QtWidgets import QApplication

    from testkit.qt_window import FONT_DPI_OVERRIDE_ENV

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_an_effect(panel)
        form = panel.property_form
        needed = []
        for panel_width in (600, 330, 900):
            set_panel_width(window, panel_width)
            QApplication.processEvents()
            width = panel.property_area.viewport().width()
            needed.append(max(form.minimumSize().height(), form.heightForWidth(width)))
            assert panel.property_host.minimumHeight() == needed[-1], (panel_width, width)
            _assert_form_rows_do_not_overlap(form)
        if not os.environ.get(FONT_DPI_OVERRIDE_ENV):
            # At the pinned baseline font a step changes the wrap; at 72 DPI no row wraps at these widths.
            assert all(a != b for a, b in pairwise(needed)), f"a step kept its height, so it checks nothing: {needed}"
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.font_sensitive
def test_no_field_demands_more_width_than_the_pane_can_give() -> None:
    """ObjectAttribute's longest entry made its combo 430 px wide against a
    340 px panel, so the form sat permanently below its own stated minimum.
    That is what made every attempt to narrow the label column tip the whole
    form into horizontal overflow instead.

    Walks every row of every entry rather than sampling one. The offending
    combo is on a single effect of a single trigger, so a test that picked "an
    effect" passed while the defect was still there -- which is exactly what
    the first version of this test did.
    """
    from PyQt5.QtWidgets import QApplication

    from descape.trigger_panel import TriggerPanel

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        checked = 0
        for i in range(panel.tree.topLevelItemCount()):
            panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
            QApplication.processEvents()
            for t in range(panel.entry_tree.topLevelItemCount()):
                top = panel.entry_tree.topLevelItem(t)
                for c in range(top.childCount()):
                    panel.entry_tree.setCurrentItem(top.child(c))
                    QApplication.processEvents()
                    # GH #138's Set / Go to / Reset rows live outside _rows.
                    groups = [
                        (f"{group} group row", row.set_button.parentWidget()) for group, row in panel._group_rows.items()
                    ]
                    if panel._instruction_preview is not None:  # GH #140's preview row, also outside _rows
                        groups.append(("instruction preview", panel._instruction_preview))
                    for name, widget in [(spec.name, widget) for spec, *_rest, widget in panel._rows] + groups:
                        checked += 1
                        assert (
                            widget.minimumSizeHint().width() <= TriggerPanel.MIN_USEFUL_WIDTH
                        ), (
                            f"{name}'s editor demands "
                            f"{widget.minimumSizeHint().width()} px of a "
                            f"{TriggerPanel.MIN_USEFUL_WIDTH} px panel"
                        )
                    assert (
                        panel.property_form.minimumSize().width() <= panel.property_host.width()
                    ), "the form's minimum width exceeds the space it is given"
                    assert not panel.property_area.horizontalScrollBar().isVisible()
        # 43 on the shipped fixture. A floor, so a fixture that lost its
        # entries cannot make this pass vacuously.
        assert checked >= 40, f"only {checked} fields walked -- the fixture changed"
    finally:
        window.edit_history.mark_saved()
        window.close()
