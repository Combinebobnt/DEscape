"""The width-driven "More Tools" dropdown for toolbar tool buttons that
don't fit at the current window width.

Two layers: descape/toolbar_overflow.partition() is pure arithmetic, tested
with no QApplication at all. Everything else drives a real, shown, offscreen
ViewerWindow (conftest.shown_terrain_window()) -- membership/visibility
here depends on real sizeHint()s and real QToolBar layout, which an unshown
window can't give (see that fixture's own docstring).

One deliberate deviation from the plan's original wording: mode-inapplicable
tools (ToolDef.modes) are hidden via QAction.setVisible(), not removed from
the toolbar -- that's how Stage 2 actually shipped, and a hidden action's
sizeHint() and layout footprint are both unaffected (confirmed empirically),
so overflow partitioning works the same either way. Assertions below check
"invisible everywhere" for an inapplicable tool, not "absent from the
toolbar's actions() list".

Every ViewerWindow() here must call edit_history.mark_saved() before
close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape import settings, viewer_common
from descape.toolbar_overflow import partition

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# ---------------------------------------------------------------------------
# Pure partition() tests -- no QApplication.
# ---------------------------------------------------------------------------


def test_partition_nothing_overflows_when_everything_fits():
    items = [("a", 40), ("b", 50), ("c", 30)]
    on_bar, overflowed = partition(200, items, more_button_px=90)
    assert on_bar == ["a", "b", "c"]
    assert overflowed == []


def test_partition_empty_input():
    assert partition(500, [], more_button_px=90) == ([], [])


def test_partition_two_pass_boundary():
    """Without reserving the button, "a"+"b" (90) fit in 100 but "c" (30)
    would make 120 -- so pass 1 already needs a second pass. With
    more_button_px=20 reserved, budget drops to 80: "a"+"b" (90) no longer
    fit either, only "a" (40) does. This is the case a naive single pass
    gets wrong -- fitting "a"+"b" against the unreserved 100 and never
    re-checking against the reserved 80."""
    items = [("a", 40), ("b", 50), ("c", 30)]
    on_bar, overflowed = partition(100, items, more_button_px=20)
    assert on_bar == ["a"]
    assert overflowed == ["b", "c"]


def test_partition_is_a_strict_suffix_not_a_knapsack():
    """"c" (5) would fit in the leftover space after "b" is skipped, but
    overflow must be a strict right-hand suffix -- no reordering, or the bar
    would reshuffle under the cursor during a drag-resize."""
    items = [("a", 40), ("b", 70), ("c", 5)]
    on_bar, overflowed = partition(95, items, more_button_px=0)
    assert on_bar == ["a"]
    assert overflowed == ["b", "c"]


def test_partition_reserves_button_only_on_the_overflow_path():
    """If pass 1 already fits, more_button_px is never subtracted -- a huge
    more_button_px must not force a button/overflow that pass 1 said wasn't
    needed."""
    items = [("a", 40), ("b", 50)]
    on_bar, overflowed = partition(90, items, more_button_px=10_000)
    assert on_bar == ["a", "b"]
    assert overflowed == []


# ---------------------------------------------------------------------------
# Real-window membership/visibility tests.
# ---------------------------------------------------------------------------

_WIDTHS = [800, 1000, 1280, 1600]
_MODES = [("View", "view"), ("Terrain", "terrain"), ("Units", "units")]


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("mode_text,mode_id", _MODES)
def test_every_tool_is_on_bar_xor_in_menu_or_neither_if_inapplicable(mode_text, mode_id, width):
    window = conftest.shown_terrain_window(width=width)
    try:
        window.mode_combo.setCurrentText(mode_text)
        conftest.ensure_qapp()
        from PyQt5.QtWidgets import QApplication

        QApplication.processEvents()

        menu_actions = window.more_tools_menu.actions()
        for tool in settings.TOOLS:
            action = getattr(window, f"{tool.tool_id}_action")
            on_bar_visible = action in window.main_toolbar.actions() and action.isVisible()
            in_menu = action in menu_actions
            if viewer_common.tool_applicable(tool.tool_id, mode_id):
                assert on_bar_visible != in_menu, (tool.tool_id, mode_text, width)
            else:
                assert not on_bar_visible and not in_menu, (tool.tool_id, mode_text, width)
    finally:
        conftest.close_window(window)


def test_nothing_overflows_at_a_generously_wide_window():
    window = conftest.shown_terrain_window(width=3000)
    try:
        assert not window.more_tools_action.isVisible()
        assert window.more_tools_menu.actions() == []
        assert window.more_tools_button.text() == "More Tools ▾"
    finally:
        conftest.close_window(window)


def test_something_overflows_at_the_app_minimum_width():
    """settings.MIN_WINDOW_WIDTH is the narrowest width the app allows at
    all -- Terrain mode's six tools plus the pinned prefix don't fit there,
    so the button must be live (Stage 0's own measurement: pinned ~514px,
    six tool buttons ~500px combined, leaving well under 800px total)."""
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        assert window.more_tools_action.isVisible()
        assert window.more_tools_menu.actions() != []
        # Pan is always first in settings.TOOLS -- the suffix rule means it
        # is the last thing to ever overflow.
        assert window.pan_action in window.main_toolbar.actions()
        assert window.pan_action.isVisible()
    finally:
        conftest.close_window(window)


def test_overflow_is_a_suffix_of_tools_declaration_order():
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        applicable = [t.tool_id for t in settings.TOOLS if viewer_common.tool_applicable(t.tool_id, "terrain")]
        on_bar = [
            t for t in applicable
            if getattr(window, f"{t}_action") in window.main_toolbar.actions()
            and getattr(window, f"{t}_action").isVisible()
        ]
        overflowed = [t for t in applicable if t not in on_bar]
        assert on_bar == applicable[: len(on_bar)]
        assert overflowed == applicable[len(on_bar) :]
    finally:
        conftest.close_window(window)


def test_active_tool_label_tracks_when_overflowed():
    window = conftest.shown_terrain_window(width=settings.MIN_WINDOW_WIDTH)
    try:
        overflowed_tool = None
        for tool in settings.TOOLS:
            if viewer_common.tool_applicable(tool.tool_id, "terrain"):
                action = getattr(window, f"{tool.tool_id}_action")
                if action in window.more_tools_menu.actions():
                    overflowed_tool = tool
                    break
        assert overflowed_tool is not None, "expected at least one overflowed tool at MIN_WINDOW_WIDTH"

        window._on_tool_selected(overflowed_tool.tool_id)
        assert window.more_tools_button.text() == f"{overflowed_tool.label} ▾"

        window._on_tool_selected("pan")
        assert window.more_tools_button.text() == "More Tools ▾"
    finally:
        conftest.close_window(window)


def test_resize_wide_then_narrow_then_wide_returns_to_the_same_state():
    """No accumulated drift across a resize round-trip -- e.g. a tool action
    left behind in the menu (or the toolbar) after its width class changes
    twice."""
    window = conftest.shown_terrain_window(width=3000)
    try:
        before = {
            t.tool_id: getattr(window, f"{t.tool_id}_action") in window.main_toolbar.actions()
            for t in settings.TOOLS
        }
        window.resize(settings.MIN_WINDOW_WIDTH, 800)
        conftest.ensure_qapp()
        from PyQt5.QtWidgets import QApplication

        QApplication.processEvents()
        window.resize(3000, 800)
        QApplication.processEvents()
        after = {
            t.tool_id: getattr(window, f"{t.tool_id}_action") in window.main_toolbar.actions()
            for t in settings.TOOLS
        }
        assert before == after
        assert not window.more_tools_action.isVisible()
    finally:
        conftest.close_window(window)
