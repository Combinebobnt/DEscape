"""TASK-173 (GH #55): every Map Options row carries a tooltip.

The Custom-victory walk (tests/test_map_options_panel.py) checks only the four
custom rows, and the Diplomacy walk (tests/test_diplomacy_panel.py) only that
panel's rows. This walks every Map Options row in a real viewer, so a new spec
added without a tooltip, or a build path that drops it, fails here.
"""

from __future__ import annotations

import pytest
from test_map_options_panel import BLANK_FIXTURE, TRIGGER_FIXTURE, _close, _options_window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def test_every_map_options_spec_has_its_own_tooltip() -> None:
    """Pass condition: every OptionFieldSpec the Map Options panel renders has a non-empty tooltip."""
    from descape.option_fields import _SPECS

    specs = [spec for spec in _SPECS if spec.panel == "map_options"]
    assert specs, "no Map Options specs -- this would pass vacuously"
    missing = [spec.field_id for spec in specs if not spec.tooltip.strip()]
    assert not missing, f"Map Options rows with no tooltip: {missing}"


@pytest.mark.parametrize("path", [BLANK_FIXTURE, TRIGGER_FIXTURE], ids=lambda p: p.stem)
def test_every_map_options_row_shows_a_tooltip_in_the_viewer(path) -> None:
    """Pass conditions, in a shown viewer switched to Map Options on `path`:
    - the panel built one row for every Map Options spec specs_for() finds on the file;
    - every row's editor shows a non-empty tooltip;
    - every enabled row's tooltip is its own description (the spec tooltip), not a
      read-only or Custom-only reason.
    """
    from descape import option_fields

    window = _options_window(path)
    try:
        panel = window.map_options_panel
        expected = {
            spec.field_id: spec
            for spec in option_fields.specs_for(window.scenario)
            if spec.panel == "map_options"
        }
        assert expected, "no Map Options rows on this file -- this would pass vacuously"
        assert set(panel._widgets) == set(expected)
        enabled = 0
        for field_id, widget in panel._widgets.items():
            tip = widget.toolTip()
            assert tip.strip(), f"{field_id} has no tooltip"
            if widget.isEnabled():
                enabled += 1
                assert tip == expected[field_id].tooltip, (field_id, tip)
        assert enabled, "no enabled row -- the own-description check would pass vacuously"
    finally:
        _close(window)
