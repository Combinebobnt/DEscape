"""load_scenario() freezes the loaded document out of the collector's reach
(gc.freeze()), and every way a document ends unfreezes it, so a closed
document is still freed. Asserted as == 0 after the unfreeze rather than a
decrease, since the freeze is process-wide and an earlier test in the same
worker may have left a window loaded."""

from __future__ import annotations

import gc

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _thaw_after():
    yield
    gc.unfreeze()


def test_a_load_freezes_and_close_scenario_unfreezes() -> None:
    gc.unfreeze()
    window = conftest.blank_window()
    try:
        assert gc.get_freeze_count() > 0, "load_scenario did not freeze the loaded document"
        assert window._load_gc_collect_s is not None
        window.close_scenario()
        assert window.scenario is None, "close_scenario refused -- vacuous"
        assert gc.get_freeze_count() == 0, "close_scenario left the closed document frozen"
    finally:
        conftest.close_window(window)


def test_closing_the_window_unfreezes() -> None:
    window = conftest.blank_window()
    assert gc.get_freeze_count() > 0
    conftest.close_window(window)
    assert gc.get_freeze_count() == 0, "closeEvent left the document frozen"


def test_the_next_load_parses_with_the_previous_document_unfrozen(monkeypatch) -> None:
    from descape import viewer

    window = conftest.blank_window()
    try:
        assert gc.get_freeze_count() > 0
        seen: list[int] = []
        real = viewer.load_map_and_units

        def recording(path):
            seen.append(gc.get_freeze_count())
            return real(path)

        monkeypatch.setattr(viewer, "load_map_and_units", recording)
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert seen == [0], "the parse ran with the previous document still frozen"
        assert gc.get_freeze_count() > 0, "the second load did not freeze"
    finally:
        conftest.close_window(window)
