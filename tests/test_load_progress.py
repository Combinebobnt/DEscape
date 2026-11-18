"""GH #359: the status bar's load progress widget, its owners and captions."""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture()
def progress():
    from PyQt5.QtWidgets import QStatusBar

    from descape.load_progress import LoadProgress

    conftest.ensure_qapp()
    bar = QStatusBar()
    widget = LoadProgress()
    bar.addPermanentWidget(widget)
    bar.show()
    yield widget
    bar.close()
    bar.deleteLater()


def test_hidden_until_a_job_starts_and_after_the_last_finishes(progress):
    assert progress.isHidden()
    progress.start("load", "Loading a: reading the file", 10)
    assert not progress.isHidden()
    assert progress.caption.text() == "Loading a: reading the file" and progress.bar.value() == 10
    progress.step("load", 35, "Loading a: preparing the map")
    assert progress.caption.text() == "Loading a: preparing the map" and progress.bar.value() == 35
    progress.finish("load")
    assert progress.isHidden() and progress.job("load") is None


def test_a_step_without_a_caption_keeps_the_caption(progress):
    progress.start("data_mod", "Parsing M: starting")
    progress.step("data_mod", 50)
    assert progress.job("data_mod") == ("Parsing M: starting", 50)
    assert progress.bar.text() == "50%"


def test_the_load_outranks_a_data_mod_parse_which_shows_again_after(progress):
    progress.start("data_mod", "Parsing M: starting", 5)
    progress.start("load", "Loading a: reading the file", 10)
    assert progress.shown_owner() == "load" and progress.caption.text() == "Loading a: reading the file"
    progress.step("data_mod", 45, "Parsing M: building the unit art (2 of 12)")
    assert progress.caption.text() == "Loading a: reading the file", "the hidden job moves on unseen"
    progress.finish("load")
    assert not progress.isHidden()
    assert progress.caption.text() == "Parsing M: building the unit art (2 of 12)" and progress.bar.value() == 45


def test_a_finished_job_ignores_late_steps_and_a_restart_replaces(progress):
    progress.start("load", "Loading a: reading the file", 10)
    progress.finish("load")
    progress.step("load", 60)
    progress.finish("load")
    assert progress.isHidden() and progress.job("load") is None
    progress.start("load", "Loading a: drawing the map", 80)
    progress.start("load", "Loading b: reading the file", 10)
    assert progress.job("load") == ("Loading b: reading the file", 10)


def test_percent_is_clamped_and_an_unknown_owner_raises(progress):
    progress.start("load", "x", -5)
    assert progress.bar.value() == 0
    progress.step("load", 250)
    assert progress.bar.value() == 100
    for call in (lambda: progress.start("save", "x"), lambda: progress.step("save", 1), lambda: progress.finish("save")):
        with pytest.raises(ValueError, match="unknown load progress owner 'save'"):
            call()


def test_scenario_captions_and_stages():
    from descape.load_progress import SCENARIO_STAGES, scenario_caption

    assert list(SCENARIO_STAGES) == ["read", "prepare", "panels", "draw"]
    percents = [percent for _label, percent in SCENARIO_STAGES.values()]
    assert percents == sorted(percents) and percents[0] > 0 and percents[-1] < 100
    assert scenario_caption("a.aoe2scenario", "open", "read") == "Loading a.aoe2scenario: reading the file"
    assert scenario_caption("Untitled", "new", "draw") == "Creating new map: drawing the map"
    assert scenario_caption("a.aoe2scenario", "resize", "panels") == "Resizing a.aoe2scenario: filling the panels"


def test_a_new_caption_is_laid_out_without_the_event_loop(progress):
    """A blocked load runs no posted events: each step lays the caption out itself, and the bar stays its own width."""
    from descape.load_progress import BAR_WIDTH

    progress.start("load", "Loading a: reading the file", 10)
    progress.step("load", 35, "Loading a.aoe2scenario: preparing the map, a much longer caption than before")
    assert progress.caption.width() >= progress.caption.sizeHint().width()
    assert progress.bar.width() == BAR_WIDTH


def test_a_wide_status_bar_keeps_the_widget_at_its_size_hint(progress):
    progress.parentWidget().resize(2400, 40)
    progress.start("load", "Loading a: reading the file", 10)
    assert progress.width() == progress.sizeHint().width()
