"""GH #359 in the window: a scenario load's stages on the status bar's
progress bar, up to the first paint, and none left behind by a failed load."""

from __future__ import annotations

import shutil
import time

import pytest

from descape import scenario_io
from descape.load_progress import SCENARIO_STAGES

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

STAGE_PERCENTS = [percent for _label, percent in SCENARIO_STAGES.values()]


def _spy(window, monkeypatch) -> list[tuple[str, str, int]]:
    """Every load job start and step, as (owner, caption, percent)."""
    progress = window.load_progress
    heard: list[tuple[str, str, int]] = []
    real_start, real_step = progress.start, progress.step

    def start(owner, caption, percent=0):
        heard.append((owner, caption, percent))
        real_start(owner, caption, percent)

    def step(owner, percent, caption=None):
        heard.append((owner, caption, percent))
        real_step(owner, percent, caption)

    monkeypatch.setattr(progress, "start", start)
    monkeypatch.setattr(progress, "step", step)
    return heard


def _loads(heard) -> list[tuple[str, int]]:
    return [(caption, percent) for owner, caption, percent in heard if owner == "load"]


def _pump_until(condition, timeout: float = 15.0) -> None:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting on the first paint"
        time.sleep(0.01)
        QApplication.processEvents()


@pytest.fixture()
def window():
    window = conftest.blank_window(load=False)
    yield window
    conftest.close_window(window)


@pytest.fixture()
def shown():
    window = conftest.shown_window(800, 600)
    yield window
    conftest.close_window(window)


def test_an_open_steps_through_the_stages_and_an_unshown_window_hides_at_return(window, tmp_path, monkeypatch):
    path = tmp_path / "a.aoe2scenario"
    shutil.copy(scenario_io.BLANK_TEMPLATE_PATH, path)
    heard = _spy(window, monkeypatch)
    window.load_scenario(path)
    assert _loads(heard) == [(f"Loading a.aoe2scenario: {label}", percent) for label, percent in SCENARIO_STAGES.values()]
    assert window.load_progress.isHidden() and window.load_progress.job("load") is None, "no paint will come"


def test_a_new_map_says_creating_new_map(window, monkeypatch):
    heard = _spy(window, monkeypatch)
    window._create_new_map(120)
    assert _loads(heard) == [(f"Creating new map: {label}", percent) for label, percent in SCENARIO_STAGES.values()]


def test_a_shown_window_keeps_drawing_the_map_until_the_first_paint(shown):
    shown.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    progress = shown.load_progress
    assert not progress.isHidden(), "the first composite paint comes after load_scenario() returns"
    assert progress.job("load") == (
        f"Loading {scenario_io.BLANK_TEMPLATE_PATH.name}: drawing the map",
        STAGE_PERCENTS[-1],
    )
    _pump_until(progress.isHidden)
    assert "First paint composited in " in shown.status_log.toPlainText()


def test_a_corrupt_file_hides_the_bar_before_its_dialog(window, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    open_document = window.scenario
    bad = tmp_path / "bad.aoe2scenario"
    bad.write_bytes(b"not a scenario")
    shown_then: list[bool] = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *_args, **_kwargs: shown_then.append(window.load_progress.isHidden())))
    window.load_scenario(bad)
    assert shown_then == [True], "the dialog shows no stale caption"
    assert window.load_progress.isHidden() and window.scenario is open_document


def test_a_load_that_raises_past_its_handlers_hides_the_bar(window, monkeypatch):
    def broken_render(*_args, **_kwargs):
        raise RuntimeError("a render that failed")

    monkeypatch.setattr(window, "_render_current", broken_render)
    with pytest.raises(RuntimeError, match="a render that failed"):
        window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert window.load_progress.isHidden() and not window._busy


def test_closing_the_window_drops_a_load_waiting_for_its_paint(shown):
    shown.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert shown.load_progress.job("load") is not None
    conftest.close_window(shown)
    assert shown.load_progress.job("load") is None


def test_a_recovered_untitled_load_names_its_file_and_a_resize_says_resizing(window, tmp_path, monkeypatch):
    from descape.scenario_resize import Anchor

    path = tmp_path / "slot.aoe2scenario"
    shutil.copy(scenario_io.BLANK_TEMPLATE_PATH, path)
    heard = _spy(window, monkeypatch)
    window.load_scenario(path, untitled=True)
    assert _loads(heard)[0] == ("Loading slot.aoe2scenario: reading the file", STAGE_PERCENTS[0])
    heard.clear()
    window._apply_resize(144, 144, Anchor.TOP_LEFT)
    name = window.scenario.path.name
    assert _loads(heard) == [(f"Resizing {name}: {label}", percent) for label, percent in SCENARIO_STAGES.values()]


def test_close_map_drops_a_load_waiting_for_its_paint(shown):
    shown.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert shown.load_progress.job("load") is not None
    shown.close_scenario()
    assert shown.scenario is None and shown.load_progress.isHidden()
