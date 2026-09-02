"""Coverage for MessagesPanel, standalone -- no ViewerWindow needed.

Unlike MapOptionsPanel/TriggerPanel's own tests, this one never builds a full
offscreen ViewerWindow: MessagesPanel takes a LoadedScenario directly and has
no splitter/mode-combo geometry to assert on, so a bare QApplication plus the
panel is enough. Messages mode is already wired into viewer.py; an
in-window test analogous to test_diplomacy_panel.py's has not been added yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "descape" / "templates" / "blank_120x120.aoe2scenario"
F7_YORK_PATH = Path(__file__).resolve().parent.parent / "examples" / "F7_3_York (865).aoe2scenario"


def _panel(loaded=None, **kwargs):
    from descape.messages_panel import MessagesPanel

    conftest.ensure_qapp()
    panel = MessagesPanel()
    if loaded is not None:
        panel.show_scenario(loaded, **kwargs)
    return panel


def _loaded(path=FIXTURE_PATH):
    from descape.scenario_io import load_map_and_units

    return load_map_and_units(path)


def test_clear_document_shows_no_document_status() -> None:
    panel = _panel()
    assert panel.status.text() == panel._NO_DOCUMENT


def test_show_scenario_builds_a_widget_per_field() -> None:
    from descape.messages_fields import MESSAGE_FIELDS

    panel = _panel(_loaded(), editable=True)
    for spec in MESSAGE_FIELDS:
        widget = panel.widget_for(spec.field_id)
        assert widget is not None
        assert widget.toPlainText() == panel.current_values()[spec.field_id]


def test_browsing_every_field_reports_no_edit() -> None:
    """Same load-bearing shape as MapOptionsPanel's own
    test_browsing_every_row_reports_no_edit: focusing and leaving a field
    with its own current text (editingFinished fires on focus-out whether or
    not the text changed) must not fire the callback -- a phantom record
    would dirty a document nothing was ever done to."""
    from descape.messages_fields import MESSAGE_FIELDS

    calls = []
    panel = _panel(_loaded(), editable=True)
    panel._on_message_field = lambda *args: calls.append(args)
    for spec in MESSAGE_FIELDS:
        widget = panel.widget_for(spec.field_id)
        widget.setPlainText(widget.toPlainText())
        widget.editingFinished.emit()
    assert calls == []


def test_typing_does_not_fire_the_callback_until_focus_is_lost() -> None:
    """The bug this guards against: QPlainTextEdit's only built-in signal is
    textChanged, which fires per keystroke -- wiring the callback straight to
    it would push one undo record (and one status-log line) per character
    typed. _CommitOnBlurTextEdit's editingFinished is the fix; this pins that
    setPlainText() alone (textChanged's trigger) does nothing on its own."""
    calls = []
    panel = _panel(_loaded(), editable=True)
    panel._on_message_field = lambda *args: calls.append(args)
    widget = panel.widget_for("hints")
    widget.setPlainText("a brand new hint")
    assert calls == []


def test_editing_a_field_reports_exactly_one_edit() -> None:
    calls = []
    panel = _panel(_loaded(), editable=True)
    panel._on_message_field = lambda *args: calls.append(args)
    widget = panel.widget_for("hints")
    widget.setPlainText("a brand new hint")
    widget.editingFinished.emit()
    assert calls == [("hints", "a brand new hint")]
    assert panel.current_values()["hints"] == "a brand new hint"


def test_read_only_disables_every_editor() -> None:
    panel = _panel(_loaded(), editable=False)
    from descape.messages_fields import MESSAGE_FIELDS

    for spec in MESSAGE_FIELDS:
        assert not panel.widget_for(spec.field_id).isEnabled()


def test_values_override_shows_pending_edits() -> None:
    panel = _panel(_loaded(), editable=True, values={"hints": "pending text"})
    assert panel.widget_for("hints").toPlainText() == "pending text"


def test_clear_document_after_a_scenario_resets_status() -> None:
    panel = _panel(_loaded(), editable=True)
    panel.show_scenario(None)
    assert panel.status.text() == panel._NO_DOCUMENT
    assert panel.widget_for("hints") is None


# -- set string id warning row (corpus-only: no shipped fixture has one) ----


def _requires_york():
    if not F7_YORK_PATH.exists():
        pytest.skip("examples/ corpus not present")
    return _loaded(F7_YORK_PATH)


@pytest.mark.corpus
def test_set_string_id_shows_a_warning_and_clear_button() -> None:
    panel = _panel(_requires_york(), editable=True)
    assert panel.widget_for("hints_id") is not None  # the clear button
    assert "hints" in panel._warning_labels


@pytest.mark.corpus
def test_clearing_a_string_id_reports_the_id_field_edit() -> None:
    calls = []
    panel = _panel(_requires_york(), editable=True)
    panel._on_message_field = lambda *args: calls.append(args)
    from descape.messages_fields import STRING_ID_UNSET

    panel.widget_for("hints_id").click()
    assert calls == [("hints_id", STRING_ID_UNSET)]


@pytest.mark.corpus
def test_no_set_id_shows_no_warning_row() -> None:
    panel = _panel(_loaded(), editable=True)
    assert panel.widget_for("hints_id") is None
    assert "hints" not in panel._warning_labels
