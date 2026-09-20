"""Covers scenario_io.xs_attachment() and the info panel's two read-only XS
lines (phase 5, P5-b). The file-level XS surface is read-only by design: see
the accessor's own docstring for why no write path may touch script_name."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.scenario_io import load_map_and_units, parse_triggers, xs_attachment

import conftest

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def test_embedded_xs_is_unknown_until_triggers_are_parsed() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    assert xs_attachment(loaded) == ("", None)
    assert parse_triggers(loaded) is not None
    assert xs_attachment(loaded) == ("", 0)


def test_the_accessor_never_forces_a_trigger_parse() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    xs_attachment(loaded)
    assert loaded.trigger_read_supported is None
    assert "Files" not in loaded._scenario.sections


def test_a_failed_trigger_parse_reports_unknown_not_zero() -> None:
    from descape.viewer import _xs_info_lines

    loaded = load_map_and_units(TRIGGER_FIXTURE)
    loaded.trigger_read_supported = False
    assert xs_attachment(loaded) == ("", None)
    assert _xs_info_lines(loaded)[1] == "Embedded XS: (unknown: Triggers section can't be read)"


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_entering_triggers_mode_refreshes_the_info_panel() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(TRIGGER_FIXTURE)
        QApplication.processEvents()
        text = window.info.toPlainText()
        assert "XS script file: (none)" in text
        assert "Embedded XS: (unknown until triggers are parsed)" in text

        window.mode_combo.setCurrentText("Triggers")
        QApplication.processEvents()
        assert "Embedded XS: (none)" in window.info.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.corpus
def test_no_corpus_file_uses_the_file_level_xs_surface(scenario_path) -> None:
    """Pins the census finding that 0 real files attach or embed a script.
    A new corpus file that does should fail loudly, since the read-only
    display and the no-write-path decision both rest on it."""
    loaded = load_map_and_units(scenario_path)
    parse_triggers(loaded)
    script_name, embedded = xs_attachment(loaded)
    assert script_name in ("", None), f"{scenario_path.name} names {script_name!r}"
    assert embedded in (0, None), f"{scenario_path.name} embeds {embedded} chars"
