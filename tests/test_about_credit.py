"""Help > About credits AoE2ScenarioParser (GH #113).

The credit is a static string because the frozen build ships no dist-info
(packaging/descape.spec has no copy_metadata), so these tests pin it to the
installed distribution's own METADATA and LICENSE instead. The licence comes
from the LICENSE text, not the METADATA classifier: 0.8.3's classifier says
MIT while its LICENSE file and the upstream repo are GPL-3.0.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _dist():
    from importlib.metadata import distribution

    return distribution("AoE2ScenarioParser")


def test_the_parser_credit_matches_the_installed_distribution() -> None:
    from descape.viewer import PARSER_CREDIT

    meta = _dist().metadata
    author = meta["Author-email"].split("<", 1)[0].strip()
    homepage = next(
        url.split(",", 1)[1].strip() for url in meta.get_all("Project-URL") if url.split(",", 1)[0] == "Homepage"
    )
    assert f"{meta['Name']}, by {author}" in PARSER_CREDIT
    assert homepage in PARSER_CREDIT
    licence = _dist().read_text("LICENSE")
    assert licence is not None, "the dist-info carries no LICENSE file"
    header = " ".join(licence.split()[:6])
    assert header.startswith("GNU GENERAL PUBLIC LICENSE Version 3"), header
    assert "GNU General Public License v3.0" in PARSER_CREDIT


def test_the_about_box_shows_the_parser_credit(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    from descape.viewer import PARSER_CREDIT, ViewerWindow

    conftest.ensure_qapp()
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "about", lambda _parent, _title, text: shown.append(text))
    window = ViewerWindow()
    try:
        window.about_action.trigger()
        assert len(shown) == 1 and PARSER_CREDIT in shown[0], shown
        assert "DEscape" in shown[0]
    finally:
        window.edit_history.mark_saved()
        window.close()
