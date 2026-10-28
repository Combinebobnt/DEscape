"""GH #160's Help menu web links: menu order and labels, the URLs each item
opens, the no-browser fallback, and what Report a Bug's prefilled body may
contain. Links go through conftest's opened_urls fixture: no browser opens."""

from __future__ import annotations

import shutil
from urllib.parse import parse_qs, unquote_plus, urlsplit

import pytest

from descape import __version__, help_guide, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_SEP = "---"
# The menu from the web guide down; Phase B's guide items sit above it.
_LINKS_AND_BELOW = [
    "AoE2 Scenario Editor Guide (ugc.aoe2.rocks)",
    _SEP,
    f"Release Notes for {__version__}",
    "Report a Bug…",
    "DEscape on GitHub",
    _SEP,
    "Debug Log",
    "Perf Trace",
    _SEP,
    "About",
]


def help_menu_texts(window) -> list[str]:
    menu = next(a.menu() for a in window.menuBar().actions() if a.text().replace("&", "") == "Help")
    return [_SEP if a.isSeparator() else a.text().replace("&", "") for a in menu.actions()]


def _bug_body(url: str) -> str:
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{help_guide.REPO_URL}/issues/new"
    return parse_qs(parts.query)["body"][0]


def test_help_menu_ends_with_the_links_then_debug_tools_then_about() -> None:
    window = conftest.blank_window(load=False)
    try:
        texts = help_menu_texts(window)
        assert texts[-len(_LINKS_AND_BELOW) :] == _LINKS_AND_BELOW
    finally:
        conftest.close_window(window)


def test_each_web_item_opens_its_page_and_says_so(opened_urls) -> None:
    window = conftest.blank_window(load=False)
    try:
        expected = {
            window.web_guide_action: "https://ugc.aoe2.rocks/scenarios/",
            window.release_notes_action: f"https://github.com/Combinebobnt/DEscape/releases/tag/v{__version__}",
            window.github_action: "https://github.com/Combinebobnt/DEscape",
        }
        for action, url in expected.items():
            assert "in your web browser" in action.toolTip(), action.text()
            opened_urls.urls.clear()
            action.trigger()
            assert opened_urls.urls == [url], action.text()
            assert window.status_log.toPlainText().splitlines()[-1] == f"Opened {url} in your web browser"
        assert "in your web browser" in window.report_bug_action.toolTip()
        opened_urls.urls.clear()
        window.report_bug_action.trigger()
        assert len(opened_urls.urls) == 1
        body = _bug_body(opened_urls.urls[0])
        assert f"DEscape version: {__version__}" in body
    finally:
        conftest.close_window(window)


def test_no_browser_copies_the_link_and_says_so(opened_urls) -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window(load=False)
    try:
        QApplication.clipboard().clear()
        opened_urls.result = False
        window.github_action.trigger()
        assert opened_urls.urls == [help_guide.REPO_URL]
        assert QApplication.clipboard().text() == help_guide.REPO_URL
        assert window.status_log.toPlainText().splitlines()[-1] == (
            f"Couldn't open a web browser; link copied: {help_guide.REPO_URL}"
        )
    finally:
        conftest.close_window(window)


def test_report_bug_url_carries_version_os_and_build() -> None:
    body = _bug_body(help_guide.report_bug_url("9.8", "Plan9 4th", "from source"))
    lines = body.splitlines()
    assert lines[:3] == ["DEscape version: 9.8", "OS: Plan9 4th", "Build: from source"]
    assert "Help > Debug Log" in body


def test_report_bug_never_carries_home_user_or_scenario_name(monkeypatch, tmp_path, opened_urls) -> None:
    """The privacy guarantee, through the real menu action with a scenario open."""
    import getpass

    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    home = tmp_path / "zz_probe_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(getpass, "getuser", lambda: "zz_probe_user")
    scenario = tmp_path / "zz_probe_scenario.aoe2scenario"
    shutil.copyfile(BLANK_TEMPLATE_PATH, scenario)

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window.load_scenario(scenario)
        assert window.scenario is not None
        window.report_bug_action.trigger()
        assert len(opened_urls.urls) == 1
        url = opened_urls.urls[0]
        for text in (url, unquote_plus(url), _bug_body(url)):
            for token in ("zz_probe_home", "zz_probe_user", "zz_probe_scenario", str(tmp_path)):
                assert token not in text, f"{token!r} leaked into the bug report: {text}"
    finally:
        conftest.close_window(window)


def test_environment_summary_names_the_macos_version(monkeypatch) -> None:
    import platform
    import sys

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    monkeypatch.setattr(platform, "release", lambda: "23.5.0")
    assert help_guide.environment_summary() == ("macOS 14.5", "from source")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert help_guide.environment_summary() == ("macOS 14.5", "release build")


def test_environment_summary_elsewhere_uses_the_release(monkeypatch) -> None:
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "11")
    assert help_guide.environment_summary()[0] == "Windows 11"


def test_help_rows_are_contiguous_and_the_links_ship_unbound() -> None:
    ids = [action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS]
    help_ids = [i for i in ids if i.startswith("help_")]
    first = ids.index(help_ids[0])
    assert ids[first : first + len(help_ids)] == help_ids
    links = ["help_web_guide", "help_release_notes", "help_report_bug", "help_github"]
    assert [i for i in help_ids if i in links] == links
    for action_id in links:
        assert settings.get_default_keybind(action_id) == ""
