"""The AoE2:DE profile folder row and its scanned-folders list in Settings > Mods
(GH #126 follow-up; their own tab since GH #244). conftest's _isolated_settings redirects config.yaml and
drops AOE2DE_PROFILE_PATH; each test builds a fake profile under tmp_path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _no_real_install(monkeypatch):
    """A run with AOE2DE_INSTALL_PATH set (the corpus tier's) would auto-detect the real profile."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()
    yield
    asset_source.clear_install_caches()


def _profile_tree(root: Path) -> Path:
    """A profile with accounts 0 and 7656; 7656 has one enabled and one disabled mod."""
    profile = root / "Age of Empires 2 DE"
    (profile / "0" / "resources" / "_common").mkdir(parents=True)
    mods = profile / "7656" / "mods"
    for name in ("12_Alpha AI", "34_Off AI"):
        (mods / "subscribed" / name / "resources" / "_common").mkdir(parents=True)
    status = {"Mods": [
        {"Enabled": True, "Path": "subscribed//12_Alpha AI"},
        {"Enabled": False, "Path": "subscribed//34_Off AI"},
    ]}
    (mods / "mod-status.json").write_text(json.dumps(status))
    return profile


def _config() -> dict:
    from descape import asset_source

    if not asset_source.CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(asset_source.CONFIG_PATH.read_text()) or {}


def _list_texts(dialog) -> list[str]:
    view = dialog.profile_roots_list
    return [view.item(i).text() for i in range(view.count())]


def _select_tab(dialog, title: str):
    """Select the Settings tab titled `title`; returns its page (the scroll area for a scrolled tab)."""
    from PyQt5.QtWidgets import QTabWidget

    tabs = dialog.findChild(QTabWidget)
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == title)
    tabs.setCurrentIndex(index)
    return tabs.widget(index)


@pytest.fixture
def opened(monkeypatch):
    """(dialog, window, rescans), with invalidate_ai_choices() counted."""
    dialog, window = conftest.dialog_and_window()
    rescans: list[bool] = []
    monkeypatch.setattr(window, "invalidate_ai_choices", lambda: rescans.append(True))
    yield dialog, window, rescans
    dialog.close()
    conftest.close_window(window)


def test_with_nothing_configured_the_row_says_so(opened) -> None:
    dialog, _window, _rescans = opened
    assert dialog.profile_path_edit.text() == ""
    assert dialog.profile_path_edit.isEnabled()
    assert not dialog.profile_auto_button.isEnabled()
    assert "No profile folder found" in dialog.profile_status_label.text()
    assert _list_texts(dialog) == ["Nothing yet: set the install path on the General tab."]
    assert not dialog.profile_skipped_label.isVisibleTo(dialog)


def test_load_writes_the_key_lists_the_folders_and_rescans(opened, tmp_path) -> None:
    from descape import content_roots

    dialog, window, rescans = opened
    profile = _profile_tree(tmp_path)
    dialog.profile_path_edit.setText(str(profile))
    dialog.load_profile_path()

    assert _config()[content_roots.PROFILE_CONFIG_KEY] == str(profile)
    assert content_roots.profile_path() == profile
    assert rescans == [True]
    assert dialog.profile_status_label.text() == "From config.yaml."
    assert dialog._profile_status_ok
    assert dialog.profile_auto_button.isEnabled()
    assert _list_texts(dialog) == [
        f"Profile 0: {profile / '0'}",
        f"Profile 7656: {profile / '7656'}",
        f"Mod (subscribed): 12_Alpha AI: {profile / '7656' / 'mods' / 'subscribed' / '12_Alpha AI'}",
    ]
    assert dialog.profile_skipped_label.text() == (
        "1 mod disabled in the game's mod list is skipped; if it is in the profile the game last used "
        "and ships art, it is listed below, off until ticked."
    )
    assert not dialog.profile_skipped_label.isHidden()
    assert f"Profile folder set to {profile}" in window.status_log.toPlainText()


def test_on_two_accounts_the_skipped_note_says_whose_art_mods_are_listed(opened, tmp_path) -> None:
    """GH #264 review: skipped_mods() counts every account's disabled mods, but the Mod art list
    shows only those of the account the game last ran. With a disabled mod in each account, the
    note counts both and says the list below is the last-used profile's."""
    import os

    from descape import content_roots

    dialog, _window, _rescans = opened
    profile = _profile_tree(tmp_path)
    old_mods = profile / "0" / "mods"
    (old_mods / "local" / "Old Art Mod" / "resources" / "_common").mkdir(parents=True)
    (old_mods / "mod-status.json").write_text(json.dumps({"Mods": [{"Enabled": False, "Path": "local//Old Art Mod"}]}))
    os.utime(old_mods / "mod-status.json", (1_000_000, 1_000_000))
    assert content_roots.active_profile_id(profile) == "7656"
    dialog.profile_path_edit.setText(str(profile))
    dialog.load_profile_path()
    note = dialog.profile_skipped_label.text()
    assert note.startswith("2 mods disabled")
    assert "any in the profile the game last used that ship art are listed below" in note


def test_the_guide_says_the_scanned_list_leaves_out_disabled_mods() -> None:
    """The list under the profile row is content_roots() without disabled mods, which the Mod art
    list and File > Data Mod still read: the guide must not call it every folder DEscape scans."""
    guide = (Path(__file__).resolve().parent.parent / "descape" / "help" / "settings.md").read_text(encoding="utf-8")
    mods = guide.split("## Mods", 1)[1].split("\n## ", 1)[0]
    profile_row = next(line for line in mods.splitlines() if line.startswith("- AoE2DE profile folder"))
    assert "disabled" in profile_row


def _pump() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(4):
        QApplication.processEvents()


def _shown_width(dialog) -> int:
    dialog.show()
    _pump()
    return dialog.width()


def test_two_disabled_mods_read_plural_and_leave_the_dialog_width_alone(tmp_path) -> None:
    """GH #126: the footnote's plural branch, and its width.

    Passes when a profile with two disabled mods on disk reads "2 mods
    disabled ... are skipped; any in the profile ..." under the scanned list,
    and the shown dialog with that section populated (list, footnote) is no
    wider than a shown dialog with nothing loaded. The width half is a guard:
    the tabs' minimum (793 px at the pinned test font) already exceeds the
    520 px default, and the footnote wraps."""
    profile = _profile_tree(tmp_path)
    mods = profile / "7656" / "mods"
    (mods / "subscribed" / "56_Second Off AI" / "resources" / "_common").mkdir(parents=True)
    status = json.loads((mods / "mod-status.json").read_text())
    status["Mods"].append({"Enabled": False, "Path": "subscribed//56_Second Off AI"})
    (mods / "mod-status.json").write_text(json.dumps(status))

    empty_dialog, empty_window = conftest.dialog_and_window()
    try:
        _select_tab(empty_dialog, "Mods")
        empty_width = _shown_width(empty_dialog)
        assert not empty_dialog.profile_skipped_label.isVisibleTo(empty_dialog)
    finally:
        empty_dialog.close()
        conftest.close_window(empty_window)

    dialog, window = conftest.dialog_and_window()
    try:
        dialog.profile_path_edit.setText(str(profile))
        dialog.load_profile_path()
        assert dialog.profile_skipped_label.text() == (
            "2 mods disabled in the game's mod list are skipped; any in the profile the game last used "
            "that ship art are listed below, off until ticked."
        )
        assert not dialog.profile_skipped_label.isHidden()
        _select_tab(dialog, "Mods")
        assert _shown_width(dialog) <= empty_width
        assert dialog.profile_skipped_label.isVisibleTo(dialog)
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.parametrize("kind", ["no-numeric-folder", "not-a-folder", "empty"])
def test_load_refuses_a_folder_without_a_profile_id(opened, tmp_path, kind: str) -> None:
    dialog, _window, rescans = opened
    if kind == "no-numeric-folder":
        folder = tmp_path / "Games"
        (folder / "logs").mkdir(parents=True)
        text = str(folder)
    elif kind == "not-a-folder":
        text = str(tmp_path / "missing")
    else:
        text = "   "
    dialog.profile_path_edit.setText(text)
    dialog.load_profile_path()
    assert _config() == {}
    assert rescans == []
    assert dialog._profile_status_ok is False
    expected = "Enter or browse" if kind == "empty" else "Nothing saved"
    assert expected in dialog.profile_status_label.text()


def test_use_auto_detect_removes_the_key(tmp_path, monkeypatch) -> None:
    from descape import asset_source, content_roots

    profile = _profile_tree(tmp_path)
    asset_source.save_profile_path_to_config(profile)
    asset_source.save_install_path_to_config(tmp_path / "AoE2DE")  # other keys survive
    dialog, window = conftest.dialog_and_window()
    try:
        rescans: list[bool] = []
        monkeypatch.setattr(window, "invalidate_ai_choices", lambda: rescans.append(True))
        assert dialog.profile_path_edit.text() == str(profile)
        assert dialog.profile_auto_button.isEnabled()
        dialog.profile_auto_button.click()
        assert content_roots.PROFILE_CONFIG_KEY not in _config()
        assert _config()["aoe2de_install"] == str(tmp_path / "AoE2DE")
        assert rescans == [True]
        assert not dialog.profile_auto_button.isEnabled()
        assert dialog.profile_path_edit.text() == ""
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_missing_config_folder_is_reported_and_can_be_cleared(tmp_path) -> None:
    from descape import asset_source

    missing = tmp_path / "gone"
    asset_source.save_profile_path_to_config(missing)
    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.profile_path_edit.text() == str(missing)
        assert dialog.profile_status_label.text() == (
            f"config.yaml points at {missing}, which is not a folder, so no profile is scanned."
        )
        assert dialog._profile_status_ok is False
        assert dialog.profile_auto_button.isEnabled()
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.parametrize("valid", [True, False], ids=["folder", "not-a-folder"])
def test_the_env_var_disables_the_row(tmp_path, monkeypatch, valid: bool) -> None:
    from descape import content_roots

    target = _profile_tree(tmp_path) if valid else tmp_path / "missing"
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, str(target))
    dialog, window = conftest.dialog_and_window()
    try:
        for widget in (dialog.profile_path_edit, dialog.profile_browse_button,
                       dialog.profile_load_button, dialog.profile_auto_button):
            assert not widget.isEnabled()
        assert dialog.profile_path_edit.text() == str(target)
        assert content_roots.PROFILE_ENV_VAR in dialog.profile_status_label.text()
        assert dialog._profile_status_ok is valid
        assert len(_list_texts(dialog)) == (3 if valid else 1)
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows auto-detects %USERPROFILE%\\Games, never from the install's Steam library"
)
def test_an_install_load_refreshes_the_auto_detected_profile(tmp_path, monkeypatch) -> None:
    """Auto-detect derives the profile from the install's Steam library."""
    from descape import asset_source, content_roots

    steamapps = tmp_path / "steamapps"
    install = steamapps / "common" / "AoE2DE"
    install.mkdir(parents=True)
    profile = steamapps / content_roots._PROTON_PROFILE_SUBPATH
    (profile / "0").mkdir(parents=True)
    monkeypatch.setattr(asset_source, "_override_path", None)  # restored after the test
    monkeypatch.setattr(asset_source, "validate_install_path", lambda path: (True, "ok"))
    monkeypatch.setattr(asset_source, "save_install_path_to_config", lambda path: None)
    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.profile_path_edit.text() == ""
        dialog.install_path_edit.setText(str(install))
        dialog.load_install_path()
        assert dialog.profile_path_edit.text() == str(profile)
        assert dialog.profile_status_label.text() == "Auto-detected from the install."
        assert _list_texts(dialog) == [f"Install: {install}", f"Profile 0: {profile / '0'}"]
    finally:
        dialog.close()
        conftest.close_window(window)
        asset_source.clear_install_caches()


def test_the_mods_tab_holds_the_profile_and_mod_art_sections(opened) -> None:
    """GH #244: the profile folder section and the Mod art list sit on their own Mods
    tab, right after General; zoom mode and the install path stay on General."""
    from PyQt5.QtWidgets import QTabWidget

    dialog, _window, _rescans = opened
    tabs = dialog.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "General", "Mods", "Appearance", "Tools", "Saving", "Keybinds", "Advanced",
    ]
    general, mods = _select_tab(dialog, "General"), _select_tab(dialog, "Mods")
    for widget in (dialog.profile_path_edit, dialog.profile_roots_list, dialog.mod_art_list):
        assert mods.isAncestorOf(widget) and not general.isAncestorOf(widget)
    for widget in (dialog.install_path_edit, dialog.zoom_cursor_radio):
        assert general.isAncestorOf(widget) and not mods.isAncestorOf(widget)


@pytest.mark.parametrize(("title", "label"), [
    ("General", "install_status_label"), ("Mods", "profile_status_label"),
])
def test_a_scrolled_tab_still_paints_the_tab_page(opened, title: str, label: str) -> None:
    """Its status labels are shaded for the tab page (status_color(on_tab_page=True));
    a plain scroll area would paint the window colour under them instead."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import painted_tab_page

    dialog, _window, _rescans = opened
    _select_tab(dialog, title)
    dialog.show()
    for _ in range(4):
        QApplication.processEvents()
    image = dialog.grab().toImage()
    page = getattr(dialog, label).parentWidget()
    corner = page.mapTo(dialog, QPoint(2, 2))
    assert image.pixelColor(corner).name() == painted_tab_page(QApplication.palette())


def test_a_theme_change_restyles_the_profile_status_line() -> None:
    """Light to parchment, as test_themes' install-label twin: their status colours differ."""
    from descape import settings
    from descape.viewer import status_color
    from descape.viewer_dialogs import apply_theme

    with conftest.restoring_app_theme() as app:
        apply_theme(app, "light", {})
        dialog, window = conftest.dialog_and_window()
        try:
            ok = dialog._profile_status_ok
            before = dialog.profile_status_label.styleSheet()
            assert before == f"color: {status_color(ok, on_tab_page=True)};"
            settings.set_theme("editor_parchment_red")
            apply_theme(app, "editor_parchment_red", {})
            after = dialog.profile_status_label.styleSheet()
            assert after != before
            assert after == f"color: {status_color(ok, on_tab_page=True)};"
        finally:
            dialog.close()
            conftest.close_window(window)


def test_long_paths_scroll_inside_the_list_not_the_dialog(opened, tmp_path) -> None:
    """The list keeps a capped height and a long path never widens the dialog."""
    from PyQt5.QtWidgets import QApplication

    dialog, _window, _rescans = opened
    width_before = dialog.minimumSizeHint().width()
    deep = tmp_path.joinpath(*(["a_rather_long_folder_name"] * 12))
    profile = _profile_tree(deep)
    dialog.profile_path_edit.setText(str(profile))
    dialog.load_profile_path()
    _select_tab(dialog, "Mods")
    dialog.show()
    for _ in range(4):
        QApplication.processEvents()
    view = dialog.profile_roots_list
    assert dialog.minimumSizeHint().width() == width_before
    assert view.horizontalScrollBar().maximum() > 0
    assert view.height() == view.maximumHeight()
    assert view.height() >= dialog._PROFILE_LIST_ROWS * view.sizeHintForRow(0)
