"""descape/qt_platform.py: the opt-in X11 compatibility fallback that runs a
Linux Wayland session under xcb, and the portal-gated desktop file dialogs
(GH #220). Qt-free, so it runs in the default tier with no display.
"""

from __future__ import annotations

import subprocess

import pytest

from descape.qt_platform import (
    PORTAL_THEME,
    apply_default,
    apply_default_theme,
    default_platform,
    default_platform_theme,
    inherited_line,
    portal_file_chooser_available,
)

WAYLAND_WITH_XWAYLAND = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}


def test_native_wayland_is_the_default_when_the_box_is_off():
    assert default_platform(WAYLAND_WITH_XWAYLAND, "linux", False) is None


def test_linux_wayland_with_xwayland_uses_xcb_when_the_box_is_on():
    assert default_platform(WAYLAND_WITH_XWAYLAND, "linux", True) == "xcb"


@pytest.mark.parametrize("value", ["wayland", "wayland-egl"])
def test_the_box_beats_an_inherited_wayland_value(value):
    env = {**WAYLAND_WITH_XWAYLAND, "QT_QPA_PLATFORM": value}
    assert default_platform(env, "linux", True) == "xcb"
    assert default_platform(env, "linux", False) is None


@pytest.mark.parametrize("value", ["xcb", "offscreen", ""])
def test_any_other_user_set_value_is_never_overridden(value):
    env = {**WAYLAND_WITH_XWAYLAND, "QT_QPA_PLATFORM": value}
    assert default_platform(env, "linux", True) is None


@pytest.mark.parametrize(
    "env",
    [
        {"WAYLAND_DISPLAY": "wayland-0"},
        {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ""},
        {"DISPLAY": ":0"},
        {"WAYLAND_DISPLAY": "", "DISPLAY": ":0"},
        {},
    ],
    ids=["no-xwayland", "empty-display", "plain-x11", "empty-wayland", "no-display"],
)
def test_missing_wayland_or_xwayland_leaves_qt_alone(env):
    assert default_platform(env, "linux", True) is None


@pytest.mark.parametrize("platform", ["win32", "darwin", "freebsd14"])
def test_non_linux_leaves_qt_alone(platform):
    assert default_platform(WAYLAND_WITH_XWAYLAND, platform, True) is None


def test_apply_sets_env_and_returns_log_line_naming_the_setting():
    env = dict(WAYLAND_WITH_XWAYLAND)
    line = apply_default(env, "linux", prefer_xcb=True)
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert line == (
        "Wayland session: using QT_QPA_PLATFORM=xcb (XWayland), the X11 compatibility setting is on."
    )


def test_apply_leaves_env_untouched_when_not_applicable():
    env = dict(WAYLAND_WITH_XWAYLAND)
    assert apply_default(env, "linux") is None
    assert "QT_QPA_PLATFORM" not in env
    env = {**WAYLAND_WITH_XWAYLAND, "QT_QPA_PLATFORM": "offscreen"}
    assert apply_default(env, "linux", prefer_xcb=True) is None
    assert env["QT_QPA_PLATFORM"] == "offscreen"
    env = {"DISPLAY": ":0"}
    assert apply_default(env, "linux", prefer_xcb=True) is None
    assert "QT_QPA_PLATFORM" not in env


def test_inherited_line_names_the_value_or_unset_on_linux_only():
    assert inherited_line({}, "linux") == "Inherited QT_QPA_PLATFORM: unset"
    assert inherited_line({"QT_QPA_PLATFORM": "wayland"}, "linux") == "Inherited QT_QPA_PLATFORM: 'wayland'"
    assert inherited_line({"QT_QPA_PLATFORM": ""}, "linux") == "Inherited QT_QPA_PLATFORM: ''"
    assert inherited_line({}, "win32") is None


def _never():
    raise AssertionError("portal check must not run")


@pytest.mark.parametrize("env", [{}, WAYLAND_WITH_XWAYLAND, {"DISPLAY": ":0"}], ids=["bare", "wayland", "x11"])
def test_linux_with_a_portal_uses_the_portal_theme(env):
    assert default_platform_theme(env, "linux", lambda: True) == PORTAL_THEME == "xdgdesktopportal"


def test_linux_without_a_portal_leaves_the_theme_alone():
    assert default_platform_theme(WAYLAND_WITH_XWAYLAND, "linux", lambda: False) is None


@pytest.mark.parametrize("value", ["kde", "gtk3", ""])
def test_a_user_set_theme_wins_without_checking_the_portal(value):
    assert default_platform_theme({"QT_QPA_PLATFORMTHEME": value}, "linux", _never) is None


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_never_checks_the_portal(platform):
    assert default_platform_theme({}, platform, _never) is None
    assert apply_default_theme({}, platform, _never) is None


class _Runner:
    """Records each call and answers with a fixed exit code or exception."""

    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return subprocess.CompletedProcess(argv, self.result)


def _which(*present):
    return lambda tool: f"/usr/bin/{tool}" if tool in present else None


@pytest.mark.parametrize(("code", "expected"), [(0, True), (1, False)])
def test_gdbus_answers_and_busctl_is_not_tried(code, expected):
    run = _Runner(code)
    assert portal_file_chooser_available({}, which=_which("gdbus", "busctl"), run=run, frozen=False) is expected
    assert len(run.calls) == 1
    argv, kwargs = run.calls[0]
    assert argv[:3] == ["/usr/bin/gdbus", "call", "--session"]
    assert argv[-2:] == ["org.freedesktop.portal.FileChooser", "version"]
    assert kwargs["timeout"] > 2


def test_busctl_is_used_when_gdbus_is_missing():
    run = _Runner(0)
    assert portal_file_chooser_available({}, which=_which("busctl"), run=run, frozen=False)
    assert run.calls[0][0][:3] == ["/usr/bin/busctl", "--user", "--timeout=2"]


def test_no_dbus_tool_means_no_portal():
    run = _Runner(0)
    assert not portal_file_chooser_available({}, which=_which(), run=run, frozen=False)
    assert run.calls == []


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("gdbus", 3), OSError("exec failed")])
def test_a_failed_query_means_no_portal(error):
    assert not portal_file_chooser_available({}, which=_which("gdbus"), run=_Runner(error), frozen=False)


def test_a_frozen_build_gives_the_tool_the_original_library_path():
    env = {"LD_LIBRARY_PATH": "/bundle", "LD_LIBRARY_PATH_ORIG": "/usr/local/lib", "PATH": "/usr/bin"}
    run = _Runner(0)
    portal_file_chooser_available(env, which=_which("gdbus"), run=run, frozen=True)
    child = run.calls[0][1]["env"]
    assert child["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert "LD_LIBRARY_PATH_ORIG" not in child
    assert child["PATH"] == "/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/bundle"


def test_a_frozen_build_without_an_original_drops_the_library_path():
    run = _Runner(0)
    portal_file_chooser_available({"LD_LIBRARY_PATH": "/bundle"}, which=_which("gdbus"), run=run, frozen=True)
    assert "LD_LIBRARY_PATH" not in run.calls[0][1]["env"]


def test_an_unfrozen_run_passes_the_library_path_through():
    run = _Runner(0)
    portal_file_chooser_available({"LD_LIBRARY_PATH": "/mine"}, which=_which("gdbus"), run=run, frozen=False)
    assert run.calls[0][1]["env"]["LD_LIBRARY_PATH"] == "/mine"


def test_apply_theme_sets_env_and_returns_log_line():
    env = dict(WAYLAND_WITH_XWAYLAND)
    line = apply_default_theme(env, "linux", lambda: True)
    assert env["QT_QPA_PLATFORMTHEME"] == "xdgdesktopportal"
    assert line == "Linux: using QT_QPA_PLATFORMTHEME=xdgdesktopportal (desktop file dialogs)"


def test_apply_theme_without_a_portal_leaves_env_untouched():
    env = dict(WAYLAND_WITH_XWAYLAND)
    line = apply_default_theme(env, "linux", lambda: False)
    assert "QT_QPA_PLATFORMTHEME" not in env
    assert line == "Linux: no xdg-desktop-portal FileChooser, keeping Qt's file dialog"


@pytest.mark.parametrize("value", ["kde", ""])
def test_apply_theme_reports_a_user_value_and_keeps_it(value):
    env = {"QT_QPA_PLATFORMTHEME": value}
    assert apply_default_theme(env, "linux", _never) == f"Inherited QT_QPA_PLATFORMTHEME: {value!r}, left as set"
    assert env["QT_QPA_PLATFORMTHEME"] == value
