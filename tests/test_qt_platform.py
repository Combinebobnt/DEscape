"""descape/qt_platform.py: the opt-in X11 compatibility fallback that runs a
Linux Wayland session under xcb. Qt-free, so it runs in the default tier with
no display.
"""

from __future__ import annotations

import pytest

from descape.qt_platform import apply_default, default_platform, inherited_line

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
