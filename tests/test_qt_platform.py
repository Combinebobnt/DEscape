"""descape/qt_platform.py: the GH #78 stopgap that defaults a Linux Wayland
session to xcb. Qt-free, so it runs in the default tier with no display.
"""

from __future__ import annotations

import pytest

from descape.qt_platform import apply_default, default_platform

WAYLAND_WITH_XWAYLAND = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}


def test_linux_wayland_with_xwayland_defaults_to_xcb():
    assert default_platform(WAYLAND_WITH_XWAYLAND, "linux") == "xcb"


@pytest.mark.parametrize("value", ["wayland", "xcb", "offscreen", ""])
def test_user_set_value_is_never_overridden(value):
    env = {**WAYLAND_WITH_XWAYLAND, "QT_QPA_PLATFORM": value}
    assert default_platform(env, "linux") is None


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
    assert default_platform(env, "linux") is None


@pytest.mark.parametrize("platform", ["win32", "darwin", "freebsd14"])
def test_non_linux_leaves_qt_alone(platform):
    assert default_platform(WAYLAND_WITH_XWAYLAND, platform) is None


def test_apply_sets_env_and_returns_log_line_with_override_hint():
    env = dict(WAYLAND_WITH_XWAYLAND)
    line = apply_default(env, "linux")
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert line is not None
    assert "QT_QPA_PLATFORM=wayland" in line


def test_apply_leaves_env_untouched_when_not_applicable():
    env = {**WAYLAND_WITH_XWAYLAND, "QT_QPA_PLATFORM": "wayland"}
    assert apply_default(env, "linux") is None
    assert env["QT_QPA_PLATFORM"] == "wayland"
    env = {"DISPLAY": ":0"}
    assert apply_default(env, "linux") is None
    assert "QT_QPA_PLATFORM" not in env
