"""descape/qt_platform.py: the opt-in X11 compatibility fallback that runs a
Linux Wayland session under xcb, and the portal-gated desktop file dialogs
(GH #220). Qt-free, so it runs in the default tier with no display, apart
from one offscreen child process that pins how QApplication reads its argv.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from descape.qt_platform import (
    PORTAL_THEME,
    apply_default,
    apply_default_theme,
    default_platform,
    inherited_line,
    portal_file_chooser_problem,
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


ARGV = ["descape", "map.aoe2scenario"]
USING = "Linux: using platform theme xdgdesktopportal (desktop file dialogs)"


@pytest.mark.parametrize("env", [{}, WAYLAND_WITH_XWAYLAND, {"DISPLAY": ":0"}], ids=["bare", "wayland", "x11"])
def test_linux_with_a_portal_passes_the_portal_theme_to_qt(env):
    argv = list(ARGV)
    assert apply_default_theme(argv, dict(env), "linux", lambda: None) == USING
    assert argv == ["descape", "-platformtheme", PORTAL_THEME, "map.aoe2scenario"]
    assert PORTAL_THEME == "xdgdesktopportal"


def test_the_theme_stays_out_of_the_environment_children_inherit():
    """GH #220 review: a program DEscape starts (Open containing folder) keeps its own theme."""
    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    apply_default_theme(list(ARGV), env, "linux", lambda: None)
    assert env == {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}


def test_linux_without_a_portal_leaves_argv_alone_and_says_why():
    argv = list(ARGV)
    line = apply_default_theme(argv, WAYLAND_WITH_XWAYLAND, "linux", lambda: "gdbus timed out after 3 s")
    assert argv == ARGV
    assert line == "Linux: no usable xdg-desktop-portal FileChooser (gdbus timed out after 3 s), keeping Qt's file dialog"


@pytest.mark.parametrize("value", ["kde", "gtk3", ""])
def test_a_user_set_theme_wins_without_checking_the_portal(value):
    argv = list(ARGV)
    env = {"QT_QPA_PLATFORMTHEME": value}
    assert apply_default_theme(argv, env, "linux", _never) == f"Inherited QT_QPA_PLATFORMTHEME: {value!r}, left as set"
    assert argv == ARGV
    assert env == {"QT_QPA_PLATFORMTHEME": value}


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_never_checks_the_portal(platform):
    argv = list(ARGV)
    assert apply_default_theme(argv, {}, platform, _never) is None
    assert argv == ARGV


def test_qt_takes_the_theme_option_back_out_of_argv():
    """viewer.main() hands sys.argv itself to QApplication: the option must not
    be left in front of the scenario path. A child process, since this one may
    already hold a QApplication."""
    pytest.importorskip("PyQt5")
    script = (
        "from PyQt5.QtWidgets import QApplication\n"
        "argv = ['descape', '-platformtheme', 'xdgdesktopportal', 'map.aoe2scenario']\n"
        "app = QApplication(argv)\n"
        "print(argv)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORMTHEME"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=True
    ).stdout
    assert out.strip().splitlines()[-1] == "['descape', 'map.aoe2scenario']"


GDBUS_V3 = "(<uint32 3>,)\n"
BUSCTL_V3 = "u 3\n"


class _Runner:
    """Records each call and answers with a fixed exit code and stdout, or an exception."""

    def __init__(self, result=0, stdout=GDBUS_V3):
        self.result = result
        self.stdout = stdout
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return subprocess.CompletedProcess(argv, self.result, stdout=self.stdout)


def _which(*present):
    return lambda tool: f"/usr/bin/{tool}" if tool in present else None


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0, None), (1, "no portal FileChooser on the session bus (gdbus exit code 1)")],
)
def test_gdbus_answers_and_busctl_is_not_tried(code, expected):
    run = _Runner(code)
    assert portal_file_chooser_problem({}, which=_which("gdbus", "busctl"), run=run, frozen=False) == expected
    assert len(run.calls) == 1
    argv, kwargs = run.calls[0]
    assert argv[:3] == ["/usr/bin/gdbus", "call", "--session"]
    assert argv[argv.index("--timeout") + 1] == "2"
    assert argv[-2:] == ["org.freedesktop.portal.FileChooser", "version"]
    assert kwargs["timeout"] > 2


def test_busctl_is_used_when_gdbus_is_missing():
    run = _Runner(0, BUSCTL_V3)
    assert portal_file_chooser_problem({}, which=_which("busctl"), run=run, frozen=False) is None
    assert run.calls[0][0][:3] == ["/usr/bin/busctl", "--user", "--timeout=2"]


@pytest.mark.parametrize(
    ("tool", "stdout", "expected"),
    [
        ("gdbus", "(<uint32 2>,)\n", "FileChooser version 2, folder pickers need 3+"),
        ("gdbus", "(<uint32 4>,)\n", None),
        ("busctl", "u 2\n", "FileChooser version 2, folder pickers need 3+"),
        ("busctl", "u 4\n", None),
        ("gdbus", "", "unreadable gdbus reply ''"),
        ("busctl", "garbage\n", "unreadable busctl reply 'garbage'"),
    ],
)
def test_a_file_chooser_older_than_version_3_means_no_portal(tool, stdout, expected):
    """Folder pickers need FileChooser version 3; an older portal keeps Qt's dialog."""
    run = _Runner(0, stdout)
    assert portal_file_chooser_problem({}, which=_which(tool), run=run, frozen=False) == expected


def test_apply_theme_asks_the_real_portal_check_when_none_is_injected(monkeypatch):
    from descape import qt_platform

    seen = []
    monkeypatch.setattr(qt_platform, "portal_file_chooser_problem", seen.append)
    env = {}
    assert apply_default_theme(list(ARGV), env, "linux") == USING
    assert seen == [env]


def test_no_dbus_tool_means_no_portal():
    run = _Runner(0)
    assert portal_file_chooser_problem({}, which=_which(), run=run, frozen=False) == "neither gdbus nor busctl is installed"
    assert run.calls == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (subprocess.TimeoutExpired("gdbus", 3), "gdbus timed out after 3 s"),
        (OSError("exec failed"), "gdbus failed to run: exec failed"),
    ],
)
def test_a_failed_query_means_no_portal(error, expected):
    assert portal_file_chooser_problem({}, which=_which("gdbus"), run=_Runner(error), frozen=False) == expected


def test_each_no_portal_reason_reads_differently_in_the_log():
    """GH #220 review: one log line used to cover every reason."""
    cases = [
        (_which(), _Runner(0)),
        (_which("gdbus"), _Runner(subprocess.TimeoutExpired("gdbus", 3))),
        (_which("gdbus"), _Runner(OSError("exec failed"))),
        (_which("gdbus"), _Runner(1)),
        (_which("gdbus"), _Runner(0, "(<uint32 2>,)\n")),
        (_which("gdbus"), _Runner(0, "garbage\n")),
    ]
    lines = set()
    for which, run in cases:
        env = {}
        problem = portal_file_chooser_problem(env, which=which, run=run, frozen=False)
        assert problem is not None
        lines.add(apply_default_theme(list(ARGV), env, "linux", lambda problem=problem: problem))
    assert len(lines) == len(cases)


def test_a_frozen_build_gives_the_tool_the_original_library_path():
    env = {"LD_LIBRARY_PATH": "/bundle", "LD_LIBRARY_PATH_ORIG": "/usr/local/lib", "PATH": "/usr/bin"}
    run = _Runner(0)
    portal_file_chooser_problem(env, which=_which("gdbus"), run=run, frozen=True)
    child = run.calls[0][1]["env"]
    assert child["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert "LD_LIBRARY_PATH_ORIG" not in child
    assert child["PATH"] == "/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/bundle"


def test_a_frozen_build_without_an_original_drops_the_library_path():
    run = _Runner(0)
    portal_file_chooser_problem({"LD_LIBRARY_PATH": "/bundle"}, which=_which("gdbus"), run=run, frozen=True)
    assert "LD_LIBRARY_PATH" not in run.calls[0][1]["env"]


def test_an_unfrozen_run_passes_the_library_path_through():
    run = _Runner(0)
    portal_file_chooser_problem({"LD_LIBRARY_PATH": "/mine"}, which=_which("gdbus"), run=run, frozen=False)
    assert run.calls[0][1]["env"]["LD_LIBRARY_PATH"] == "/mine"
