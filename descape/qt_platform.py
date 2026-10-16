"""Picks Qt's platform plugin and platform theme before the QApplication exists.

Opt-in fallback (GH #78, GH #179): Qt5's Wayland plugin delivers every
mouse-move event uncoalesced. MapView now coalesces moves itself, so native
Wayland is the default; Settings > General > "X11 compatibility on Wayland"
runs a Linux Wayland session that also has XWayland under xcb instead. With
the box ticked, an inherited `wayland*` value is overridden (a desktop session
may export one); any other user-set `QT_QPA_PLATFORM` (even an empty one)
always wins.

Desktop file dialogs (GH #220): the PyQt5 wheel ships no KDE or GTK theme
plugin, so on Linux Qt's own file dialog is used. Its one theme plugin,
`xdgdesktopportal`, routes QFileDialog through xdg-desktop-portal instead.
Qt 5.15 has no fallback when the portal is missing (the dialog never opens,
measured), so the theme is set only when a D-Bus query finds the portal's
FileChooser, and never over a user-set `QT_QPA_PLATFORMTHEME` (even an empty
one, which is the opt-out).

No PyQt import here: this must run before Qt picks a plugin, and stays a pure
function of the environment so tests need no display. The setting is read by
the caller and passed in; the portal query is injectable for the same reason.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping

FALLBACK_PLATFORM = "xcb"
PORTAL_THEME = "xdgdesktopportal"

_PORTAL_DEST = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_FILE_CHOOSER = "org.freedesktop.portal.FileChooser"
_PORTAL_QUERIES = (
    (
        "gdbus",
        ["call", "--session", "--timeout", "2", "--dest", _PORTAL_DEST, "--object-path", _PORTAL_PATH,
         "--method", "org.freedesktop.DBus.Properties.Get", _FILE_CHOOSER, "version"],
    ),
    ("busctl", ["--user", "--timeout=2", "get-property", _PORTAL_DEST, _PORTAL_PATH, _FILE_CHOOSER, "version"]),
)
# Above the D-Bus timeout, so a hung tool cannot stall startup past it.
_QUERY_TIMEOUT_S = 3


def default_platform(environ: Mapping[str, str], platform: str, prefer_xcb: bool) -> str | None:
    """The `QT_QPA_PLATFORM` value to set, or None to leave Qt's own choice."""
    if not prefer_xcb or not platform.startswith("linux"):
        return None
    inherited = environ.get("QT_QPA_PLATFORM")
    if inherited is not None and not inherited.startswith("wayland"):
        return None
    if not environ.get("WAYLAND_DISPLAY") or not environ.get("DISPLAY"):
        return None
    return FALLBACK_PLATFORM


def inherited_line(environ: Mapping[str, str] | None = None, platform: str | None = None) -> str | None:
    """The debug-log line naming the inherited `QT_QPA_PLATFORM` on Linux,
    or None elsewhere."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        return None
    value = environ.get("QT_QPA_PLATFORM")
    return f"Inherited QT_QPA_PLATFORM: {'unset' if value is None else repr(value)}"


def apply_default(
    environ: MutableMapping[str, str] | None = None,
    platform: str | None = None,
    *,
    prefer_xcb: bool = False,
) -> str | None:
    """Sets the fallback in `environ` (os.environ by default) and returns the
    debug-log line saying so, or None when it left the environment alone."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    choice = default_platform(environ, platform, prefer_xcb)
    if choice is None:
        return None
    environ["QT_QPA_PLATFORM"] = choice
    return f"Wayland session: using QT_QPA_PLATFORM={choice} (XWayland), the X11 compatibility setting is on."


def portal_file_chooser_available(
    environ: Mapping[str, str],
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    frozen: bool = bool(getattr(sys, "frozen", False)),
) -> bool:
    """Whether the session bus has xdg-desktop-portal with a FileChooser,
    asked with the first of gdbus or busctl found. Reading the property also
    starts a D-Bus-activatable portal that is not running yet."""
    env = dict(environ)
    if frozen:
        # PyInstaller points LD_LIBRARY_PATH at the bundle; a system tool needs the original.
        original = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is None:
            env.pop("LD_LIBRARY_PATH", None)
        else:
            env["LD_LIBRARY_PATH"] = original
    for tool, args in _PORTAL_QUERIES:
        exe = which(tool)
        if exe is None:
            continue
        try:
            result = run(
                [exe, *args],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_QUERY_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0
    return False


def default_platform_theme(environ: Mapping[str, str], platform: str, portal_check: Callable[[], bool]) -> str | None:
    """The `QT_QPA_PLATFORMTHEME` value to set, or None to leave Qt's own
    choice. `portal_check` runs only on Linux with the variable unset."""
    if not platform.startswith("linux") or "QT_QPA_PLATFORMTHEME" in environ:
        return None
    return PORTAL_THEME if portal_check() else None


def apply_default_theme(
    environ: MutableMapping[str, str] | None = None,
    platform: str | None = None,
    portal_check: Callable[[], bool] | None = None,
) -> str | None:
    """Sets the portal theme in `environ` (os.environ by default) when
    `default_platform_theme()` picks it, and returns the debug-log line for
    what happened on Linux, or None elsewhere."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        return None
    inherited = environ.get("QT_QPA_PLATFORMTHEME")
    if inherited is not None:
        return f"Inherited QT_QPA_PLATFORMTHEME: {inherited!r}, left as set"
    if portal_check is None:

        def portal_check() -> bool:
            return portal_file_chooser_available(environ)

    choice = default_platform_theme(environ, platform, portal_check)
    if choice is None:
        return "Linux: no xdg-desktop-portal FileChooser, keeping Qt's file dialog"
    environ["QT_QPA_PLATFORMTHEME"] = choice
    return f"Linux: using QT_QPA_PLATFORMTHEME={choice} (desktop file dialogs)"
