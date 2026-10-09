"""Picks Qt's platform plugin before the QApplication exists.

Opt-in fallback (GH #78, GH #179): Qt5's Wayland plugin delivers every
mouse-move event uncoalesced. MapView now coalesces moves itself, so native
Wayland is the default; Settings > General > "X11 compatibility on Wayland"
runs a Linux Wayland session that also has XWayland under xcb instead. With
the box ticked, an inherited `wayland*` value is overridden (a desktop session
may export one); any other user-set `QT_QPA_PLATFORM` (even an empty one)
always wins.

No PyQt import here: this must run before Qt picks a plugin, and stays a pure
function of the environment so tests need no display. The setting is read by
the caller and passed in.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping

FALLBACK_PLATFORM = "xcb"


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
