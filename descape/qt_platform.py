"""Picks Qt's platform plugin before the QApplication exists.

Stopgap for GH #78: Qt's Wayland plugin delivers every mouse-move event
uncoalesced, so a drag queues far more repaint work than under xcb, which
merges queued moves. Until MapView coalesces moves itself, a Linux Wayland
session that also has XWayland runs under xcb by default. A user-set
`QT_QPA_PLATFORM` (even an empty one) always wins.

No PyQt import here: this must run before Qt picks a plugin, and stays a pure
function of the environment so tests need no display.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping

STOPGAP_PLATFORM = "xcb"


def default_platform(environ: Mapping[str, str], platform: str) -> str | None:
    """The `QT_QPA_PLATFORM` value to set, or None to leave Qt's own choice."""
    if not platform.startswith("linux"):
        return None
    if "QT_QPA_PLATFORM" in environ:
        return None
    if not environ.get("WAYLAND_DISPLAY") or not environ.get("DISPLAY"):
        return None
    return STOPGAP_PLATFORM


def apply_default(environ: MutableMapping[str, str] | None = None, platform: str | None = None) -> str | None:
    """Sets the default in `environ` (os.environ by default) and returns the
    debug-log line saying so, or None when it left the environment alone."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    choice = default_platform(environ, platform)
    if choice is None:
        return None
    environ["QT_QPA_PLATFORM"] = choice
    return (
        f"Wayland session: using QT_QPA_PLATFORM={choice} (XWayland) by default, "
        "since Wayland drags repaint every mouse move (GH #78). "
        "Launch with QT_QPA_PLATFORM=wayland to override."
    )
