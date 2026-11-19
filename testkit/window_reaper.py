"""Deletes the ViewerWindows a finished test left alive.

A window a test built and never closed outlives it: every referrer is inside
its own graph (child panels' bound-method callbacks, MapView, ...). Either a
cycle through a child widget's C++ parent link, invisible to Python's
collector, or `_freeze_loaded_document()`'s `gc.freeze()` keeps it; which one
dominates was not established. Left alone, a worker reached 45-52k top-level
widgets, each live window re-inked its icons on every palette change, and
per-test cost grew about 10x with a worker's age. tests/conftest.py's
pytest_runtest_teardown calls reap() after every test's own finalizers.

Deletion only, never close(): by teardown _isolated_settings is undone, and
closeEvent() writes the window size to whatever CONFIG_PATH is then, which
would be the developer's real config.yaml.

A window held by a live module-, class- or package-scoped fixture survives,
whether or not the finished test requested it: protected_ids() collects those
fixtures' values (tuple elements and a `.window` attribute included). A
session-scoped fixture is not seen (pytest runs its setup hooks without
tests/conftest.py), and neither is a window held in another shape: register
those with keep_window().
"""

from __future__ import annotations

import sys

# Strong references: a kept window lives for the session anyway, and an id() could be reused.
_KEPT: list = []
# Fed by tests/conftest.py's fixture hooks: a sibling test that never requests a module fixture must spare it too.
_LIVE_WIDE: set = set()


def keep_window(window) -> None:
    """Exempts `window` from reaping for the rest of the session."""
    _KEPT.append(window)


def live_viewer_windows() -> list:
    qw = sys.modules.get("PyQt5.QtWidgets")
    viewer = sys.modules.get("descape.viewer")
    if qw is None or viewer is None or qw.QApplication.instance() is None:
        return []
    return [w for w in qw.QApplication.topLevelWidgets() if isinstance(w, viewer.ViewerWindow)]


def fixture_set_up(fixturedef) -> None:
    if fixturedef.scope != "function":
        _LIVE_WIDE.add(fixturedef)


def fixture_finalized(fixturedef) -> None:
    _LIVE_WIDE.discard(fixturedef)


def protected_ids() -> set[int]:
    """ids of keep_window()'s windows and of every value a live wider-than-function fixture holds."""
    ids = {id(w) for w in _KEPT}
    for fixturedef in list(_LIVE_WIDE):
        cached = getattr(fixturedef, "cached_result", None)
        if cached is None:
            continue
        value = cached[0]
        for v in value if isinstance(value, (tuple, list)) else (value,):
            ids.add(id(v))
            # The instance dict only: a raising property must not break every later test's teardown.
            window = getattr(v, "__dict__", {}).get("window")
            if window is not None and not callable(window):
                ids.add(id(window))
    return ids


def reap(protected: set[int]) -> int:
    """deleteLater()s every live ViewerWindow not in `protected` and flushes the
    deferred deletes. Returns how many it deleted."""
    doomed = [w for w in live_viewer_windows() if id(w) not in protected]
    if not doomed:
        return 0
    from PyQt5.QtCore import QCoreApplication, QEvent

    for window in doomed:
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    return len(doomed)
