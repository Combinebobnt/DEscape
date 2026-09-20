"""One QApplication bootstrap and one stepped-window builder, shared.

Two copies of the Qt bootstrap and four of the stepped-window builder each
carried their own forks of the same setup (tests/conftest.py,
tests/test_lazy_viewport.py, tests/test_mip_viewer.py,
tools/gen_seam_eyeball.py). Unlike qt_capture.py's five capture bodies these
were not byte-identical, so the merge needed a contract first.

Three of the differences are real and became parameters: `show` (one caller
measures a paint cycle it triggers itself and must not be shown), `size`, and
load-failure policy (skip in tests, exit in a tool).

Three were measured to be no-ops and dropped. `terrain_style_combo.
setCurrentText("Stepped")` is a same-text set on a combo _build_toolbar
already populated before connect(), and on_terrain_style_changed() early-
returns on a matching style, so it emitted nothing either way. One
processEvents() versus two unified upward on two: going 2 to 1 risks
flakiness a green run would not catch, going 1 to 2 only settles more.
QApplication(["pytest"]) versus QApplication(sys.argv[:1]) unified on a fixed
literal, so Qt can never consume a flag out of a real command line.

Three empirical facts the deleted copies carried, load-bearing here:

An unassigned `QApplication(...)` is collected by PyQt5, and the next
`ViewerWindow()` then aborts the interpreter with a Fatal Python error
rather than raising. That is why _QAPP below is a module global and not a
local.

`show()` must be called on the TOP-LEVEL window, never on `map_view`
directly. Showing a child widget alone never triggers a real paint cycle,
offscreen platform or not (confirmed empirically by
tests/test_lazy_viewport.py's own `_show_and_settle`).

Settings are pinned by assigning the settings MODULE GLOBALS, never through
`settings.set_elev_step_pct()` / `set_graphics_quality()`. Outside pytest
there is no `_isolated_settings` autouse fixture redirecting CONFIG_PATH, so
the real setters write straight through to the developer's own config.yaml.
That happened once already, to an ad-hoc debug snippet.

pytest is deliberately not imported at module level (qt_capture.py's
precedent: every third-party import here is lazy, inside a function), so a
refused scenario surfaces as ScenarioLoadError and each caller translates it
into `pytest.skip` or `SystemExit` itself.
"""

from __future__ import annotations

import importlib.util
import os

PYQT5_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_QAPP = None


class ScenarioLoadError(RuntimeError):
    """A scenario path that ViewerWindow.load_scenario() refused."""


def ensure_qapp() -> None:
    """Construct the process's one QApplication, once, before any
    ViewerWindow(). See the module docstring for why the reference is held
    in a global.

    QT_QPA_PLATFORM is set here rather than relying on any verify_*.py
    module-level side effect, since this runs before those are imported and
    QApplication() aborts immediately if it reaches for a display that does
    not exist. The argv is a fixed literal rather than `sys.argv[:1]`: Qt
    reads argv only for its own -style/-platform flags, the platform is set
    above anyway, and a literal stops Qt consuming a flag out of a real
    pytest or tool command line.
    """
    global _QAPP
    if not PYQT5_AVAILABLE or _QAPP is not None:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(["descape"])


def show_and_settle(window, size: int | None = None) -> None:
    """Resize (if `size` is given), show the top-level window, and pump the
    event loop until the initial fitInView-on-open paint has happened.

    Pumps TWICE. The four merged copies split 2/2 on this and the plan
    unified upward: going 2 to 1 is a real risk that surfaces as flakiness
    rather than a deterministic failure, so a green run would not clear it,
    while going 1 to 2 only settles more.
    """
    from PyQt5.QtWidgets import QApplication

    if size is not None:
        window.resize(size, size)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()


def stepped_window(
    path,
    *,
    elev_step_pct: int | None = None,
    graphics_quality: int | None = None,
    show: bool = True,
    size: int | None = None,
):
    """A real ViewerWindow with `path` loaded in Stepped mode, offscreen.

    elev_step_pct/graphics_quality, if given, are pinned via the settings
    module globals before ViewerWindow() is constructed (__init__ reads
    settings.get_window_size(), and _render_current() reads both of these).
    See the module docstring for why not the real setters.

    Raises ScenarioLoadError, after closing the window, if load_scenario()
    left `window.scenario` None.

    `show=False` returns the window unshown and unpumped, for a caller that
    must measure a paint cycle it triggers itself.

    Does NOT call terrain_style_combo.setCurrentText("Stepped"). Terrain
    Style already defaults to "stepped" (ViewerWindow._terrain_style), and
    _build_toolbar sets the combo before connect(), so a later same-text set
    emits nothing; on_terrain_style_changed() also early-returns whenever
    the requested style matches the current one. Two of the merged copies
    did this and were asserting the default, not exercising a real switch. A
    caller that genuinely needs to drive a style change should go through
    "Flat" first.

    Caller must call window.edit_history.mark_saved() before window.close()
    (tests/conftest.py's close_window() does exactly that). See
    tools/verify_iso_viewer_pick.py's own finally block: closeEvent() calls
    _confirm_discard_changes(), which pops a modal QMessageBox on a dirty
    document and blocks forever offscreen with nothing to click it.
    """
    ensure_qapp()
    import descape.settings as settings_module
    from descape.viewer import ViewerWindow

    if elev_step_pct is not None:
        settings_module._elev_step_pct = elev_step_pct
    if graphics_quality is not None:
        settings_module._graphics_quality = graphics_quality

    window = ViewerWindow()
    window.load_scenario(path)
    if window.scenario is None:
        window.close()
        raise ScenarioLoadError(f"{os.path.basename(str(path))} failed to load")
    if show:
        show_and_settle(window, size)
    return window
