"""Sets and reads the viewer's module globals for tests and tools.

A test that rebinds a global through `monkeypatch.setattr(descape.viewer, ...)`
only reaches code whose globals are that module's. TASK-517 split
`descape/viewer.py` into several modules, and a patch on the old module would
then miss a reader that moved, while the test still passed. So no test or tool
writes to a viewer module directly: every write goes through
`set_viewer_global()`, which sets the name in every module of the family that
binds it. `VIEWER_MODULES` lists that family; each cut appends the modules it
creates. `tests/test_viewer_family.py` fails on the raw forms. The logic is
`testkit.module_family`'s, shared with `testkit.render_family`.

`monkeypatch` is pytest's fixture, or a `pytest.MonkeyPatch()` a tool creates
and undoes itself (or never undoes, for a switch that must last the process).
"""

from __future__ import annotations

from typing import Any

from testkit.module_family import ModuleFamily

VIEWER_MODULES: tuple[str, ...] = (
    "descape.viewer",
    "descape.viewer_common",
    "descape.settings_dialog",
    "descape.mirror_dialog",
    "descape.viewer_menus",
    "descape.viewer_toolbar",
    "descape.viewer_unit_inspector",
    "descape.viewer_unit_move",
    "descape.viewer_unit_place",
    "descape.viewer_unit_ops",
    "descape.viewer_unit_edit",
    "descape.viewer_history",
    "descape.viewer_view",
    "descape.viewer_trigger_overlay",
    "descape.viewer_options",
    "descape.viewer_tool_state",
    "descape.viewer_strokes",
    "descape.viewer_region",
    "descape.viewer_render",
    "descape.viewer_triggers",
    "descape.viewer_file",
    "descape.viewer_find",
    "descape.viewer_info",
)

VIEWER = ModuleFamily("viewer", lambda: VIEWER_MODULES)


def set_viewer_global(monkeypatch: Any, name: str, value: Any) -> None:
    """Sets `name` to `value` in every family module that binds it, undone with `monkeypatch`."""
    VIEWER.set_global(monkeypatch, name, value)


def viewer_global(name: str) -> Any:
    """The current value of `name`, which every family module binding it must agree on."""
    return VIEWER.get_global(name)


def viewer_global_bound(name: str) -> bool:
    """Whether any family module binds `name`."""
    return VIEWER.is_bound(name)
