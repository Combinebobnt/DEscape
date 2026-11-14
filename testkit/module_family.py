"""Sets and reads a family of modules' globals for tests and tools.

A test that rebinds a global through `monkeypatch.setattr(module, ...)` only
reaches code whose globals are that module's. When a module is split, a patch
on the old module misses every reader that moved, and the test can still
pass. So a module that is split, or is about to be, gets a `ModuleFamily`:
the modules its code lives in, and helpers that set a name in every one of
them that binds it. `testkit.viewer_family` and `testkit.render_family` hold
the families; `testkit.module_family_scan` is the guard that bans the raw
forms.

`monkeypatch` is pytest's fixture, or a `pytest.MonkeyPatch()` a tool creates
and undoes itself. A switch that must last the process (a bench child's arm,
a worker's patch) uses a `pytest.MonkeyPatch()` that is never undone.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any


class ModuleFamily:
    """`label` names the family in errors; `modules` returns its module names, read on every call."""

    # `RENDER.X = 1` raises rather than setting X on the family object, a patch that reaches no module.
    __slots__ = ("_modules", "label")

    def __init__(self, label: str, modules: Callable[[], tuple[str, ...]]) -> None:
        self.label = label
        self._modules = modules

    def __repr__(self) -> str:
        return f"ModuleFamily({self.label!r})"

    def module_names(self) -> tuple[str, ...]:
        return tuple(self._modules())

    def binders(self, name: str) -> list[ModuleType]:
        modules = [importlib.import_module(module_name) for module_name in self.module_names()]
        return [module for module in modules if name in vars(module)]

    def _require_binders(self, name: str) -> list[ModuleType]:
        binders = self.binders(name)
        if not binders:
            family = ", ".join(self.module_names())
            raise LookupError(f"no {self.label} module binds {name!r} (family: {family})")
        return binders

    def set_global(self, monkeypatch: Any, name: str, value: Any) -> None:
        """Sets `name` to `value` in every family module that binds it, undone with `monkeypatch`."""
        for module in self._require_binders(name):
            monkeypatch.setattr(module, name, value)

    def get_global(self, name: str, *default: Any) -> Any:
        """The current value of `name`, which every family module binding it must agree on.

        A `default`, as for `getattr()`, is returned when no family module binds `name`.
        """
        if default and not self.is_bound(name):
            return default[0]
        values = [vars(module)[name] for module in self._require_binders(name)]
        first = values[0]
        if any(value is not first for value in values[1:]):
            raise AssertionError(f"{self.label} modules bind different objects to {name!r}")
        return first

    def is_bound(self, name: str) -> bool:
        """Whether any family module binds `name`."""
        return bool(self.binders(name))

    @staticmethod
    def patch(monkeypatch: Any, target: Any, name: str, value: Any) -> None:
        """One entry of a patch list: a family goes to its `set_global`, anything else to `monkeypatch.setattr`.

        A patch list names a family by its `ModuleFamily`, never by a module
        object, so the scan can see every write (`testkit.module_family_scan`).
        """
        if isinstance(target, ModuleFamily):
            target.set_global(monkeypatch, name, value)
        else:
            monkeypatch.setattr(target, name, value)

    @staticmethod
    def current(target: Any, name: str) -> Any:
        """The value a patch list entry replaces: a family's `get_global`, anything else's attribute."""
        return target.get_global(name) if isinstance(target, ModuleFamily) else getattr(target, name)
