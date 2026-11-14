"""TriggerPanel's mixin split keeps the rules in descape/trigger_panel.py's docstring."""

from __future__ import annotations

import ast
import itertools
import sys
import types
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from descape.trigger_panel import TriggerPanel

MIXINS = [cls for cls in TriggerPanel.__mro__ if cls.__module__.startswith("descape.trigger_panel_")]


def _defined(cls: type) -> set[str]:
    """Names the class body defines, Python's own dunder bookkeeping (__module__, __doc__, ...) aside."""
    names = set()
    for name, value in vars(cls).items():
        if not (name.startswith("__") and name.endswith("__")) or isinstance(value, types.FunctionType):
            names.add(name)
    return names


def test_eight_mixins_precede_qwidget():
    mro = TriggerPanel.__mro__
    assert len(MIXINS) == 8
    assert all(mro.index(mixin) < mro.index(QWidget) for mixin in MIXINS)


def test_no_name_is_defined_by_two_classes():
    clashes = {
        (a.__name__, b.__name__): shared
        for a, b in itertools.combinations([TriggerPanel, *MIXINS], 2)
        if (shared := _defined(a) & _defined(b))
    }
    assert not clashes


def test_mixins_hold_no_state_setup_signals_or_qt_overrides():
    qt_names = set(dir(QWidget))
    for mixin in MIXINS:
        assert "__init__" not in vars(mixin), mixin.__name__
        assert not [name for name, value in vars(mixin).items() if isinstance(value, pyqtSignal)], mixin.__name__
        assert not _defined(mixin) & qt_names, mixin.__name__


def test_no_mixin_module_imports_trigger_panel_or_a_sibling():
    family = {"trigger_panel", *(mixin.__module__.removeprefix("descape.") for mixin in MIXINS)}
    for mixin in MIXINS:
        tree = ast.parse(Path(sys.modules[mixin.__module__].__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").removeprefix("descape.") not in family, mixin.__module__
                assert not (node.module == "descape" and any(a.name in family for a in node.names)), mixin.__module__
            elif isinstance(node, ast.Import):
                assert not any(a.name.removeprefix("descape.") in family for a in node.names), mixin.__module__
