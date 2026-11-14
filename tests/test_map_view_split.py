"""MapView's mixin split keeps the rules in descape/map_view.py's docstring."""

from __future__ import annotations

import ast
import itertools
import sys
import types
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGraphicsView

from descape.map_view import MapView

MIXINS = [cls for cls in MapView.__mro__ if cls.__module__.startswith("descape.map_view_")]


def _defined(cls: type) -> set[str]:
    """Names the class body defines, Python's own dunder bookkeeping (__module__, __doc__, ...) aside."""
    names = set()
    for name, value in vars(cls).items():
        if not (name.startswith("__") and name.endswith("__")) or isinstance(value, types.FunctionType):
            names.add(name)
    return names


def test_six_mixins_precede_qgraphicsview():
    mro = MapView.__mro__
    assert len(MIXINS) == 6
    assert all(mro.index(mixin) < mro.index(QGraphicsView) for mixin in MIXINS)


def test_no_name_is_defined_by_two_classes():
    clashes = {
        (a.__name__, b.__name__): shared for a, b in itertools.combinations([MapView, *MIXINS], 2) if (shared := _defined(a) & _defined(b))
    }
    assert not clashes


def test_mixins_hold_no_state_setup_and_no_signals():
    for mixin in MIXINS:
        assert "__init__" not in vars(mixin), mixin.__name__
        assert not [name for name, value in vars(mixin).items() if isinstance(value, pyqtSignal)], mixin.__name__


def test_no_mixin_module_imports_map_view():
    for mixin in MIXINS:
        tree = ast.parse(Path(sys.modules[mixin.__module__].__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "descape.map_view", mixin.__module__
                assert not (node.module == "descape" and any(a.name == "map_view" for a in node.names)), mixin.__module__
            elif isinstance(node, ast.Import):
                assert not any(a.name == "descape.map_view" for a in node.names), mixin.__module__
