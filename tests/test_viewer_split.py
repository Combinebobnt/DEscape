"""ViewerWindow's mixin split keeps the rules in descape/viewer.py's docstring."""

from __future__ import annotations

import ast
import itertools
import sys
import types
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from descape.viewer import ViewerWindow
from testkit.viewer_family import VIEWER_MODULES

MIXINS = [cls for cls in ViewerWindow.__mro__ if cls.__module__.startswith("descape.viewer_")]


def _defined(cls: type) -> set[str]:
    """Names the class body defines, Python's own dunder bookkeeping (__module__, __doc__, ...) aside."""
    names = set()
    for name, value in vars(cls).items():
        if not (name.startswith("__") and name.endswith("__")) or isinstance(value, types.FunctionType):
            names.add(name)
    return names


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.Module):
    """Import nodes outside an `if TYPE_CHECKING:` block, at any depth."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _absolute(node: ast.ImportFrom) -> str:
    """The module an ImportFrom names, a relative one resolved from a top-level descape module."""
    if not node.level:
        return node.module or ""
    prefix = "descape" if node.level == 1 else ""  # level 2 and up leave the package: an ImportError anyway
    return ".".join(part for part in (prefix, node.module) if part)


def _viewer_imports(source: str) -> list[str]:
    """Runtime imports of descape.viewer in a descape module's source."""
    found = []
    for node in _runtime_imports(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = _absolute(node)
            if module == "descape.viewer" or (module == "descape" and any(a.name == "viewer" for a in node.names)):
                found.append(ast.unparse(node))
        elif any(a.name == "descape.viewer" for a in node.names):
            found.append(ast.unparse(node))
    return found


def test_nineteen_mixins_precede_qmainwindow():
    mro = ViewerWindow.__mro__
    assert len(MIXINS) == 19
    assert all(mro.index(mixin) < mro.index(QMainWindow) for mixin in MIXINS)


def test_no_name_is_defined_by_two_classes():
    clashes = {
        (a.__name__, b.__name__): shared
        for a, b in itertools.combinations([ViewerWindow, *MIXINS], 2)
        if (shared := _defined(a) & _defined(b))
    }
    assert not clashes


def test_mixins_hold_no_state_setup_event_override_or_signal():
    for mixin in MIXINS:
        assert "__init__" not in vars(mixin), mixin.__name__
        assert not [name for name in vars(mixin) if name.endswith("Event")], mixin.__name__
        assert not [name for name, value in vars(mixin).items() if isinstance(value, pyqtSignal)], mixin.__name__


def test_no_mixin_shadows_a_qmainwindow_attribute():
    """So the order of the bases cannot change which function a name resolves to."""
    qt_names = set(dir(QMainWindow))
    shadowed = {
        mixin.__name__: names
        for mixin in MIXINS
        if (names := sorted(n for n in vars(mixin) if n in qt_names and not (n.startswith("__") and n.endswith("__"))))
    }
    assert not shadowed


def test_no_mixin_module_imports_viewer_at_runtime():
    found = {
        mixin.__module__: hits
        for mixin in MIXINS
        if (hits := _viewer_imports(Path(sys.modules[mixin.__module__].__file__).read_text(encoding="utf-8")))
    }
    assert not found


def test_viewer_family_lists_every_module_in_the_mro():
    modules = {cls.__module__ for cls in ViewerWindow.__mro__ if cls.__module__.split(".")[0] == "descape"}
    assert modules <= set(VIEWER_MODULES), sorted(modules - set(VIEWER_MODULES))


def test_viewer_import_scan_skips_type_checking_and_resolves_relative_imports():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from descape.viewer import ViewerWindow\n"
        "else:\n    import descape.viewer\n"
        "def f():\n    from descape import viewer\n"
        "from .viewer import main\n"
        "from . import viewer as v\n"
        "from .viewer_common import status_color\n"
        "import descape.viewer_common\n"
    )
    assert sorted(_viewer_imports(source)) == sorted(
        ["import descape.viewer", "from descape import viewer", "from .viewer import main", "from . import viewer as v"]
    )
