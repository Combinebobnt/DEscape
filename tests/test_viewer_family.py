"""Tests and tools set a viewer module's globals only through `testkit.viewer_family`.

TASK-517 splits `descape/viewer.py` into several modules. A test that rebinds
a global on the old module would then miss every reader that moved, and could
still pass. The scan (`testkit.module_family_scan`, which lists the forms)
fails on each raw form of module-dict access through an alias of a family
module (`testkit.viewer_family.VIEWER_MODULES`), in every Python file under
tests/, tools/ and testkit/. A module reached some other way is listed in
`_HAND_LISTED_ALIASES`. The scanner's own cases live here;
`tests/test_render_family.py` adds the render families' rules.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from testkit import viewer_family
from testkit.module_family_scan import module_defines, python_files, scan_files, scan_source
from testkit.viewer_family import set_viewer_global, viewer_global, viewer_global_bound

ROOT = Path(__file__).resolve().parent.parent
_SCANNED_DIRS = ("tests", "tools", "testkit")
_HELPERS = (ROOT / "testkit" / "viewer_family.py", ROOT / "testkit" / "module_family.py")

# A family module reached through something the scan cannot follow, such as a fixture's value: file -> {name: module}.
_HAND_LISTED_ALIASES = {
    "tests/test_crash_dialog.py": {"_crash_hooks_installed": "descape.viewer"},
}


def test_no_test_or_tool_reaches_into_a_viewer_modules_globals() -> None:
    files = python_files(ROOT, _SCANNED_DIRS, exclude=_HELPERS)
    problems = scan_files(ROOT, files, viewer_family.VIEWER_MODULES, module_defines, _HAND_LISTED_ALIASES)
    assert not problems, "\n".join(problems)


def test_the_hand_listed_aliases_still_exist() -> None:
    for rel, names in _HAND_LISTED_ALIASES.items():
        path = ROOT / rel
        assert path.is_file(), rel
        source = path.read_text(encoding="utf-8")
        for name, module_name in names.items():
            assert module_name in viewer_family.VIEWER_MODULES, (rel, module_name)
            assert any(isinstance(n, ast.arg) and n.arg == name for n in ast.walk(ast.parse(source))), (rel, name)


_FAKE_DEFINES = {"descape.viewer": frozenset({"ViewerWindow", "THRESHOLD"})}.__getitem__
_IMPORT = "import descape.viewer as vm\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_IMPORT + "monkeypatch.setattr(vm, 'THRESHOLD', 1)", "setattr() on descape.viewer"),
        (_IMPORT + "setattr(vm, 'THRESHOLD', 1)", "setattr() on descape.viewer"),
        (_IMPORT + "real = getattr(vm, name)", "getattr() on descape.viewer"),
        (_IMPORT + "if hasattr(vm, 'THRESHOLD'): pass", "hasattr() on descape.viewer"),
        (_IMPORT + "monkeypatch.delattr(vm, 'THRESHOLD')", "delattr() on descape.viewer"),
        (_IMPORT + "monkeypatch.setattr(target=vm, name='THRESHOLD', value=1)", "setattr() on descape.viewer"),
        (_IMPORT + "object.__setattr__(vm, 'THRESHOLD', 1)", "__setattr__() on descape.viewer"),
        ("mock.patch.multiple('descape.viewer', THRESHOLD=1)", "multiple() on descape.viewer"),
        ("from descape.viewer import QMessageBox, ViewerWindow", "import of descape.viewer.QMessageBox"),
        ("import sys\nv = sys.modules.get('descape.viewer')\nv.X = 1", "assignment to descape.viewer.X"),
        (
            (
                "from descape import viewer\nclass TestA:\n    viewer = None\n"
                "    def test(self, monkeypatch):\n        monkeypatch.setattr(viewer, 'THRESHOLD', 1)"
            ),
            "setattr() on descape.viewer",
        ),
        (_IMPORT + "with mock.patch.object(vm, 'THRESHOLD', 1): pass", "object() on descape.viewer"),
        (_IMPORT + "monkeypatch.setitem(vm.__dict__, 'THRESHOLD', 1)", "__dict__ of descape.viewer"),
        (_IMPORT + "values = vars(vm)", "vars() on descape.viewer"),
        (_IMPORT + "vm.THRESHOLD = 1", "assignment to descape.viewer.THRESHOLD"),
        (_IMPORT + "del vm.THRESHOLD", "assignment to descape.viewer.THRESHOLD"),
        ("monkeypatch.setattr('descape.viewer.QMessageBox.question', f)", "string target 'descape.viewer.QMessageBox"),
        ("mock.patch(f'descape.viewer.{name}')", "string target 'descape.viewer.'"),
        (_IMPORT + "vm.QMessageBox.warning(None, 't', 'x')", "read of descape.viewer.QMessageBox, a name it only imports"),
        ("from descape import viewer\ndef f():\n    viewer.THRESHOLD = 1", "assignment to descape.viewer.THRESHOLD"),
        ("from descape import viewer as v\nw = v\nsetattr(w, 'X', 1)", "setattr() on descape.viewer"),
        ("import importlib\nv = importlib.import_module('descape.viewer')\nv.X = 1", "assignment to descape.viewer.X"),
        ("import sys\nv = sys.modules['descape.viewer']\nv.X = 1", "assignment to descape.viewer.X"),
        ("import descape.viewer\ndescape.viewer.THRESHOLD = 1", "assignment to descape.viewer.THRESHOLD"),
        ("def test(fixture_module):\n    m = fixture_module\n    m.X = 1", "assignment to descape.viewer.X"),
    ],
)
def test_the_scan_flags_each_raw_form(source: str, expected: str) -> None:
    found = scan_source(source, ("descape.viewer",), _FAKE_DEFINES, {"fixture_module": "descape.viewer"})
    assert any(expected in message for _line, message in found), found


@pytest.mark.parametrize(
    "source",
    [
        "from PyQt5.QtWidgets import QMessageBox\nmonkeypatch.setattr(QMessageBox, 'question', f)",
        _IMPORT + "monkeypatch.setattr(vm.ViewerWindow, 'undo', f)",
        _IMPORT + "assert vm.THRESHOLD > 1\nwindow = vm.ViewerWindow()",
        _IMPORT + "set_viewer_global(monkeypatch, 'THRESHOLD', 1)\nreal = viewer_global('QMessageBox')",
        "from descape import viewer\ndef test(viewer):\n    viewer.zoom = 2\n    setattr(viewer, 'zoom', 3)",
        "from descape import viewer\ndef test():\n    viewer = make_window()\n    viewer.zoom = 2",
        "from descape import viewer_common as vc\nvc.X = 1",
        "'descape.viewer.X is named in a bare string, as a docstring would'",
        "monkeypatch.setattr('descape.viewer_common.X', 1)",
        "from descape.viewer import THRESHOLD, ViewerWindow",
        "class T:\n    from descape import viewer as w\n    def g(self):\n        w.zoom = 1",
    ],
)
def test_the_scan_passes_what_is_not_a_module_dict_access(source: str) -> None:
    assert scan_source(source, ("descape.viewer",), _FAKE_DEFINES) == []


def test_a_class_object_patch_is_not_flagged_as_a_module_write() -> None:
    found = scan_source(_IMPORT + "monkeypatch.setattr(vm.QMessageBox, 'question', f)", ("descape.viewer",), _FAKE_DEFINES)
    assert [message for _line, message in found] == [
        "read of descape.viewer.QMessageBox, a name it only imports: import it from its own module"
    ]


@pytest.fixture
def fake_family(monkeypatch):
    modules = (types.ModuleType("_viewer_family_fake_a"), types.ModuleType("_viewer_family_fake_b"))
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(viewer_family, "VIEWER_MODULES", tuple(m.__name__ for m in modules))
    return modules


def test_a_name_two_modules_bind_is_set_in_both_and_undone(fake_family) -> None:
    first, second = fake_family
    shared = object()
    first.SHARED = second.SHARED = shared
    first.ONLY_FIRST = "first"
    patch = pytest.MonkeyPatch()
    set_viewer_global(patch, "SHARED", 5)
    set_viewer_global(patch, "ONLY_FIRST", 6)
    assert (first.SHARED, second.SHARED, first.ONLY_FIRST) == (5, 5, 6)
    assert "ONLY_FIRST" not in vars(second), "a module that does not bind the name is left alone"
    assert viewer_global("SHARED") == 5
    patch.undo()
    assert first.SHARED is shared and second.SHARED is shared and first.ONLY_FIRST == "first"


def test_an_unknown_name_raises_naming_the_family(fake_family) -> None:
    patch = pytest.MonkeyPatch()
    with pytest.raises(LookupError, match="_viewer_family_fake_a, _viewer_family_fake_b"):
        set_viewer_global(patch, "MISSING", 1)
    with pytest.raises(LookupError):
        viewer_global("MISSING")
    assert not viewer_global_bound("MISSING")
    fake_family[1].MISSING = None
    assert viewer_global_bound("MISSING")


def test_viewer_global_refuses_bindings_that_disagree(fake_family) -> None:
    first, second = fake_family
    first.NAME, second.NAME = object(), object()
    with pytest.raises(AssertionError, match="NAME"):
        viewer_global("NAME")
