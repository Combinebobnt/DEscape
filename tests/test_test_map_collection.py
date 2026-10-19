"""Guards the `test_map` mark: a test that reads the generated test map or
its resave without it would silently run in the default tier, which stays
the same as before the map existed. The map items run in the quick and
stress tiers only."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

import conftest

TESTS_DIR = conftest.ROOT / "tests"
MAP_NAMES = frozenset({"TEST_MAP_PATH", "TEST_MAP_RESAVE_PATH"})
MAP_FILES = (conftest.TEST_MAP_PATH.name, conftest.TEST_MAP_RESAVE_PATH.name)


def _mentions(node: ast.AST, names: frozenset[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in names:
            return True
        if isinstance(sub, ast.alias) and sub.name in names:
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and any(f in sub.value for f in MAP_FILES):
            return True
    return False


def _has_test_map_mark(decorators: list[ast.expr]) -> bool:
    for deco in decorators:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Attribute) and target.attr == "test_map":
            return True
    return False


def _pytestmark_has_test_map(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets)
        and _mentions(stmt.value, frozenset({"test_map"}))
        for stmt in body
    )


def unmarked_map_tests(source: str) -> list[str]:
    """Tests (top-level or in a class) that reach the map unmarked: directly, by file
    name, or through one module-level constant or helper function."""
    tree = ast.parse(source)
    if _pytestmark_has_test_map(tree.body):
        return []
    constants = {
        t.id
        for stmt in tree.body
        if isinstance(stmt, ast.Assign) and _mentions(stmt.value, MAP_NAMES)
        for t in stmt.targets
        if isinstance(t, ast.Name)
    }
    functions = [s for s in tree.body if isinstance(s, ast.FunctionDef)]
    names = MAP_NAMES | constants
    helpers = {f.name for f in functions if not f.name.startswith("test_") and _mentions(f, names)}
    reach = frozenset(names | helpers)
    tests = [(f.name, f) for f in functions]
    for cls in (s for s in tree.body if isinstance(s, ast.ClassDef)):
        if _has_test_map_mark(cls.decorator_list) or _pytestmark_has_test_map(cls.body):
            continue
        tests += [(f"{cls.name}::{f.name}", f) for f in cls.body if isinstance(f, ast.FunctionDef)]
    return [
        label
        for label, f in tests
        if f.name.startswith("test_") and _mentions(f, reach) and not _has_test_map_mark(f.decorator_list)
    ]


def test_every_map_reading_test_carries_the_test_map_mark() -> None:
    unmarked = [
        f"{path.name}::{name}"
        for path in sorted(TESTS_DIR.glob("test_*.py"))
        if path.name != "test_test_map_collection.py"
        for name in unmarked_map_tests(path.read_text(encoding="utf-8"))
    ]
    assert unmarked == []


@pytest.mark.parametrize("name", ["test_test_map_fixture.py", "test_test_map_game_resave.py"])
def test_the_map_modules_would_be_caught_without_their_pytestmark(name: str) -> None:
    source = (TESTS_DIR / name).read_text(encoding="utf-8")
    assert "pytestmark = pytest.mark.test_map\n" in source
    assert unmarked_map_tests(source.replace("pytestmark = pytest.mark.test_map\n", "")), name


def test_the_guard_catches_each_way_of_reaching_the_map() -> None:
    source = (
        "import conftest\n"
        "MAP = conftest.TEST_MAP_PATH\n"
        "LITERAL = 'fixtures/test_map_game_resave.aoe2scenario'\n"
        "def _load():\n    return conftest.TEST_MAP_PATH\n"
        "def test_direct():\n    conftest.TEST_MAP_RESAVE_PATH\n"
        "def test_via_helper():\n    _load()\n"
        "def test_via_constant():\n    MAP\n"
        "def test_via_literal():\n    LITERAL\n"
        "class TestX:\n    def test_method(self):\n        MAP\n"
        "@pytest.mark.test_map\ndef test_marked():\n    _load()\n"
        "def test_unrelated():\n    pass\n"
    )
    assert unmarked_map_tests(source) == [
        "test_direct",
        "test_via_helper",
        "test_via_constant",
        "test_via_literal",
        "TestX::test_method",
    ]
    assert unmarked_map_tests("pytestmark = pytest.mark.test_map\n" + source) == []


@pytest.mark.parametrize("attr", sorted(MAP_NAMES))  # by name: a Path param would itself get test_map
def test_the_collection_hook_marks_an_item_parametrized_with_a_map(attr: str) -> None:
    path = getattr(conftest, attr)
    marked, plain = [], []
    map_item = SimpleNamespace(callspec=SimpleNamespace(params={"path": path}), add_marker=marked.append)
    other = conftest.ROOT / "tests" / "fixtures" / "units_120x120.aoe2scenario"
    other_item = SimpleNamespace(callspec=SimpleNamespace(params={"path": other}), add_marker=plain.append)
    unparametrized = SimpleNamespace(add_marker=plain.append)
    conftest.pytest_collection_modifyitems(None, [map_item, other_item, unparametrized])
    assert [m.name for m in marked] == ["test_map"]
    assert plain == []
