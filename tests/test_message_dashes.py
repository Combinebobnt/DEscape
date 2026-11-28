"""TASK-702: no message a user can read uses a double hyphen as a clause
separator. Status log lines, labels, dialog text and the refusal exceptions
a save or edit failure shows are all plain string literals, so one source
walk covers them; TASK-612's tooltip pin (tests/test_menu_tooltips.py) is the
Qt-side half.

Exempt: docstrings (module, class, function and attribute ones), comments,
assert messages and the raises in _INTERNAL_RAISES, whose text only a
developer reads in a traceback, plus the literals in _ALLOWED. RuntimeError is
not exempt: undo/redo and jump refusals print it to the status log.
"""

from __future__ import annotations

import ast

import conftest

_INTERNAL_RAISES = {"TypeError", "KeyError", "AssertionError"}

# Not a separator: the zoom label's unknown-value placeholder.
_ALLOWED = {("viewer_toolbar.py", "  Zoom: --  ")}


def _exempt_nodes(tree: ast.AST) -> set[int]:
    exempt: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            # A bare string statement is a docstring, at the top of a body or after an assignment.
            exempt.update(
                id(stmt.value) for stmt in body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            )
        if isinstance(node, ast.Assert) and node.msg is not None:
            exempt.update(id(n) for n in ast.walk(node.msg))
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _INTERNAL_RAISES:
                exempt.update(id(n) for n in ast.walk(node.exc))
    return exempt


def _double_hyphen_literals(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    exempt = _exempt_nodes(tree)
    return [
        f"{name}:{node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exempt
        and (name, node.value) not in _ALLOWED
        and (" -- " in node.value or node.value.startswith("-- ") or node.value.endswith(" --"))
    ]


def test_no_message_literal_in_the_source_uses_a_double_hyphen_separator() -> None:
    found = []
    for path in sorted((conftest.ROOT / "descape").rglob("*.py")):
        found += _double_hyphen_literals(path.read_text(encoding="utf-8"), path.name)
    assert not found, "Use a new sentence, a comma or a colon instead:\n" + "\n".join(found)


def test_the_guard_catches_a_message_and_skips_the_exempt_kinds() -> None:
    source = '''
"""Module docstring -- exempt."""
X = 1
"""Attribute docstring -- exempt."""

def f(log):
    """Function docstring -- exempt."""
    assert X, "assert message -- exempt"
    if X:
        raise TypeError(f"invariant {X} -- exempt")
    log(f"status line {X} -- caught")
    raise ValueError("refusal -- caught")
    raise RuntimeError("undo refusal -- caught")
'''
    found = _double_hyphen_literals(source, "probe.py")
    assert sorted(line.split(": ", 1)[1] for line in found) == [
        "' -- caught'", "'refusal -- caught'", "'undo refusal -- caught'"
    ], found
