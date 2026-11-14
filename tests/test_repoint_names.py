"""tools/repoint_names.py: callers of moved names are repointed by token, refused forms are left alone, and --check
proves a repoint changed module qualifiers only."""

from __future__ import annotations

import importlib
import sys

import pytest

from tools import repoint_names as rn

BASE_CORE = '''"""Core."""

from __future__ import annotations

SCALE = 2
_PRIVATE = 3


def draw(x):
    return x * SCALE


def kept(x):
    return draw(x) + _PRIVATE
'''

SPLIT_CORE = '''"""Core."""

from __future__ import annotations

from pkg.core_draw import _PRIVATE, draw


def kept(x):
    return draw(x) + _PRIVATE
'''

CORE_DRAW = '''"""Drawing."""

from __future__ import annotations

SCALE = 2
_PRIVATE = 3


def draw(x):
    return x * SCALE
'''

USER = '''"""Uses core: a module alias, a `from` import mixing moved and kept names, a function-scope import."""

from __future__ import annotations

from pkg import core
from pkg.core import SCALE, kept


def total(x):
    return core.draw(x) + core._PRIVATE + kept(x) + SCALE


def lazy(x):
    from pkg import core as c

    return f"{c.draw(x)}!"


def shadow(core):
    return core.draw
'''

REL = '''from __future__ import annotations

from . import core
from .core import draw


def twice(x):
    return draw(x) + core.kept(x)
'''

TOOL = '''import pkg.core as m


def go():
    return m.draw(1)
'''

FILES = {"pkg/user.py": USER, "pkg/rel.py": REL, "extra/t.py": TOOL}


def _fake_git(base: dict[str, str], root):
    def git(*args: str) -> str:
        working = {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*.py")}
        if args[0] == "show":
            return base[args[1].partition(":")[2]]
        if args[:2] == ("ls-tree", "-r"):
            files = list(base)
        elif args[0] == "ls-tree":
            files = [p for p in base if p.startswith(args[-1])]
        elif args[0] == "diff":
            files = [p for p in base if working.get(p) != base[p]]
        elif args[0] == "ls-files":
            files = [p for p in working if p not in base]
        else:
            raise AssertionError(args)
        return "".join(f"{p}\n" for p in files)

    return git


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic package at its base, then split the way a cut leaves it; yields the root."""
    base = {"pkg/__init__.py": "", "pkg/core.py": BASE_CORE, **FILES}
    for path, text in {**base, "pkg/core.py": SPLIT_CORE, "pkg/core_draw.py": CORE_DRAW}.items():
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_text(text)
    monkeypatch.setattr(rn, "ROOT", tmp_path)
    monkeypatch.setattr(rn, "DEFAULT_DIRS", ("pkg", "extra"))
    monkeypatch.setattr(rn, "_git", _fake_git(base, tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [m for m in sys.modules if m.partition(".")[0] in {"pkg", "extra"}]:
        del sys.modules[name]


def _run(*extra: str) -> int:
    return rn.main(["--base", "base", "--module", "pkg.core", *extra])


def test_repoint_rewrites_every_form_runs_and_is_idempotent(tree):
    assert _run() == 0
    user, rel, tool = ((tree / p).read_text() for p in FILES)
    assert "return core_draw.draw(x) + core_draw._PRIVATE + kept(x) + SCALE" in user  # public and private names
    assert "from pkg.core import kept\nfrom pkg.core_draw import SCALE\n" in user  # split by home
    assert "from pkg import core\n" not in user  # the alias went unused, ruff dropped it
    assert '    from pkg import core_draw\n\n    return f"{core_draw.draw(x)}!"' in user  # function scope, f-string
    assert "def shadow(core):\n    return core.draw" in user  # a parameter is not the module
    assert "from .core_draw import draw" in rel and "core.kept(x)" in rel  # relative forms, a kept name
    assert "import pkg.core_draw as core_draw\n" in tool and "core_draw.draw(1)" in tool  # `import ... as` form
    after = {p: (tree / p).read_text() for p in FILES}
    assert _run() == 0
    assert {p: (tree / p).read_text() for p in FILES} == after
    user_module = importlib.import_module("pkg.user")
    assert (user_module.total(1), user_module.lazy(1), importlib.import_module("pkg.rel").twice(1)) == (12, "2!", 7)
    assert importlib.import_module("extra.t").go() == 2


def test_check_passes_a_repoint_and_fails_a_one_token_edit(tree, capsys):
    assert _run("--check") == 1  # nothing repointed yet: every caller would change
    assert _run() == 0
    assert _run("--check") == 0
    user = tree / "pkg/user.py"
    user.write_text(user.read_text().replace("+ SCALE", "- SCALE"))
    capsys.readouterr()
    assert _run("--check") == 1
    assert "pkg/user.py: line" in capsys.readouterr().out
    assert _run("--check", "--allow", "pkg/user.py") == 0


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("from pkg.core import *\n", "star import"),
        ("from pkg import core\nmods = (core,)\nx = core.draw\n", "used as a value"),
        ("from pkg import core\ncore.draw = None\n", "a write or del"),
        ("import pkg.core\nx = pkg.core.draw\n", "not an import alias"),
        ("from pkg import core\nc = core\nx = c.draw\n", "used as a value"),
        ("from pkg import core\ncore_draw = 1\nx = core.draw\n", "already bound"),
        ("from pkg import core\n\n\ndef f(core_draw):\n    return core.draw\n", "already bound"),
        ("from pkg import core; y = 1\nx = core.draw\n", "shares its line"),
        ("from pkg.core import (  # why\n    draw,\n)\n", "comment inside"),
        ("from pkg import core\n\n\ndef f(x: 'core.draw'):\n    return core.kept\n", "a string naming core.draw"),
        ("from pkg import core\n\nmatch 1:\n    case core_draw:\n        x = core.draw(core_draw)\n", "already bound"),  # a capture
        ("from pkg import core\n\nmatch [1]:\n    case [*core_draw]:\n        x = core.draw(core_draw)\n", "already bound"),
        ("from pkg import core\n\nmatch {}:\n    case {**core_draw}:\n        x = core.draw(core_draw)\n", "already bound"),
        ("from pkg import core\n\n\ndef f():\n    x = core.draw(1)\n    from pkg import core_draw\n\n    return x\n", "after the read"),
    ],
)
def test_refused_forms_leave_the_file_alone(source, why):
    result = rn.repoint_source(source, "pkg.core", {"draw": "pkg.core_draw", "SCALE": "pkg.core_draw"}, "pkg")
    assert result.text == source
    assert any(why in r for r in result.refusals), result.refusals


def test_a_docstring_and_a_string_outside_the_alias_scope_are_not_refused():
    source = '"""See core.draw."""\nfixture = "core.draw(1)"\n\n\ndef f():\n    from pkg import core\n\n    return core.draw\n'
    result = rn.repoint_source(source, "pkg.core", {"draw": "pkg.core_draw"}, "pkg")
    assert result.refusals == [] and "from pkg import core_draw" in result.text and 'fixture = "core.draw(1)"' in result.text


@pytest.mark.parametrize(
    "source",
    [
        # The home imported for type checkers only.
        "from typing import TYPE_CHECKING\n\nfrom pkg import core\n\nif TYPE_CHECKING:\n    from pkg import core_draw\nX = core.draw(1)\n",
        # Imported below a read in the same scope, the alias bound there too.
        "def f():\n    from pkg import core\n\n    x = core.draw(1)\n    from pkg import core_draw\n\n    return x\n\n\nX = f()\n",
        "from pkg import core\n\nX = core.draw(1)\nfrom pkg import core_draw\n",
    ],
)
def test_a_home_imported_only_conditionally_or_below_the_read_still_gets_its_import(tree, source):
    result = rn.repoint_source(source, "pkg.core", {"draw": "pkg.core_draw"}, "pkg")
    assert result.refusals == [] and result.imports == 1
    assert rn.repoint_source(result.text, "pkg.core", {"draw": "pkg.core_draw"}, "pkg").text == result.text  # a second run
    (tree / "pkg/probe.py").write_text(result.text)
    assert importlib.import_module("pkg.probe").X == 2


def test_check_fails_a_read_made_bare_and_a_shadowing_import():
    family = ("pkg.core", "pkg.core_draw")
    old = "from pkg import core\nfrom pkg.core import kept, draw\n\nfoo = 1\n\n\ndef f():\n    return core.draw(foo) + kept\n"
    repointed = old.replace("from pkg import core\n", "from pkg import core_draw\n").replace("core.draw(", "core_draw.draw(")
    repointed = repointed.replace("from pkg.core import kept, draw\n", "from pkg.core import kept\nfrom pkg.core_draw import draw\n")
    assert rn.proof_problems(old, repointed, family, "pkg") is None
    assert rn.proof_problems(old, repointed.replace("core_draw.draw(", "draw("), family, "pkg")  # the bare-name swap
    shadowed = repointed.replace("def f():\n", "def f():\n    from pkg.core_draw import foo\n\n")
    assert rn.proof_problems(old, shadowed, family, "pkg")


def test_a_moved_name_with_two_homes_or_none_fails_before_writing(tree, capsys):
    (tree / "pkg/core_extra.py").write_text("def draw(x):\n    return x\n")
    assert _run() == 2
    assert "draw: defined now by pkg.core_draw and pkg.core_extra" in capsys.readouterr().err
    (tree / "pkg/core_extra.py").unlink()
    (tree / "pkg/core_draw.py").write_text(CORE_DRAW.replace("SCALE = 2\n", ""))
    assert _run() == 2
    assert "SCALE: no module defines it now" in capsys.readouterr().err
    assert (tree / "pkg/user.py").read_text() == USER
