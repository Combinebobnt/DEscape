"""tools/split_class.py: a table cut is pure code motion, and verify catches anything that is not."""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from tools import split_class as sc

SOURCE = '''"""A synthetic module to split."""

from __future__ import annotations

import math

# A comment inside the import block.
from collections import OrderedDict

# Rides with _SCALE.
_SCALE = 2


def _helper(x):
    return x * _SCALE


class Base:
    pass


class Widget(Base):
    LIMIT = 3

    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        self._value = new

    # A comment that rides with area_one.
    @staticmethod
    def area_one(x):
        return _helper(x) + math.floor(x)

    AREA_CONST = 5

    def area_two(self):
        return OrderedDict(a=self.AREA_CONST)

    def stays(self):
        return self.LIMIT
'''

SRC = "splitpkg/widget.py"
AREA = "splitpkg/widget_area.py"


def _table(ranges=(("value", "area_two"),), names=("_SCALE", "_helper")):
    area = {"mixin": "_WidgetAreaMixin", "ranges": [list(r) for r in ranges], "module_names": list(names)}
    return {"source": SRC, "class": "Widget", "modules": {AREA: area}}


def _split(source=SOURCE, table=None):
    return sc._render(sc.table_layout(sc._Module(SRC, source), table or _table()))


def _problems(texts, source=SOURCE):
    return sc.verify_texts(sc._Module(SRC, source), {p: sc._Module(p, t) for p, t in texts.items()}, SRC)


def _import_tree(tmp_path, monkeypatch, texts):
    for path, text in texts.items():
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))


def _forget_splitpkg():
    for name in [m for m in sys.modules if m.startswith("splitpkg")]:
        del sys.modules[name]


def test_split_is_motion_and_runs(tmp_path, monkeypatch):
    texts = _split()
    assert _problems(texts) == []
    assert "class Widget(\n    _WidgetAreaMixin, Base,\n):" in texts[SRC]
    assert "from __future__ import annotations\n\nimport math\n\n# A comment inside the import block." in texts[AREA]
    assert "_helper" not in texts[SRC] and "def area_one" in texts[AREA]
    assert "# A comment that rides with area_one." in texts[AREA]
    _import_tree(tmp_path, monkeypatch, texts)
    try:
        widget = importlib.import_module("splitpkg.widget").Widget()
        widget.value = 4
        assert (widget.value, widget.area_one(1.5), widget.area_two()["a"], widget.stays()) == (4, 4, 5, 3)
    finally:
        _forget_splitpkg()


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        (AREA, "math.floor(x)", "math.ceil(x)"),  # a one-token edit in a moved body
        (AREA, "    AREA_CONST = 5\n", ""),  # a dropped member
        (AREA, "    AREA_CONST = 5\n", "    AREA_CONST = 5\n    AREA_CONST = 5\n"),  # a duplicated member
        (AREA, "    AREA_CONST = 5\n", "    AREA_CONST = 5\n    print(AREA_CONST)\n"),  # a bare call in a mixin body
        (AREA, "\n\nclass _WidgetAreaMixin", "\n_WidgetAreaMixin = None\n\n\nclass _WidgetAreaMixin"),  # an added module-level statement
        (SRC, "    def stays", "    Widget_extra = 1\n\n    def stays"),  # an added class attribute
        (AREA, "# A comment that rides", "# A changed comment that rides"),  # a comment edit
    ],
)
def test_verify_catches_non_motion(path, old, new):
    texts = _split()
    assert old in texts[path]
    texts[path] = texts[path].replace(old, new, 1)
    assert _problems(texts)


def test_verify_catches_a_module_level_assignment_to_a_class_attribute():
    texts = _split()
    texts[AREA] += "\n_WidgetAreaMixin.LIMIT = 4\n"
    assert _problems(texts)


def test_verify_catches_lines_swapped_between_bodies():
    """The same lines, so only the statement comparison sees it."""
    texts = _split()
    a, b = "        return OrderedDict(a=self.AREA_CONST)\n", "        return self.LIMIT\n"
    texts[AREA], texts[SRC] = texts[AREA].replace(a, b), texts[SRC].replace(b, a)
    assert any("statement" in problem for problem in _problems(texts))
    assert not any(problem.startswith("line") for problem in _problems(texts))


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        (AREA, "class _WidgetAreaMixin:", "class _WidgetAreaMixin(dict):"),
        (SRC, "    _WidgetAreaMixin, Base,\n", "    _WidgetAreaMixin,\n"),
        (SRC, "    _WidgetAreaMixin, Base,\n", "    _WidgetAreaMixin, Base, metaclass=type,\n"),
        (AREA, "\n\nclass _WidgetAreaMixin", "\nfrom math import ceil as _helper\n\n\nclass _WidgetAreaMixin"),
    ],
)
def test_verify_catches_class_statement_and_late_import_changes(path, old, new):
    """Changes inside the exempt class statements, and an import after the leading block."""
    texts = _split()
    assert old in texts[path]
    texts[path] = texts[path].replace(old, new, 1)
    assert _problems(texts)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (_table(ranges=(("value", "nope"),)), "unknown anchor"),
        (_table(ranges=(("area_two", "value"),)), "inverted range"),
        (_table(ranges=(("value", "area_one"), ("area_one", "area_two"))), "overlaps"),
        (_table(names=("_SCALE", "_missing")), "no module-level name"),
        (_table(names=("_helper",)), "would import _SCALE"),  # a circular import back into the source
    ],
)
def test_bad_tables_fail(table, message):
    with pytest.raises(sc.SplitError, match=message):
        _split(table=table)


def test_cli_writes_nothing_on_a_bad_table_and_refuses_existing_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    (tmp_path / "splitpkg").mkdir()
    (tmp_path / SRC).write_text(SOURCE)
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_table(ranges=(("area_two", "value"),))))
    assert sc.main(["split", "--table", str(table)]) == 2
    assert (tmp_path / SRC).read_text() == SOURCE and not (tmp_path / AREA).exists()
    table.write_text(json.dumps(_table()))
    assert sc.main(["split", "--table", str(table)]) == 0
    assert (tmp_path / AREA).exists()
    (tmp_path / SRC).write_text(SOURCE)
    assert sc.main(["split", "--table", str(table)]) == 2
    assert (tmp_path / SRC).read_text() == SOURCE


def test_like_replays_a_cut_onto_an_edited_source():
    rev = _split()
    rev[AREA] = rev[AREA].replace('"""Part of Widget."""', '"""Widget\'s area members."""')
    lane = SOURCE.replace("math.floor(x)", "math.floor(x) + 1")
    lane = lane.replace("    def stays", "    def area_three(self):\n        return 3\n\n    def stays")
    lane = lane.replace("class Base:", "_NEW = 1\n\n\nclass Base:")
    rev_source, rev_targets = sc._Module(SRC, rev[SRC]), {AREA: sc._Module(AREA, rev[AREA])}
    with pytest.raises(sc.SplitError, match=r"class member area_three(.|\n)*module-level name _NEW"):
        sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {})
    texts = sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {"area_three": AREA, "_NEW": SRC}))
    assert _problems(texts, source=lane) == []
    assert "math.floor(x) + 1" in texts[AREA] and "def area_three" in texts[AREA]
    assert "\"\"\"Widget's area members.\"\"\"" in texts[AREA]
    assert texts[SRC].count("class Widget(") == 1


def test_like_moves_an_unnamed_member_with_the_one_before_it():
    lane = SOURCE.replace("    AREA_CONST = 5\n", "    AREA_CONST = 5\n    LO, HI = 1, 2\n")
    rev = _split()
    rev_source, rev_targets = sc._Module(SRC, rev[SRC]), {AREA: sc._Module(AREA, rev[AREA])}
    texts = sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {}))
    assert "LO, HI = 1, 2" in texts[AREA] and "LO, HI" not in texts[SRC]
    assert _problems(texts, source=lane) == []


def test_an_attribute_docstring_moves_with_its_assignment():
    """A string statement after an assignment ending a range or a module-level run is its doc, not a kept statement."""
    source = SOURCE.replace("_SCALE = 2\n", '_SCALE = 2\n"""Doc of _SCALE."""\n')
    source = source.replace("    AREA_CONST = 5\n", '    AREA_CONST = 5\n    """Doc of AREA_CONST."""\n')
    texts = _split(source, _table(ranges=(("value", "AREA_CONST"),)))
    assert _problems(texts, source=source) == []
    for doc in ("Doc of _SCALE.", "Doc of AREA_CONST."):
        assert doc in texts[AREA] and doc not in texts[SRC]
    rev_source, rev_targets = sc._Module(SRC, texts[SRC]), {AREA: sc._Module(AREA, texts[AREA])}
    assert sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(SRC, source), {})) == texts


def test_a_tuple_assignment_moves_by_any_of_its_names_and_its_other_names_are_imported(tmp_path, monkeypatch):
    source = SOURCE.replace("_SCALE = 2\n", "_SCALE = 2\nLOW, HIGH = 1, 9\n")
    source = source.replace("return _helper(x) + math.floor(x)", "return _helper(x) + math.floor(x) + LOW")
    source = source.replace("return self.LIMIT", "return self.LIMIT + HIGH")
    texts = _split(source, _table(names=("_SCALE", "_helper", "HIGH", "LOW")))
    assert _problems(texts, source=source) == []
    assert "LOW, HIGH = 1, 9" in texts[AREA] and "from splitpkg.widget_area import HIGH" in texts[SRC]
    rev_source, rev_targets = sc._Module(SRC, texts[SRC]), {AREA: sc._Module(AREA, texts[AREA])}
    replayed = sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(SRC, source), {}))
    assert replayed == texts
    lane = source.replace("LOW, HIGH = 1, 9\n", "LOW, HIGH = 1, 9\nMID, TOP = 5, 7\n")
    with pytest.raises(sc.SplitError, match="module-level name MID"):
        sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {})
    placed = sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {"TOP": AREA}))
    assert "MID, TOP = 5, 7" in placed[AREA] and "MID, TOP" not in placed[SRC]
    with pytest.raises(sc.SplitError, match="different modules"):
        sc.like_layout(rev_source, rev_targets, sc._Module(SRC, lane), {"MID": SRC, "TOP": AREA})
    _import_tree(tmp_path, monkeypatch, texts)
    try:
        widget = importlib.import_module("splitpkg.widget").Widget()
        assert (widget.area_one(1.5), widget.stays()) == (5, 12)
    finally:
        _forget_splitpkg()


def _fake_git(revs: dict[str, dict[str, str]], root):
    """git as split_class calls it, over revisions held as {path: text} and the files under `root`."""

    def working() -> dict[str, str]:
        return {str(p.relative_to(root)): p.read_text() for p in root.rglob("*.py")}

    def git(*args: str) -> str:
        if args[0] == "show":
            rev, _, path = args[1].partition(":")
            return revs[rev][path]
        if args[0] == "ls-tree":
            files = revs[args[-1]]
        elif args[0] == "ls-files":
            files = working()
        elif args[:3] == ("diff", "--name-only", "--diff-filter=A"):
            files = {p: t for p, t in revs[args[4]].items() if p not in revs[args[3]]}
        elif args[:3] == ("diff", "--name-only", "--diff-filter=M"):
            files = {p: t for p, t in revs[args[4]].items() if p in revs[args[3]] and revs[args[3]][p] != t}
        elif args[:2] == ("diff", "--name-only"):
            files = {p: t for p, t in working().items() if revs[args[2]].get(p, t) != t}
        else:
            raise AssertionError(args)
        return "".join(f"{p}\n" for p in files)

    return git


def test_cli_like_then_verify_infer_source_and_class(tmp_path, monkeypatch):
    """The lane-port recipe through main(): `split --like <cut>` then `verify --base <lane>`, no other arguments."""
    cut = _split()
    lane = SOURCE.replace("math.floor(x)", "math.floor(x) + 1")
    uses_area = f"from {AREA[:-3].replace('/', '.')} import _helper\n"  # ties with the source on modules imported
    revs = {"cut~1": {SRC: SOURCE, "tests/t.py": "x = 1\n"}, "cut": {**cut, "tests/t.py": uses_area}}
    revs["lane"] = {SRC: lane, "tests/t.py": "x = 1\n"}
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "_git", _fake_git(revs, tmp_path))
    (tmp_path / "splitpkg").mkdir()
    (tmp_path / SRC).write_text("<<<<<<< conflict markers\n")
    assert sc.main(["split", "--like", "cut", "--assign", "oops"]) == 2
    assert sc.main(["split", "--like", "cut"]) == 2  # the working file does not parse
    revs["HEAD"] = {SRC: lane}
    assert sc.main(["split", "--like", "cut", "--source-rev", "HEAD"]) == 0
    assert "math.floor(x) + 1" in (tmp_path / AREA).read_text()
    assert sc.main(["verify", "--base", "lane"]) == 0
    (tmp_path / AREA).write_text((tmp_path / AREA).read_text().replace("+ 1", "+ 2"))
    assert sc.main(["verify", "--base", "lane"]) == 1


def test_pick_source_takes_the_file_importing_most_new_modules():
    new = {"pkg/a.py", "pkg/b.py"}
    candidates = {"pkg/core.py": "from pkg.a import A\nfrom pkg.b import B\n", "tests/t.py": "from pkg.a import A\n"}
    assert sc.pick_source(candidates, new) == "pkg/core.py"
    with pytest.raises(sc.SplitError):
        sc.pick_source({"x.py": "from pkg.a import A\n", "y.py": "from pkg.b import B\n"}, new)
    both = "from pkg.a import A\nfrom pkg.b import B\n"
    old = {"pkg/core.py": "class A:\n    pass\n\n\nclass B:\n    pass\n", "tests/t.py": ""}
    assert sc.pick_source({"pkg/core.py": both, "tests/t.py": both}, new, old) == "pkg/core.py"  # it used to define them


# A module-only cut (no mixin) that appends to a module the source already imports, and moves a
# class the kept class inherits from.
MOD_SOURCE = '''"""A synthetic module whose module-level names move out."""

from __future__ import annotations

import math

from splitpkg.common import BASE_LIMIT

# Rides with _MEMO.
_MEMO: dict = {}


def _shared(x):
    return math.floor(x) + BASE_LIMIT


class Panel:
    LIMIT = 2

    def size(self):
        return _shared(self.LIMIT) + len(_MEMO)


class Window(Panel):  # the header is kept as written
    """Kept whole."""

    def value(self):
        return _shared(1.5)
'''

COMMON = '''"""Shared names."""

from __future__ import annotations

BASE_LIMIT = 10
'''

MSRC, COMMON_PATH, PANEL = "splitpkg/window.py", "splitpkg/common.py", "splitpkg/panel.py"


def _module_table():
    modules = {COMMON_PATH: {"mixin": None, "module_names": ["_MEMO", "_shared"]}, PANEL: {"mixin": None, "module_names": ["Panel"]}}
    return {"source": MSRC, "class": "Window", "modules": modules}


def _module_split(source=MOD_SOURCE, common=COMMON):
    layout = sc.table_layout(sc._Module(MSRC, source), _module_table(), {COMMON_PATH: sc._Module(COMMON_PATH, common)})
    return sc._render(layout)


def _module_problems(texts, source=MOD_SOURCE, common=COMMON):
    new = {p: sc._Module(p, t) for p, t in texts.items()}
    return sc.verify_texts(sc._Module(MSRC, source), new, MSRC, [sc._Module(COMMON_PATH, common)])


def test_module_only_cut_appends_to_an_existing_module_and_runs(tmp_path, monkeypatch):
    texts = _module_split()
    assert _module_problems(texts) == []
    assert "class Window(Panel):  # the header is kept as written\n" in texts[MSRC]
    common = texts[COMMON_PATH]
    assert common.startswith('"""Shared names."""\n\nfrom __future__ import annotations\nimport math\n')
    assert common.count("from __future__") == 1 and "from splitpkg.common" not in common
    assert common.index("BASE_LIMIT = 10") < common.index("# Rides with _MEMO.\n_MEMO: dict = {}") < common.index("def _shared")
    assert "class Panel" in texts[PANEL] and "class Panel" not in texts[MSRC]
    _import_tree(tmp_path, monkeypatch, texts)
    try:
        window = importlib.import_module("splitpkg.window").Window()
        assert (window.value(), window.size()) == (11, 12)
    finally:
        _forget_splitpkg()


def test_module_only_verify_counts_the_existing_module_on_the_old_side():
    texts = _module_split()
    assert sc.verify_texts(sc._Module(MSRC, MOD_SOURCE), {p: sc._Module(p, t) for p, t in texts.items()}, MSRC)
    texts[COMMON_PATH] = texts[COMMON_PATH].replace("BASE_LIMIT = 10", "BASE_LIMIT = 11")  # an edit to the old part
    assert _module_problems(texts)


@pytest.mark.parametrize(
    ("block", "exempt"),
    [
        ("if TYPE_CHECKING:\n    # A comment inside.\n    from splitpkg.window import Window\n", True),
        ("if typing.TYPE_CHECKING:\n    from splitpkg.window import Window\n", True),
        ("if TYPE_CHECKING:\n    from splitpkg.window import Window\nelse:\n    Window = None\n", False),
        ("if TYPE_CHECKING:\n    Window = None\n", False),
    ],
)
def test_verify_exempts_type_checking_import_blocks_only(block, exempt):
    texts = _module_split()
    texts[PANEL] = texts[PANEL].replace("\n\nclass Panel", f"\n{block}\n\nclass Panel", 1)
    assert (_module_problems(texts) == []) is exempt


def test_module_only_table_problems():
    table = _module_table()
    table["modules"][PANEL]["ranges"] = [["value", "value"]]
    with pytest.raises(sc.SplitError, match="ranges but no mixin"):
        sc.table_layout(sc._Module(MSRC, MOD_SOURCE), table)
    with pytest.raises(sc.SplitError, match="only a module-only target"):
        sc.table_layout(sc._Module(SRC, SOURCE), _table(), {AREA: sc._Module(AREA, COMMON)})
    with pytest.raises(sc.SplitError, match="lacks the source's __future__"):
        _module_split(common=COMMON.replace("from __future__ import annotations\n", ""))


def test_module_only_like_keeps_the_lane_copy_of_the_existing_module():
    """And keeps the cut's own hand edits outside the moved code: the existing module's new docstring
    and an added `if TYPE_CHECKING:` block."""
    rev = _module_split()
    rev[COMMON_PATH] = rev[COMMON_PATH].replace('"""Shared names."""', '"""Shared names, and the moved ones."""')
    block = "if TYPE_CHECKING:\n    from splitpkg.window import Window\n"
    rev[PANEL] = rev[PANEL].replace("\n\nclass Panel", f"\n{block}\n\nclass Panel", 1)
    lane = MOD_SOURCE.replace("+ BASE_LIMIT", "+ BASE_LIMIT + 1")
    lane_common = COMMON + "\nLANE_ONLY = 1\n"
    rev_source = sc._Module(MSRC, rev[MSRC])
    rev_targets = {p: sc._Module(p, rev[p]) for p in (COMMON_PATH, PANEL)}
    existing = {COMMON_PATH: sc._Module(COMMON_PATH, lane_common)}
    base = {COMMON_PATH: sc._Module(COMMON_PATH, COMMON)}
    old = sc._Module(MSRC, MOD_SOURCE)
    texts = sc._render(sc.like_layout(rev_source, rev_targets, sc._Module(MSRC, lane), {}, existing, old, base))
    assert "LANE_ONLY = 1" in texts[COMMON_PATH] and "+ BASE_LIMIT + 1" in texts[COMMON_PATH]
    assert texts[COMMON_PATH].count("BASE_LIMIT = 10") == 1
    assert texts[COMMON_PATH].startswith('"""Shared names, and the moved ones."""\n\nfrom __future__')
    assert block in texts[PANEL]
    assert _module_problems(texts, source=lane, common=lane_common) == []
    both = {COMMON_PATH: sc._Module(COMMON_PATH, lane_common.replace('"""Shared names."""', '"""The lane\'s."""'))}
    with pytest.raises(sc.SplitError, match="both the cut and the source revision changed its docstring"):
        sc.like_layout(rev_source, rev_targets, sc._Module(MSRC, lane), {}, both, old, base)


def test_like_refuses_a_cut_whose_mixin_base_it_cannot_see():
    """A mixin base spelled as an attribute is not detected; the port must fail, not keep the class whole."""
    rev = _split()
    rev[SRC] = rev[SRC].replace("    _WidgetAreaMixin, Base,\n", "    widget_area._WidgetAreaMixin, Base,\n")
    rev_targets = {AREA: sc._Module(AREA, rev[AREA])}
    with pytest.raises(sc.SplitError, match="not mixins of a split class"):
        sc.like_layout(sc._Module(SRC, rev[SRC]), rev_targets, sc._Module(SRC, SOURCE), {}, rev_old=sc._Module(SRC, SOURCE))


def test_module_only_table_needs_no_class():
    table = _module_table()
    del table["class"]
    existing = {COMMON_PATH: sc._Module(COMMON_PATH, COMMON)}
    assert sc._render(sc.table_layout(sc._Module(MSRC, MOD_SOURCE), table, existing)) == _module_split()
    table["modules"][PANEL]["mixin"] = "_PanelMixin"
    with pytest.raises(sc.SplitError, match="a table with a mixin names its class"):
        sc.table_layout(sc._Module(MSRC, MOD_SOURCE), table)


def test_cli_module_only_split_like_and_verify(tmp_path, monkeypatch):
    """split --table onto an existing module, then the lane-port recipe, through main() with nothing passed but revs."""
    cut = _module_split()
    imports_both = "from splitpkg.common import _shared\nfrom splitpkg.panel import Panel\n"  # ties on modules imported
    base = {MSRC: MOD_SOURCE, COMMON_PATH: COMMON, "tests/t.py": "x = 1\n"}
    revs = {"cut~1": base, "cut": {**cut, "tests/t.py": imports_both}}
    revs["lane"] = {**base, MSRC: MOD_SOURCE.replace("+ BASE_LIMIT", "+ BASE_LIMIT + 1"), COMMON_PATH: COMMON + "\nLANE_ONLY = 1\n"}
    revs["HEAD"] = revs["lane"]
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "_git", _fake_git(revs, tmp_path))
    (tmp_path / "splitpkg").mkdir()
    (tmp_path / "tests").mkdir()
    table = tmp_path / "table.json"
    table.write_text(json.dumps(_module_table()))
    (tmp_path / MSRC).write_text(MOD_SOURCE)
    (tmp_path / COMMON_PATH).write_text(COMMON)
    assert sc.main(["split", "--table", str(table)]) == 0
    assert (tmp_path / COMMON_PATH).read_text() == cut[COMMON_PATH]
    (tmp_path / MSRC).write_text(MOD_SOURCE)
    (tmp_path / COMMON_PATH).write_text(COMMON)
    assert sc.main(["split", "--table", str(table)]) == 2  # panel.py is a leftover, not at HEAD: refused, not appended to
    assert (tmp_path / COMMON_PATH).read_text() == COMMON and (tmp_path / PANEL).read_text() == cut[PANEL]
    assert sc.main(["split", "--table", str(table), "--source-rev", "cut~1"]) == 2  # panel.py is already there
    (tmp_path / PANEL).unlink()
    assert sc.main(["split", "--table", str(table), "--source-rev", "cut~1"]) == 0  # common.py read at the rev
    assert (tmp_path / COMMON_PATH).read_text() == cut[COMMON_PATH]
    (tmp_path / "tests/t.py").write_text(imports_both)
    assert sc.main(["verify", "--base", "cut~1"]) == 0
    (tmp_path / MSRC).write_text("<<<<<<< conflict markers\n")
    assert sc.main(["split", "--like", "cut", "--source-rev", "HEAD"]) == 0
    assert "LANE_ONLY = 1" in (tmp_path / COMMON_PATH).read_text()
    (tmp_path / "tests/t.py").write_text("x = 1\n")
    assert sc.main(["verify", "--base", "lane"]) == 0
    (tmp_path / COMMON_PATH).write_text((tmp_path / COMMON_PATH).read_text().replace("LANE_ONLY = 1", "LANE_ONLY = 2"))
    assert sc.main(["verify", "--base", "lane"]) == 1


# A three-layer module-only cut: the source imports only the top layer, which reaches the other two.
LAYER_SOURCE = '''"""A synthetic module cut into three layers."""

from __future__ import annotations

import math

_BASE = 2


def low(x):
    return math.floor(x) * _BASE


def mid(x):
    return low(x) + 1


def top(x):
    return mid(x) * 3


def entry(x):
    return top(x) - 1
'''

LSRC = "splitpkg/layers.py"
LAYERS = ("splitpkg/layers_low.py", "splitpkg/layers_mid.py", "splitpkg/layers_top.py")


def _layer_table():
    names = (["_BASE", "low"], ["mid"], ["top"])
    return {"source": LSRC, "modules": {p: {"mixin": None, "module_names": n} for p, n in zip(LAYERS, names, strict=True)}}


def test_cli_verify_and_like_follow_targets_through_targets(tmp_path, monkeypatch):
    """verify compares the two layers the source reaches only through the top one, and --like replays all three."""
    cut = sc._render(sc.table_layout(sc._Module(LSRC, LAYER_SOURCE), _layer_table()))
    assert "from splitpkg.layers_top import top" in cut[LSRC] and "layers_mid" not in cut[LSRC]
    lane = LAYER_SOURCE.replace("* _BASE", "* _BASE + 1")
    revs = {"cut~1": {LSRC: LAYER_SOURCE}, "cut": cut, "lane": {LSRC: lane}, "HEAD": {LSRC: lane}}
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "_git", _fake_git(revs, tmp_path))
    _import_tree(tmp_path, monkeypatch, cut)
    assert sc.main(["verify", "--base", "cut~1"]) == 0
    low = tmp_path / LAYERS[0]
    low.write_text(low.read_text().replace("math.floor(x)", "math.ceil(x)"))  # a one-token edit in the bottom layer
    assert sc.main(["verify", "--base", "cut~1"]) == 1
    for path in LAYERS:
        (tmp_path / path).unlink()
    assert sc.main(["split", "--like", "cut", "--source-rev", "HEAD"]) == 0
    assert "* _BASE + 1" in low.read_text()
    assert sc.main(["verify", "--base", "lane"]) == 0
    try:
        assert importlib.import_module("splitpkg.layers").entry(1.5) == 11
    finally:
        _forget_splitpkg()


LOW_OLD = '''"""An existing module the cut appends its bottom layer to."""

from __future__ import annotations

FLOOR = 0
'''


def test_cli_verify_and_like_find_an_existing_bottom_layer_through_added_targets(tmp_path, monkeypatch, capsys):
    """The bottom layer is an existing module, reached from the source only through the two added layers."""
    low_path = LAYERS[0]
    cut = sc._render(sc.table_layout(sc._Module(LSRC, LAYER_SOURCE), _layer_table(), {low_path: sc._Module(low_path, LOW_OLD)}))
    assert "FLOOR = 0" in cut[low_path] and "layers_low" not in cut[LSRC] and "layers_mid" not in cut[LSRC]
    lane = LAYER_SOURCE.replace("* _BASE", "* _BASE + 1")
    lane_low = LOW_OLD + "\nLANE_ONLY = 1\n"
    revs = {"cut~1": {LSRC: LAYER_SOURCE, low_path: LOW_OLD}, "cut": cut, "lane": {LSRC: lane, low_path: lane_low}}
    revs["HEAD"] = revs["lane"]
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "_git", _fake_git(revs, tmp_path))
    _import_tree(tmp_path, monkeypatch, cut)
    assert sc.main(["verify", "--base", "cut~1"]) == 0
    assert f"verify: appended to {low_path}," in capsys.readouterr().out
    low = tmp_path / low_path
    low.write_text(low.read_text().replace("FLOOR = 0", "FLOOR = 1"))  # an edit to the existing part
    assert sc.main(["verify", "--base", "cut~1"]) == 1
    _import_tree(tmp_path, monkeypatch, {LSRC: lane, low_path: lane_low})
    for path in LAYERS[1:]:
        (tmp_path / path).unlink()
    assert sc.main(["split", "--like", "cut", "--source-rev", "HEAD"]) == 0
    assert "* _BASE + 1" in low.read_text() and "LANE_ONLY = 1" in low.read_text() and "FLOOR = 0" in low.read_text()
    assert sc.main(["verify", "--base", "lane"]) == 0
    try:
        assert importlib.import_module("splitpkg.layers").entry(1.5) == 11
    finally:
        _forget_splitpkg()
