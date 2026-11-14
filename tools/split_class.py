"""Splits one class into per-area mixin modules by a cut table, and proves the result is code motion.

Usage, from the repo root:

    python3 tools/split_class.py split --table T.json [--source-rev REV]
    python3 tools/split_class.py split --like REV [--source-rev REV2] [--source PATH] [--assign NAME=MODULE ...]
    python3 tools/split_class.py verify --base REV [--source PATH]
    python3 tools/split_class.py surface package.module:Class

The table is JSON:

    {"source": "descape/map_view.py", "class": "MapView",
     "modules": {"descape/map_view_input.py": {"mixin": "_MapViewInputMixin",
                                               "ranges": [["first", "last"], ...],
                                               "module_names": ["_HELPER", ...]}}}

A range is [first member, last member] of the class body in source order,
both inclusive. A member's segment is every line after the previous member
through its own last line, so the comments and decorators above it move with
it. A tuple assignment (`A, B = ...`) is listed in `module_names` by any of
its names. Members in no range and module-level names in no `module_names` list stay
in the source, whose class gains the mixins as bases ahead of its own. A
module with `"mixin": null` takes `module_names` only and gets no class; with
no mixin in the table at all the class statement is kept as it is.

A module-only target that already exists (at `--source-rev`, else at HEAD
and in the working tree) is appended to: the moved segments go after its last statement,
and the source's import block joins its own, less any `__future__` import and
any import of the target itself.

`split --table` fails before writing anything on an unknown anchor, an
inverted or overlapping range, a missing module name, a mixin target that
already exists, a new target the working tree already has (a leftover of an
earlier run), or a module that
would need a name the source keeps (a circular import). Each new module
starts with the source's import block plus an import of every moved
module-level name it uses from a sibling; prune the rest with
`ruff check --select F401,I001 --fix`.

`split --like REV` replays the cut REV made onto a source that still has the
uncut class (a lane port): each member goes where REV put the member of that
name, and everything outside the moved segments (module docstrings, mixin
class lines and their docstrings, the split class statement) is copied from
REV; the split class's own docstring comes from the source. A member or
module-level name REV lacks fails the run unless placed with `--assign
NAME=MODULE` (MODULE is a target path, or the source path to keep it). A
tuple assignment goes where `--assign` puts any of its names, else where REV
put its first name. An
unnamed class-body statement goes with the named member before it. A target
REV appended to (one that existed at REV~1) is rebuilt from the source
revision's own copy of that file, never REV's, so a lane's edits to it stay;
it takes REV's docstring unless only the source revision changed it (both
changing it fails the run). The `if TYPE_CHECKING:` import blocks REV added
are carried into each file. A class REV's targets define that is neither a
mixin of the split class nor a name the source had fails the run, so a
mixin base the detection misses is not taken for a module-only cut.

`verify --base REV` compares the working tree against REV: the multisets of
`ast.dump()` over the class-body statements and over the module-level
statements, and the multiset of non-blank lines, must be equal both ways.
The cut's targets are the new files and the existing targets (a changed file
that now defines a name the old source defined) that the source takes a
`from` import from, directly or through another target, so a module the
source reaches only through a lower layer is compared too. The old side is
the source at REV plus each existing target's content at REV. Exempt: the leading import block (its comments included),
`if TYPE_CHECKING:` blocks of imports, module docstrings, the class
docstrings of the split class and its mixins, the mixin class lines and the
split class statement. A mixin may have no bases, keywords or decorators,
and the split class keeps its own bases and keywords after the mixins.

`surface` prints every name the class resolves with the type of the object
found statically, for a before/after comparison of the runtime surface.

The source and class of `--like` and `verify` are inferred when not given:
the source is the changed file whose classes take the most bases from the
new modules (then the one importing the most names it used to define from
them, then the one importing the most of them), and the class is the one
with the most bases defined in those modules.
"""

from __future__ import annotations

import argparse
import ast
import collections
import functools
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ""  # the owner value of a segment that stays in the source


class SplitError(Exception):
    pass


def _member_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _bound_names(node: ast.stmt) -> list[str]:
    """The names a module-level statement defines: its member name, or each name of a tuple assignment."""
    if (name := _member_name(node)) is not None:
        return [name]
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List)):
        elts = node.targets[0].elts
        if all(isinstance(e, ast.Name) for e in elts):
            return [e.id for e in elts]
    return []


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _attribute_doc(prev: ast.stmt | None, node: ast.stmt) -> bool:
    """`node` is the string statement documenting the assignment `prev`, so it moves with it."""
    return isinstance(prev, (ast.Assign, ast.AnnAssign)) and _is_docstring(node)


def _follow_attribute_docs(nodes: list[ast.stmt], owner: list[str]) -> None:
    for i in range(1, len(nodes)):
        if _attribute_doc(nodes[i - 1], nodes[i]):
            owner[i] = owner[i - 1]


def _dotted(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _header_end(lines: list[str], node: ast.ClassDef) -> int:
    """The line number of the colon that ends a class statement's header."""
    readline = io.StringIO("".join(lines[node.lineno - 1 :])).readline
    depth = 0
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.OP:
            continue
        if tok.string in {"(", "[", "{"}:
            depth += 1
        elif tok.string in {")", "]", "}"}:
            depth -= 1
        elif tok.string == ":" and depth == 0:
            return node.lineno + tok.start[0] - 1
    raise SplitError(f"no header colon for class {node.name}")


def _class_doc(node: ast.ClassDef) -> ast.stmt | None:
    return node.body[0] if node.body and _is_docstring(node.body[0]) else None


def _is_future(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def _is_type_checking_imports(node: ast.stmt) -> bool:
    """`if TYPE_CHECKING:` (or `typing.TYPE_CHECKING`) holding imports only, with no else."""
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    named = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )
    return named and all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in node.body)


@dataclass
class _Module:
    """A parsed file: docstring, the leading import block, and what follows it."""

    path: str
    text: str
    lines: list[str] = field(init=False)
    tree: ast.Module = field(init=False)
    doc_end: int = field(init=False)  # last line of the module docstring, 0 if none
    imports: list[ast.stmt] = field(init=False)  # the leading run of import statements
    imports_end: int = field(init=False)  # last line of that run (doc_end if there is none)
    body: list[ast.stmt] = field(init=False)  # module-level statements after the run

    def __post_init__(self) -> None:
        self.lines = self.text.splitlines(keepends=True)
        if self.lines and not self.lines[-1].endswith("\n"):
            self.lines[-1] += "\n"
        try:
            self.tree = ast.parse(self.text, filename=self.path)
        except SyntaxError as error:
            raise SplitError(f"{self.path} does not parse (conflict markers? pass --source-rev): {error}") from error
        stmts = list(self.tree.body)
        self.doc_end = stmts.pop(0).end_lineno if stmts and _is_docstring(stmts[0]) else 0
        self.imports = []
        while stmts and isinstance(stmts[0], (ast.Import, ast.ImportFrom)):
            self.imports.append(stmts.pop(0))
        self.imports_end = self.imports[-1].end_lineno if self.imports else self.doc_end
        self.body = stmts

    def span(self, first: int, last: int) -> str:
        """Lines first..last, 1-based and inclusive."""
        return "".join(self.lines[first - 1 : last])

    def head(self) -> str:
        """The docstring and anything above it."""
        return self.span(1, self.doc_end)

    def import_block(self) -> str:
        """The import run with the comments and blank lines inside it."""
        if not self.imports:
            return ""
        return self.span(self.imports[0].lineno, self.imports_end)

    def top_names(self) -> set[str]:
        """The names the module-level statements after the import run define."""
        return {name for node in self.body for name in _bound_names(node)}

    def find_class(self, name: str) -> ast.ClassDef:
        for node in self.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise SplitError(f"{self.path}: no top-level class {name}")

    def class_head(self, node: ast.ClassDef, with_doc: bool) -> str:
        """A class statement through its header colon, plus its docstring if asked."""
        if node.decorator_list:
            raise SplitError(f"{self.path}: decorated class {node.name} is not supported")
        doc = _class_doc(node) if with_doc else None
        return self.span(node.lineno, doc.end_lineno if doc else _header_end(self.lines, node))


def _members(module: _Module, cls: ast.ClassDef) -> list[tuple[ast.stmt, str]]:
    """Each class-body statement but the docstring, with its segment text."""
    doc = _class_doc(cls)
    prev = doc.end_lineno if doc else _header_end(module.lines, cls)
    out = []
    for node in cls.body:
        if node is doc:
            continue
        out.append((node, module.span(prev + 1, node.end_lineno)))
        prev = node.end_lineno
    return out


def _module_segments(module: _Module) -> list[tuple[ast.stmt, str]]:
    out = []
    prev = module.imports_end
    for node in module.body:
        out.append((node, module.span(prev + 1, node.end_lineno)))
        prev = node.end_lineno
    return out


def _strip_blank(text: str) -> str:
    lines = text.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "".join(lines)


def _joined(segments: list[tuple[int, str]], gap: int, prev: int | None = None) -> str:
    """Segments in order; one whose source neighbour is not the one before it here gets `gap` blank lines.
    `prev` is the index of the source segment that precedes the first here, None if none does."""
    out = []
    for index, text in segments:
        adjacent = prev is not None and index == prev + 1
        out.append(text if adjacent else "\n" * gap + _strip_blank(text))
        prev = index
    return "".join(out)


@dataclass
class _Layout:
    """Where every segment goes, and the text that is not a segment."""

    source: _Module
    cls: ast.ClassDef | None  # None: a module-only cut, the class (if any) is an ordinary statement
    targets: dict[str, str | None]  # target path -> mixin class name (None: module names only)
    member_owner: list[str]  # per _members() entry: a target path or CORE
    module_owner: list[str]  # per source.body entry
    target_heads: dict[str, str] = field(default_factory=dict)  # path -> module docstring text
    mixin_heads: dict[str, str] = field(default_factory=dict)  # path -> mixin class line(s) and docstring
    core_head: str | None = None  # the source's new module docstring text, None to keep it
    class_statement: str | None = None  # the split class's new header, None to generate it
    import_bases: dict[str, _Module] = field(default_factory=dict)  # path -> module whose import block leads
    existing: dict[str, _Module] = field(default_factory=dict)  # module-only target path -> the file appended to
    existing_heads: dict[str, str] = field(default_factory=dict)  # existing target path -> its new docstring text
    type_checking: dict[str, str] = field(default_factory=dict)  # path -> `if TYPE_CHECKING:` blocks to add after its imports


def _imported(nodes: list[ast.stmt]) -> set[tuple[str, str]]:
    out = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            out.update((node.module or "", alias.asname or alias.name) for alias in node.names)
    return out


def _import_text(base: _Module | None, source: _Module, needed: dict[str, set[str]], own: str | None = None) -> str:
    """`base`'s import block, then the source's statements it lacks, then `needed` (dotted module -> names).
    `own` is the dotted name of the module written when that is not the source: an import of it, or a
    `__future__` import (legal only as the first statement), is left out of what is added."""
    text = base.import_block() if base else ""
    have_nodes = list(base.imports) if base else []
    seen = {ast.dump(n) for n in have_nodes}
    for node in source.imports:
        if own is not None and (_is_future(node) or (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == own)):
            continue
        if ast.dump(node) not in seen:
            seen.add(ast.dump(node))
            have_nodes.append(node)
            text += ast.get_source_segment(source.text, node) + "\n"
    have = _imported(have_nodes)
    for module_name in sorted(needed):
        names = sorted(n for n in needed[module_name] if (module_name, n) not in have)
        if names:
            text += f"from {module_name} import {', '.join(names)}\n"
    return text


def _loaded_names(nodes: list[ast.stmt]) -> set[str]:
    return {n.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.Name)}


def _wrap_bases(names: list[str], width: int = 79) -> list[str]:
    rows: list[str] = []
    row = "   "
    for name in names:
        piece = f" {name},"
        if len(row) + len(piece) > width and row.strip():
            rows.append(row)
            row = "   "
        row += piece
    if row.strip():
        rows.append(row)
    return rows


def _appended(
    target: _Module,
    source: _Module,
    needed: dict[str, set[str]],
    segments: list[tuple[int, str]],
    head: str | None = None,
    type_checking: str = "",
) -> str:
    """`target` with the source's imports merged into its block and `segments` after its last statement.
    `head` replaces its docstring; `type_checking` goes after its imports."""
    if any(map(_is_future, source.imports)) and not any(map(_is_future, target.imports)):
        raise SplitError(f"{target.path} lacks the source's __future__ import: add it to {target.path} first")
    imports = _import_text(target, source, needed, own=_dotted(target.path))
    head = target.head() if head is None else head
    if target.imports:
        before = head + target.span(target.doc_end + 1, target.imports[0].lineno - 1)
        after = target.span(target.imports_end + 1, len(target.lines))
    else:
        before, after = head + "\n", target.span(target.doc_end + 1, len(target.lines))
    return before + imports + ("\n" + type_checking if type_checking else "") + after + _joined(segments, 2)


def _type_checking_blocks(module: _Module, *without: _Module | None) -> str:
    """The text of `module`'s top-level `if TYPE_CHECKING:` import blocks that every module in `without` lacks."""
    have = {ast.dump(n) for other in without if other is not None for n in other.body}
    blocks = [n for n in module.body if _is_type_checking_imports(n) and ast.dump(n) not in have]
    return "".join(module.span(n.lineno, n.end_lineno) for n in blocks)


def _render(layout: _Layout) -> dict[str, str]:
    """The new text of the source and of every target."""
    src, cls = layout.source, layout.cls
    members = _members(src, cls) if cls is not None else []
    segments = _module_segments(src)

    defined: dict[str, str] = {}  # module-level name -> owning path
    for (node, _text), owner in zip(segments, layout.module_owner, strict=True):
        if node is not cls:
            for name in _bound_names(node):
                defined[name] = owner or src.path

    def needs(owner: str, nodes: list[ast.stmt]) -> dict[str, set[str]]:
        here = owner or src.path
        out: dict[str, set[str]] = collections.defaultdict(set)
        for name in _loaded_names(nodes):
            home = defined.get(name)
            if home is None or home == here:
                continue
            if owner and home == src.path:
                raise SplitError(f"{owner} would import {name} from {src.path}, which imports it: move {name} too")
            out[_dotted(home)].add(name)
        return out

    texts = {}
    for path, mixin in layout.targets.items():
        mod_nodes = [(i, seg) for i, (seg, owner) in enumerate(zip(segments, layout.module_owner, strict=True)) if owner == path]
        cls_nodes = [(i, seg) for i, (seg, owner) in enumerate(zip(members, layout.member_owner, strict=True)) if owner == path]
        if mixin is None and cls_nodes:
            raise SplitError(f"{path} has no mixin but is assigned class members")
        used = needs(path, [seg[0] for _i, seg in mod_nodes + cls_nodes])
        type_checking = layout.type_checking.get(path, "")
        if path in layout.existing:
            segs = [(i, text) for i, (_n, text) in mod_nodes]
            texts[path] = _appended(layout.existing[path], src, used, segs, layout.existing_heads.get(path), type_checking)
            continue
        what = f"{cls.name} members" if mixin is not None else "Names"
        doc = layout.target_heads.get(path) or f'"""{what} split out of {src.path}."""\n'
        out = doc + "\n" + _import_text(layout.import_bases.get(path), src, used)
        out += "\n" + type_checking if type_checking else ""
        out += _joined([(i, text) for i, (_n, text) in mod_nodes], 2)
        if mixin is not None:
            head = layout.mixin_heads.get(path) or f'class {mixin}:\n    """Part of {cls.name}."""\n'
            out += "\n\n" + head + _joined([(i, text) for i, (_n, text) in cls_nodes], 1)
        texts[path] = out

    kept_members = [(i, seg) for i, (seg, owner) in enumerate(zip(members, layout.member_owner, strict=True)) if not owner]
    kept_module = [(i, seg) for i, (seg, owner) in enumerate(zip(segments, layout.module_owner, strict=True)) if not owner]
    used = needs(CORE, [seg[0] for _i, seg in kept_members + kept_module if seg[0] is not cls])
    if cls is not None:
        for name, home in needs(CORE, [*cls.bases, *(k.value for k in cls.keywords), *cls.decorator_list]).items():
            used.setdefault(name, set()).update(home)
    for path, mixin in layout.targets.items():
        if mixin is not None:
            used.setdefault(_dotted(path), set()).add(mixin)

    statement = layout.class_statement
    mixins = [m for m in layout.targets.values() if m is not None]
    if cls is not None and statement is None and not mixins:
        statement = src.class_head(cls, with_doc=False)
    elif cls is not None and statement is None:
        bases = mixins + [ast.get_source_segment(src.text, b) for b in cls.bases]
        bases += [f"{k.arg}={ast.get_source_segment(src.text, k.value)}" for k in cls.keywords]
        statement = f"class {cls.name}(\n" + "\n".join(_wrap_bases(bases)) + "\n):\n"
    doc = _class_doc(cls) if cls is not None else None
    if doc is not None:
        statement += src.span(_header_end(src.lines, cls) + 1, doc.end_lineno)

    body_pieces = []
    for i, (node, text) in kept_module:
        if node is not cls:
            body_pieces.append((i, text))
            continue
        start = src.body[i - 1].end_lineno + 1 if i else src.imports_end + 1
        class_body = _joined([(j, t) for j, (_n, t) in kept_members], 1, prev=-1)
        body_pieces.append((i, src.span(start, node.lineno - 1) + statement + class_body))
    head = layout.core_head if layout.core_head is not None else src.head()
    between = src.span(src.doc_end + 1, src.imports[0].lineno - 1) if src.imports else "\n"
    trailer = src.span(src.body[-1].end_lineno + 1, len(src.lines)) if src.body else ""
    imports = _import_text(layout.import_bases.get(src.path), src, used)
    type_checking = layout.type_checking.get(src.path, "")
    imports += "\n" + type_checking if type_checking else ""
    texts[src.path] = head + between + imports + _joined(body_pieces, 2, prev=-1) + trailer
    return texts


def table_layout(source: _Module, table: dict, existing: dict[str, _Module] | None = None) -> _Layout:
    """`existing` holds each target that is already a file, by path; it must be module-only."""
    existing = existing or {}
    problems = []
    if "class" in table:
        cls = source.find_class(table["class"])
    elif any(spec.get("mixin") for spec in table["modules"].values()):
        raise SplitError("table problems:\n  a table with a mixin names its class")
    else:
        cls = None
    members = _members(source, cls) if cls is not None else []
    names = [_member_name(node) for node, _text in members]
    owner = [CORE] * len(members)
    for path, spec in table["modules"].items():
        if spec.get("mixin") is None and spec.get("ranges"):
            problems.append(f"{path}: ranges but no mixin")
            continue
        if path in existing and spec.get("mixin") is not None:
            problems.append(f"{path}: already exists, and only a module-only target can be appended to")
        for first, last in spec.get("ranges", []):
            if first not in names or last not in names:
                problems.append(f"{path}: unknown anchor in [{first}, {last}]")
                continue
            a, b = names.index(first), len(names) - 1 - names[::-1].index(last)
            if a > b:
                problems.append(f"{path}: inverted range [{first}, {last}]")
            for i in range(a, b + 1):
                if owner[i]:
                    problems.append(f"{path}: [{first}, {last}] overlaps {owner[i]} at {names[i]}")
                owner[i] = path
    module_names = [_bound_names(node) for node in source.body]
    module_owner = [CORE] * len(source.body)
    for path, spec in table["modules"].items():
        for name in spec.get("module_names", []):
            hits = [i for i, n in enumerate(module_names) if name in n and source.body[i] is not cls]
            if not hits:
                problems.append(f"{path}: no module-level name {name}")
            for i in hits:
                if module_owner[i] == path and len(module_names[i]) > 1:
                    continue  # another name of the same tuple assignment, already placed
                if module_owner[i]:
                    problems.append(f"{path}: module name {name} is already in {module_owner[i]}")
                module_owner[i] = path
    _follow_attribute_docs([node for node, _text in members], owner)
    _follow_attribute_docs(source.body, module_owner)
    if problems:
        raise SplitError("table problems:\n  " + "\n  ".join(problems))
    targets = {path: spec.get("mixin") for path, spec in table["modules"].items()}
    layout = _Layout(source, cls, targets, owner, module_owner, existing=dict(existing))
    layout.import_bases = dict.fromkeys([source.path, *targets], source)
    return layout


def _split_class_of(
    source: _Module, new_modules: dict[str, _Module], old: _Module | None = None
) -> tuple[ast.ClassDef | None, dict[str, str]]:
    """The class whose bases are classes the new modules define, and each such base's module.
    (None, {}) when no class has one: a module-only cut. A base the class already had in `old`
    (a moved class it inherits from) is not a mixin."""
    defined = {}
    for path, module in new_modules.items():
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                defined[node.name] = path
    had: dict[str, set[str]] = {}
    for node in old.body if old is not None else []:
        if isinstance(node, ast.ClassDef):
            had[node.name] = {b.id for b in node.bases if isinstance(b, ast.Name)}
    best: tuple[ast.ClassDef | None, dict[str, str]] = None, {}
    for node in source.body:
        if isinstance(node, ast.ClassDef):
            mixins = {
                b.id: defined[b.id]
                for b in node.bases
                if isinstance(b, ast.Name) and b.id in defined and b.id not in had.get(node.name, ())
            }
            if len(mixins) > len(best[1]):
                best = node, mixins
    return best


def _old_top_names(text: str | None) -> set[str]:
    try:
        return _Module("<old>", text).top_names() if text is not None else set()
    except SplitError:
        return set()


def pick_source(candidates: dict[str, str], new_paths: set[str], old: dict[str, str] | None = None) -> str:
    """The candidate with the most class bases imported from `new_paths`, then the most names imported
    from them that its `old` text defined, then the most of them imported."""
    dotted = {_dotted(p) for p in new_paths}
    old = old or {}
    scores = {}
    for path, text in candidates.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module in dotted]
        names = {alias.asname or alias.name for n in imports for alias in n.names}
        bases = sum(isinstance(b, ast.Name) and b.id in names for n in tree.body if isinstance(n, ast.ClassDef) for b in n.bases)
        moved_out = len({alias.name for n in imports for alias in n.names} & _old_top_names(old.get(path)))
        scores[path] = (bases, moved_out, len({n.module for n in imports}))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1][-1] == 0 or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
        raise SplitError(f"cannot tell the split's source among {sorted(candidates)}: pass --source")
    return ranked[0][0]


def _imported_targets(source: _Module, paths: set[str], load: Callable[[str], _Module | None]) -> list[str]:
    """The `paths` that `source` takes a `from` import from, directly or through another one found, in the order
    first reached. `load(path)` is the path's new module, or None when it is not one of the cut's targets."""
    by_dotted = {_dotted(p): p for p in paths}
    out: list[str] = []
    seen = {source.path}
    queue = collections.deque([source])
    while queue:
        module = queue.popleft()
        for node in module.imports + module.body:
            path = by_dotted.get(node.module) if isinstance(node, ast.ImportFrom) else None
            if path is None or path in seen:
                continue
            seen.add(path)
            if (target := load(path)) is not None:
                out.append(path)
                queue.append(target)
    return out


def _target_loader(texts: dict[str, str]) -> Callable[[str], _Module]:
    """`_imported_targets()`'s `load` over path -> new text, each parsed once."""
    return functools.cache(lambda path: _Module(path, texts[path]))


def existing_targets(
    source: _Module, old_source: _Module, candidates: dict[str, tuple[str, str]], added: dict[str, str] | None = None
) -> list[str]:
    """The candidates (path -> (old text, new text)) the cut appended to: reached from `source` as
    `_imported_targets()` follows, through the added modules (`added`, path -> text) and the appended ones,
    and now defining a name `old_source` defined and their old text did not."""
    moved = old_source.top_names() - source.top_names()
    added = added or {}
    load_added = _target_loader(added)

    def load(path: str) -> _Module | None:
        if path in added:
            return load_added(path)
        old_text, new_text = candidates[path]
        module = _Module(path, new_text)
        return module if (module.top_names() - _old_top_names(old_text)) & moved else None

    return [p for p in _imported_targets(source, set(candidates) | set(added), load) if p in candidates]


def like_layout(
    rev_source: _Module,
    rev_targets: dict[str, _Module],
    source: _Module,
    assign: dict[str, str],
    existing: dict[str, _Module] | None = None,
    rev_old: _Module | None = None,
    existing_base: dict[str, _Module] | None = None,
) -> _Layout:
    """Places `source`'s members where `rev_source`'s cut put the members of the same name.
    `existing` holds the source revision's own copy of each target the cut appended to, and
    `existing_base` the copy before the cut; `rev_old` is the source before the cut, which tells a
    moved base class from a mixin. `if TYPE_CHECKING:` import blocks the cut added are carried over."""
    existing, existing_base = existing or {}, existing_base or {}
    rev_cls, mixins = _split_class_of(rev_source, rev_targets, rev_old)
    mixin_of = {path: name for name, path in mixins.items()}
    cls = source.find_class(rev_cls.name) if rev_cls is not None else None
    if set(existing) & set(mixin_of):
        raise SplitError(f"only a module-only target can be appended to, not {sorted(set(existing) & set(mixin_of))}")
    if rev_old is not None:
        known = rev_old.top_names() | set(mixins)
        stray = [
            f"{path}: {node.name}"
            for path, module in rev_targets.items()
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name not in known
            and (path not in existing_base or node.name not in existing_base[path].top_names())
        ]
        if stray:
            raise SplitError(f"classes the cut added that are not mixins of a split class (an unrecognised base?): {stray}")

    member_home: dict[str, str] = {}
    module_home: dict[str, str] = {}
    for node, _text in _members(rev_source, rev_cls) if rev_cls is not None else []:
        if (name := _member_name(node)) is not None:
            member_home[name] = CORE
    for node in rev_source.body:
        if node is not rev_cls:
            module_home.update(dict.fromkeys(_bound_names(node), CORE))
    for path, module in rev_targets.items():
        mixin = mixin_of.get(path)
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == mixin:
                for member, _text in _members(module, node):
                    if (name := _member_name(member)) is not None:
                        member_home[name] = path
            else:
                module_home.update(dict.fromkeys(_bound_names(node), path))

    for name, path in assign.items():
        home = CORE if path == source.path else path
        if home and home not in rev_targets:
            raise SplitError(f"--assign {name}={path}: not one of {sorted(rev_targets)} or {source.path}")
        member_home[name] = module_home[name] = home

    unknown = []
    owner: list[str] = []
    for node, _text in _members(source, cls) if cls is not None else []:
        name = _member_name(node)
        if name is None:
            owner.append(owner[-1] if owner else CORE)
            continue
        if name not in member_home:
            unknown.append(f"class member {name}")
        owner.append(member_home.get(name, CORE))
    module_owner = []
    for node in source.body:
        bound = _bound_names(node) if node is not cls else []
        assigned = [n for n in bound if n in assign]
        if len({module_home[n] for n in assigned}) > 1:
            raise SplitError(f"--assign puts the names of one tuple assignment in different modules: {assigned}")
        name = assigned[0] if assigned else bound[0] if bound else None
        if name is not None and name not in module_home:
            unknown.append(f"module-level name {name}")
        module_owner.append(module_home.get(name, CORE) if name is not None else CORE)
    _follow_attribute_docs(source.body, module_owner)
    if unknown:
        raise SplitError("not in the reference cut (place each with --assign NAME=MODULE):\n  " + "\n  ".join(unknown))

    layout = _Layout(source, cls, {p: mixin_of.get(p) for p in rev_targets}, owner, module_owner, existing=dict(existing))
    layout.core_head = rev_source.head()
    if rev_cls is not None:
        layout.class_statement = rev_source.class_head(rev_cls, with_doc=False)
    layout.import_bases = {rev_source.path: rev_source, **rev_targets}
    layout.type_checking[source.path] = _type_checking_blocks(rev_source, rev_old, source) if rev_old is not None else ""
    for path, lane_copy in existing.items():
        base = existing_base.get(path)
        heads = {"lane": lane_copy.head(), "cut": rev_targets[path].head(), "base": base.head() if base else None}
        if heads["lane"] == heads["base"] or heads["lane"] == heads["cut"]:
            layout.existing_heads[path] = heads["cut"]
        elif heads["cut"] != heads["base"]:
            raise SplitError(f"{path}: both the cut and the source revision changed its docstring; merge it by hand first")
    for path, module in rev_targets.items():
        layout.type_checking[path] = _type_checking_blocks(module, existing_base.get(path), existing.get(path))
        layout.target_heads[path] = module.head()
        mixin = mixin_of.get(path)
        if mixin is not None:
            layout.mixin_heads[path] = module.class_head(module.find_class(mixin), with_doc=True)
    return layout


def _census(module: _Module, split_names: set[str]) -> tuple[collections.Counter, collections.Counter, collections.Counter, dict]:
    """Class-body statements of the split classes, other module-level statements and non-blank lines, exemptions removed."""
    class_stmts: collections.Counter = collections.Counter()
    module_stmts: collections.Counter = collections.Counter()
    where: dict[str, str] = {}
    exempt = set(range(1, module.doc_end + 1))
    if module.imports:
        exempt.update(range(module.imports[0].lineno, module.imports_end + 1))

    def note(counter: collections.Counter, node: ast.stmt) -> None:
        key = ast.dump(node)
        counter[key] += 1
        where.setdefault(key, f"{module.path}:{node.lineno}: {ast.unparse(node).splitlines()[0][:100]}")

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name in split_names and not node.decorator_list:
            head = module.class_head(node, with_doc=True)
            exempt.update(range(node.lineno, node.lineno + len(head.splitlines())))
            for member, _text in _members(module, node):
                note(class_stmts, member)
        elif _is_type_checking_imports(node):
            exempt.update(range(node.lineno, node.end_lineno + 1))
        else:
            note(module_stmts, node)
    lines = collections.Counter(
        line.rstrip() for number, line in enumerate(module.lines, 1) if number not in exempt and line.strip()
    )
    return class_stmts, module_stmts, lines, where


def verify_texts(old: _Module, new: dict[str, _Module], source_path: str, old_existing: list[_Module] | None = None) -> list[str]:
    """Every difference between the old source (plus the old copies of the targets that were appended to)
    and the new group; empty means pure code motion."""
    targets = {p: m for p, m in new.items() if p != source_path}
    cls, mixins = _split_class_of(new[source_path], targets, old)
    names = {cls.name, *mixins} if cls is not None else set()
    old_class, old_module, old_lines, where = _census(old, names)
    for module in old_existing or []:
        class_stmts, module_stmts, lines, module_where = _census(module, names)
        old_class += class_stmts
        old_module += module_stmts
        old_lines += lines
        for key, value in module_where.items():
            where.setdefault(key, value)
    new_class: collections.Counter = collections.Counter()
    new_module: collections.Counter = collections.Counter()
    new_lines: collections.Counter = collections.Counter()
    for module in new.values():
        class_stmts, module_stmts, lines, module_where = _census(module, names)
        new_class += class_stmts
        new_module += module_stmts
        new_lines += lines
        for key, value in module_where.items():
            where.setdefault(key, value)
    problems = []
    for module in new.values():
        problems.extend(
            f"{module.path}:{node.lineno}: mixin {node.name} has bases, keywords or decorators"
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name in mixins and (node.bases or node.keywords or node.decorator_list)
        )
    if cls is not None:
        old_cls = old.find_class(cls.name)
        own_bases = [ast.dump(b) for b in cls.bases if not (isinstance(b, ast.Name) and b.id in mixins)]
        keywords_changed = [ast.dump(k) for k in cls.keywords] != [ast.dump(k) for k in old_cls.keywords]
        if own_bases != [ast.dump(b) for b in old_cls.bases] or keywords_changed:
            problems.append(f"{source_path}: {cls.name}'s own bases or keywords changed")
        if [ast.dump(d) for d in cls.decorator_list] != [ast.dump(d) for d in old_cls.decorator_list]:
            problems.append(f"{source_path}: {cls.name}'s decorators changed")
    for label, before, after in (("class-body statement", old_class, new_class), ("module-level statement", old_module, new_module)):
        problems += [f"{label} lost x{n}: {where[k]}" for k, n in (before - after).items()]
        problems += [f"{label} added x{n}: {where[k]}" for k, n in (after - before).items()]
    problems += [f"line lost x{n}: {k.strip()[:100]}" for k, n in (old_lines - new_lines).items()]
    problems += [f"line added x{n}: {k.strip()[:100]}" for k, n in (new_lines - old_lines).items()]
    return problems


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _show(rev: str, path: str) -> str:
    return _git("show", f"{rev}:{path}")


def _lines(text: str) -> set[str]:
    return {line for line in text.splitlines() if line}


def _write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _cmd_split(args: argparse.Namespace) -> int:
    def read(path: str) -> str:
        return _show(args.source_rev, path) if args.source_rev else (ROOT / path).read_text(encoding="utf-8")

    if args.table:
        table = json.loads(Path(args.table).read_text(encoding="utf-8"))
        path = table["source"]
        # A target is appended to if the base revision has it (HEAD without --source-rev); one only on disk is a leftover.
        module_only_on_disk = [p for p, spec in table["modules"].items() if spec.get("mixin") is None and (ROOT / p).exists()]
        in_base = args.source_rev or ("HEAD" if module_only_on_disk else None)
        at_rev = _lines(_git("ls-tree", "-r", "--name-only", in_base)) if in_base else set()
        present = [p for p in table["modules"] if p in at_rev and (args.source_rev or (ROOT / p).exists())]
        layout = table_layout(_Module(path, read(path)), table, {p: _Module(p, read(p)) for p in present})
        stale = [p for p in layout.targets if p not in layout.existing and (ROOT / p).exists()]
        if stale:
            raise SplitError(f"target modules already exist: {stale}")
    else:
        rev = args.like
        prev = f"{rev}~1"
        added = _lines(_git("diff", "--name-only", "--diff-filter=A", prev, rev, "--", "*.py"))
        changed = _lines(_git("diff", "--name-only", "--diff-filter=M", prev, rev, "--", "*.py"))
        old_texts = {p: _show(prev, p) for p in changed}
        path = args.source or pick_source({p: _show(rev, p) for p in changed}, added, old_texts)
        rev_source = _Module(path, _show(rev, path))
        rev_old = _Module(path, old_texts.get(path) or _show(prev, path))
        candidates = {p: (old_texts[p], _show(rev, p)) for p in changed if p != path}
        added_texts = {p: _show(rev, p) for p in added}
        appended = existing_targets(rev_source, rev_old, candidates, added_texts)
        target_texts = {**added_texts, **{p: candidates[p][1] for p in appended}}
        load = _target_loader(target_texts)
        rev_targets = {p: load(p) for p in _imported_targets(rev_source, set(target_texts), load)}
        bad = [item for item in args.assign if "=" not in item]
        if bad:
            raise SplitError(f"--assign takes NAME=MODULE, not {bad}")
        assign = dict(item.split("=", 1) for item in args.assign)
        existing = {p: _Module(p, read(p)) for p in appended}
        existing_base = {p: _Module(p, old_texts[p]) for p in appended}
        layout = like_layout(rev_source, rev_targets, _Module(path, read(path)), assign, existing, rev_old, existing_base)
    texts = _render(layout)
    for out_path, out_text in texts.items():
        ast.parse(out_text, filename=out_path)
    missing = sorted(str(Path(p).parent) for p in texts if not (ROOT / p).parent.is_dir())
    if missing:
        raise SplitError(f"no such directory: {missing}")
    for out_path, out_text in texts.items():
        _write(ROOT / out_path, out_text)
        print(f"wrote {out_path} ({len(out_text.splitlines())} lines)")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    base = args.base
    at_base = _lines(_git("ls-tree", "-r", "--name-only", base))
    tracked = _lines(_git("ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"))
    added = {p for p in tracked if p not in at_base and (ROOT / p).is_file()}
    changed = _lines(_git("diff", "--name-only", base, "--", "*.py")) & at_base
    new_texts = {p: (ROOT / p).read_text(encoding="utf-8") for p in changed if (ROOT / p).is_file()}
    old_texts = {p: _show(base, p) for p in new_texts}
    path = args.source or pick_source(new_texts, added, old_texts)
    source = _Module(path, (ROOT / path).read_text(encoding="utf-8"))
    old_source = _Module(path, _show(base, path))
    added_texts = {p: (ROOT / p).read_text(encoding="utf-8") for p in added}
    appended = existing_targets(source, old_source, {p: (old_texts[p], new_texts[p]) for p in new_texts if p != path}, added_texts)
    target_texts = {**added_texts, **{p: new_texts[p] for p in appended}}
    load = _target_loader(target_texts)
    group = {path: source}
    for target in _imported_targets(source, set(target_texts), load):
        group[target] = load(target)
    problems = verify_texts(old_source, group, path, [_Module(p, old_texts[p]) for p in appended])
    print(f"verify: {path} against {base}, with {', '.join(sorted(set(group) - {path}))}")
    if appended:
        print(f"verify: appended to {', '.join(appended)}, whose {base} content counts on the old side")
    for problem in problems:
        print("  " + problem)
    print("verify: zero differences" if not problems else f"verify: {len(problems)} difference(s)")
    return 1 if problems else 0


def surface(cls: type) -> list[tuple[str, str]]:
    out = []
    for name in dir(cls):
        try:
            kind = type(inspect.getattr_static(cls, name)).__name__
        except AttributeError:
            kind = "<dynamic>"
        out.append((name, kind))
    return sorted(out)


def _cmd_surface(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    module_name, _, class_name = args.target.partition(":")
    for name, kind in surface(getattr(importlib.import_module(module_name), class_name)):
        print(name, kind)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split", help="cut by a table, or replay a cut onto an uncut source")
    how = split.add_mutually_exclusive_group(required=True)
    how.add_argument("--table", help="the cut table (JSON)")
    how.add_argument("--like", metavar="REV", help="replay the cut commit REV")
    split.add_argument("--source-rev", metavar="REV", help="read the source from git at REV instead of the working file")
    split.add_argument("--source", help="with --like: the split's source path, when it cannot be inferred")
    split.add_argument("--assign", action="append", default=[], metavar="NAME=MODULE", help="with --like: place a name REV lacks")
    verify = sub.add_parser("verify", help="prove the working tree's split is code motion from REV")
    verify.add_argument("--base", required=True, metavar="REV")
    verify.add_argument("--source", help="the split's source path, when it cannot be inferred")
    surf = sub.add_parser("surface", help="print a class's names and static types")
    surf.add_argument("target", help="package.module:Class")
    args = parser.parse_args(argv)
    try:
        return {"split": _cmd_split, "verify": _cmd_verify, "surface": _cmd_surface}[args.command](args)
    except SplitError as error:
        print(f"split_class: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
