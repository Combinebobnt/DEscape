"""The scan behind each module family's guard test (`testkit.module_family`).

A test or tool that writes to a family module's dict directly reaches only
that one module, so after a split it can miss the reader it meant to patch
and still pass. `scan_source()` finds each raw form of module-dict access
through an alias of a family module:

- `setattr`, `getattr`, `hasattr`, `delattr` (plain or `monkeypatch.`) on the
  alias, `patch.object`/`patch.multiple` on it, `vars()` of it, its `__dict__`;
- an assignment to or `del` of `alias.X`;
- a string that starts with `"<module>."`, the `monkeypatch.setattr` target
  form;
- a read of `alias.X`, or a `from <module> import X`, where the module does
  not define `X` at top level (a name it only imports);
- with `bare_values`, any other load of the alias: a module held in a tuple,
  passed as an argument, returned or assigned is one the scan cannot follow.

An alias is a name bound by `import pkg.mod as X`, `from pkg import mod [as
X]`, `importlib.import_module(...)` or `sys.modules[...]` of a family module,
or assigned from another alias, plus the dotted `pkg.mod` itself. With a
`package`, relative imports resolve against it. A module reached some other
way is passed in `extra_aliases`. Not followed: an alias bound by unpacking,
a loop, `with ... as` or a walrus; one imported under `global` in another
function; a comprehension variable of the alias's name, which hides it for
the whole function; a family module reached as an attribute of its package
(`descape.render` after `import descape as d`); a target string built at run
time.

`reads_only` keeps only the read rule, for production code (`descape/`),
which may import and call a family module but has no reason to patch one.

`alias_scopes()` is the alias resolution on its own, shared with
`tools/repoint_names.py`, so the guard and the repoint see the same reads.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ATTR_FUNCS = frozenset({"setattr", "getattr", "hasattr", "delattr", "__setattr__", "__delattr__"})
PATCH_FUNCS = frozenset({"object", "multiple"})
MODULE_DUNDERS = frozenset({"__file__", "__name__", "__spec__"})
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
VIEWER_FIX = "set it with testkit.viewer_family.set_viewer_global(), read it with viewer_global()"

Fix = str | Callable[[str], str]


@functools.cache
def module_defines(module_name: str) -> frozenset[str]:
    """The names `module_name` defines at top level, by its source: not the names it only imports."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin, module_name
    return defined_names(Path(spec.origin).read_text(encoding="utf-8"))


def defined_names(source: str) -> frozenset[str]:
    """The names the module `source` defines at top level, inside `if`, `try` and `with` blocks too."""
    names: set[str] = set()

    def visit(statements) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for block in ("body", "orelse", "finalbody"):
                    visit(getattr(node, block, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(ast.parse(source).body)
    return frozenset(names)


def dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def from_module(node: ast.ImportFrom, package: str | None) -> str | None:
    """The absolute module an ImportFrom names, or None for a relative one with no `package`."""
    if not node.level:
        return node.module
    if package is None:
        return None
    parts = package.split(".")
    if node.level > 1:
        parts = parts[: -(node.level - 1)]
    return ".".join(parts + ([node.module] if node.module else []))


def family_module(node: ast.AST, aliases: dict[str, str], family: tuple[str, ...]) -> str | None:
    """The family module `node` evaluates to, if it is one the scan can see."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    dotted = dotted_name(node)
    if dotted in family:
        return dotted
    if isinstance(node, ast.Call) and node.args:
        func = (dotted_name(node.func) or "").rsplit(".", 1)[-1]
        first = node.args[0]
        if func in {"import_module", "__import__"} and isinstance(first, ast.Constant) and first.value in family:
            return first.value
        if func == "reload":
            return family_module(first, aliases, family)
        if dotted_name(node.func) == "sys.modules.get" and isinstance(first, ast.Constant) and first.value in family:
            return first.value
    if isinstance(node, ast.Subscript) and dotted_name(node.value) == "sys.modules":
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in family:
            return key.value
    return None


def _own_nodes(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """The nodes evaluated in `scope` itself, and the scopes nested directly in it. A nested function's
    decorators, defaults and annotations belong to `scope`, as do a nested class's decorators and bases."""
    own: list[ast.AST] = []
    nested: list[ast.AST] = []
    stack: list[ast.AST] = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            nested.append(node)
            own.append(node)
            stack.extend(getattr(node, "decorator_list", []))
            if not isinstance(node, ast.ClassDef):
                args = node.args
                stack.extend(args.defaults + [d for d in args.kw_defaults if d is not None])
                every = [*args.posonlyargs, *args.args, *args.kwonlyargs, *filter(None, (args.vararg, args.kwarg))]
                stack.extend(a.annotation for a in every if a.annotation is not None)
                stack.extend([node.returns] if getattr(node, "returns", None) is not None else [])
            else:
                stack.extend(node.bases + [k.value for k in node.keywords])
            continue
        own.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return own, nested


def _bound_names(scope: ast.AST, own: list[ast.AST]) -> set[str]:
    names: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        names.update(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)
        names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    for node in own:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _SCOPES) and not isinstance(node, ast.Lambda) and node is not scope:
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _fix_for(fix: Fix, module_name: str) -> str:
    return fix if isinstance(fix, str) else fix(module_name)


def _string_findings(tree: ast.AST, family: tuple[str, ...], fix: Fix) -> list[tuple[int, str]]:
    bare = {id(s.value) for s in ast.walk(tree) if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
            text = node.values[0].value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare:
            text = node.value
        else:
            continue
        for module_name in family:
            if isinstance(text, str) and text.startswith(module_name + "."):
                found.append((node.lineno, f"string target {text!r}: {_fix_for(fix, module_name)}"))
                break
    return found


@dataclass
class AliasScope:
    """One scope of a module, as `alias_scopes()` sees it."""

    node: ast.AST  # the module, function, lambda or class
    own: list[ast.AST]  # the nodes evaluated in it, not in a nested scope
    visible: dict[str, str]  # alias name -> the family module it is bound to here
    binders: dict[str, list[ast.stmt]]  # alias name -> the statements that bound it, here or in an enclosing scope


def alias_scopes(
    tree: ast.Module, family: tuple[str, ...], extra_aliases: dict[str, str] | None = None, *, package: str | None = None
) -> list[AliasScope]:
    """Every scope of `tree` with the family aliases visible in it, by the rules in this module's docstring.
    An alias from `extra_aliases` has no binder."""
    extra = dict(extra_aliases or {})
    out: list[AliasScope] = []

    def walk(scope: ast.AST, inherited: dict[str, str], inherited_binders: dict[str, list[ast.stmt]]) -> None:
        own, nested = _own_nodes(scope)
        bound = _bound_names(scope, own)
        here: dict[str, str] = {}
        binders: dict[str, list[ast.stmt]] = {}
        for node in own:
            if isinstance(node, ast.Import):
                pairs = [(a.asname, a.name) for a in node.names if a.asname and a.name in family]
            elif isinstance(node, ast.ImportFrom) and (base := from_module(node, package)):
                pairs = [(a.asname or a.name, f"{base}.{a.name}") for a in node.names if f"{base}.{a.name}" in family]
            else:
                continue
            for name, module_name in pairs:
                here[name] = module_name
                binders.setdefault(name, []).append(node)
        changed = True
        while changed:
            visible = {k: v for k, v in inherited.items() if k not in bound} | here | extra
            changed = False
            for node in own:
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    module_name = family_module(node.value, visible, family)
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if module_name and isinstance(target, ast.Name) and here.get(target.id) != module_name:
                            here[target.id] = module_name
                            binders.setdefault(target.id, []).append(node)
                            changed = True
        seen_binders = {k: v for k, v in inherited_binders.items() if k not in bound} | binders
        out.append(AliasScope(scope, own, visible, seen_binders))
        # A class body's own names are not visible inside its methods.
        if isinstance(scope, ast.ClassDef):
            seen_by_children, binders_by_children = inherited | extra, inherited_binders
        else:
            seen_by_children, binders_by_children = visible, seen_binders
        for child in nested:
            walk(child, seen_by_children, binders_by_children)

    walk(tree, {}, {})
    return out


def scan_source(
    source: str,
    family: tuple[str, ...],
    defines: Callable[[str], frozenset[str]],
    extra_aliases: dict[str, str] | None = None,
    *,
    fix: Fix = VIEWER_FIX,
    package: str | None = None,
    reads_only: bool = False,
    bare_values: bool = False,
    bare_fix: str = "",
) -> list[tuple[int, str]]:
    """Every raw module-dict access through a family alias in `source`, as (line, message)."""
    tree = ast.parse(source)
    found = [] if reads_only else _string_findings(tree, family, fix)
    # A load that is the object of an attribute read is how a module is meant to be used.
    attribute_objects = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    reported: set[int] = set()

    for scope in alias_scopes(tree, family, extra_aliases, package=package):
        own, visible = scope.own, scope.visible
        for node in own:
            if isinstance(node, ast.Call) and not reads_only:
                func = (dotted_name(node.func) or "").rsplit(".", 1)[-1]
                first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "target"), None)
                alias = family_module(first, visible, family) if first is not None else None
                if alias is None and func in PATCH_FUNCS and isinstance(first, ast.Constant) and first.value in family:
                    alias = first.value
                if alias and func in ATTR_FUNCS | PATCH_FUNCS | {"vars"}:
                    found.append((node.lineno, f"{func}() on {alias}: {_fix_for(fix, alias)}"))
                    reported.add(id(first))
            elif isinstance(node, ast.ImportFrom):
                base = from_module(node, package)
                if base in family:
                    found.extend(
                        (node.lineno, f"import of {base}.{a.name}, a name it only imports: import it from its own module")
                        for a in node.names
                        if a.name != "*" and a.name not in defines(base)
                    )
            elif isinstance(node, ast.Attribute):
                module_name = family_module(node.value, visible, family)
                if module_name is None:
                    continue
                if node.attr == "__dict__" and not reads_only:
                    found.append((node.lineno, f"__dict__ of {module_name}: {_fix_for(fix, module_name)}"))
                elif isinstance(node.ctx, (ast.Store, ast.Del)):
                    if not reads_only:
                        found.append((node.lineno, f"assignment to {module_name}.{node.attr}: {_fix_for(fix, module_name)}"))
                elif node.attr not in defines(module_name) and node.attr not in MODULE_DUNDERS:
                    found.append(
                        (node.lineno, f"read of {module_name}.{node.attr}, a name it only imports: import it from its own module")
                    )
        if bare_values and not reads_only:
            for node in own:
                if id(node) in attribute_objects or id(node) in reported:
                    continue
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    module_name = visible.get(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    module_name = dotted_name(node) if dotted_name(node) in family else None
                else:
                    continue
                if module_name:
                    found.append((node.lineno, f"{module_name} used as a value, which the scan cannot follow: {bare_fix}"))
    return sorted(set(found))


def python_files(root: Path, dirs: Iterable[str], exclude: Iterable[Path] = ()) -> list[Path]:
    """Every Python file under `root`/each of `dirs`, outside `__pycache__` and `exclude`."""
    skipped = set(exclude)
    files: list[Path] = []
    for name in dirs:
        files += [p for p in sorted((root / name).rglob("*.py")) if "__pycache__" not in p.parts and p not in skipped]
    return files


def package_of(root: Path, path: Path) -> str | None:
    """The dotted package a file under `root` sits in, if its directory is one (has `__init__.py`)."""
    parent = path.parent
    if not (parent / "__init__.py").is_file():
        return None
    return ".".join(parent.relative_to(root).parts)


def scan_files(
    root: Path,
    files: Iterable[Path],
    family: tuple[str, ...],
    defines: Callable[[str], frozenset[str]] = module_defines,
    hand_listed: dict[str, dict[str, str]] | None = None,
    **options,
) -> list[str]:
    """`scan_source()` over `files`, as `rel:line: message` lines."""
    problems = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        extra = (hand_listed or {}).get(rel)
        for line, message in scan_source(source, family, defines, extra, package=package_of(root, path), **options):
            problems.append(f"{rel}:{line}: {message}")
    return problems
