"""Repoints the callers of names a split moved out of a module, and proves the repoint changed qualifiers only.

Usage, from the repo root:

    python3 tools/repoint_names.py --base REV --module descape.render [--skip PATH ...] [PATH ...]
    python3 tools/repoint_names.py --check --base REV --module descape.render [--skip PATH ...] [--allow PATH ...] [PATH ...]

The moved map: every top-level name the module's file defined at REV and no
longer defines, mapped to the one sibling module (a `.py` file in the same
directory) that defines it now and did not at REV. A name with no such home,
or with two, fails the run before anything is written.

The rewrite runs over descape/, tests/, tools/, testkit/ and native/, or the
files and directories given, edits by token so nothing else in a file
changes, and finds aliases by the rules of `testkit.module_family_scan`:

1. `alias.X`, where `alias` is bound to the module and X moved: the
   qualifier becomes the home's last name (`render_units.X`), and an import
   of the home goes directly after each statement that bound the alias, at
   its indentation and in its form (`from descape import render_units`,
   `from . import render_units`, `import descape.render_units as
   render_units`). A read inside an f-string is found by its AST position
   like any other. No import is added where the home is already imported.
2. `from descape.render import X, Y` and the relative `from .render import
   X`: split into one statement per home, with the kept names staying.

Then `ruff check --select F401,I001 --fix` over the files it changed drops
each import that went unused and sorts.

A file is left as it is and listed, for a hand edit and a second run, when
it has: a star import from the module; a moved name read through anything
but an import alias (`descape.render.X`, an alias bound by assignment or
`import_module()`); the alias used as a value rather than the object of an
attribute read (`getattr(render, name)`, `for m in (render, x)`); a write
or `del` of a moved name through it; a home's name already bound to
something else; an import it must split or follow that shares its line with
another statement, or one it must split with a comment inside; or a string,
in a scope where the alias is bound, other than a bare
string statement such as a docstring, naming `alias.X` or `module.X` for a
moved X (a quoted annotation never runs, so it never fails). A shadowing
parameter or local is not the module. A second run changes nothing, which
is what makes this the port tool for a lane or a pull request cut before
the split.

`--check` is the repoint proof. No file in scope would change, or is
refused outside `--allow`; and every file that differs from REV (tracked,
or a new `.py`), outside the split itself (the module's file and the
homes) and `--allow`, and inside the PATHs if any are given, is Python
whose AST at REV and now is equal once both sides drop their imports of the
module and the homes as modules, read `from <any of them> import a, b` as
one import of the same names from a placeholder module (merged within each
run of imports, as a split by home and an import sort leave them), and
reduce each attribute read through an alias of one to a placeholder
`<family>.X` that no bare name equals. So those files changed module
qualifiers and nothing else: a read made bare, or a `from` import that now
binds another name, still differs.

A read of a home's alias counts as imported only where an unconditional
import of the home (a statement of its scope's body, not one under `if
TYPE_CHECKING:` or `try`) binds it in the reading scope, above the read, or
in an enclosing one. Otherwise the import is added beside the alias's
binding as above, and a file whose reading scope imports the home only
below the read or under a condition, where the added import would not
reach, is refused. A `match` capture of a home's name is a binding of it.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testkit.module_family_scan import AliasScope, alias_scopes, defined_names, family_module, from_module, package_of

DEFAULT_DIRS = ("descape", "tests", "tools", "testkit", "native")


class RepointError(Exception):
    pass


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


def _module_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _short(module: str) -> str:
    return module.rpartition(".")[2]


def moved_map(module: str, base: str) -> dict[str, str]:
    """Each name `module` defined at `base` and defines no longer -> the one sibling module that took it."""
    path = _module_path(module)
    moved = defined_names(_git("show", f"{base}:{path}")) - defined_names((ROOT / path).read_text(encoding="utf-8"))
    folder, package = path.rpartition("/")[0], module.rpartition(".")[0]
    at_base = set(_lines(_git("ls-tree", "--name-only", base, f"{folder}/")))
    homes: dict[str, list[str]] = {}
    for file in sorted((ROOT / folder).glob("*.py")):
        rel = file.relative_to(ROOT).as_posix()
        if rel == path:
            continue
        gained = defined_names(file.read_text(encoding="utf-8")) & moved
        if gained and rel in at_base:
            gained -= defined_names(_git("show", f"{base}:{rel}"))
        for name in gained:
            homes.setdefault(name, []).append(f"{package}.{file.stem}")
    problems = [f"{name}: no module defines it now" for name in sorted(moved - set(homes))]
    problems += [f"{name}: defined now by {' and '.join(found)}" for name, found in sorted(homes.items()) if len(found) > 1]
    if problems:
        raise RepointError("moved names without exactly one home:\n  " + "\n  ".join(problems))
    return {name: found[0] for name, found in homes.items()}


@dataclass
class Repoint:
    """One file's rewrite: the new text (the old one when refused or unchanged) and why it was refused."""

    text: str
    refusals: list[str] = field(default_factory=list)
    reads: int = 0
    imports: int = 0


def _binding_refusal(tree: ast.Module, name: str, home: str, package: str | None) -> int | None:
    """The line of a binding of `name` in `tree` that is not an import of `home`, if there is one."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound == name and not (alias.asname and alias.name == home):
                    return node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = from_module(node, package)
            if any((a.asname or a.name) == name and f"{base}.{a.name}" != home for a in node.names):
                return node.lineno
        elif name in _other_bindings(node):
            return node.lineno
    return None


def _other_bindings(node: ast.AST) -> list[str]:
    """The names a node that is not an import binds or declares."""
    if isinstance(node, ast.Name):
        return [] if isinstance(node.ctx, ast.Load) else [node.id]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler):
        return [node.name] if node.name else []
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    return list(node.names) if isinstance(node, (ast.Global, ast.Nonlocal)) else []


def repoint_source(source: str, module: str, moved: dict[str, str], package: str | None = None) -> Repoint:
    """`source` with every read and import of a moved name repointed to its home (`moved`: name -> home module).
    `package` resolves relative imports. Any refusal leaves the text as it is."""
    tree = ast.parse(source)
    family = (module, *sorted(set(moved.values())))
    data = source.encode("utf-8")
    lines = data.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    result = Repoint(source)
    edits: list[tuple[int, int, str, str | None]] = []  # (start, end, replacement, the text it must replace)
    after: dict[int, tuple[ast.stmt, set[str]]] = {}  # id(binder) -> (binder, homes to import after it)
    homes_used: set[str] = set()
    attribute_objects = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    def offset(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    def span(node: ast.AST) -> tuple[int, int]:
        return offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)

    def refuse(node: ast.AST, why: str) -> None:
        result.refusals.append(f"{node.lineno}: {why}")

    def alone(node: ast.stmt) -> bool:
        before = lines[node.lineno - 1][: node.col_offset]
        rest = lines[node.end_lineno - 1][node.end_col_offset :].strip()
        return not before.strip() and (not rest or rest.startswith(b"#"))

    def editable_import(node: ast.stmt, rewritten: bool) -> bool:
        """Whether an import can be followed by a new line or, if `rewritten`, replaced; refuses it if not."""
        if not alone(node):
            refuse(node, "an import that shares its line with another statement")
            return False
        start, end = span(node)
        if rewritten and b"#" in data[start:end]:
            refuse(node, "an import with a comment inside")
            return False
        return True

    scopes = alias_scopes(tree, family, package=package)
    owner = {id(n): s for s in scopes for n in s.own}
    scope_ids = {id(s.node) for s in scopes}
    parent = {id(n): s for s in scopes for n in s.own if id(n) in scope_ids}

    def home_reach(scope: AliasScope, read: ast.AST, home: str, binder: ast.stmt) -> str | None:
        """None if `home` is imported for `read` already, "add" to import it beside `binder`, else a refusal.
        Imported already: by a statement of its scope's body, and in the read's own scope, above the read."""
        short = _short(home)
        path = [scope]
        while id(path[-1].node) in parent:
            path.append(parent[id(path[-1].node)])
        path = [path[0], *(s for s in path[1:] if not isinstance(s.node, ast.ClassDef))]  # a class's names are not seen inside it
        def binds_short(n: ast.AST) -> bool:
            return isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split(".")[0]) == short for a in n.names)

        imports = [[n for n in s.own if binds_short(n)] for s in path]
        depth = next((d for d, binds in enumerate(imports) if binds), None)
        if depth is None:
            return "add"
        binds, outer = imports[depth], path[depth]
        body = outer.node.body if isinstance(getattr(outer.node, "body", None), list) else []
        if any(n in body and (depth or (n.lineno, n.col_offset) < (read.lineno, read.col_offset)) for n in binds):
            return None
        if depth >= next((i for i, s in enumerate(path) if s is owner.get(id(binder))), len(path)):
            return "add"
        return f"{short} is imported in this scope only after the read or under a condition: import it above the read by hand"

    for scope in scopes:
        for node in scope.own:
            if isinstance(node, ast.Attribute) and node.attr in moved and family_module(node.value, scope.visible, family) == module:
                home = moved[node.attr]
                alias = node.value.id if isinstance(node.value, ast.Name) else None
                binders = scope.binders.get(alias, []) if alias else []
                if not isinstance(node.ctx, ast.Load):
                    refuse(node, f"a write or del of {module}.{node.attr}")
                elif alias is None or not binders or not all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in binders):
                    refuse(node, f"{module}.{node.attr} read through {ast.unparse(node.value)}, not an import alias")
                else:
                    value = node.value
                    edits.append((*span(value), _short(home), alias))
                    homes_used.add(home)
                    result.reads += 1
                    reach = home_reach(scope, node, home, binders[0])
                    if reach == "add":
                        for binder in binders:
                            after.setdefault(id(binder), (binder, set()))[1].add(home)
                    elif reach is not None:
                        refuse(node, reach)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and scope.visible.get(node.id) == module:
                if id(node) not in attribute_objects:
                    refuse(node, f"{node.id} ({module}) used as a value")
            elif isinstance(node, ast.ImportFrom) and from_module(node, package) == module:
                if any(a.name == "*" for a in node.names):
                    refuse(node, f"a star import from {module}")
                    continue
                by_home: dict[str, list[ast.alias]] = {}
                for a in node.names:
                    by_home.setdefault(moved.get(a.name, module), []).append(a)
                if set(by_home) == {module} or not editable_import(node, rewritten=True):
                    continue
                parts = (node.module or "").split(".")
                statements = [
                    f"from {'.' * node.level}{'.'.join([*parts[:-1], _short(home)])} import "
                    + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in names)
                    for home, names in sorted(by_home.items(), key=lambda item: (item[0] != module, item[0]))
                ]
                indent = lines[node.lineno - 1][: node.col_offset].decode()
                edits.append((*span(node), f"\n{indent}".join(statements), None))
                result.imports += len(statements) - (module in by_home)

    for home in sorted(homes_used):
        line = _binding_refusal(tree, _short(home), home, package)
        if line is not None:
            result.refusals.append(f"{line}: {_short(home)} is already bound to something other than {home}")
    for binder, homes in after.values():
        if not editable_import(binder, rewritten=False):
            continue
        if isinstance(binder, ast.Import):
            new = [f"import {home} as {_short(home)}" for home in sorted(homes)]
        else:
            new = [f"from {'.' * binder.level}{binder.module or ''} import {_short(home)}" for home in sorted(homes)]
        indent = lines[binder.lineno - 1][: binder.col_offset].decode()
        position = starts[binder.end_lineno]
        text = "".join(f"{indent}{statement}\n" for statement in new)
        edits.append((position, position, text if data.endswith(b"\n") or position < len(data) else "\n" + text, None))
        result.imports += len(new)

    bare = {id(s.value) for s in ast.walk(tree) if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)}
    for scope in scopes:
        aliases = sorted({name for name, value in scope.visible.items() if value == module} | {module})
        named = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, aliases)) + r")\.(\w+)")
        for node in scope.own:
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare:
                hits = sorted({m.group(0) for m in named.finditer(node.value) if m.group(1) in moved})
                if hits:
                    refuse(node, f"a string naming {', '.join(hits)}, which never runs: edit it by hand")

    if result.refusals:
        result.refusals.sort(key=lambda r: int(r.split(":", 1)[0]))
        result.reads = result.imports = 0
        return result
    out = bytearray(data)
    for start, end, replacement, expected in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        if expected is not None and out[start:end].decode() != expected:
            raise RepointError(f"AST position {start}..{end} holds {out[start:end].decode()!r}, not {expected!r}")
        out[start:end] = replacement.encode()
    result.text = out.decode("utf-8")
    ast.parse(result.text)
    return result


FAMILY = "<family>"  # the module a normalized family import or attribute read names; no source spells it


def _merged_imports(run: list[ast.stmt]) -> list[ast.stmt]:
    """A run of import statements with its family `from` imports as one, names sorted, unless another import in the
    run binds one of their names too (then their order matters)."""
    family = [s for s in run if isinstance(s, ast.ImportFrom) and s.module == FAMILY]
    others = [s for s in run if s not in family]
    bound = {a.asname or a.name for s in family for a in s.names}
    if not family or bound & {a.asname or a.name.split(".")[0] for s in others for a in s.names}:
        return run
    names = sorted((a for s in family for a in s.names), key=lambda a: (a.name, a.asname or ""))
    return [ast.copy_location(ast.ImportFrom(module=FAMILY, names=names, level=0), family[0]), *others]


def _normalized(source: str, family: tuple[str, ...], package: str | None) -> list[tuple[int, str]]:
    """Each top-level statement of `source` as (line, dump), with the family's module imports dropped, each `from`
    import of a family module made one of `FAMILY` (merged per run of imports), and every attribute read through a
    family alias reduced to `FAMILY.X`, which no bare name equals."""
    tree = ast.parse(source)
    reduce = {
        id(node)
        for scope in alias_scopes(tree, family, package=package)
        for node in scope.own
        if isinstance(node, ast.Attribute) and family_module(node.value, scope.visible, family)
    }

    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            self.generic_visit(node)
            return ast.copy_location(ast.Name(id=f"{FAMILY}.{node.attr}", ctx=node.ctx), node) if id(node) in reduce else node

        def visit_Import(self, node: ast.Import) -> ast.AST | None:
            node.names = [a for a in node.names if a.name not in family]
            return node if node.names else None

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
            base = from_module(node, package)
            if base in family:
                node.module, node.level = FAMILY, 0
                return node
            if base:
                node.names = [a for a in node.names if f"{base}.{a.name}" not in family]
            return node if node.names else None

    tree = Normalize().visit(tree)
    for node in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if not isinstance(block, list):
                continue
            out: list[ast.stmt] = []
            run: list[ast.stmt] = []
            for statement in [*block, None]:
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    run.append(statement)
                    continue
                out += _merged_imports(run)
                run = []
                if statement is not None:
                    out.append(statement)
            block[:] = out
    return [(getattr(node, "lineno", 0), ast.dump(node)) for node in tree.body]


def proof_problems(old: str, new: str, family: tuple[str, ...], package: str | None) -> str | None:
    """Where `new` differs from `old` other than by the family's qualifiers and imports, or None."""
    before, after = _normalized(old, family, package), _normalized(new, family, package)
    for (_line, a), (line, b) in zip(before, after, strict=False):
        if a != b:
            return f"line {line}: differs beyond module qualifiers"
    if len(before) != len(after):
        return f"{len(after) - len(before):+d} top-level statements"
    return None


def _scope_files(paths: list[str], skip: set[str]) -> list[Path]:
    files: set[Path] = set()
    for name in paths or DEFAULT_DIRS:
        target = ROOT / name
        if target.is_file():
            files.add(target)
        elif target.is_dir():
            files.update(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        elif paths:
            raise RepointError(f"no such file or directory: {name}")
    return sorted(p for p in files if p.relative_to(ROOT).as_posix() not in skip)


def _write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _ruff() -> str:
    beside = [ROOT / ".venv" / "bin" / "ruff", Path(sys.executable).parent / "ruff"]
    found = next((str(p) for p in beside if p.exists()), None) or shutil.which("ruff")
    if found is None:
        raise RepointError("ruff is not installed (requirements-dev.txt)")
    return found


def _repoint_all(args: argparse.Namespace, moved: dict[str, str]) -> dict[str, tuple[str, Repoint]]:
    """rel path -> (old text, its Repoint), for every file in scope that names the module's last part."""
    out = {}
    needle = _short(args.module)
    for path in _scope_files(args.paths, set(args.skip)):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            out[path.relative_to(ROOT).as_posix()] = (text, repoint_source(text, args.module, moved, package_of(ROOT, path)))
    return out


def _print_refusals(results: dict[str, tuple[str, Repoint]], allow: set[str]) -> int:
    count = 0
    for rel, (_text, result) in sorted(results.items()):
        if rel in allow:
            continue
        for refusal in result.refusals:
            print(f"  {rel}:{refusal}")
            count += 1
    return count


def _cmd_repoint(args: argparse.Namespace, moved: dict[str, str]) -> int:
    results = _repoint_all(args, moved)
    changed = [rel for rel, (text, result) in sorted(results.items()) if result.text != text]
    for rel in changed:
        _write(ROOT / rel, results[rel][1].text)
    if changed:
        ruff = subprocess.run([_ruff(), "check", "--select", "F401,I001", "--fix", "--quiet", *changed], cwd=ROOT, check=False)
        if ruff.returncode:
            raise RepointError("ruff could not fix every F401/I001 finding in the files above")
    reads = sum(results[rel][1].reads for rel in changed)
    imports = sum(results[rel][1].imports for rel in changed)
    print(f"repoint_names: {len(moved)} moved names; {reads} reads and {imports} import statements repointed in {len(changed)} files")
    refused = [rel for rel, (_text, result) in results.items() if result.refusals]
    if refused:
        print(f"repoint_names: {len(refused)} files refused, left as they are; edit by hand, then run again:")
        _print_refusals(results, set())
    return 1 if refused else 0


def _cmd_check(args: argparse.Namespace, moved: dict[str, str]) -> int:
    allow = set(args.allow)
    homes = sorted(set(moved.values()))
    family = (args.module, *homes)
    problems = []
    results = _repoint_all(args, moved)
    problems += [f"{rel}: the repoint would change it" for rel, (text, result) in sorted(results.items()) if result.text != text]
    if _print_refusals(results, allow):
        problems.append("refused files outside --allow (listed above)")
    split = {_module_path(m) for m in family}
    differing = set(_lines(_git("diff", "--name-only", args.base)))
    differing |= set(_lines(_git("ls-files", "--others", "--exclude-standard", "--", "*.py")))
    in_scope = {p.relative_to(ROOT).as_posix() for p in _scope_files(args.paths, set())} if args.paths else None
    at_base = set(_lines(_git("ls-tree", "-r", "--name-only", args.base)))
    proved = 0
    for rel in sorted(differing - split - allow):
        if in_scope is not None and rel not in in_scope:
            continue
        if not rel.endswith(".py") or not (ROOT / rel).is_file() or rel not in at_base:
            problems.append(f"{rel}: differs from {args.base} and is not a changed Python file (--allow it after reading it)")
            continue
        old, new = _git("show", f"{args.base}:{rel}"), (ROOT / rel).read_text(encoding="utf-8")
        found = proof_problems(old, new, family, package_of(ROOT, ROOT / rel))
        if found:
            problems.append(f"{rel}: {found}")
        proved += 1
    print(f"repoint_names --check: {proved} files changed qualifiers only, against {args.base}; split: {', '.join(sorted(split))}")
    for problem in problems:
        print(f"  {problem}")
    print("repoint_names --check: zero differences" if not problems else f"repoint_names --check: {len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, metavar="REV", help="the commit before the split")
    parser.add_argument("--module", required=True, help="the split's source module, e.g. descape.render")
    parser.add_argument("--check", action="store_true", help="prove the repoint instead of making it")
    parser.add_argument("--skip", action="append", default=[], metavar="PATH", help="a file the rewrite leaves alone")
    parser.add_argument("--allow", action="append", default=[], metavar="PATH", help="with --check: a hand-edited file")
    parser.add_argument("paths", nargs="*", metavar="PATH", help=f"files or directories (default: {' '.join(DEFAULT_DIRS)})")
    args = parser.parse_args(argv)
    try:
        moved = moved_map(args.module, args.base)
        return (_cmd_check if args.check else _cmd_repoint)(args, moved)
    except (RepointError, subprocess.CalledProcessError) as error:
        print(f"repoint_names: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
