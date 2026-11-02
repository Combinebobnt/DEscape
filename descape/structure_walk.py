"""Reads a scenario file's bytes against an AoE2ScenarioParser-shaped
structure dict, without the library, and records where every field sits.

Why DEscape owns a walker: inferring a new scenario version's layout
(tools/infer_structure.py) tries many candidate structures over the same
bytes, and a misaligned candidate must fail at a named field in milliseconds.
The library reads a garbage repeat count first and fails later, after
building millions of objects (a 2^31 `map_width * map_height` builds
TerrainStructs until memory runs out). This walker checks every repeat count
and string length against the bytes left before reading anything, and keeps
per-field offsets, which is what localizing a layout change needs.

Reading rules follow the library (sections/aoe2_file_section.py,
helper/bytes_parser.py, sections/retrievers/retriever.py,
sections/dependencies/): static `repeat`, `on_construct` SET_REPEAT and
REFRESH_SELF (applying `on_refresh` SET_REPEAT), the "value or list" rule
with `is_list` (implicitly true under any SET_REPEAT), a negative string
length reading as empty, a negative repeat reading nothing, nested structs
resolved only in the enclosing struct's own `structs`, and a short
`__END_OF_FILE_MARK__` being the normal end of the file.
tests/test_structure_walk.py pins this module to the library field by field.

No AoE2ScenarioParser import, on purpose: the in-app compat view (TASK-057)
can use it in the frozen build. Dependency `eval`s run through a small AST
interpreter with a whitelist, never Python's eval().
"""

from __future__ import annotations

import ast
import operator
import struct
import zlib
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

EOF_MARK = "__END_OF_FILE_MARK__"
HEADER_SECTION = "FileHeader"

DEFAULT_READ_BUDGET = 20_000_000  # field reads plus struct instances; the heaviest corpus file (100k effects) needs ~10M
DEFAULT_STRING_CAP = 1 << 26  # 64 MiB; the largest real string (an XS script) is far below this
DEFAULT_DEPTH_LIMIT = 16  # real structures nest 3 deep (Triggers > TriggerStruct > EffectStruct)
DEFAULT_BODY_CAP = 1 << 30  # decompressed body size cap for walk_file()

_INT_FORMATS = {("u", 1): "B", ("s", 1): "b", ("u", 2): "H", ("s", 2): "h", ("u", 4): "I", ("s", 4): "i", ("u", 8): "Q", ("s", 8): "q"}
_FLOAT_FORMATS = {4: "f", 8: "d"}


class StructureError(ValueError):
    """The structure dict itself uses something the walker can't read."""


class WalkError(Exception):
    """The bytes don't fit the structure at `path` (a dotted field path with
    instance indices), reading at `offset` of the walked buffer."""

    def __init__(self, path: str, offset: int, reason: str) -> None:
        super().__init__(f"{path} @ {offset}: {reason}")
        self.path = path
        self.offset = offset
        self.reason = reason
        self.partial: Walk | None = None  # what walk() read before the failure


# --- dependency evals -------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_FUNCS = {"round": round, "type": type, "len": len}
_CONST_NAMES = {"list": list}


def _check_eval_node(node: ast.AST, code: str) -> None:
    if isinstance(node, ast.Expression):
        _check_eval_node(node.body, code)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, str)):
            raise StructureError(f"eval constant not allowed: {code!r}")
    elif isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise StructureError(f"eval name store not allowed: {code!r}")
    elif isinstance(node, ast.Subscript):
        _check_eval_node(node.value, code)
        _check_eval_node(node.slice, code)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise StructureError(f"eval operator not allowed: {code!r}")
        _check_eval_node(node.left, code)
        _check_eval_node(node.right, code)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise StructureError(f"eval operator not allowed: {code!r}")
        _check_eval_node(node.operand, code)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _check_eval_node(value, code)
    elif isinstance(node, ast.Compare):
        if any(type(op) not in _CMPOPS for op in node.ops):
            raise StructureError(f"eval comparison not allowed: {code!r}")
        _check_eval_node(node.left, code)
        for comparator in node.comparators:
            _check_eval_node(comparator, code)
    elif isinstance(node, ast.IfExp):
        _check_eval_node(node.test, code)
        _check_eval_node(node.body, code)
        _check_eval_node(node.orelse, code)
    elif isinstance(node, ast.List):
        if node.elts:
            raise StructureError(f"eval list literal must be empty: {code!r}")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise StructureError(f"eval call not allowed: {code!r}")
        for arg in node.args:
            _check_eval_node(arg, code)
    else:
        raise StructureError(f"eval syntax not allowed ({type(node).__name__}): {code!r}")


def compile_eval(code: str) -> ast.Expression:
    """Parse a dependency eval and check it against the whitelist: names,
    constants, subscripts, arithmetic, comparisons (`is` included),
    `x if c else y`, `and`/`or`/`not`, the empty list literal, the name
    `list`, and calls to round, type and len. Raises StructureError."""
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise StructureError(f"eval does not parse: {code!r}") from e
    _check_eval_node(tree, code)
    return tree


def _run(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _run(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _CONST_NAMES:
            return _CONST_NAMES[node.id]
        raise KeyError(node.id)
    if isinstance(node, ast.Subscript):
        return _run(node.value, env)[_run(node.slice, env)]
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_run(node.left, env), _run(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_run(node.operand, env))
    if isinstance(node, ast.BoolOp):
        result = None
        for value in node.values:
            result = _run(value, env)
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _run(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _run(comparator, env)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _run(node.body, env) if _run(node.test, env) else _run(node.orelse, env)
    if isinstance(node, ast.List):
        return []
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*(_run(arg, env) for arg in node.args))
    raise StructureError(f"unexpected eval node {type(node).__name__}")  # unreachable after compile_eval


# --- compiled structure -----------------------------------------------------


@dataclass(frozen=True)
class RepeatRule:
    """How a retriever's repeat is set while reading: an eval over earlier
    retrievers of the same struct (`targets` are their names)."""

    code: str
    tree: ast.Expression
    targets: tuple[str, ...]


@dataclass
class FieldSpec:
    name: str
    var: str  # the structure's `type` string
    kind: str  # "int", "float", "chars", "data", "str" or "struct"
    size: int  # item size; for "str" the length prefix size; 0 for "struct"
    fmt: str  # struct format char for int/float, "" otherwise
    signed: bool
    repeat: int  # static repeat
    is_list: bool | None
    rule: RepeatRule | None
    struct: StructSpec | None = None
    eof: bool = False
    unpacker: struct.Struct | None = None  # one item of an int, float or string length prefix

    def __post_init__(self) -> None:
        if self.fmt:
            self.unpacker = struct.Struct("<" + self.fmt)


@dataclass
class StructSpec:
    name: str
    fields: list[FieldSpec]
    min_size: int = 0
    fixed: struct.Struct | None = None  # every field a scalar int/float/chars/data with repeat 1, no rule
    structs: dict[str, StructSpec] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


@dataclass(frozen=True)
class Step:
    """One read in a struct: a single field (`run` None, `indices` one
    field), or a run of consecutive scalar fields unpacked in one call."""

    indices: tuple[int, ...]
    run: struct.Struct | None = None
    offsets: tuple[int, ...] = ()
    chars: tuple[bool, ...] = ()
    keep: tuple[bool, ...] = ()  # the value is an input to a later repeat eval


def _runnable(f: FieldSpec) -> bool:
    return f.kind in ("int", "float", "chars", "data") and f.repeat == 1 and f.rule is None and not f.eof and f.is_list is not True


def _steps(fields: list[FieldSpec]) -> list[Step]:
    needed = {t for f in fields if f.rule is not None for t in f.rule.targets}
    steps: list[Step] = []
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        offsets, offset = [], 0
        for i in run:
            offsets.append(offset)
            offset += fields[i].size
        fmt = "".join(fields[i].fmt if fields[i].kind in ("int", "float") else f"{fields[i].size}s" for i in run)
        steps.append(
            Step(
                tuple(run),
                struct.Struct("<" + fmt),
                tuple(offsets),
                tuple(fields[i].kind == "chars" for i in run),
                tuple(fields[i].name in needed for i in run),
            )
        )
        run.clear()

    for i, f in enumerate(fields):
        if _runnable(f):
            run.append(i)
            continue
        flush()
        steps.append(Step((i,)))
    flush()
    return steps


def _dependency_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _parse_target(target: str, where: str) -> str:
    section, _, name = target.partition(":")
    if section != "self" or not name:
        raise StructureError(f"{where}: only self: targets are read-path dependencies here, got {target!r}")
    return name


def _repeat_source(name: str, definition: Mapping[str, Any], where: str) -> tuple[str, tuple[str, ...]] | None:
    """The eval and targets that set this retriever's repeat while reading,
    or None for a static repeat."""
    deps = definition.get("dependencies", {}) or {}
    construct = _dependency_list(deps.get("on_construct"))
    if not construct:
        return None
    if len(construct) != 1:
        raise StructureError(f"{where}.{name}: more than one on_construct dependency")
    dep = construct[0]
    action = dep.get("action")
    if action == "REFRESH_SELF":
        refresh = _dependency_list(deps.get("on_refresh"))
        if len(refresh) != 1 or refresh[0].get("action") != "SET_REPEAT":
            raise StructureError(f"{where}.{name}: REFRESH_SELF without a single on_refresh SET_REPEAT")
        dep = refresh[0]
    elif action != "SET_REPEAT":
        raise StructureError(f"{where}.{name}: on_construct {action} is not supported")
    raw_targets = dep.get("target")
    if raw_targets is None:
        raise StructureError(f"{where}.{name}: SET_REPEAT without a target")
    targets = tuple(_parse_target(t, f"{where}.{name}") for t in ([raw_targets] if isinstance(raw_targets, str) else raw_targets))
    return dep.get("eval") or targets[0], targets


def _repeat_rule(name: str, definition: Mapping[str, Any], where: str) -> tuple[RepeatRule | None, bool]:
    deps = definition.get("dependencies", {}) or {}
    implied_list = any(d.get("action") == "SET_REPEAT" for v in deps.values() for d in _dependency_list(v))
    source = _repeat_source(name, definition, where)
    if source is None:
        return None, implied_list
    code, targets = source
    return RepeatRule(code, compile_eval(code), targets), implied_list


def _field_spec(name: str, definition: Mapping[str, Any], where: str) -> FieldSpec:
    var = definition.get("type")
    if not isinstance(var, str) or not var:
        raise StructureError(f"{where}.{name}: missing type")
    repeat = definition.get("repeat", 1)
    if type(repeat) is not int:
        raise StructureError(f"{where}.{name}: repeat must be an int, got {repeat!r}")
    rule, implied_list = _repeat_rule(name, definition, where)
    is_list = definition.get("is_list")
    if is_list is None and implied_list:
        is_list = True
    common = {"name": name, "var": var, "repeat": repeat, "is_list": is_list, "rule": rule, "eof": name == EOF_MARK}
    if var.startswith("struct:"):
        return FieldSpec(kind="struct", size=0, fmt="", signed=False, **common)
    letters = "".join(ch for ch in var if ch.isalpha())
    digits = "".join(ch for ch in var if ch.isdigit())
    if not digits:
        raise StructureError(f"{where}.{name}: type {var!r} has no size")
    bits = int(digits)
    if letters in ("u", "s"):
        size = bits // 8
        fmt = _INT_FORMATS.get((letters, size))
        if fmt is None:
            raise StructureError(f"{where}.{name}: unsupported int type {var!r}")
        return FieldSpec(kind="int", size=size, fmt=fmt, signed=letters == "s", **common)
    if letters == "f":
        size = bits // 8
        if size not in _FLOAT_FORMATS:
            raise StructureError(f"{where}.{name}: unsupported float type {var!r}")
        return FieldSpec(kind="float", size=size, fmt=_FLOAT_FORMATS[size], signed=True, **common)
    if letters == "str":
        size = bits // 8
        if (("s", size)) not in _INT_FORMATS:
            raise StructureError(f"{where}.{name}: unsupported string prefix {var!r}")
        return FieldSpec(kind="str", size=size, fmt=_INT_FORMATS[("s", size)], signed=True, **common)
    if letters == "c":
        return FieldSpec(kind="chars", size=bits, fmt="", signed=False, **common)
    if letters == "":
        return FieldSpec(kind="data", size=bits, fmt="", signed=False, **common)
    raise StructureError(f"{where}.{name}: unknown type {var!r}")


def _compile_block(name: str, block: Mapping[str, Any], depth: int) -> StructSpec:
    if depth > DEFAULT_DEPTH_LIMIT:
        raise StructureError(f"{name}: structs nest deeper than {DEFAULT_DEPTH_LIMIT}")
    retrievers = block.get("retrievers")
    if not isinstance(retrievers, Mapping):
        raise StructureError(f"{name}: no retrievers")
    spec = StructSpec(name=name, fields=[_field_spec(n, d, name) for n, d in retrievers.items()])
    spec.structs = {sn: _compile_block(sn, sb, depth + 1) for sn, sb in (block.get("structs") or {}).items()}
    for f in spec.fields:
        if f.kind == "struct":
            sub = f.var[len("struct:") :]
            if sub not in spec.structs:
                raise StructureError(f"{name}.{f.name}: struct {sub!r} is not defined in {name}'s own structs")
            f.struct = spec.structs[sub]
    spec.min_size = sum(_min_field_size(f) for f in spec.fields)
    spec.steps = _steps(spec.fields)
    if not spec.fields:
        raise StructureError(f"{name}: no retrievers")
    if all(_runnable(f) for f in spec.fields):
        fmt = "".join(f.fmt if f.kind in ("int", "float") else f"{f.size}s" for f in spec.fields)
        spec.fixed = struct.Struct("<" + fmt)
    return spec


def _min_field_size(f: FieldSpec) -> int:
    if f.rule is not None or f.eof:
        return 0
    count = max(f.repeat, 0)
    if f.kind == "struct":
        return count * f.struct.min_size
    return count * f.size


@dataclass
class CompiledStructure:
    sections: dict[str, StructSpec]


def compile_structure(structure: Mapping[str, Any] | CompiledStructure) -> CompiledStructure:
    """Check and pre-digest a structure dict. Raises StructureError on
    anything the walker can't read faithfully."""
    if isinstance(structure, CompiledStructure):
        return structure
    return CompiledStructure({name: _compile_block(name, block, 0) for name, block in structure.items()})


# --- the walk record --------------------------------------------------------


class Column:
    """Every occurrence of one field path, in read order: offset, size and
    value. A field inside a struct has exactly one occurrence per instance of
    that struct, so occurrence i belongs to instance i. Scalar ints and
    floats are stored in typed arrays."""

    __slots__ = ("offsets", "sizes", "values")

    def __init__(self) -> None:
        self.offsets = array("q")
        self.sizes = array("q")
        self.values: array | list | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _store(self, value: Any) -> None:
        values = self.values
        if values is None:
            if type(value) is int and -(1 << 63) <= value < (1 << 63):
                self.values = array("q", [value])
            elif type(value) is float:
                self.values = array("d", [value])
            else:
                self.values = [value]
            return
        if type(values) is list:
            values.append(value)
            return
        if (values.typecode == "q" and type(value) is int and -(1 << 63) <= value < (1 << 63)) or (
            values.typecode == "d" and type(value) is float
        ):
            values.append(value)
            return
        self.values = [*values, value]

    def add(self, offset: int, size: int, value: Any) -> None:
        self.offsets.append(offset)
        self.sizes.append(size)
        values = self.values
        kind = type(value)
        if type(values) is list or (
            type(values) is array and (kind is float if values.typecode == "d" else kind is int and -(1 << 63) <= value < (1 << 63))
        ):
            values.append(value)
        else:
            self._store(value)

    def extend(self, offsets: Iterable[int], size: int, values: Sequence[Any]) -> None:
        start = len(self.offsets)
        self.offsets.extend(offsets)
        self.sizes.extend([size] * (len(self.offsets) - start))
        if not values:
            return
        if self.values is None:
            first = values[0]
            if type(first) is int and all(type(v) is int and -(1 << 63) <= v < (1 << 63) for v in values):
                self.values = array("q", values)
                return
            if type(first) is float and all(type(v) is float for v in values):
                self.values = array("d", values)
                return
            self.values = list(values)
            return
        for value in values:
            self._store(value)

    def value(self, index: int) -> Any:
        return self.values[index]

    def value_list(self) -> list[Any]:
        return [] if self.values is None else list(self.values)


@dataclass
class Walk:
    """What a walk read. `columns` maps a field path (section, then retriever
    names down through struct fields) to its Column. A struct field's own
    column holds one entry per occurrence whose value is its instance count;
    the `path + ("#",)` column holds one entry per instance (offset and size
    of the instance, value its index within its occurrence)."""

    sections: dict[str, tuple[int, int]] = field(default_factory=dict)
    columns: dict[tuple[str, ...], Column] = field(default_factory=dict)
    end: int = 0
    reads: int = 0

    def column(self, *path: str) -> Column:
        return self.columns[tuple(path)]

    def values(self, *path: str) -> list[Any]:
        return self.columns[tuple(path)].value_list()


# --- decoding ---------------------------------------------------------------


def _decode_text(raw: bytes) -> str | bytes:
    if raw.endswith(b"\x00"):
        raw = raw[:-1]
    try:
        return raw.decode("utf-8")
    except ValueError:
        return raw.decode("latin-1")


def _decode_chars(raw: bytes) -> str | bytes:
    end = raw.find(b"\x00")
    return _decode_text(raw if end == -1 else raw[:end])


def _vorl(f: FieldSpec, repeat: int, values: list[Any]) -> Any:
    """The library's "value or list" rule (helper/bytes_parser.vorl)."""
    if repeat != 1:
        return values
    if f.is_list is not None:
        return values if f.is_list else values[0]
    return values[0] if len(values) == 1 else values


# --- walking ----------------------------------------------------------------


class _Walker:
    def __init__(self, data: bytes, walk: Walk, budget: int, string_cap: int, depth_limit: int, record: bool) -> None:
        self.data = data
        self.view = memoryview(data)
        self.size = len(data)
        self.walk = walk
        self.budget = budget
        self.string_cap = string_cap
        self.depth_limit = depth_limit
        self.record = record
        self.trail: list[list[Any]] = []  # [name, index or None] per level, formatted only on error
        self._block_columns: dict[tuple[str, ...], list[Column]] = {}

    def where(self, name: str | None = None) -> str:
        parts = [n if i is None else f"{n}[{i}]" for n, i in self.trail]
        if name is not None:
            parts.append(name)
        return ".".join(parts)

    def _col(self, path: tuple[str, ...]) -> Column:
        col = self.walk.columns.get(path)
        if col is None:
            col = self.walk.columns[path] = Column()
        return col

    def _columns(self, spec: StructSpec, path: tuple[str, ...]) -> list[Column]:
        cols = self._block_columns.get(path)
        if cols is None:
            cols = self._block_columns[path] = [self._col((*path, f.name)) for f in spec.fields]
        return cols

    def _spend(self, n: int, name: str | None, offset: int) -> None:
        self.walk.reads += n
        if self.walk.reads > self.budget:
            raise WalkError(self.where(name), offset, f"read budget of {self.budget} exceeded")

    def _repeat(self, f: FieldSpec, env: Mapping[str, Any], pos: int) -> int:
        # Like the library, the eval sees its target names only.
        scope = {name: env[name] for name in f.rule.targets if name in env}
        try:
            value = _run(f.rule.tree, scope)
        except Exception as e:
            raise WalkError(self.where(f.name), pos, f"repeat eval {f.rule.code!r} failed: {type(e).__name__}: {e}") from e
        if type(value) is not int:
            raise WalkError(self.where(f.name), pos, f"repeat eval {f.rule.code!r} gave {value!r}, not an int")
        return value

    def block(self, spec: StructSpec, pos: int, path: tuple[str, ...], depth: int) -> int:
        if depth > self.depth_limit:
            raise WalkError(self.where(), pos, f"struct depth over {self.depth_limit}")
        cols = self._columns(spec, path) if self.record else None
        fields = spec.fields
        env: dict[str, Any] = {}
        for step in spec.steps:
            run = step.run
            if run is not None:
                if run.size > self.size - pos:
                    self._run_overflow(fields, step, pos)
                values = run.unpack_from(self.data, pos)
                self.walk.reads += len(values)
                if self.walk.reads > self.budget:
                    raise WalkError(self.where(fields[step.indices[0]].name), pos, f"read budget of {self.budget} exceeded")
                for k, index in enumerate(step.indices):
                    value = values[k]
                    if step.chars[k]:
                        value = _decode_chars(value)
                    if cols is not None:
                        cols[index].add(pos + step.offsets[k], fields[index].size, value)
                    if step.keep[k]:
                        env[fields[index].name] = value
                pos += run.size
                continue
            index = step.indices[0]
            f = fields[index]
            repeat = f.repeat if f.rule is None else self._repeat(f, env, pos)
            count = repeat if repeat > 0 else 0
            start = pos
            if f.kind == "struct":
                need = count * f.struct.min_size
                if need > self.size - pos:
                    raise WalkError(
                        self.where(f.name), pos, f"{count} x {f.struct.name} needs at least {need} bytes, {self.size - pos} left"
                    )
                pos = self.struct_list(f, count, pos, (*path, f.name), depth + 1)
                value: Any = count
                env[f.name] = [None] * count
            elif f.eof and self.size - pos < f.size * count:
                value = env[f.name] = []
            else:
                values, pos = self.primitive(f, count, pos)
                value = env[f.name] = _vorl(f, repeat, values)
            if cols is not None:
                cols[index].add(start, pos - start, value)
        return pos

    def _run_overflow(self, fields: list[FieldSpec], step: Step, pos: int) -> None:
        for k, index in enumerate(step.indices):
            f = fields[index]
            at = pos + step.offsets[k]
            if f.size > self.size - at:
                raise WalkError(self.where(f.name), at, f"1 x {f.var} needs {f.size} bytes, {self.size - at} left")
        raise AssertionError("run overflow not located")  # unreachable: the run's size is the sum of its fields

    def primitive(self, f: FieldSpec, count: int, pos: int) -> tuple[list[Any], int]:
        left = self.size - pos
        if count * f.size > left:
            raise WalkError(self.where(f.name), pos, f"{count} x {f.var} needs {count * f.size} bytes, {left} left")
        self._spend(count if count > 1 else 1, f.name, pos)
        data = self.data
        kind = f.kind
        if kind == "int" or kind == "float":
            if count == 1:
                return [f.unpacker.unpack_from(data, pos)[0]], pos + f.size
            values = list(struct.unpack_from(f"<{count}{f.fmt}", data, pos)) if count else []
            return values, pos + count * f.size
        if kind == "data":
            values = [bytes(data[pos + i * f.size : pos + (i + 1) * f.size]) for i in range(count)]
            return values, pos + count * f.size
        if kind == "chars":
            values = [_decode_chars(bytes(data[pos + i * f.size : pos + (i + 1) * f.size])) for i in range(count)]
            return values, pos + count * f.size
        values = []
        for i in range(count):
            if f.size > self.size - pos:
                raise WalkError(self.where(f"{f.name}[{i}]"), pos, f"string length prefix needs {f.size} bytes, {self.size - pos} left")
            (length,) = f.unpacker.unpack_from(data, pos)
            pos += f.size
            if length <= 0:
                values.append("")
                continue
            if length > self.size - pos or length > self.string_cap:
                raise WalkError(self.where(f"{f.name}[{i}]"), pos - f.size, f"string length {length} with {self.size - pos} bytes left")
            values.append(_decode_text(bytes(data[pos : pos + length])))
            pos += length
        return values, pos

    def struct_list(self, f: FieldSpec, count: int, pos: int, path: tuple[str, ...], depth: int) -> int:
        if count == 0:
            return pos
        spec = f.struct
        self._spend(count, f.name, pos)
        inst = self._col((*path, "#")) if self.record else None
        if spec.fixed is not None:
            stride = spec.fixed.size
            end = pos + stride * count
            if end > self.size:
                raise WalkError(self.where(f.name), pos, f"{count} x {spec.name} needs {stride * count} bytes, {self.size - pos} left")
            self._spend(count * len(spec.fields), f.name, pos)
            if inst is not None:
                rows = spec.fixed.iter_unpack(self.view[pos:end])
                columns = list(zip(*rows, strict=True))
                inst.extend(range(pos, end, stride), stride, list(range(count)))
                field_offset = 0
                for sub, col, raw_values in zip(spec.fields, self._columns(spec, path), columns, strict=True):
                    if sub.kind == "chars":
                        values = [_decode_chars(bytes(v)) for v in raw_values]
                    elif sub.kind == "data":
                        values = [bytes(v) for v in raw_values]
                    else:
                        values = list(raw_values)
                    col.extend(range(pos + field_offset, end, stride), sub.size, values)
                    field_offset += sub.size
            return end
        crumb = [f.name, 0]
        self.trail.append(crumb)
        for i in range(count):
            crumb[1] = i
            start = pos
            pos = self.block(spec, pos, path, depth)
            if inst is not None:
                inst.add(start, pos - start, i)
        self.trail.pop()
        return pos


def walk(
    structure: Mapping[str, Any] | CompiledStructure,
    data: bytes,
    start: int = 0,
    sections: Sequence[str] | None = None,
    budget: int = DEFAULT_READ_BUDGET,
    *,
    string_cap: int = DEFAULT_STRING_CAP,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
    record: bool = True,
) -> Walk:
    """Walk `sections` (default: every section in the structure, in order)
    over `data` from `start`. Raises WalkError at the first field that
    doesn't fit, StructureError if the structure can't be read. With
    record=False only spans, `end` and `reads` are kept."""
    compiled = compile_structure(structure)
    names = list(compiled.sections) if sections is None else list(sections)
    result = Walk()
    walker = _Walker(data, result, budget, string_cap, depth_limit, record)
    pos = start
    for name in names:
        spec = compiled.sections.get(name)
        if spec is None:
            raise StructureError(f"no section {name!r} in the structure")
        begin = pos
        walker.trail = [[name, None]]
        try:
            pos = walker.block(spec, pos, (name,), 0)
        except WalkError as e:
            e.partial = result
            raise
        result.sections[name] = (begin, pos)
    result.end = pos
    return result


def inflate(raw: bytes, start: int, cap: int = DEFAULT_BODY_CAP) -> bytes:
    """Raw-deflate decompress `raw[start:]` the way the library does
    (zlib, -MAX_WBITS), refusing output over `cap` bytes. Raises
    zlib.error or ValueError."""
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    body = inflater.decompress(memoryview(raw)[start:], cap + 1)
    if len(body) > cap:
        raise ValueError(f"body decompresses to more than {cap} bytes")
    if not inflater.eof:
        raise zlib.error("deflate stream does not end inside the file")
    return body


@dataclass
class FileWalk:
    """A whole file: the FileHeader over the raw bytes, the rest over the
    decompressed body. `remainder` is body bytes past the last section."""

    header: Walk
    body: Walk
    body_bytes: bytes

    @property
    def remainder(self) -> int:
        return len(self.body_bytes) - self.body.end


def walk_file(
    structure: Mapping[str, Any] | CompiledStructure,
    raw: bytes,
    budget: int = DEFAULT_READ_BUDGET,
    *,
    body_cap: int = DEFAULT_BODY_CAP,
    record: bool = True,
) -> FileWalk:
    """Walk a whole scenario file: FileHeader, then every other section of
    the structure over the inflated body. Raises WalkError, StructureError,
    zlib.error or ValueError."""
    compiled = compile_structure(structure)
    header = walk(compiled, raw, 0, [HEADER_SECTION], budget, record=record)
    try:
        body = inflate(raw, header.end, body_cap)
    except (zlib.error, ValueError) as e:
        raise WalkError("body", header.end, f"does not inflate: {e}") from e
    rest = [n for n in compiled.sections if n != HEADER_SECTION]
    return FileWalk(header, walk(compiled, body, 0, rest, budget - header.reads, record=record), body)


def read_path_evals(structure: Mapping[str, Any]) -> list[str]:
    """Every eval string the walker would run for `structure` (read-path
    SET_REPEATs), for the whitelist test."""
    found: list[str] = []

    def visit(block: Mapping[str, Any]) -> None:
        for name, definition in (block.get("retrievers") or {}).items():
            source = _repeat_source(name, definition, "?")
            if source is not None:
                found.append(source[0])
        for sub in (block.get("structs") or {}).values():
            visit(sub)

    for block in structure.values():
        visit(block)
    return found
