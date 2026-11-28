"""Finding, consenting to and importing addons, and a report of what became
of each one.

Qt-free. The consent box is a dialog, so it is injected: `load_addons()`
takes `ask(path, digest) -> bool` from the window, and the decision of what
to ask about, the exact (resolved path, digest) pair looked up in the
confirmed set, stays here where it is tested headless. A set keyed on path
alone would confirm a file once and then run whatever that path later holds.
`ask` defaults to refusing: a caller with no way to ask cannot consent.

Nothing here raises for anything a folder can hold: an unreadable file, a
module that raises at import and one that calls `sys.exit()` each come back
as a failed record. `KeyboardInterrupt` and its kind propagate, so Ctrl-C
during a slow import still stops the process.

A failed import is unwound: every registration it made is rolled back, and
its module leaves `sys.modules`. An earlier addon that loaded is untouched.
`sys.path` is never changed: an addon is imported from its own location
under a namespaced module name, so `json.py` in the folder cannot shadow the
real module.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Iterable, Mapping, MutableSet, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

from descape import settings
from descape.registry import Registry
from descape.scripting import content_digest, describe, detail

# Points the addons folder somewhere else, replacing the default outright.
ENV_ADDON_PATH = "DESCAPE_ADDON_PATH"
# An addon's module is imported as this prefix plus its file or folder name.
MODULE_PREFIX = "descape_addon_"

LOADED = "loaded"
DECLINED = "declined"
FAILED = "failed"

AskConsent = Callable[[Path, str], bool]


def refuse_consent(path: Path, digest: str) -> bool:
    return False


@dataclass(frozen=True)
class AddonRecord:
    name: str
    path: str
    status: str
    message: str = ""
    detail: str = ""


@dataclass(frozen=True)
class AddonReport:
    """One pass over the addons folder. `disabled_by` names the switch that
    skipped it (then `root` is None and there are no records)."""

    root: str | None = None
    disabled_by: str = ""
    records: tuple[AddonRecord, ...] = field(default_factory=tuple)

    def with_status(self, status: str) -> tuple[AddonRecord, ...]:
        return tuple(record for record in self.records if record.status == status)

    @property
    def loaded(self) -> tuple[AddonRecord, ...]:
        return self.with_status(LOADED)

    @property
    def declined(self) -> tuple[AddonRecord, ...]:
        return self.with_status(DECLINED)

    @property
    def failed(self) -> tuple[AddonRecord, ...]:
        return self.with_status(FAILED)


_last_report: AddonReport | None = None


def last_report() -> AddonReport | None:
    """The most recent load_addons() report in this process, or None."""
    return _last_report


def clear_last_report() -> None:
    global _last_report
    _last_report = None


def addon_root(env: Mapping[str, str] | None = None) -> Path:
    """The addons folder: `DESCAPE_ADDON_PATH` if set, else `addons/` under the
    per-user data directory, in every build. Not created here."""
    environ = os.environ if env is None else env
    override = environ.get(ENV_ADDON_PATH, "")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("DEscape", appauthor=False)) / "addons"


def discover(root: Path) -> list[Path]:
    """The addons in `root` by name: `.py` files and folders holding an
    `__init__.py`. Names starting with `.` or `_` are skipped."""
    units = []
    for entry in sorted(root.iterdir(), key=lambda entry: entry.name):
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir():
            if (entry / "__init__.py").is_file():
                units.append(entry)
        elif entry.suffix == ".py":
            units.append(entry)
    return units


def module_name_for(path: Path) -> str:
    return MODULE_PREFIX + path.stem


def _import_unit(path: Path, source: bytes | None, digest: str) -> None:
    """Imports one addon from its location. A file runs `source`, the bytes
    hashed; a package is hashed again after its import and fails on a change,
    since its submodules are read from disk. Its modules are in `sys.modules`
    while it executes (a package's `from . import x` needs that) and leave
    again if it raised."""
    module_name = module_name_for(path)
    if path.is_dir():
        spec = importlib.util.spec_from_file_location(module_name, path / "__init__.py", submodule_search_locations=[str(path)])
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no import machinery for {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        if source is None:
            spec.loader.exec_module(module)
            if content_digest(path) != digest:
                raise ImportError(f"{path.name} changed while it loaded: restart DEscape to be asked about it")
        else:
            exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- consented code
    except BaseException:
        for name in [name for name in sys.modules if name == module_name or name.startswith(module_name + ".")]:
            sys.modules.pop(name, None)
        raise


def load_addons(
    registries: Iterable[Registry],
    *,
    root: Path | str | None = None,
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    ask: AskConsent | None = None,
    confirmed: MutableSet[tuple[str, str]] | None = None,
) -> AddonReport:
    """Imports every addon in the folder and reports on each.

    `registries` are what an addon may register into, needed here only to
    unwind a failed import. `confirmed` holds the (path, digest) pairs already
    agreed to and gains each one `ask` accepts. `ask` gets the resolved path,
    the key the pair holds, and persists its own Yes."""
    global _last_report
    containers = tuple(registries)
    switch = settings.addons_disabled_by(argv, env)
    if switch:
        _last_report = AddonReport(disabled_by=switch)
        return _last_report
    search = Path(root) if root is not None else addon_root(env)
    agreed: MutableSet[tuple[str, str]] = set() if confirmed is None else confirmed
    consent = refuse_consent if ask is None else ask
    records: list[AddonRecord] = []
    units: list[Path] = []
    try:
        if search.is_dir():
            units = discover(search)
    except OSError as exc:
        records.append(AddonRecord(search.name, str(search), FAILED, describe(exc)))
    taken: set[str] = set()
    for unit in units:
        module_name = module_name_for(unit)
        if not module_name.isidentifier() or module_name in taken or module_name in sys.modules:
            why = "its name is not a Python identifier" if not module_name.isidentifier() else "another addon has its name"
            records.append(AddonRecord(unit.name, str(unit), FAILED, f"not loaded: {why}"))
            continue
        taken.add(module_name)
        try:
            resolved = unit.resolve()
            source = None if unit.is_dir() else unit.read_bytes()
            digest = content_digest(unit if source is None else source)
        except OSError as exc:
            records.append(AddonRecord(unit.name, str(unit), FAILED, describe(exc)))
            continue
        pair = (str(resolved), digest)
        if pair not in agreed:
            if not consent(resolved, digest):
                records.append(AddonRecord(unit.name, str(resolved), DECLINED))
                continue
            agreed.add(pair)
        marks = [container.generation for container in containers]
        try:
            _import_unit(unit, source, digest)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- any raise from addon code fails that addon
            for container, mark in zip(containers, marks, strict=True):
                container.rollback_to(mark)
            records.append(AddonRecord(unit.name, str(resolved), FAILED, describe(exc), detail(exc)))
        else:
            records.append(AddonRecord(unit.name, str(resolved), LOADED))
    _last_report = AddonReport(root=str(search), records=tuple(records))
    return _last_report


def report_lines(report: AddonReport) -> list[str]:
    """Help > Debug Log's lines for a report: one per addon, a failure's
    traceback indented under it."""
    if report.disabled_by:
        return [f"Addons are off for this session ({report.disabled_by})"]
    lines = [f"Addons folder: {report.root}"]
    if not report.records:
        lines.append("  no addons found")
    for record in report.records:
        line = f"  {record.name}: {record.status}"
        if record.message:
            line += f": {record.message}"
        lines.append(line)
        if record.detail:
            lines.extend(f"    {part}" for part in record.detail.splitlines())
    return lines


def summary_text(report: AddonReport | None) -> str:
    """Tools > Addons' closing row: "2 loaded, 1 failed: see Help > Debug Log"."""
    if report is None:
        return "Addons have not loaded yet"
    if report.disabled_by:
        return f"Addons are off for this session ({report.disabled_by})"
    if not report.records:
        return "No addons found"
    counts = [
        f"{len(records)} {status}"
        for status, records in ((LOADED, report.loaded), (DECLINED, report.declined), (FAILED, report.failed))
        if records
    ]
    return f"{', '.join(counts)}: see Help > Debug Log"


def count_text(report: AddonReport | None) -> str:
    """The addon line of Report a Bug's environment, with no names or paths."""
    if report is None:
        return "not loaded"
    if report.disabled_by:
        return f"off ({report.disabled_by})"
    return f"{len(report.loaded)} loaded, {len(report.failed)} failed"


__all__ = [
    "DECLINED",
    "ENV_ADDON_PATH",
    "FAILED",
    "LOADED",
    "MODULE_PREFIX",
    "AddonRecord",
    "AddonReport",
    "addon_root",
    "clear_last_report",
    "count_text",
    "discover",
    "last_report",
    "load_addons",
    "refuse_consent",
    "report_lines",
    "summary_text",
]
