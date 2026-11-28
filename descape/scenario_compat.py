"""Opens a scenario saved by a newer game version than any structure DEscape
has, read-only, in compatibility mode (TASK-057).

The file is analysed under the newest known structure (the base) with no
library import: format_scan finds the body and the terrain grid, and the
guarded walker (structure_walk) checks which byte ranges read cleanly under
the base. Three groups are trusted on positive evidence only:

- Terrain: the scan finds exactly one grid. Without it the open is refused.
- Settings (FileHeader, DataHeader..Map up to map_width): the base header
  walk ends where the body starts, and the base walk of DataHeader..Map lands
  terrain_data on the scanned grid with the same width and height.
- Units: the base Units walk from the grid's end completes, ends on a
  plausible trigger version, and every unit passes format_scan's filters.

Triggers and Files are never read. The open then builds a proxy: a complete
body of the base version made of the file's trusted bytes and a donor's
(the shipped blank template) everywhere else, and loads it through the
normal loader, so every reader works unchanged. Anything showing a Settings
area checks `compat.settings_trusted` first, since an untrusted area holds
the donor's values. The proxy is never written: scenario_write refuses a
document with `compat` set.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from descape import format_scan as fs
from descape import game_dataset, games, library_compat
from descape import structure_walk as sw

UNKNOWN = "Unknown in this version"

# Test seam for the viewer's open path: the base version and donor bytes
# open_compat() uses when its caller passes none. Production never sets them.
TEST_BASE: str | None = None
TEST_DONOR: bytes | None = None

_F64 = struct.Struct("<d")


@dataclass(frozen=True)
class CompatPlan:
    """What the analysis trusts in a newer file. `reasons` holds one line
    per area ("terrain", "settings", "units", "triggers") for the log."""

    file_version: str
    base: str
    game: str
    raw: bytes = field(repr=False)
    body: fs.Body = field(repr=False)
    grid: fs.GridMatch
    header_ok: bool
    settings_trusted: bool
    units_span: tuple[int, int] | None
    trigger_version: float | None
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def units_trusted(self) -> bool:
        return self.units_span is not None


@dataclass(frozen=True)
class CompatRefusal:
    reason: str


def version_key(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, ValueError):
        return None


def applies(version: str) -> bool:
    """True for a dotted-number version newer than every structure DEscape has."""
    from descape import scenario_io

    key = version_key(version)
    newest = version_key(scenario_io.newest_structure_version())
    return key is not None and newest is not None and key > newest


def base_structure(version: str) -> dict:
    """A known version's structure.json, library first, then the repo's,
    with library_compat.fix_structure() applied."""
    for root in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR):
        path = root / f"v{version}" / "structure.json"
        if path.is_file():
            structure = json.loads(path.read_text(encoding="utf-8"))
            library_compat.fix_structure(structure, version)
            return structure
    raise LookupError(f"no structure for scenario version {version}")


def _terrain_stride(compiled: sw.CompiledStructure) -> int | None:
    terrain = next((f for f in compiled.sections["Map"].fields if f.name == "terrain_data"), None)
    if terrain is None or terrain.struct is None or terrain.struct.fixed is None:
        return None
    return terrain.struct.fixed.size


def _terrain_ids(game_id: str) -> set[int]:
    text = game_dataset.table_path("terrain_classes.json", dataset=game_id).read_text(encoding="utf-8")
    return {int(k) for k in json.loads(text)["classes"]}


def _first(walk: sw.Walk | None, *path: str) -> Any:
    col = None if walk is None else walk.columns.get(tuple(path))
    return None if col is None or not len(col) else col.value(0)


def _pre_grid_sections(compiled: sw.CompiledStructure) -> list[str]:
    names = list(compiled.sections)
    return names[names.index(sw.HEADER_SECTION) + 1 : names.index("Map") + 1]


def _settings_verdict(compiled: sw.CompiledStructure, body: bytes, grid: fs.GridMatch, header_ok: bool) -> str | None:
    """None when Settings is trusted, else why not."""
    if not header_ok:
        return "the file header does not read as the base version's"
    try:
        walk = sw.walk(compiled, body, 0, _pre_grid_sections(compiled))
    except sw.WalkError as e:
        return f"DataHeader..Map does not read as the base version's ({e})"
    col = walk.columns.get(("Map", "terrain_data"))
    if col is None or not len(col) or col.offsets[0] != grid.tiles_start:
        return "DataHeader..Map does not end on the terrain grid"
    if (_first(walk, "Map", "map_width"), _first(walk, "Map", "map_height")) != (grid.width, grid.height):
        return "the map size read before the grid differs from the grid's"
    if walk.end != grid.end:
        return "Map does not end where the grid does"
    return None


def _units_verdict(compiled: sw.CompiledStructure, body: bytes, grid: fs.GridMatch, base: str) -> tuple[str | None, int, float | None]:
    """(None, end, trigger version) when Units is trusted, else (why not, -1, None)."""
    known = fs.KNOWN_TRIGGER_VERSIONS.get(base)
    if known is None:
        return f"no trigger version is known for base {base}", -1, None
    try:
        walk = sw.walk(compiled, body, grid.end, ["Units"])
    except sw.WalkError as e:
        return f"Units does not read as the base version's ({e})", -1, None
    if walk.end + _F64.size > len(body):
        return "Units runs to the end of the body", -1, None
    (version,) = _F64.unpack_from(body, walk.end)
    if not fs.plausible_trigger_version(version, known):
        return f"Units does not end on a trigger version (read {version!r})", -1, None
    bad = fs.unit_filters(walk, grid.width, grid.height)
    if bad is not None:
        return f"a unit fails the plausibility filters: {bad}", -1, None
    return None, walk.end, version


def _find_body(compiled: sw.CompiledStructure, raw: bytes) -> tuple[fs.Body | str, bool, sw.Walk | None]:
    """(body or refusal reason, header_ok, the header walk or its partial)."""
    try:
        header = sw.walk(compiled, raw, 0, [sw.HEADER_SECTION])
    except sw.WalkError as e:
        header = e.partial
    else:
        found = fs.inflate_at(raw, header.end)
        if found is not None and len(found.data) >= fs.COMPAT_MIN_BODY_BYTES:
            return found, True, header
    body = fs.find_body(raw, min_bytes=fs.COMPAT_MIN_BODY_BYTES)
    return body, False, header


def analyse(raw: bytes, base: str | None = None) -> CompatPlan | CompatRefusal:
    """Which areas of `raw` read as the `base` version (default the newest
    known). Refuses when the body or the terrain grid can't be found."""
    from descape import scenario_io

    base = base or TEST_BASE or scenario_io.newest_structure_version()
    file_version = raw[:4].decode("latin-1")
    try:
        compiled = sw.compile_structure(base_structure(base))
    except (LookupError, sw.StructureError) as e:
        return CompatRefusal(f"the base structure {base} can't be read: {e}")
    body, header_ok, header = _find_body(compiled, raw)
    if isinstance(body, str):
        return CompatRefusal(f"the scenario's data could not be found: {body}")
    game = games.for_variant(_first(header, sw.HEADER_SECTION, "unknown_value_2")).id
    stride = _terrain_stride(compiled)
    if stride is None:
        return CompatRefusal(f"base {base}'s terrain is not a fixed-size struct")
    grid = fs.scan_terrain_grid(body.data, stride, _terrain_ids(game), (fs.COMPAT_MAP_TILES_MIN, fs.MAP_TILES_MAX))
    if isinstance(grid, fs.GridMiss):
        return CompatRefusal(f"the terrain could not be found: {grid.reason}")
    reasons = {"terrain": f"{grid.width}x{grid.height} grid at body offset {grid.offset}"}
    settings_why = _settings_verdict(compiled, body.data, grid, header_ok)
    reasons["settings"] = settings_why or "read"
    units_why, units_end, trigger_version = _units_verdict(compiled, body.data, grid, base)
    reasons["units"] = units_why or "read"
    reasons["triggers"] = "never read in compatibility mode"
    return CompatPlan(
        file_version=file_version,
        base=base,
        game=game,
        raw=raw,
        body=body,
        grid=grid,
        header_ok=header_ok,
        settings_trusted=settings_why is None,
        units_span=None if units_why else (grid.end, units_end),
        trigger_version=trigger_version,
        reasons=reasons,
    )


@dataclass(frozen=True)
class _Donor:
    header: bytes
    body: bytes
    grid_offset: int
    units_span: tuple[int, int]


def _read_donor(compiled: sw.CompiledStructure, raw: bytes, base: str) -> _Donor | CompatRefusal:
    version = raw[:4].decode("latin-1")
    if version != base:
        return CompatRefusal(
            f"this build's templates are version {version}, older than its newest structure {base}"
        )
    try:
        walked = sw.walk_file(compiled, raw)
    except sw.WalkError as e:
        return CompatRefusal(f"the template does not read as version {base}: {e}")
    if walked.remainder != 0:
        return CompatRefusal(f"the template has {walked.remainder} bytes past its last section")
    terrain = walked.body.columns.get(("Map", "terrain_data"))
    if terrain is None or not len(terrain):
        return CompatRefusal("the template has no terrain")
    counts = walked.body.columns.get(("Units", "players_units", "unit_count"))
    if counts is not None and any(counts.value_list()):
        return CompatRefusal("the template holds units")
    return _Donor(raw[: walked.header.end], walked.body_bytes, terrain.offsets[0] - 8, walked.body.sections["Units"])


def _default_donor(game: str) -> bytes:
    from descape import scenario_new

    return scenario_new.blank_template_path_for(game).read_bytes()


def proxy_bytes(plan: CompatPlan, donor_raw: bytes) -> tuple[bytes, bytes] | CompatRefusal:
    """(header, body) of a base-version file: the plan's trusted bytes, the
    donor's everywhere else, the donor's Triggers-and-Files tail."""
    compiled = sw.compile_structure(base_structure(plan.base))
    donor = _read_donor(compiled, donor_raw, plan.base)
    if isinstance(donor, CompatRefusal):
        return donor
    body, grid = plan.body.data, plan.grid
    if plan.settings_trusted:
        header, pre = plan.raw[: plan.body.start], body[: grid.offset]
    else:
        header, pre = donor.header, donor.body[: donor.grid_offset]
    units = body[slice(*plan.units_span)] if plan.units_span else donor.body[slice(*donor.units_span)]
    tail = donor.body[donor.units_span[1] :]
    return header, pre + body[grid.offset : grid.end] + units + tail


def open_compat(raw: bytes, path: str | Path, base: str | None = None, donor: bytes | None = None):
    """The file at `path` (bytes `raw`) as a read-only LoadedScenario of the
    base version with `.compat` set, or a CompatRefusal. `donor` defaults to
    the shipped blank template of the file's game."""
    from descape import scenario_io, scenario_write

    plan = analyse(raw, base)
    if isinstance(plan, CompatRefusal):
        return plan
    donor_raw = donor if donor is not None else TEST_DONOR if TEST_DONOR is not None else _default_donor(plan.game)
    built = proxy_bytes(plan, donor_raw)
    if isinstance(built, CompatRefusal):
        return built
    header, body = built
    proxy_raw = header + scenario_write._compress_bytes(body)
    loaded = scenario_io._load_map_and_units(proxy_raw, Path(path), structure_version=plan.base, body=body)
    for ok, what in (
        (loaded.terrain_write_supported, "terrain"),
        (loaded.units_write_supported, "units"),
        (loaded.messages_write_supported, "messages"),
    ):
        if not ok:
            return CompatRefusal(f"the compatibility view's {what} block failed its load-time check")
    loaded.structure_source = "compat"
    loaded.trigger_read_supported = False
    loaded.trigger_version = plan.trigger_version if plan.trigger_version is not None else -1.0
    loaded.compat = plan
    return loaded


def load_for_viewer(path: str | Path, load):
    """The viewer's open: `load` (viewer_file's load_map_and_units), and for a
    version newer than every known structure, the compatibility view. A
    refusal raises the original UnsupportedStructureVersion with the
    refusal's reason added."""
    from descape import scenario_io

    path = Path(path)
    try:
        return load(path)
    except scenario_io.UnsupportedStructureVersion as e:
        if not applies(e.scenario_version):
            raise
        result = open_compat(path.read_bytes(), path)
        if isinstance(result, CompatRefusal):
            raise scenario_io.UnsupportedStructureVersion(
                e.scenario_version, f"{e}\n\nCompatibility mode could not read it either: {result.reason}."
            ) from e
        return result


def plan_of(loaded) -> CompatPlan | None:
    return getattr(loaded, "compat", None)


def display_version(loaded) -> str:
    """The version a reader shows: the file's own in compatibility mode."""
    plan = plan_of(loaded)
    return plan.file_version if plan is not None else loaded.scenario_version


def settings_unknown(loaded) -> bool:
    """True when the open document's Settings areas hold a donor's values."""
    plan = plan_of(loaded)
    return plan is not None and not plan.settings_trusted


def units_unknown(loaded) -> bool:
    plan = plan_of(loaded)
    return plan is not None and not plan.units_trusted


TERRAIN_LATER = "Terrain can be edited once DEscape can read this file's map settings and units."


def terrain_writable(loaded) -> bool:
    """B2: a compat document's terrain and elevation may be written only when
    Settings and Units both read (the terrain tools' cliff and tree checks read units)."""
    plan = plan_of(loaded)
    return plan is not None and plan.settings_trusted and plan.units_trusted


def lock_reason(loaded) -> str | None:
    """Why a compat document's units, settings and triggers may not be edited
    (the status bar banner), or None for any other document."""
    plan = getattr(loaded, "compat", None)
    if plan is None:
        return None
    if terrain_writable(loaded):
        return (
            f"Compatibility mode: scenario version {plan.file_version} is newer than DEscape knows. "
            "Terrain and elevation can be edited; units, players and map settings are shown read-only; triggers Unknown."
        )
    return (
        f"Read-only: scenario version {plan.file_version} is newer than DEscape knows, opened in compatibility mode. "
        f"Terrain read; units {'shown' if plan.units_trusted else 'Unknown'}; "
        f"players and map settings {'shown' if plan.settings_trusted else 'Unknown'}; triggers Unknown."
    )


def terrain_lock_reason(loaded) -> str | None:
    """Why a compat document's terrain may not be written, None when it may
    (or for any other document)."""
    if plan_of(loaded) is None or terrain_writable(loaded):
        return None
    return f"{lock_reason(loaded)} {TERRAIN_LATER}"


def write_compat(loaded, out_path: str | Path, backup: bool):
    """Writes a compat document's terrain into its original file's bytes: the
    original header and body with only each tile's terrain_id, elevation and
    layer patched at the scanned grid, every other byte (trailing ones
    included) verbatim, and the original bytes unchanged when no tile
    changed. The proxy is never written. Returns scenario_write.WriteResult;
    raises WriteBlockedError for any other document, an untrusted area, a
    guarded destination or a tile list of the wrong size."""
    from descape import scenario_write as sw_

    plan = plan_of(loaded)
    if plan is None:
        raise sw_.WriteBlockedError("This document is not open in compatibility mode.")
    if not terrain_writable(loaded):
        raise sw_.WriteBlockedError(f"Scenario version {plan.file_version}: {TERRAIN_LATER}")
    out_path = Path(out_path)
    sw_.check_destination(loaded, out_path)
    grid, terrain = plan.grid, loaded.map_manager.terrain
    if len(terrain) != grid.width * grid.height:
        raise sw_.WriteBlockedError(
            f"The map holds {len(terrain)} tiles but the file's grid has {grid.width}x{grid.height}: refusing to write."
        )
    body = bytearray(plan.body.data)
    sw_.patch_tiles(body, grid.tiles_start, grid.stride, loaded.terrain_has_layer, terrain)
    raw = plan.raw
    if body == plan.body.data:
        final = raw
    else:
        tail = raw[len(raw) - plan.body.trailing :] if plan.body.trailing else b""
        final = raw[: plan.body.start] + sw_._compress_bytes(bytes(body)) + tail
    return sw_._write_file_bytes(out_path, final, backup)
