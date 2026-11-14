"""Running a user's Python code once against the open scenario.

A script is a `.py` file in the scripts folder. It runs with one global,
`editor`, an `Editor`, which is the whole supported API: reads of the map and
its units, terrain and elevation writes, unit writes, progress. An addon
command runs through the same transaction (`call()`).

**One undo entry.** A run that finishes pushes one `CompositeDiffRecord`
holding a tile record and a unit record, whichever it changed, so one Undo
backs out the whole run. A run that changed nothing pushes nothing.

**A failed or cancelled run rolls back everything**, tiles and units alike,
and nothing reaches the history. A rollback that cannot finish is reported
as `incomplete`, never silently.

**Only `editor` changes a unit.** `ScriptRun` keeps the values each unit is
expected to hold: its run-start values, or the ones recorded right after an
`editor` write that touched it. A unit or a unit list that differs from that
fails the run and rolls it back (`unit_findings()`), checked before each
`editor` unit write on the units it is about to touch, in full on the first
such write, and in full at the end. Failing is the only safe answer: a direct
write to an untouched unit would be dropped at save (its bytes are spliced
back verbatim), and one to a unit `editor` had touched would ride past the
rotation, const and garrison guards of `UnitEditModel` into the file.

**A tile may be written directly.** The whole-map diff records it either
way, so it is undoable. The run still fails if a tile it changed could not be
saved, or would save something the game cannot load (`terrain_findings()`).

**The facade hands out no edit model** and adds no caller to
`set_wall_variant()`, `set_cliff_frame()`, `set_unit_const()` or
`replace_type()`: each of those is scoped to named gestures by `AGENTS.md`.
There is no save here either; the window's File > Save keeps its own rules.

**No sandbox.** A script is ordinary Python with the user's full privileges.
The checks keep the supported API inside the write rules and catch mistakes;
they do not contain hostile code. Consent is keyed on (resolved path, sha256
of the bytes), asked again whenever the bytes change, and the bytes hashed
are the bytes run (`content_digest()`, `settings.add_confirmed_script()`).

Qt-free, so the transaction and the whole API are testable headless.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import math
import operator
import os
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs
from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError
from AoE2ScenarioParser.objects.data_objects.terrain_tile import TerrainTile
from AoE2ScenarioParser.objects.data_objects.unit import Unit

from descape import (
    batch_api,
    elevation_tools,
    garrison,
    object_catalog,
    place_rotation,
    render_units,
    scenario_io,
    unit_kind,
    unit_rotation,
    unit_variant,
)
from descape.edit_history import CompositeDiffRecord, DiffRecord, EditHistory, TileState, tile_state
from descape.iso_geometry import MAX_ELEVATION, MIN_ELEVATION
from descape.scenario_io import LoadedScenario
from descape.unit_model import UnitEditModel, UnitEditsUnavailableError, capture_flag_of, unit_state

# Points the scripts folder somewhere else, replacing the default outright.
ENV_SCRIPT_PATH = "DESCAPE_SCRIPT_PATH"

# The `__name__` a script runs under, so an `if __name__ == "__main__":` guard runs nothing.
SCRIPT_MODULE_NAME = "__descape_script__"

# saved_values()'s fields, in order: unit_state()'s nine, then the three the encoder also writes.
FIELD_NAMES = (
    "x",
    "y",
    "z",
    "rotation",
    "unit_const",
    "garrisoned_in_id",
    "initial_animation_frame",
    "status",
    "capture_flag",
    "reference_id",
    "caption_string_id",
    "caption_string",
)

# How many units or tiles one check names before it only counts the rest.
FINDING_LIMIT = 5

_NOT_RUNNING = "editor can only change the scenario while its script or addon command runs"
_NOT_SQUARE = "elevation can only be edited on a square map"
_TERRAIN_READ_ONLY = "this file's terrain failed load-time verification and cannot be written"
_NO_LAYER = "this file's terrain has no layer field, so a layer would never be saved"
_INCOMPLETE = "the scenario could not be fully rolled back: close it without saving"

_ALL_FIELDS = (True, True, True)

SavedValues = tuple


class ScriptError(Exception):
    """Something a script asked `editor` for cannot be done. Raised into the
    script, which may catch it; uncaught, it fails the run like any other."""


class Cancelled(BaseException):
    """The user cancelled the run. A BaseException, so a script's
    `except Exception` cannot swallow it."""


@dataclass(frozen=True)
class ScriptEntry:
    """One script in the folder: its file stem, its menu text, and where it is."""

    name: str
    title: str
    path: Path


@dataclass(frozen=True)
class ScriptOutcome:
    """What one run did.

    `changed` names what a finished run changed ("terrain", "units"), in the
    one history entry `record`. `rolled_back` names what a failed or cancelled
    run had changed and put back, and `touched` the tile indices among it, for
    the caller's repaint. `incomplete` means the rollback could not finish.
    `findings` are the checks' reasons, `error` one line naming what stopped
    the run and `detail` its traceback; both empty on success and on a cancel.
    """

    title: str
    label: str = ""
    changed: tuple[str, ...] = ()
    rolled_back: tuple[str, ...] = ()
    incomplete: bool = False
    output: str = ""
    error: str = ""
    detail: str = ""
    cancelled: bool = False
    findings: tuple[str, ...] = ()
    record: DiffRecord | None = None
    touched: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error and not self.cancelled


# -- folder, discovery, consent ----------------------------------------------


def script_root(env: Mapping[str, str] | None = None) -> Path:
    """The scripts folder: `DESCAPE_SCRIPT_PATH` if set, else `scripts/` under
    the per-user data directory, in every build. Not created here."""
    environ = os.environ if env is None else env
    override = environ.get(ENV_SCRIPT_PATH, "")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("DEscape", appauthor=False)) / "scripts"


def discover_scripts(root) -> list[ScriptEntry]:
    """Every `.py` file directly in `root`, by file name. Names starting with
    `.` or `_` are skipped and a directory is never a script. An absent or
    unreadable `root` is no scripts."""
    root = Path(root)
    try:
        if not root.is_dir():
            return []
        paths = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    return [
        ScriptEntry(path.stem, path.stem, path)
        for path in paths
        if path.suffix == ".py" and not path.name.startswith((".", "_")) and path.is_file()
    ]


def content_digest(source) -> str:
    """sha256 of the code a run would execute: of `source` itself when it is
    bytes, else of the file at that path. A package directory hashes every
    `.py` beneath it in sorted relative-path order, each name folded in, so
    moving code between two files changes the answer."""
    digest = hashlib.sha256()
    if isinstance(source, (bytes, bytearray, memoryview)):
        digest.update(source)
        return digest.hexdigest()
    path = Path(source)
    if path.is_dir():
        for member in sorted(path.rglob("*.py"), key=lambda p: p.relative_to(path).parts):
            digest.update(member.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(member.read_bytes())
            digest.update(b"\0")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def describe(exc: BaseException) -> str:
    """One line naming a failure, for a status line."""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def detail(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()


# -- the run-start unit image --------------------------------------------------


def field_support(loaded: LoadedScenario) -> tuple[bool, bool, bool]:
    """Whether this file stores capture_flag, caption_string_id and
    caption_string. An unsupported one is never read: its link raises."""
    return tuple(
        scenario_io.unit_field_supported(loaded, name)
        for name in ("capture_flag", "caption_string_id", "caption_string")
    )


def _optional(unit: Unit, name: str):
    try:
        return getattr(unit, name)
    except (UnsupportedAttributeError, AttributeError):
        return None


def saved_values(unit: Unit, support: tuple[bool, bool, bool] = _ALL_FIELDS) -> SavedValues:
    """Every field the unit encoder writes, in FIELD_NAMES order:
    unit_state()'s nine plus reference_id and the two caption fields. A field
    the file does not store (`support`, from field_support()) reads as None."""
    return (
        *unit_state(unit, support[0]),
        unit.reference_id,
        _optional(unit, "caption_string_id") if support[1] else None,
        _optional(unit, "caption_string") if support[2] else None,
    )


@dataclass
class UnitImage:
    """The units as a run found them: each player's list (strong references,
    so no id() here can be reused during the run) and each unit's values."""

    lists: tuple[tuple, ...]
    values: dict[int, SavedValues]
    support: tuple[bool, bool, bool]


def capture_image(loaded: LoadedScenario) -> UnitImage:
    support = field_support(loaded)
    lists = tuple(tuple(units) for units in loaded.unit_manager.units)
    values = {id(unit): saved_values(unit, support) for units in lists for unit in units}
    return UnitImage(lists, values, support)


def _same_items(a: Sequence, b: Iterable) -> bool:
    return all(map(operator.is_, a, b))


def _same_values(a: SavedValues, b: SavedValues) -> bool:
    """Equal, and of the same types: a float where the file holds an int
    compares equal and then breaks the encoder at save."""
    return a == b and [*map(type, a)] == [*map(type, b)]


def _field_differs(old, new) -> bool:
    return old is not new and (type(old) is not type(new) or old != new)


def restore_image(image: UnitImage, loaded: LoadedScenario, lists: bool = True) -> bool:
    """Puts the units back to `image`: each player's list (unless `lists` is
    False, when a UnitEditModel.rollback_to() already did) and every field
    that differs. Returns whether anything was written. Writes only what
    differs, so a run that touched no unit assigns nothing."""
    manager = loaded.unit_manager
    changed = False
    if lists:
        for player, units in enumerate(image.lists):
            live = manager.units[player]
            if len(live) != len(units) or not _same_items(live, units):
                # Slice assignment, never rebinding: the setter re-wraps the lists.
                live[:] = list(units)
                changed = True
        if changed:
            # The cached owner render reads; never the banned `unit.player` setter.
            manager.update_unit_player_values()
    for units in image.lists:
        for unit in units:
            want = image.values[id(unit)]
            now = saved_values(unit, image.support)
            if _same_values(now, want):
                continue
            for name, old, new in zip(FIELD_NAMES, now, want, strict=True):
                # None is a field this file does not store: assigning it would raise.
                if new is not None and _field_differs(old, new):
                    setattr(unit, name, new)
            changed = True
    if changed:
        # A direct write bumped nothing, so render's memos must be told.
        loaded.unit_gen += 1
    return changed


def image_differs(image: UnitImage, loaded: LoadedScenario) -> bool:
    """Whether any unit list or unit value has moved off `image`."""
    live_lists = loaded.unit_manager.units
    if len(live_lists) != len(image.lists):
        return True
    for live, units in zip(live_lists, image.lists, strict=True):
        if len(live) != len(units) or not _same_items(live, units):
            return True
    return any(
        not _same_values(saved_values(unit, image.support), image.values[id(unit)])
        for units in image.lists
        for unit in units
    )


class Membership:
    """Which units each player holds, as `editor` left them: the run-start
    lists with every facade add, remove and reassign applied. A dict per
    player keyed by id(unit), so a removal is O(1) and order is the list's
    own (a delete keeps it, an add or a reassign appends)."""

    def __init__(self, lists: Sequence[Sequence]):
        self.lists: list[dict[int, Unit]] = [{id(unit): unit for unit in units} for units in lists]

    def owner(self, unit) -> int | None:
        key = id(unit)
        for player, units in enumerate(self.lists):
            if units.get(key) is unit:
                return player
        return None

    def add(self, player: int, unit: Unit) -> None:
        self.lists[player][id(unit)] = unit

    def remove(self, units: Iterable[Unit]) -> None:
        for unit in units:
            for held in self.lists:
                held.pop(id(unit), None)

    def reassign(self, unit: Unit, player: int) -> None:
        self.remove([unit])
        self.add(player, unit)


# -- the checks ---------------------------------------------------------------


def _name(const: int) -> str:
    return object_catalog.display_name(const)


def _owner_name(player: int) -> str:
    return "GAIA" if player == 0 else f"Player {player}"


def _unit_label(unit) -> str:
    return f"{_name(unit.unit_const)} (id {unit.reference_id})"


def _capped(items: list[str], limit: int, noun: str) -> list[str]:
    if len(items) <= limit:
        return items
    return [*items[:limit], f"and {len(items) - limit} more {noun}"]


def _value_diff(unit, want: SavedValues, now: SavedValues) -> str:
    fields = [
        f"{name} {old!r} -> {new!r}"
        for name, old, new in zip(FIELD_NAMES, want, now, strict=True)
        if _field_differs(old, new)
    ]
    return f"{_unit_label(unit)} changed without editor: {', '.join(fields)}"


def unit_findings(
    image: UnitImage,
    expected: Mapping[int, tuple],
    membership: Membership,
    manager,
    model: UnitEditModel | None = None,
    limit: int = FINDING_LIMIT,
) -> list[str]:
    """Why the units are not what `editor` left them, as readable lines, or
    [] when they are. `expected` maps id(unit) to (unit, saved_values) for
    every unit a facade write touched; any other unit must hold its `image`
    values. Every list must equal `membership`'s, and a `model` must still be
    aligned with the lists. One pass over the units."""
    live_lists = manager.units
    if len(live_lists) != len(membership.lists):
        return [f"the scenario holds {len(live_lists)} unit lists, not {len(membership.lists)}"]
    findings: list[str] = []
    for player, (live, want) in enumerate(zip(live_lists, membership.lists, strict=True)):
        if len(live) == len(want) and _same_items(live, want.values()):
            continue
        present = {id(unit) for unit in live}
        extra = [unit for unit in live if want.get(id(unit)) is not unit]
        missing = [unit for unit in want.values() if id(unit) not in present]
        parts = []
        if extra:
            parts.append("added " + ", ".join(_capped([_unit_label(u) for u in extra], limit, "units")))
        if missing:
            parts.append("removed " + ", ".join(_capped([_unit_label(u) for u in missing], limit, "units")))
        findings.append(
            f"{_owner_name(player)}'s unit list changed without editor: {'; '.join(parts) or 'reordered'}"
        )
    differing: list[str] = []
    for live in live_lists:
        for unit in live:
            entry = expected.get(id(unit))
            want = entry[1] if entry is not None and entry[0] is unit else image.values.get(id(unit))
            if want is None:
                continue  # a unit the run never knew: the list pass named it
            now = saved_values(unit, image.support)
            if not _same_values(now, want):
                differing.append(_value_diff(unit, want, now))
    findings.extend(_capped(differing, limit, "units"))
    if model is not None:
        try:
            model._check_alignment()
        except RuntimeError as error:
            findings.append(f"the unit lists are out of step with their edit model: {error}")
    return findings


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def terrain_findings(
    changes: Iterable[tuple[int, TileState, TileState]],
    tiles: Sequence,
    width: int,
    height: int,
    *,
    write_supported: bool,
    has_layer: bool,
    limit: int = FINDING_LIMIT,
) -> list[str]:
    """Why the tiles a run changed cannot be kept, or []. `changes` is a
    TileDiffRecord's: (index, (terrain_id, elevation, layer) before, after).

    Only changed tiles are judged, so an illegal pair the run did not touch
    stays: any change on a file whose terrain cannot be written; a terrain_id
    or layer (other than -1) outside 0..255; a layer changed on a file with
    no layer field; a layer equal to its own terrain_id; and a changed
    elevation outside the legal range or more than 1 from an 8-adjacent
    tile, a seam that can crash the game on load."""
    changes = list(changes)
    if not changes:
        return []
    if not write_supported:
        return [f"{len(changes)} tile(s) changed, but {_TERRAIN_READ_ONLY}"]
    findings: list[str] = []
    pairs: set[frozenset[int]] = set()
    for index, (old_terrain, old_elevation, old_layer), (terrain, elevation, layer) in changes:
        y, x = divmod(index, width)
        where = f"tile ({x}, {y})"
        if not _is_int(terrain) or not 0 <= terrain <= 255:
            findings.append(f"{where} terrain_id {terrain!r} is not an int in 0..255")
        if not _is_int(layer) or not (layer == -1 or 0 <= layer <= 255):
            findings.append(f"{where} layer {layer!r} is not -1 or an int in 0..255")
        elif layer != old_layer and not has_layer:
            findings.append(f"{where} layer changed, but {_NO_LAYER}")
        elif layer != -1 and layer == terrain and (layer != old_layer or terrain != old_terrain):
            findings.append(f"{where} layer {layer} equals its own terrain_id; clear it with -1")
        if elevation == old_elevation and type(elevation) is type(old_elevation):
            continue
        if not _is_int(elevation) or not MIN_ELEVATION <= elevation <= MAX_ELEVATION:
            findings.append(f"{where} elevation {elevation!r} is not an int in {MIN_ELEVATION}..{MAX_ELEVATION}")
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if (dx or dy) and 0 <= nx < width and 0 <= ny < height:
                    other = tiles[ny * width + nx].elevation
                    pair = frozenset((index, ny * width + nx))
                    if _is_int(other) and abs(other - elevation) > 1 and pair not in pairs:
                        pairs.add(pair)
                        findings.append(
                            f"tiles ({x}, {y}) and ({nx}, {ny}) differ by {abs(other - elevation)} "
                            "elevation levels; the game may crash loading this map"
                        )
    return _capped(findings, limit, "tile problems")


_TILE_FIELDS = ("terrain_id", "elevation", "layer")


def mistyped_tiles(tiles: Sequence) -> list[int]:
    """Indices of tiles holding anything but an exact int. The stroke diff
    compares by value, so a float equal to the int it replaced is in no
    record, yet the terrain patch cannot pack it. One fast pass per field;
    the slow walk runs only when something is wrong."""
    if all(set(map(type, map(operator.attrgetter(name), tiles))) <= {int} for name in _TILE_FIELDS):
        return []
    return [i for i, tile in enumerate(tiles) if any(type(getattr(tile, name)) is not int for name in _TILE_FIELDS)]


# -- the transaction ----------------------------------------------------------


@dataclass
class _Closed:
    changed: tuple[str, ...] = ()
    rolled_back: tuple[str, ...] = ()
    incomplete: bool = False
    findings: tuple[str, ...] = ()
    record: DiffRecord | None = None
    touched: tuple[int, ...] = ()
    errors: list[BaseException] = field(default_factory=list)


class ScriptRun:
    """One run's transaction over a document: begin(), the code, close().

    `model` is the document's UnitEditModel if it has one. `model_getter`
    builds it on the run's first unit write otherwise, and raises
    UnitEditsUnavailableError when units cannot be edited (the default
    constructs one; the window passes its own, which also keeps it).
    `selected_units` and `selected_tiles` supply the window's selection, and
    `progress(done, total, note)` its progress dialog, raising Cancelled.
    """

    def __init__(
        self,
        loaded: LoadedScenario,
        history: EditHistory,
        *,
        model: UnitEditModel | None = None,
        model_getter: Callable[[], UnitEditModel] | None = None,
        selected_units: Callable[[], Iterable] | None = None,
        selected_tiles: Callable[[], Iterable] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        self.loaded = loaded
        self.history = history
        self.model = model
        self._model_getter = model_getter or (lambda: UnitEditModel(loaded))
        self._selected_units = selected_units
        self._selected_tiles = selected_tiles
        self._progress = progress
        self.editor = Editor(self)
        # True only while execute()/call() runs the code: a kept `editor` refuses afterwards.
        self.running = False
        self.cancelled = False
        # Reasons a check already failed this run; the run fails even if the script caught the raise.
        self.findings: list[str] = []
        self.image: UnitImage | None = None
        self.membership: Membership | None = None
        # id(unit) -> (unit, saved_values) after each facade write; holding the unit pins its id.
        self.expected: dict[int, tuple] = {}
        self._state = "new"
        self._edit_open = False
        self._units_ready = False
        # The unit record _commit() built, until its composite is pushed.
        self._unit_record = None
        # (index, run-start state) of tiles found holding a non-int at close().
        self._mistyped: list[tuple[int, TileState]] = []

    @property
    def tiles(self) -> Sequence:
        return self.loaded.map_manager.terrain

    def begin(self) -> None:
        """Opens the transaction: the run-start unit image, the model's
        whole-list unit edit if it has a model, and a whole-map stroke, which
        also makes the window's stroke-gated paths (autosave) wait. Refuses,
        changing nothing, while a stroke or a unit edit is already open."""
        if self._state != "new":
            raise RuntimeError("a ScriptRun runs once")
        if self.history.in_stroke:
            raise ScriptError("an edit is still in progress")
        self.image = capture_image(self.loaded)
        self.membership = Membership(self.image.lists)
        if self.model is not None:
            try:
                self.model.begin_unit_edit(range(9))
            except RuntimeError:
                raise ScriptError("a unit edit is still in progress") from None
            self._edit_open = True
        self.history.begin_stroke(self.tiles)
        self._state = "open"

    # The facade's side.

    def fail(self, findings: list[str]):
        self.findings.extend(findings)
        raise ScriptError("; ".join(findings))

    def unit_model(self) -> UnitEditModel:
        """The model for a facade unit write. The first call runs the full
        unit check, then builds the model if the document had none and opens
        its edit, so that snapshot equals the run-start image."""
        if self._units_ready:
            return self.model
        findings = unit_findings(self.image, self.expected, self.membership, self.loaded.unit_manager, self.model)
        if findings:
            self.fail(findings)
        if self.model is None:
            try:
                model = self._model_getter()
            except UnitEditsUnavailableError as error:
                raise ScriptError(f"units are read-only for this file: {error}") from None
            if model is None:
                raise ScriptError("units cannot be edited in this scenario")
            try:
                model.begin_unit_edit(range(9))
            except RuntimeError:
                raise ScriptError("a unit edit is still in progress") from None
            self.model = model
            self._edit_open = True
        self._units_ready = True
        return self.model

    def expected_values(self, unit) -> SavedValues:
        if self.membership.owner(unit) is None:
            raise ScriptError("that is not a unit of this scenario (was it removed?)")
        entry = self.expected.get(id(unit))
        return entry[1] if entry is not None and entry[0] is unit else self.image.values[id(unit)]

    def check(self, units: Iterable) -> None:
        """Before a facade write: its targets must hold their expected
        values, so a direct write cannot become the expected state."""
        differing = []
        for unit in units:
            want = self.expected_values(unit)
            now = saved_values(unit, self.image.support)
            if not _same_values(now, want):
                differing.append(_value_diff(unit, want, now))
        if differing:
            self.fail(_capped(differing, FINDING_LIMIT, "units"))

    def touched(self, units: Iterable) -> None:
        for unit in units:
            self.expected[id(unit)] = (unit, saved_values(unit, self.image.support))

    def occupants(self, model: UnitEditModel, unit) -> list:
        """The units garrisoned directly inside `unit`. A reference_id of -1
        holds nothing: -1 is "inside nothing" on every other unit."""
        return [] if unit.reference_id == -1 else model.referencing(unit)

    def riders(self, model: UnitEditModel, unit) -> list:
        """Everything garrisoned inside `unit`, transitively: what moves and
        changes owner with it, as it does under a dragged host."""
        found: list = []
        seen = {id(unit)}
        stack = [unit]
        while stack:
            for occupant in self.occupants(model, stack.pop()):
                if id(occupant) not in seen:
                    seen.add(id(occupant))
                    found.append(occupant)
                    stack.append(occupant)
        return found

    def report_progress(self, done: int, total: int, note: str) -> None:
        if self._progress is None:
            return
        try:
            self._progress(done, total, note)
        except Cancelled:
            self.cancelled = True
            raise

    # The runner's side.

    def close(self, label: str, keep: bool) -> _Closed:
        """Ends the transaction. With `keep`, runs the run-end checks and
        pushes one composite record; a finding, or `keep` False, rolls
        everything back instead. Never raises."""
        if self._state != "open":
            return _Closed()
        self._state = "closed"
        self.running = False
        errors: list[BaseException] = []
        findings = list(self.findings)
        tile_record = None
        try:
            # Before the record ends the stroke: their run-start states are only known until then.
            self._mistyped = [(i, self.history.stroke_start_state(i)) for i in mistyped_tiles(self.tiles)]
            tile_record = self.history.build_stroke_record(label, self.tiles)
        except Exception as error:  # noqa: BLE001 -- close() never raises: it reports
            errors.append(error)
            return self._roll_back(None, findings, errors, tiles_lost=True)
        mm = self.loaded.map_manager
        for index, _start in self._mistyped[:FINDING_LIMIT]:
            y, x = divmod(index, mm.map_width)
            findings.append(f"tile ({x}, {y}) holds {tile_state(self.tiles[index])!r}: every tile field must be an int")
        if keep and not findings:
            try:
                findings += terrain_findings(
                    tile_record.changes if tile_record is not None else (),
                    self.tiles,
                    mm.map_width,
                    mm.map_height,
                    write_supported=self.loaded.terrain_write_supported,
                    has_layer=self.loaded.terrain_has_layer,
                )
                findings += unit_findings(
                    self.image, self.expected, self.membership, self.loaded.unit_manager, self.model
                )
                if not findings:
                    return self._commit(label, tile_record)
            except Exception as error:  # noqa: BLE001 -- close() never raises: it reports
                errors.append(error)
        return self._roll_back(tile_record, findings, errors)

    def _commit(self, label: str, tile_record) -> _Closed:
        children: list[DiffRecord] = []
        changed: list[str] = []
        if tile_record is not None:
            children.append(tile_record)
            changed.append("terrain")
        if self._edit_open:
            unit_record = self.model.commit_unit_edit(label, self.history, push=False)
            self._edit_open = False
            self._unit_record = unit_record
            # A unit moved away and back, or added then removed, is no edit: its bytes stay the file's.
            self.model.settle_unchanged(unit_record)
            if unit_record.before.players:
                children.append(unit_record)
                changed.append("units")
            else:
                self._unit_record = None
        if not children:
            return _Closed()
        record = CompositeDiffRecord(label, children)
        errors: list[BaseException] = []
        try:
            self.history.push_composite_record(record)
        except Exception as error:
            # A listener's raise must not undo a push that landed.
            if not (self.history.records and self.history.records[-1] is record):
                raise
            # The record is on the stack: only history.on_change raised. The run is kept.
            errors.append(error)
        self._unit_record = None
        return _Closed(changed=tuple(changed), record=record, errors=errors)

    def _roll_back(self, tile_record, findings: list[str], errors: list, tiles_lost: bool = False) -> _Closed:
        rolled: list[str] = []
        touched: tuple[int, ...] = ()
        incomplete = tiles_lost
        try:
            if tile_record is not None:
                touched = tuple(tile_record.undo(self.tiles, None))
                rolled.append("terrain")
            for index, start in self._mistyped:
                tile = self.tiles[index]
                tile.terrain_id, tile.elevation, tile.layer = start
            if self._mistyped and not rolled:
                rolled.append("terrain")
            touched = tuple(dict.fromkeys([*touched, *(index for index, _start in self._mistyped)]))
        except Exception as error:  # noqa: BLE001 -- close() never raises: it reports
            errors.append(error)
            incomplete = True
        units_moved = True
        try:
            # Its own try: one unreadable unit must not stop the restore below.
            units_moved = image_differs(self.image, self.loaded)
        except Exception as error:  # noqa: BLE001 -- close() never raises: it reports
            errors.append(error)
        try:
            if self._edit_open:
                self._edit_open = False
                self.model.rollback_to(self.image.lists)
                restore_image(self.image, self.loaded, lists=False)
            elif self._unit_record is not None:
                # _commit() closed the edit, then raised before its push.
                self.model.restore(self._unit_record.before)
                restore_image(self.image, self.loaded, lists=False)
            else:
                restore_image(self.image, self.loaded)
            if units_moved:
                rolled.append("units")
            if image_differs(self.image, self.loaded):
                incomplete = True
        except Exception as error:  # noqa: BLE001 -- close() never raises: it reports
            errors.append(error)
            incomplete = True
        return _Closed(
            rolled_back=tuple(rolled),
            incomplete=incomplete,
            findings=tuple(findings),
            touched=touched,
            errors=errors,
        )


# -- the facade ---------------------------------------------------------------


@contextmanager
def _model_errors():
    """A model raise inside a facade call reaches the script as ScriptError
    with the model's message."""
    try:
        yield
    except (ValueError, UnitEditsUnavailableError) as error:
        raise ScriptError(str(error)) from None


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScriptError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _whole(value, what: str) -> int:
    if not _is_int(value):
        raise ScriptError(f"{what} must be an int, got {value!r}")
    return value


class Editor:
    """The `editor` a script sees. Everything here is supported; anything
    else reached through it is not.

    The reads work at any time and are read-only by contract. Every other
    method works only while the run's code is executing, so an `editor` kept
    by an addon refuses afterwards.
    """

    ScriptError = ScriptError
    Cancelled = Cancelled

    def __init__(self, run: ScriptRun):
        self._run = run

    # What a script reads.

    @property
    def loaded(self) -> LoadedScenario:
        """The open scenario, for batch_api's read helpers (buildings_of,
        neighbors, tile_under, is_water)."""
        return self._run.loaded

    @property
    def width(self) -> int:
        return self._run.loaded.map_manager.map_width

    @property
    def height(self) -> int:
        return self._run.loaded.map_manager.map_height

    @property
    def name(self) -> str:
        """The scenario's file name."""
        return Path(self._run.loaded.path).name

    def tile(self, x: int, y: int) -> TerrainTile:
        """The terrain tile at (x, y). Raises ScriptError off the map."""
        x, y = _whole(x, "x"), _whole(y, "y")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ScriptError(f"tile ({x}, {y}) is off the {self.width}x{self.height} map")
        return self._run.tiles[y * self.width + x]

    def units(self, player: int | None = None) -> list:
        """Every unit, or one player's (0 is GAIA), as a new list."""
        lists = self._run.loaded.unit_manager.units
        if player is None:
            return [unit for units in lists for unit in units]
        if not _is_int(player) or not 0 <= player < len(lists):
            raise ScriptError(f"player must be 0 (GAIA)..{len(lists) - 1}, got {player!r}")
        return list(lists[player])

    @property
    def selected_units(self) -> tuple:
        """The Units-mode selection when the run started from the window; empty elsewhere."""
        source = self._run._selected_units
        return () if source is None else tuple(source())

    @property
    def selected_tiles(self) -> tuple:
        """The committed Select region's (x, y) tiles; empty when there is none."""
        source = self._run._selected_tiles
        return () if source is None else tuple(source())

    # Running.

    def _writing(self) -> ScriptRun:
        run = self._run
        if not run.running:
            raise ScriptError(_NOT_RUNNING)
        if run.cancelled:
            raise Cancelled()
        if run.findings:
            raise ScriptError(f"this run has already failed: {run.findings[0]}")
        return run

    def progress(self, done: int, total: int, note: str = "") -> None:
        """Report how far the script has got. Raises Cancelled once the user
        presses Cancel, which rolls the whole run back."""
        self._writing().report_progress(done, total, note)

    # Terrain.

    def _terrain_run(self) -> ScriptRun:
        run = self._writing()
        if not run.loaded.terrain_write_supported:
            raise ScriptError(_TERRAIN_READ_ONLY)
        return run

    def _elevation_run(self) -> ScriptRun:
        run = self._terrain_run()
        if not run.loaded.map_is_square:
            raise ScriptError(_NOT_SQUARE)
        return run

    def set_terrain(self, x: int, y: int, terrain_id: int) -> None:
        """Sets the tile's terrain and clears its layer, as the Draw tool does."""
        self._terrain_run()
        tile = self.tile(x, y)
        if not _is_int(terrain_id) or not 0 <= terrain_id <= 255:
            raise ScriptError(f"terrain_id must be an int in 0..255, got {terrain_id!r}")
        batch_api.set_terrain(tile, terrain_id)

    def set_layer(self, x: int, y: int, terrain_id: int | None) -> None:
        """Sets the tile's layered terrain, or removes it with None (or with
        the tile's own terrain). Refused on a file with no layer field."""
        run = self._terrain_run()
        if not run.loaded.terrain_has_layer:
            raise ScriptError(_NO_LAYER)
        tile = self.tile(x, y)
        if terrain_id is not None and not _is_int(terrain_id):
            raise ScriptError(f"a layer is a terrain id or None, got {terrain_id!r}")
        with _model_errors():
            batch_api.set_layer(tile, terrain_id)

    def set_elevation(self, x: int, y: int, level: int) -> None:
        """Sets the tile's elevation, clamped to the legal range, sloping
        its neighbours as the Elevation tool does. Square maps only."""
        run = self._elevation_run()
        tile = self.tile(x, y)
        level = max(MIN_ELEVATION, min(MAX_ELEVATION, _whole(level, "level")))
        elevation_tools.set_tile_elevation(run.loaded.map_manager, tile.x, tile.y, level)

    def raise_elevation(self, units_or_tiles, amount: int = 1) -> int:
        """Raises (or lowers) the tile under each unit, or each tile or
        (x, y) given, by `amount`. Every target is computed from one
        snapshot before any is written, so shared and adjacent tiles do not
        compound. Returns how many tiles were set. Square maps only."""
        run = self._elevation_run()
        amount = _whole(amount, "amount")
        targets: dict[tuple[int, int], int] = {}
        for item in units_or_tiles:
            if isinstance(item, Unit):
                tile = batch_api.tile_under(run.loaded, item)
            elif isinstance(item, TerrainTile):
                tile = item
            else:
                try:
                    x, y = item
                except (TypeError, ValueError):
                    raise ScriptError(f"raise_elevation takes units, tiles or (x, y) pairs, got {item!r}") from None
                tile = self.tile(x, y)
            if tile is not None and tile.xy not in targets:
                targets[tile.xy] = max(MIN_ELEVATION, min(MAX_ELEVATION, tile.elevation + amount))
        for (x, y), level in targets.items():
            elevation_tools.set_tile_elevation(run.loaded.map_manager, x, y, level)
        return len(targets)

    # Units.

    def _point(self, x, y) -> tuple[float, float]:
        x, y = _number(x, "x"), _number(y, "y")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ScriptError(f"({x}, {y}) is off the {self.width}x{self.height} map")
        return x, y

    def _player(self, player) -> int:
        if not _is_int(player) or not 0 <= player <= 8:
            raise ScriptError(f"player must be 0 (GAIA)..8, got {player!r}")
        return player

    def move_unit(self, unit, x: float, y: float) -> None:
        """Moves `unit` to (x, y), its height kept. A host carries the units inside it."""
        run = self._writing()
        x, y = self._point(x, y)
        model = run.unit_model()
        group = [unit, *run.riders(model, unit)]
        run.check(group)
        with _model_errors():
            for member in group:
                if (member.x, member.y) != (x, y):
                    model.set_position(member, x, y, member.z)
        run.touched(group)

    def set_status(self, unit, value: int) -> None:
        """Sets the unit's state (0 Foundation, 2 Alive, ...): any int in 0..255."""
        run = self._writing()
        model = run.unit_model()
        run.check([unit])
        if _is_int(value) and unit.status == value:
            return
        with _model_errors():
            model.set_status(unit, value)
        run.touched([unit])

    def set_capture_flag(self, unit, value: int) -> None:
        """Sets the unit's capture flag (-1 Default, ...). Refused on a file saved before 1.59."""
        run = self._writing()
        model = run.unit_model()
        run.check([unit])
        if model.supports_capture_flag and _is_int(value) and capture_flag_of(unit) == value:
            return
        with _model_errors():
            model.set_capture_flag(unit, value)
        run.touched([unit])

    def reassign(self, unit, player: int) -> None:
        """Gives `unit`, and the units inside it, to `player` (0 is GAIA).
        A GAIA-only object stays GAIA's, so that is refused."""
        run = self._writing()
        player = self._player(player)
        model = run.unit_model()
        group = [unit, *run.riders(model, unit)]
        run.check(group)
        for member in group:
            if unit_kind.owner_refused(member.unit_const, player):
                raise ScriptError(f"{_name(member.unit_const)} is GAIA-only; only GAIA can own it")
        with _model_errors():
            for member in group:
                if run.membership.owner(member) != player:
                    model.reassign(member, player)
                    run.membership.reassign(member, player)
        run.touched(group)

    def remove(self, units) -> None:
        """Deletes one unit or several. Refused, removing none, while a unit
        outside the batch is garrisoned inside one of them."""
        run = self._writing()
        batch = [units] if isinstance(units, Unit) else list(units)
        unique = list({id(unit): unit for unit in batch}.values())
        if not unique:
            return
        model = run.unit_model()
        run.check(unique)
        with _model_errors():
            model.remove_many(unique)
        run.membership.remove(unique)
        for unit in unique:
            run.expected.pop(id(unit), None)

    def add_unit(self, const: int, x: float, y: float, player: int, facing: float | None = None, variant: int | None = None):
        """Places a new unit and returns it. A GAIA-only object goes to GAIA
        whatever `player` says, as Place Unit does. `facing` (radians, for a
        unit that faces) snaps to the unit's own directions; `variant` picks
        a graphic variant of a tree or other scenery."""
        run = self._writing()
        if not _is_int(const) or not object_catalog.is_known_object(const):
            raise ScriptError(f"there is no object with id {const!r}")
        x, y = self._point(x, y)
        owner = unit_kind.placement_owner(const, self._player(player))
        state = place_rotation.PlaceRotation()
        if facing is not None:
            if not unit_rotation.rotation_is_angle(const):
                raise ScriptError(f"{_name(const)} has no facing to set")
            state = place_rotation.PlaceRotation(facing=_number(facing, "facing"))
        if variant is not None:
            if not unit_variant.is_cyclable(const):
                raise ScriptError(f"{_name(const)} has no graphic variants to pick")
            count = unit_variant.variant_count_for(const)
            if not _is_int(variant) or not 0 <= variant < count:
                raise ScriptError(f"variant must be an int in 0..{count - 1} for {_name(const)}, got {variant!r}")
            state = place_rotation.PlaceRotation(variant=variant, variant_const=const)
        # The placement exception's own guard: the only source of a new unit's rotation.
        rotation = place_rotation.rotation_for(state, const, owner)
        model = run.unit_model()
        with _model_errors():
            unit = model.add(owner, const, x, y, rotation=rotation)
        run.membership.add(owner, unit)
        run.touched([unit])
        return unit

    def rotate(self, unit, facing: float) -> None:
        """Turns a unit that faces to `facing` (radians), snapped to its own
        directions. Refused for anything whose rotation is not a facing
        (walls, gates, trees and other scenery)."""
        run = self._writing()
        facing = _number(facing, "facing")
        model = run.unit_model()
        run.check([unit])
        const = unit.unit_const
        if unit_rotation.rotation_is_angle(const):
            count = unit_rotation.angle_count_for(const)
            facing = unit_rotation.facing_to_rotation(unit_rotation.rotation_to_facing(facing, count), count)
            if unit.rotation == facing:
                return
        with _model_errors():
            model.set_rotation(unit, facing)
        run.touched([unit])

    def set_variant(self, unit, index: int) -> None:
        """Picks graphic variant `index` of a tree or other scenery. Refused
        for a wall, cliff or gate, whose shape the game derives itself."""
        run = self._writing()
        index = _whole(index, "index")
        model = run.unit_model()
        run.check([unit])
        const = unit.unit_const
        if unit_variant.is_cyclable(const):
            count = unit_variant.variant_count_for(const)
            if not 0 <= index < count:
                raise ScriptError(f"variant must be in 0..{count - 1} for {_name(const)}, got {index}")
            if unit.rotation == float(index):
                return
        with _model_errors():
            # Always the literal integer, the Cycle Variant rule.
            model.set_variant(unit, float(index))
        run.touched([unit])

    def garrison(self, unit, host) -> None:
        """Puts `unit` inside `host`, at its point, if the game would allow
        it (type and base capacity) and the link is sound. `host` None
        unloads it onto a tile beside its host."""
        run = self._writing()
        model = run.unit_model()
        if host is None:
            self._unload(run, model, unit)
            return
        run.check([unit, host])
        if unit.garrisoned_in_id == host.reference_id and unit is not host:
            return
        message = garrison.refusal(
            host.unit_const, _name(host.unit_const), len(run.occupants(model, host)), [unit.unit_const], name_of=_name
        )
        if message is not None:
            raise ScriptError(message)
        with _model_errors():
            model.set_garrisoned_in(unit, host.reference_id)
            model.set_position(unit, host.x, host.y, getattr(host, "z", 0.0))
        run.touched([unit])

    def _unload(self, run: ScriptRun, model: UnitEditModel, unit) -> None:
        run.check([unit])
        if unit.garrisoned_in_id == -1:
            return
        host = next(
            (
                candidate
                for units in run.loaded.unit_manager.units
                for candidate in units
                if candidate.reference_id == unit.garrisoned_in_id and candidate is not unit
            ),
            None,
        )
        x, y = unit.x, unit.y
        if host is not None:
            bounds = render_units.unit_tile_bounds(host, self.width, self.height)
            tile = None if bounds is None else garrison.unload_tile(bounds, self.width, self.height)
            if tile is None:
                raise ScriptError(f"no tile beside {_name(host.unit_const)} is on the map")
            x, y = tile[0] + 0.5, tile[1] + 0.5
        with _model_errors():
            model.set_garrisoned_in(unit, -1)
            model.set_position(unit, x, y, unit.z)
        run.touched([unit])


# -- running code -------------------------------------------------------------


def _run_guarded(body: Callable[[], None], run: ScriptRun, title: str, label: str) -> ScriptOutcome:
    try:
        run.begin()
    except ScriptError as error:
        return ScriptOutcome(title=title, label=label, error=str(error))
    captured = io.StringIO()
    failure: BaseException | None = None
    cancelled = False
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            body()
    except Cancelled:
        cancelled = True
    except (Exception, SystemExit) as error:  # noqa: BLE001 -- any raise from user code fails the run
        failure = error
    except BaseException:
        # KeyboardInterrupt and its kind: someone asking the process to stop, not a
        # script failing. Put everything back, then let it go.
        run.close(label, keep=False)
        raise
    finally:
        run.running = False
    # A cancel some `except BaseException` swallowed is still a cancel.
    cancelled = cancelled or run.cancelled
    closed = run.close(label, keep=failure is None and not cancelled)
    error = ""
    details: list[str] = []
    if failure is not None:
        error = describe(failure)
        details.append(detail(failure))
    elif closed.findings and not cancelled:
        error = closed.findings[0]
        if len(closed.findings) > 1:
            error += f" (and {len(closed.findings) - 1} more)"
    elif closed.errors and closed.record is None and not cancelled:
        error = describe(closed.errors[0])
    details.extend(detail(problem) for problem in closed.errors)
    return ScriptOutcome(
        title=title,
        label=label,
        changed=closed.changed,
        rolled_back=closed.rolled_back,
        incomplete=closed.incomplete,
        output=captured.getvalue(),
        error=error,
        detail="\n\n".join(details),
        cancelled=cancelled,
        findings=closed.findings,
        record=closed.record,
        touched=closed.touched,
    )


def execute(source, namespace: dict, run: ScriptRun, *, filename: str, title: str, label: str) -> ScriptOutcome:
    """Compiles `source` and runs it in `namespace` inside `run`'s
    transaction, then keeps what it did as one `label` history entry or
    rolls all of it back.

    Compiling happens inside the output capture, so a SyntaxError is a
    failure like any other. KeyboardInterrupt rolls back, then propagates;
    SystemExit is a failure."""

    def body() -> None:
        code = compile(source, filename, "exec")
        run.running = True
        exec(code, namespace)  # noqa: S102 -- running the user's own consented code is the feature

    return _run_guarded(body, run, title, label)


def call(function: Callable[[Editor], object], run: ScriptRun, *, title: str, label: str) -> ScriptOutcome:
    """Runs `function(editor)` inside `run`'s transaction, as execute() runs
    a script: an addon command's way in."""

    def body() -> None:
        run.running = True
        function(run.editor)

    return _run_guarded(body, run, title, label)


def run_script(entry: ScriptEntry, source: bytes, run: ScriptRun) -> ScriptOutcome:
    """Runs `source`, the bytes of `entry` the user agreed to, with
    `run.editor` as `editor`."""
    namespace = {
        "__name__": SCRIPT_MODULE_NAME,
        "__file__": str(entry.path),
        "__builtins__": builtins,
        "editor": run.editor,
    }
    return execute(
        source, namespace, run, filename=str(entry.path), title=entry.title, label=f'Script "{entry.title}"'
    )


def summarise(outcome: ScriptOutcome) -> tuple[str, list[str]]:
    """`(short, lines)` for a run: `short` fits the status line, `lines`
    adds what a reader may need to act on. "Nothing changed" is said only
    when it is true: an incomplete rollback contradicts it."""
    title = outcome.label or f'Script "{outcome.title}"'
    settled = ", nothing changed" if not outcome.incomplete else ""
    if outcome.cancelled:
        short = f"{title} was cancelled{settled}"
    elif outcome.error:
        short = f"{title} failed{settled}: {outcome.error}"
    elif outcome.changed:
        short = f"{title} changed {' and '.join(outcome.changed)}"
    else:
        short = f"{title} ran, but changed nothing"
    lines = list(outcome.findings)
    if outcome.rolled_back:
        lines.append(f"Rolled back: {' and '.join(outcome.rolled_back)}")
    if outcome.incomplete:
        lines.append(_INCOMPLETE[0].upper() + _INCOMPLETE[1:])
        short += f"; {_INCOMPLETE}"
    return short, lines


__all__ = [
    "ENV_SCRIPT_PATH",
    "FIELD_NAMES",
    "SCRIPT_MODULE_NAME",
    "Cancelled",
    "Editor",
    "Membership",
    "ScriptEntry",
    "ScriptError",
    "ScriptOutcome",
    "ScriptRun",
    "UnitImage",
    "call",
    "capture_image",
    "content_digest",
    "discover_scripts",
    "execute",
    "field_support",
    "image_differs",
    "mistyped_tiles",
    "restore_image",
    "run_script",
    "saved_values",
    "script_root",
    "summarise",
    "terrain_findings",
    "unit_findings",
]
