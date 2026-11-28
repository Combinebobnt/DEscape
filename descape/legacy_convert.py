"""File > Convert to .aoe2scenario (GH #315): a new DE document from an open
pre-DE (header-2) one, as AoE2:DE's own editor import makes it. Qt-free.

The legacy document is only read. The new one starts as the default game's
blank template at the legacy map's size (scenario_new.blank_body()) and is
filled through the same edit models a save uses: terrain tiles,
OptionsEditModel (options, players, diplomacy, disables, AI),
MessagesEditModel and UnitEditModel.import_units(). One
scenario_write.build_patched_body(), then the player cameras patched in
place on the reloaded result (fixed width, the clamp_cameras() shape).

What the importer rewrites is matched where the AoC DE-import pairs measure
it (tests/test_legacy_convert_oracle.py): terrain 33/34, the player value
remaps, and unit images by rule (plan_unit_image()). Its random picks use
`rng`, so only the rule's set can be compared. Everything unmeasured passes
through as stored. Triggers are not carried yet; the report counts them,
and every other thing left out, so nothing is dropped silently.
"""

from __future__ import annotations

import random
import struct
from collections import Counter
from dataclasses import dataclass, field

from AoE2ScenarioParser.datasets.object_support import Civilization, CivilizationOld

from descape import (
    ai_scripts,
    diplomacy_fields,
    disables_fields,
    game_dataset,
    games,
    gate_orientation,
    legacy_import,
    messages_fields,
    option_fields,
    player_fields,
    scenario_format,
    scenario_new,
    unit_kind,
    unit_sprites,
)
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.render_units import unit_occupied_tiles, unit_tile_bounds
from descape.scenario_io import LoadedScenario, load_map_and_units, load_map_and_units_from_bytes, reactivate
from descape.scenario_write import _compress_bytes, build_patched_body
from descape.unit_model import UnitEditModel

# Measured importer remaps on the AoC pairs (2026-11-22). The oracle test imports these.
TERRAIN_REMAP = {33: (6, 32), 34: (0, 32)}
CIV_TOKENS = {
    1: "BRITON-CIV", 2: "FRANKISH-CIV", 3: "GOTHIC-CIV", 4: "TEUTONIC-CIV", 5: "JAPANESE-CIV",
    6: "CHINESE-CIV", 7: "BYZANTINE-CIV", 8: "PERSIAN-CIV", 9: "SARACEN-CIV", 12: "MONGOL-CIV", 17: "HUN-CIV",
}  # fmt: skip
DE_POP_LIMIT = 200
STRING_ID_NONE = (-1, -2)  # player_name_string_id: legacy -> DE
# DE-only map options the importer sets the same on both pairs.
DE_OPTION_DEFAULTS = {"villager_force_drop": 1}
_NO_CINEMATIC = {"", "<None>"}
_TYPE_BUILDING = 80

_NEXT_UNIT_ID = struct.Struct("<i")
_CAMERA = {"f32": struct.Struct("<f"), "s16": struct.Struct("<h"), "s32": struct.Struct("<i")}
_PLAYER_DATA_3_CAMERAS = ("editor_camera_x", "editor_camera_y", "initial_camera_x", "initial_camera_y")


@dataclass
class ConversionReport:
    """What the conversion changed by rule, and everything it left out."""

    triggers_left_out: int = 0
    dropped_consts: Counter = field(default_factory=Counter)  # const -> units, unknown to the AoE2 tables
    links_cut: int = 0  # garrison links to a missing host, or to the unit itself
    unmeasured_band: str | None = None  # a band with no DE-import pair: rules are The Conquerors'
    aok_links_cleared: int = 0  # AoK's 0 "not garrisoned"
    renumbered: int = 0  # negative or repeated reference_ids given fresh ones
    rewritten: Counter = field(default_factory=Counter)  # rule -> units whose image it set
    kept_gate_rotations: int = 0  # gates with a nonzero stored rotation, unmeasured
    elevation_steps: int = 0  # neighbour pairs more than 1 apart
    refused: list[str] = field(default_factory=list)  # fields the new document would not take
    not_carried: list[str] = field(default_factory=list)  # data with no write path

    @property
    def left_out(self) -> bool:
        """Whether anything did not come across as stored (rule rewrites aside)."""
        return bool(
            self.triggers_left_out or self.dropped_consts or self.links_cut or self.renumbered
            or self.elevation_steps or self.refused or self.not_carried
        )  # fmt: skip

    def lines(self) -> list[str]:
        out = [f"Unit images set as AoE2:DE's import sets them: {n} {rule}" for rule, n in sorted(self.rewritten.items())]
        if self.unmeasured_band:
            out.append(
                f"The import rules were measured on The Conquerors files; this {self.unmeasured_band} file "
                "may differ from the game's own import of it."
            )
        if self.triggers_left_out:
            out.append(f"Left out {self.triggers_left_out} triggers: converting triggers is not supported yet.")
        if self.dropped_consts:
            n = sum(self.dropped_consts.values())
            out.append(f"Left out {n} units of {len(self.dropped_consts)} object types AoE2:DE does not have.")
        if self.links_cut:
            out.append(f"Unloaded {self.links_cut} garrisoned units whose host is not on the map or is the unit itself.")
        if self.aok_links_cleared:
            out.append(f"Read {self.aok_links_cleared} Age of Kings garrison fields of 0 as not garrisoned.")
        if self.renumbered:
            out.append(f"Gave {self.renumbered} units with a negative or repeated id a new one.")
        if self.kept_gate_rotations:
            out.append(f"Kept the stored rotation of {self.kept_gate_rotations} gates.")
        if self.elevation_steps:
            out.append(
                f"{self.elevation_steps} neighbouring tiles differ in elevation by more than 1, "
                "which AoE2:DE may refuse: smooth them before saving."
            )
        out += [f"Not converted: {line}" for line in self.refused + self.not_carried]
        return out


def legacy_text(value) -> str:
    """A legacy string as the game wrote it, cp1252, undoing the library's
    utf-8-then-latin-1 decode; one trailing NUL dropped."""
    if isinstance(value, bytes):
        raw = value
    elif value.isascii():
        return value.removesuffix("\x00")
    else:
        try:
            raw = value.encode("latin-1")
            raw.decode("utf-8")
            return value.removesuffix("\x00")  # bytes that are valid utf-8 were read as utf-8
        except UnicodeEncodeError:
            return value.removesuffix("\x00")  # past latin-1: a utf-8 read
        except UnicodeDecodeError:
            pass  # the latin-1 fallback: re-decode its bytes
    return raw.removesuffix(b"\x00").decode("cp1252", errors="replace")


def civ_to_de(value):
    """A legacy civilization int as DE's token, through the library's names."""
    if value in CIV_TOKENS:
        return CIV_TOKENS[value]
    try:
        return Civilization[CivilizationOld(value).name].value
    except (ValueError, KeyError):
        return value


def player_value_to_de(field_id: str, value):
    """The value DE's importer writes for one legacy player field."""
    if field_id in ("civilization", "architecture") and isinstance(value, int):
        return civ_to_de(value)
    if field_id == "starting_age":
        return legacy_import.starting_age_to_de(value)
    if field_id == "pop_limit":
        return DE_POP_LIMIT
    if field_id == "player_name_string_id" and value == STRING_ID_NONE[0]:
        return STRING_ID_NONE[1]
    if isinstance(value, str):
        return legacy_text(value)
    return value


def terrain_to_de(terrain_id: int, layer: int) -> tuple[int, int]:
    """(id, layer) DE's importer writes for a legacy tile."""
    return TERRAIN_REMAP.get(terrain_id, (terrain_id, layer))


def elevation_steps(width: int, height: int, elevations) -> int:
    """Neighbour pairs (along x and y) whose elevations differ by more than 1."""
    steps = 0
    for i, e in enumerate(elevations):
        if i % width + 1 < width and abs(elevations[i + 1] - e) > 1:
            steps += 1
        if i // width + 1 < height and abs(elevations[i + width] - e) > 1:
            steps += 1
    return steps


# -- units -------------------------------------------------------------------


def unit_rule(unit_const: int) -> str | None:
    """Which measured import rule sets this const's image, or None (as stored)."""
    if unit_const in legacy_import.TREE_CONSTS:
        return "trees"
    if unit_sprites.rotation_variant_eligible(unit_const):
        return "walls"
    frames = _frame_count(unit_const)
    if unit_const in unit_kind.fish_consts() and frames > 1:
        return "fish"
    if unit_kind.dat_type(unit_const) == _TYPE_BUILDING and frames and unit_const not in unit_sprites.cliff_const_set():
        return "gates" if unit_const in _gate_consts() else "buildings"
    return None


def _frame_count(unit_const: int) -> int:
    return int((unit_sprites.graphic_map().get(unit_const) or {}).get("frame_count") or 0)


def _gate_consts() -> frozenset[int]:
    return frozenset(c for group in gate_orientation.groups().values() for c in group)


def plan_unit_image(rule: str | None, unit_const: int, rotation, frame: int, rng: random.Random, wall_shape: int = 0):
    """(rotation, frame) for one unit under `rule` (unit_rule()). Walls take
    `wall_shape` (wall_shapes()); every other unmeasured value as stored."""
    if rule == "trees":
        drawn = legacy_import.tree_variant(rotation, rng.randrange(3))
        return (rotation, frame) if drawn is None else (drawn, int(drawn))
    if rule == "walls":
        return float(wall_shape), frame
    if rule == "fish":
        pick = rng.randrange(_frame_count(unit_const))
        return float(pick), pick
    if rule in ("buildings", "gates"):
        pick = rng.randrange(_frame_count(unit_const))
        return (rotation if rule == "gates" else 0.0), pick
    return rotation, frame


def wall_shapes(units, width: int, height: int) -> dict[int, int]:
    """id(unit) -> the shape its wall-connector neighbours give it (the tower
    when they give none), as the importer re-derives it (514 of 540 on Attila)."""
    connectors = unit_sprites.wall_connector_consts()
    tiles: set[tuple[int, int]] = set()
    for unit in units:
        if unit.unit_const in connectors:
            tiles.update(unit_occupied_tiles(unit, width, height) or ())
    shapes = {}
    for unit in units:
        if unit_sprites.rotation_variant_eligible(unit.unit_const):
            bounds = unit_tile_bounds(unit, width, height)
            mask = unit_sprites.neighbour_mask(bounds[0], bounds[2], tiles, diagonals=True) if bounds else 0
            shape = unit_sprites.wall_variant_from_neighbours8(mask)
            shapes[id(unit)] = unit_sprites.WALL_TOWER if shape is None else shape
    return shapes


def plan_units(legacy: LoadedScenario, rng: random.Random, report: ConversionReport) -> tuple[list[list[dict]], int]:
    """import_units()'s nine lists and counter. Drops consts the active game
    does not know, gives negative or repeated ids fresh ones, cuts links to
    a missing host, and sets each unit's image by its rule."""
    mm = legacy.map_manager
    lists = legacy.unit_manager.units
    kept = []
    for units in lists:
        row = []
        for unit in units:
            if unit_kind.dat_type(unit.unit_const) is None:
                report.dropped_consts[unit.unit_const] += 1
            else:
                row.append(unit)
        kept.append(row)
    flat = [u for row in kept for u in row]
    shapes = wall_shapes(flat, mm.map_width, mm.map_height)
    legacy_counter = _NEXT_UNIT_ID.unpack_from(legacy.decompressed_body, 0)[0]
    next_id = max([legacy_counter] + [u.reference_id + 1 for u in flat])
    seen: set[int] = set()
    ids: dict[int, int] = {}
    for unit in flat:
        rid = unit.reference_id
        if rid < 0 or rid in seen:
            rid, next_id = next_id, next_id + 1
            report.renumbered += 1
        seen.add(rid)
        ids[id(unit)] = rid
    aok = legacy.source_format.game == scenario_format.AOK
    out: list[list[dict]] = []
    for row in kept:
        specs = []
        for unit in row:
            host = unit.garrisoned_in_id
            if aok and host == 0:
                host = -1
                report.aok_links_cleared += 1
            elif host != -1 and (host not in seen or host == ids[id(unit)]):
                host = -1
                report.links_cut += 1
            rule = unit_rule(unit.unit_const)
            rotation, frame = plan_unit_image(
                rule, unit.unit_const, unit.rotation, unit.initial_animation_frame, rng, shapes.get(id(unit), 0)
            )
            if rule == "gates" and rotation:
                report.kept_gate_rotations += 1
            if (rotation, frame) != (unit.rotation, unit.initial_animation_frame):
                report.rewritten[rule] += 1
            specs.append(
                {
                    "unit_const": unit.unit_const, "x": unit.x, "y": unit.y, "z": unit.z, "status": unit.status,
                    "rotation": rotation, "initial_animation_frame": frame, "garrisoned_in_id": host,
                    "reference_id": ids[id(unit)],
                }
            )  # fmt: skip
        out.append(specs)
    return out, max(next_id, legacy_counter)


# -- players, options, messages ----------------------------------------------


def _set(model, field_id: str, value, label: str, report: ConversionReport) -> None:
    try:
        model.set_value(field_id, value)
    except (KeyError, ValueError) as e:
        report.refused.append(f"{label} ({value!r}): {e}")


def _shown(options: OptionsEditModel, key: str):
    try:
        return options.current_value(key)
    except KeyError:
        return None


def fill_options(legacy: LoadedScenario, options: OptionsEditModel, report: ConversionReport) -> None:
    """Map options, players, diplomacy, disables and custom AIs, remapped as the importer does."""
    target = {s.field_id for s in options.specs}
    for spec in option_fields.specs_for(legacy):
        if spec.field_id in target:
            _set(options, spec.field_id, option_fields.current_value(legacy, spec), spec.field_id, report)
    count = player_fields.defined_player_count(legacy)
    if count is not None:
        _set(options, player_fields.PLAYER_COUNT_FIELD_ID, count, "number of players", report)
    for field_id, value in DE_OPTION_DEFAULTS.items():
        if field_id in target:
            _set(options, field_id, value, field_id, report)
    legacy_map = legacy._scenario.sections["Map"].retriever_map
    if "unknown_1" in legacy_map and "ai_map_type" in target:
        _set(options, "ai_map_type", legacy_map["unknown_1"].data, "AI map type", report)
    for player_id in range(player_fields.NUM_PLAYERS + 1):
        values = {
            spec.field_id: player_value_to_de(spec.field_id, player_fields.current_value(legacy, spec, player_id))
            for spec in player_fields.specs_for_player(legacy, player_id)
        }
        values.update({k: v for k, v in _imported_player_values(legacy_map, values, player_id).items() if k not in values})
        for field_id, value in values.items():
            key = player_fields.player_field_id(field_id, player_id)
            if options.can_set(key):
                _set(options, key, value, f"player {player_id} {field_id}", report)
            elif field_id != "personality" and _shown(options, key) not in (None, value):
                report.refused.append(f"player {player_id} {field_id} ({value!r}): not writable, kept the template's")
    for row in range(1, diplomacy_fields.NUM_PLAYERS + 1):
        for col in range(1, diplomacy_fields.NUM_PLAYERS + 1):
            key = diplomacy_fields.stance_cell_id(row, col)
            if options.can_set(key):
                _set(options, key, diplomacy_fields.stance_value(legacy, row, col), key, report)
        key = diplomacy_fields.allied_victory_cell_id(row)
        if options.can_set(key):
            _set(options, key, diplomacy_fields.allied_victory_value(legacy, row), key, report)
    for key in disables_fields.all_field_ids():
        category, player_id = disables_fields.parse_disables_field_id(key)
        if options.can_set(key):
            _set(options, key, disables_fields.current_ids(legacy, category, player_id), key, report)
    _fill_ai(legacy, options, report)


def _imported_player_values(legacy_map: dict, values: dict, player_id: int) -> dict:
    """Player fields a legacy file has no slot for, as the importer fills
    them: architecture from the civilization, and every player's initial
    view from the Map section's one camera (stored y, x), GAIA's left unset."""
    out = {}
    if "civilization" in values:
        out["architecture"] = values["civilization"]
    if player_id and "player_1_camera_x" in legacy_map:
        out[player_fields.POV_X_FIELD] = legacy_map["player_1_camera_y"].data
        out[player_fields.POV_Y_FIELD] = legacy_map["player_1_camera_x"].data
    return out


def _fill_ai(legacy: LoadedScenario, options: OptionsEditModel, report: ConversionReport) -> None:
    """A custom AI (stored with its script) carries its name, type and text.
    A standard one becomes DEscape's built-in standard AI: DE writes its own
    RandomGame stub there, which DEscape does not ship."""
    layout = player_fields.personality_layout(legacy)
    if layout is None:
        return
    for player_id, stored in player_fields.stored_personalities(legacy, layout).items():
        if stored.ai_type == ai_scripts.AI_TYPE_STANDARD:
            choice = ai_scripts.STANDARD
        elif stored.text:
            choice = ai_scripts.stored_choice(
                f"legacy:{player_id}", legacy_text(stored.stored_name), stored.ai_type, stored.text, ()
            )
        else:
            if stored.stored_name:
                report.not_carried.append(f"player {player_id} AI {stored.stored_name!r} (its script is not in the file)")
            continue
        try:
            options.set_personality(player_id, choice)
        except (KeyError, ValueError) as e:
            report.refused.append(f"player {player_id} AI {choice.stored_name!r}: {e}")


def fill_messages(legacy: LoadedScenario, messages: MessagesEditModel, report: ConversionReport) -> None:
    """The six texts re-decoded as cp1252, newlines as stored, and their string ids."""
    retrievers = legacy._scenario.sections["Messages"].retriever_map
    for spec in messages_fields.present_fields(retrievers):
        text = legacy_text(retrievers[spec.retriever].data)
        if len(text.encode("utf-8")) > messages_fields.MAX_FIELD_BYTES:
            text = text.encode("utf-8")[: messages_fields.MAX_FIELD_BYTES].decode("utf-8", errors="ignore")
            report.refused.append(f"{spec.label} text: cut to the {messages_fields.MAX_FIELD_BYTES}-byte limit")
        _set(messages, spec.field_id, text, f"{spec.label} text", report)
        if not messages_fields.has_string_id(spec, retrievers):
            continue
        string_id = retrievers[spec.id_retriever].data
        string_id = messages_fields.STRING_ID_UNSET if string_id == legacy_import.U32_MINUS_ONE else string_id
        _set(messages, f"{spec.field_id}_id", string_id, f"{spec.label} string id", report)


def _not_carried(legacy: LoadedScenario, report: ConversionReport) -> None:
    sections = legacy._scenario.sections
    picture = sections.get("BackgroundImage")
    if picture is not None and picture.retriever_map["ascii_filename"].data:
        report.not_carried.append(f"background image {picture.retriever_map['ascii_filename'].data!r}")
    cinematics = sections.get("Cinematics")
    if cinematics is not None:
        for name, r in cinematics.retriever_map.items():
            if isinstance(r.data, str) and r.data.strip() not in _NO_CINEMATIC:
                report.not_carried.append(f"{name.removeprefix('ascii_')} cinematic {r.data.strip()!r}")


# -- the whole document ------------------------------------------------------


def legacy_trigger_count(legacy: LoadedScenario) -> int:
    """The trigger count a legacy file stores (the s32 at [9:13] of its trigger tail)."""
    tail = legacy.trigger_tail
    return max(0, int.from_bytes(tail[9:13], "little", signed=True)) if len(tail) >= 13 else 0


def _patch_cameras(data: bytes, legacy: LoadedScenario, display_path: str, report: ConversionReport) -> bytes:
    """The legacy player_data_3 cameras written over the converted file's, in
    place, each clamped onto the map like clamp_cameras()."""
    out = load_map_and_units_from_bytes(data, display_path)
    fields = player_fields.camera_fields(out)
    pd3 = legacy._scenario.sections["Units"].retriever_map.get("player_data_3")
    names = _PLAYER_DATA_3_CAMERAS  # camera_fields()'s per-entry order
    if fields is None or pd3 is None or len(fields) < len(names) * len(pd3.data):
        report.not_carried.append("player cameras (the new file's could not be located)")
        return data
    body = bytearray(out.decompressed_body)
    for i, entry in enumerate(pd3.data):
        for k, name in enumerate(names):
            target, axis = fields[i * len(names) + k].target, fields[i * len(names) + k].axis
            value = entry.retriever_map[name].data
            bound = (out.map_manager.map_width if axis == "x" else out.map_manager.map_height) - 1
            value = min(value, bound) if value >= 0 else value
            _CAMERA[target.codec].pack_into(body, target.offset, float(value) if target.codec == "f32" else int(value))
    return out.header_bytes + _compress_bytes(bytes(body))


def converted_name(legacy: LoadedScenario) -> str:
    """The converted document's file name: the legacy stem as .aoe2scenario."""
    return f"{legacy.path.stem if legacy.path else 'converted'}.aoe2scenario"


def convert(legacy: LoadedScenario, *, rng: random.Random | None = None) -> tuple[bytes, ConversionReport]:
    """The converted .aoe2scenario bytes and what the conversion did. Only
    reads `legacy`; raises ValueError for a document that is not legacy."""
    if not legacy.source_format.legacy:
        raise ValueError("Only a pre-DE document converts to .aoe2scenario")
    if not scenario_format.converts(legacy.source_format):
        raise ValueError(f"{legacy.source_format.display_name} files do not convert to .aoe2scenario")
    rng = random.Random() if rng is None else rng
    report = ConversionReport(triggers_left_out=legacy_trigger_count(legacy))
    if not legacy_import.applies(legacy):
        report.unmeasured_band = legacy.source_format.display_name
    try:
        return _convert(legacy, rng, report)
    except Exception:
        reactivate(legacy)  # the donor and the new document loaded over its library state
        raise


def _convert(legacy: LoadedScenario, rng: random.Random, report: ConversionReport) -> tuple[bytes, ConversionReport]:
    mm = legacy.map_manager
    with game_dataset.activated(games.default().id):
        donor = load_map_and_units(scenario_new.blank_template_path_for(games.default().id))
        body = scenario_new.blank_body(donor, mm.map_width, mm.map_height)
        name = converted_name(legacy)
        new = load_map_and_units_from_bytes(donor.header_bytes + _compress_bytes(body), name)
        for src, dst in zip(mm.terrain, new.map_manager.terrain, strict=True):
            dst.terrain_id, dst.layer = terrain_to_de(src.terrain_id, getattr(src, "layer", -1))
            dst.elevation = src.elevation
        report.elevation_steps = elevation_steps(mm.map_width, mm.map_height, [t.elevation for t in mm.terrain])
        options = OptionsEditModel(new)
        fill_options(legacy, options, report)
        messages = MessagesEditModel(new)
        fill_messages(legacy, messages, report)
        units = UnitEditModel(new)
        lists, counter = plan_units(legacy, rng, report)
        units.import_units(lists, counter)
        _not_carried(legacy, report)
        header, out = build_patched_body(new, options=options, units=units, messages=messages)
        data = _patch_cameras(header + _compress_bytes(out), legacy, name, report)
    return data, report


def load_converted(legacy: LoadedScenario, *, rng: random.Random | None = None) -> tuple[LoadedScenario, ConversionReport]:
    """convert(), reloaded through the real loader as an untitled document."""
    data, report = convert(legacy, rng=rng)
    return load_map_and_units_from_bytes(data, converted_name(legacy)), report


__all__ = [
    "CIV_TOKENS",
    "DE_POP_LIMIT",
    "TERRAIN_REMAP",
    "ConversionReport",
    "civ_to_de",
    "convert",
    "converted_name",
    "elevation_steps",
    "legacy_text",
    "legacy_trigger_count",
    "load_converted",
    "plan_unit_image",
    "plan_units",
    "player_value_to_de",
    "terrain_to_de",
    "unit_rule",
    "wall_shapes",
]
