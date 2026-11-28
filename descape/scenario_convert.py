"""Convert an open scenario to the newest file version DEscape writes (GH #307).

Qt-free, the same relationship to descape/scenario_write.py that
descape/scenario_resize.py has. One step exists, 1.58 to 1.59, as a table
keyed by source version, so the next game update adds a row rather than a
branch. The September 2026 update's whole format delta is three hunks: each
unit gains `capture_flag` (s8) after `garrisoned_in_id`, each condition gains
`allow_in_fog` (s32) right before `xs_function` with its field count 33 to
34, and the version carriers move (FileHeader.version, DataHeader.version,
the Triggers section's `trigger_version` 4.9 to 5.0). Effects are unchanged.

The pipeline, and why each step sits where it does:

1. scenario_write.build_patched() builds the document's current body, every
   pending edit included, and it is reloaded as `current`: a clean parse of
   the document as it stands, at fresh offsets.
2. A byte transform of that body: overwrites first (the two body carriers
   and each condition's count), then one insert per unit and per condition
   at its default value. Never a re-serialization through the library, which
   writes an empty str32 one byte longer than the game does.
3. The result is reloaded as `converted` and verify_conversion() checks it
   against `current` field by field, from each load's own unit layout and
   section retrievers, so a wrong insert offset cannot pass by re-checking
   itself.

Conditions are located from each trigger's end, never by a forward walk:
retriever_length() re-serializes a trigger's str32 fields and drifts, while
each parsed entry's byte_length is exact and condition_data is followed only
by condition_display_order_array (one s32 per condition).

The library's version state is process-global and the last load wins.
`converted` is the last load when convert_scenario() returns; a caller that
loads anything else afterwards calls scenario_io.reactivate(result.loaded)
before editing it. On failure the source document is reactivated instead.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from AoE2ScenarioParser.exceptions.asp_exceptions import EndOfFileError

from descape import library_compat, scenario_io
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_io import (
    LoadedScenario,
    UnsupportedStructureVersion,
    load_map_and_units_from_bytes,
    parse_triggers,
    retriever_length,
    unit_codec,
)
from descape.scenario_write import WriteBlockedError, _compress_bytes, build_patched, refuse_legacy
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")
_S32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_DATA_VERSION_AT = 4  # DataHeader.version, an f32 after next_unit_id_to_place
_CONDITION_HEAD = 8  # condition_type, then the field count


@dataclass(frozen=True)
class Step:
    """One version step: what changes between `source` and `target`."""

    source: str
    target: str
    trigger_versions: tuple[float, float]
    unit_field: str  # inserted right after unit_after
    unit_after: str
    unit_fill: bytes
    condition_counts: tuple[int, int]
    condition_field: str  # appended as the last counted field, before xs_function
    condition_fill: bytes


STEPS: dict[str, Step] = {
    "1.58": Step(
        source="1.58",
        target="1.59",
        trigger_versions=(4.9, 5.0),
        unit_field="capture_flag",
        unit_after="garrisoned_in_id",
        unit_fill=struct.pack("<b", -1),
        condition_counts=(33, 34),
        condition_field="allow_in_fog",
        condition_fill=_S32.pack(-1),
    ),
}

# The newest version with a vocabulary; tests/test_scenario_convert.py trips when a game update adds one.
TARGET_VERSION = "1.59"


class ConversionRefusedError(ValueError):
    """Raised by convert_scenario() for a document source_refusal() refuses."""


@dataclass
class ConversionResult:
    """The converted document, reloaded. Saving it means write_scenario(loaded, path)."""

    loaded: LoadedScenario
    data: bytes
    source_version: str
    target_version: str
    units: int
    conditions: int


def source_refusal(scenario_version: str) -> str | None:
    """Why a `scenario_version` file cannot be converted, or None."""
    if scenario_version == TARGET_VERSION:
        return f"This scenario is already version {TARGET_VERSION}."
    if scenario_version not in STEPS:
        sources = " or ".join(STEPS)
        return f"Only version {sources} scenarios can be converted; this one is {scenario_version}."
    return None


def action_label() -> str:
    return f"Convert to Version {TARGET_VERSION}…"


def confirm_text(name: str | None, source_version: str) -> str:
    """The confirm dialog's text. `name` is the file Save would overwrite, None for an untitled document."""
    save = (
        f"Nothing is written until you save: Save overwrites {name} as {TARGET_VERSION} (Save As keeps the "
        f"{source_version} file)."
        if name
        else "Nothing is written until you save."
    )
    return (
        f"This scenario is version {source_version} and becomes {TARGET_VERSION}, the format the "
        f"September 2026 game update saves. A {TARGET_VERSION} scenario does not open in an older game.\n\n"
        "The conversion cannot be undone and clears the undo history. Unsaved changes are kept. "
        f"{save}\n\nConvert?"
    )


def _structure_json(version: str) -> dict:
    for base in (library_compat.VERSIONS_DIR, library_compat.REPO_VERSIONS_DIR):
        path = base / f"v{version}" / "structure.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No structure.json for scenario version {version}")


def unit_struct_retrievers(version: str) -> dict:
    """The UnitStruct retrievers of `version`'s structure.json, in field order."""
    units = _structure_json(version)["Units"]
    player_units = units["structs"][units["retrievers"]["players_units"]["type"].removeprefix("struct:")]
    return player_units["structs"][player_units["retrievers"]["units"]["type"].removeprefix("struct:")]["retrievers"]


@cache
def unit_insert_offset(step: Step) -> int:
    """Where `step.unit_field` lands in a unit's bytes: the fixed widths before it in the target struct."""
    retrievers = unit_struct_retrievers(step.target)
    names = list(retrievers)
    before = names[: names.index(step.unit_field)]
    if not before or before[-1] != step.unit_after:
        raise ValueError(f"{step.target}'s {step.unit_field} does not follow {step.unit_after}")
    codes = scenario_io._FIXED_WIDTH_CODES
    return struct.calcsize("<" + "".join(codes[retrievers[name]["type"]] for name in before))


@dataclass(frozen=True)
class _TriggerSpan:
    start: int
    length: int
    region: int  # where condition_data starts
    conditions: tuple[tuple[int, int], ...]  # (offset, byte_length) per condition
    entry: Any


def _trigger_layout(loaded: LoadedScenario) -> tuple[int, int, list[_TriggerSpan]]:
    """(trigger_data start, its end, one span per trigger) in loaded's body. Raises WriteBlockedError."""
    body = loaded.decompressed_body
    retrievers = loaded._scenario.sections["Triggers"].retriever_map
    pos = loaded.units_section_end
    for name, retriever in retrievers.items():
        if name == "trigger_data":
            break
        pos += retriever_length(retriever)
    start = pos
    spans = []
    for index, entry in enumerate(retrievers["trigger_data"].data or []):
        fields = entry.retriever_map
        if list(fields)[-2:] != ["condition_data", "condition_display_order_array"]:
            raise WriteBlockedError("This version's triggers do not end with their conditions.")
        conditions = fields["condition_data"].data or []
        region = pos + entry.byte_length - 4 * len(conditions) - sum(c.byte_length for c in conditions)
        if region < pos + 4 or _S32.unpack_from(body, region - 4)[0] != len(conditions):
            raise WriteBlockedError(f"Trigger {index}'s conditions could not be located.")
        offsets = []
        at = region
        for condition in conditions:
            offsets.append((at, condition.byte_length))
            at += condition.byte_length
        spans.append(_TriggerSpan(pos, entry.byte_length, region, tuple(offsets), entry))
        pos += entry.byte_length
    return start, pos, spans


def _check_carriers(loaded: LoadedScenario, step: Step, which: int) -> str | None:
    body = loaded.decompressed_body
    version = (step.source, step.target)[which]
    trigger_version = step.trigger_versions[which]
    if loaded.header_bytes[:4] != version.encode("ascii"):
        return f"the file header does not read {version}"
    if body[_DATA_VERSION_AT : _DATA_VERSION_AT + 4] != _F32.pack(float(version)):
        return f"the data header does not read {version}"
    if body[loaded.units_section_end : loaded.units_section_end + 8] != _F64.pack(trigger_version):
        return f"the trigger version does not read {trigger_version}"
    return None


def _source_gate(current: LoadedScenario, step: Step) -> None:
    """Raises ConversionRefusedError when `current`'s units or triggers cannot be located."""
    if not current.units_write_supported or current.unit_spans is None or unit_codec(current) is None:
        raise ConversionRefusedError("This scenario's objects could not be verified, so they cannot be converted.")
    if parse_triggers(current) is None or not current.trigger_write_supported:
        raise ConversionRefusedError("This scenario's triggers could not be verified, so they cannot be converted.")
    if reason := _check_carriers(current, step, 0):
        raise ConversionRefusedError(f"This scenario cannot be converted: {reason}.")


def convert_body(current: LoadedScenario, step: Step) -> tuple[bytes, int, int]:
    """(converted body, units, conditions) for a `current` that passed _source_gate()."""
    body = current.decompressed_body
    old_count, new_count = step.condition_counts
    overwrites = [
        (_DATA_VERSION_AT, _F32.pack(float(step.target))),
        (current.units_section_end, _F64.pack(step.trigger_versions[1])),
    ]
    offset = unit_insert_offset(step)
    inserts = [(start + offset, step.unit_fill) for player in current.unit_spans for start, _length in player]
    units = len(inserts)
    conditions = 0
    for trigger in _trigger_layout(current)[2]:
        for at, length in trigger.conditions:
            count = _S32.unpack_from(body, at + 4)[0]
            if count != old_count:
                raise WriteBlockedError(f"A condition holds {count} fields, not {old_count}.")
            xs_at = at + _CONDITION_HEAD + 4 * count
            if xs_at + 4 + _U32.unpack_from(body, xs_at)[0] != at + length:
                raise WriteBlockedError("A condition's length does not match its fields.")
            overwrites.append((at + 4, _S32.pack(new_count)))
            inserts.append((xs_at, step.condition_fill))
            conditions += 1
    patched = bytearray(body)
    for at, data in overwrites:
        patched[at : at + len(data)] = data
    out = bytearray()
    last = 0
    for at, data in sorted(inserts):
        out += patched[last:at]
        out += data
        last = at
    out += patched[last:]
    return bytes(out), units, conditions


def _field_spans(loaded: LoadedScenario) -> dict[str, tuple[int, int]]:
    """name -> (offset, size) of each fixed unit field in loaded's own layout."""
    unit_format = unit_codec(loaded).unit_format
    codes = unit_format.fixed.format.removeprefix("<")
    spans = {}
    at = 0
    for name, code in zip(unit_format.fields, codes, strict=True):
        size = struct.calcsize("<" + code)
        spans[name] = (at, size)
        at += size
    return spans


def _outside(body: bytes, start: int, end: int, spans: Sequence[Sequence[tuple[int, int]]]) -> bytes:
    """body[start:end] minus every unit span: per-player counts and the trailer."""
    out = bytearray()
    at = start
    for player in spans:
        for offset, length in player:
            out += body[at:offset]
            at = offset + length
    return bytes(out + body[at:end])


def _verify_units(current: LoadedScenario, converted: LoadedScenario, step: Step) -> None:
    if [len(p) for p in current.unit_spans] != [len(p) for p in converted.unit_spans]:
        raise WriteBlockedError("The converted file's per-player object counts differ.")
    old, new = current.decompressed_body, converted.decompressed_body
    old_fields, new_fields = _field_spans(current), _field_spans(converted)
    if set(new_fields) - set(old_fields) != {step.unit_field}:
        raise WriteBlockedError(f"The converted objects gained {sorted(set(new_fields) - set(old_fields))}.")
    old_size, new_size = unit_codec(current).unit_format.fixed.size, unit_codec(converted).unit_format.fixed.size
    for old_player, new_player in zip(current.unit_spans, converted.unit_spans, strict=True):
        for (a, a_len), (b, b_len) in zip(old_player, new_player, strict=True):
            for name, (offset, size) in old_fields.items():
                b_offset = new_fields[name][0]
                if old[a + offset : a + offset + size] != new[b + b_offset : b + b_offset + size]:
                    raise WriteBlockedError(f"An object's {name} changed in conversion.")
            flag_at = b + new_fields[step.unit_field][0]
            if new[flag_at : flag_at + len(step.unit_fill)] != step.unit_fill:
                raise WriteBlockedError(f"An object's {step.unit_field} is not the default.")
            if old[a + old_size : a + a_len] != new[b + new_size : b + b_len]:
                raise WriteBlockedError("An object's caption changed in conversion.")
    if _outside(old, current.units_block_offset, current.units_section_end, current.unit_spans) != _outside(
        new, converted.units_block_offset, converted.units_section_end, converted.unit_spans
    ):
        raise WriteBlockedError("The Units section's counts or trailer changed in conversion.")


def _verify_triggers(current: LoadedScenario, converted: LoadedScenario, step: Step) -> None:
    old, new = current.decompressed_body, converted.decompressed_body
    old_start, old_end, old_triggers = _trigger_layout(current)
    new_start, new_end, new_triggers = _trigger_layout(converted)
    if old[current.units_section_end + 8 : old_start] != new[converted.units_section_end + 8 : new_start]:
        raise WriteBlockedError("The Triggers header changed in conversion.")
    if len(old_triggers) != len(new_triggers):
        raise WriteBlockedError("The converted file's trigger count differs.")
    fill = _S32.unpack(step.condition_fill)[0]
    for index, (a, b) in enumerate(zip(old_triggers, new_triggers, strict=True)):
        n = len(a.conditions)
        if old[a.start : a.region] != new[b.start : b.region] or b.length != a.length + 4 * n or len(b.conditions) != n:
            raise WriteBlockedError(f"Trigger {index} changed outside its conditions.")
        if old[a.start + a.length - 4 * n : a.start + a.length] != new[b.start + b.length - 4 * n : b.start + b.length]:
            raise WriteBlockedError(f"Trigger {index}'s condition order changed.")
        pairs = zip(a.entry.retriever_map["condition_data"].data or [], b.entry.retriever_map["condition_data"].data or [], strict=True)
        for source, target in pairs:
            fields = target.retriever_map
            # The count against the parsed struct's own s32 fields (all but type, count and xs_function), not the step's.
            if fields["static_value_33"].data != len(fields) - 3 or fields[step.condition_field].data != fill:
                raise WriteBlockedError(f"Trigger {index} has a condition without the {TARGET_VERSION} fields.")
            for name, retriever in source.retriever_map.items():
                if name != "static_value_33" and fields[name].data != retriever.data:
                    raise WriteBlockedError(f"Trigger {index} has a condition whose {name} changed.")
    if old[old_end : current.triggers_section_end] != new[new_end : converted.triggers_section_end]:
        raise WriteBlockedError("The data after the triggers changed in conversion.")


def verify_conversion(current: LoadedScenario, converted: LoadedScenario, step: Step, units: int, conditions: int) -> None:
    """Raises WriteBlockedError naming the first way `converted` is not `current` at `step.target`."""
    if reason := _check_carriers(converted, step, 1):
        raise WriteBlockedError(f"The converted file failed to verify: {reason}.")
    if len(converted.decompressed_body) - len(current.decompressed_body) != units + 4 * conditions:
        raise WriteBlockedError("The converted file grew by an unexpected number of bytes.")
    for gate in ("terrain_write_supported", "units_write_supported", "messages_write_supported"):
        if getattr(current, gate) and not getattr(converted, gate):
            raise WriteBlockedError(f"The converted file failed its {gate.removesuffix('_supported')} check.")
    if parse_triggers(converted) is None or not converted.trigger_write_supported or unit_codec(converted) is None:
        raise WriteBlockedError("The converted file's triggers or objects failed to verify.")
    old, new = current.decompressed_body, converted.decompressed_body
    lo, hi = _DATA_VERSION_AT, _DATA_VERSION_AT + 4
    if old[:lo] + old[hi : current.units_block_offset] != new[:lo] + new[hi : converted.units_block_offset]:
        raise WriteBlockedError("The data before the objects changed in conversion.")
    if old[current.triggers_section_end :] != new[converted.triggers_section_end :]:
        raise WriteBlockedError("The data after the triggers changed in conversion.")
    _verify_units(current, converted, step)
    _verify_triggers(current, converted, step)


def convert_scenario(
    scenario: LoadedScenario,
    *,
    triggers: TriggerEditModel | None = None,
    options: OptionsEditModel | None = None,
    units: UnitEditModel | None = None,
    messages: MessagesEditModel | None = None,
) -> ConversionResult:
    """Converts `scenario` with its pending edits (the models write_scenario()
    takes) to TARGET_VERSION. Raises ConversionRefusedError for what
    source_refusal() refuses or a document whose units or triggers cannot be
    located, WriteBlockedError when the result fails to verify. `scenario`
    is left untouched. The result is the library's last load: a caller that
    loads anything else afterwards calls scenario_io.reactivate(result.loaded)
    before editing it."""
    refuse_legacy(scenario)
    if reason := source_refusal(scenario.scenario_version):
        raise ConversionRefusedError(reason)
    step = STEPS[scenario.scenario_version]
    try:
        patched = build_patched(scenario, triggers, options, units, messages)
        current = load_map_and_units_from_bytes(patched.header + _compress_bytes(patched.body), scenario.path)
        _source_gate(current, step)
        body, unit_count, condition_count = convert_body(current, step)
        data = step.target.encode("ascii") + current.header_bytes[4:] + _compress_bytes(body)
        try:
            converted = load_map_and_units_from_bytes(data, scenario.path)
        except (EndOfFileError, struct.error, ValueError, UnsupportedStructureVersion) as e:
            raise WriteBlockedError(f"The converted file does not load: {type(e).__name__}: {e}") from e
        verify_conversion(current, converted, step, unit_count, condition_count)
    except Exception:
        scenario_io.reactivate(scenario)
        raise
    return ConversionResult(converted, data, step.source, step.target, unit_count, condition_count)


__all__ = [
    "STEPS",
    "TARGET_VERSION",
    "ConversionRefusedError",
    "ConversionResult",
    "Step",
    "action_label",
    "confirm_text",
    "convert_body",
    "convert_scenario",
    "source_refusal",
    "unit_insert_offset",
    "unit_struct_retrievers",
    "verify_conversion",
]
