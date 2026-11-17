"""Qt-free sentences for the Triggers panel's Detail column (GH #388).

Each condition and effect type has a Short and a Full template in
trigger_summaries.json, keyed by type name like trigger_requirements.json,
since names are stable across versions and ids are not. A template is text
with placeholders:

- `{field}`: the field's value, rendered by its vocabulary presentation
  (a player, an object, an enum word, quoted text). Unset reads "?".
- `{field!}`: only when the value differs from the type's default;
  `{field!0}` also hides 0 (any comma-separated ints).
- `{field?|on text|off text}`: a yes/no field as one of two texts.
- `{field:Enum}`: a plain int read through a named enum (DifficultyLevel).
- `{area}`, `{location}`, `{wall}`: coordinates, a link to centre the map.
- `{obj}`: the object filter (selected objects, or player plus object,
  class or type, plus area and max count); `{objtype}` without area and max;
  `{objname}` without the player as well.
- `{amount}`: the quantity slot the library will write (armour and attack,
  float, or plain).
- `{op|VALUE|TARGET}`: the operation, Short `TARGET +VALUE`, Full
  `Add VALUE to TARGET`. VALUE and TARGET are templates themselves.
- `{either|A|B}`: A, or B when A has an unset placeholder.
- `{cost}`: the set resource amounts, comma-joined.
- `{hp}`: " HP" when the show_hp option is on, else nothing.
- `[...]`: dropped when any placeholder inside is unset.

An entry's `"object_link": "location"` sends its object link to the location
instead (an object that is about to be created), and `"link_as_object"`
gives the fields it names the `{obj}` link (a replacement points at what it
replaces). A condition's `inverted`
adds `not ` (Full: `Not: `) on its own. A field the
template does not show and the JSON's `omit` does not name is appended as
`+ label: value` when it differs from the type's default, so nothing set is
ever hidden. A type with no template (a newer version) reads `label: value`
pairs.

Read-only: nothing here writes an entry.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

from AoE2ScenarioParser.datasets import trigger_lists

from descape import message_markup, player_labels, trigger_fields
from descape.trigger_geometry import AREA_FIELDS, LOCATION_FIELDS, LOCATION_OBJECT_FIELD, SELECTED_OBJECTS_FIELD, WALL_FIELDS

TABLE_PATH = Path(__file__).resolve().parent / "trigger_summaries.json"

READABLE, RAW = "readable", "raw"
SHORT, FULL = "short", "full"
PLAYER_NUMBER, PLAYER_FULL, PLAYER_NAME = "number", "full", "name"

KIND_KEYS = {"condition": "conditions", "effect": "effects"}
TYPE_ATTRIBUTES = {"condition": "condition_type", "effect": "effect_type"}
INVERTED = "inverted"

# Link targets a Segment carries; the window follows them (viewer_trigger_overlay.follow_trigger_link()).
LINK_AREA, LINK_TILE, LINK_UNIT, LINK_UNITS, LINK_MATCH, LINK_TRIGGER = "area", "tile", "unit", "units", "match", "trigger"

# Fields whose presentation is "" in some versions but hold the same kind of value.
_PRESENTATION_FALLBACK = {
    "unit_object": "Unit",
    "next_object": "Unit",
    "trigger_id": "TriggerId",
    "variable": "VariableId",
    "variable2": "VariableId",
    "allow_in_fog": "bool",
}
_OBJECT_CONST_FIELDS = ("object_list_unit_id", "object_list")
_FILTER_FIELDS = ("source_player", "object_group", "object_type")
_AMOUNT_FIELDS = ("quantity", "quantity_float", "armour_attack_quantity", "armour_attack_class")
_PSEUDO_FIELDS: Mapping[str, tuple[str, ...]] = {
    "area": AREA_FIELDS,
    "location": (*LOCATION_FIELDS, LOCATION_OBJECT_FIELD),
    "wall": WALL_FIELDS,
    "obj": (*_OBJECT_CONST_FIELDS, *_FILTER_FIELDS, SELECTED_OBJECTS_FIELD, *AREA_FIELDS, "max_units_affected"),
    "objtype": (*_OBJECT_CONST_FIELDS, *_FILTER_FIELDS),
    "objname": (*_OBJECT_CONST_FIELDS, *_FILTER_FIELDS),
    "amount": _AMOUNT_FIELDS,
    "op": ("operation",),
    "either": (),
    "hp": (),
    "cost": ("food", "wood", "stone", "gold", *(f"resource_{n}{suffix}" for n in (1, 2, 3) for suffix in ("", "_quantity"))),
}
_COST_FIELDS = (("food", "Food"), ("wood", "Wood"), ("stone", "Stone"), ("gold", "Gold"))

# A yes/no field's word when on: (Short, Full).
_FLAG_WORDS: Mapping[str, tuple[str, str]] = {
    "force_research_technology": ("forced", "forced"),
    "mutual_diplomacy": ("mutual", "both ways"),
    "global_sound": ("global", "heard everywhere"),
    "flash_object": ("flash", "flashing"),
    "queue_action": ("queued", "queued"),
    "issue_group_command": ("group command", "as a group command"),
    "scroll": ("scroll", "scrolling"),
    "reset_timer": ("reset", "resetting it"),
    "disable_sound": ("silent", "silently"),
    "disable_garrison_unload_sound": ("no unload sound", "without the unload sound"),
    "play_sound": ("sound", "with its sound"),
    "use_tag_color_for_icon": ("tag colour icon", "with the tag colour icon"),
    "include_changeable_weapon_objects": ("incl. other weapon mode", "counting the other weapon mode"),
    "allow_in_fog": ("in fog", "even in fog"),
}

_OPERATION_SHORT = {1: "{t} set to {v}", 2: "{t} +{v}", 3: "{t} -{v}", 4: "{t} x{v}", 5: "{t} /{v}"}
_OPERATION_FULL = {1: "Set {t} to {v}", 2: "Add {v} to {t}", 3: "Take {v} from {t}", 4: "Multiply {t} by {v}", 5: "Divide {t} by {v}"}
_COMPARISON_SHORT = {0: "=", 1: "<", 2: ">", 3: "<=", 4: ">="}
_COMPARISON_FULL = {0: "exactly", 1: "less than", 2: "more than", 3: "at most", 4: "at least"}
_RESOURCE_WORDS = {0: "Food", 1: "Wood", 2: "Stone", 3: "Gold"}
_DAMAGE_CLASS_WORDS = {3: "Pierce", 4: "Melee"}
# Presentations whose words start with a capital; the rest read lower case.
_CAPITALISED = frozenset({"ObjectClass", "ObjectType", "DamageClass", "ColorMood", "PlayerColorId", "Attribute"})
# Enums a template names with `{field:Enum}`, for plain ints the vocabulary gives no presentation: (enum, capitalised).
_NAMED_ENUMS = {
    "DifficultyLevel": (trigger_lists.DifficultyLevel, True),
    "TechnologyState": (trigger_lists.TechnologyState, False),
    "DiplomacyState": (trigger_lists.DiplomacyState, False),
}


@dataclass(frozen=True)
class Options:
    """Settings > Appearance's condition and effect details (settings.get_trigger_details())."""

    mode: str = READABLE
    sentence_style: str = SHORT
    player_text: str = PLAYER_NUMBER
    # The player colour as a box before the player, the text itself, both, or "off".
    player_colour: str = "box"
    show_off_flags: bool = False
    show_ids: bool = False
    text_cap: int = 40
    link_coords: bool = True
    link_objects: bool = True
    link_triggers: bool = True
    # A placed object's reference id after it, and how many of several selected ids to list (0: the count only).
    show_unit_ids: bool = True
    # "HP" after a heal or damage amount; off by default, since "1000 HP" reads as a gain on a damage row.
    show_hp: bool = False
    # Quoted text's <COLOR> tags: "colour" draws the text in it, "raw" shows the tags as written.
    text_tags: str = "colour"
    unit_id_limit: int = 0
    # An effect with both selected objects and an area reads the area's filter, the objects, or both in either order.
    selected_with_area: str = "area"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Options:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class Context:
    """What a sentence reads besides the entry: player labels, and the open
    document's trigger names, variable names and placed units."""

    labels: player_labels.PlayerLabels = player_labels.DEFAULT_LABELS
    # trigger_id -> its name, or None when no trigger has that id.
    trigger_name: Callable[[int], str | None] = lambda _id: None
    variable_name: Callable[[int], str] = lambda _id: ""
    # reference_id -> unit_references.UnitRef or None.
    unit: Callable[[int], Any] = lambda _id: None


@dataclass(frozen=True)
class ObjectFilter:
    """What an object link matches on the map; -1 or None means any."""

    player: int = -1
    unit_const: int = -1
    group: int = -1
    object_type: int = -1
    area: tuple[int, int, int, int] | None = None
    label: str = ""


@dataclass(frozen=True)
class Segment:
    text: str
    link: tuple | None = None
    # Player whose colour swatch goes before the text.
    player: int | None = None
    # Player the text names, swatch or not, so a sentence knows which players it shows.
    pid: int | None = field(default=None, compare=False)
    # How `player`'s colour shows: "box", "text" or "both".
    mark: str = field(default="box", compare=False)
    # A colour tag's text colour (message_markup), for quoted text.
    rgb: tuple[int, int, int] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Summary:
    segments: tuple[Segment, ...]
    raw: str
    text: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", "".join(segment.text for segment in self.segments))


# -- the table ---------------------------------------------------------------


@cache
def load_table(path: Path = TABLE_PATH) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """kind key ("conditions"/"effects") -> type name -> {"short", "full", "omit"}."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class _Text:
    text: str


@dataclass(frozen=True)
class _Field:
    name: str
    modifier: str
    args: tuple[tuple, ...]


@dataclass(frozen=True)
class _Optional:
    parts: tuple


def _split_top(text: str, separator: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _closing(text: str, start: int) -> int:
    """Index of the bracket closing the one at `start`; ValueError when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] in "{[":
            depth += 1
        elif text[i] in "}]":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced template: {text!r}")


@cache
def parse(template: str) -> tuple:
    """A template as a tuple of _Text, _Field and _Optional parts."""
    parts: list = []
    i = 0
    while i < len(template):
        char = template[i]
        if char in "{[":
            end = _closing(template, i)
            inner = template[i + 1 : end]
            if char == "[":
                parts.append(_Optional(parse(inner)))
            else:
                head, *args = _split_top(inner, "|")
                name, modifier = head, ""
                for mark in ("!", "?", ":"):
                    if mark in head:
                        name, rest = head.split(mark, 1)
                        modifier = mark + rest
                        break
                if not name.replace("_", "").isalnum():
                    raise ValueError(f"bad placeholder {{{inner}}} in {template!r}")
                parts.append(_Field(name, modifier, tuple(parse(arg) for arg in args)))
            i = end + 1
        elif char in "}]":
            raise ValueError(f"unbalanced template: {template!r}")
        else:
            end = i
            while end < len(template) and template[end] not in "{[}]":
                end += 1
            parts.append(_Text(template[i:end]))
            i = end
    return tuple(parts)


def placeholders(template: str) -> set[str]:
    """Every placeholder name a template uses, nested ones included."""
    found: set[str] = set()

    def walk(parts) -> None:
        for part in parts:
            if isinstance(part, _Field):
                found.add(part.name)
                for arg in part.args:
                    walk(arg)
            elif isinstance(part, _Optional):
                walk(part.parts)

    walk(parse(template))
    return found


def covered_fields(template: str) -> set[str]:
    """The vocabulary fields a template shows, pseudo-fields expanded."""
    found: set[str] = set()
    for name in placeholders(template):
        found.update(_PSEUDO_FIELDS.get(name, (name,)))
    return found


# -- rendering ---------------------------------------------------------------


def _is_unset(value: Any) -> bool:
    """-1, None, "" and an empty list are the library's "unset" across trigger fields."""
    if isinstance(value, bool):
        return False
    return value is None or value == trigger_fields.UNSET or value == "" or (isinstance(value, (list, tuple)) and not value)


def _number(value: Any) -> str:
    # :g, as the form shows a float: quantity_float is an f32, so 1.2 reads back as 1.2000000476837158.
    return f"{value:g}" if isinstance(value, float) else str(value)


def _hidden(modifier: str) -> tuple[int, ...]:
    """The extra values a `{field!0,1}` modifier hides."""
    return tuple(int(v) for v in modifier[1:].split(",") if v.strip().lstrip("-").isdigit())


def _sentence_case(label: str) -> str:
    label = label.lower()
    return label[:1].upper() + label[1:]


class _Renderer:
    """One entry's sentence. Each placeholder renders to a list of Segments,
    or None when unset (which drops an enclosing [...])."""

    def __init__(self, kind, entry, definition, presentation, context, options, read) -> None:
        self.kind = kind
        self.entry = entry
        self.definition = definition
        self.presentation = presentation
        self.context = context
        self.options = options
        self.read = read
        self.full = options.sentence_style == FULL
        self.listed = frozenset(definition.attributes)
        # Set fields the sentence rendered, and ones it hid on purpose (an off flag, a `!` value).
        # Any other set field goes to extras().
        self.used: set[str] = set()
        self.hidden: set[str] = set()
        # The rendered {obj}'s link, and the fields that share it (the template's "link_as_object").
        self.object_link: tuple | None = None
        self.link_as_object: frozenset[str] = frozenset()

    # -- values

    def value(self, name: str) -> Any:
        if name not in self.listed:
            return None
        if name == "quantity_float":
            name = "quantity"
        return self.read(self.entry, name)

    def is_default(self, name: str, value: Any) -> bool:
        return value == self.definition.default_attributes.get(name, trigger_fields.UNSET)

    def presentation_of(self, name: str) -> str:
        return self.presentation.get(name) or _PRESENTATION_FALLBACK.get(name, "")

    # -- the parts

    def render(self, parts) -> list[Segment] | None:
        before = set(self.used)
        out: list[Segment] = []
        for part in parts:
            if isinstance(part, _Text):
                out.append(Segment(part.text))
            elif isinstance(part, _Optional):
                out.extend(self.render(part.parts) or ())
            else:
                rendered = self.field(part)
                if rendered is None:
                    # A dropped part shows none of its fields, so they stay extras.
                    self.used = before
                    return None
                out.extend(rendered)
        return out

    def render_top(self, parts) -> list[Segment]:
        out: list[Segment] = []
        for part in parts:
            if isinstance(part, _Text):
                out.append(Segment(part.text))
            elif isinstance(part, _Optional):
                out.extend(self.render(part.parts) or ())
            else:
                rendered = self.field(part)
                out.extend([Segment("?")] if rendered is None else rendered)
        return out

    def field(self, part: _Field) -> list[Segment] | None:
        pseudo = getattr(self, f"_pseudo_{part.name}", None)
        if pseudo is not None and part.name in _PSEUDO_FIELDS:
            rendered = pseudo(part)
            if rendered is not None:
                self.used.update(_PSEUDO_FIELDS[part.name])
                if part.name == "obj":
                    self.object_link = next((s.link for s in rendered if s.link), None)
            return rendered
        value = self.value(part.name)
        if _is_unset(value):
            return None
        if part.modifier.startswith("?"):
            rendered = self.render(part.args[0 if value else 1]) if len(part.args) == 2 else None
        elif part.modifier.startswith("!") and (self.is_default(part.name, value) or value in _hidden(part.modifier)):
            rendered = None
        elif part.modifier.startswith(":"):
            rendered = self.named_enum(part.modifier[1:], value)
        else:
            rendered = self.value_segments(part.name, value)
        (self.hidden if rendered is None else self.used).add(part.name)
        if rendered and part.name in self.link_as_object and self.object_link is not None:
            rendered = [replace(s, link=self.object_link) for s in rendered]
        return rendered

    def value_segments(self, name: str, value: Any) -> list[Segment] | None:
        presentation = self.presentation_of(name)
        if presentation == "PlayerId":
            return [self.player(value)]
        if presentation in ("UnitInfo", "BuildingInfo", "TechInfo"):
            return [Segment(self.catalog_name(presentation, value))]
        if presentation == "Unit":
            return self.placed_unit(value)
        if presentation == "Unit[]":
            return self.placed_units(value)
        if presentation == "TriggerId":
            return [self.trigger(value)]
        if presentation == "VariableId":
            name = self.context.variable_name(value) if isinstance(value, int) else ""
            if name:
                return [Segment(f'"{name}"')]
            # Full templates already say "variable".
            return [Segment(str(value) if self.full else f"var {value}")]
        if presentation == "bool":
            return self.flag(name, value)
        if presentation == "str" or isinstance(value, str):
            return self.quoted_segments(value)
        if presentation:
            return [Segment(self.enum_word(presentation, value))]
        return [Segment(_number(value))]

    # -- value kinds

    def player(self, pid: Any) -> Segment:
        if not isinstance(pid, int) or not 0 <= pid < len(self.context.labels.text):
            return Segment(f"P{pid}")
        label = self.context.labels.text[pid]
        if self.options.player_text == PLAYER_NUMBER:
            label = "GAIA" if pid == 0 else f"P{pid}"
        elif self.options.player_text == PLAYER_NAME and " - " in label:
            label = label.split(" - ", 1)[1]
        colour = self.options.player_colour
        return Segment(label, player=None if colour == "off" else pid, pid=pid, mark=colour)

    @staticmethod
    def owned(text: str, link: tuple | None, owner: Segment) -> Segment:
        """`text` coloured as the player segment `owner` is."""
        return Segment(text, link, owner.player, owner.pid, owner.mark)

    def catalog_name(self, presentation: str, value: Any) -> str:
        name = trigger_fields.resolve_reference(presentation, value)
        if not name:
            return f"{'tech' if presentation == 'TechInfo' else 'object'} {value}"
        name = _sentence_case(name) if name.isupper() else name
        return f"{name} #{value}" if self.options.show_ids else name

    def unit_link(self, link: tuple) -> tuple | None:
        return link if self.options.link_objects else None

    def placed_unit(self, ref_id: Any) -> list[Segment]:
        ref = self.context.unit(ref_id)
        if ref is None:
            return [Segment(f"object {ref_id} (not on the map)")]
        name = self.catalog_name("UnitInfo", ref.unit_const)
        link = self.unit_link((LINK_UNIT, ref_id)) if ref.bounds is not None else None
        owner = self.player(ref.player_id)
        text = f"{owner.text} {name} #{ref_id}" if self.options.show_unit_ids else f"{owner.text} {name}"
        return [self.owned(text, link, owner)]

    def placed_units(self, value: Any) -> list[Segment] | None:
        ids = tuple(v for v in (value or ()) if isinstance(v, int) and v >= 0)
        if not ids:
            return None
        if len(ids) == 1:
            return self.placed_unit(ids[0])
        listed = ""
        limit = self.options.unit_id_limit
        if limit > 0:
            shown = ", ".join(f"#{i}" for i in ids[:limit])
            listed = f" ({shown}, …)" if len(ids) > limit else f" ({shown})"
        link = self.unit_link((LINK_UNITS, ids))
        source = self.value("source_player")
        if _is_unset(source):
            return [Segment(f"{len(ids)} selected objects{listed}", link)]
        owner = self.player(source)
        return [self.owned(f"{len(ids)} selected {owner.text} objects{listed}", link, owner)]

    def trigger(self, trigger_id: Any) -> Segment:
        name = self.context.trigger_name(trigger_id) if isinstance(trigger_id, int) else None
        if name is None:
            return Segment(f"trigger {trigger_id} (missing)")
        text = f'"{name}"' if name.strip() else f"trigger {trigger_id}"
        return Segment(text, (LINK_TRIGGER, trigger_id) if self.options.link_triggers else None)

    def flag(self, name: str, value: Any) -> list[Segment] | None:
        short, full = _FLAG_WORDS.get(name, (name.replace("_", " "),) * 2)
        word = full if self.full else short
        if value and value != trigger_fields.UNSET:
            return [Segment(word)]
        return [Segment(f"not {word}")] if self.options.show_off_flags else None

    def quoted(self, text: str) -> str:
        text = " ".join(str(text).split())
        cap = self.options.text_cap
        if cap and len(text) > cap:
            text = text[: max(cap - 1, 1)] + "…"
        return f'"{text}"'

    def quoted_segments(self, text: Any) -> list[Segment]:
        """Quoted text, its colour tags drawn as colours unless the text_tags option shows them;
        the cap counts the visible characters."""
        flat = " ".join(str(text).split())
        if self.options.text_tags == "raw":
            return [Segment(self.quoted(flat))]
        runs = message_markup.parse(flat)
        cap = self.options.text_cap
        total = sum(len(run.text) for run in runs)
        room = max(cap - 1, 1) if cap and total > cap else total
        out = [Segment('"')]
        for run in runs:
            if room <= 0:
                break
            piece = run.text[:room]
            room -= len(piece)
            out.append(Segment(piece, rgb=run.color))
        if cap and total > cap:
            out.append(Segment("…"))
        out.append(Segment('"'))
        return out

    def enum_word(self, presentation: str, value: Any) -> str:
        if presentation == "Comparison":
            return (_COMPARISON_FULL if self.full else _COMPARISON_SHORT).get(value, str(value))
        if presentation == "Attribute" and value in _RESOURCE_WORDS:
            return _RESOURCE_WORDS[value]
        if presentation == "DamageClass" and value in _DAMAGE_CLASS_WORDS:
            return _DAMAGE_CLASS_WORDS[value]
        if presentation == "DecisionOption":
            return f"option {value}"
        label = next((label for label, v in trigger_fields.enum_choices(presentation) if v == value), None)
        if label is None:
            return str(value)
        if presentation == "ObjectAttribute":
            return f'"{_sentence_case(label)}"'
        return _sentence_case(label) if presentation in _CAPITALISED else label.lower()

    def named_enum(self, enum_name: str, value: Any) -> list[Segment]:
        enum_type, capitalised = _NAMED_ENUMS.get(enum_name, ((), False))
        member = next((m for m in enum_type if m.value == value), None)
        if member is None:
            return [Segment(str(value))]
        label = member.name.replace("_", " ")
        return [Segment(_sentence_case(label) if capitalised else label.lower())]

    # -- pseudo-fields

    def coords(self, names: tuple[str, ...]) -> tuple[int, ...] | None:
        values = tuple(self.value(name) for name in names)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            return None
        return values

    def area_segment(self, coords: tuple[int, ...]) -> Segment:
        x1, y1, x2, y2 = coords
        # Not "to" in Full: "in 1,1 to 9,9 to 5,5" would read as two moves.
        text = f"{x1},{y1} - {x2},{y2}" if self.full else f"{x1},{y1} {x2},{y2}"
        return Segment(text, (LINK_AREA, *coords) if self.options.link_coords else None)

    def _pseudo_area(self, _part) -> list[Segment] | None:
        coords = self.coords(AREA_FIELDS)
        return None if coords is None else [self.area_segment(coords)]

    def _pseudo_wall(self, _part) -> list[Segment] | None:
        coords = self.coords(WALL_FIELDS)
        return None if coords is None else [self.area_segment(coords)]

    def _pseudo_location(self, _part) -> list[Segment] | None:
        ref_id = self.value(LOCATION_OBJECT_FIELD)
        if not _is_unset(ref_id):
            return self.placed_unit(ref_id)
        coords = self.coords(LOCATION_FIELDS)
        if coords is None:
            return None
        return [Segment(f"{coords[0]},{coords[1]}", (LINK_TILE, *coords) if self.options.link_coords else None)]

    def object_type_segments(self, with_player: bool, area: tuple[int, ...] | None = None) -> list[Segment]:
        const_field = next((f for f in _OBJECT_CONST_FIELDS if f in self.listed), None)
        const = self.value(const_field) if const_field else None
        player, group, object_type = (self.value(name) for name in _FILTER_FIELDS)
        if not _is_unset(const):
            what = self.catalog_name("UnitInfo", const)
        elif not _is_unset(group):
            what = f"class {self.enum_word('ObjectClass', group)}"
        elif not _is_unset(object_type):
            what = self.enum_word("ObjectType", object_type)
        else:
            what = "objects"
        owner = None if _is_unset(player) else self.player(player)
        label = what if owner is None else f"{owner.text} {what}"
        unset = trigger_fields.UNSET
        found = ObjectFilter(
            player=unset if _is_unset(player) else player,
            unit_const=unset if _is_unset(const) else const,
            group=unset if _is_unset(group) else group,
            object_type=unset if _is_unset(object_type) else object_type,
            area=area,
            label=label,
        )
        text = label if with_player else what
        shown = owner if with_player and owner is not None else Segment("")
        return [self.owned(text, self.unit_link((LINK_MATCH, found)), shown)]

    # The area, when the type has one, narrows what the object's link steps through.
    def _pseudo_objtype(self, _part) -> list[Segment] | None:
        const_field = next((f for f in _OBJECT_CONST_FIELDS if f in self.listed), None)
        named = const_field, "object_group", "object_type"
        if all(_is_unset(self.value(name)) for name in named if name):
            # No object, class or type: the effect names only its player (modify attribute's "Population").
            player = self.value("source_player")
            return None if _is_unset(player) else [self.player(player)]
        return self.object_type_segments(with_player=True, area=self.coords(AREA_FIELDS))

    def _pseudo_objname(self, _part) -> list[Segment]:
        return self.object_type_segments(with_player=False, area=self.coords(AREA_FIELDS))

    def _pseudo_obj(self, _part) -> list[Segment]:
        selected = self.placed_units(self.value(SELECTED_OBJECTS_FIELD))
        area = self.coords(AREA_FIELDS)
        if selected is not None and (area is None or self.options.selected_with_area == "objects"):
            return selected
        if selected is not None and self.options.selected_with_area == "objects_first":
            return [*selected, Segment(" in "), self.area_segment(area)]
        if selected is not None and self.options.selected_with_area == "area_first":
            in_area = [*self.object_type_segments(with_player=True, area=area), Segment(" in "), self.area_segment(area)]
            return [*in_area, Segment(" and "), *selected]
        out = self.object_type_segments(with_player=True, area=area)
        most = self.value("max_units_affected")
        limited = isinstance(most, int) and not isinstance(most, bool) and most > 0
        if limited and self.full:
            out.insert(0, Segment(f"up to {most} "))
        elif self.full:
            out.insert(0, Segment("all "))
        if area is not None:
            out += [Segment(" in "), self.area_segment(area)]
        if limited and not self.full:
            out.append(Segment(f" (max {most})"))
        return out

    def _pseudo_amount(self, _part) -> list[Segment] | None:
        slot = trigger_fields.live_quantity_slot(self.entry) if self.kind == "effect" else trigger_fields.QUANTITY
        if slot == trigger_fields.ARMOUR_ATTACK or (
            slot == trigger_fields.QUANTITY and "quantity" not in self.listed and "armour_attack_quantity" in self.listed
        ):
            amount, damage = self.value("armour_attack_quantity"), self.value("armour_attack_class")
            if _is_unset(amount):
                return None
            text = _number(amount) if _is_unset(damage) else f"{_number(amount)} {self.enum_word('DamageClass', damage)}"
            return [Segment(text)]
        if slot == trigger_fields.VARIABLE:
            return self.field(_Field("variable", "", ()))
        quantity = self.value("quantity")
        return None if _is_unset(quantity) else [Segment(_number(quantity))]

    def _pseudo_hp(self, _part) -> list[Segment]:
        return [Segment(" HP")] if self.options.show_hp else []

    def _pseudo_either(self, part: _Field) -> list[Segment] | None:
        for arg in part.args:
            rendered = self.render(arg)
            if rendered is not None:
                return rendered
        return None

    def _pseudo_cost(self, _part) -> list[Segment] | None:
        pieces = [(self.value(name), word) for name, word in _COST_FIELDS]
        for n in (1, 2, 3):
            resource, amount = self.value(f"resource_{n}"), self.value(f"resource_{n}_quantity")
            # An untouched slot holds its defaults (Food, 0), not a real cost.
            untouched = self.is_default(f"resource_{n}", resource) and self.is_default(f"resource_{n}_quantity", amount)
            if not _is_unset(resource) and not untouched:
                pieces.append((amount, self.enum_word("Attribute", resource)))
        texts = [f"{_number(amount)} {word}" for amount, word in pieces if not _is_unset(amount)]
        return [Segment(", ".join(texts))] if texts else None

    def _pseudo_op(self, part: _Field) -> list[Segment] | None:
        if len(part.args) != 2:
            return None
        amount, target = self.render(part.args[0]), self.render_top(part.args[1])
        if amount is None:
            amount = [Segment("?")]
        operation = self.value("operation")
        forms = _OPERATION_FULL if self.full else _OPERATION_SHORT
        form = forms.get(operation, "Change {t} by {v}" if self.full else "{t} {v}")
        out: list[Segment] = []
        for i, piece in enumerate(form.replace("{v}", "\0v\0").replace("{t}", "\0t\0").split("\0")):
            if i % 2 == 0:
                if piece:
                    out.append(Segment(piece))
            else:
                out.extend(amount if piece == "v" else target)
        return out

    # -- the whole sentence

    def sentence(self, template: Mapping[str, Any] | None) -> list[Segment]:
        if template is None:
            return self.fallback()
        text = template.get(FULL if self.full else SHORT, "")
        self.link_as_object = frozenset(template.get("link_as_object", ()))
        out = self.render_top(parse(text))
        if template.get("object_link") == "location":
            out = self.links_to_location(out)
        if self.full and out and out[0].text[:1].islower():
            # "all P1 Militia cannot be selected" starts a sentence.
            out[0] = replace(out[0], text=out[0].text[:1].upper() + out[0].text[1:])
        out = self.with_player(out)
        inverted = self.value(INVERTED) if self.kind == "condition" else None
        if isinstance(inverted, int) and inverted > 0:
            out.insert(0, Segment("Not: " if self.full else "not "))
        shown = self.used | self.hidden | {INVERTED, TYPE_ATTRIBUTES[self.kind]} | set(template.get("omit", ()))
        out += self.extras(shown)
        return out

    def links_to_location(self, out: list[Segment]) -> list[Segment]:
        """Object links pointed at the entry's location, or dropped when it has none."""
        coords = self.coords(LOCATION_FIELDS)
        tile = (LINK_TILE, *coords) if coords is not None and self.options.link_objects else None
        return [replace(s, link=tile) if s.link and s.link[0] == LINK_MATCH else s for s in out]

    def with_player(self, out: list[Segment]) -> list[Segment]:
        """`out` led by the type's player when it has one set that the sentence does not show."""
        source = self.value("source_player")
        if not isinstance(source, int) or isinstance(source, bool) or _is_unset(source):
            return out
        if any(segment.pid == source for segment in out):
            return out
        return [self.player(source), Segment(" "), *out]

    def listed_order(self) -> list[str]:
        type_attribute = TYPE_ATTRIBUTES[self.kind]
        return [
            name
            for name in sorted(self.definition.attributes, key=trigger_fields.form_rank)
            if name != type_attribute and name not in trigger_fields.HIDDEN_FIELDS
        ]

    def extras(self, shown: set[str]) -> list[Segment]:
        out: list[Segment] = []
        for name in self.listed_order():
            if name in shown:
                continue
            value = self.value(name)
            if _is_unset(value) or self.is_default(name, value):
                continue
            label = trigger_fields.field_label(self.definition, name)
            if self.presentation_of(name) == "bool":
                out.append(Segment(f" + {label}" if value else f" + not {label}"))
                continue
            rendered = self.value_segments(name, value) or [Segment(_number(value))]
            out += [Segment(f" + {label}: "), *rendered]
        return out

    def fallback(self) -> list[Segment]:
        out: list[Segment] = []
        for name in self.listed_order():
            value = self.value(name)
            if _is_unset(value):
                continue
            rendered = self.value_segments(name, value)
            if rendered is None:
                continue
            if out:
                out.append(Segment(", "))
            out += [Segment(f"{trigger_fields.field_label(self.definition, name)}: "), *rendered]
        return out


def _safe_read(entry: Any, attribute: str) -> Any:
    try:
        return getattr(entry, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def raw_detail(kind: str, entry: Any, definition: Any, read: Callable[[Any, str], Any] = _safe_read) -> str:
    """The `attribute=value` pairs the Detail column showed before GH #388."""
    type_attribute = TYPE_ATTRIBUTES[kind]
    details = []
    for attribute in sorted(definition.attributes, key=trigger_fields.form_rank):
        if attribute == type_attribute or attribute in trigger_fields.HIDDEN_FIELDS:
            continue
        value = read(entry, attribute)
        # -1 is the library's "unset" sentinel across almost every numeric
        # trigger field, so showing it would bury the fields that are set.
        if value is None or value == -1 or value == "" or value == []:
            continue
        details.append(f"{attribute}={value}")
    return ", ".join(details)


def _bridge(segments: list[Segment]) -> list[Segment]:
    """A blank gap between two parts with the same link joins it, so the underline is unbroken."""
    out = list(segments)
    for i in range(1, len(out) - 1):
        gap, before, after = out[i], out[i - 1], out[i + 1]
        if gap.link is None and not gap.text.strip() and before.link is not None and before.link == after.link:
            out[i] = replace(gap, link=before.link)
    return out


def _merge(segments: list[Segment]) -> tuple[Segment, ...]:
    """Joins neighbouring plain segments, so a row paints few runs."""
    out: list[Segment] = []
    for segment in segments:
        if not segment.text:
            continue
        plain = segment.link is None and segment.player is None and segment.rgb is None
        if out and plain and out[-1].link is None and out[-1].player is None and out[-1].rgb is None:
            out[-1] = Segment(out[-1].text + segment.text)
        else:
            out.append(segment)
    return tuple(out)


def summarize(
    kind: str,
    entry: Any,
    definition: Any,
    presentation: Mapping[str, str],
    context: Context | None = None,
    options: Options | None = None,
    read: Callable[[Any, str], Any] = _safe_read,
) -> Summary:
    """One condition or effect's Detail: its sentence, or the raw pairs in Raw mode."""
    context = context or Context()
    options = options or Options()
    raw = raw_detail(kind, entry, definition, read)
    if options.mode == RAW:
        return Summary((Segment(raw),) if raw else (), raw)
    template = load_table().get(KIND_KEYS[kind], {}).get(definition.name)
    renderer = _Renderer(kind, entry, definition, presentation, context, options, read)
    return Summary(_merge(_bridge(renderer.sentence(template))), raw)


def matches(found: ObjectFilter, ref: Any, class_of: Callable[[int], Any], type_of: Callable[[int], Any]) -> bool:
    """Whether a placed unit (unit_references.UnitRef) passes an object link's filter."""
    unset = trigger_fields.UNSET
    if found.player != unset and ref.player_id != found.player:
        return False
    if found.unit_const != unset and ref.unit_const != found.unit_const:
        return False
    if found.group != unset and class_of(ref.unit_const) != found.group:
        return False
    if found.object_type != unset and type_of(ref.unit_const) != found.object_type:
        return False
    if found.area is not None:
        x1, y1, x2, y2 = found.area
        x, y = ref.own_tile
        if not (min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)):
            return False
    return True
