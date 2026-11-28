"""Pasting copied triggers between scenarios of different versions (GH #317).

Qt-free. A TriggerBlock or EntryBlock copied at scenario version A pastes into
a document at B, both with a vocabulary, when every trigger, condition and
effect in it passes four rules; any failure refuses the whole block:

1. Its condition or effect type exists in both A's and B's vocabulary.
2. No set value is lost. An attribute A's entry type lists that B cannot store
   (the library's link is unsupported at B) or B's entry type does not list
   must be unset: A's per-type default, A's type-0 default, -1, None, [] or "".
   A Trigger attribute B cannot store must hold its constructor default. A
   float quantity has no storage of its own below 1.55, so a non-integral one
   is refused there.
3. Attributes B stores and A does not are overwritten with B's per-type
   default after the import, never only when missing: a field A lacks can
   carry None. Attributes B cannot store are set to None, as a native B
   object holds them.
4. An armour/attack class and quantity (or variable) fit 0..255 when B's
   Triggers section packs them 8+8 (below trigger version 2.5). At 2.5 and
   above (16+16) every value pastes: a same-width paste keeps a negative one.

Which attribute exists in which version comes from the library's own
`Support(since=, until=)` metadata on each class's `_link_list`, never from
names in structure.json. Values of version-gated attributes are read from the
object's `__dict__`, never through getattr: depending on what loaded last,
the classes are pristine or poisoned for another version (library_compat),
and a poisoned property raises. Ungated attributes are never poisoned, so
those are read through getattr, which keeps the library's computed meaning
(`item_id`, an armour/attack `quantity`).

The import itself runs inside stripped(): every attribute B cannot store is
popped from the block's objects for its length and restored after, since the
library's deepcopy sets each through a property B's version disables.
"""

from __future__ import annotations

import contextlib
import inspect
import weakref
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from AoE2ScenarioParser.datasets.trigger_lists import ObjectAttribute
from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.effect import Effect
from AoE2ScenarioParser.objects.data_objects.trigger import Trigger

from . import library_compat, trigger_fields

CLASSES = (Trigger, Condition, Effect)
_KIND = {Condition: "condition", Effect: "effect"}
_TYPE_ATTRIBUTE = {Condition: "condition_type", Effect: "effect_type"}
_UNSET = (-1, None, [], "")
_AA_WIDE_FROM = 2.5  # Effect._merge_aa_values(): 16+16 bits from this trigger version, 8+8 below
_FLOAT_QUANTITY = "quantity_float"
_SHOWN_REASONS = 3


@dataclass(frozen=True)
class Field:
    """One version-gated link of a class. `name` is the link's `link`, the
    vocabulary's name for it; `stored` is its `__dict__` key, None for a
    computed property the library derives (never dropped or filled)."""

    name: str
    link_name: str
    stored: str | None
    support: Any


def _pristine_attribute(cls: type, name: str) -> Any:
    for klass in cls.__mro__:
        state = library_compat.PRISTINE_CLASS_STATE.get(klass, vars(klass))
        if name in state:
            return state[name]
    return None


def _stored_key(cls: type, link_name: str) -> str | None:
    # A settable property stores under "_" + its name (player_color); one with no setter is computed.
    attribute = _pristine_attribute(cls, link_name)
    if isinstance(attribute, property):
        return f"_{link_name}" if attribute.fset is not None else None
    return link_name


def _build_fields(cls: type) -> tuple[Field, ...]:
    return tuple(
        Field(link.link, link.name, _stored_key(cls, link.name), link.support)
        for link in library_compat._iter_links(cls._link_list)
        if link.support is not None
    )


FIELDS: dict[type, tuple[Field, ...]] = {cls: _build_fields(cls) for cls in CLASSES}


@cache
def supported(cls: type, version: str) -> frozenset[str]:
    """The vocabulary names of `cls`'s links a `version` document stores."""
    return frozenset(
        link.link
        for link in library_compat._iter_links(cls._link_list)
        if link.support is None or link.support.supports(version)
    )


@cache
def _unstored(cls: type, version: str) -> tuple[Field, ...]:
    return tuple(f for f in FIELDS[cls] if not f.support.supports(version))


def _added(cls: type, source: str, here: str) -> tuple[Field, ...]:
    return tuple(f for f in FIELDS[cls] if f.stored and f.support.supports(here) and not f.support.supports(source))


@cache
def _trigger_defaults() -> dict[str, Any]:
    parameters = inspect.signature(Trigger.__init__).parameters
    return {f.stored: parameters[f.link_name].default for f in FIELDS[Trigger] if f.link_name in parameters}


@dataclass(frozen=True)
class PastePlan:
    """What a paste of a block copied at `source` into a `here` document
    does: `drop` per class (Fields popped around the import), `fill` per
    class (Fields set to B's default after it), and why it is refused."""

    source: str
    here: str
    drop: dict = field(default_factory=dict)
    fill: dict = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def crosses(self) -> bool:
        return bool(self.here) and self.source != self.here

    @property
    def refusal(self) -> str:
        if not self.reasons:
            return ""
        shown = list(self.reasons[:_SHOWN_REASONS])
        if len(self.reasons) > _SHOWN_REASONS:
            shown.append(f"and {len(self.reasons) - _SHOWN_REASONS} more")
        return f"Copied from a {self.source} scenario; this one is {self.here}: {'; '.join(shown)}."


def _numeric(version: str) -> bool:
    try:
        float(version)
    except ValueError:
        return False
    return True


def _vocabulary(version: str):
    if version and _numeric(version) and library_compat.vocabulary_is_available(version):
        return library_compat.load_vocabulary(version)
    return None


def _read(obj: Any, name: str, gated: dict[str, Field]) -> Any:
    gate = gated.get(name)
    if gate is not None:
        return vars(obj).get(gate.stored) if gate.stored else None
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 -- a library computed property can raise on a half-built object
        return None


def _is_unset(value: Any, *defaults: Any) -> bool:
    return any(value == unset for unset in (*_UNSET, *defaults))


def _float_quantity(entry: Any, obj: Any) -> bool:
    return _FLOAT_QUANTITY in entry.attributes and ObjectAttribute._storage_type(getattr(obj, "object_attributes", None)) == "float"


def _aa_values(obj: Any) -> tuple:
    source = vars(obj).get("_armour_attack_source")
    if source == "quantity":
        return vars(obj).get("_armour_attack_class"), vars(obj).get("_armour_attack_quantity")
    if source == "variable":
        return vars(obj).get("_armour_attack_class"), vars(obj).get("variable")
    return ()


class _Reasons:
    """Reasons in first-seen order, entries of one cause counted together."""

    def __init__(self) -> None:
        self._counts: dict[tuple, int] = {}

    def add(self, key: tuple) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def texts(self, source: str, here: str) -> tuple[str, ...]:
        out = []
        for (what, kind, label), n in self._counts.items():
            noun = f"{n} {kind}{'' if n == 1 else 's'}"
            if what == "lost":
                out.append(f"{noun} set {label}, which {here} does not store")
            elif what == "type":
                out.append(f"{label} does not exist in {here}")
            elif what == "unknown":
                out.append(f"{noun} of type {label}, which {source} does not define")
            else:
                out.append(f"{noun} {'holds' if n == 1 else 'hold'} an armour/attack value too large for {here}")
        return tuple(out)


def _check_entry(obj: Any, cls: type, vocab_a, vocab_b, here: str, trigger_version: float | None, reasons: _Reasons) -> None:
    kind = _KIND[cls]
    definitions_a = vocab_a.conditions if cls is Condition else vocab_a.effects
    definitions_b = vocab_b.conditions if cls is Condition else vocab_b.effects
    type_id = getattr(obj, _TYPE_ATTRIBUTE[cls], None)
    entry_a, entry_b = definitions_a.get(type_id), definitions_b.get(type_id)
    if entry_a is None:
        reasons.add(("unknown", kind, str(type_id)))
        return
    if entry_b is None:
        reasons.add(("type", kind, entry_a.label.title()))
        return
    gated = {f.name: f for f in FIELDS[cls]}
    base_a = definitions_a[0].default_attributes if 0 in definitions_a else {}
    defaults_a = {**base_a, **entry_a.default_attributes}
    for name in entry_a.attributes:
        gate = gated.get(name)
        if (gate is None or gate.support.supports(here)) and name in entry_b.attributes:
            continue
        if name == _FLOAT_QUANTITY:
            value = vars(obj).get("_quantity")
            if _float_quantity(entry_a, obj) and isinstance(value, float) and not value.is_integer():
                reasons.add(("lost", kind, "a fractional Quantity"))
            continue
        if not _is_unset(_read(obj, name, gated), defaults_a.get(name), base_a.get(name)):
            reasons.add(("lost", kind, trigger_fields.field_label(entry_a, name).title()))
    narrow = trigger_version is not None and trigger_version < _AA_WIDE_FROM
    if narrow and any(isinstance(v, int) and not 0 <= v <= 0xFF for v in _aa_values(obj)):
        reasons.add(("aa", kind, ""))


def _objects(block: Any) -> Iterator[tuple[type, Any]]:
    for trigger in getattr(block, "triggers", ()):
        yield Trigger, trigger
        yield from ((Condition, c) for c in trigger.conditions)
        yield from ((Effect, e) for e in trigger.effects)
    yield from ((Condition, c) for c in getattr(block, "conditions", ()))
    yield from ((Effect, e) for e in getattr(block, "effects", ()))


def _walk(block: Any, here: str, trigger_version: float | None) -> PastePlan:
    source = block.scenario_version
    vocab_a, vocab_b = _vocabulary(source), _vocabulary(here)
    if vocab_a is None or vocab_b is None:
        missing = source if vocab_a is None else here
        return PastePlan(source, here, reasons=(f"no trigger vocabulary for {missing}",))
    reasons = _Reasons()
    trigger_defaults = _trigger_defaults()
    for cls, obj in _objects(block):
        if cls is Trigger:
            for f in _unstored(Trigger, here):
                value = vars(obj).get(f.stored)
                if not _is_unset(value, trigger_defaults.get(f.stored), 0):
                    reasons.add(("lost", "trigger", f.name.replace("_", " ").title()))
        else:
            _check_entry(obj, cls, vocab_a, vocab_b, here, trigger_version, reasons)
    drop = {cls: tuple(f for f in _unstored(cls, here) if f.stored) for cls in CLASSES}
    fill = {cls: _added(cls, source, here) for cls in CLASSES}
    return PastePlan(source, here, drop, fill, reasons.texts(source, here))


_memo: dict[tuple, tuple] = {}


def plan_paste(block: Any, here: str, *, trigger_version: float | None = None) -> PastePlan:
    """The PastePlan for `block` pasted into a `here` document whose Triggers
    section is at `trigger_version` (None skips rule 4). Equal versions, or
    an empty `here`, give an empty plan without walking. Memoized per
    (block kind, here, trigger_version) on the last block, held weakly, so
    the trigger and entry clipboards keep a slot each."""
    if not here or block.scenario_version == here:
        return PastePlan(block.scenario_version, here)
    key = (type(block), here, trigger_version)
    cached = _memo.get(key)
    if cached is not None and cached[0]() is block:
        return cached[1]
    plan = _walk(block, here, trigger_version)
    _memo[key] = (weakref.ref(block), plan)
    return plan


@contextlib.contextmanager
def stripped(plan: PastePlan, objects: Iterable[Any]) -> Iterator[None]:
    """Pop every attribute `plan.here` cannot store from `objects`' instance
    dicts for the length of the block, restoring them in `finally`, so the
    clipboard block is unchanged after any paste, a failed one included."""
    popped = []
    try:
        for obj in objects:
            state = vars(obj)
            popped.extend((state, f.stored, state.pop(f.stored)) for f in plan.drop.get(type(obj), ()) if f.stored in state)
        yield
    finally:
        for state, key, value in popped:
            state[key] = value


def block_objects(block: Any) -> list:
    """Every Trigger, Condition and Effect object in `block`, for stripped()."""
    return [obj for _cls, obj in _objects(block)]


def _entry_defaults(vocabulary, cls: type, obj: Any) -> dict[str, Any]:
    definitions = vocabulary.conditions if cls is Condition else vocabulary.effects
    base = definitions[0].default_attributes if 0 in definitions else {}
    entry = definitions.get(getattr(obj, _TYPE_ATTRIBUTE[cls], None))
    return {**base, **(entry.default_attributes if entry is not None else {})}


def apply(plan: PastePlan, triggers: Sequence, conditions: Sequence, effects: Sequence) -> None:
    """Rule 3 over the pasted objects: B's defaults into the attributes it
    adds, None into the ones it cannot store, as a native B object holds."""
    if not plan.crosses:
        return
    vocabulary = library_compat.load_vocabulary(plan.here)
    trigger_defaults = _trigger_defaults()
    for cls, objects in ((Trigger, triggers), (Condition, conditions), (Effect, effects)):
        for obj in objects:
            state = vars(obj)
            for f in plan.drop.get(cls, ()):
                state[f.stored] = None
            defaults = trigger_defaults if cls is Trigger else _entry_defaults(vocabulary, cls, obj)
            for f in plan.fill.get(cls, ()):
                state[f.stored] = defaults.get(f.stored if cls is Trigger else f.name)
            if cls is Effect and _FLOAT_QUANTITY not in supported(Effect, plan.here):
                quantity = state.get("_quantity")
                if isinstance(quantity, float) and quantity.is_integer():
                    state["_quantity"] = int(quantity)


def paste_refusal(block: Any, here: str, trigger_version: float | None = None, *, convertible: bool) -> str:
    """The paste refusal for a block of another version, or "" when it pastes.
    `convertible` is whether File > Convert is enabled for this document; the
    hint is added only when converting it would make the paste legal."""
    from . import scenario_convert  # scenario_convert imports scenario_io's write stack; keep this module light

    plan = plan_paste(block, here, trigger_version=trigger_version)
    if not plan.reasons:
        return ""
    target = scenario_convert.TARGET_VERSION
    target_triggers = max(step.trigger_versions[1] for step in scenario_convert.STEPS.values())
    if convertible and here != target and not plan_paste(block, target, trigger_version=target_triggers).reasons:
        return f"{plan.refusal} File > {scenario_convert.action_label().replace('…', '')} upgrades this one."
    return plan.refusal
