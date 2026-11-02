"""Keeps an effect's stored quantity pair through a trigger re-serialize.

An effect stores quantity twice, `quantity` (int) and `quantity_float`, and the
library's Effect holds one value for the two. A dirtied trigger therefore comes
back from a commit with the live half kept and the other reset to its sentinel,
and the library's reader, `_quantity_float or quantity`, swaps a stored 0.0 for
the int half, which a commit then writes into the float slot.

The fix, the same shape as unlinked_fields and for the same reasons (a plain
instance attribute travels with the effect through AoE2Object.__deepcopy__, so
through undo snapshots and copy/paste, and survives list moves that shift
slots):

- pull(): at load, carry each effect's stored pair on the effect, and put back
  the 0.0 the reader lost. Only a float-live effect's stored 0.0 is healed: it
  is the one value `or` loses.
- push(): after the library's commit and before any slot is serialized, write
  the carried pair into the slot of each effect whose quantity did not change.

An effect whose quantity was edited keeps the library's result, live half new
and dead half sentinel: only one real value then exists, so no reading rule can
mask the edit. An edit means the live half, the effect type or
`object_attributes` differs from what was parsed; changing a value back to the
one the file stores counts as unchanged, since the result equals the file.

A new effect carries nothing, and neither does one from a file before scenario
version 1.55, which has no float field. push()'s own check for that field is
for a batch script's paste: the UI refuses one into such a file.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from typing import Any

from AoE2ScenarioParser.objects.data_objects.effect import _is_float_quantity_effect

_CARRIED = "_file_quantity_pair"


def pull(effects: Iterable[Any], entries: Iterable[Any]) -> None:
    """Carries each entry's stored pair on the effect built from it. `effects`
    and `entries` are parallel, in struct order."""
    for effect, entry in zip(effects, entries, strict=True):
        fields = entry.retriever_map
        if "quantity_float" not in fields:  # before scenario version 1.55
            return
        whole, part = fields["quantity"].data, fields["quantity_float"].data
        setattr(effect, _CARRIED, (effect.effect_type, effect.object_attributes, whole, part))
        # The library's `_quantity_float or quantity` loses a stored 0.0.
        if part == 0.0 and _is_float_quantity_effect(effect.effect_type, effect.object_attributes):
            effect.quantity = part


def push(effects: Iterable[Any], entries: Iterable[Any]) -> None:
    """Writes the carried pair into the slot each unchanged effect now occupies.
    Call after the library's commit (which re-slots effects) and before the
    slots are serialized."""
    for effect, entry in zip(effects, entries, strict=True):
        carried, fields = getattr(effect, _CARRIED, None), entry.retriever_map
        if carried is None or "quantity_float" not in fields:
            continue
        effect_type, attribute, whole, part = carried
        if (effect.effect_type, effect.object_attributes) != (effect_type, attribute):
            continue
        # The float compares as stored: f32, so NaN matches and a retyped 1.9 equals a stored 1.899999976.
        if _is_float_quantity_effect(effect_type, attribute):
            unchanged = struct.pack("<f", fields["quantity_float"].data) == struct.pack("<f", part)
        else:
            unchanged = fields["quantity"].data == whole
        if unchanged:
            fields["quantity"].set_data(whole, affect_dirty=False)
            fields["quantity_float"].set_data(part, affect_dirty=False)
