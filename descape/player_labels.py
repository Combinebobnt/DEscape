"""One label per player for every player selector (GH #130).

Qt-free. `tribe_names()` reads each player's Tribe name as edited (pending
Players mode edits over the file's own value) and `player_label()` turns a
(pid, name) pair into the shared "P8 - Horde Army" form. The viewer computes
the nine labels once and pushes them to each selector; see
ViewerWindow._refresh_player_labels().
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from descape import player_fields

TRIBE_NAME_FIELD = "tribe_name"
# Corpus max is 25 chars; a 255-byte name must not widen the Units panel.
LABEL_NAME_MAX = 32
PLAYER_SLOTS = 9  # GAIA plus P1..P8


def full_player_label(pid: int, name: str = "") -> str:
    """"GAIA" for pid 0, "P8" for an empty or blank name, else "P8 - <name>"."""
    if pid == 0:
        return "GAIA"
    name = name.strip()
    return f"P{pid} - {name}" if name else f"P{pid}"


def player_label(pid: int, name: str = "") -> str:
    """full_player_label(), with a name over LABEL_NAME_MAX chars cut to
    LABEL_NAME_MAX - 1 plus an ellipsis."""
    name = name.strip()
    if len(name) > LABEL_NAME_MAX:
        name = name[: LABEL_NAME_MAX - 1] + "…"
    return full_player_label(pid, name)


class PlayerLabels(NamedTuple):
    """Nine labels indexed by pid: `text` for the item, `full` (never cut)
    for its tooltip."""

    text: tuple[str, ...]
    full: tuple[str, ...]


def labels_for(names: tuple[str, ...]) -> PlayerLabels:
    """PlayerLabels for a tribe_names() tuple."""
    return PlayerLabels(
        tuple(player_label(pid, name) for pid, name in enumerate(names)),
        tuple(full_player_label(pid, name) for pid, name in enumerate(names)),
    )


DEFAULT_LABELS = labels_for(("",) * PLAYER_SLOTS)


def tribe_names(loaded, pending: Mapping[int, str] | None = None) -> tuple[str, ...]:
    """A 9-tuple of tribe names indexed by pid, slot 0 (GAIA) always "".

    A pending edit wins over the file's value: a tribe name edit is a byte
    patch that never updates the parsed retriever. All "" when the file has
    no tribe_name spec."""
    pending = pending or {}
    spec = next(
        (s for s in player_fields.specs_for(loaded) if s.field_id == TRIBE_NAME_FIELD), None
    )
    if spec is None:
        return ("",) * PLAYER_SLOTS
    names = [""]
    for pid in range(1, PLAYER_SLOTS):
        value = pending[pid] if pid in pending else player_fields.current_value(loaded, spec, pid)
        names.append(value.strip() if isinstance(value, str) else "")
    return tuple(names)
