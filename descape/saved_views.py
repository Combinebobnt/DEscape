"""View > Saved Views (TASK-270): nine editor camera views per scenario file.

A view is the continuous map point under the viewport centre, in tile units,
plus the zoom as a multiple of fit-to-view. It names no Elevation View, mode
or tool, so a recall keeps whichever ones are showing. Nothing here touches the
scenario: settings.py keeps the views in config.yaml, keyed by the scenario's
resolved path, and this module holds their cleaning and the store's list shape.
Qt-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SLOTS = tuple(range(1, 10))


@dataclass(frozen=True)
class SavedView:
    x: float
    y: float
    zoom: float


def _finite(value) -> float | None:
    # Exact types: a YAML `true` is no coordinate, and 10**400 overflows float().
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def clean_view(raw) -> SavedView | None:
    """A stored {"x", "y", "zoom"} row as a SavedView, or None unless x and y are finite and zoom > 0."""
    if not isinstance(raw, dict):
        return None
    x, y, zoom = (_finite(raw.get(name)) for name in ("x", "y", "zoom"))
    if x is None or y is None or zoom is None or zoom <= 0:
        return None
    return SavedView(x, y, zoom)


def clean_slots(raw) -> dict[int, SavedView]:
    """A stored {slot: row} mapping with every non-slot key and bad row dropped."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for slot, row in raw.items():
        view = clean_view(row)
        if type(slot) is int and slot in SLOTS and view is not None:
            out[slot] = view
    return out


def rounded(view: SavedView) -> SavedView:
    """x/y to 4 decimals and zoom to 6 significant digits: a readable file, still far under a scene pixel."""
    return SavedView(round(view.x, 4), round(view.y, 4), float(f"{view.zoom:.6g}"))


def clean_store(raw, cap: int) -> list[tuple[str, dict[int, SavedView]]]:
    """The config's saved_views list as (scenario key, views) pairs, most recent first: a
    malformed entry, an empty one or a repeat of an earlier key is dropped, then the cap applies."""
    store: list[tuple[str, dict[int, SavedView]]] = []
    if not isinstance(raw, list):
        return store
    seen = set()
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("scenario"), str):
            continue
        key = entry["scenario"]
        views = {slot: rounded(view) for slot, view in clean_slots(entry.get("views")).items()}
        if views and key not in seen:
            seen.add(key)
            store.append((key, views))
    return store[:cap]


def with_views(store, key: str, views: dict[int, SavedView], cap: int) -> list[tuple[str, dict[int, SavedView]]]:
    """A new store with `key`'s views replaced and moved to the front, capped; empty views drop the entry."""
    rest = [(k, v) for k, v in store if k != key]
    kept = {slot: rounded(view) for slot, view in sorted(views.items()) if slot in SLOTS}
    return ([(key, kept), *rest] if kept else rest)[:cap]


def store_rows(store) -> list[dict]:
    """The store as config.yaml rows, order kept (the config writer does not sort)."""
    return [
        {"scenario": key, "views": {slot: {"x": v.x, "y": v.y, "zoom": v.zoom} for slot, v in sorted(views.items())}}
        for key, views in store
    ]
