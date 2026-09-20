"""The terrain-style vocabulary: the three style tokens, the elevated
subset, and the display labels derived from them.

A Qt-free leaf module importing nothing from descape, so render.py (which
imports no PyQt5 at all) and the Qt-side viewer.py/map_view.py can share one
source of truth without the constants dragging Qt into the pure-numpy render
path. viewer_common.py -- the obvious existing home -- imports
PyQt5.QtWidgets, and iso_geometry.py's own docstring scopes it to the
"Stepped" terrain rendering mode, which is precisely the mode "flat" is not.

Deliberately not a home for the ~30 single-value comparisons (== "flat",
!= "sloped") scattered across viewer.py, map_view.py and unit_pick.py: a
constant does not make those clearer. What this fixes is the vocabulary
being *enumerated* in several independent places in two different cases,
with no single source of truth."""

from __future__ import annotations

# Canonical order -- the toolbar combo's own item order is STYLE_LABELS,
# built from this, so the two can no longer drift.
TERRAIN_STYLES = ("flat", "stepped", "sloped")

# The two non-Flat styles, which share a projected pick plane and a diamond
# ground outline.
ELEVATED_STYLES = ("stepped", "sloped")


def label_for(style: str) -> str:
    """The toolbar combo's Title-Case label for a style token."""
    return style.capitalize()


def style_for_label(label: str) -> str:
    """The style token behind a combo label. Raises on anything that isn't
    one of the three, which is stricter than the bare .lower() it replaces
    at viewer.on_terrain_style_changed()."""
    style = label.lower()
    if style not in TERRAIN_STYLES:
        raise ValueError(f"unknown terrain style label {label!r}")
    return style


# Derived with .capitalize() rather than mapped, which round-trips all three
# exactly and is already the de facto contract at both viewer.py sites. The
# accepted cost: display text is now coupled to the token, so a future
# two-word style ("Sloped (smooth)") needs a real label mapping here instead.
STYLE_LABELS = tuple(label_for(s) for s in TERRAIN_STYLES)
