"""Display Instructions text markup (GH #140), Qt-free.

A message may carry `<COLOR>` tags (the nine colour tables of the install's
`widgetui/UIColors.json`), each switching the colour for the rest of the
string, and `<name>` tags naming a trigger variable. Any other `<...>` is
literal text: corpus messages such as `<AQUA><<<<<EPIC!>>>>>` expect the
brackets to show. Effects 3 and 44 (tech/object descriptions) also use
`<b>`, `<i>` and `<cost>`; this module keeps those literal too.

Whether the game draws instruction text with each table's `Text` entry, and
honours a mid-string switch, is a guess still owed an in-game check.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from typing import NamedTuple

from descape import asset_source

RGB = tuple[int, int, int]

# widgetui/UIColors.json ColorTables[*].Text (RGB), used when no install is configured.
TEXT_COLORS: dict[str, RGB] = {
    "BLUE": (110, 166, 235),
    "RED": (255, 100, 100),
    "GREEN": (0, 255, 0),
    "YELLOW": (255, 255, 0),
    "AQUA": (0, 255, 225),
    "PURPLE": (241, 108, 232),
    "GREY": (172, 172, 172),
    "ORANGE": (255, 180, 21),
    "WHITE": (232, 238, 255),
}

UI_COLORS_SUBPATH = "widgetui/UIColors.json"

TEXT = "text"
VARIABLE = "variable"
LITERAL = "literal"

_TAG = re.compile(r"<([^<>]*)>")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


class Run(NamedTuple):
    """One stretch of a message. `color` None is the default text colour."""

    text: str
    color: RGB | None
    kind: str


@lru_cache(maxsize=1)
def text_colors() -> dict[str, RGB]:
    """The install's `Text` colour per table, keyed by uppercase name, with
    TEXT_COLORS filling any table (or the whole file) the install lacks."""
    colors = dict(TEXT_COLORS)
    install = asset_source.get_install_path()
    if install is None:
        return colors
    path = install / UI_COLORS_SUBPATH
    try:
        tables = json.loads(path.read_text(encoding="utf-8"))["ColorTables"]
    except (OSError, ValueError, KeyError, TypeError):
        return colors
    if not isinstance(tables, dict):
        return colors
    for name, table in tables.items():
        try:
            r, g, b = (int(channel) for channel in table["Text"][:3])
        except (KeyError, TypeError, ValueError):
            continue
        if name.upper() in colors:
            colors[name.upper()] = (r, g, b)
    return colors


def clear_caches() -> None:
    """asset_source.clear_install_caches() calls this on an install change."""
    text_colors.cache_clear()


def parse(message: str, variable_names=(), colors: dict[str, RGB] | None = None) -> list[Run]:
    """Split `message` into runs. Colour tags and variable names match
    case-insensitively; empty runs are dropped."""
    colors = text_colors() if colors is None else colors
    variables = {name.lower() for name in variable_names if name}
    runs: list[Run] = []
    color: RGB | None = None

    def add(text: str, kind: str) -> None:
        if text:
            runs.append(Run(text, color, kind))

    position = 0
    for match in _TAG.finditer(message):
        add(message[position : match.start()], TEXT)
        inner = match.group(1)
        if inner.upper() in colors:
            color = colors[inner.upper()]
        elif inner.lower() in variables:
            add(inner, VARIABLE)
        else:
            add(match.group(0), LITERAL)
        position = match.end()
    add(message[position:], TEXT)
    return runs


def leading_color(runs: list[Run]) -> RGB | None:
    """The colour a message opens with, or None if its first run is untagged."""
    return runs[0].color if runs else None


def _hex(color: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def to_html(runs: list[Run], default_color: RGB) -> str:
    """Qt rich text for `runs`: escaped, one span per run, line breaks as <br>,
    a variable as an italic `[name]` placeholder."""
    parts = []
    for run in runs:
        text = f"[{run.text}]" if run.kind == VARIABLE else run.text
        body = "<br>".join(html.escape(line, quote=False) for line in _LINE_BREAK.split(text))
        if run.kind == VARIABLE:
            body = f"<i>{body}</i>"
        parts.append(f'<span style="color:{_hex(run.color or default_color)}">{body}</span>')
    return "".join(parts)
