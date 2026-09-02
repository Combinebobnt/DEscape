"""
Trigger organization derived from conventions already present in real files:
a leading `[tag]`/`(tag)`/`<tag>` name prefix, and `--- Section ---`-style
divider triggers that partition the display-order list into sections.

Deliberately Qt-free, mirroring trigger_fields.py's convention, so the
heuristic is testable in the default tier without a QApplication.
TriggerPanel is the only importer. No write path and no persistence: both
functions here read a name/order pair and return derived data, nothing more
-- name and trigger_display_order already survive a reorder, which is why
that's sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Characters that read as a divider rule when repeated. Fitted against 1570
# trigger names across 16 examples/ scenarios; re-measured against a much
# larger private scenario collection (not shipped in this repo) with
# tools/census_trigger_names.py before this shipped. Provisional in the
# sense that a wider corpus could still move it.
_DIVIDER_CHARS = frozenset("-=~*_#")
_DIVIDER_RUN = 2
_DIVIDER_BARE_RUN = 3

_TAG_DELIMITERS = {"[": "]", "(": ")", "<": ">"}


def parse_tag(name: str) -> str | None:
    """The leading `[tag]`/`(tag)`/`<tag>` prefix, or None.

    A *prefix* match only: the delimiter must open at position 0. `[h/m]`
    matches (the census's own tag corpus has 86 distinct tags, some
    slash-bearing); `Escort [VIP]` does not, since the bracket isn't leading.
    """
    stripped = name.strip()
    if not stripped:
        return None
    close = _TAG_DELIMITERS.get(stripped[0])
    if close is None:
        return None
    end = stripped.find(close, 1)
    if end <= 1:
        return None
    tag = stripped[1:end].strip()
    return tag or None


def leading_divider_run(name: str) -> int:
    """How many divider characters `name` (stripped) starts with.

    Exposed for tools/census_trigger_names.py's near-miss diagnostic (a name
    starting with >=2 divider characters that is_divider() still rejects) --
    not used by is_divider() itself, which needs the trailing run too.
    """
    stripped = name.strip()
    lead = 0
    while lead < len(stripped) and stripped[lead] in _DIVIDER_CHARS:
        lead += 1
    return lead


def is_divider(name: str) -> bool:
    """Whether `name` reads as a `--- Section ---`-style header.

    A name is a divider if, stripped, it either starts with >=2 AND ends with
    >=2 characters from _DIVIDER_CHARS with something in between (`--- Setup
    ---`, `---Setup---`, `--Start--`), or consists entirely of >=3 of them (a
    bare `------`). The two cases are kept disjoint on purpose: a 2-character
    `--` is neither a real rule nor two overlapping runs of one.

    A false positive here costs a cosmetic nesting level, never data: the
    header row IS the divider trigger, so it stays selectable, editable, and
    movable either way. That is what makes a fuzzy heuristic acceptable --
    see the plan's "the one invariant that makes false positives safe".
    """
    stripped = name.strip()
    if not stripped:
        return False
    if all(c in _DIVIDER_CHARS for c in stripped):
        return len(stripped) >= _DIVIDER_BARE_RUN

    trail = 0
    while trail < len(stripped) and stripped[-(trail + 1)] in _DIVIDER_CHARS:
        trail += 1
    return leading_divider_run(name) >= _DIVIDER_RUN and trail >= _DIVIDER_RUN


@dataclass(frozen=True)
class Section:
    """One section of the display-order partition.

    `header_index` is the divider trigger's own list index, or None for the
    synthetic leading section (triggers before the first divider, or every
    trigger when the file has none). `member_indices` never includes the
    header itself. `title` is "" for the synthetic section -- the panel
    supplies its own display label for that row.
    """

    title: str
    header_index: int | None
    member_indices: tuple[int, ...]


def sections(names: Sequence[str], order: Sequence[int]) -> list[Section]:
    """Partition `order` (trigger list indices, typically
    trigger_display_order) into sections at each divider-named trigger.

    `order` may be any permutation of a subset of range(len(names)); nothing
    here assumes it is the full trigger list, so a filtered view can reuse it
    too. The leading section is omitted entirely when it would be empty --
    i.e. when the very first trigger in `order` is itself a divider -- so a
    file with no divider at all still comes back as one non-empty section
    with header_index=None (the caller renders that case flat).
    """
    result: list[Section] = []
    title = ""
    header_index: int | None = None
    members: list[int] = []

    def flush() -> None:
        if header_index is not None or members:
            result.append(Section(title, header_index, tuple(members)))

    for index in order:
        name = names[index]
        if is_divider(name):
            flush()
            title = name.strip()
            header_index = index
            members = []
        else:
            members.append(index)
    flush()
    return result
