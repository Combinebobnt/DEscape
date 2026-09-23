"""
Trigger organization derived from conventions already present in real files:
a leading `[tag]`/`(tag)`/`<tag>` name prefix, and `--- Section ---`-style
divider triggers that partition the display-order list into sections.

Deliberately Qt-free, mirroring trigger_fields.py's convention, so the
heuristic is testable in the default tier without a QApplication.
No write path and no persistence: every function here reads a name/order
pair and returns derived data, nothing more -- name and
trigger_display_order already survive a reorder, which is why that's
sufficient. The tag rewriters (retag_name/strip_tag) and the divider
formatters (format_divider/retitle_divider) return a new name; the window's
funnels are what write it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


def split_tag(name: str) -> tuple[str, str, str] | None:
    """`name` cut as (prefix, tag, suffix) around parse_tag()'s tag, or None.

    Cut from the unstripped name, so `prefix + tag + suffix == name`: prefix
    holds any leading whitespace, the opener and inner left padding; suffix
    the inner right padding, the closer and the rest of the name. None exactly
    when parse_tag() is None.
    """
    tag = parse_tag(name)
    if tag is None:
        return None
    lead = len(name) - len(name.lstrip())
    body_start = lead + 1
    tag_start = body_start + (len(name[body_start:]) - len(name[body_start:].lstrip()))
    tag_end = tag_start + len(tag)
    return name[:tag_start], name[tag_start:tag_end], name[tag_end:]


def retag_name(name: str, new_tag: str) -> str:
    """`name` with its tag replaced by `new_tag`, every other character kept."""
    parts = split_tag(name)
    if parts is None:
        raise ValueError(f"{name!r} carries no tag")
    return parts[0] + new_tag + parts[2]


def strip_tag(name: str) -> str:
    """`name` with its tag, delimiters and the whitespace after the closer
    removed. Leading whitespace and the rest of the name are kept as is."""
    parts = split_tag(name)
    if parts is None:
        raise ValueError(f"{name!r} carries no tag")
    prefix, _, suffix = parts
    lead = prefix[: len(prefix) - len(prefix.lstrip())]
    # suffix is right padding, then the closer, then the rest.
    after_closer = suffix.lstrip()[1:]
    return lead + after_closer.lstrip()


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
class DividerFormat:
    """The decoration of a titled divider: `lead` then `trail` copies of
    `char` around the title, with one space on each inner side when `spaced`."""

    char: str = "-"
    lead: int = 3
    trail: int = 3
    spaced: bool = True


# `--- X ---`: 96 of the census's 153 spaced dividers, the corpus's mode.
DEFAULT_DIVIDER_FORMAT = DividerFormat()


def split_divider(name: str) -> tuple[str, str, str] | None:
    """`name` cut as (prefix, title, suffix), or None.

    Cut from the unstripped name, so `prefix + title + suffix == name`: prefix
    is leading whitespace, the maximal leading divider run and the whitespace
    after it; suffix the mirror image, trailing whitespace included. None for
    a non-divider and for a divider with no title slot (a bare run, or runs
    around nothing but whitespace).
    """
    if not is_divider(name):
        return None
    stripped = name.strip()
    if all(c in _DIVIDER_CHARS for c in stripped):
        return None
    start = len(name) - len(name.lstrip())
    while name[start] in _DIVIDER_CHARS:
        start += 1
    while name[start].isspace():
        start += 1
    end = len(name.rstrip())
    while name[end - 1] in _DIVIDER_CHARS:
        end -= 1
    while name[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return name[:start], name[start:end], name[end:]


def _format_of(name: str) -> DividerFormat | None:
    """The DividerFormat `name` was written in, or None when it has no title
    slot or its runs mix characters (no single `char` describes it)."""
    parts = split_divider(name)
    if parts is None:
        return None
    prefix, _, suffix = parts
    lead_ws = prefix[: len(prefix) - len(prefix.lstrip())]
    lead_run = prefix[len(lead_ws):].rstrip()
    trail_run = suffix.strip()
    chars = set(lead_run) | set(trail_run)
    if len(chars) != 1:
        return None
    spaced = prefix[-1].isspace() and suffix[0].isspace()
    return DividerFormat(lead_run[0], len(lead_run), len(trail_run), spaced)


def divider_format(names: Sequence[str]) -> DividerFormat:
    """The most common titled-divider format among `names`, so an emitted
    divider matches the file it lands in.

    Ties prefer DEFAULT_DIVIDER_FORMAT, then the smallest (lead, trail). No
    titled divider at all returns the default.
    """
    counts: dict[DividerFormat, int] = {}
    for name in names:
        fmt = _format_of(name)
        if fmt is not None:
            counts[fmt] = counts.get(fmt, 0) + 1
    if not counts:
        return DEFAULT_DIVIDER_FORMAT
    best = max(counts.values())
    tied = [fmt for fmt, count in counts.items() if count == best]
    if DEFAULT_DIVIDER_FORMAT in tied:
        return DEFAULT_DIVIDER_FORMAT
    return min(tied, key=lambda f: (f.lead, f.trail, not f.spaced, f.char))


def format_divider(title: str, fmt: DividerFormat = DEFAULT_DIVIDER_FORMAT) -> str:
    """A divider name for `title` in `fmt`."""
    gap = " " if fmt.spaced else ""
    return f"{fmt.char * fmt.lead}{gap}{title}{gap}{fmt.char * fmt.trail}"


def retitle_divider(name: str, title: str) -> str:
    """`name` with its title replaced, its own decoration kept byte for byte."""
    parts = split_divider(name)
    if parts is None:
        raise ValueError(f"{name!r} has no divider title to replace")
    return parts[0] + title + parts[2]


def divider_title_error(name: str, title: str) -> str:
    """Why `name` cannot carry `title` as its divider title, or "" when it
    reads back as exactly that title (a title ending in a divider character
    would merge into the run)."""
    if not title:
        return "the title is empty"
    parts = split_divider(name)
    if parts is None or parts[1] != title:
        return "it would not read back as that title"
    return ""


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


def section_end_slot(names: Sequence[str], order: Sequence[int], header_index: int | None) -> int:
    """The display slot just past the section headed by `header_index` (None
    for the synthetic leading section): past its last member, past its header
    when it has none, or 0 for an empty leading section. Raises ValueError
    when no divider in `order` has that index."""
    slots = {index: slot for slot, index in enumerate(order)}
    for section in sections(names, order):
        if section.header_index != header_index:
            continue
        last = section.member_indices[-1] if section.member_indices else section.header_index
        return slots[last] + 1
    if header_index is None:
        return 0
    raise ValueError(f"trigger {header_index} does not head a section")


def section_header_of(names: Sequence[str], order: Sequence[int], index: int) -> int | None:
    """The header_index of the section trigger `index` sits in (its own index
    when it is a divider), None for the leading section."""
    for section in sections(names, order):
        if index == section.header_index or index in section.member_indices:
            return section.header_index
    raise ValueError(f"trigger {index} is not in the display order")


UNNAMED_SECTION = "(unnamed section)"


def section_key(parts: Sequence[Section], position: int) -> tuple[str, int]:
    """The in-session collapse-state key for `parts[position]`: its title,
    plus how many earlier sections share that exact title.

    Not header_index: remove/reorder renumber trigger ids across the whole
    list, so an index key goes stale on the very edits that rebuild the tree.
    A title key only misses when a same-titled section is added or removed
    above it, and a miss just renders the section expanded.
    """
    title = parts[position].title
    ordinal = sum(1 for section in parts[:position] if section.title == title)
    return title, ordinal


def section_label(title: str) -> str:
    """A menu label for a section titled `title`: the title itself, or
    UNNAMED_SECTION for a pure ruler (`------`) with no letter or digit in it.
    Elision is the caller's; this module stays Qt-free."""
    if any(c.isalnum() for c in title):
        return title
    return UNNAMED_SECTION
