"""
Trigger organization derived from conventions already present in real files:
a leading `[tag]`/`(tag)`/`<tag>` name prefix, and `--- Section ---`-style
divider triggers that partition the display-order list into sections.

Deliberately Qt-free, mirroring trigger_fields.py's convention, so the
heuristic is testable in the default tier without a QApplication.
No write path of its own and no persistence: every function here reads a
name/order pair and returns derived data, nothing more -- name and
trigger_display_order already survive a reorder, which is why that's
sufficient. The tag rewriters (retag_name/strip_tag and their chain-wide
retag_in_chain/strip_tag_from_chain), the divider formatters
(format_divider/retitle_divider) and copied_name() return a new name; the
window's funnels are what write it. Written data does derive from here,
though: trigger_model's copy and paste placement reads sections() and
section_end_slot() to pick the display slots it writes, and its section move
(moved_section_display_order()) writes a permutation of sections() blocks.
That is safe because the partition written is the partition displayed: a
false-positive divider moves a block the user can see, whole, and costs a
nesting level, never data.

Which names are dividers is one process-global DividerRule
(divider_rule()/set_divider_rule()), read by every function here. Only the
viewer sets it, from Settings > General > Trigger section headers; batch
scripts and tools keep DEFAULT_DIVIDER_RULE, the built-in rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Characters that read as a divider rule when repeated. Fitted against 1570
# trigger names across 16 examples/ scenarios; re-measured against a much
# larger private scenario collection (not shipped in this repo) with
# tools/census_trigger_names.py before this shipped, and again on 2026-10-27
# over 42,140 names in 144 parsed files, when is_divider() was widened to
# marker-style headers. Provisional in the sense that a wider corpus could
# still move it.
_DIVIDER_CHARS = frozenset("-=~*_#")
_DIVIDER_RUN = 2
_DIVIDER_BARE_RUN = 3
# The one character authors wrap a title in singly (`~Part I~`).
_DIVIDER_WRAP = "~"

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


def tag_chain(name: str) -> list[str]:
    """Every tag in `name`'s leading run, outermost first: parse_tag() then
    strip_tag() until no tag leads, so "[P1][D1] x" gives ["P1", "D1"]."""
    chain = []
    while (tag := parse_tag(name)) is not None:
        chain.append(tag)
        name = strip_tag(name)
    return chain


def chain_segments(name: str) -> tuple[list[tuple[str, str]], str]:
    """`name`'s leading tag run cut into (segment, tag) pairs plus the rest,
    with `"".join(segments) + rest == name`. A segment is split_tag()'s prefix,
    tag, inner right padding and closer; the tags are tag_chain()'s."""
    segments = []
    pos = 0
    while (parts := split_tag(name[pos:])) is not None:
        prefix, tag, suffix = parts
        end = pos + len(prefix) + len(tag) + (len(suffix) - len(suffix.lstrip())) + 1
        segments.append((name[pos:end], tag))
        pos = end
    return segments, name[pos:]


def retag_in_chain(name: str, old: str, new: str) -> str:
    """`name` with every `old` in its leading tag chain replaced by `new` in
    place (retag_name() per segment), every other character kept."""
    segments, rest = chain_segments(name)
    if all(tag != old for _, tag in segments):
        raise ValueError(f"{name!r} carries no tag {old!r}")
    return "".join(retag_name(text, new) if tag == old else text for text, tag in segments) + rest


def strip_tag_from_chain(name: str, tag: str) -> str:
    """`name` with every `tag` in its leading tag chain removed, wherever it
    sits. Dropping the first segment follows strip_tag(): leading whitespace
    kept, the whitespace after the removed closer dropped."""
    segments, rest = chain_segments(name)
    if all(t != tag for _, t in segments):
        raise ValueError(f"{name!r} carries no tag {tag!r}")
    kept = "".join(text for text, t in segments if t != tag) + rest
    if segments[0][1] != tag:
        return kept
    return name[: len(name) - len(name.lstrip())] + kept.lstrip()


def leading_divider_run(name: str) -> int:
    """How many divider characters `name` (stripped) starts with.

    Used by is_divider(), and by tools/census_trigger_names.py's near-miss
    diagnostic (a name starting with >=2 divider characters that
    is_divider() still rejects).
    """
    stripped = name.strip()
    lead = 0
    while lead < len(stripped) and stripped[lead] in _DIVIDER_CHARS:
        lead += 1
    return lead


MAX_PREFIXES = 8
MAX_PREFIX_LEN = 16
_LONE_TAG_OPENERS = frozenset(_TAG_DELIMITERS)


def prefix_error(token: str) -> str:
    """Why `token` cannot be an extra header prefix, or "" when it can: the one
    validator for Settings load and the dialog."""
    if not token:
        return "a prefix is empty"
    if any(c.isspace() for c in token):
        return f"{token!r} contains whitespace"
    if len(token) > MAX_PREFIX_LEN:
        return f"{token!r} is longer than {MAX_PREFIX_LEN} characters"
    if token in _LONE_TAG_OPENERS:
        return f"{token!r} alone would make every {token}tag{_TAG_DELIMITERS[token]} trigger a header"
    if all(c in _DIVIDER_CHARS for c in token):
        return f"{token!r} is a divider rule: use the shape checkboxes instead"
    return ""


@dataclass(frozen=True)
class DividerRule:
    """Which name shapes head a section. The two-sided `--- X ---` shape is
    always on (New Section writes it); the other built-in shapes can be
    switched off, and `prefixes` adds starting tokens of the user's own.
    `prefixes` is kept longest first so two equal rules compare equal."""

    bare_rule: bool = True
    leading_rule: bool = True
    tilde_wrap: bool = True
    prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefixes", tuple(sorted(self.prefixes, key=lambda t: (-len(t), t))))

    @classmethod
    def with_prefixes(cls, prefixes: Sequence[str], **toggles: bool) -> DividerRule:
        return cls(prefixes=tuple(prefixes), **toggles)

    def error(self) -> str:
        """Why this rule cannot be made active, or "" when it can."""
        if len(self.prefixes) > MAX_PREFIXES:
            return f"at most {MAX_PREFIXES} prefixes"
        if len(set(self.prefixes)) != len(self.prefixes):
            return "a prefix is listed twice"
        return next((err for err in map(prefix_error, self.prefixes) if err), "")

    @classmethod
    def from_config(cls, value: object) -> DividerRule:
        """The rule a `trigger_section_headers` config value describes. Never
        raises: a malformed value or toggle falls back to its default, and an
        invalid, duplicate or ninth-and-later prefix is dropped."""
        if not isinstance(value, dict):
            return DEFAULT_DIVIDER_RULE
        toggles = {}
        for key in ("bare_rule", "leading_rule", "tilde_wrap"):
            flag = value.get(key)
            if isinstance(flag, bool):
                toggles[key] = flag
        prefixes: list[str] = []
        raw = value.get("prefixes")
        for token in raw if isinstance(raw, list) else ():
            if isinstance(token, str) and not prefix_error(token) and token not in prefixes:
                prefixes.append(token)
        return cls(prefixes=tuple(prefixes[:MAX_PREFIXES]), **toggles)

    def to_config(self) -> dict[str, object]:
        return {
            "bare_rule": self.bare_rule,
            "leading_rule": self.leading_rule,
            "tilde_wrap": self.tilde_wrap,
            "prefixes": list(self.prefixes),
        }


DEFAULT_DIVIDER_RULE = DividerRule()
_rule = DEFAULT_DIVIDER_RULE


def divider_rule() -> DividerRule:
    """The active rule every function here reads."""
    return _rule


def set_divider_rule(rule: DividerRule) -> None:
    """Make `rule` active process-wide. Only the viewer calls this, from
    Settings; batch scripts and tools keep DEFAULT_DIVIDER_RULE. Raises
    ValueError for a rule DividerRule.error() refuses (a caller bug: the
    dialog and the settings loader both validate first)."""
    global _rule
    err = rule.error()
    if err:
        raise ValueError(err)
    _rule = rule


def _builtin_shape(stripped: str) -> str | None:
    """Which built-in shape a stripped name has: "two_sided", "bare_rule",
    "leading_rule", "tilde_wrap", or None."""
    if all(c in _DIVIDER_CHARS for c in stripped):
        return "bare_rule" if len(stripped) >= _DIVIDER_BARE_RUN else None
    lead = leading_divider_run(stripped)
    trail = 0
    while trail < len(stripped) and stripped[-(trail + 1)] in _DIVIDER_CHARS:
        trail += 1
    if lead >= _DIVIDER_RUN:
        if trail >= _DIVIDER_RUN:
            return "two_sided"
        after = stripped[lead]
        return "leading_rule" if after.isspace() or after.isalnum() else None
    if lead == 1 and trail == 1 and stripped[0] == stripped[-1] == _DIVIDER_WRAP and stripped[1:-1].strip():
        return "tilde_wrap"
    return None


def _matched_prefix(stripped: str, rule: DividerRule) -> str | None:
    for token in rule.prefixes:
        if stripped.startswith(token):
            rest = stripped[len(token):].strip()
            if rest and not all(c in _DIVIDER_CHARS for c in rest):
                return token
    return None


def _divider_match(name: str) -> tuple[str, str | None] | None:
    """("builtin", shape) or ("prefix", token) for a header under the active
    rule, else None. Built-in shapes are tried first and keep their own cut."""
    stripped = name.strip()
    if not stripped:
        return None
    rule = _rule
    shape = _builtin_shape(stripped)
    if shape == "two_sided" or (shape is not None and getattr(rule, shape)):
        return "builtin", shape
    token = _matched_prefix(stripped, rule)
    return ("prefix", token) if token is not None else None


def is_divider(name: str) -> bool:
    """Whether `name` reads as a `--- Section ---`-style header under the
    active DividerRule (divider_rule()).

    A name consisting entirely of >=3 characters from _DIVIDER_CHARS is a
    bare divider (`------`, toggle `bare_rule`); a 2-character `--` is
    neither a real rule nor two overlapping runs of one. Otherwise, stripped,
    a name is a divider in any of three built-in shapes:

    - it starts with >=2 AND ends with >=2 of them (`--- Setup ---`,
      `---Setup---`, `--Start--`), always on;
    - it starts with >=2 of them followed by whitespace or a letter or digit
      (`----- ambush-setup`, `--lost-both-path`, toggle `leading_rule`). The
      character after the run keeps an arrow such as `--> Attack` out;
    - one `~` on each side of a title (`~Part I~`, toggle `tilde_wrap`). `~`
      is the only character the corpus wraps singly; `*Boss*` reads as
      emphasis and stays out.

    Failing those, it is a divider when it starts with one of the rule's
    `prefixes` followed by something other than whitespace or divider
    characters (`>> Setup` under prefix `>>`).

    A false positive here costs a cosmetic nesting level, never data: the
    header row IS the divider trigger, so it stays selectable, editable, and
    movable either way. That is what makes a fuzzy heuristic acceptable --
    see the plan's "the one invariant that makes false positives safe".
    """
    return _divider_match(name) is not None


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
    around nothing but whitespace). A prefix header's prefix is its leading
    whitespace, the prefix token and the whitespace after it; its suffix is
    trailing whitespace only.
    """
    match = _divider_match(name)
    if match is None or match[1] == "bare_rule":
        return None
    start = len(name) - len(name.lstrip())
    if match[0] == "prefix":
        start += len(match[1])
        while name[start].isspace():
            start += 1
        end = len(name.rstrip())
        return name[:start], name[start:end], name[end:]
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
    slot, its runs mix characters (no single `char` describes it), or either
    run is shorter than two (a marker such as `----- ambush-setup` or `~P2~`),
    so New Section only ever emits the two-sided shape."""
    parts = split_divider(name)
    if parts is None:
        return None
    prefix, _, suffix = parts
    lead_ws = prefix[: len(prefix) - len(prefix.lstrip())]
    lead_run = prefix[len(lead_ws):].rstrip()
    trail_run = suffix.strip()
    if len(lead_run) < _DIVIDER_RUN or len(trail_run) < _DIVIDER_RUN:
        return None
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


COPY_SUFFIX = " (copy)"


def copied_name(name: str, suffix: str = COPY_SUFFIX) -> str:
    """The name a copy of `name` gets (GH #134): `suffix` inside a titled
    divider's title, so the copy still heads a section of its own; a bare
    run like `------` verbatim; anything else with `suffix` appended.

    `name` verbatim too whenever the suffixed name's header status would
    differ (`------` with `bare_rule` off, a lone `>>` under a `>>` prefix):
    trigger_model.display_order_with_copies_inserted() places copies from the
    source names, so a copy that turned into a header would take over the
    rest of its section."""
    parts = split_divider(name)
    if parts is not None:
        copied = parts[0] + parts[1] + suffix + parts[2]
    elif is_divider(name):
        return name
    else:
        copied = name + suffix
    return copied if is_divider(copied) == is_divider(name) else name


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
    A title key alone misses when same-titled sections swap or one is added or
    removed above another, so the panel re-keys across structural edits and
    undo by header identity (TriggerPanel.follow_collapse(), TASK-225).
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
