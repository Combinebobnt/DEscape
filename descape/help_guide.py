"""The built-in guide's topics and the Help menu's web links (GH #160). No Qt
imports: the viewer and the help window call these, and tests check them
without a QApplication.

The guide is one Markdown file per topic under GUIDE_DIR, because Qt's
Markdown import gives headings no anchors: a link can only reach a whole file.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from descape import settings

REPO_URL = "https://github.com/Combinebobnt/DEscape"
UGC_SCENARIOS_URL = "https://ugc.aoe2.rocks/scenarios/"

# Module level so the frozen self-check finds it (packaging/entry_frozen.py).
GUIDE_DIR = Path(__file__).resolve().parent / "help"

KEYS_TOPIC_ID = "keys"


@dataclass(frozen=True)
class Topic:
    topic_id: str
    filename: str
    modes: tuple[str, ...] = ()


# Display order. `modes` are the mode ids (ViewerWindow.mode) whose context help opens the topic.
TOPICS: tuple[Topic, ...] = (
    Topic("getting_started", "getting-started.md", ("view",)),
    Topic("terrain", "terrain.md", ("terrain",)),
    Topic("units", "units.md", ("units",)),
    Topic("triggers", "triggers.md", ("triggers",)),
    Topic("map_options", "map-options.md", ("map_options",)),
    Topic("players", "players.md", ("players",)),
    Topic("diplomacy", "diplomacy.md", ("diplomacy",)),
    Topic("messages", "messages.md", ("messages",)),
    Topic("map_tools", "map-tools.md"),
    Topic("view_and_filters", "view-and-filters.md"),
    Topic("settings", "settings.md"),
    Topic(KEYS_TOPIC_ID, "keyboard-and-mouse.md"),
)
_FALLBACK_TOPIC_ID = "getting_started"


def topic_by_id(topic_id: str) -> Topic:
    """Raises KeyError for an unknown id."""
    for topic in TOPICS:
        if topic.topic_id == topic_id:
            return topic
    raise KeyError(topic_id)


def topic_by_filename(filename: str) -> Topic | None:
    return next((topic for topic in TOPICS if topic.filename == filename), None)


def topic_for_mode(mode_id: str) -> Topic:
    """The topic F1 opens in that mode; Getting Started for any other."""
    for topic in TOPICS:
        if mode_id in topic.modes:
            return topic
    return topic_by_id(_FALLBACK_TOPIC_ID)


def topic_path(topic: Topic) -> Path:
    # GUIDE_DIR read at call time, so a test can point it elsewhere.
    return GUIDE_DIR / topic.filename


def topic_text(topic: Topic) -> str:
    return topic_path(topic).read_text(encoding="utf-8")


def topic_title(topic: Topic) -> str:
    """The file's first `# ` line, so the title has one source."""
    for line in topic_text(topic).splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return topic.filename


_MD_SPECIAL = set("\\`*_<>|[]~")


def _md_escape(text: str) -> str:
    return "".join(f"\\{ch}" if ch in _MD_SPECIAL else ch for ch in text)


def keys_markdown(
    bindings: Mapping[str, str],
    wheel_bindings: Mapping[str, str],
    fmt: Callable[[str], str] = lambda s: s,
) -> str:
    """The Keyboard and Mouse page's live block: the wheel, then every action
    with a key, grouped under the Keybinds tab's section titles, then how many
    have none. `fmt` turns a stored key into display text."""
    lines = [
        "## Your current keys",
        "",
        "### Mouse wheel over the map",
        "",
        "| Wheel | Does |",
        "|---|---|",
    ]
    for slot in settings.WHEEL_SLOTS:
        action = wheel_bindings.get(slot, settings.DEFAULT_WHEEL_BINDINGS[slot])
        label = settings.WHEEL_ACTION_LABELS.get(action, action)
        lines.append(f"| {_md_escape(settings.WHEEL_SLOT_LABELS[slot])} | {_md_escape(label)} |")
    unbound = 0
    section = None
    for action_id, label, _default in settings.REBINDABLE_ACTIONS:
        key = bindings.get(action_id, "")
        if not key:
            unbound += 1
            continue
        prefix = action_id.split("_", 1)[0]
        if prefix != section:
            section = prefix
            title = settings.KEYBIND_SECTION_TITLES.get(prefix, prefix.title())
            lines += ["", f"### {_md_escape(title)}", "", "| Command | Key |", "|---|---|"]
        # A mode-scoped command names its modes, so two rows on one key read as each mode's.
        lines.append(f"| {_md_escape(label + settings.keybind_mode_tag(action_id))} | {_md_escape(fmt(key))} |")
    noun = "command has" if unbound == 1 else "commands have"
    lines += ["", f"{unbound} {noun} no key. Set one in Settings > Keybinds.", ""]
    return "\n".join(lines)


def splice_keys(static_text: str, block: str) -> str:
    """static_text with block inserted before its first `## ` line, or appended."""
    lines = static_text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("## "):
            return "".join(lines[:index]) + block.rstrip("\n") + "\n\n" + "".join(lines[index:])
    return static_text.rstrip("\n") + "\n\n" + block


def release_notes_url(version: str) -> str:
    return f"{REPO_URL}/releases/tag/v{version}"


def report_bug_url(version: str, os_text: str, build_kind: str) -> str:
    """A new-issue link with the environment filled in. Never a path, user
    name or scenario name: the body is built from these three values only."""
    body = "\n".join(
        [
            f"DEscape version: {version}",
            f"OS: {os_text}",
            f"Build: {build_kind}",
            "",
            "What happened:",
            "",
            "",
            "What you expected:",
            "",
            "",
            "Steps to reproduce:",
            "1. ",
            "",
            "If it is useful, paste the relevant lines from Help > Debug Log.",
        ]
    )
    return f"{REPO_URL}/issues/new?{urlencode({'body': body})}"


def environment_summary() -> tuple[str, str]:
    """(OS text, build kind). macOS's platform.release() is the Darwin kernel
    version, so its own version comes from mac_ver()."""
    system = platform.system()
    if system == "Darwin":
        os_text = f"macOS {platform.mac_ver()[0]}"
    else:
        os_text = f"{system} {platform.release()}"
    build_kind = "release build" if getattr(sys, "frozen", False) else "from source"
    return os_text, build_kind
