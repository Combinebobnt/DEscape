"""GH #160's built-in guide: the topic registry, the lints every topic file
passes (links, menu paths, typed keys, style), and the Keyboard and Mouse
page's generated block. Each lint is a function checked against a bad sample
too, so a lint that silently accepts everything fails here."""

from __future__ import annotations

import re

import pytest

from descape import help_guide, settings

import conftest

_FILES = [topic.filename for topic in help_guide.TOPICS]

# Edit-menu texts that only show with a focus or selection the mode switch below does not set up.
_FOCUS_ONLY_TEXTS = {
    "Edit": {
        "Cut",
        "Copy",
        "Paste",
        "Copy Units",
        "Paste Units",
        "Cut Conditions/Effects",
        "Copy Conditions/Effects",
        "Paste Conditions/Effects",
    },
    # The item reads "Release Notes for <version>"; the guide names it without the version.
    "Help": {"Release Notes"},
}

# (file, key): a fixed gesture that equals a rebindable default. The info tree's Ctrl+C.
_TYPED_KEY_ALLOWLIST = {("keyboard-and-mouse.md", "Ctrl+C")}


def _texts() -> dict[str, str]:
    return {topic.filename: help_guide.topic_text(topic) for topic in help_guide.TOPICS}


# -- the registry -------------------------------------------------------------


def test_every_topic_file_exists_and_starts_with_its_title() -> None:
    assert len(set(_FILES)) == len(_FILES)
    assert len({t.topic_id for t in help_guide.TOPICS}) == len(help_guide.TOPICS)
    for topic in help_guide.TOPICS:
        path = help_guide.topic_path(topic)
        assert path.is_file(), path
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("# "), (topic.filename, first)
        assert help_guide.topic_title(topic) == first[2:].strip()


def test_no_guide_file_is_missing_from_topics() -> None:
    on_disk = sorted(p.name for p in help_guide.GUIDE_DIR.glob("*.md"))
    assert on_disk == sorted(_FILES)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_every_mode_has_a_topic_and_only_view_opens_getting_started() -> None:
    """Read from Topic.modes, not topic_for_mode(), whose fallback would pass any mode."""
    from descape.viewer import _mode_id

    window = conftest.blank_window(load=False)
    try:
        combo = window.mode_combo
        mode_ids = {_mode_id(combo.itemText(i)) for i in range(combo.count())}
    finally:
        conftest.close_window(window)
    claimed = [mode for topic in help_guide.TOPICS for mode in topic.modes]
    assert len(claimed) == len(set(claimed)), "a mode claimed by two topics"
    assert mode_ids <= set(claimed), mode_ids - set(claimed)
    assert help_guide.topic_by_id("getting_started").modes == ("view",)
    assert help_guide.topic_for_mode("view").topic_id == "getting_started"
    assert help_guide.topic_for_mode("units").topic_id == "units"


# -- links ----------------------------------------------------------------------

_LINK = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def _link_problems(filename: str, text: str) -> list[str]:
    problems = []
    for target in _LINK.findall(text):
        if target.startswith("https://"):
            continue
        if "#" in target:
            problems.append(f"{filename}: {target} (Qt cannot follow a #heading link)")
        elif target not in _FILES:
            problems.append(f"{filename}: {target} (not https:// and not a topic file)")
    return problems


def test_link_lint_catches_bad_links() -> None:
    sample = "[a](units.md) [b](https://ugc.aoe2.rocks/x/#y) [c](units.md#sel) [d](http://x.org) [e](nope.md)"
    assert _link_problems("x.md", sample) == [
        "x.md: units.md#sel (Qt cannot follow a #heading link)",
        "x.md: http://x.org (not https:// and not a topic file)",
        "x.md: nope.md (not https:// and not a topic file)",
    ]


def test_every_link_is_https_or_a_topic_file() -> None:
    problems = [p for name, text in _texts().items() for p in _link_problems(name, text)]
    assert not problems, problems


# -- menu paths -----------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _merge(into: dict, menu) -> None:
    for action in menu.actions():
        if action.isSeparator():
            continue
        text = action.text().replace("&", "")
        sub = into.setdefault(text, {})
        if action.menu() is not None:
            _merge(sub, action.menu())


def _menu_resolver():
    """A function path -> bool over every menu text any mode shows, plus the
    Settings tabs, the Filters popup and the focus-only aliases."""
    from PyQt5.QtWidgets import QTabWidget

    from descape.viewer import SettingsDialog

    window = conftest.blank_window()
    try:
        bar: dict = {}
        combo = window.mode_combo
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            for action in window.menuBar().actions():
                _merge(bar.setdefault(action.text().replace("&", ""), {}), action.menu())
        for title, texts in _FOCUS_ONLY_TEXTS.items():
            for text in texts:
                bar[title].setdefault(text, {})
        filters: dict = {}
        _merge(filters, window.filters_button.menu())
        dialog = SettingsDialog(window)
        try:
            tabs = dialog.findChild(QTabWidget)
            settings_tabs = {tabs.tabText(i).replace("&", "") for i in range(tabs.count())}
        finally:
            dialog.close()
    finally:
        conftest.close_window(window)

    def resolves(path: str) -> bool:
        parts = [part.strip() for part in path.split(" > ")]
        if parts[0] == "Settings":
            return len(parts) >= 2 and parts[1] in settings_tabs
        node = {"Filters": filters, **bar}
        for part in parts:
            if part not in node:
                return False
            node = node[part]
        return True

    return resolves


def _menu_path_problems(filename: str, text: str, resolves) -> list[str]:
    return [
        f"{filename}: **{span}**"
        for span in (m.group(1) for m in _BOLD.finditer(text))
        if " > " in span and not resolves(span)
    ]


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_menu_path_lint_resolves_real_paths_and_rejects_others() -> None:
    resolves = _menu_resolver()
    for good in (
        "Edit > Find and Replace…",
        "Edit > Copy Region",
        "Edit > Paste Triggers",
        "Edit > Copy Conditions/Effects",
        "Map > Resize Map…",
        "File > Recover from Autosave…",
        "Help > Release Notes",
        "Help > DEscape Guide",
        "Settings > Keybinds",
        "Settings > Appearance > Theme",
        "Filters > Show Trees",
    ):
        assert resolves(good), good
    for bad in (
        "Edit > Frobnicate",
        "Edit > Find and Replace...",
        "Settings > Frobnicate",
        "Filters > Frobnicate",
        "Nowhere > Thing",
        "Help > Release Notes for 0.0",
    ):
        assert not resolves(bad), bad
    assert _menu_path_problems("x.md", "Use **Edit > Frobnicate** or **bold** text.", resolves) == [
        "x.md: **Edit > Frobnicate**"
    ]


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_every_bold_menu_path_in_the_guide_resolves() -> None:
    resolves = _menu_resolver()
    problems = [p for name, text in _texts().items() for p in _menu_path_problems(name, text, resolves)]
    assert not problems, f"menu paths that resolve nowhere: {problems}"


# -- typed keys -----------------------------------------------------------------


def _typed_key_problems(filename: str, text: str) -> list[str]:
    defaults = sorted({default for _, _, default in settings.REBINDABLE_ACTIONS if "+" in default})
    return [
        f"{filename}: {key}"
        for key in defaults
        if (filename, key) not in _TYPED_KEY_ALLOWLIST and re.search(rf"(?<![\w+]){re.escape(key)}(?![\w+])", text)
    ]


def test_typed_key_lint_catches_a_default_but_not_a_gesture() -> None:
    assert _typed_key_problems("x.md", "Press Ctrl+V to paste.") == ["x.md: Ctrl+V"]
    assert _typed_key_problems("x.md", "Ctrl+click toggles, Ctrl+drag toggles a box, ctrl+v is not a key.") == []
    assert _typed_key_problems("x.md", "Ctrl+Shift+S saves as.") == ["x.md: Ctrl+Shift+S"]
    assert _typed_key_problems("keyboard-and-mouse.md", "Ctrl+C copies the rows.") == []
    assert _typed_key_problems("units.md", "Ctrl+C copies the rows.") == ["units.md: Ctrl+C"]


def test_no_topic_types_a_rebindable_key() -> None:
    problems = [p for name, text in _texts().items() for p in _typed_key_problems(name, text)]
    assert not problems, f"name the command instead; the Keyboard and Mouse page shows its key: {problems}"


# -- style ----------------------------------------------------------------------

_EM_DASH = chr(0x2014)
# Spelled in two parts so tests/test_no_maintainer_refs.py doesn't read it as a citation.
_PRIVATE_DIR = "maintainer" + "/"
_STYLE = [
    ("an em dash", re.compile(_EM_DASH)),
    ("a double hyphen", re.compile(" -- ")),
    ("a task id", re.compile(r"TASK-\d")),
    ("an issue number", re.compile("GH #")),
    ("a private-repo path", re.compile(re.escape(_PRIVATE_DIR))),
]


def _style_problems(filename: str, text: str) -> list[str]:
    return [f"{filename}: {what}" for what, pattern in _STYLE if pattern.search(text)]


def test_style_lint_catches_each_pattern() -> None:
    sample = f"a {_EM_DASH} b -- c TASK-12 GH #5 {_PRIVATE_DIR}x"
    assert len(_style_problems("x.md", sample)) == len(_STYLE)


def test_guide_style() -> None:
    problems = [p for name, text in _texts().items() for p in _style_problems(name, text)]
    assert not problems, problems


# -- the Keyboard and Mouse page's live block ---------------------------------


def _unbound(**bound: str) -> dict[str, str]:
    bindings = {action_id: "" for action_id, _, _ in settings.REBINDABLE_ACTIONS}
    bindings.update(bound)
    return bindings


def test_keys_markdown_lists_bound_actions_under_their_section() -> None:
    block = help_guide.keys_markdown(
        _unbound(edit_paste="Ctrl+V", mode_units="Ctrl+U"), settings.DEFAULT_WHEEL_BINDINGS
    )
    lines = block.splitlines()
    assert lines.index("### Edit") < lines.index("| Paste | Ctrl+V |") < lines.index("### Modes")
    assert lines.index("### Modes") < lines.index("| Units Mode | Ctrl+U |")
    assert "| Copy |" not in block and "### File" not in block
    total = len(settings.REBINDABLE_ACTIONS)
    assert f"{total - 2} commands have no key. Set one in Settings > Keybinds." in lines
    for slot in settings.WHEEL_SLOTS:
        action = settings.WHEEL_ACTION_LABELS[settings.DEFAULT_WHEEL_BINDINGS[slot]]
        assert f"| {settings.WHEEL_SLOT_LABELS[slot]} | {action} |" in lines
    assert lines.index("### Mouse wheel over the map") < lines.index("### Edit")


def test_keys_markdown_formats_keys_and_escapes_markdown(monkeypatch) -> None:
    monkeypatch.setattr(
        settings, "REBINDABLE_ACTIONS", [("edit_x", "A|B", ""), ("edit_y", "Turn <", ""), ("edit_z", "Z", "")]
    )
    block = help_guide.keys_markdown(
        {"edit_x": "Ctrl+X", "edit_y": "<", "edit_z": ""},
        {"plain": "pan_vertical"},
        fmt=lambda key: key.replace("Ctrl", "Cmd"),
    )
    assert "| A\\|B | Cmd+X |" in block
    assert "| Turn \\< | \\< |" in block
    assert "1 command has no key." in block
    assert "| Wheel | Pan up / down |" in block


def test_splice_keys_puts_the_block_after_the_intro_and_before_the_first_section() -> None:
    static = "# Keyboard and Mouse\n\nIntro line.\n\n## Map\n\nMiddle-drag pans.\n\n## Units\n\nMore.\n"
    spliced = help_guide.splice_keys(static, "## Your current keys\n\nBLOCK\n")
    assert spliced.index("Intro line.") < spliced.index("BLOCK") < spliced.index("## Map")
    assert spliced.count("## Map") == 1 and spliced.endswith("More.\n")
    assert help_guide.splice_keys("# T\n\nOnly intro.\n", "BLOCK\n") == "# T\n\nOnly intro.\n\nBLOCK\n"
