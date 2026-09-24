from __future__ import annotations

import pytest

from descape import changelog
from tools import release_notes

_SAMPLE = """\
# Changelog

## [0.4] - 2026-09-07

### Added

- A Cliff tool in Terrain mode.

## [0.3] - 2026-08-15

### Fixed

- An older fix.
"""


@pytest.fixture()
def sample_changelog(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    return tmp_path


def test_section_is_collapsed_and_footer_is_not(sample_changelog):
    notes = release_notes.render("0.4")
    opened, closed = notes.index("<details>"), notes.index("</details>")
    assert notes.startswith("<details>\n<summary><b>Full changelog for v0.4</b>")
    assert opened < notes.index("- A Cliff tool in Terrain mode.") < closed
    assert closed < notes.index("**Corresponding Source (GPL-3.0):**")
    assert "An older fix." not in notes


def test_blank_lines_keep_markdown_rendering(sample_changelog):
    # GitHub renders a <details> body as markdown only after a blank line, and a "---"
    # straight under a text line would become a heading instead of a rule.
    notes = release_notes.render("0.4")
    assert "</summary>\n\n### Added" in notes
    assert "in Terrain mode.\n\n</details>\n\n---\n" in notes


def test_unknown_version_raises(sample_changelog):
    with pytest.raises(ValueError):
        release_notes.render("0.9")


def test_main_prints_render(sample_changelog, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["release_notes.py", "v0.4"])
    assert release_notes.main() == 0
    assert capsys.readouterr().out == release_notes.render("0.4")
