from __future__ import annotations

import pytest

from descape import changelog

_SAMPLE = """\
# Changelog

## [Unreleased]

### Added

- Something not yet released.

## [0.4] - 2026-09-07

### Added

- A Cliff tool in Terrain mode.
- Another entry.

## [0.3] - 2026-08-15

### Fixed

- An older fix.
"""


@pytest.fixture()
def sample_changelog(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    return tmp_path


def test_newest_released_version_skips_unreleased(sample_changelog):
    assert changelog.newest_released_version() == "0.4"


def test_section_for_newest_version(sample_changelog):
    section = changelog.section_for("0.4")
    assert "A Cliff tool in Terrain mode." in section
    assert "Another entry." in section
    assert "An older fix." not in section
    assert "Unreleased" not in section


def test_section_for_older_version(sample_changelog):
    section = changelog.section_for("0.3")
    assert "An older fix." in section
    assert "A Cliff tool in Terrain mode." not in section


def test_section_for_missing_version_raises(sample_changelog):
    with pytest.raises(ValueError):
        changelog.section_for("9.9")


def test_newest_released_version_raises_if_none(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n- x\n", encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        changelog.newest_released_version()
