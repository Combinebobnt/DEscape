from __future__ import annotations

import pytest

from descape import changelog
from tools import check_release_tag

_SAMPLE = """\
# Changelog

## [Unreleased]

- Something not yet released.

## [0.4] - 2026-09-07

### Added

- A Cliff tool in Terrain mode.
"""

_SAMPLE_EMPTY_SECTION = """\
# Changelog

## [0.4] - 2026-09-07
## [0.3] - 2026-08-15

- old
"""


@pytest.fixture()
def sample_changelog(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    monkeypatch.setattr(check_release_tag, "__version__", "0.4")
    return tmp_path


def test_matching_tag_passes(sample_changelog):
    assert check_release_tag.check("v0.4") is None


def test_version_mismatch_fails(sample_changelog, monkeypatch):
    monkeypatch.setattr(check_release_tag, "__version__", "0.5")
    error = check_release_tag.check("v0.4")
    assert error is not None
    assert "__version__" in error


def test_changelog_mismatch_fails(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    monkeypatch.setattr(check_release_tag, "__version__", "0.9")
    error = check_release_tag.check("v0.9")
    assert error is not None
    assert "CHANGELOG.md" in error


def test_empty_changelog_section_fails(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(_SAMPLE_EMPTY_SECTION, encoding="utf-8")
    monkeypatch.setattr(changelog, "ROOT", tmp_path)
    monkeypatch.setattr(check_release_tag, "__version__", "0.4")
    error = check_release_tag.check("v0.4")
    assert error is not None
    assert "section body" in error


def test_tag_without_v_prefix(sample_changelog):
    assert check_release_tag.check("0.4") is None
