"""Coverage for descape/asset_source.py's string-table reader and language
config (phase 4d, slice 4): resource_string(), get_language(), and the cache
invalidation set_install_path_override() drives.

_isolated_settings (conftest.py, autouse) already redirects CONFIG_PATH to a
throwaway path for every test here, so get_language() sees no config unless
a test writes one itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import asset_source

STRINGS_TEXT = '''\
// a comment line, and a blank line below

5164 "Town Center"
7427 "Anarchy"
3125 "Sent to \\"%s\\":"
'''


def _install_with_strings(tmp_path: Path, lang: str = "en") -> Path:
    install = tmp_path / "install"
    strings_dir = install / "resources" / lang / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text(STRINGS_TEXT, encoding="utf-8")
    return install


@pytest.fixture
def installed(tmp_path):
    install = _install_with_strings(tmp_path)
    asset_source.set_install_path_override(install)
    yield install
    asset_source.set_install_path_override(None)


def test_resource_string_resolves_a_known_key(installed) -> None:
    assert asset_source.resource_string(5164) == "Town Center"
    assert asset_source.resource_string(7427) == "Anarchy"


def test_resource_string_unescapes_embedded_quotes(installed) -> None:
    assert asset_source.resource_string(3125) == 'Sent to "%s":'


def test_resource_string_is_none_for_an_unknown_key(installed) -> None:
    assert asset_source.resource_string(999999) is None


def test_resource_string_is_none_with_no_install_configured() -> None:
    assert asset_source.resource_string(5164) is None


def test_resource_string_is_none_for_a_language_with_no_strings_file(installed) -> None:
    assert asset_source.resource_string(5164, lang="de") is None


def test_resource_string_uses_get_language_by_default(tmp_path) -> None:
    install = tmp_path / "install"
    (install / "resources" / "de" / "strings" / "key-value").mkdir(parents=True)
    (install / "resources" / "de" / "strings" / "key-value" / "key-value-strings-utf8.txt").write_text(
        '5164 "Stadtzentrum"\n', encoding="utf-8"
    )
    asset_source.set_install_path_override(install)
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"language": "de"}))
    try:
        assert asset_source.resource_string(5164) == "Stadtzentrum"
    finally:
        asset_source.set_install_path_override(None)


def test_set_install_path_override_invalidates_the_string_table_cache(tmp_path) -> None:
    """A key resolved under one install must not survive switching to a
    second install (or to none) that does not carry it."""
    install = _install_with_strings(tmp_path)
    asset_source.set_install_path_override(install)
    try:
        assert asset_source.resource_string(5164) == "Town Center"
    finally:
        asset_source.set_install_path_override(None)
    assert asset_source.resource_string(5164) is None


def test_get_language_defaults_to_en_with_no_config() -> None:
    assert asset_source.get_language() == "en"


def test_get_language_reads_the_config_key() -> None:
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"language": "fr"}))
    assert asset_source.get_language() == "fr"


def test_get_language_tolerates_a_malformed_config() -> None:
    asset_source.CONFIG_PATH.write_text("not: valid: yaml: at: all: [")
    assert asset_source.get_language() == "en"
