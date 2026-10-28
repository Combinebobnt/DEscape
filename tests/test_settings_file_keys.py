"""TASK-497: every config.yaml key is either portable or local, exactly once.

Discovery reads descape/'s sources for the shapes config keys are read and
written through (`config["k"]`, `config.get("k"`, `config.pop("k"`,
`_load_config().get("k"`), plus the two names that reach config through a
variable: settings._LEGACY_TOOL_OPTION_KEYS' legacy keys and
content_roots.PROFILE_CONFIG_KEY. A key in neither map would drop out of
exported files without anyone noticing, or a path would leak into them; a
registry key discovery never finds is a retired key lingering there.
"""

from __future__ import annotations

import re
from pathlib import Path

from descape import content_roots, settings, settings_file, tool_options

DESCAPE = Path(__file__).resolve().parent.parent / "descape"

_READ = re.compile(r'\b(?:config\[|config\.get\(|config\.pop\(|_load_config\(\)\.get\()"([^"]+)"')
_WRITE = re.compile(r'\bconfig\["([^"]+)"\]\s*=(?!=)')

# Discovery hits that are not config.yaml keys. None today.
NOT_CONFIG_KEYS: frozenset[str] = frozenset()
# Read but never written, and not legacy: set by hand in config.yaml only.
READ_ONLY_KEYS = frozenset({"language"})


def _scan() -> tuple[set[str], set[str]]:
    read: set[str] = set()
    written: set[str] = set()
    for path in sorted(DESCAPE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        read.update(_READ.findall(text))
        written.update(_WRITE.findall(text))
    read.update(legacy for legacy, _name in settings._LEGACY_TOOL_OPTION_KEYS)
    read.add(content_roots.PROFILE_CONFIG_KEY)
    return read - NOT_CONFIG_KEYS, written


def test_every_config_key_is_portable_or_local() -> None:
    discovered, _written = _scan()
    unclassified = discovered - set(settings_file.PORTABLE_KEYS) - set(settings_file.LOCAL_KEYS)
    assert not unclassified, (
        f"config keys in neither settings_file.PORTABLE_KEYS nor LOCAL_KEYS: {sorted(unclassified)}. "
        "A preference goes in PORTABLE_KEYS; a path, last-used state or screen geometry in LOCAL_KEYS."
    )


def test_no_key_is_both_portable_and_local() -> None:
    assert not set(settings_file.PORTABLE_KEYS) & set(settings_file.LOCAL_KEYS)


def test_every_registry_key_is_one_the_code_uses() -> None:
    discovered, _written = _scan()
    stale = (set(settings_file.PORTABLE_KEYS) | set(settings_file.LOCAL_KEYS)) - discovered
    assert not stale, f"registry keys no code reads or writes any more: {sorted(stale)}"


def test_every_legacy_key_has_a_successor_the_registry_carries() -> None:
    discovered, written = _scan()
    legacy = discovered - written - READ_ONLY_KEYS
    assert legacy == set(settings_file.LEGACY_SUCCESSORS)
    for legacy_key, (successor, option) in settings_file.LEGACY_SUCCESSORS.items():
        assert legacy_key in settings_file.PORTABLE_KEYS, legacy_key
        assert successor in settings_file.PORTABLE_KEYS, (legacy_key, successor)
        assert option is None or option in tool_options.BY_NAME, (legacy_key, option)
