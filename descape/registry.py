"""A name-to-entry container that refuses a taken name and can roll back a
batch of registrations.

The addon loader reads `generation` before importing an addon and rolls back
to it if the import raises, so an addon that registers one command and then
fails leaves nothing behind: a half-registered entry would be offered in a
menu with nothing to say why it does not work.

Qt-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class ShadowedNameError(ValueError):
    """A registration asked for a name that is already taken."""


class Registry:
    """Names to entries, in registration order. An entry is anything with a
    non-empty string `name` attribute, the key it is filed under."""

    def __init__(self, kind: str = "entry") -> None:
        self._kind = kind
        self._entries: dict[str, Any] = {}
        self._added_at: dict[str, int] = {}
        self._generation = 0

    def register(self, entry: Any) -> None:
        """Files `entry` under its `name`. Raises ShadowedNameError for a taken
        name and ValueError for a missing one, changing nothing either way."""
        name = getattr(entry, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"a {self._kind} must carry a non-empty name: {entry!r}")
        existing = self._entries.get(name)
        if existing is not None:
            raise ShadowedNameError(f"{self._kind} name {name!r} is already registered to {existing!r}")
        self._generation += 1
        self._entries[name] = entry
        self._added_at[name] = self._generation

    def get(self, name: str) -> Any | None:
        return self._entries.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def entries(self) -> Mapping[str, Any]:
        """A read-only live view of name to entry, in registration order."""
        return MappingProxyType(self._entries)

    @property
    def generation(self) -> int:
        """Bumped by every register, rollback and reset; never reused."""
        return self._generation

    def rollback_to(self, generation: int) -> None:
        """Drops everything registered after `generation`."""
        for name in [name for name, at in self._added_at.items() if at > generation]:
            del self._entries[name]
            del self._added_at[name]
        self._generation += 1

    def reset(self) -> None:
        """Drops every entry. For test teardown."""
        self._entries.clear()
        self._added_at.clear()
        self._generation += 1


__all__ = ["Registry", "ShadowedNameError"]
