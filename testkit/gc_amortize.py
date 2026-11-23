"""Keeps full collections from costing a test suite more than its tests do.

Every scenario load runs `_freeze_loaded_document()`: one full collection,
then `gc.freeze()`. In a long-lived test worker that collection walks every
object the worker holds (pytest's collected items, other tests' tables), so
it took 333 ms in-suite against 70 ms alone, and full collections were ~40%
of the quick suite's summed time.

- `install()` wraps the method once: during the call, `descape.viewer_file`'s
  module-level `gc` is a stand-in whose `collect()` and `freeze()` do
  nothing (every viewer-family module's `gc`, through `set_viewer_global()`),
  so the method's own body (timing, gc_hold) still runs. A test
  marked `real_gc_freeze` gets the real `gc`. `gc` itself is never patched.
- `after_test()`, from `tests/conftest.py`'s teardown after the window
  reaper, runs one full collection every `COLLECT_EVERY` tests instead,
  which frees the reaped windows' graphs. Freeze and collect go together:
  freezing without that collect would keep each reaped window alive.
- `drop_data_mod_caches()` drops every `game_dataset.cache()` entry kept for
  a data mod. Each data-mod test keys its own tables by its `tmp_path`
  install, so they would otherwise pile up for the rest of the worker.
"""

from __future__ import annotations

import functools
import gc
import sys

COLLECT_EVERY = 25

_state = {"real": False, "tests": 0}


class _NoCollectGc:
    """descape.viewer_file's `gc` for one `_freeze_loaded_document()` call."""

    @staticmethod
    def collect(generation: int = 2) -> int:
        return 0

    @staticmethod
    def freeze() -> None:
        return None

    def __getattr__(self, name):
        return getattr(gc, name)


_SHIM = _NoCollectGc()


def install() -> bool:
    """Wraps `_freeze_loaded_document()` once `descape.viewer_file` is imported. True once installed."""
    viewer_file = sys.modules.get("descape.viewer_file")
    if viewer_file is None:
        return False
    mixin = viewer_file._ViewerFileMixin
    real = mixin._freeze_loaded_document
    if getattr(real, "_gc_amortized", False):
        return True

    @functools.wraps(real)
    def freeze_loaded_document(self) -> None:
        if _state["real"]:
            return real(self)
        import pytest

        from testkit.viewer_family import set_viewer_global

        # Every viewer-family module's gc: the call is synchronous, so only this method sees it.
        with pytest.MonkeyPatch.context() as mp:
            set_viewer_global(mp, "gc", _SHIM)
            return real(self)

    freeze_loaded_document._gc_amortized = True
    mixin._freeze_loaded_document = freeze_loaded_document
    return True


def begin_test(real_gc_freeze: bool, gui: bool) -> None:
    """Before a test's setup: install, importing the viewer first for a gui test (its fixtures
    may load before anything else imports it), and choose the real or stubbed load collect."""
    if gui:
        import descape.viewer_file  # noqa: F401
    install()
    _state["real"] = real_gc_freeze


def after_test() -> bool:
    """After a test's teardown: drop data-mod tables, and every COLLECT_EVERY tests collect. True when it collected."""
    drop_data_mod_caches()
    _state["real"] = False
    _state["tests"] += 1
    if _state["tests"] % COLLECT_EVERY:
        return False
    gc.collect()
    return True


def drop_data_mod_caches() -> int:
    """Deletes every `game_dataset.cache()` entry keyed by a data mod; returns how many."""
    game_dataset = sys.modules.get("descape.game_dataset")
    if game_dataset is None:
        return 0
    dropped = 0
    for per_dataset in game_dataset._caches:
        for key in [key for key in per_dataset if isinstance(key, tuple)]:
            del per_dataset[key]
            dropped += 1
    return dropped
