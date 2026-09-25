"""Which composite backend render.py uses: the native kernel
(descape/_composite_native.pyx, built by tools/build_native.py) or the numpy
code it replaces, which stays as the fallback and the byte-identity oracle.

render.py reads `native` at call time, never caching it, so use_backend()
can flip it mid-process (tests/test_native_composite.py runs both backends
in one process). DESCAPE_COMPOSITE=numpy|native forces one at import; forcing
native when it can't load raises rather than silently running numpy.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import ModuleType

# Must equal _composite_native.KERNEL_ABI; a stale build from an older
# checkout is ignored until rebuilt, never run against newer callers.
EXPECTED_KERNEL_ABI = 1

BACKENDS = ("numpy", "native")


def _load() -> tuple[ModuleType | None, str | None]:
    try:
        from descape import _composite_native
    except ImportError as exc:
        return None, f"native kernel not built ({exc}); build it with tools/build_native.py, which needs a C compiler"
    abi = getattr(_composite_native, "KERNEL_ABI", None)
    if abi != EXPECTED_KERNEL_ABI:
        return None, f"native kernel is stale (KERNEL_ABI {abi}, expected {EXPECTED_KERNEL_ABI}); rebuild with tools/build_native.py"
    return _composite_native, None


_module, unavailable_reason = _load()


def available() -> bool:
    return _module is not None


def _resolve(name: str) -> ModuleType | None:
    if name not in BACKENDS:
        raise ValueError(f"unknown composite backend {name!r}, expected one of {BACKENDS}")
    if name == "numpy":
        return None
    if _module is None:
        raise RuntimeError(f"composite backend 'native' requested but {unavailable_reason}")
    return _module


# The active kernel module, or None for numpy. render.py's only hook.
native: ModuleType | None = _resolve(os.environ["DESCAPE_COMPOSITE"]) if os.environ.get("DESCAPE_COMPOSITE") else _module


def active_backend() -> str:
    return "numpy" if native is None else "native"


@contextmanager
def use_backend(name: str):
    """Runs the block on backend `name`, restoring the previous one after."""
    global native
    previous = native
    native = _resolve(name)
    try:
        yield
    finally:
        native = previous


def describe() -> str:
    """One line for the debug log: the active backend and, when it is numpy
    by necessity rather than by choice, why."""
    line = f"Composite backend: {active_backend()}"
    if native is None and _module is None:
        line += f" ({unavailable_reason})"
    elif native is None:
        line += " (forced by DESCAPE_COMPOSITE)"
    return line
