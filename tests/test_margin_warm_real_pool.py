"""TASK-031.11: a real viewer's margin warm runs its chunk kernels on the real
worker pool, and Perf Trace's `warm:` bucket reports them as `worker N chunks`.

tests/test_margin_warm.py's real-window test drains with run_to_completion(),
which is synchronous by design, and its pool tests use an inline executor.
tests/test_native_composite.py's real-pool test drives a bare cache. This one
pumps a shown viewer's own event loop with the shared ThreadPoolExecutor.
"""

from __future__ import annotations

import os
import re
import threading
import time

import pytest
from test_margin_warm import _zoomed_window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def native_kernel():
    """tests/test_native_composite.py's gate: worker jobs exist only on the native backend."""
    from descape import composite_backend

    if composite_backend.available():
        return
    if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
    pytest.skip(composite_backend.unavailable_reason)


def test_a_real_windows_margin_warm_runs_on_the_worker_pool_and_reports_worker_chunks(
    native_kernel, monkeypatch
) -> None:
    """Pass conditions, for the ring a zoomed Stepped viewer queues on a viewport change,
    drained only by pumping the event loop:
    - at least one chunk kernel ran, and every one ran on a `margin-warm` pool thread,
      never the GUI thread;
    - the `warm:` lines' `worker N chunks` add up to exactly the kernels that ran;
    - every ring chunk is cached afterwards.
    """
    from PyQt5.QtWidgets import QApplication

    from descape import debug_log, margin_warm, perf_trace

    if margin_warm.WORKERS <= 0:
        pytest.skip("margin_warm.WORKERS is 0: every chunk stays on the GUI thread")
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    runs: list[tuple[object, int, str]] = []
    real_run_job = margin_warm._run_job

    def run_job(deliver, token, job, bridge) -> None:
        runs.append((deliver.__self__, token, threading.current_thread().name))
        real_run_job(deliver, token, job, bridge)

    monkeypatch.setattr(margin_warm, "_run_job", run_job)
    window = _zoomed_window()
    try:
        # The resize and scale queued a 150 ms poll that would retarget the ring on a slow runner.
        window.map_view.stop_viewport_poll()
        warmer = window._margin_warmer
        # The load-time warm reports into the same bucket; only this ring is under test.
        window._load_warmer.cancel()
        window._on_viewport_changed()
        # Its first fire re-arms the level warm, whose on_job_done chains a load warm into the same bucket.
        window._level_warmer.cancel()
        window._load_warmer.cancel()
        assert warmer.is_active, "no ring was queued -- vacuous"
        cache, mip, ring, token = warmer._cache, warmer._mip, list(warmer._queue), warmer._token
        monkeypatch.setattr(perf_trace, "_enabled", True)
        debug_log.clear()
        deadline = time.monotonic() + 60
        while warmer.is_active:
            assert time.monotonic() < deadline, "the pooled warm never drained"
            QApplication.processEvents()
            time.sleep(0.001)
        perf_trace.flush_idle()
        assert warmer._token == token, "the ring was retargeted mid-test, so its counts are not this ring's"

        # Runs from a ring this one replaced carry an older token and are dropped unreported by design.
        ran_on = [name for owner, token, name in runs if owner is warmer and token == warmer._token]
        assert ran_on, "no chunk reached the pool -- every chunk ran synchronously"
        gui = threading.main_thread().name
        assert all(name.startswith("margin-warm") and name != gui for name in ran_on), ran_on
        text = debug_log.get_log_text()
        reported = [int(n) for n in re.findall(r"warm: [^\n]*?worker (\d+) chunks", text)]
        assert sum(reported) == len(ran_on), (reported, len(ran_on), text[-2000:])
        missing = [chunk for chunk in ring if not cache.has_chunk(mip, *chunk)]
        assert not missing, f"ring chunks never cached: {missing}"
    finally:
        conftest.close_window(window)
