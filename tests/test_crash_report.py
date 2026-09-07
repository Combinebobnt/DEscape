"""descape/crash_report.py, in isolation: report text assembly, the atomic
write, retention pruning, the pending/.reported handshake, the rate
limiter, and the faulthandler-log rotation. Qt-free, so this runs in the
default tier like descape/backup.py's own tests.
"""

from __future__ import annotations

import time

from descape.crash_report import (
    RateLimiter,
    build_report,
    extract_summary,
    fingerprint,
    mark_reported,
    pending_reports,
    prune,
    rotate_faulthandler_log,
    write_report,
)


def _make_exc():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        return sys.exc_info()


def test_build_report_contains_traceback_version_and_log():
    exc_type, exc_value, exc_tb = _make_exc()
    text = build_report(
        exc_type, exc_value, exc_tb, version="0.3", log_text="[12:00:00] did a thing"
    )
    assert "RuntimeError: boom" in text
    assert "Version: 0.3" in text
    assert "did a thing" in text


def test_extract_summary_gets_the_exception_line():
    exc_type, exc_value, exc_tb = _make_exc()
    text = build_report(exc_type, exc_value, exc_tb, version="0.3", log_text="(empty)")
    assert extract_summary(text) == "RuntimeError: boom"


def test_write_report_lands_in_dump_dir(tmp_path):
    path = write_report("hello", tmp_path)
    assert path.parent == tmp_path
    assert path.read_text() == "hello"
    assert not list(tmp_path.glob("*.tmp"))


def test_prune_drops_oldest_beyond_keep(tmp_path):
    paths = []
    for i in range(7):
        p = tmp_path / f"crash-2026010{i}-000000-aaaaaaaa.txt"
        p.write_text(str(i))
        paths.append(p)
        time.sleep(0.01)
    prune(tmp_path, keep=5)
    remaining = sorted(p.name for p in tmp_path.glob("crash-*.txt"))
    assert len(remaining) == 5
    assert paths[0].name not in remaining
    assert paths[-1].name in remaining


def test_pending_reports_and_mark_reported_round_trip(tmp_path):
    a = write_report("a", tmp_path)
    b = write_report("b", tmp_path)
    assert set(pending_reports(tmp_path)) == {a, b}

    reported = mark_reported(a)
    assert reported.name == a.name + ".reported"
    assert reported.exists()
    assert not a.exists()
    assert pending_reports(tmp_path) == [b]


def test_rate_limiter_writes_once_per_fingerprint():
    limiter = RateLimiter(max_dumps=3)
    exc_type, _, exc_tb = _make_exc()
    fp = fingerprint(exc_type, exc_tb)

    assert limiter.should_write(fp) is True
    assert limiter.should_write(fp) is False


def test_rate_limiter_writes_for_each_distinct_fingerprint():
    limiter = RateLimiter(max_dumps=3)
    exc_type, _, exc_tb = _make_exc()
    fp1 = fingerprint(exc_type, exc_tb)
    fp2 = ("SomethingElse", ("other.py", 1))

    assert limiter.should_write(fp1) is True
    assert limiter.should_write(fp2) is True


def test_rate_limiter_hard_caps_total_dumps():
    limiter = RateLimiter(max_dumps=2)
    fps = [("Err", ("f.py", i)) for i in range(5)]
    results = [limiter.should_write(fp) for fp in fps]
    assert results == [True, True, False, False, False]


def test_rotate_faulthandler_log_moves_nonempty_log_into_pending(tmp_path):
    log_path = tmp_path / "faulthandler.log"
    log_path.write_text("Fatal Python error: Segmentation fault\n")

    dest = rotate_faulthandler_log(tmp_path)

    assert dest is not None
    assert not log_path.exists()
    assert dest in pending_reports(tmp_path)
    assert "Segmentation fault" in dest.read_text()


def test_rotate_faulthandler_log_leaves_empty_log_alone(tmp_path):
    log_path = tmp_path / "faulthandler.log"
    log_path.write_text("")

    dest = rotate_faulthandler_log(tmp_path)

    assert dest is None
    assert log_path.exists()


def test_rotate_faulthandler_log_no_log_present(tmp_path):
    assert rotate_faulthandler_log(tmp_path) is None
