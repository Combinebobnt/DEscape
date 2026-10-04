"""descape/crash_report.py, in isolation: report text assembly, the atomic
write, retention pruning, the pending/.reported handshake, the rate
limiter, and the faulthandler-log rotation. Qt-free, so this runs in the
default tier like descape/backup.py's own tests.
"""

from __future__ import annotations

import getpass
import os
import stat
import time

import pytest

from descape import crash_report, debug_log
from descape.crash_report import (
    RateLimiter,
    build_report,
    extract_summary,
    fingerprint,
    mark_reported,
    pending_reports,
    prune,
    rotate_faulthandler_log,
    scrub_user_paths,
    write_report,
)


@pytest.fixture
def fake_user(monkeypatch):
    """Home /home/al and user `al`, never the real ones; returns a setter."""

    def set_user(home: str, user: str | None = "al") -> None:
        monkeypatch.setenv("HOME", home)
        # Windows Python reads USERPROFILE, not HOME.
        monkeypatch.setenv("USERPROFILE", home)

        def getuser() -> str:
            if user is None:
                raise OSError("no user")
            return user

        monkeypatch.setattr(getpass, "getuser", getuser)

    set_user("/home/al")
    return set_user


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


def test_a_report_scrubs_the_home_prefix_from_every_section(fake_user):
    exc_type, exc_value, exc_tb = _make_exc()
    text = build_report(
        exc_type,
        exc_value,
        exc_tb,
        version="0.3",
        log_text="Loaded /home/al/maps/a.aoe2scenario\nCrash report written to /home/al/.config/x.txt",
    )
    assert "/home/al" not in text
    assert "Loaded ~/maps/a.aoe2scenario" in text
    assert "written to ~/.config/x.txt" in text


def test_the_home_prefix_never_matches_a_longer_sibling_dir(fake_user):
    fake_user("/home/al", user=None)
    text = "/home/alpha/x.py /home/al.bak/y ('/home/al') /home/al: /home/al/z"
    assert scrub_user_paths(text) == "/home/alpha/x.py /home/al.bak/y ('~') ~: ~/z"


def test_a_root_or_empty_home_is_left_alone(fake_user):
    for home in ("/", ""):
        fake_user(home, user=None)
        assert scrub_user_paths("/usr/lib/x.py and /home/al/y") == "/usr/lib/x.py and /home/al/y"


def test_a_username_segment_outside_home_is_scrubbed(fake_user):
    text = "File /media/al/stick/mod.py\n/mnt/al\n'/srv/al' end"
    assert scrub_user_paths(text) == "File /media/<user>/stick/mod.py\n/mnt/<user>\n'/srv/<user>' end"


def test_the_home_dirs_last_component_counts_as_the_username(fake_user):
    fake_user("/home/al", user=None)
    assert scrub_user_paths("/media/al/stick") == "/media/<user>/stick"


def test_a_username_never_matches_inside_a_longer_segment(fake_user):
    text = "/alpha/x.py /data/al.txt /data/al-2/ /opt/pal/ al wrote /x/al"
    assert scrub_user_paths(text) == "/alpha/x.py /data/al.txt /data/al-2/ /opt/pal/ al wrote /x/<user>"


def test_windows_separators_are_scrubbed(fake_user):
    fake_user("C:\\Users\\al")
    text = 'File "C:\\Users\\al\\AppData\\x.py"\nD:\\al\\games\\y E:/al/z'
    assert scrub_user_paths(text) == 'File "~\\AppData\\x.py"\nD:\\<user>\\games\\y E:/<user>/z'


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


def test_rotate_faulthandler_log_scrubs_the_rotated_copy_and_keeps_its_mode(tmp_path, fake_user):
    log_path = tmp_path / "faulthandler.log"
    log_path.write_text('Fatal Python error: Aborted\n  File "/home/al/src/a.py"\n  File "/media/al/b.py"\n')
    os.chmod(log_path, 0o644)

    dest = rotate_faulthandler_log(tmp_path)

    assert dest is not None
    assert dest.read_text() == 'Fatal Python error: Aborted\n  File "~/src/a.py"\n  File "/media/<user>/b.py"\n'
    assert stat.S_IMODE(dest.stat().st_mode) == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == [dest.name], "a temp file was left behind"


def test_a_failed_scrub_still_rotates_and_returns_normally(tmp_path, fake_user, monkeypatch):
    log_path = tmp_path / "faulthandler.log"
    log_path.write_text('  File "/home/al/a.py"\n')

    def refuse(_path):
        raise OSError("disk full")

    monkeypatch.setattr(crash_report, "_scrub_file", refuse)
    debug_log.clear()
    dest = rotate_faulthandler_log(tmp_path)

    assert dest is not None and dest in pending_reports(tmp_path)
    assert not log_path.exists()
    assert "could not scrub" in debug_log.get_log_text()


def test_a_second_failed_scrub_reuses_the_same_temp_name(tmp_path, fake_user, monkeypatch):
    real_replace = os.replace

    def replace(src, dst):
        # Only the scrub's replace fails; the rotation's own rename goes through.
        if os.path.basename(src) != crash_report.FAULTHANDLER_LOG_NAME:
            raise OSError("replace refused")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    log_path = tmp_path / "faulthandler.log"
    for stamp in (1_000_000_000, 1_000_000_100):
        log_path.write_text('  File "/home/al/a.py"\n')
        os.utime(log_path, (stamp, stamp))
        assert rotate_faulthandler_log(tmp_path) is not None

    temps = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp"))
    assert temps == [crash_report.SCRUB_TMP_NAME], temps
    assert len(pending_reports(tmp_path)) == 2


def test_rotate_faulthandler_log_leaves_empty_log_alone(tmp_path):
    log_path = tmp_path / "faulthandler.log"
    log_path.write_text("")

    dest = rotate_faulthandler_log(tmp_path)

    assert dest is None
    assert log_path.exists()


def test_rotate_faulthandler_log_no_log_present(tmp_path):
    assert rotate_faulthandler_log(tmp_path) is None
