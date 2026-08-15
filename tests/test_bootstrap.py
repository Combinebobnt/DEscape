"""bootstrap.py's own logic, isolated from the real venv/pip/tkinter it
drives: interpreter/venv path resolution, feedback-mode selection (splash /
console / relaunch-in-terminal), the ready-file wait loop, and that the
pip-upgrade step only runs on first-time venv creation, not every launch.
Never spawns a real subprocess or a real Tk window -- subprocess.Popen and
Splash are always faked or monkeypatched."""

from __future__ import annotations

import time
from pathlib import Path

import bootstrap


class _FakeStdout:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class _FakeReporter:
    def __init__(self) -> None:
        self.pumps = 0
        self.statuses: list[str] = []

    def set_status(self, text: str) -> None:
        self.statuses.append(text)

    def pump(self) -> None:
        self.pumps += 1

    def close(self) -> None:
        pass


class _FakeProc:
    """poll() returns None until the `exits_after`-th call, then a fake exit code."""

    def __init__(self, exits_after: int | None = None) -> None:
        self._exits_after = exits_after
        self._calls = 0

    def poll(self):
        self._calls += 1
        if self._exits_after is not None and self._calls >= self._exits_after:
            return 0
        return None


def _boom():
    raise RuntimeError("no display")


def test_venv_python_path_windows() -> None:
    assert bootstrap.venv_python_path(Path("/x/.venv"), "nt") == Path("/x/.venv/Scripts/python.exe")


def test_venv_python_path_posix() -> None:
    assert bootstrap.venv_python_path(Path("/x/.venv"), "posix") == Path("/x/.venv/bin/python3")


def test_make_reporter_uses_splash_when_available(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(bootstrap, "Splash", lambda: sentinel)
    assert bootstrap.make_reporter() is sentinel


def test_make_reporter_falls_back_to_console_when_tty(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "Splash", _boom)
    relaunch_calls = []
    monkeypatch.setattr(bootstrap, "relaunch_in_terminal", lambda: relaunch_calls.append(True) or False)
    reporter = bootstrap.make_reporter(stdout=_FakeStdout(True))
    assert isinstance(reporter, bootstrap.ConsoleReporter)
    assert reporter.log_path is None
    assert relaunch_calls == []


def test_make_reporter_relaunches_when_no_tty_and_relaunch_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "Splash", _boom)
    monkeypatch.setattr(bootstrap, "relaunch_in_terminal", lambda: True)
    try:
        bootstrap.make_reporter(stdout=_FakeStdout(False))
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit(0) after a successful relaunch")


def test_make_reporter_logs_when_no_tty_and_relaunch_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bootstrap, "Splash", _boom)
    monkeypatch.setattr(bootstrap, "relaunch_in_terminal", lambda: False)
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    reporter = bootstrap.make_reporter(stdout=_FakeStdout(False))
    assert isinstance(reporter, bootstrap.ConsoleReporter)
    assert reporter.log_path == tmp_path / "bootstrap_log.txt"
    assert reporter.log_path.exists()


def test_wait_until_ready_returns_immediately_when_file_already_exists(tmp_path) -> None:
    ready_file = tmp_path / "ready"
    ready_file.touch()
    reporter = _FakeReporter()
    bootstrap.wait_until_ready(reporter, ready_file, _FakeProc(), timeout=5.0)
    assert reporter.pumps == 0


def test_wait_until_ready_returns_when_child_exits_first(tmp_path) -> None:
    ready_file = tmp_path / "never-appears"
    reporter = _FakeReporter()
    bootstrap.wait_until_ready(reporter, ready_file, _FakeProc(exits_after=1), timeout=5.0)
    assert not ready_file.exists()


def test_wait_until_ready_times_out(tmp_path) -> None:
    ready_file = tmp_path / "never-appears"
    reporter = _FakeReporter()
    start = time.monotonic()
    bootstrap.wait_until_ready(reporter, ready_file, _FakeProc(), timeout=0.15)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1
    assert reporter.pumps > 0


def _make_recorder(calls: list[list[str]]):
    class _Recorder:
        def __init__(self, cmd, **kwargs) -> None:
            calls.append(list(cmd))
            self.returncode = 0

        def poll(self):
            return self.returncode

    return _Recorder


def test_find_or_create_venv_runs_pip_upgrade_when_missing(monkeypatch, tmp_path) -> None:
    fake_venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(bootstrap, "VENV_DIR", fake_venv_dir)
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", bootstrap.venv_python_path(fake_venv_dir))
    monkeypatch.setattr(bootstrap, "VENV_SENTINEL", fake_venv_dir / ".bootstrap_complete")
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        # The real `python -m venv` call is mocked out, so nothing actually
        # creates fake_venv_dir on disk -- do it here so the later
        # VENV_SENTINEL.touch() has a directory to land in, same as a real
        # (successful) venv creation would leave behind.
        fake_venv_dir.mkdir(parents=True, exist_ok=True)
        return _make_recorder(calls)(cmd, **kwargs)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)

    bootstrap.find_or_create_venv(_FakeReporter())

    assert any("venv" in c for c in calls[0])
    assert calls[1][-4:] == ["install", "--quiet", "--upgrade", "pip"]
    assert bootstrap.VENV_SENTINEL.exists()


def test_find_or_create_venv_skips_pip_upgrade_when_venv_exists(monkeypatch, tmp_path) -> None:
    fake_venv_dir = tmp_path / ".venv"
    fake_venv_python = bootstrap.venv_python_path(fake_venv_dir)
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.touch()
    fake_sentinel = fake_venv_dir / ".bootstrap_complete"
    fake_sentinel.touch()
    monkeypatch.setattr(bootstrap, "VENV_DIR", fake_venv_dir)
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", fake_venv_python)
    monkeypatch.setattr(bootstrap, "VENV_SENTINEL", fake_sentinel)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _make_recorder(calls))

    bootstrap.find_or_create_venv(_FakeReporter())

    assert calls == []


def test_find_or_create_venv_wipes_and_recreates_when_incomplete(monkeypatch, tmp_path) -> None:
    """Regression test, ported from daubED after the same bug was found and
    fixed there: force-quitting mid-setup leaves VENV_PYTHON on disk (python
    -m venv creates the interpreter early) but the venv otherwise broken.
    Without the sentinel, find_or_create_venv would trust that half-built
    venv forever and every later launch would need a manual `.venv` delete
    to recover."""
    fake_venv_dir = tmp_path / ".venv"
    fake_venv_python = bootstrap.venv_python_path(fake_venv_dir)
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.touch()
    canary = fake_venv_dir / "leftover_from_interrupted_run"
    canary.touch()
    # Deliberately no sentinel -- this is the "interrupted mid-setup" state.
    monkeypatch.setattr(bootstrap, "VENV_DIR", fake_venv_dir)
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", fake_venv_python)
    monkeypatch.setattr(bootstrap, "VENV_SENTINEL", fake_venv_dir / ".bootstrap_complete")
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        fake_venv_dir.mkdir(parents=True, exist_ok=True)
        return _make_recorder(calls)(cmd, **kwargs)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)

    bootstrap.find_or_create_venv(_FakeReporter())

    assert not canary.exists()
    assert any("venv" in c for c in calls[0])
    assert bootstrap.VENV_SENTINEL.exists()


def test_install_dependencies_uses_requirements_txt(monkeypatch, tmp_path) -> None:
    fake_venv_python = tmp_path / "python3"
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", fake_venv_python)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _make_recorder(calls))

    bootstrap.install_dependencies(_FakeReporter())

    assert calls[0][0] == str(fake_venv_python)
    assert calls[0][-1].endswith("requirements.txt")
