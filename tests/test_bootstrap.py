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
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _make_recorder(calls))

    bootstrap.find_or_create_venv(_FakeReporter())

    assert any("venv" in c for c in calls[0])
    assert calls[1][-4:] == ["install", "--quiet", "--upgrade", "pip"]


def test_find_or_create_venv_skips_pip_upgrade_when_venv_exists(monkeypatch, tmp_path) -> None:
    fake_venv_dir = tmp_path / ".venv"
    fake_venv_python = bootstrap.venv_python_path(fake_venv_dir)
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.touch()
    monkeypatch.setattr(bootstrap, "VENV_DIR", fake_venv_dir)
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", fake_venv_python)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _make_recorder(calls))

    bootstrap.find_or_create_venv(_FakeReporter())

    assert calls == []


def test_install_dependencies_uses_requirements_txt(monkeypatch, tmp_path) -> None:
    fake_venv_python = tmp_path / "python3"
    monkeypatch.setattr(bootstrap, "VENV_PYTHON", fake_venv_python)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _make_recorder(calls))

    bootstrap.install_dependencies(_FakeReporter())

    assert calls[0][0] == str(fake_venv_python)
    assert calls[0][-1].endswith("requirements.txt")
