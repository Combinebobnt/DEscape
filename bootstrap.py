#!/usr/bin/env python3
"""Sets up the venv, installs dependencies, and starts the editor -- with
progress feedback visible even when there's no attached terminal (a
double-clicked LAUNCH_DESCAPE_*).

Invoked by LAUNCH_DESCAPE_LinuxMac.sh / LAUNCH_DESCAPE_Windows.bat, which only
locate a Python 3 interpreter before handing off here. Stdlib only -- this
runs before PyQt5/numpy/etc. are guaranteed to be installed, so it must not
import anything from descape/.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_SENTINEL = VENV_DIR / ".bootstrap_complete"
READY_TIMEOUT = 30.0

_TERMINAL_CANDIDATES = [
    ("x-terminal-emulator", ["-e"]),
    ("gnome-terminal", ["--"]),
    ("konsole", ["-e"]),
    ("xterm", ["-e"]),
]


def venv_python_path(venv_dir: Path, os_name: str = os.name) -> Path:
    # venv layout differs by OS: Scripts/python.exe on Windows, bin/python3
    # elsewhere -- mirrors map_editor.py's own VENV_PYTHON logic.
    return venv_dir / "Scripts" / "python.exe" if os_name == "nt" else venv_dir / "bin" / "python3"


VENV_PYTHON = venv_python_path(VENV_DIR)


class ConsoleReporter:
    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        if log_path is not None:
            log_path.write_text("", encoding="utf-8")

    def set_status(self, text: str) -> None:
        print(text)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    def pump(self) -> None:
        pass

    def close(self) -> None:
        pass


class Splash:
    def __init__(self) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("DEscape")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        # width, height sized for the longest real status string (see
        # install_dependencies()) at two wrapped lines -- a status label
        # with no wraplength silently clips instead of wrapping or growing
        # the window, since resizable() is off.
        width, height = 360, 130
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        tk.Label(self.root, text="DEscape", font=("", 12, "bold")).pack(pady=(16, 4))
        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status_var, wraplength=width - 30, justify="center").pack(padx=15)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.pump()

    def pump(self) -> None:
        self.root.update()

    def close(self) -> None:
        self.root.destroy()


def relaunch_in_terminal() -> bool:
    """Best-effort: re-run this script inside a real terminal so console
    fallback output isn't lost on a double-click launch with neither tkinter
    nor an attached tty. The env flag stops a relaunched copy from trying
    again if the new window somehow still reports no tty."""
    if os.name == "nt" or os.environ.get("DESCAPE_BOOTSTRAP_RELAUNCHED"):
        return False
    env = dict(os.environ, DESCAPE_BOOTSTRAP_RELAUNCHED="1")
    args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    if sys.platform == "darwin":
        # `open -a Terminal` can't forward argv cleanly; osascript is the
        # standard workaround, at the cost of not forwarding a scenario-path
        # arg and not inheriting the relaunch-guard env var into the new
        # window -- acceptable for a fallback-of-a-fallback this narrow.
        cmd = " ".join(shlex.quote(a) for a in args)
        script = f'tell application "Terminal" to do script "{cmd}"'
        try:
            subprocess.Popen(["osascript", "-e", script])
            return True
        except OSError:
            return False
    for exe, flags in _TERMINAL_CANDIDATES:
        if shutil.which(exe) is None:
            continue
        try:
            subprocess.Popen([exe, *flags, *args], env=env)
            return True
        except OSError:
            continue
    return False


def make_reporter(*, stdout=None):
    stdout = stdout if stdout is not None else sys.stdout
    try:
        return Splash()
    except Exception:
        pass
    if stdout.isatty():
        return ConsoleReporter()
    if relaunch_in_terminal():
        sys.exit(0)
    return ConsoleReporter(log_path=ROOT / "bootstrap_log.txt")


def fail(reporter, message: str) -> None:
    reporter.close()
    print()
    print(f"Something went wrong: {message}")
    log_path = getattr(reporter, "log_path", None)
    if log_path is not None:
        print(f"(a log was also written to {log_path})")
    input("Press Enter to close this window...")
    sys.exit(1)


def run_step(reporter, cmd: list[str], status_text: str) -> int:
    reporter.set_status(status_text)
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        reporter.pump()
        time.sleep(0.05)
    return proc.returncode


def find_or_create_venv(reporter) -> None:
    if VENV_PYTHON.exists() and VENV_SENTINEL.exists():
        return
    if VENV_DIR.exists():
        # VENV_PYTHON exists without the sentinel (or the directory is
        # otherwise present) means a previous run was interrupted mid-setup
        # -- `python -m venv` creates the interpreter early, well before the
        # venv is actually usable. Trusting a half-built venv produces
        # confusing pip/import errors instead of just fixing itself, so wipe
        # it and start clean.
        shutil.rmtree(VENV_DIR)
    if run_step(reporter, [sys.executable, "-m", "venv", str(VENV_DIR)], "First-time setup -- this can take a minute...") != 0:
        fail(reporter, "couldn't create the Python virtual environment (.venv).")
    if run_step(reporter, [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], "Upgrading pip...") != 0:
        fail(reporter, "couldn't update pip.")
    VENV_SENTINEL.touch()


def install_dependencies(reporter) -> None:
    cmd = [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "-r", str(ROOT / "requirements.txt")]
    status = "Checking dependencies (only downloads anything the first time, or after an update)..."
    if run_step(reporter, cmd, status) != 0:
        fail(reporter, "couldn't install dependencies -- check your internet connection.")


def wait_until_ready(reporter, ready_file: Path, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists() or proc.poll() is not None:
            return
        reporter.pump()
        time.sleep(0.05)


def launch_app(reporter) -> None:
    reporter.set_status("Starting DEscape...")
    ready_file = Path(tempfile.gettempdir()) / f"descape_ready_{os.getpid()}_{int(time.time() * 1000)}"
    env = dict(os.environ, DESCAPE_READY_FILE=str(ready_file))
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(ROOT / "map_editor.py"), *sys.argv[1:]],
        cwd=str(ROOT),
        env=env,
    )
    wait_until_ready(reporter, ready_file, proc, READY_TIMEOUT)
    reporter.close()
    ready_file.unlink(missing_ok=True)
    rc = proc.wait()
    if rc != 0:
        print(f"The editor closed with an error (exit code {rc}).")
    sys.exit(rc)


def main() -> None:
    reporter = make_reporter()
    try:
        find_or_create_venv(reporter)
        install_dependencies(reporter)
        launch_app(reporter)
    except SystemExit:
        raise
    except Exception as exc:
        # Last-resort guard: an uncaught exception here would otherwise just
        # vanish the window on a double-click launch with no console attached.
        fail(reporter, str(exc))


if __name__ == "__main__":
    main()
