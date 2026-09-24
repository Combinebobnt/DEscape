#!/usr/bin/env python3
"""Builds the native composite kernel, descape/_composite_native.pyx, in place.
The app runs without it (descape/composite_backend.py falls back to numpy),
just slower.

    python tools/build_native.py            # build if missing or stale
    python tools/build_native.py --force    # always rebuild
    python tools/build_native.py --check    # exit 0 current, 1 stale, 2 stale and a build already failed
    python tools/build_native.py --probe    # exit 0 if a C compiler and Python headers look present

Needs cython and setuptools (packaging/requirements-native.txt) for a build,
nothing for --check/--probe. The top level is stdlib-only because bootstrap.py
runs those two in the venv before either is installed.

A build writes a stamp keyed on the .pyx, the compile flags and the
interpreter's extension suffix, so a changed source or a new Python version
reads as stale. A failed build writes a failed stamp on the same key, which
only bootstrap.py consults (--check exit 2), so a launch never retries a
build that is bound to fail again.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYX = ROOT / "descape" / "_composite_native.pyx"
STAMP = ROOT / "descape" / "_composite_native.stamp"
FAILED_STAMP = ROOT / "descape" / "_composite_native.failed"
MODULE = "descape._composite_native"

# No FMA contraction anywhere, so the float paths round exactly as numpy's do.
UNIX_ARGS = ("-O3", "-ffp-contract=off")
MSVC_ARGS = ("/O2", "/fp:precise")

EXIT_CURRENT, EXIT_STALE, EXIT_FAILED_BEFORE = 0, 1, 2


def ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def built_module_path() -> Path:
    return PYX.with_name(PYX.stem + ext_suffix())


def build_key() -> str:
    digest = hashlib.sha256(PYX.read_bytes())
    digest.update(repr((UNIX_ARGS, MSVC_ARGS, ext_suffix())).encode())
    return digest.hexdigest()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def check() -> int:
    key = build_key()
    if built_module_path().exists() and _read(STAMP) == key:
        return EXIT_CURRENT
    return EXIT_FAILED_BEFORE if _read(FAILED_STAMP) == key else EXIT_STALE


def _install_hint() -> str:
    if sys.platform == "darwin":
        return "install the Xcode Command Line Tools (xcode-select --install)"
    if sys.platform == "win32":
        return "install Microsoft C++ Build Tools (the 'Desktop development with C++' workload)"
    return "install a C compiler and the Python headers (Debian/Ubuntu: build-essential python3-dev)"


def _has_compiler() -> bool:
    if sys.platform == "darwin":
        # Never run cc blind here: without the Command Line Tools it pops the
        # system install dialog.
        try:
            return subprocess.run(["xcode-select", "-p"], capture_output=True, check=False).returncode == 0
        except OSError:
            return False
    if sys.platform == "win32":
        program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.exists():
            return False
        cmd = [str(vswhere), "-latest", "-products", "*", "-requires",
               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]
        try:
            return bool(subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip())
        except OSError:
            return False
    cc = (sysconfig.get_config_var("CC") or "").split()
    candidates = ([cc[0]] if cc else []) + ["cc", "gcc", "clang"]
    return any(shutil.which(c) for c in candidates)


def probe() -> str | None:
    """None if a build looks possible, else why not and what to install."""
    if not _has_compiler():
        return f"no C compiler found; {_install_hint()}"
    include = sysconfig.get_paths().get("include")
    if not include or not (Path(include) / "Python.h").exists():
        return f"no Python headers (Python.h) found; {_install_hint()}"
    return None


def build(force: bool = False) -> int:
    key = build_key()
    if not force and check() == EXIT_CURRENT:
        print(f"build_native: {built_module_path().name} is current")
        return 0
    # setuptools reports a compile error as SystemExit.
    try:
        _run_setuptools(force)
    except (Exception, SystemExit) as exc:
        FAILED_STAMP.write_text(key + "\n", encoding="utf-8")
        STAMP.unlink(missing_ok=True)
        print(f"build_native: build failed: {exc}", file=sys.stderr)
        return 1
    if not built_module_path().exists():
        FAILED_STAMP.write_text(key + "\n", encoding="utf-8")
        print(f"build_native: build ran but {built_module_path().name} is missing", file=sys.stderr)
        return 1
    STAMP.write_text(key + "\n", encoding="utf-8")
    FAILED_STAMP.unlink(missing_ok=True)
    print(f"build_native: built {built_module_path().relative_to(ROOT)}")
    return 0


def _run_setuptools(force: bool) -> None:
    from Cython.Build import cythonize
    from setuptools import Extension, setup
    from setuptools.command.build_ext import build_ext

    class _BuildExt(build_ext):
        def build_extensions(self):
            args = MSVC_ARGS if self.compiler.compiler_type == "msvc" else UNIX_ARGS
            for ext in self.extensions:
                ext.extra_compile_args = list(args)
            super().build_extensions()

    ext = Extension(MODULE, [PYX.relative_to(ROOT).as_posix()])
    old_cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        setup(
            name="descape-native",
            ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}, force=force, quiet=True),
            cmdclass={"build_ext": _BuildExt},
            script_args=["build_ext", "--inplace"] + (["--force"] if force else []),
        )
    finally:
        os.chdir(old_cwd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="rebuild even if the stamp is current")
    mode.add_argument("--check", action="store_true", help="report staleness via the exit code, build nothing")
    mode.add_argument("--probe", action="store_true", help="check for a C compiler and Python headers")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if args.probe:
        reason = probe()
        print(reason or "build_native: compiler and Python headers found")
        return 1 if reason else 0
    return build(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
