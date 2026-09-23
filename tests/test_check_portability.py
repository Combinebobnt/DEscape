from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tools import check_portability

needs_objdump = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("objdump") is None,
    reason="needs Linux ELF binaries and objdump",
)


def test_versions_compare_numerically() -> None:
    assert check_portability.parse_version("2.2.5") < check_portability.parse_version("2.28")
    assert check_portability.parse_version("3.4.9") < check_portability.parse_version("3.4.21")
    assert check_portability.format_version((2, 2, 5)) == "2.2.5"


def test_non_elf_files_are_not_scanned(tmp_path: Path) -> None:
    (tmp_path / "script.sh").write_text("#!/bin/sh\n")
    (tmp_path / "data.so").write_bytes(b"not an elf")
    assert check_portability.iter_elf_files(tmp_path) == []


@needs_objdump
def test_floor_fails_below_a_real_binarys_requirement(capsys) -> None:
    ls = Path(shutil.which("ls")).resolve()
    assert check_portability.main([str(ls), "--max-glibc", "2.0"]) == 1
    assert "needs GLIBC_" in capsys.readouterr().err


@needs_objdump
def test_floor_passes_above_a_real_binarys_requirement() -> None:
    ls = Path(shutil.which("ls")).resolve()
    assert check_portability.main([str(ls), "--max-glibc", "99.0"]) == 0


@needs_objdump
def test_host_libs_requires_a_qxcb_plugin(tmp_path: Path) -> None:
    errors = check_portability.check_host_libs(tmp_path)
    assert errors == [f"expected exactly one libqxcb.so under {tmp_path}, found 0"]


@needs_objdump
def test_host_libs_flags_unbundled_deps_and_bundled_libstdcxx(tmp_path: Path) -> None:
    pytest.importorskip("PyInstaller")
    ls = Path(shutil.which("ls")).resolve()
    unbundled = [n for n in check_portability.needed_entries(ls) if not check_portability.pyinstaller_excludes(n)]
    if not unbundled:
        pytest.skip(f"{ls} needs only libraries PyInstaller leaves to the host")
    plugin_dir = tmp_path / "_internal" / "PyQt5"
    plugin_dir.mkdir(parents=True)
    shutil.copy(ls, plugin_dir / check_portability.QXCB_NAME)
    (tmp_path / "_internal" / "libstdc++.so.6").write_bytes(b"stub")

    errors = check_portability.check_host_libs(tmp_path)
    assert any(unbundled[0] in e and "missing from the build image" in e for e in errors)
    assert any(e.startswith("libstdc++.so.6 is bundled") for e in errors)


def test_skip_env_short_circuits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SKIP_PORTABILITY_CHECK", "1")
    assert check_portability.main([str(tmp_path / "missing"), "--max-glibc", "2.0"]) == 0
