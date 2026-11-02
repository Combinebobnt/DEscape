"""Config and autosave-index readers never see a partially written YAML file."""

from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from test_collection_env import ROOT, sanitized_env

from descape import autosave, settings

import conftest


@pytest.fixture(params=["config", "index"])
def persistence(request, tmp_path, monkeypatch):
    if request.param == "config":
        # Deliberately not asset_source.CONFIG_PATH: the explicit path must win.
        path = tmp_path / "settings" / "config.yaml"
        monkeypatch.setattr(settings, "CONFIG_PATH", path)
        return path, settings._save_config, settings._load_config
    return autosave.index_path(), autosave._save_index, autosave._load_index


def test_concurrent_reader_sees_only_whole_old_or_new_file(persistence, monkeypatch):
    path, save, load = persistence
    old = {"generation": "old", "values": ["old"] * 64}
    new = {"generation": "new", "values": ["new"] * 96}
    save(old)
    old_bytes = path.read_bytes()
    new_bytes = yaml.safe_dump(new, default_flow_style=False, sort_keys=False).replace("\n", os.linesep).encode("utf-8")
    partial_written, reader_done = Event(), Event()

    @contextmanager
    def paused_writer(handle):
        def write(data):
            middle = len(data) // 2
            written = handle.write(data[:middle])
            handle.flush()
            partial_written.set()
            assert reader_done.wait(10), "reader never acknowledged the partial-write window"
            return written + handle.write(data[middle:])

        with handle:
            yield SimpleNamespace(write=write)

    real_open = io.open

    def open_with_pause(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        return paused_writer(handle) if "w" in mode else handle

    # Both os.fdopen and Path.write_text delegate to io.open.
    monkeypatch.setattr(io, "open", open_with_pause)
    observed_bytes, observed_docs = [old_bytes], [load()]
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(save, new)
        try:
            assert partial_written.wait(10), "writer never reached the partial-write window"
            observed_bytes.append(path.read_bytes())
            observed_docs.append(load())
        finally:
            reader_done.set()
        writer.result(timeout=10)
    observed_bytes.append(path.read_bytes())
    observed_docs.append(load())

    assert all(data in (old_bytes, new_bytes) for data in observed_bytes), "reader saw a truncated or partial file"
    assert observed_bytes == [old_bytes, old_bytes, new_bytes]
    assert observed_docs == [old, old, new]
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failure_keeps_old_bytes_and_removes_temp(persistence, monkeypatch, failure):
    path, save, load = persistence
    old = {"generation": "old"}
    save(old)
    before = path.read_bytes()
    replaced = []

    @contextmanager
    def failing_writer(fd, *args, **kwargs):
        def write(data):
            handle.write(data[: len(data) // 2])
            handle.flush()
            raise OSError("injected write failure")

        with real_fdopen(fd, *args, **kwargs) as handle:
            yield SimpleNamespace(write=write)

    def failing_replace(src, dst):
        replaced.append(Path(src))
        assert Path(src).parent == path.parent
        assert Path(dst) == path
        assert yaml.safe_load(Path(src).read_bytes()) == {"generation": "new"}
        assert path.read_bytes() == before
        raise OSError("injected replace failure")

    real_fdopen = os.fdopen
    if failure == "write":
        monkeypatch.setattr(os, "fdopen", failing_writer)
    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match=f"injected {failure} failure"):
        save({"generation": "new"})

    assert bool(replaced) == (failure == "replace")
    assert path.read_bytes() == before
    assert load() == old
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits")
def test_existing_mode_survives_a_more_restrictive_umask(persistence):
    path, save, _load = persistence
    save({"generation": "old"})
    path.chmod(0o640)
    previous = os.umask(0o077)
    try:
        save({"generation": "new"})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits")
@pytest.mark.parametrize("umask", [0o022, 0o002, 0o077])
def test_new_file_mode_is_normal_0666_less_umask(persistence, umask):
    path, save, _load = persistence
    assert not path.parent.exists()
    previous = os.umask(umask)
    try:
        save({"generation": "new"})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


def test_config_and_index_creation_never_change_the_process_umask(persistence, monkeypatch):
    _path, save, load = persistence
    mask = Mock(side_effect=AssertionError("a writer must not change the process umask"))
    monkeypatch.setattr(os, "umask", mask)
    save({"generation": "new"})
    assert load() == {"generation": "new"}
    mask.assert_not_called()


@pytest.mark.parametrize("data", ["export: yes\n", b"export: yes\n"])
def test_export_creation_never_changes_the_process_umask(tmp_path, monkeypatch, data):
    from descape.yaml_file_io import atomic_write

    mask = Mock(side_effect=AssertionError("a writer must not change the process umask"))
    monkeypatch.setattr(os, "umask", mask)
    path = tmp_path / "export.yaml"
    atomic_write(path, data)
    expected = data if isinstance(data, bytes) else data.replace("\n", os.linesep).encode()
    assert path.read_bytes() == expected
    mask.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits")
def test_private_config_stays_private_even_while_the_temporary_is_written(persistence, monkeypatch):
    path, save, _load = persistence
    save({"generation": "old"})
    path.chmod(0o600)
    real_fdopen = os.fdopen
    modes = []

    @contextmanager
    def inspect_writer(fd, *args, **kwargs):
        modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        with real_fdopen(fd, *args, **kwargs) as handle:
            yield handle

    monkeypatch.setattr(os, "fdopen", inspect_writer)
    previous = os.umask(0o022)
    try:
        save({"generation": "new"})
    finally:
        os.umask(previous)
    assert modes == [0o600]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_read_only_temporary_cleanup_preserves_the_publication_error(tmp_path, monkeypatch, cleanup_fails):
    from descape.yaml_file_io import atomic_write

    path = tmp_path / "config.yaml"
    path.write_bytes(b"generation: old\n")
    real_unlink = Path.unlink
    attempts = []

    def refuse_replace(*_args):
        raise OSError("original publication failure")

    def windows_unlink(temporary, *args, **kwargs):
        attempts.append(temporary)
        if cleanup_fails or not temporary.stat().st_mode & stat.S_IWRITE:
            raise PermissionError("read-only temporary cannot be removed")
        return real_unlink(temporary, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", refuse_replace)
        patch.setattr(Path, "unlink", windows_unlink)
        with pytest.raises(OSError, match="original publication failure"):
            atomic_write(path, "generation: new\n", mode=0o444)

    assert len(attempts) == 2
    assert path.read_bytes() == b"generation: old\n"
    if cleanup_fails:
        attempts[-1].unlink()  # Remove the owned residue once the injected cleanup failure ends.
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="native Windows read-only file semantics")
def test_windows_read_only_target_failure_leaves_no_temporary(tmp_path):
    from descape import asset_source

    path = tmp_path / "config.yaml"
    path.write_bytes(b"generation: old\n")
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            asset_source.write_config_file(path, {"generation": "new"})
        assert path.read_bytes() == b"generation: old\n"
        assert list(tmp_path.iterdir()) == [path]
    finally:
        path.chmod(0o600)


@conftest.requires_symlinks
@pytest.mark.parametrize("existing", [False, True])
def test_symlink_survives_and_resolved_target_parents_are_created(persistence, tmp_path, existing):
    path, save, load = persistence
    path.parent.mkdir(parents=True)
    target = tmp_path / "dotfiles" / "deep" / path.name
    if existing:
        target.parent.mkdir(parents=True)
        target.write_text("generation: old\n", encoding="utf-8")
    link = Path("..") / "dotfiles" / "deep" / path.name
    path.symlink_to(link)

    save({"generation": "new"})

    assert path.is_symlink()
    assert path.readlink() == link
    assert path.resolve() == target
    assert yaml.safe_load(target.read_bytes()) == {"generation": "new"}
    assert load() == {"generation": "new"}
    assert list(path.parent.iterdir()) == [path]
    assert list(target.parent.iterdir()) == [target]


@conftest.requires_symlinks
@pytest.mark.skipif(os.name == "nt", reason="Windows path length depends on long-path host policy")
def test_symlink_to_a_near_component_limit_target_remains_writable(persistence, tmp_path):
    path, save, load = persistence
    path.parent.mkdir(parents=True)
    target = tmp_path / ("a" * 250)
    target.write_text("generation: old\n", encoding="utf-8")
    path.symlink_to(target)
    save({"generation": "new"})
    assert path.is_symlink()
    assert path.resolve() == target
    assert load() == {"generation": "new"}


@pytest.mark.parametrize("first", ["asset_source", "content_roots", "scenario_io", "yaml_file_io"])
def test_atomic_writer_has_no_import_order_cycle(tmp_path, first):
    path = tmp_path / "config.yaml"
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib, sys
from pathlib import Path
importlib.import_module("descape." + sys.argv[1])
from descape import asset_source
asset_source.write_config_file(Path(sys.argv[2]), {"first": sys.argv[1]})
""", first, str(path)],
        cwd=ROOT,
        env=sanitized_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert yaml.safe_load(path.read_bytes()) == {"first": first}
