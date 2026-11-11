"""TASK-606: conftest's autouse TMPDIR redirect survives a test's own monkeypatch.undo()."""

import os
import tempfile


def test_tmpdir_stays_in_tmp_path_after_monkeypatch_undo(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DESCAPE_UNRELATED_PROBE", "1")
    monkeypatch.undo()
    assert os.environ["TMPDIR"] == str(tmp_path)
    assert tempfile.gettempdir() == str(tmp_path)
