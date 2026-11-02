"""conftest.require_reference_corpus: a check whose counts were measured on
examples/ runs there and skips when --scenario-dir points at other
scenarios, so the full suite run on a contributor's own files stays green."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import conftest


def _config(scenario_dir: str | None) -> SimpleNamespace:
    return SimpleNamespace(getoption=lambda name: scenario_dir if name == "--scenario-dir" else None)


def test_examples_runs_by_default_and_when_named() -> None:
    conftest.require_reference_corpus(_config(None))
    conftest.require_reference_corpus(_config(str(conftest.ROOT / "examples")))


def test_another_scenario_dir_skips(tmp_path) -> None:
    with pytest.raises(pytest.skip.Exception, match="measured on examples/"):
        conftest.require_reference_corpus(_config(str(tmp_path)))
