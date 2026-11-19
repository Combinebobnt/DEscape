"""conftest.py's quick_keep thinning: a combination no keep names goes slow,
a kept one stays quick, and a keep that names a missing param or matches
nothing is a usage error rather than a silently all-slow test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest


class _Item:
    def __init__(self, keeps: list[dict], **params) -> None:
        self.callspec = SimpleNamespace(params=params)
        self.path = Path("test_fake.py")
        self.originalname = "test_fake"
        self.nodeid = f"test_fake.py::test_fake[{params}]"
        self._keeps = [pytest.mark.quick_keep(**k).mark for k in keeps]
        self.added: list[str] = []

    def iter_markers(self, name: str):
        return iter(self._keeps if name == "quick_keep" else [])

    def add_marker(self, mark) -> None:
        self.added.append(mark.name)


def _grid(keeps: list[dict]) -> list[_Item]:
    return [_Item(keeps, style=s, sprites=p) for s in ("stepped", "sloped") for p in (False, True)]


def test_only_the_kept_combinations_stay_quick() -> None:
    items = _grid([{"style": "stepped", "sprites": False}, {"style": "sloped", "sprites": True}])
    conftest._thin_to_quick_keeps(items)
    slow = [(i.callspec.params["style"], i.callspec.params["sprites"]) for i in items if "slow" in i.added]
    assert slow == [("stepped", True), ("sloped", False)]


def test_a_keep_naming_fewer_params_keeps_every_combination_it_matches() -> None:
    items = _grid([{"style": "sloped"}])
    conftest._thin_to_quick_keeps(items)
    assert [i.callspec.params["style"] for i in items if "slow" not in i.added] == ["sloped", "sloped"]


def test_a_keep_naming_a_param_the_test_lacks_is_a_usage_error() -> None:
    with pytest.raises(pytest.UsageError, match="names no param"):
        conftest._thin_to_quick_keeps(_grid([{"styel": "stepped"}]))


def test_a_keep_matching_no_combination_is_a_usage_error() -> None:
    with pytest.raises(pytest.UsageError, match="matches none of its test's params"):
        conftest._thin_to_quick_keeps(_grid([{"style": "Stepped"}]))


def test_one_dead_keep_beside_a_live_one_is_a_usage_error() -> None:
    with pytest.raises(pytest.UsageError, match="keep #2"):
        conftest._thin_to_quick_keeps(_grid([{"style": "stepped"}, {"style": "Sloped"}]))


def test_a_narrowed_collection_may_match_nothing() -> None:
    """A node-id or --lf rerun of thinned combinations collects only those."""
    items = [i for i in _grid([{"style": "sloped"}]) if i.callspec.params["style"] == "stepped"]
    conftest._thin_to_quick_keeps(items, whole=False)
    assert [i.added for i in items] == [["slow"], ["slow"]]


def test_node_ids_and_last_failed_count_as_a_narrowed_collection() -> None:
    def config(args, lf=False):
        return SimpleNamespace(args=args, getoption=lambda name, default=None: lf if name == "lf" else default)

    assert conftest._collected_whole(config(["tests"]))
    assert not conftest._collected_whole(config(["tests/test_x.py::test_y[a]"]))
    assert not conftest._collected_whole(config(["tests"], lf=True))
