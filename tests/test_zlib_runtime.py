"""testkit/zlib_runtime.py: zlib-ng detection, the trailing-data refusal and
the generators' warning, none of which the stock-zlib default tier reaches."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH
from testkit import zlib_runtime


@pytest.mark.parametrize(
    "module, stock",
    [
        (SimpleNamespace(ZLIB_RUNTIME_VERSION="1.2.13"), True),
        (SimpleNamespace(ZLIB_RUNTIME_VERSION="1.3.1.zlib-ng"), False),  # zlib-ng-compat
        (SimpleNamespace(ZLIB_RUNTIME_VERSION="1.2.12", ZLIBNG_VERSION="2.2.5"), False),  # PyPI zlib_ng
    ],
)
def test_is_stock(module, stock: bool) -> None:
    assert zlib_runtime._is_stock(module) is stock


def test_scenario_parts_refuses_bytes_after_the_deflate_stream() -> None:
    raw = BLANK_TEMPLATE_PATH.read_bytes()
    zlib_runtime.scenario_parts(raw)
    with pytest.raises(ValueError, match="3 bytes after the deflate stream"):
        zlib_runtime.scenario_parts(raw + b"abc")


def test_generators_warn_only_off_stock_zlib(monkeypatch, capsys) -> None:
    monkeypatch.setattr(zlib_runtime, "STOCK_ZLIB", True)
    zlib_runtime.warn_if_not_stock("out.aoe2scenario")
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(zlib_runtime, "STOCK_ZLIB", False)
    zlib_runtime.warn_if_not_stock("out.aoe2scenario")
    err = capsys.readouterr().err
    assert err.startswith("warning: zlib runtime ") and "out.aoe2scenario will not match CI's bytes" in err
