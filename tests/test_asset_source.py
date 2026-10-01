"""Coverage for descape/asset_source.py's string-table reader and language
config (phase 4d, slice 4): resource_string(), get_language(), and the cache
invalidation set_install_path_override() drives.

_isolated_settings (conftest.py, autouse) already redirects CONFIG_PATH to a
throwaway path for every test here, so get_language() sees no config unless
a test writes one itself.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest
import yaml

from descape import asset_source

import conftest

STRINGS_TEXT = '''\
// a comment line, and a blank line below

5164 "Town Center"
7427 "Anarchy"
3125 "Sent to \\"%s\\":"
'''


def _install_with_strings(tmp_path: Path, lang: str = "en") -> Path:
    install = tmp_path / "install"
    strings_dir = install / "resources" / lang / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text(STRINGS_TEXT, encoding="utf-8")
    return install


@pytest.fixture
def installed(tmp_path):
    install = _install_with_strings(tmp_path)
    asset_source.set_install_path_override(install)
    yield install
    asset_source.set_install_path_override(None)


def test_resource_string_resolves_a_known_key(installed) -> None:
    assert asset_source.resource_string(5164) == "Town Center"
    assert asset_source.resource_string(7427) == "Anarchy"


def test_resource_string_unescapes_embedded_quotes(installed) -> None:
    assert asset_source.resource_string(3125) == 'Sent to "%s":'


def test_resource_string_is_none_for_an_unknown_key(installed) -> None:
    assert asset_source.resource_string(999999) is None


def test_resource_string_is_none_with_no_install_configured() -> None:
    assert asset_source.resource_string(5164) is None


def test_resource_string_is_none_for_a_language_with_no_strings_file(installed) -> None:
    assert asset_source.resource_string(5164, lang="de") is None


def test_resource_string_uses_get_language_by_default(tmp_path) -> None:
    install = tmp_path / "install"
    (install / "resources" / "de" / "strings" / "key-value").mkdir(parents=True)
    (install / "resources" / "de" / "strings" / "key-value" / "key-value-strings-utf8.txt").write_text(
        '5164 "Stadtzentrum"\n', encoding="utf-8"
    )
    asset_source.set_install_path_override(install)
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"language": "de"}))
    try:
        assert asset_source.resource_string(5164) == "Stadtzentrum"
    finally:
        asset_source.set_install_path_override(None)


def test_set_install_path_override_invalidates_the_string_table_cache(tmp_path) -> None:
    """A key resolved under one install must not survive switching to a
    second install (or to none) that does not carry it."""
    install = _install_with_strings(tmp_path)
    asset_source.set_install_path_override(install)
    try:
        assert asset_source.resource_string(5164) == "Town Center"
    finally:
        asset_source.set_install_path_override(None)
    assert asset_source.resource_string(5164) is None


def test_clear_install_caches_forgets_an_install_read_from_another_config(tmp_path, monkeypatch) -> None:
    """The between-tests leak: a lookup cached against the developer's real
    config (install visible) must not survive into the next test's fake one."""
    from descape import object_catalog

    install = tmp_path / "install"
    strings_dir = install / "resources" / "en" / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text('5397 "Fake Oak"\n', encoding="utf-8")
    real_config = tmp_path / "real" / "config.yaml"
    real_config.parent.mkdir()
    real_config.write_text(f"aoe2de_install: {install}\n")
    fake_config = asset_source.CONFIG_PATH
    monkeypatch.setattr(asset_source, "CONFIG_PATH", real_config)
    asset_source.clear_install_caches()
    try:
        assert object_catalog.display_name(349) == "Fake Oak"
        monkeypatch.setattr(asset_source, "CONFIG_PATH", fake_config)
        assert asset_source.get_install_path() == install, "not cached -- the test proves nothing"

        asset_source.clear_install_caches()

        assert asset_source.get_install_path() is None
        assert object_catalog.display_name(349) == "Tree Oak"
    finally:
        asset_source.clear_install_caches()


def test_clear_install_caches_keeps_the_override(tmp_path) -> None:
    asset_source.set_install_path_override(tmp_path)
    try:
        asset_source.clear_install_caches()
        assert asset_source.get_install_path() == tmp_path
    finally:
        asset_source.set_install_path_override(None)


def test_every_test_starts_with_the_install_caches_cleared() -> None:
    """conftest's _isolated_settings must call clear_install_caches() before
    its yield: pytest-qt's post-teardown processEvents() can cache the real
    config's install between tests, after any teardown-side clear has run."""
    source = inspect.getsource(conftest._isolated_settings)
    body = ast.parse(textwrap.dedent(source)).body[0].body
    yield_at = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Yield))
    calls = [
        node.func.attr
        for stmt in body[:yield_at]
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "clear_install_caches" in calls


def test_get_language_defaults_to_en_with_no_config() -> None:
    assert asset_source.get_language() == "en"


def test_get_language_reads_the_config_key() -> None:
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"language": "fr"}))
    assert asset_source.get_language() == "fr"


def test_get_language_tolerates_a_malformed_config() -> None:
    asset_source.CONFIG_PATH.write_text("not: valid: yaml: at: all: [")
    assert asset_source.get_language() == "en"


# --- terrain average colour, from the cached texture array (copy-paste perf plan, Step 2) ---


@pytest.fixture
def fresh_average_cache():
    asset_source.get_terrain_average_color.cache_clear()
    yield
    asset_source.get_terrain_average_color.cache_clear()


def test_the_average_colour_comes_from_the_cached_array_without_opening_the_texture(
    fresh_average_cache, monkeypatch
) -> None:
    """A Copy Region thumbnail must not re-decode a .dds the renderer already
    holds as an array: with the array available, Image.open never runs."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((asset_source.LOADED_TEXTURE_SIZE, asset_source.LOADED_TEXTURE_SIZE, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 40, 120, 200
    arr[: arr.shape[0] // 2, :, 0] = 80  # top half redder: the mean is not one pixel's value
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", lambda _tid: arr)

    def refuse(*_a, **_k):
        raise AssertionError("get_terrain_average_color decoded the texture file")

    monkeypatch.setattr(Image, "open", refuse)
    assert asset_source.get_terrain_average_color(10) == (60, 120, 200)


def test_the_average_colour_is_none_without_a_texture_array(fresh_average_cache, monkeypatch) -> None:
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", lambda _tid: None)
    assert asset_source.get_terrain_average_color(10) is None


def _dds_average(path: Path) -> tuple[int, int, int]:
    """The pre-2026-09-29 definition: the full .dds resized straight to 32x32."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        pixels = np.asarray(img.convert("RGB").resize((32, 32))).reshape(-1, 3)
    return tuple(int(total) // len(pixels) for total in pixels.sum(axis=0, dtype=np.int64))


@pytest.mark.corpus
def test_the_terrain_colour_from_the_array_matches_the_dds_average_on_every_real_terrain(fresh_average_cache) -> None:
    """Needs AOE2DE_INSTALL_PATH. At most one terrain may move, by at most 1 per
    channel (measured: terrain 23, g_wt3.dds, (24,82,126) -> (23,82,126))."""
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    differing = {}
    checked = 0
    for terrain_id in sorted(asset_source._terrain_texture_map()):
        path = asset_source.get_terrain_texture_path(terrain_id)
        if path is None:
            continue
        checked += 1
        old, new = _dds_average(path), asset_source.get_terrain_average_color(terrain_id)
        if old != new:
            differing[terrain_id] = (old, new)
            assert max(abs(a - b) for a, b in zip(old, new, strict=True)) <= 1, f"terrain {terrain_id}: {old} -> {new}"
    assert checked > 100, f"only {checked} terrains have a texture -- is the install complete?"
    assert len(differing) <= 1, f"more than one terrain's average colour moved: {differing}"


# --- terrain texture loads (first-paint-terrain-textures plan) ---


def _shared_file_ids() -> tuple[int, int]:
    """Two terrain ids the committed map points at the same texture file."""
    by_file: dict[str, list[int]] = {}
    for tid, name in sorted(asset_source._terrain_texture_map().items()):
        by_file.setdefault(name, []).append(tid)
    return next(tuple(ids[:2]) for ids in by_file.values() if len(ids) >= 2)


def _texture_install(root: Path, terrain_ids) -> Path:
    """A fake install whose mapped .dds names hold small PNGs (PIL sniffs content), one colour per file."""
    from PIL import Image

    tex_dir = root / asset_source.TERRAIN_TEXTURE_SUBPATH
    tex_dir.mkdir(parents=True, exist_ok=True)
    for tid in terrain_ids:
        name = asset_source._terrain_texture_map()[tid]
        Image.new("RGB", (16, 16), (tid % 251, 40, 200)).save(tex_dir / name, format="PNG")
    return root


@pytest.fixture
def texture_install(tmp_path):
    ids = _shared_file_ids()
    other = next(t for t in sorted(asset_source._terrain_texture_map()) if t not in ids)
    asset_source.set_install_path_override(_texture_install(tmp_path / "install", (*ids, other)))
    yield (*ids, other)
    asset_source.set_install_path_override(None)


@pytest.fixture
def traced():
    from descape import perf_trace

    perf_trace.reset()
    perf_trace.enable(True)
    yield perf_trace
    perf_trace.enable(False)
    perf_trace.reset()


def test_a_texture_load_is_recorded_for_perf_trace_and_a_hit_is_not(texture_install, traced) -> None:
    _a, _b, other = texture_install
    assert asset_source.get_terrain_texture_array(other) is not None
    assert asset_source.get_terrain_texture_array(other) is not None
    traced._drain_textures()
    assert traced._textures_pending.files == 1


@pytest.fixture
def counted_opens(monkeypatch):
    from PIL import Image

    opened: list = []
    real_open = Image.open

    def counting_open(path, *args, **kwargs):
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_open)
    return opened


def test_two_ids_sharing_a_file_share_one_read_only_array_and_one_open(texture_install, counted_opens) -> None:
    a, b, other = texture_install
    arr_a = asset_source.get_terrain_texture_array(a)
    arr_b = asset_source.get_terrain_texture_array(b)
    assert arr_a is arr_b
    assert asset_source.get_terrain_texture_array(other) is not arr_a
    assert len(counted_opens) == 2, counted_opens
    assert arr_a.shape == (asset_source.LOADED_TEXTURE_SIZE, asset_source.LOADED_TEXTURE_SIZE, 3)
    assert not arr_a.flags.writeable
    with pytest.raises(ValueError):
        arr_a[0, 0, 0] = 1


def test_clear_install_caches_drops_the_loaded_textures(texture_install, counted_opens) -> None:
    a, _b, _other = texture_install
    first = asset_source.get_terrain_texture_array(a)
    asset_source.clear_install_caches()
    assert asset_source.get_terrain_texture_array(a) is not first
    assert len(counted_opens) == 2


class _InlineExecutor:
    """Runs a prefetch at submit() and hands back its finished Future."""

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


@pytest.fixture(params=["inline", "threads"])
def pool(request, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    executor = _InlineExecutor() if request.param == "inline" else ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(asset_source, "_prefetch_executor", executor)
    yield request.param
    if request.param == "threads":
        executor.shutdown(wait=True)


@pytest.fixture
def threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(asset_source, "_prefetch_executor", executor)
    yield executor
    executor.shutdown(wait=True)


def _blocking_loader(monkeypatch, release):
    """_load_texture that holds pool threads until `release` is set; returns its call log."""
    import threading

    calls: list[str] = []
    real = asset_source._load_texture

    def loader(path):
        name = threading.current_thread().name
        calls.append(name)
        if name != "MainThread":
            assert release.wait(10), "the test never released the pool load"
        return real(path)

    monkeypatch.setattr(asset_source, "_load_texture", loader)
    return calls


def test_prefetch_then_get_equals_a_serial_load_and_opens_each_file_once(texture_install, counted_opens, pool) -> None:
    import numpy as np

    a, b, other = texture_install
    serial = {tid: asset_source._load_texture(asset_source.get_terrain_texture_path(tid)) for tid in texture_install}
    counted_opens.clear()
    assert asset_source.prefetch_terrain_textures([a, b, other, a]) == 2, "one load per file"
    for tid in texture_install:
        assert np.array_equal(asset_source.get_terrain_texture_array(tid), serial[tid])
    assert sorted(counted_opens) == sorted({asset_source.get_terrain_texture_path(t) for t in texture_install})
    assert asset_source.prefetch_terrain_textures([a, other]) == 0, "cached files are not loaded again"
    assert not asset_source._pending


def test_prefetch_skips_ids_without_a_texture(texture_install, pool) -> None:
    missing = next(t for t in sorted(asset_source._terrain_texture_map()) if t not in texture_install)
    assert asset_source.prefetch_terrain_textures([missing, 10**6]) == 0


def test_a_get_racing_a_pending_load_waits_for_it_instead_of_reloading(texture_install, threads, monkeypatch) -> None:
    import threading

    from descape import perf_trace

    _a, _b, other = texture_install
    release = threading.Event()
    calls = _blocking_loader(monkeypatch, release)
    perf_trace.reset()
    perf_trace.enable(True)
    try:
        assert asset_source.prefetch_terrain_textures([other]) == 1
        timer = threading.Timer(0.1, release.set)
        timer.start()
        arr = asset_source.get_terrain_texture_array(other)
        timer.join()
        perf_trace._drain_textures()
        agg = perf_trace._textures_pending
    finally:
        perf_trace.enable(False)
        perf_trace.reset()
    assert arr is not None and len(calls) == 1 and calls[0] != "MainThread", calls
    assert agg.files == 0 and agg.pool_files == 1 and agg.wait_ms > 0, "the get waited on the pool's load"


def test_a_failing_prefetch_raises_at_the_get(texture_install, pool, monkeypatch) -> None:
    _a, _b, other = texture_install
    real = asset_source._load_texture
    failures = [OSError("truncated texture")]

    def fails_once(path):
        if failures:
            raise failures.pop()
        return real(path)

    monkeypatch.setattr(asset_source, "_load_texture", fails_once)
    asset_source.prefetch_terrain_textures([other])
    with pytest.raises(OSError, match="truncated texture"):
        asset_source.get_terrain_texture_array(other)
    assert asset_source.get_terrain_texture_array(other) is not None, "a failed prefetch is not cached"


def test_clear_during_a_pending_load_discards_it(texture_install, threads, monkeypatch) -> None:
    import threading

    _a, _b, other = texture_install
    release = threading.Event()
    calls = _blocking_loader(monkeypatch, release)
    asset_source.prefetch_terrain_textures([other])
    [(_gen, future)] = asset_source._pending.values()
    asset_source.clear_install_caches()
    release.set()
    stale = future.result(timeout=10)
    arr = asset_source.get_terrain_texture_array(other)
    assert arr is not stale, "a load from before the clear was served"
    assert calls[-1] == "MainThread" and len(calls) == 2, calls
