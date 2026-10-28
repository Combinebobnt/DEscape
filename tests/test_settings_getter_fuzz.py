"""TASK-497 Step 2: every settings getter survives any YAML value under any portable key.

A settings import writes another machine's raw config entries and leaves
per-value checks to the getters (plan D5), as a hand-edited config.yaml
always has. So for every portable key: a config holding only that key, every
memo reset, then every settings getter called (each per-id one over its ids)
plus asset_source.get_language(). None may raise, and each result must have
its annotated type exactly: `type(v) is int`, so a bool or a float never
passes as an int. Each key's failures are reported together.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import types
import typing
from pathlib import Path

import pytest
import yaml

from descape import asset_source, settings, settings_file, themes, tool_options, view_layers
from testkit.settings_isolation import MEMOIZED_GLOBALS

ROOT = Path(__file__).resolve().parent.parent

# The plan's six, plus three a YAML file can also hold: .inf, .nan and an int past a float's range.
HOSTILE: tuple = ("x", 7, True, None, [1, "a"], {"a": {"b": 1}}, math.inf, math.nan, 10**400)

# text_box_lines has no registry; these are its callers' shapes ("messages.<field id>", "trigger.<name>").
_TEXT_BOX_IDS = ("messages.instructions", "trigger.message")
_KEYBIND_IDS = tuple(action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS)
_OVERLAY_COLOR_IDS = tuple(cid for cid, _label, _default in settings.OVERLAY_COLORS)
_OVERLAY_OPACITY_IDS = tuple(gid for gid, _label, _default in settings.OVERLAY_OPACITIES)
_TRIGGER_STATUS_IDS = tuple(sid for sid, _label, _default in settings.TRIGGER_STATUS_COLORS)
_TOOL_OPTION_NAMES = tuple(spec.name for spec in tool_options.OPTIONS)

# Dict-valued portable keys -> the ids a hostile value goes under. keybinds also takes the
# renamed ids, whose value the rename copies before the collision reconciler runs.
DICT_KEY_IDS: dict[str, tuple] = {
    "keybinds": _KEYBIND_IDS + tuple(settings._RENAMED_KEYBINDS),
    "wheel_bindings": settings.WHEEL_SLOTS,
    "tool_options": _TOOL_OPTION_NAMES,
    "overlay_colors": _OVERLAY_COLOR_IDS,
    "view_layers": tuple(spec.layer_id for spec in view_layers.LAYERS),
    "overlay_opacity": _OVERLAY_OPACITY_IDS,
    "trigger_status_colors": _TRIGGER_STATUS_IDS,
    "theme_colors": themes.ROLE_IDS,
    "text_box_lines": _TEXT_BOX_IDS,
}

# One-argument getters -> the ids they are called with.
PER_ID_GETTERS: dict[str, tuple] = {
    "get_keybind": _KEYBIND_IDS,
    "get_default_keybind": _KEYBIND_IDS,
    "get_action_label": _KEYBIND_IDS,
    "get_overlay_color": _OVERLAY_COLOR_IDS,
    "get_default_overlay_color": _OVERLAY_COLOR_IDS,
    "get_overlay_color_label": _OVERLAY_COLOR_IDS,
    "get_overlay_opacity": _OVERLAY_OPACITY_IDS,
    "get_default_overlay_opacity": _OVERLAY_OPACITY_IDS,
    "get_trigger_status_color": _TRIGGER_STATUS_IDS,
    "get_default_trigger_status_color": _TRIGGER_STATUS_IDS,
    "get_trigger_status_color_label": _TRIGGER_STATUS_IDS,
    "get_text_box_lines": _TEXT_BOX_IDS,
    "get_tool_option_default": _TOOL_OPTION_NAMES,
}
# Takes a scenario path, not an id, and its key (data_mod_by_scenario) is local.
EXCLUDED_GETTERS = frozenset({"get_data_mod_for"})


def _getters() -> tuple[list[str], list[str]]:
    zero, one = [], []
    for name, fn in inspect.getmembers(settings, inspect.isfunction):
        if not name.startswith("get_") or fn.__module__ != settings.__name__ or name in EXCLUDED_GETTERS:
            continue
        required = [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty]
        (zero if not required else one).append(name)
    return zero, one


def _option_hint(spec: tool_options.ToolOption):
    if spec.kind == tool_options.BOOL:
        return bool
    if spec.kind == tool_options.INT:
        return int
    if spec.kind == tool_options.CHOICE:
        (choice_type,) = {type(choice) for choice in spec.choices}
        return choice_type
    if spec.kind == tool_options.ID:
        return spec.id_type | None if spec.nullable else spec.id_type
    (member_type,) = {type(member) for member in spec.members}
    return list[member_type]


def _mismatch(value, hint) -> str | None:
    """Why `value` is not exactly of type `hint`, or None. Containers are checked entry by entry."""
    if hint is object:
        return None
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if any(_mismatch(value, option) is None for option in typing.get_args(hint)):
            return None
        return f"{value!r:.80} is a {type(value).__name__}, not {hint}"
    expected = type(None) if hint is None else (origin or hint)
    if type(value) is not expected:
        return f"{value!r:.80} is a {type(value).__name__}, not {hint}"
    args = typing.get_args(hint)
    if expected is dict and args:
        items = [x for k, v in value.items() for x in ((k, args[0]), (v, args[1]))]
    elif expected in (list, frozenset) and args:
        items = [(member, args[0]) for member in value]
    elif expected is tuple and args:
        items = list(zip(value, args, strict=False)) if len(value) == len(args) else [(value, None)]
    elif dataclasses.is_dataclass(expected):
        field_hints = typing.get_type_hints(expected)
        items = [(getattr(value, f.name), field_hints[f.name]) for f in dataclasses.fields(value)]
    else:
        items = []
    for item, item_hint in items:
        if item_hint is None:
            return f"{value!r:.80} has the wrong length for {hint}"
        why = _mismatch(item, item_hint)
        if why is not None:
            return f"in {value!r:.60}: {why}"
    return None


def _example_float(key: str) -> float:
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    value = example.get(key)
    return float(value) if type(value) is int else 3.0


def _configs(key: str) -> list[dict]:
    configs = [{key: value} for value in HOSTILE]
    if key in DICT_KEY_IDS:
        for value in HOSTILE:
            entries = {entry_id: copy.deepcopy(value) for entry_id in DICT_KEY_IDS[key]}
            configs.append({key: {**entries, "zz_unknown_id": copy.deepcopy(value), 7: copy.deepcopy(value)}})
    else:
        configs += [{key: 3.0}, {key: _example_float(key)}]
    return configs


def _sweep(config: dict, config_path: Path, monkeypatch) -> list[str]:
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = yaml.safe_load(asset_source.read_config_text(config_path))
    # Every getter parses the same file; one parse, a fresh copy per read, keeps the sweep fast.
    monkeypatch.setattr(settings, "_load_config", lambda: copy.deepcopy(loaded))
    for name in MEMOIZED_GLOBALS:
        setattr(settings, name, None)
    zero, _one = _getters()
    calls = [(f"settings.{name}", ()) for name in zero]
    calls += [(f"settings.{name}", (entry_id,)) for name, ids in PER_ID_GETTERS.items() for entry_id in ids]
    calls.append(("asset_source.get_language", ()))
    failures = []
    for qualified, args in calls:
        module, name = qualified.split(".")
        fn = getattr(settings if module == "settings" else asset_source, name)
        if name == "get_tool_option_default":
            hint = _option_hint(tool_options.BY_NAME[args[0]])
        else:
            hint = typing.get_type_hints(fn)["return"]
        try:
            why = _mismatch(fn(*args), hint)
        except Exception as e:  # noqa: BLE001 -- every raise is a finding, reported below
            why = f"raised {type(e).__name__}: {e}"
        if why is not None:
            failures.append(f"{name}{args!r:.40}: {why}")
    return failures


def test_every_one_argument_getter_has_ids_to_call() -> None:
    _zero, one = _getters()
    assert sorted(one) == sorted(PER_ID_GETTERS)


def test_every_dict_key_is_portable() -> None:
    assert set(DICT_KEY_IDS) <= set(settings_file.PORTABLE_KEYS)


@pytest.mark.parametrize("key", list(settings_file.PORTABLE_KEYS))
def test_no_getter_raises_or_returns_a_wrong_type_on_a_hostile_value(key, tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    assert settings.CONFIG_PATH == config_path == asset_source.CONFIG_PATH
    failures = [
        f"{config!r:.100} -> {failure}" for config in _configs(key) for failure in _sweep(config, config_path, monkeypatch)
    ]
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:60])
