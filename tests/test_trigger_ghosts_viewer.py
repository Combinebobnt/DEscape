"""GH #167's trigger-created unit ghosts, through a real offscreen
ViewerWindow: the triggers fixture's Create Object (and Place Foundations
added through the panel) -> trigger_geometry.CreatedUnit -> MapView's two
ghost items. The ghost's place, its facing seam, the emphasis split, the
resolve memo and its re-resolve hooks, the re-derive paths, the View toggle
and the caps.

The fixture's geometry Create Object has no player, so it draws no ghost
until a test sets one. The shared fixture is never regenerated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from descape import asset_source, render, unit_sprites

import conftest
from test_unit_sprites import CONST as SPRITE_CONST
from test_unit_sprites import FILE_NAME, build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
GEOMETRY_TRIGGER = 3
CREATE = ("effect", 2)
CREATE_TILE = (70, 80)
_ARCHER = 4
_CASTLE = 82
_CREATE_OBJECT = 11
_PLACE_FOUNDATION = 25


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """test_place_preview.py's fixture: one synthetic sprite for SPRITE_CONST
    only, so every other const resolves no sprite. Warms viewer._wall_consts()
    first, for the reason given there."""
    from descape import viewer

    viewer._wall_consts.cache_clear()
    assert viewer._wall_consts(), "the real wall set must be cached before the patch"
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {SPRITE_CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                                "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    yield tmp_path
    asset_source.set_install_path_override(None)


def _window(style: str = "Stepped"):
    window = conftest.shown_window(900, 700)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(GEOMETRY_TRIGGER)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    conftest.close_window(window)


def _set(window, entry_ref, **values) -> None:
    """Writes fields of one effect of the geometry trigger through the panel's funnel."""
    panel = window.trigger_panel
    kind, index = entry_ref
    for name, value in values.items():
        entry = panel._manager().triggers[GEOMETRY_TRIGGER].effects[index]
        spec = next(s for s in panel._specs_for(kind, entry) if s.name == name)
        window.set_entry_field(GEOMETRY_TRIGGER, kind, index, spec, value)


def _new_effect(window, type_id: int, **values) -> tuple[str, int]:
    """New, then the type, as the panel's New button and Type… do; returns its ref."""
    window.entry_structural_edit("new", GEOMETRY_TRIGGER, "effect", -1, type_id)
    ref = ("effect", len(window.trigger_panel._manager().triggers[GEOMETRY_TRIGGER].effects) - 1)
    _set(window, ref, **values)
    return ref


def _items(view) -> dict:
    return {item.data(0): item for item in view.trigger_ghost_items()}


def _strong(view):
    return _items(view)["ghosts_strong"]


def _tile_centre(view, tile):
    return view._tile_polygon(*tile).boundingRect().center()


def _sprite_content(item) -> list:
    return [(x, y, arr.shape, arr.tobytes()) for (_img, x, y), arr in zip(item._images, item._arrays, strict=True)]


def _expected_draws(window, entry) -> list:
    view = window.map_view
    cache = view._sloped_cache()
    draws = render.unit_sprite_draws_at(
        window.scenario, view._iso_proj, view._iso_elevations, None if cache is None else cache.corner_rise,
        entry.player_id, entry.unit, tree_scale=window._layers.tree_scale, hero_glow=window._layers.hero_glow,
        wall_alpha=window._layers.wall_alpha,
    )
    out = []
    for draw, ax, ay in draws:
        rgba = np.ascontiguousarray(draw.rgba)
        out.append((float(ax - draw.hotspot_x), float(ay - draw.hotspot_y), rgba.shape, rgba.tobytes()))
    return out


def _resolver_spy(monkeypatch, view) -> list:
    calls: list = []
    real = view.resolve_trigger_ghost
    monkeypatch.setattr(view, "resolve_trigger_ghost", lambda record, sprites: (calls.append(record), real(record, sprites))[1])
    return calls


# -- what draws ------------------------------------------------------------------


def test_the_unset_player_draws_no_ghost_until_one_is_set() -> None:
    window = _window()
    try:
        view = window.map_view
        assert [r.tile for r in view._trigger_ghost_records] == [CREATE_TILE], "the record is read either way"
        assert view.trigger_ghost_items() == [], "an unset player names no colour slot"
        _set(window, CREATE, source_player=1)
        (item,) = view.trigger_ghost_items()
        assert item.data(0) == "ghosts_strong" and item.scene() is view.scene()
        assert item._marks and not item._images, "no install: the coloured mark"
        assert item.sceneBoundingRect().contains(_tile_centre(view, CREATE_TILE))
    finally:
        _close(window)


def test_a_place_foundation_added_through_the_panel_ghosts_its_const_on_its_location() -> None:
    window = _window()
    try:
        view = window.map_view
        tile = (30, 90)
        ref = _new_effect(window, _PLACE_FOUNDATION, object_list_unit_id=_CASTLE, source_player=2,
                          location_x=tile[0], location_y=tile[1])
        (record,) = [r for r in view._trigger_ghost_records if r.entry_ref == ref]
        assert (record.effect, record.unit_const, record.player, record.tile) == ("place_foundation", _CASTLE, 2, tile)
        strong = _strong(view)
        assert strong.sceneBoundingRect().contains(_tile_centre(view, tile))
        # A 4x4 castle's mark is wider than its one location tile.
        assert strong.sceneBoundingRect().width() > view._tile_polygon(*tile).boundingRect().width() * 2
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_ghost_stands_where_place_unit_puts_the_same_const_on_that_tile(sprite_install, style) -> None:
    """The position oracle: a Place Unit click on the location tile."""
    window = _window(style)
    try:
        view = window.map_view
        assert view._terrain_style == style.lower()
        assert window._sprites_enabled
        _set(window, CREATE, object_list_unit_id=SPRITE_CONST, source_player=1)
        got = _sprite_content(_strong(view))
        assert got, "the sprite must resolve, or this test proves nothing"

        window.mode_combo.setCurrentText("Units")
        window.units_panel.select_owner(1)
        window.units_panel._set_pending_object(SPRITE_CONST)
        window.place_unit_action.setChecked(True)
        window.on_unit_place(_tile_centre(view, CREATE_TILE), Qt.NoModifier)
        entry = view._unit_index.entry_for_key(window._selection[0])
        assert (entry.unit.unit_const, entry.unit.x, entry.unit.y) == (SPRITE_CONST, 70.5, 80.5)
        want = _expected_draws(window, entry)
        assert want, "the placed unit must resolve a sprite too"
        assert got == want
    finally:
        _close(window)


def test_flat_and_show_sprites_off_draw_marks(sprite_install) -> None:
    window = _window("Stepped")
    try:
        view = window.map_view
        _set(window, CREATE, object_list_unit_id=SPRITE_CONST, source_player=1)
        assert _strong(view)._images
        window.show_sprites_action.setChecked(False)
        assert _strong(view)._marks and not _strong(view)._images, "Show sprites off"
        window.show_sprites_action.setChecked(True)
        assert _strong(view)._images
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        assert _strong(view)._marks and not _strong(view)._images, "Flat"
    finally:
        _close(window)


# -- facing ----------------------------------------------------------------------


def test_every_ghost_takes_the_default_facing_until_the_constant_is_on(monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import facet_preview

    window = _window()
    try:
        built = []
        real = facet_preview.ghost_rotation
        monkeypatch.setattr(facet_preview, "ghost_rotation", lambda c, f: (built.append(real(c, f)), built[-1])[1])
        _set(window, CREATE, object_list_unit_id=SPRITE_CONST, source_player=1, facet=4)
        assert built and set(built) == {0.0}, "facet 4 with the constant off"
        built.clear()
        _set(window, CREATE, facet=-1)
        assert built == [0.0]

        monkeypatch.setattr(facet_preview, "GHOST_FACING_MEASURED", True)
        panel = window.trigger_panel
        panel.select_entry(*CREATE)
        built.clear()
        # The panel's own spinbox, so the open preview follows the same edit.
        next(w for spec, _kind, _index, w in panel._rows if spec.name == "facet").setValue(4)
        QApplication.processEvents()
        assert panel._manager().triggers[GEOMETRY_TRIGGER].effects[CREATE[1]].facet == 4
        preview = panel._facet_preview
        assert preview is not None and preview.rotation is not None
        assert built == [pytest.approx(preview.rotation)], "the ghost and the panel's preview agree"
        assert preview.rotation != 0.0
    finally:
        _close(window)


# -- emphasis and the memo ---------------------------------------------------------


def _two_ghost_window():
    """The geometry Create Object plus a second one far from it, both player 1."""
    window = _window()
    _set(window, CREATE, source_player=1)
    other = _new_effect(window, _CREATE_OBJECT, object_list_unit_id=_ARCHER, source_player=1, location_x=20, location_y=30)
    return window, other


def test_one_selected_entry_draws_strong_and_its_sibling_dim_with_no_resolve(monkeypatch) -> None:
    window, other = _two_ghost_window()
    try:
        view = window.map_view
        window.trigger_panel.select_trigger(GEOMETRY_TRIGGER)
        calls = _resolver_spy(monkeypatch, view)
        window.trigger_panel.select_entry(*CREATE)
        items = _items(view)
        strong, dim = items["ghosts_strong"], items["ghosts_dim"]
        assert strong.sceneBoundingRect().contains(_tile_centre(view, CREATE_TILE))
        assert not strong.sceneBoundingRect().contains(_tile_centre(view, (20, 30)))
        assert dim.sceneBoundingRect().contains(_tile_centre(view, (20, 30)))
        assert not dim.sceneBoundingRect().contains(_tile_centre(view, CREATE_TILE))
        assert strong.opacity() == pytest.approx(0.6) and dim.opacity() == pytest.approx(0.3)
        window.trigger_panel.select_entry(*other)
        assert _strong(view).sceneBoundingRect().contains(_tile_centre(view, (20, 30)))
        assert calls == [], "an emphasis change re-splits cached parts"
    finally:
        _close(window)


def test_a_unit_edit_re_derives_without_resolving_but_the_three_look_inputs_re_resolve(monkeypatch) -> None:
    from descape import player_fields, trigger_geometry

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        derived = []
        real_reader = trigger_geometry.created_units_for_trigger
        monkeypatch.setattr(
            trigger_geometry, "created_units_for_trigger", lambda *a, **k: (derived.append(1), real_reader(*a, **k))[1]
        )
        calls = _resolver_spy(monkeypatch, view)

        window.mode_combo.setCurrentText("Units")
        window.units_panel.select_owner(1)
        window.units_panel.pick_object(83)
        window.place_unit_action.setChecked(True)
        window.on_unit_place(_tile_centre(view, (40, 40)), Qt.NoModifier)
        assert derived, "the unit edit did not re-derive the overlay"
        assert calls == [] and view.trigger_ghost_items()

        window.show_sprites_action.setChecked(False)
        assert len(calls) == 1, "Show sprites"
        calls.clear()
        layer = window.layer_actions["small_trees"]
        layer.setChecked(not layer.isChecked())
        assert len(calls) == 1, "View > Layers"
        calls.clear()
        spec = {s.field_id: s for s in player_fields.specs_for(window.scenario)}["color"]
        current = player_fields.current_value(window.scenario, spec, 1)
        window.set_player_field(spec, 1, next(value for value, _label in spec.choices if value != current))
        assert len(calls) == 1, "a player colour"
    finally:
        _close(window)


def test_a_rebuild_depoisons_once_and_a_memo_rebuild_not_at_all(monkeypatch) -> None:
    """batch_adds()'s rule: one library_compat.depoison() per rebuild, never one per resolve."""
    from descape import library_compat

    window, _other = _two_ghost_window()
    try:
        view = window.map_view
        depoisons = []
        real = library_compat.depoison
        monkeypatch.setattr(library_compat, "depoison", lambda: (depoisons.append(1), real())[1])
        calls = _resolver_spy(monkeypatch, view)
        view.refresh_trigger_ghosts()
        assert len(calls) == 2, "both ghosts must resolve, or this test proves nothing"
        assert len(depoisons) == 1
        depoisons.clear()
        view.set_trigger_ghosts_enabled(True)
        assert len(calls) == 2 and depoisons == [], "a rebuild the memo answers"
    finally:
        _close(window)


# -- re-derive and lifecycle --------------------------------------------------------


def test_a_location_edit_moves_the_ghost_and_undo_moves_it_back_and_a_delete_drops_it() -> None:
    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        before = _strong(view).sceneBoundingRect()
        _set(window, CREATE, location_x=75)
        assert _strong(view).sceneBoundingRect().contains(_tile_centre(view, (75, 80)))
        window.mode_combo.setCurrentText("View")
        window.undo()
        assert _strong(view).sceneBoundingRect() == before

        window.mode_combo.setCurrentText("Triggers")
        window.entry_structural_edit("delete", GEOMETRY_TRIGGER, "effect", [CREATE], -1)
        assert view._trigger_ghost_records == [] and view.trigger_ghost_items() == []
        window.undo()
        assert _strong(view).sceneBoundingRect() == before
    finally:
        _close(window)


def test_a_mode_change_and_a_restyle_keep_the_ghost_and_close_and_reload_leave_none() -> None:
    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        records = list(view._trigger_ghost_records)
        window.mode_combo.setCurrentText("View")
        assert view._trigger_ghost_records == records and view.trigger_ghost_items()
        for style in ("Sloped", "Flat", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            assert view._trigger_ghost_records == records, style
            (item,) = view.trigger_ghost_items()
            assert item.scene() is view.scene(), style
            assert item.sceneBoundingRect().contains(_tile_centre(view, CREATE_TILE)), style

        window.edit_history.mark_saved()
        window.close_scenario()
        assert view._trigger_ghost_records == [] and view.trigger_ghost_items() == []
        window.load_scenario(FIXTURE_PATH)
        assert view._trigger_ghost_records == [] and view.trigger_ghost_items() == []
    finally:
        _close(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevation_edit_moves_the_ghost_after_the_flush_and_not_during_a_stroke(style: str) -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window(style)
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        before = _strong(view).sceneBoundingRect()
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")

        # Direct calls hold no MapView stroke: the queued flush moves it.
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*CREATE_TILE, Qt.NoModifier)
        window.on_edit_stroke_end()
        assert _strong(view).sceneBoundingRect() == before, "rebuilt per touched batch, not deferred"
        QApplication.processEvents()
        raised = _strong(view).sceneBoundingRect()
        assert raised != before, "the ghost stayed at the pre-edit ground height"

        # Inside a live stroke the flush holds it to stroke end.
        window.on_edit_stroke_start()
        view._stroke_active = True
        window.on_edit_stroke_tile(*CREATE_TILE, Qt.NoModifier)
        QApplication.processEvents()
        assert _strong(view).sceneBoundingRect() == raised, "moved mid-stroke"
        assert view._trigger_ghosts_stale
        view._end_stroke()
        QApplication.processEvents()
        assert _strong(view).sceneBoundingRect() != raised, "not moved at stroke end"
    finally:
        if window.map_view._stroke_active:
            window.map_view._end_stroke()
        _close(window)


# -- the View toggle ----------------------------------------------------------------


def _copy_geometry_trigger(window) -> list[int]:
    """Duplicates the geometry trigger; returns the pair to select as a two-trigger union."""
    panel = window.trigger_panel
    name = panel._manager().triggers[GEOMETRY_TRIGGER].name
    window.trigger_structural_edit("copy", [GEOMETRY_TRIGGER])
    triggers = window.trigger_edits.manager().triggers
    return [
        next(i for i, t in enumerate(triggers) if t.name == name),
        next(i for i, t in enumerate(triggers) if t.name == f"{name} (copy)"),
    ]


def test_the_toggle_removes_the_items_skips_resolving_and_persists(monkeypatch) -> None:
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        assert window.trigger_ghosts_action.isChecked() and view.trigger_ghost_items()
        calls = _resolver_spy(monkeypatch, view)
        window.trigger_ghosts_action.setChecked(False)
        assert settings.get_trigger_ghosts() is False
        assert view.trigger_ghost_items() == []
        assert view._trigger_ghost_records, "the records are kept"
        _set(window, CREATE, location_x=72)
        assert calls == [] and view.trigger_ghost_items() == [], "resolved while off"
        assert view.trigger_overlay_items(), "the overlay is independent of it"
        window.trigger_ghosts_action.setChecked(True)
        assert _strong(view).sceneBoundingRect().contains(_tile_centre(view, (72, 80)))
        assert len(calls) == 1
        window.trigger_ghosts_action.setChecked(False)
    finally:
        _close(window)

    monkeypatch.setattr(settings, "_trigger_ghosts", None)
    fresh = _window()
    try:
        assert not fresh.trigger_ghosts_action.isChecked()
        _set(fresh, CREATE, source_player=1)
        assert fresh.map_view.trigger_ghost_items() == []
    finally:
        _close(fresh)


def test_turning_it_on_derives_a_union_skipped_while_both_were_off(monkeypatch) -> None:
    from descape import trigger_geometry

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        window.trigger_overlay_action.setChecked(False)
        window.trigger_ghosts_action.setChecked(False)
        derived = []
        real_reader = trigger_geometry.created_units_for_trigger
        monkeypatch.setattr(
            trigger_geometry, "created_units_for_trigger", lambda *a, **k: (derived.append(1), real_reader(*a, **k))[1]
        )
        pair = _copy_geometry_trigger(window)
        # As the TASK-352 test does: one trigger alone is always derived.
        window.trigger_panel.select_trigger(pair[0])
        derived.clear()
        window.trigger_panel.select_triggers(pair)
        assert derived == [] and view._trigger_ghost_records == []
        window.trigger_ghosts_action.setChecked(True)
        assert derived, "turning it on did not derive the union"
        assert sorted(r.trigger_index for r in view._trigger_ghost_records) == sorted(pair)
        (item,) = view.trigger_ghost_items()
        assert item.data(0) == "ghosts_strong", "a union draws unemphasised"
        assert view._trigger_shapes == [], "the overlay stays off"
    finally:
        _close(window)


def _place_unit(window, tile) -> None:
    window.mode_combo.setCurrentText("Units")
    window.units_panel.select_owner(1)
    window.units_panel.pick_object(83)
    window.place_unit_action.setChecked(True)
    window.on_unit_place(_tile_centre(window.map_view, tile), Qt.NoModifier)


def test_a_unit_edit_parses_no_triggers_for_a_ghosts_only_union(monkeypatch) -> None:
    """The work TASK-352 removed: with the overlay off a union draws only
    ghosts, read off trigger fields alone, so a unit edit leaves them be."""
    import descape.viewer as viewer_module

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        window.trigger_overlay_action.setChecked(False)
        pair = _copy_geometry_trigger(window)
        window.trigger_panel.select_triggers(pair)
        records = list(view._trigger_ghost_records)
        assert sorted(r.trigger_index for r in records) == sorted(pair)
        window.mode_combo.setCurrentText("Units")
        owned = window.scenario.unit_manager.units[1]
        before = len(owned)
        parses = []
        real = viewer_module.parse_triggers
        monkeypatch.setattr(viewer_module, "parse_triggers", lambda loaded: (parses.append(1), real(loaded))[1])
        _place_unit(window, (40, 40))
        assert len(owned) == before + 1, "the unit edit must land, or this test proves nothing"
        assert parses == [], "a unit edit re-derived a ghosts-only union"
        assert view._trigger_ghost_records == records and view.trigger_ghost_items()

        # With the overlay on its shapes name units, so the same edit re-derives.
        window.trigger_overlay_action.setChecked(True)
        parses.clear()
        _place_unit(window, (44, 40))
        assert parses
    finally:
        _close(window)


def test_turning_the_ghosts_back_on_says_a_union_cap_again(monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        _set(window, CREATE, source_player=1)
        window.trigger_overlay_action.setChecked(False)
        monkeypatch.setattr(viewer_module, "MAX_UNION_GHOSTS", 1)
        pair = _copy_geometry_trigger(window)
        window.trigger_panel.select_triggers(pair)
        line = "Trigger-created units hidden"
        assert window.status_log.toPlainText().count(line) == 1
        window.trigger_ghosts_action.setChecked(False)
        window.trigger_ghosts_action.setChecked(True)
        assert window.status_log.toPlainText().count(line) == 2, "the same cap key kept it silent"
    finally:
        _close(window)


def test_turning_the_ghosts_back_on_re_derives_one_trigger_and_says_the_sprite_cap_again(sprite_install, monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, object_list_unit_id=SPRITE_CONST, source_player=1)
        _new_effect(window, _CREATE_OBJECT, object_list_unit_id=SPRITE_CONST, source_player=1, location_x=20, location_y=30)
        monkeypatch.setattr(viewer_module, "MAX_SPRITE_GHOSTS", 1)
        window._refresh_trigger_overlay()
        line = "drawn as marks"
        assert window.status_log.toPlainText().count(line) == 1
        window.trigger_ghosts_action.setChecked(False)
        window.trigger_ghosts_action.setChecked(True)
        assert window.status_log.toPlainText().count(line) == 2, "one trigger only rebuilt, never re-derived"
        items = view.trigger_ghost_items()
        assert sum(len(i._marks) for i in items) == 2 and not any(i._images for i in items)
    finally:
        _close(window)


# -- caps ---------------------------------------------------------------------------


def test_past_the_sprite_cap_every_ghost_is_a_mark_said_once(sprite_install, monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, object_list_unit_id=SPRITE_CONST, source_player=1)
        _new_effect(window, _CREATE_OBJECT, object_list_unit_id=SPRITE_CONST, source_player=1, location_x=20, location_y=30)
        items = view.trigger_ghost_items()
        assert sum(len(i._images) for i in items) == 2 and not any(i._marks for i in items)
        assert "drawn as marks" not in window.status_log.toPlainText()

        monkeypatch.setattr(viewer_module, "MAX_SPRITE_GHOSTS", 1)
        window._refresh_trigger_overlay()
        items = view.trigger_ghost_items()
        assert not any(i._images for i in items), "never a mix by count"
        assert sum(len(i._marks) for i in items) == 2
        window._refresh_trigger_overlay()
        log = window.status_log.toPlainText()
        assert log.count("Trigger-created units drawn as marks: 2 is more than the 1 drawn as sprites") == 1
    finally:
        _close(window)


def test_a_union_past_the_ghost_cap_with_the_overlay_off_names_the_ghosts(monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        window.trigger_overlay_action.setChecked(False)
        monkeypatch.setattr(viewer_module, "MAX_UNION_GHOSTS", 1)
        pair = _copy_geometry_trigger(window)
        window.trigger_panel.select_triggers(pair)
        assert view.trigger_ghost_items() == [] and view._trigger_ghost_records == []
        log = window.status_log.toPlainText()
        assert log.splitlines()[-1] == "Trigger-created units hidden: the 2 selected triggers create more than 1 units to draw"
        assert "Trigger overlay hidden" not in log
        assert window._trigger_overlay_ref == (tuple(pair), frozenset()), "kept for turning the overlay on"
        window._refresh_trigger_overlay()
        assert window.status_log.toPlainText().count("Trigger-created units hidden") == 1, "said once per selection"
    finally:
        _close(window)


# -- layering -----------------------------------------------------------------------


def test_the_ghosts_sit_between_the_area_fill_and_the_marks_and_place_units_ghost_leaves_them() -> None:
    window = _window()
    try:
        view = window.map_view
        _set(window, CREATE, source_player=1)
        (item,) = view.trigger_ghost_items()
        assert view.TRIGGER_AREA_Z < item.zValue() < view.TRIGGER_MARK_Z

        window.mode_combo.setCurrentText("Units")
        window.units_panel.select_owner(1)
        window.units_panel.pick_object(83)
        window.place_unit_action.setChecked(True)
        window.on_place_preview(_tile_centre(view, (40, 40)), Qt.NoModifier)
        assert view._unit_ghost_item is not None
        window.pan_action.setChecked(True)
        assert view._unit_ghost_item is None
        assert view.trigger_ghost_items() == [item] and item.scene() is view.scene()
    finally:
        _close(window)
