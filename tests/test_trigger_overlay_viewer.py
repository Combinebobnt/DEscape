"""GH #41's trigger map overlay, driven through a real offscreen ViewerWindow:
panel selection -> trigger_geometry -> MapView scene items, the re-derive
paths (field edit, structural edit, undo), the lifecycle hooks, and pixel
captures at 1:1 in all three terrain styles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
GEOMETRY_TRIGGER = 3
CONDITION_AREA = ("condition", 1)
PATROL = ("effect", 1)
CREATE = ("effect", 2)
WHOLE_MAP = ("effect", 4)
OUTLINE_RGB = (255, 0, 255)


def _gen():
    return conftest.load_verify_module("gen_trigger_fixture")


def _window(tmp_path=None, monkeypatch=None, *, style: str = "Stepped", outline: str | None = None):
    if outline is not None:
        from descape import settings

        (tmp_path / "config.yaml").write_text(f"overlay_colors:\n  trigger_area_outline: '{outline}'\n")
        monkeypatch.setattr(settings, "_overlay_colors", None)
    window = conftest.shown_window(900, 700)
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None
    window.terrain_style_combo.setCurrentText(style)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_trigger(GEOMETRY_TRIGGER)
    return window


def _shape_refs(view):
    return [(s.entry_ref, s.shape, s.coords) for s in view._trigger_shapes]


def _items_by_role(view):
    return {item.data(0): item for item in view.trigger_overlay_items()}


def _labels(view):
    return [i.text() for i in view.trigger_overlay_items() if hasattr(i, "text")]


# -- selection -> overlay ------------------------------------------------------


def test_selecting_the_trigger_draws_every_shape_unemphasised() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        # Referenced units are pinned by the unit tests at the end of this file.
        assert [r for r in _shape_refs(view) if r[1] != "unit"] == [
            (CONDITION_AREA, "area", gen.GEOMETRY_CONDITION_AREA),
            (PATROL, "area", gen.GEOMETRY_PATROL_AREA),
            (PATROL, "location", gen.GEOMETRY_PATROL_LOCATION),
            (CREATE, "location", gen.GEOMETRY_CREATE_LOCATION),
            (WHOLE_MAP, "area", gen.GEOMETRY_WHOLE_MAP_AREA),
            (("effect", gen.TASK_OBJECT_EFFECT), "location", gen.HOUSE_TILE),
        ]
        assert view._trigger_emphasis is None
        assert view.trigger_overlay_items()
    finally:
        conftest.close_window(window)


def test_selecting_an_entry_emphasises_it_and_a_coordinate_free_trigger_draws_nothing() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        assert view._trigger_emphasis == frozenset({PATROL})
        window.trigger_panel.select_trigger(0)
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


def _ring_polygons(path) -> set:
    """Each subpath of an overlay path as a hashable vertex tuple."""
    return {tuple((round(p.x(), 3), round(p.y(), 3)) for p in polygon) for polygon in path.toSubpathPolygons()}


def _area_ring(view, area) -> tuple:
    """The one ring an inclusive trigger area draws, in _ring_polygons() form."""
    (ring,) = _ring_polygons(view._rect_ring_path(view._trigger_area_rect(area)))
    return ring


def _mark_centre(view, tile):
    return view._location_mark(tile).boundingRect().center()


def test_several_selected_entries_all_draw_strong() -> None:
    """GH #168: every selected condition and effect draws at full strength,
    not only the panel's current one. The unselected ones stay dimmed."""
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entries([PATROL, CREATE])
        assert window.trigger_panel.current_entry_ref() == PATROL
        roles = _items_by_role(view)
        create = _mark_centre(view, gen.GEOMETRY_CREATE_LOCATION)
        assert roles["marks_strong"].path().contains(create), "the second selected effect's location is dimmed"
        assert not roles["marks_dim"].path().contains(create)
        strong, dim = _ring_polygons(roles["outline_strong"].path()), _ring_polygons(roles["outline_dim"].path())
        assert _area_ring(view, gen.GEOMETRY_PATROL_AREA) in strong
        assert _area_ring(view, gen.GEOMETRY_CONDITION_AREA) in dim
        assert _area_ring(view, gen.GEOMETRY_CONDITION_AREA) not in strong
        assert view._trigger_emphasis == frozenset({PATROL, CREATE})

        # Back to one entry: the other selected one dims again.
        window.trigger_panel.select_entry(*PATROL)
        assert view._trigger_emphasis == frozenset({PATROL})
        assert _items_by_role(view)["marks_dim"].path().contains(create)
    finally:
        conftest.close_window(window)


def test_the_panel_hook_fires_on_user_selection_not_on_populate() -> None:
    window = _window()
    try:
        panel = window.trigger_panel
        calls = []
        panel.on_focus_changed = lambda: calls.append(1)
        panel.select_trigger(1)
        assert calls, "a trigger change did not reach the hook"
        calls.clear()
        panel.select_entry("effect", 0)
        assert calls, "an entry change did not reach the hook"
        calls.clear()
        panel._populating = True
        try:
            panel._sync_entry_form()
        finally:
            panel._populating = False
        assert calls == [], "a programmatic populate fired the hook"
    finally:
        conftest.close_window(window)


def test_several_triggers_draw_their_union_unemphasised() -> None:
    """GH #168: several selected triggers draw every shape of each, all at
    full strength. The second trigger is a Duplicate of the first, so every
    entry ref exists in both: a run must still pair within one trigger."""
    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        single = _shape_refs(view)
        single_labels = _labels(view)
        single_roles = set(_items_by_role(view))
        assert single and single_labels, "the fixture trigger should carry shapes and a labelled run"
        name = window.trigger_panel._manager().triggers[GEOMETRY_TRIGGER].name

        window.trigger_structural_edit("copy", [GEOMETRY_TRIGGER])
        triggers = window.trigger_edits.manager().triggers
        source = next(i for i, t in enumerate(triggers) if t.name == name)
        copy = next(i for i, t in enumerate(triggers) if t.name == f"{name} (copy)")
        panel.select_triggers([source, copy])
        assert panel.selected_trigger_indices() == [source, copy]

        by_trigger: dict[int, list] = {}
        for shape in view._trigger_shapes:
            by_trigger.setdefault(shape.trigger_index, []).append((shape.entry_ref, shape.shape, shape.coords))
        assert by_trigger == {source: single, copy: single}
        assert view._trigger_emphasis is None
        # The roles one unemphasised trigger draws (marks_dim is its superseded location), none dimmed.
        roles = set(_items_by_role(view))
        assert roles == single_roles and not roles & {"outline_dim", "fill_dim", "units_dim"}, sorted(roles)
        # One label per run-bearing entry per trigger: a cross-trigger pairing would add more.
        assert sorted(_labels(view)) == sorted(single_labels * 2)

        panel.select_trigger(source)
        assert _shape_refs(view) == single and _labels(view) == single_labels
    finally:
        conftest.close_window(window)


def test_a_union_one_shape_past_the_cap_draws_nothing_and_says_why(monkeypatch) -> None:
    """The union is re-derived on every unit edit, so past MAX_OVERLAY_SHAPES
    it draws nothing (never a partial union) and stops being re-derived. One
    trigger alone is never capped."""
    import descape.viewer as viewer_module

    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        name = panel._manager().triggers[GEOMETRY_TRIGGER].name
        window.trigger_structural_edit("copy", [GEOMETRY_TRIGGER])
        triggers = window.trigger_edits.manager().triggers
        pair = [
            next(i for i, t in enumerate(triggers) if t.name == name),
            next(i for i, t in enumerate(triggers) if t.name == f"{name} (copy)"),
        ]
        panel.select_triggers(pair)
        total = len(view._trigger_shapes)
        assert total > 0 and total < viewer_module.MAX_OVERLAY_SHAPES

        monkeypatch.setattr(viewer_module, "MAX_OVERLAY_SHAPES", total)
        panel.select_trigger(pair[0])
        panel.select_triggers(pair)
        assert len(view._trigger_shapes) == total, "a union exactly at the cap still draws"
        assert "Trigger overlay hidden" not in window.status_log.toPlainText()

        monkeypatch.setattr(viewer_module, "MAX_OVERLAY_SHAPES", total - 1)
        panel.select_trigger(pair[0])
        panel.select_triggers(pair)
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
        assert window._trigger_overlay_ref is None, "a unit edit would re-derive the whole union"
        said = window.status_log.toPlainText().splitlines()[-1]
        assert said == f"Trigger overlay hidden: the 2 selected triggers have more than {total - 1} areas, locations and units to draw"

        monkeypatch.setattr(viewer_module, "MAX_OVERLAY_SHAPES", 0)
        panel.select_trigger(pair[0])
        assert len(view._trigger_shapes) == total // 2, "one trigger is never capped"
    finally:
        conftest.close_window(window)


def test_with_the_overlay_off_a_union_is_not_derived_or_reported_until_it_is_on(monkeypatch) -> None:
    """TASK-352: View > Show Trigger Overlay off skips the union and the cap
    message on every selection change; turning it on derives and reports."""
    import descape.viewer as viewer_module
    from descape import trigger_geometry

    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        name = panel._manager().triggers[GEOMETRY_TRIGGER].name
        window.trigger_structural_edit("copy", [GEOMETRY_TRIGGER])
        triggers = window.trigger_edits.manager().triggers
        pair = [
            next(i for i, t in enumerate(triggers) if t.name == name),
            next(i for i, t in enumerate(triggers) if t.name == f"{name} (copy)"),
        ]
        panel.select_triggers(pair)
        total = len(view._trigger_shapes)
        assert total > 0
        calls = []
        real = trigger_geometry.shapes_for_trigger
        monkeypatch.setattr(trigger_geometry, "shapes_for_trigger", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

        window.trigger_overlay_action.setChecked(False)
        monkeypatch.setattr(viewer_module, "MAX_OVERLAY_SHAPES", total - 1)
        panel.select_trigger(pair[0])
        calls.clear()
        panel.select_triggers(pair)
        assert calls == [], "the union was derived for a hidden overlay"
        assert "Trigger overlay hidden" not in window.status_log.toPlainText()
        assert window._trigger_overlay_ref == (tuple(pair), frozenset()), "the ref is kept for turning it on"

        window.trigger_overlay_action.setChecked(True)
        said = window.status_log.toPlainText().splitlines()[-1]
        assert said == f"Trigger overlay hidden: the 2 selected triggers have more than {total - 1} areas, locations and units to draw"
        assert view._trigger_shapes == [] and window._trigger_overlay_ref is None

        window.trigger_overlay_action.setChecked(False)
        monkeypatch.setattr(viewer_module, "MAX_OVERLAY_SHAPES", total)
        panel.select_trigger(pair[0])
        panel.select_triggers(pair)
        window.trigger_overlay_action.setChecked(True)
        assert len(view._trigger_shapes) == total, "turning it on draws the union"
        assert view.trigger_overlay_items() and all(i.isVisible() for i in view.trigger_overlay_items())
    finally:
        conftest.close_window(window)


def test_runs_never_pair_entries_across_triggers() -> None:
    """Two triggers' ("effect", 1) are different entries: an area in one and a
    location in the other is no run. The same pair inside one trigger is."""
    from descape import trigger_geometry
    from descape.trigger_geometry import TriggerShape

    gen = _gen()
    window = _window()
    try:
        view = window.map_view

        def shapes(location_trigger: int) -> list:
            return [
                TriggerShape(0, "effect", 1, "area", trigger_geometry.AREA_FIELDS, gen.GEOMETRY_PATROL_AREA),
                TriggerShape(
                    location_trigger, "effect", 1, "location", trigger_geometry.LOCATION_FIELDS, gen.GEOMETRY_PATROL_LOCATION
                ),
            ]

        view.show_trigger_overlay(shapes(location_trigger=0), None)
        assert len(_labels(view)) == 1
        # The location mark plus the run segment.
        assert len(_items_by_role(view)["marks_strong"].path().toSubpathPolygons()) == 2

        view.show_trigger_overlay(shapes(location_trigger=1), None)
        assert _labels(view) == []
        assert len(_items_by_role(view)["marks_strong"].path().toSubpathPolygons()) == 1, "a run crossed two triggers"
        assert "outline_strong" in _items_by_role(view)
    finally:
        conftest.close_window(window)


def test_adding_a_second_entry_to_the_selection_disarms_a_map_pick() -> None:
    """A pick targets one entry's field, so it ends once the form shows two
    entries, even when the picked entry stays the current row."""
    gen = _gen()
    window = _window()
    try:
        panel = window.trigger_panel
        picked = ("effect", gen.TASK_OBJECT_EFFECT)
        panel.select_entry(*picked)
        target = (GEOMETRY_TRIGGER, *picked, "selected_object_ids", True)
        window.arm_unit_picker(target)
        # The compare, not a plain disarm: a refocus on the same entry keeps the pick.
        window._on_trigger_focus_changed()
        assert window._unit_picker == target

        panel.select_entries([picked, ("effect", gen.PATROL_IDS_EFFECT)])
        assert panel.current_entry_ref() == picked
        assert window._unit_picker is None and not window.map_view.unit_picker_active()
    finally:
        conftest.close_window(window)


def test_the_run_label_reads_the_rulers_own_number() -> None:
    from descape import ruler

    gen = _gen()
    window = _window()
    try:
        window.trigger_panel.select_entry(*PATROL)
        expected = ruler.format_measurement(ruler.measure((32, 43), gen.GEOMETRY_PATROL_LOCATION))
        assert _labels(window.map_view) == [expected]
    finally:
        conftest.close_window(window)


def test_an_unparseable_trigger_section_clears_the_overlay(monkeypatch) -> None:
    import descape.viewer as viewer_module

    window = _window()
    try:
        assert window.map_view._trigger_shapes
        # A scoped context: a bare undo() would also revert the autouse settings isolation.
        with monkeypatch.context() as patch:
            patch.setattr(viewer_module, "parse_triggers", lambda loaded: None)
            window._refresh_trigger_overlay()
        assert window.map_view._trigger_shapes == [] and window.map_view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


# -- re-derive after edits -------------------------------------------------------


def _area_x2_spec(window):
    from descape import library_compat, trigger_fields

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    definition = vocabulary.effects[19]  # patrol
    specs = trigger_fields.field_specs(definition, vocabulary.effect_presentation, "effect_type")
    return next(s for s in specs if s.attribute == "area_x2")


def _patrol_area(view):
    return next(s.coords for s in view._trigger_shapes if s.entry_ref == PATROL and s.shape == "area")


def _outline_path_rect(view):
    return _items_by_role(view)["outline_strong"].path().boundingRect()


def _outline_path(view):
    from PyQt5.QtGui import QPainterPath

    return QPainterPath(_items_by_role(view)["outline_strong"].path())


def test_a_field_edit_moves_the_outline_and_undo_moves_it_back_even_in_view_mode() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        before_rect = _outline_path_rect(view)
        window.set_entry_fields(GEOMETRY_TRIGGER, [PATROL], _area_x2_spec(window), 44)
        assert _patrol_area(view)[2] == 44
        assert _outline_path_rect(view) != before_rect, "the outline did not move"
        window.mode_combo.setCurrentText("View")
        window.undo()
        assert _patrol_area(view)[2] == 34
        assert _outline_path_rect(view) == before_rect
    finally:
        conftest.close_window(window)


def test_a_structural_delete_re_derives_the_entry_indices() -> None:
    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        panel.select_entry(*PATROL)
        patrol_rect = _outline_path_rect(view)
        # As in the GUI: the effect above is selected to delete it.
        panel.select_entry("effect", 0)
        before_delete = _overlay_snapshot(view)
        window.entry_structural_edit("delete", GEOMETRY_TRIGGER, "effect", [("effect", 0)], -1)
        # The change_variable effect is gone: the patrol is effect 0 now.
        assert any(s.entry_ref == ("effect", 0) and s.shape == "area" for s in view._trigger_shapes)
        # The whole-map area moved down to effect 3 with it.
        assert any(s.entry_ref == ("effect", 3) and s.coords == _gen().GEOMETRY_WHOLE_MAP_AREA for s in view._trigger_shapes)
        assert not any(s.entry_ref == ("effect", 4) and s.shape == "area" for s in view._trigger_shapes)
        # GH #41: the delete lands on the patrol, and the emphasised ring is the patrol's.
        assert panel.current_entry_ref() == ("effect", 0)
        assert _outline_path_rect(view) == patrol_rect
        assert _overlay_snapshot(view) != before_delete
        window.undo()
        assert any(s.entry_ref == WHOLE_MAP for s in view._trigger_shapes)
        # Back to the pre-delete state: the restored effect above selected, the same items drawn.
        assert panel.current_entry_ref() == ("effect", 0)
        assert _overlay_snapshot(view) == before_delete
    finally:
        conftest.close_window(window)


def _overlay_snapshot(view) -> dict:
    """Each overlay item's role and scene bounds: what the user sees drawn."""
    return {item.data(0): item.boundingRect() for item in view.trigger_overlay_items()}


# -- lifecycle ---------------------------------------------------------------------


def test_the_overlay_survives_a_mode_change_and_a_restyle() -> None:
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        shapes = list(view._trigger_shapes)
        window.mode_combo.setCurrentText("View")
        assert view._trigger_shapes == shapes and view.trigger_overlay_items()
        for style in ("Sloped", "Flat", "Stepped"):
            window.terrain_style_combo.setCurrentText(style)
            assert view._trigger_shapes == shapes, style
            assert view._trigger_emphasis == frozenset({PATROL}), style
            assert all(item.scene() is view.scene() for item in view.trigger_overlay_items()), style
    finally:
        conftest.close_window(window)


def test_close_and_reload_leave_no_overlay() -> None:
    window = _window()
    try:
        view = window.map_view
        window.close_scenario()
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
        window.load_scenario(FIXTURE_PATH)
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_an_elevation_edit_moves_the_outline(style: str) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    gen = _gen()
    window = _window(style=style)
    try:
        view = window.map_view
        assert view._terrain_style == style.lower(), "the restyle did not reach the view"
        window.trigger_panel.select_entry(*CONDITION_AREA)
        before = _outline_path(view)
        x1, y1, _x2, _y2 = gen.GEOMETRY_CONDITION_AREA
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(x1, y1, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert _outline_path(view) != before, "the outline stayed at the pre-edit ground height"
    finally:
        conftest.close_window(window)


def test_the_view_toggle_hides_without_dropping_and_persists(monkeypatch) -> None:
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        assert window.trigger_overlay_action.isChecked()
        window.trigger_overlay_action.setChecked(False)
        assert settings.get_trigger_overlay() is False
        assert view.trigger_overlay_items() and not any(i.isVisible() for i in view.trigger_overlay_items())
        window.trigger_panel.select_entry(*PATROL)
        assert not any(i.isVisible() for i in view.trigger_overlay_items()), "a rebuild re-showed hidden items"
        window.trigger_overlay_action.setChecked(True)
        assert all(i.isVisible() for i in view.trigger_overlay_items())
        window.trigger_overlay_action.setChecked(False)
    finally:
        conftest.close_window(window)

    # GH #41's restart: drop the memo so the fresh window has to read config.yaml back.
    monkeypatch.setattr(settings, "_trigger_overlay", None)
    fresh = _window()
    try:
        assert not fresh.trigger_overlay_action.isChecked()
        items = fresh.map_view.trigger_overlay_items()
        assert items and not any(i.isVisible() for i in items)
    finally:
        conftest.close_window(fresh)


def test_the_z_band_sits_between_the_selection_and_the_region_and_above_the_ruler_label() -> None:
    from descape.map_view import MapView

    assert MapView.UNIT_SELECT_OUTLINE_Z < MapView.TRIGGER_AREA_Z < MapView.TRIGGER_MARK_Z < MapView.REGION_SELECT_Z
    assert MapView.RULER_LABEL_Z < MapView.TRIGGER_LABEL_Z < MapView.ANALYSIS_MARKER_Z < MapView.MIRROR_OVERLAY_Z
    window = _window()
    try:
        window.trigger_panel.select_entry(*PATROL)
        z = {role: item.zValue() for role, item in _items_by_role(window.map_view).items()}
        assert z["outline_strong"] == z["fill_strong"] == MapView.TRIGGER_AREA_Z
        assert z["marks_strong"] == MapView.TRIGGER_MARK_Z
        assert z["label"] == MapView.TRIGGER_LABEL_Z
    finally:
        conftest.close_window(window)


def test_a_whole_map_outline_is_o_perimeter_not_o_area() -> None:
    """The assertion that stops a refactor reintroducing one polygon per tile:
    a 120x120 area is 14400 tiles but only 480 border tiles. Doubling the side
    must roughly double the element count (O(perimeter)), not quadruple it."""
    for style in ("Stepped", "Sloped", "Flat"):
        window = _window(style=style)
        try:
            view = window.map_view
            window.trigger_panel.select_entry(*WHOLE_MAP)
            whole = _items_by_role(view)["outline_strong"].path().elementCount()
            half = view._rect_ring_path((0, 0, 60, 60)).elementCount()
            assert whole == view._rect_ring_path((0, 0, 120, 120)).elementCount(), style
            assert 1.8 <= whole / half <= 2.2, (style, whole, half)
            assert whole < 4 * 120 * 120, (style, whole)
        finally:
            conftest.close_window(window)


# -- pixels ----------------------------------------------------------------------


def _colour_hits(array, rgb, tolerance):
    return (np.abs(array.astype(np.int16) - np.array(rgb, dtype=np.int16)) <= tolerance).all(axis=2)


def _area_capture(view, area):
    from PyQt5.QtCore import QRectF
    from PyQt5.QtWidgets import QApplication

    x1, y1, x2, y2 = area
    bounds = None
    for tx, ty in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        rect = view._tile_polygon(tx, ty).boundingRect()
        bounds = rect if bounds is None else bounds.united(rect)
    rect = QRectF(bounds.adjusted(-8, -8, 8, 8).toAlignedRect())
    QApplication.processEvents()
    return conftest.scene_rect_to_array(view.scene(), rect)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_area_outline_paints_in_every_style_and_emphasis_is_stronger(tmp_path, monkeypatch, style) -> None:
    gen = _gen()
    window = _window(tmp_path, monkeypatch, style=style, outline="#ff00ff")
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*CONDITION_AREA)
        strong = _area_capture(view, gen.GEOMETRY_CONDITION_AREA)
        dimmed = _area_capture(view, gen.GEOMETRY_PATROL_AREA)
        strong_hits = int(_colour_hits(strong, OUTLINE_RGB, 12).sum())
        dim_hits = int(_colour_hits(dimmed, OUTLINE_RGB, 12).sum())
        assert strong_hits >= 40, (style, strong_hits)
        assert dim_hits == 0, (style, dim_hits)
        # Dimmed, not gone: the same hue at reduced alpha still marks the area.
        assert _colour_hits(dimmed, OUTLINE_RGB, 150).sum() > 0, style
    finally:
        conftest.close_window(window)


# -- referenced units (trigger unit references, slice C) ---------------------------


def _unit_shapes(view):
    return [(s.entry_ref, s.fields[0], s.coords) for s in view._trigger_shapes if s.shape == "unit"]


def test_referenced_units_are_outlined_and_the_task_runs_to_its_location_object() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        house = (40, 62, 41, 63)
        assert (("condition", gen.DESTROY_CONDITION), "unit_object", house) in _unit_shapes(view)
        assert (("effect", gen.TASK_OBJECT_EFFECT), "location_object_reference", house) in _unit_shapes(view)
        roles = _items_by_role(view)
        assert "units_strong" in roles and "fill_units_strong" not in roles
        window.trigger_panel.select_entry("effect", gen.TASK_OBJECT_EFFECT)
        roles = _items_by_role(view)
        strong, dim = roles["marks_strong"].path(), roles["marks_dim"].path()
        centre = view._location_mark(gen.HOUSE_TILE).boundingRect().center()
        # The stored location is an authoring copy of the house's tile: dimmed, never strong.
        assert dim.contains(centre) and not strong.contains(centre)
        assert _labels(view), "the task's run, archers' centroid to the house, is labelled"
    finally:
        conftest.close_window(window)


def test_moving_a_referenced_unit_moves_its_outline() -> None:
    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        before = _items_by_role(view)["units_strong"].path().boundingRect()
        unit = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == gen.REF_HOUSE)
        unit.x, unit.y = 81.0, 21.0
        window._after_unit_mutation()
        moved = [c for e, f, c in _unit_shapes(view) if f == "unit_object"]
        assert moved == [(80, 20, 81, 21)]
        after = _items_by_role(view)["units_strong"].path().boundingRect()
        assert after != before
    finally:
        conftest.close_window(window)


def test_the_unit_outline_is_on_its_own_colour_row(tmp_path, monkeypatch) -> None:
    from descape import settings

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  trigger_unit_ref: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    window = _window()
    try:
        pen = _items_by_role(window.map_view)["units_strong"].pen()
        assert pen.color().name() == "#ff00ff"
    finally:
        conftest.close_window(window)


def test_a_colour_change_reinks_a_live_outline() -> None:
    """GH #41: Settings > Appearance recolours a drawn overlay straight away,
    through the same call its colour swatch makes."""
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        assert _items_by_role(view)["outline_strong"].pen().color().name() != "#ff00ff"
        settings.set_overlay_color("trigger_area_outline", "#ff00ff")
        view.apply_overlay_colors()
        # Re-fetched: a colour change rebuilds the items rather than re-inking them.
        assert _items_by_role(view)["outline_strong"].pen().color().name() == "#ff00ff"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_unit_outline_paints_on_the_house_footprint_in_every_style(tmp_path, monkeypatch, style) -> None:
    from descape import settings

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  trigger_unit_ref: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    window = _window(style=style)
    try:
        view = window.map_view
        # Only the house's own entry, so no area outline shares the frame.
        window.trigger_panel.select_entry("condition", _gen().DESTROY_CONDITION)
        array = _area_capture(view, (40, 62, 41, 63))
        assert _colour_hits(array, OUTLINE_RGB, 40).sum() > 20
        # Countable: the ring's extent is exactly the footprint tiles' own polygons.
        footprint = None
        for tx, ty in ((40, 62), (41, 62), (41, 63), (40, 63)):
            rect = view._tile_polygon(tx, ty).boundingRect()
            footprint = rect if footprint is None else footprint.united(rect)
        ring = _items_by_role(view)["units_strong"].path().boundingRect()
        for got, want in zip(ring.getCoords(), footprint.getCoords(), strict=True):
            assert abs(got - want) < 0.5, (style, ring, footprint)
    finally:
        conftest.close_window(window)


# -- the owed in-app passes, offscreen (TASK-087) ------------------------------------


def _area_tiles(area):
    x1, y1, x2, y2 = area
    return [(tx, ty) for tx in range(x1, x2 + 1) for ty in range(y1, y2 + 1)]


def _tiles_bounds(view, tiles):
    bounds = None
    for tile in tiles:
        rect = view._tile_polygon(*tile).boundingRect()
        bounds = rect if bounds is None else bounds.united(rect)
    return bounds


def _ring_around(view, role: str, area):
    """The one subpath of `role`'s ring whose bounds match `area`'s tiles, or None."""
    from PyQt5.QtGui import QPolygonF

    want = _tiles_bounds(view, _area_tiles(area))
    for polygon in _items_by_role(view)[role].path().toSubpathPolygons():
        got = QPolygonF(polygon).boundingRect()
        if all(abs(g - w) < 0.5 for g, w in zip(got.getCoords(), want.getCoords(), strict=True)):
            return QPolygonF(polygon)
    return None


def _tile_centre(view, tile):
    return view._tile_polygon(*tile).boundingRect().center()


_STYLE_CASES = [("Flat", False), ("Flat", True), ("Stepped", False), ("Sloped", False)]


@pytest.mark.parametrize(("style", "iso"), _STYLE_CASES)
def test_the_patrol_is_drawn_over_exactly_its_tiles_with_its_mark_run_and_label_in_every_style(style: str, iso: bool) -> None:
    """GH #41 steps 2 and 5: the Patrol (area 30-34 x 40-46, location 60,50) selected in
    Triggers, then View mode and the style (Flat, Flat + Isometric View, Stepped, Sloped).

    Pass conditions, in that style's own tile geometry:
    - the strong outline has a ring whose bounds are exactly the area's 35 tiles' (both corner
      tiles included), and the strong fill is the same path;
    - every area tile's centre is inside that ring and every tile bordering the area is not;
    - the location's mark is drawn inside the location tile (its centre is the tile's centre);
    - the run is one segment from the area centre tile's anchor to the location's anchor;
    - the one label reads the run's distance and is pinned at the location's anchor.
    """
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtWidgets import QApplication

    from descape import ruler

    gen = _gen()
    area, location = gen.GEOMETRY_PATROL_AREA, gen.GEOMETRY_PATROL_LOCATION
    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        window.mode_combo.setCurrentText("View")
        window.iso_action.setChecked(iso)
        window.terrain_style_combo.setCurrentText(style)
        QApplication.processEvents()
        assert view._terrain_style == ("stepped" if iso else style.lower())
        assert view._trigger_emphasis == frozenset({PATROL})

        ring = _ring_around(view, "outline_strong", area)
        assert ring is not None, f"no strong ring over the patrol's tiles in {style} iso={iso}"
        roles = _items_by_role(view)
        assert roles["fill_strong"].path() == roles["outline_strong"].path()
        x1, y1, x2, y2 = area
        for tile in _area_tiles(area):
            assert ring.containsPoint(_tile_centre(view, tile), Qt.OddEvenFill), tile
        border = [(x, y) for x in range(x1 - 1, x2 + 2) for y in (y1 - 1, y2 + 1)]
        border += [(x, y) for x in (x1 - 1, x2 + 1) for y in range(y1, y2 + 1)]
        for tile in border:
            assert not ring.containsPoint(_tile_centre(view, tile), Qt.OddEvenFill), tile

        mark_centre = _mark_centre(view, location)
        assert view._tile_polygon(*location).containsPoint(mark_centre, Qt.OddEvenFill)
        assert roles["marks_strong"].path().contains(mark_centre)
        middle = ruler.measure((32, 43), location)
        a, b = view._ruler_anchor((32, 43)), view._ruler_anchor(location)
        segments = [p for p in roles["marks_strong"].path().toSubpathPolygons() if p.count() == 2]
        assert any(
            (QPointF(p[0]) - a).manhattanLength() < 0.01 and (QPointF(p[1]) - b).manhattanLength() < 0.01 for p in segments
        ), "no run segment from the area centre to the location"
        (label,) = [i for i in view.trigger_overlay_items() if i.data(0) == "label"]
        assert label.text() == ruler.format_measurement(middle)
        assert (label.pos() - b).manhattanLength() < 0.01
    finally:
        conftest.close_window(window)


def test_another_entry_draws_strongest_the_patrol_fades_but_stays_and_the_trigger_row_draws_all_strong() -> None:
    """GH #41 step 4. Pass conditions:
    - with the condition area selected, its ring is strong and the Patrol's ring is in the dim
      outline, not the strong one;
    - with the trigger row clicked, it is the only selected row, no dim area roles are drawn
      and both rings are strong.
    """
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    gen = _gen()
    window = _window()
    try:
        view = window.map_view
        panel = window.trigger_panel
        panel.select_entry(*PATROL)
        panel.select_entry(*CONDITION_AREA)
        assert view._trigger_emphasis == frozenset({CONDITION_AREA})
        roles = _items_by_role(view)
        strong, dim = _ring_polygons(roles["outline_strong"].path()), _ring_polygons(roles["outline_dim"].path())
        assert _area_ring(view, gen.GEOMETRY_CONDITION_AREA) in strong
        assert _area_ring(view, gen.GEOMETRY_PATROL_AREA) in dim
        assert _area_ring(view, gen.GEOMETRY_PATROL_AREA) not in strong

        # A plain click on the entry list's own "Trigger" row. Not setCurrentItem(): under
        # ExtendedSelection that reads QApplication.keyboardModifiers(), which an earlier
        # QTest Ctrl+key click in the same process can leave latched, turning it into a Ctrl-click.
        tree = panel.entry_tree
        root = tree.topLevelItem(0)
        assert root.data(0, Qt.UserRole) == ("trigger", -1)
        # The short entry list has scrolled the root out of view, as a user would scroll it back.
        tree.scrollToItem(root)
        rect = tree.visualItemRect(root)
        point = QPoint(rect.left() + 5, rect.center().y())
        assert tree.itemAt(point) is root, (rect, tree.viewport().size())
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        assert tree.currentItem() is root
        assert tree.selectedItems() == [root]
        assert panel.current_entry_ref() is None
        assert view._trigger_emphasis is None
        roles = _items_by_role(view)
        assert not {"outline_dim", "fill_dim"} & set(roles)
        strong = _ring_polygons(roles["outline_strong"].path())
        assert {_area_ring(view, gen.GEOMETRY_CONDITION_AREA), _area_ring(view, gen.GEOMETRY_PATROL_AREA)} <= strong
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_outline_climbs_a_raised_corner_tile(style: str) -> None:
    """GH #41 step 6. Pass conditions, the condition area's topmost on-screen tile raised with
    the Elevation tool: before and after, the strong ring's bounds are exactly the area tiles'
    own current polygons, and its top edge rose with the raised tile's top.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    gen = _gen()
    area = gen.GEOMETRY_CONDITION_AREA
    window = _window(style=style)
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*CONDITION_AREA)
        before = _ring_around(view, "outline_strong", area)
        assert before is not None
        top_tile = min(_area_tiles(area), key=lambda t: view._tile_polygon(*t).boundingRect().top())
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("elevation")
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*top_tile, Qt.NoModifier)
        window.on_edit_stroke_end()
        QApplication.processEvents()
        assert window.scenario.map_manager.get_tile(*top_tile).elevation > 0
        after = _ring_around(view, "outline_strong", area)
        assert after is not None, "the ring no longer matches the area tiles' raised polygons"
        raised_top = view._tile_polygon(*top_tile).boundingRect().top()
        assert after.boundingRect().top() < before.boundingRect().top()
        assert abs(after.boundingRect().top() - raised_top) < 0.5
    finally:
        conftest.close_window(window)


def test_the_four_trigger_overlay_swatches_recolour_the_drawing_at_once(monkeypatch) -> None:
    """GH #41 step 11, through Settings > Appearance's own swatches (colour dialog answered).

    Pass conditions, the Task Object effect selected (an area-free run to a referenced house,
    plus the patrol area dimmed): after each swatch, the items that colour inks carry it at
    once: the area outline pen, the area fill's RGB (its alpha kept), the marks/run pen and the
    run label, and the referenced-unit outline pen.
    """
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog

    from descape.viewer import SettingsDialog

    gen = _gen()
    window = _window()
    dialog = SettingsDialog(window)
    try:
        view = window.map_view
        window.trigger_panel.select_entries([PATROL, ("effect", gen.TASK_OBJECT_EFFECT)])
        picks = {
            "trigger_area_outline": "#ff00ff",
            "trigger_area_fill": "#00ffff",
            "trigger_run": "#ffff00",
            "trigger_unit_ref": "#0000ff",
        }
        for color_id, hex_str in picks.items():
            monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, h=hex_str, **k: QColor(h)))
            dialog._overlay_swatches[color_id].click()
            assert _settings_color(color_id) == hex_str
        roles = _items_by_role(view)
        assert roles["outline_strong"].pen().color().name() == "#ff00ff"
        fill = roles["fill_strong"].brush().color()
        assert fill.name() == "#00ffff" and fill.alpha() == view.TRIGGER_FILL_ALPHA
        assert roles["marks_strong"].pen().color().name() == "#ffff00"
        labels = [i for i in view.trigger_overlay_items() if i.data(0) == "label"]
        assert labels and all(i.brush().color().name() == "#ffff00" for i in labels)
        assert roles["units_strong"].pen().color().name() == "#0000ff"
    finally:
        dialog.close()
        conftest.close_window(window)


def _settings_color(color_id: str) -> str:
    from descape import settings

    return settings.get_overlay_color(color_id)


def test_opening_another_scenario_then_closing_leaves_nothing_of_the_drawing() -> None:
    """GH #41 step 12. Pass conditions: with the Patrol drawn, opening the blank template
    drops every overlay item and shape, and leaves no trigger-overlay item in the scene; File >
    Close afterwards leaves none either.
    """
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = _window()
    try:
        view = window.map_view
        window.trigger_panel.select_entry(*PATROL)
        assert view.trigger_overlay_items()
        roles = {"label", *(f"{k}_{s}" for k in ("fill", "outline", "marks", "units") for s in ("strong", "dim"))}

        def leftovers() -> list:
            return [i for i in view.scene().items() if i.data(0) in roles] if view.scene() is not None else []

        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None and window.scenario.path == BLANK_TEMPLATE_PATH
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
        assert leftovers() == []
        window.close_scenario()
        assert view._trigger_shapes == [] and view.trigger_overlay_items() == []
        assert leftovers() == []
    finally:
        conftest.close_window(window)
