"""ViewerWindow's Clone Stamp (GH #285): the source a Ctrl+click sets, its
marker query, and the stroke's start, per-dab write and end, which
viewer_strokes.py dispatches here. The copy and the seam are
descape/clone_stamp.py."""

from __future__ import annotations

from descape import clone_stamp, debug_log, perf_trace
from descape.edit_history import TileDiffRecord

ILLEGAL_SOURCE_WARNING = (
    "The copy kept an elevation step of more than one level that the map already had at the source or beside the copy; "
    "check Tools > Map Analysis"
)


class _ViewerCloneStampMixin:
    """ViewerWindow's Clone Stamp (viewer.py's docstring has the split's rules)."""

    def set_clone_source(self, tile: tuple[int, int]) -> None:
        """MapView's Ctrl+press: the anchor for the next press's offset. Records nothing."""
        mm = self.scenario.map_manager if self.scenario is not None else None
        # Flat picks x == map_width on the map rect's far edge.
        if mm is None or not (0 <= tile[0] < mm.map_width and 0 <= tile[1] < mm.map_height):
            return
        self._clone.set_anchor(tile)
        self.map_view.refresh_clone_marker(self._hover_tile)
        self._log_status(f"Clone source set at ({tile[0]}, {tile[1]})")

    def _clone_marker_tiles(self, hover: tuple[int, int] | None) -> tuple:
        """MapView's marker query: the brush footprint the next dab samples from, () for none."""
        origin = self._clone.marker_origin(hover) if self.scenario is not None else None
        return () if origin is None else tuple(self._brush_footprint(*origin))

    def _clone_elevation_ok(self) -> bool:
        """The Elevation box's gate, Paste Region's: elevation_ok in _update_tool_enabled()
        (write_ok and a square map), so a compat document clones terrain only."""
        scenario = self.scenario
        return (
            self.clone_action.isEnabled()
            and scenario.terrain_write_supported
            and self.game_data_lock_reason() is None
            and scenario.map_is_square
        )

    def _begin_clone_stroke(self) -> None:
        """on_edit_stroke_start() for the Clone Stamp: the full snapshot the copy
        reads, or no stroke at all when there is nothing to copy."""
        terrain = self.clone_terrain_check.isChecked()
        elevation = self.clone_elevation_check.isChecked() and self._clone_elevation_ok()
        if self._clone.anchor is None:
            self._log_status("Ctrl+click a tile to set the clone source")
            return
        if not terrain and not elevation:
            self._log_status("Tick Terrain or Elevation to clone")
            return
        mm = self.scenario.map_manager
        with perf_trace.phase("stroke_snapshot"):
            self.edit_history.begin_stroke(mm.terrain)
        with perf_trace.phase("stroke_start"):
            self._init_stroke_state()
            # Snapshotted like the Target: a mid-stroke tick can't half-apply across one record.
            self._stroke_clone = (terrain, elevation)
            self._clone_stroke = (
                clone_stamp.CloneStroke.from_tiles(mm.terrain, mm.map_width, mm.map_height) if elevation else None
            )

    def _clone_press(self, tiles) -> None:
        """The stroke's first touch is the press tile (MapView touches it straight after the start)."""
        if self._current_tool == "clone" and self._stroke_clone is not None and tiles:
            self._clone.fix_offset(tiles[0])

    def _clone_footprint(self, mm, footprint) -> set[int]:
        """_mutate_footprint()'s Clone Stamp write: one dab, the clamp's ring included."""
        terrain, elevation = self._stroke_clone
        targets = clone_stamp.clone_targets(
            footprint, self._clone.offset, self.edit_history.stroke_start_state, mm.map_width, mm.map_height
        )
        _cloned, written = clone_stamp.apply_clone(
            mm, self._clone_stroke, targets, terrain=terrain, elevation=elevation, carry_layer=self._layer_write_ok()
        )
        # A set even when nothing landed, so the footprint still counts as visited.
        return written

    def _end_clone_stroke(self, label: str) -> None:
        """on_edit_stroke_end() for the Clone Stamp: Draw's record, Trees and Eye
        candy when Terrain was ticked, and the illegal-source warning."""
        terrain, elevation = self._stroke_clone
        mm = self.scenario.map_manager
        with perf_trace.phase("stroke_record"):
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain, self._stroke_written)
        self._log_status(self._clone_status(tile_record))
        if terrain:
            with perf_trace.phase("unit_plan"):
                record, splices = self._apply_terrain_unit_plan(tile_record, mm, with_splices=True)
        else:
            record, splices = tile_record, None
        # Every tile in the record was already repainted live, step by step.
        self._push_terrain_unit_record(record, splices, live_painted=True)
        if elevation and self._clone_stroke.violations(self._stroke_written):
            # Only a source that already held one leaves one; the edit stays, one undo away.
            self._log_status(ILLEGAL_SOURCE_WARNING, level=debug_log.WARNING)

    def _clone_status(self, tile_record: TileDiffRecord | None) -> str:
        dx, dy = self._clone.offset or (0, 0)
        changed = len(tile_record.changes) if tile_record is not None else 0
        line = f"Clone Stamp: {changed} tiles from offset ({dx}, {dy})"
        return line if tile_record is not None else f"{line} (no change)"
