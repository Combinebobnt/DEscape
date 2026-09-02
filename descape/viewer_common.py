"""Widget helpers and tool-id lookups shared by the viewer's panels.

Split out of viewer.py so trigger_panel and map_options_panel can share
them without importing viewer.py, which imports both of them. The
settings.TOOLS-derived sets live here as one documented family even
though their consumers are split across map_view.py and viewer.py.

They are deliberately not in settings.py next to the TOOLS registry they
derive from: they are view-layer lookups rather than settings (three of the
seven are private by name), and settings.py's import surface is depended on
directly by a lot of tests."""

from __future__ import annotations



from PyQt5.QtWidgets import (
    QComboBox,
    QSpinBox,
)


from descape import (
    settings,
)


# Derived from settings.TOOLS -- the single tool registry -- rather than
# hand-duplicated here, so a new tool added there can't silently miss an
# entry in any of these. Internal tool identifiers (e.g. "elevation",
# "set_level") are unchanged even when a display label is renamed later
# (Elevate / Set Elevation), so settings.get_keybind()'s persisted
# "tool_elevation"/"tool_set_level" keys and every other internal reference
# keep working regardless.
EDIT_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.is_edit_tool)
_TOOL_LABELS = {t.tool_id: t.label for t in settings.TOOLS}
_STROKE_LABELS = {t.tool_id: t.stroke_label for t in settings.TOOLS if t.is_edit_tool}
# One-shot click tools (Paint Can) that must never enter the drag-stroke
# path -- see MapView.mousePressEvent's early CLICK_TOOLS branch and
# ViewerWindow.on_fill.
CLICK_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.click_only)
# Which toolbar param widget ("terrain" | "level" | "") the active tool
# reads -- see ViewerWindow._update_tool_enabled's tool-param visibility
# block.
_TOOL_PARAM = {t.tool_id: t.param_widget for t in settings.TOOLS}
# Tools whose stroke applies across a brush footprint (size + shape) rather
# than always exactly one tile -- see ToolDef.supports_brush's own comment.
BRUSH_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_brush)
# The Ruler is the only tool that is neither an edit tool nor Pan, so it
# can't be recognised by set membership the way the four sets above are.
TOOL_RULER = "ruler"

# Which mode(s) each tool's toolbar button shows in -- see
# settings.ToolDef.modes.
_TOOL_MODES = {t.tool_id: t.modes for t in settings.TOOLS}


def tool_applicable(tool_id: str, mode: str) -> bool:
    """True if `tool_id`'s toolbar button should be visible in `mode`.

    Empty ToolDef.modes means every mode. Used by
    ViewerWindow._update_tool_enabled() to hide mode-inapplicable tools
    instead of just greying them out.
    """
    modes = _TOOL_MODES.get(tool_id, ())
    return not modes or mode in modes

# How many characters a property-form combo box sizes itself for. Without it a
# combo's minimumSizeHint is its longest *item*, and ObjectAttribute's longest
# is 430 px in a 340 px panel -- so the form sat permanently below its own
# stated minimum, which is what made every attempt to narrow the label column
# tip the whole form into horizontal overflow.
_COMBO_CONTENTS_CHARS = 12


def _fit_combo_width(combo: QComboBox) -> None:
    """Size a combo for a fixed character count rather than for its longest
    item -- see _COMBO_CONTENTS_CHARS. The popup still shows every name in
    full. Shared by TriggerPanel and MapOptionsPanel: VictoryCondition and
    SecondaryGameMode are the same class of long-label enum ObjectAttribute
    is."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(_COMBO_CONTENTS_CHARS)


def _make_spinbox(
    value,
    editable: bool,
    *,
    minimum: int,
    maximum: int,
    special_value_text: str = "",
) -> QSpinBox:
    """A property-form spinbox with keyboard tracking off.

    `setKeyboardTracking(False)` is the load-bearing one: with it on, typing
    "12" over a 5 emits valueChanged(1) and then valueChanged(12), so a
    two-digit edit records two undo steps and briefly writes a value the user
    never asked for.

    Range is a parameter rather than the hardcoded trigger range this was
    promoted from, because Options.ai_map_type's own minimum is INT32_MIN and
    a -1 floor would silently clamp it. `special_value_text` renders
    `minimum` as text instead of a number whose meaning the user would have to
    know -- TriggerPanel passes it for trigger_fields.UNSET's -1 sentinel, and
    Map Options, which has no unset sentinel, passes nothing.
    """
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    if special_value_text:
        spin.setSpecialValueText(special_value_text)
    # Falls back to `minimum` rather than to 0, so TriggerPanel keeps landing
    # on UNSET (and therefore on special_value_text) for a None-valued field.
    spin.setValue(value if isinstance(value, int) and not isinstance(value, bool) else minimum)
    spin.setEnabled(editable)
    spin.setKeyboardTracking(False)
    return spin
