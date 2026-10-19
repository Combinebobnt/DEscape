"""Names to QIcons, and which icon each tool, menu command and toolbar button gets.

`apply_icons(window)` is the one place icons are set; it runs once at the end of
`ViewerWindow.__init__` and again after each palette change, so a theme switch
re-inks everything. Checkable toggles get no icon: under Fusion a menu icon
replaces the checkmark.
"""

from __future__ import annotations

from PyQt5.QtGui import QIcon

from descape import icon_art, settings

# 16 is the menus', 24 the toolbar's; 32 covers a 2x menu.
RENDERED_SIZES = (16, 24, 32)
RENDERED_RATIOS = (1.0, 2.0)

TOOL_ICONS: dict[str, str] = {tool.tool_id: f"tool-{tool.tool_id}" for tool in settings.TOOLS}

# settings.REBINDABLE_ACTIONS id -> icon, reached through ViewerWindow._keybind_actions.
ACTION_ICONS: dict[str, str] = {
    "file_new": "file-new",
    "file_new_default": "file-new",
    "file_open": "file-open",
    "file_close": "file-close",
    "file_save": "file-save",
    "file_save_as": "file-save-as",
    "file_recover_autosave": "file-recover",
    "file_exit": "file-exit",
    "edit_undo": "edit-undo",
    "edit_redo": "edit-redo",
    "edit_cut": "edit-cut",
    "edit_copy": "edit-copy",
    "edit_paste": "edit-paste",
    "edit_clipboard_history": "edit-clipboard-history",
    "edit_history": "edit-history",
    "edit_select_all": "select-all",
    "edit_deselect": "select-none",
    "edit_scatter_units": "edit-scatter",
    "edit_find_replace": "edit-find",
    "edit_flip_horizontal": "flip-horizontal",
    "edit_flip_vertical": "flip-vertical",
    "edit_settings": "settings",
    "unit_rotate_ccw": "rotate-ccw",
    "unit_rotate_cw": "rotate-cw",
    "unit_rotate_ccw_coarse": "rotate-ccw-quarter",
    "unit_rotate_cw_coarse": "rotate-cw-quarter",
    "unit_variant_prev": "variant-prev",
    "unit_variant_next": "variant-next",
    "unit_variant_random": "variant-random",
    "unit_select_stack": "select-stack",
    "map_resize": "map-resize",
    "map_mirror": "map-mirror",
    "map_disables": "map-disables",
    "analysis_run": "map-analysis",
    "help_about": "help-about",
    "help_debug_log": "help-debug-log",
    "filter_all_players": "filter-all-players",
    "filter_no_players": "filter-no-players",
    "filter_show_all": "filter-show-all",
    "filter_hide_all": "filter-hide-all",
}

# ViewerWindow attribute -> icon, for what has no rebindable row: the toolbar's own
# Rotate pair (separate QActions from the menu's) and the two popup buttons.
WINDOW_ICONS: dict[str, str] = {
    "rotate_ccw_toolbar_action": "rotate-ccw",
    "rotate_cw_toolbar_action": "rotate-cw",
    "filters_button": "filters",
    "more_tools_button": "more-tools",
}
# Open Recent's Clear entry is rebuilt with the menu, so it reads this name itself.
CLEAR_RECENT_ICON = "file-clear-recent"
# Every preset size under File > New Map shares the New icon.
NEW_SIZE_ICON = "file-new"

_cache: dict[str, QIcon] = {}
_cache_colours: tuple[int, ...] = ()


def icon(name: str) -> QIcon:
    """Cached for the current colours only, so a theme change neither serves a stale icon nor keeps old sets."""
    global _cache_colours
    if name not in icon_art.ICON_SHAPES:
        raise KeyError(f"{name!r} has no artwork in icon_art.ICON_SHAPES")
    ink = icon_art.ink_colour()
    backdrops = icon_art.backdrop_colours()
    colours = (ink.rgba(), *(b.rgba() for b in backdrops))
    if colours != _cache_colours:
        _cache.clear()
        _cache_colours = colours
    built = _cache.get(name)
    if built is None:
        built = QIcon()
        for size in RENDERED_SIZES:
            for ratio in RENDERED_RATIOS:
                built.addPixmap(icon_art.paint_icon(name, size, ink, ratio, backdrops))
        _cache[name] = built
    return built


def tool_icon(tool_id: str) -> QIcon:
    return icon(TOOL_ICONS[tool_id])


def apply_icons(window) -> None:
    """Every tool, menu command and toolbar button icon. Raises on a table entry the window
    lacks, so a renamed action fails here rather than losing its icon."""
    for tool_id, name in TOOL_ICONS.items():
        action = getattr(window, f"{tool_id}_action")
        action.setIcon(icon(name))
        # Tools are checkable, so More Tools' menu keeps its radio mark rather than the icon.
        action.setIconVisibleInMenu(False)
    for action_id, name in ACTION_ICONS.items():
        try:
            action = window._keybind_actions[action_id]
        except KeyError:
            raise KeyError(f"ui_icons.ACTION_ICONS names {action_id!r}, which has no action") from None
        action.setIcon(icon(name))
    for attr, name in WINDOW_ICONS.items():
        getattr(window, attr).setIcon(icon(name))
    for action in window.new_size_actions:
        action.setIcon(icon(NEW_SIZE_ICON))
    if window.clear_recent_action is not None:
        window.clear_recent_action.setIcon(icon(CLEAR_RECENT_ICON))
    # The More Tools button wears the active tool's icon while that tool is in its menu.
    window._update_more_tools_button()
