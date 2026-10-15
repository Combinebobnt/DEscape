"""descape/unit_card.py (GH #148, TASK-337): the portrait, name, Unit ID and
base-stats card the Units panel shows twice, standalone.

Geometry is read on a shown offscreen card after processEvents(), from
height() and geometry(), never from a repeated size hint: an unshown
widget's hint goes stale after its first query. Where a hint is the subject,
each state gets a fresh card.
"""

from __future__ import annotations

import pytest

from descape import asset_source, object_catalog, unit_card
from descape.unit_card import UnitCard

import conftest
from conftest import _UNIT_RED, ManualExecutor, _fake_install

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

@pytest.fixture(autouse=True)
def _no_install_from_the_environment(monkeypatch):
    """The no-install cases stay so with AOE2DE_INSTALL_PATH set, which outranks the config conftest hides."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


_ARCHER = 4  # all five stats
_TREE = 349  # HP only
_UNKNOWN = 999999  # no stats, no portrait
_ARCHER_STATS = {"hp": "30", "attack": "4", "melee_armour": "0", "pierce_armour": "0", "range": "4"}


def _card(**kwargs) -> UnitCard:
    conftest.ensure_qapp()
    return UnitCard(**kwargs)


def _shown_stats(card) -> dict[str, str]:
    return {
        field_id: value.text()
        for field_id, (caption, value) in card.stat_rows.items()
        if caption.isVisibleTo(card) and value.isVisibleTo(card)
    }


def _hosted(card, font=None):
    """The card in a shown window, above a stretch, so it gets its own height."""
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

    host = QWidget()
    if font is not None:
        host.setFont(font)
    layout = QVBoxLayout(host)
    layout.addWidget(card)
    layout.addStretch(1)
    host.resize(500, 600)
    host.show()
    QApplication.processEvents()
    return host


def _height(card) -> int:
    from PyQt5.QtWidgets import QApplication

    # Twice: the card's own layout request settles first, its host's on the next pass (measured).
    QApplication.processEvents()
    QApplication.processEvents()
    return card.height()


def _portrait_centre(card) -> tuple[int, int, int, int]:
    image = card.portrait.pixmap().toImage()
    colour = image.pixelColor(image.width() // 2, image.height() // 2)
    return colour.red(), colour.green(), colour.blue(), colour.alpha()


def _pump(until=lambda: True, timeout=5.0) -> None:
    import time

    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    QApplication.processEvents()
    while not until() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.002)
    QApplication.processEvents()


# -- what a card shows ------------------------------------------------------------


@pytest.mark.parametrize("stable", [False, True])
def test_show_const_fills_the_name_unit_id_and_stats(stable) -> None:
    card = _card(stable=stable)
    card.show_const(_ARCHER, prefix="Placing: ")
    assert card.name_label.text() == f"Placing: {object_catalog.display_name(_ARCHER)}"
    assert card.const_label.text() == str(_ARCHER)
    assert card.const_label.isVisibleTo(card)
    assert _shown_stats(card) == _ARCHER_STATS

    card.show_const(_TREE, name="An oak")
    assert card.name_label.text() == "An oak"
    assert _shown_stats(card) == {"hp": "20"}

    card.show_const(_UNKNOWN)
    assert card.name_label.text() == object_catalog.display_name(_UNKNOWN)
    assert card.const_label.text() == str(_UNKNOWN)
    assert _shown_stats(card) == {}


@pytest.mark.parametrize("stable", [False, True])
def test_show_mixed_and_clear(stable) -> None:
    card = _card(stable=stable)
    card.show_const(_ARCHER)
    card.show_mixed()
    assert (card.name_label.text(), card.const_label.text()) == ("(mixed)", "(mixed)")
    assert _shown_stats(card) == {}
    assert card.portrait_source is None

    card.show_const(_ARCHER)
    card.clear("Placing: (none)")
    assert card.name_label.text() == "Placing: (none)"
    assert card.const_label.text() == ""
    assert _shown_stats(card) == {}
    assert card.portrait_source is None


# -- a stable card never changes height -------------------------------------------


def _stable_heights(card, slot_visible: bool) -> dict[str, int]:
    """Per state; the slot shows in every state with an install and in none without (user ruling, TASK-337)."""
    heights = {}
    for state, show in (
        ("archer", lambda: card.show_const(_ARCHER)),
        ("tree", lambda: card.show_const(_TREE)),
        ("unknown", lambda: card.show_const(_UNKNOWN)),
        ("cleared", lambda: card.clear("Placing: (none)")),
    ):
        show()
        heights[state] = _height(card)
        assert card.portrait.isVisibleTo(card) == slot_visible, state
    return heights


@pytest.fixture(params=[False, True], ids=["no_install", "fake_install"])
def install_state(request, tmp_path) -> bool:
    """Whether the fake install is configured for the test."""
    if request.param:
        asset_source.set_install_path_override(_fake_install(tmp_path))
    yield request.param
    if request.param:
        asset_source.set_install_path_override(None)


def _bigger_font():
    from PyQt5.QtWidgets import QApplication

    font = QApplication.font()
    font.setPointSize(font.pointSize() + 6)
    return font


def test_a_stable_card_keeps_one_height_at_or_above_a_five_stat_card(install_state) -> None:
    card = _card(stable=True)
    host = _hosted(card)
    reference = _card()
    reference_host = _hosted(reference)
    try:
        heights = _stable_heights(card, slot_visible=install_state)
        assert len(set(heights.values())) == 1, heights
        reference.show_const(_ARCHER)
        assert heights["archer"] >= _height(reference)
    finally:
        host.close()
        reference_host.close()


def test_a_stable_card_keeps_one_height_after_a_font_change(install_state) -> None:
    """GH #142 scales the app font at runtime: the layout holds the size, nothing re-measures."""
    big = _bigger_font()
    card = _card(stable=True)
    host = _hosted(card)
    reference = _card()
    reference_host = _hosted(reference, font=big)
    try:
        before = _stable_heights(card, slot_visible=install_state)["cleared"]
        host.setFont(big)
        heights = _stable_heights(card, slot_visible=install_state)
        assert len(set(heights.values())) == 1, heights
        assert heights["archer"] > before, "the font change must reach the card, or this proves nothing"
        reference.show_const(_ARCHER)
        assert heights["archer"] >= _height(reference)
    finally:
        host.close()
        reference_host.close()


def test_the_inspector_card_is_shorter_for_a_tree_than_for_an_archer() -> None:
    card = _card()
    host = _hosted(card)
    try:
        card.show_const(_ARCHER)
        archer = _height(card)
        card.show_const(_TREE)
        assert _height(card) < archer
    finally:
        host.close()


def test_the_longest_name_does_not_widen_a_stable_card() -> None:
    longest = max(object_catalog.objects(), key=lambda entry: len(entry.name))
    long_card = _card(stable=True)
    long_card.show_const(longest.id, prefix="Placing: ", name=longest.name)
    archer_card = _card(stable=True)
    archer_card.show_const(_ARCHER, prefix="Placing: ")
    assert long_card.minimumSizeHint().width() == archer_card.minimumSizeHint().width()
    assert long_card.name_label.text() == f"Placing: {longest.name}"


# -- the portrait comes from the shared loader (GH #68) -----------------------------


def _with_icons() -> list[int]:
    """Catalog consts with a portrait index: each one a worker decode."""
    return [entry.id for entry in object_catalog.objects() if object_catalog.icon_for(entry.id) is not None]


def _manual_card(**kwargs):
    from descape.catalog_thumbs import ThumbLoader

    executor = ManualExecutor()
    conftest.ensure_qapp()
    return _card(loader=ThumbLoader(executor=executor), **kwargs), executor


def test_rapid_picks_keep_one_decode_in_flight_and_skip_the_middle(fake_install) -> None:
    """The loader replaces TASK-337's debounce: holding an arrow key decodes the
    row in flight and then the row the user stops on, nothing in between."""
    card, executor = _manual_card(stable=True)
    picks = [*[const for const in _with_icons() if const != _ARCHER][:29], _ARCHER]
    for const in picks:
        card.show_const(const)
        assert card.portrait_source is None
    assert len(executor.submitted) == 1, "one decode in flight, never a queue"
    executor.run(0)
    _pump(lambda: len(executor.submitted) == 2)
    assert executor.icons() == [object_catalog.icon_for(picks[0]), object_catalog.icon_for(_ARCHER)]
    executor.run(1)
    _pump(lambda: card.portrait_source is not None)
    assert card.portrait_source == "icon"
    assert _portrait_centre(card) == _UNIT_RED


def test_a_portrait_one_card_resolved_shows_at_once_on_the_other(fake_install) -> None:
    from descape.catalog_thumbs import ThumbLoader

    executor = ManualExecutor()
    loader = ThumbLoader(executor=executor)
    inspector = _card(loader=loader)
    placing = _card(stable=True, loader=loader)
    inspector.show_const(_ARCHER)
    executor.run(0)
    _pump(lambda: inspector.portrait_source is not None)
    assert inspector.portrait_source == "icon"
    placing.show_const(_ARCHER)
    assert len(executor.submitted) == 1, "a memo hit asks for nothing"
    assert placing.portrait_source == "icon"
    assert _portrait_centre(placing) == _UNIT_RED


@pytest.mark.parametrize("queued", [False, True], ids=["served", "still_queued"])
def test_a_dropped_card_is_freed_by_refcount_alone(queued, fake_install) -> None:
    """The loader must not hold its clients: a card and its own loader in a
    cycle would wait for the cyclic GC, which may run on the catalog-thumb
    worker and free Qt objects off the GUI thread."""
    import gc
    import weakref

    card, executor = _manual_card(stable=True)
    card.show_const(_ARCHER)
    if queued:
        card.show_const(82)  # the castle waits behind the archer in flight
    else:
        executor.run(0)
        _pump(lambda shown=card: shown.portrait_source is not None)
        assert card.loader.idle()
    assert (len(card.loader._pending) == 1) == queued, "a served client leaves the pending map"
    dead = weakref.ref(card)
    gc.collect()
    gc.disable()
    try:
        del card
        assert dead() is None
    finally:
        gc.enable()


def test_a_portrait_for_a_const_no_longer_shown_is_not_drawn(fake_install) -> None:
    card, executor = _manual_card(stable=True)
    card.show_const(_ARCHER)
    card.clear("Placing: (none)")
    executor.run(0)
    _pump()
    assert card.portrait_source is None
    assert card.portrait.pixmap() is None or card.portrait.pixmap().isNull()


def test_a_portrait_from_an_older_install_is_dropped(fake_install) -> None:
    card, executor = _manual_card(stable=True)
    card.show_const(_ARCHER)
    asset_source.clear_install_caches()
    executor.run(0)
    _pump()
    assert card.portrait_source is None
    assert card.loader.cached(_ARCHER, unit_card.PORTRAIT_PX) == (False, None)


@pytest.mark.parametrize("answer", ["portrait", "nothing"])
@pytest.mark.parametrize("stable", [False, True])
def test_the_slot_holds_its_height_before_and_after_the_loader_answers(stable, answer, fake_install) -> None:
    """The tree's HP-only column (61 px at the test font) is shorter than the
    64 px slot. "nothing": no portrait file and no sprite, so the loader answers None."""
    from PIL import Image

    icon, _building = object_catalog.portrait_for(_TREE)
    if answer == "portrait":
        portrait = asset_source.get_install_path() / asset_source.UNIT_ICON_DIR / asset_source.UNIT_ICON_NAME.format(icon=icon)
        Image.new("RGBA", (32, 32), _UNIT_RED).save(portrait, format="DDS")
    card, executor = _manual_card(stable=stable)
    host = _hosted(card)
    try:
        card.show_const(_TREE)
        before, slot_before = _height(card), card.portrait.geometry().getRect()
        assert card.portrait.isVisibleTo(card), "a placeholder until the loader answers"
        executor.run(0)
        _pump(lambda: not card._awaiting)
        assert (card.portrait_source is None) == (answer == "nothing")
        assert card.loader.cached(_TREE, unit_card.PORTRAIT_PX)[1] is None or answer == "portrait"
        assert card.portrait.isVisibleTo(card)
        assert (_height(card), card.portrait.geometry().getRect()) == (before, slot_before)
        # A second show of the same const answers from the memo, and keeps the slot the same.
        card.show_const(_ARCHER)
        card.show_const(_TREE)
        assert card.portrait.isVisibleTo(card)
    finally:
        host.close()


def test_with_no_install_the_inspector_slot_stays_shut() -> None:
    card, executor = _manual_card()
    card.show_const(_ARCHER)
    assert executor.submitted == []
    assert not card.portrait.isVisibleTo(card)


def test_the_trebuchet_and_town_center_share_icon_28_but_not_its_folder(tmp_path) -> None:
    """TASK-075's S1 check on TASK-191: siege is .dat type 80 too, and its
    portrait lives in units/, where the Town Center's lives in buildings/."""
    from PIL import Image

    install = _fake_install(tmp_path)
    ingame = install / "widgetui" / "textures" / "ingame"
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(ingame / "units" / "028_50730.DDS", format="DDS")
    Image.new("RGBA", (8, 8), (4, 5, 6, 255)).save(ingame / "buildings" / "028_town_center.DDS", format="DDS")
    asset_source.set_install_path_override(install)
    try:
        assert object_catalog.portrait_for(42) == (28, False)
        assert object_catalog.portrait_for(109) == (28, True)
        assert asset_source.unit_icon_path(*object_catalog.portrait_for(42)).parent.name == "units"
        assert asset_source.unit_icon_path(*object_catalog.portrait_for(109)).parent.name == "buildings"
    finally:
        asset_source.set_install_path_override(None)


def test_mixed_counts_the_types_and_shows_no_stats() -> None:
    card = _card()
    card.show_const(_ARCHER)
    card.show_mixed(3)
    assert card.name_label.text() == "(mixed)"
    assert card.const_label.text() == "(mixed: 3 types)"
    assert _shown_stats(card) == {}


def test_range_reads_min_to_max_where_they_differ() -> None:
    card = _card()
    card.show_const(938)  # range 2.5, max_range 3
    assert _shown_stats(card)["range"] == "2.5-3"
    card.show_const(_ARCHER)
    assert _shown_stats(card)["range"] == "4"


def test_refresh_thumbnails_follows_an_install_change(tmp_path) -> None:
    card = _card(stable=True)
    card.show_const(_ARCHER)
    hp_caption = card.stat_rows["hp"][0]
    assert hp_caption.text() == "Hit points:"
    asset_source.set_install_path_override(_fake_install(tmp_path))
    try:
        card.refresh_thumbnails()
        conftest.settle_thumbs(card.loader)
        assert hp_caption.text() == ""
        assert hp_caption.pixmap().height() == hp_caption.fontMetrics().height()
        assert card.portrait_source == "icon"
        assert _portrait_centre(card) == _UNIT_RED
    finally:
        asset_source.set_install_path_override(None)
    card.refresh_thumbnails()
    conftest.settle_thumbs(card.loader)
    assert hp_caption.text() == "Hit points:"
    assert card.portrait_source != "icon"


@pytest.mark.parametrize("shown", [True, False], ids=["a_const_shown", "cleared"])
def test_refresh_thumbnails_shows_the_slot_with_an_install_and_hides_it_without(shown, tmp_path) -> None:
    """User ruling (TASK-337): no install, no slot, so the card costs the pane no width."""
    card = _card(stable=True)
    if shown:
        card.show_const(_ARCHER)
    else:
        card.clear("Placing: (none)")
    assert not card.portrait.isVisibleTo(card)
    no_install_width = card.minimumSizeHint().width()
    asset_source.set_install_path_override(_fake_install(tmp_path))
    try:
        card.refresh_thumbnails()
        assert card.portrait.isVisibleTo(card)
    finally:
        asset_source.set_install_path_override(None)
    card.refresh_thumbnails()
    assert not card.portrait.isVisibleTo(card)
    assert card.minimumSizeHint().width() == no_install_width
