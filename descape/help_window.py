"""The built-in guide's window and the Help menu's browser links (GH #160).

Links never use setOpenExternalLinks(): that calls the C++ openUrl, which a
test cannot patch, so every link goes through anchorClicked and open_url()."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QKeySequence, QTextCursor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from descape import help_guide, settings


def open_url(url: str, report: Callable[[str], None]) -> bool:
    """Opens url in the web browser. When none opens, the link goes to the
    clipboard instead, and report() says which happened."""
    if QDesktopServices.openUrl(QUrl(url)):
        report(f"Opened {url} in your web browser")
        return True
    QGuiApplication.clipboard().setText(url)
    report(f"Couldn't open a web browser; link copied: {url}")
    return False


def topic_url(topic: help_guide.Topic) -> QUrl:
    return QUrl.fromLocalFile(str(help_guide.topic_path(topic)))


def topic_for_url(url: QUrl) -> help_guide.Topic | None:
    """The registered topic a local URL names, or None for anything outside the guide."""
    if not url.isLocalFile():
        return None
    path = Path(url.toLocalFile()).resolve()
    if path.parent != help_guide.GUIDE_DIR.resolve():
        return None
    return help_guide.topic_by_filename(path.name)


def _native_key(sequence: str) -> str:
    return QKeySequence(sequence).toString(QKeySequence.NativeText)


def keys_page_text() -> str:
    """The Keyboard and Mouse file with the user's live keys spliced in."""
    topic = help_guide.topic_by_id(help_guide.KEYS_TOPIC_ID)
    bindings = {action_id: settings.get_keybind(action_id) for action_id, _, _ in settings.REBINDABLE_ACTIONS}
    block = help_guide.keys_markdown(bindings, settings.get_wheel_bindings(), fmt=_native_key)
    return help_guide.splice_keys(help_guide.topic_text(topic), block)


class HelpBrowser(QTextBrowser):
    """Serves the guide's files itself (UTF-8, and the keys page generated),
    and hands every clicked link to the window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)

    def loadResource(self, resource_type: int, url: QUrl):
        topic = topic_for_url(url)
        if topic is None:
            return super().loadResource(resource_type, url)
        if topic.topic_id == help_guide.KEYS_TOPIC_ID:
            return keys_page_text()
        return help_guide.topic_text(topic)


class HelpWindow(QDialog):
    """The guide: a topic list, the page, Back and Forward, and a search box.
    Non-modal and shared: ViewerWindow keeps one and re-selects its topic."""

    def __init__(self, parent: QWidget | None, report: Callable[[str], None]) -> None:
        super().__init__(parent)
        self._report = report
        self.setWindowTitle("DEscape Guide")
        self.setModal(False)
        self.resize(900, 640)

        self.back_button = QPushButton("Back")
        self.back_button.setToolTip("The page you were on before this one")
        self.back_button.setEnabled(False)
        self.forward_button = QPushButton("Forward")
        self.forward_button.setToolTip("The page you went Back from")
        self.forward_button.setEnabled(False)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search the guide")
        self.search_edit.setToolTip(
            "Shows only the topics that contain this text. Enter finds it in the page you are reading"
        )
        self.search_edit.setClearButtonEnabled(True)

        self.topic_list = QListWidget()
        self._topic_texts: dict[str, str] = {}
        for topic in help_guide.TOPICS:
            item = QListWidgetItem(help_guide.topic_title(topic))
            item.setData(Qt.UserRole, topic.topic_id)
            self.topic_list.addItem(item)
            self._topic_texts[topic.topic_id] = help_guide.topic_text(topic).casefold()
        self.browser = HelpBrowser()

        top = QHBoxLayout()
        top.addWidget(self.back_button)
        top.addWidget(self.forward_button)
        top.addWidget(self.search_edit, 1)
        splitter = QSplitter()
        splitter.addWidget(self.topic_list)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 680])
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)

        self.back_button.clicked.connect(self.browser.backward)
        self.forward_button.clicked.connect(self.browser.forward)
        self.browser.backwardAvailable.connect(self.back_button.setEnabled)
        self.browser.forwardAvailable.connect(self.forward_button.setEnabled)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        self.browser.sourceChanged.connect(self._on_source_changed)
        self.topic_list.currentRowChanged.connect(self._on_row_changed)
        self.search_edit.textChanged.connect(self._filter_topics)
        self.search_edit.returnPressed.connect(self._find_in_page)

    def current_topic_id(self) -> str | None:
        topic = topic_for_url(self.browser.source())
        return topic.topic_id if topic is not None else None

    def select_topic(self, topic_id: str) -> None:
        """Shows the topic and selects its row. Re-selecting the shown topic
        reloads it, so the keys page picks up a rebinding made since."""
        topic = help_guide.topic_by_id(topic_id)
        self._set_row(topic.topic_id)
        url = topic_url(topic)
        if self.browser.source() == url:
            self.browser.reload()
        else:
            self.browser.setSource(url)

    def _set_row(self, topic_id: str) -> None:
        for row in range(self.topic_list.count()):
            if self.topic_list.item(row).data(Qt.UserRole) == topic_id:
                with QSignalBlocker(self.topic_list):
                    self.topic_list.setCurrentRow(row)
                return

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.select_topic(self.topic_list.item(row).data(Qt.UserRole))

    def _on_source_changed(self, url: QUrl) -> None:
        topic = topic_for_url(url)
        if topic is not None:
            self._set_row(topic.topic_id)

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() in ("http", "https"):
            open_url(url.toString(), self._report)
            return
        topic = topic_for_url(self.browser.source().resolved(url))
        if topic is None:
            self._report(f"The guide only opens its own pages and web links, not {url.toString()}")
            return
        self.select_topic(topic.topic_id)

    def _filter_topics(self, text: str) -> None:
        query = text.strip().casefold()
        for row in range(self.topic_list.count()):
            item = self.topic_list.item(row)
            item.setHidden(bool(query) and query not in self._topic_texts[item.data(Qt.UserRole)])

    def _find_in_page(self) -> None:
        query = self.search_edit.text().strip()
        if not query or self.browser.find(query):
            return
        self.browser.moveCursor(QTextCursor.Start)
        self.browser.find(query)
