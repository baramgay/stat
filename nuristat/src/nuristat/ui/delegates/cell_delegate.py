"""Cell Editor Delegate — SPSS 스타일 셀 편집기."""


from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate


class CellDelegate(QStyledItemDelegate):
    """SPSS 스타일 셀 delegate.

    핵심 동작:
    - _initial_text: DataView가 printable 키 입력 시 설정. setEditorData에서 사용.
    - eventFilter: Enter/Tab/Arrow → commitData + closeEditor + _pending_navigate 설정
    - DataView는 closeEditor 시그널 연결 후 _pending_navigate로 이동.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initial_text: str = ""
        self._pending_navigate: tuple[int, int, int, int] | None = None  # (dc, dr, src_row, src_col)
        self._editor_initialized: bool = False

    def set_initial_text(self, text: str) -> None:
        """DataView에서 printable 키 입력 시 호출. 다음 setEditorData에서 사용."""
        self._initial_text = text

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        self._editor_initialized = False  # 새 에디터마다 리셋
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        if self._editor_initialized:
            return  # 동일 에디터에 두 번 호출되는 경우 무시
        self._editor_initialized = True

        if self._initial_text:
            # printable 키로 편집 시작: clear() 후 insert()로 선택 문제 없이 입력
            editor.clear()
            editor.insert(self._initial_text)
            self._initial_text = ""
        else:
            # F2 또는 더블클릭으로 편집 시작: 기존값 보여주고 전체 선택
            value = index.data(Qt.ItemDataRole.EditRole)
            editor.setText(str(value) if value is not None and value != "" else "")
            editor.selectAll()

    def setModelData(self, editor: QLineEdit, model, index):
        model.setData(index, editor.text(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def eventFilter(self, editor, event):
        """편집기의 키 입력을 가로채서 커밋/내비게이션 처리."""
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(editor, event)

        key = event.key()
        modifiers = event.modifiers()

        NOHINT = QStyledItemDelegate.EndEditHint.NoHint
        REVERT = QStyledItemDelegate.EndEditHint.RevertModelCache

        def _commit_and_close(dc: int, dr: int) -> bool:
            # Save source index BEFORE commitData — setData may trigger beginResetModel
            # which clears currentIndex() before we can read it
            table = self.parent()
            src = table.currentIndex() if table is not None else None
            src_row = src.row() if src is not None and src.isValid() else -1
            src_col = src.column() if src is not None and src.isValid() else -1
            self._pending_navigate = (dc, dr, src_row, src_col)
            self.commitData.emit(editor)
            self.closeEditor.emit(editor, NOHINT)
            return True

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return _commit_and_close(0, 1)

        if key == Qt.Key.Key_Tab:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                return _commit_and_close(-1, 0)
            return _commit_and_close(1, 0)

        if key == Qt.Key.Key_Escape:
            self.closeEditor.emit(editor, REVERT)
            return True

        if key == Qt.Key.Key_Up:
            return _commit_and_close(0, -1)

        if key == Qt.Key.Key_Down:
            return _commit_and_close(0, 1)

        if key == Qt.Key.Key_Left and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == 0 and not editor.hasSelectedText():
                return _commit_and_close(-1, 0)

        if key == Qt.Key.Key_Right and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == len(editor.text()) and not editor.hasSelectedText():
                return _commit_and_close(1, 0)

        return super().eventFilter(editor, event)
