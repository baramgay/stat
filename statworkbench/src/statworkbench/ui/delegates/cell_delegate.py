"""Cell Editor Delegate — SPSS 스타일 셀 편집기."""

from PySide6.QtWidgets import QStyledItemDelegate, QLineEdit
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent
from typing import Optional


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
        self._pending_navigate: Optional[tuple[int, int]] = None

    def set_initial_text(self, text: str) -> None:
        """DataView에서 printable 키 입력 시 호출. 다음 setEditorData에서 사용."""
        self._initial_text = text

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        if self._initial_text:
            # printable 키로 편집 시작: 기존값 지우고 첫 글자만
            editor.setText(self._initial_text)
            editor.setCursorPosition(len(self._initial_text))
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

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.commitData.emit(editor)
            self._pending_navigate = (0, 1)   # 아래
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
            return True

        if key == Qt.Key.Key_Tab:
            self.commitData.emit(editor)
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self._pending_navigate = (-1, 0)  # 왼쪽
            else:
                self._pending_navigate = (1, 0)   # 오른쪽
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
            return True

        if key == Qt.Key.Key_Escape:
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelData)
            return True

        if key == Qt.Key.Key_Up:
            self.commitData.emit(editor)
            self._pending_navigate = (0, -1)  # 위
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
            return True

        if key == Qt.Key.Key_Down:
            self.commitData.emit(editor)
            self._pending_navigate = (0, 1)   # 아래
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
            return True

        # 왼쪽: 커서가 맨 앞이면 셀 이동
        if key == Qt.Key.Key_Left and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == 0 and not editor.hasSelectedText():
                self.commitData.emit(editor)
                self._pending_navigate = (-1, 0)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
                return True

        # 오른쪽: 커서가 맨 뒤면 셀 이동
        if key == Qt.Key.Key_Right and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == len(editor.text()) and not editor.hasSelectedText():
                self.commitData.emit(editor)
                self._pending_navigate = (1, 0)
                self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.NoHint)
                return True

        return super().eventFilter(editor, event)
