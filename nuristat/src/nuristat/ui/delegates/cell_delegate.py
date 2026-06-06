"""Cell Editor Delegate — SPSS 스타일 셀 편집기."""


from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QComboBox, QLineEdit, QStyledItemDelegate


def _fmt_code(code: object) -> str:
    """값 라벨 코드 표시 — 정수형 실수(1.0)는 '1'로 깔끔하게."""
    if isinstance(code, float) and code.is_integer():
        return str(int(code))
    return str(code)


class CellDelegate(QStyledItemDelegate):
    """SPSS 스타일 셀 delegate.

    핵심 동작:
    - 값 라벨이 정의된 범주형 열 → 콤보박스(코드 = 라벨)로 빠른 선택(SPSS식).
      그 외 열 → 일반 QLineEdit (타이핑 즉시 편집).
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

    @staticmethod
    def _value_labels(index) -> dict | None:
        """해당 셀 열의 값 라벨 사전 반환 (모델이 제공하면)."""
        model = index.model()
        if model is not None and hasattr(model, "value_labels_for_col"):
            return model.value_labels_for_col(index.column())
        return None

    def createEditor(self, parent, option, index):
        self._editor_initialized = False  # 새 에디터마다 리셋
        labels = self._value_labels(index)
        if labels:
            # 범주형(값 라벨 보유) → '코드 = 라벨' 드롭다운. 자유 입력도 허용(라벨 외 코드).
            combo = QComboBox(parent)
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            combo.setFrame(False)
            for code, lab in labels.items():
                combo.addItem(f"{_fmt_code(code)} = {lab}", _fmt_code(code))
            return combo
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        if self._editor_initialized:
            return  # 동일 에디터에 두 번 호출되는 경우 무시
        self._editor_initialized = True

        if isinstance(editor, QComboBox):
            value = index.data(Qt.ItemDataRole.EditRole)
            code = _fmt_code(value) if value is not None and value != "" else ""
            i = editor.findData(code)
            if i >= 0:
                editor.setCurrentIndex(i)
            else:
                editor.setEditText(code)
            return

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

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            # 표시 텍스트가 항목과 일치하면 코드(itemData), 아니면 자유 입력값 그대로.
            text = editor.currentText()
            di = editor.findText(text)
            if di >= 0 and editor.itemData(di) is not None:
                code = editor.itemData(di)
            else:
                code = text
            model.setData(index, code, Qt.ItemDataRole.EditRole)
            return
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

        # 콤보박스(값 라벨)에서는 위/아래로 항목을 선택하므로 셀 이동으로 가로채지 않는다.
        is_combo = isinstance(editor, QComboBox)

        if key == Qt.Key.Key_Up and not is_combo:
            return _commit_and_close(0, -1)

        if key == Qt.Key.Key_Down and not is_combo:
            return _commit_and_close(0, 1)

        if key == Qt.Key.Key_Left and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == 0 and not editor.hasSelectedText():
                return _commit_and_close(-1, 0)

        if key == Qt.Key.Key_Right and isinstance(editor, QLineEdit):
            if editor.cursorPosition() == len(editor.text()) and not editor.hasSelectedText():
                return _commit_and_close(1, 0)

        return super().eventFilter(editor, event)
