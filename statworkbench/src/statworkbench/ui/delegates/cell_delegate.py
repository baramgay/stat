"""Cell Editor Delegate — SPSS 스타일 셀 편집기.

Features:
- 즉시 포커스 (딜레이 없음)
- 키 입력 즉시 반영
- Enter: 커밋 + 아래 이동
- Tab: 커밋 + 오른쪽 이동
- Esc: 취소
- 화살표: 커밋 + 이동
"""

from PySide6.QtWidgets import (
    QStyledItemDelegate, QLineEdit, QTableView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtGui import QKeyEvent


class CellEditor(QLineEdit):
    """SPSS 스타일 셀 편집기."""
    
    commit_requested = Signal()
    cancel_requested = Signal()
    navigate_requested = Signal(Qt.Key)  # 방향키
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrame(False)
        # 즉시 포커스
        self.setFocusPolicy(Qt.StrongFocus)
    
    def focusInEvent(self, event):
        """포커스 시 전체 선택."""
        super().focusInEvent(event)
        self.selectAll()
    
    def keyPressEvent(self, event):
        """키 입력 처리."""
        key = event.key()
        modifiers = event.modifiers()
        
        # Enter: 커밋
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.commit_requested.emit()
            return
        
        # Esc: 취소
        if key == Qt.Key_Escape:
            self.cancel_requested.emit()
            return
        
        # Tab: 커밋 + 이동
        if key == Qt.Key_Tab:
            self.commit_requested.emit()
            return
        
        # 화살표 키: 편집 중에도 이동
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            # 텍스트 커서가 끝/처음에 있을 때만 이동
            if key == Qt.Key_Left and self.cursorPosition() == 0:
                self.navigate_requested.emit(key)
                return
            if key == Qt.Key_Right and self.cursorPosition() == len(self.text()):
                self.navigate_requested.emit(key)
                return
            if key in (Qt.Key_Up, Qt.Key_Down):
                self.navigate_requested.emit(key)
                return
        
        super().keyPressEvent(event)


class CellDelegate(QStyledItemDelegate):
    """SPSS 스타일 셀 delegate."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def createEditor(self, parent, option, index):
        """편집기 생성."""
        editor = CellEditor(parent)
        return editor
    
    def setEditorData(self, editor, index):
        """편집기에 데이터 설정."""
        value = index.data(Qt.ItemDataRole.EditRole)
        if value is None or value == "":
            editor.setText("")
        else:
            editor.setText(str(value))
    
    def setModelData(self, editor, model, index):
        """모델에 데이터 저장."""
        text = editor.text()
        model.setData(index, text, Qt.ItemDataRole.EditRole)
    
    def updateEditorGeometry(self, editor, option, index):
        """편집기 위치 업데이트."""
        editor.setGeometry(option.rect)
