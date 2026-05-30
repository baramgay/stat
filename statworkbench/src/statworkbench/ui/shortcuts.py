"""Keyboard Shortcuts — 단축키 정의.

StatWorkbench 전역 단축키를 정의합니다.
"""


from PySide6.QtGui import QKeySequence


class ShortcutManager:
    """단축키 관리자."""

    # 파일 메뉴
    FILE_NEW = QKeySequence.New          # Ctrl+N
    FILE_OPEN = QKeySequence.Open        # Ctrl+O
    FILE_SAVE = QKeySequence.Save        # Ctrl+S
    FILE_SAVE_AS = QKeySequence.SaveAs   # Ctrl+Shift+S

    # 편집 메뉴
    EDIT_UNDO = QKeySequence.Undo        # Ctrl+Z
    EDIT_REDO = QKeySequence.Redo        # Ctrl+Shift+Z
    EDIT_CUT = QKeySequence.Cut          # Ctrl+X
    EDIT_COPY = QKeySequence.Copy        # Ctrl+C
    EDIT_PASTE = QKeySequence.Paste      # Ctrl+V
    EDIT_SELECT_ALL = QKeySequence.SelectAll  # Ctrl+A

    # 보기 메뉴
    VIEW_DATA = "Ctrl+1"
    VIEW_VARIABLE = "Ctrl+2"
    VIEW_SYNTAX = "Ctrl+3"
    VIEW_OUTPUT = "Ctrl+Shift+O"
    VIEW_THEME = "Ctrl+Shift+D"

    # 분석 메뉴
    ANALYZE_SCRIPT = "Ctrl+Shift+R"
    ANALYZE_FREQ = "Ctrl+Shift+F"
    ANALYZE_DESC = "Ctrl+Shift+D"
    ANALYZE_TTEST = "Ctrl+Shift+T"
    ANALYZE_REGRESSION = "Ctrl+Shift+L"

    # 차트 메뉴
    GRAPH_VISUAL = "Ctrl+Shift+V"

    # 데이터 메뉴
    DATA_SORT = "Ctrl+Shift+S"
    DATA_SELECT = "Ctrl+Shift+C"
    DATA_WEIGHT = "Ctrl+Shift+W"

    # 도움말
    HELP_ABOUT = "F1"

    @classmethod
    def get_all_shortcuts(cls) -> dict[str, str]:
        """모든 단축키 반환."""
        return {
            "파일 > 새로 만들기": "Ctrl+N",
            "파일 > 열기": "Ctrl+O",
            "파일 > 저장": "Ctrl+S",
            "파일 > 다른 이름으로 저장": "Ctrl+Shift+S",
            "편집 > 실행 취소": "Ctrl+Z",
            "편집 > 다시 실행": "Ctrl+Shift+Z",
            "편집 > 잘라내기": "Ctrl+X",
            "편집 > 복사": "Ctrl+C",
            "편집 > 붙여넣기": "Ctrl+V",
            "편집 > 모두 선택": "Ctrl+A",
            "보기 > 데이터 보기": "Ctrl+1",
            "보기 > 변수 보기": "Ctrl+2",
            "보기 > 구문 편집기": "Ctrl+3",
            "보기 > 결과 창": "Ctrl+Shift+O",
            "보기 > 다크 모드": "Ctrl+Shift+D",
            "분석 > 스크립트 실행": "Ctrl+Shift+R",
            "분석 > 빈도": "Ctrl+Shift+F",
            "분석 > 기술통계": "Ctrl+Shift+D",
            "분석 > T 검정": "Ctrl+Shift+T",
            "분석 > 회귀": "Ctrl+Shift+L",
            "차트 > 고급 시각화": "Ctrl+Shift+V",
        }
