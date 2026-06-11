"""Syntax Editor — SPSS 스타일 구문 편집기.

SPSS Syntax Editor 기능:
- 구문 입력 및 편집
- 자동 완성 (명령어, 함수, 변수)
- 구문 강조 (Syntax Highlighting)
- 구문 실행 및 로그 출력
- 실행 히스토리 관리
"""


import pandas as pd
from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset

# SPSS Syntax 키워드
SPSS_COMMANDS = [
    "GET", "SAVE", "IMPORT", "EXPORT", "DATASET", "SELECT", "SORT", "SPLIT",
    "COMPUTE", "RECODE", "COUNT", "RANK", "AGGREGATE", "MATCH", "ADD",
    "FREQUENCIES", "DESCRIPTIVES", "CROSSTABS", "MEANS", "T-TEST",
    "ONEWAY", "GLM", "UNIANOVA", "MANOVA", "CORRELATIONS", "REGRESSION",
    "LOGISTIC", "NONPAR", "NPAR", "EXAMINE", "GRAPH", "CHART",
    "IF", "DO", "LOOP", "END", "EXECUTE", "LIST", "DISPLAY",
    "VARIABLE", "VALUE", "LABELS", "MISSING", "FORMATS", "WIDTH",
    "BEGIN", "PROGRAM", "PYTHON", "END", "PROGRAM",
]

SPSS_FUNCTIONS = [
    "ABS", "SQRT", "LN", "LOG10", "EXP", "ROUND", "TRUNC",
    "MEAN", "SUM", "SD", "VARIANCE", "MIN", "MAX",
    "RND", "MOD", "LAG", "LEAD", "ANY", "RANGE",
    "SYSMIS", "MISSING", "VALUE", "LABEL",
]

SPSS_SUBCOMMANDS = [
    "VARIABLES", "STATISTICS", "CHART", "FORMAT", "ORDER", "MISSING",
    "CELLS", "COUNT", "ROW", "COLUMN", "TOTAL", "LAYER",
    "GROUPS", "PAIRS", "PAIR", "WITH", "BY", "ON", "OFF",
    "SORT", "DESCRIPTIVES", "PLOT", "HISTOGRAM", "BARCHART",
]


class SPSSSyntaxHighlighter(QSyntaxHighlighter):
    """SPSS Syntax 구문 강조."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_formats()

    def _setup_formats(self):
        """구문 강조 형식 설정."""
        # 명령어 (파랑, 굵게)
        self.cmd_format = QTextCharFormat()
        self.cmd_format.setForeground(QColor("#0066CC"))
        self.cmd_format.setFontWeight(QFont.Weight.Bold)

        # 함수 (볼록, 초록)
        self.func_format = QTextCharFormat()
        self.func_format.setForeground(QColor("#009900"))
        self.func_format.setFontWeight(QFont.Weight.Bold)

        # 서브명령어 (볼록, 주황)
        self.sub_format = QTextCharFormat()
        self.sub_format.setForeground(QColor("#CC6600"))
        self.sub_format.setFontWeight(QFont.Weight.Bold)

        # 문자열 (빨강)
        self.str_format = QTextCharFormat()
        self.str_format.setForeground(QColor("#CC0000"))

        # 숫자 (볼록, 볼록)
        self.num_format = QTextCharFormat()
        self.num_format.setForeground(QColor("#990099"))

        # 주석 (회색, 기울임)
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#808080"))
        self.comment_format.setFontItalic(True)

        # 연산자 (볼록)
        self.op_format = QTextCharFormat()
        self.op_format.setForeground(QColor("#666666"))

    def highlightBlock(self, text: str):
        """블록 구문 강조."""
        import re

        upper_text = text.upper()

        # 주석 (* 또는 COMMENT)
        if text.strip().startswith("*") or text.strip().startswith("/*"):
            self.setFormat(0, len(text), self.comment_format)
            return

        # 명령어
        for cmd in SPSS_COMMANDS:
            pattern = r'\b' + cmd + r'\b'
            for match in re.finditer(pattern, upper_text):
                self.setFormat(match.start(), match.end() - match.start(), self.cmd_format)

        # 함수
        for func in SPSS_FUNCTIONS:
            pattern = r'\b' + func + r'\b'
            for match in re.finditer(pattern, upper_text):
                self.setFormat(match.start(), match.end() - match.start(), self.func_format)

        # 서브명령어
        for sub in SPSS_SUBCOMMANDS:
            pattern = r'\b' + sub + r'\b'
            for match in re.finditer(pattern, upper_text):
                self.setFormat(match.start(), match.end() - match.start(), self.sub_format)

        # 문자열 (작은따옴표)
        for match in re.finditer(r"'[^']*'", text):
            self.setFormat(match.start(), match.end() - match.start(), self.str_format)

        # 숫자
        for match in re.finditer(r'\b\d+\.?\d*\b', text):
            self.setFormat(match.start(), match.end() - match.start(), self.num_format)

        # 연산자
        for match in re.finditer(r'[\+\-\*/\=\(\)\,\;\.]', text):
            self.setFormat(match.start(), 1, self.op_format)


class SyntaxEditor(QWidget):
    """SPSS Syntax Editor 위젯."""

    syntax_executed = Signal(str, str)   # 명령어, 결과 텍스트
    analysis_ready  = Signal(object)     # AnalysisResult — 결과창 전달용

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset: Dataset | None = None
        self._history: list[str] = []
        self._history_index = -1
        self._max_history = 50
        self._setup_ui()
        self._connect_editor_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 도구 모음
        toolbar = QHBoxLayout()

        self.btn_run = QPushButton("▶ 실행")
        self.btn_run.setToolTip("선택한 구문을 실행합니다 (Ctrl+R)")
        self.btn_run.clicked.connect(self._run_selected)
        toolbar.addWidget(self.btn_run)

        self.btn_run_all = QPushButton("▶▶ 모두 실행")
        self.btn_run_all.setToolTip("모든 구문을 실행합니다")
        self.btn_run_all.clicked.connect(self._run_all)
        toolbar.addWidget(self.btn_run_all)

        toolbar.addSpacing(20)

        self.btn_undo = QPushButton("↩ 실행 취소")
        self.btn_undo.clicked.connect(self._undo)
        toolbar.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↪ 다시 실행")
        self.btn_redo.clicked.connect(self._redo)
        toolbar.addWidget(self.btn_redo)

        toolbar.addStretch()

        self.btn_save = QPushButton("💾 저장")
        self.btn_save.clicked.connect(self._save_syntax)
        toolbar.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂 불러오기")
        self.btn_load.clicked.connect(self._load_syntax)
        toolbar.addWidget(self.btn_load)

        layout.addLayout(toolbar)

        # 스플리터: 에디터 | 로그
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 에디터 영역
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        editor_layout.addWidget(QLabel("SPSS Syntax:"))

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "* SPSS 구문을 입력하세요.\n"
            "* 예: FREQUENCIES VARIABLES=age gender.\n"
            "* 예: DESCRIPTIVES VARIABLES=income /STATISTICS=MEAN STDDEV.\n"
            "* 예: COMPUTE new_var = (var1 + var2) / 2.\n"
            "* Ctrl+Space: 자동 완성"
        )

        # 고정폭 글꼴
        font = QFont("Consolas", 11)
        if not QFontDatabase.hasFamily("Consolas"):
            font = QFont("Courier New", 11)
        self.editor.setFont(font)

        # 구문 강조
        self.highlighter = SPSSSyntaxHighlighter(self.editor.document())

        # 자동 완성
        self._setup_completer()

        editor_layout.addWidget(self.editor)
        splitter.addWidget(editor_widget)

        # 로그 영역
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_layout.addWidget(QLabel("실행 로그:"))

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        self.log.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas, Courier New; font-size: 11px;"
        )
        log_layout.addWidget(self.log)

        splitter.addWidget(log_widget)
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)

        # 히스토리
        self._add_to_history("* NuriStat Syntax Editor")

    def _setup_completer(self):
        """자동 완성 설정."""
        self.completer = QCompleter(self)
        self.completer.setWidget(self.editor)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)

        # 키워드 목록
        keywords = SPSS_COMMANDS + SPSS_FUNCTIONS + SPSS_SUBCOMMANDS
        model = QStringListModel(keywords, self.completer)
        self.completer.setModel(model)

    def set_dataset(self, dataset: Dataset):
        """데이터셋 설정."""
        self._dataset = dataset
        # 변수명을 자동 완성에 추가
        if dataset and dataset.data is not None:
            keywords = (SPSS_COMMANDS + SPSS_FUNCTIONS + SPSS_SUBCOMMANDS +
                       list(dataset.data.columns))
            model = QStringListModel(keywords, self.completer)
            self.completer.setModel(model)

    def _run_selected(self):
        """선택한 구문 실행."""
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
        else:
            # 현재 줄 실행
            cursor.select(cursor.LineUnderCursor)
            text = cursor.selectedText()

        if text.strip():
            self._execute_syntax(text)

    def _run_all(self):
        """모든 구문 실행."""
        text = self.editor.toPlainText()
        if text.strip():
            self._execute_syntax(text)

    def _execute_syntax(self, syntax: str):
        """구문 실행."""
        self._log(f">>> {syntax.strip()}")

        try:
            result = self._parse_and_execute(syntax)
            if result:
                self._log(f"[성공] {result}")
                self.syntax_executed.emit(syntax, result)
            else:
                self._log("[완료]")
        except Exception as exc:
            self._log(f"[오류] {exc}")

        self._add_to_history(syntax)

    def _parse_and_execute(self, syntax: str) -> str:
        """구문 파싱 및 실행 — 분석 모듈 실제 호출."""
        import re
        if not self._dataset or self._dataset.data is None:
            raise ValueError("데이터셋이 없습니다")

        lines = syntax.strip().split('\n')
        results = []

        for line in lines:
            line = line.strip().rstrip('.')
            if not line or line.startswith('*'):
                continue
            upper = line.upper()

            # ── FREQUENCIES VARIABLES=v1 v2 ─────────────────────────────
            if upper.startswith('FREQUENCIES'):
                vars_list = self._extract_variables(line)
                if vars_list:
                    from nuristat.analysis.frequencies import run_analysis as _freq
                    spec = {"variables": {"target": vars_list}, "options": {"include_missing": False, "show_cumulative": True}}
                    result = _freq(self._dataset, spec)
                    self.analysis_ready.emit(result)
                    results.append(f"빈도분석: {len(vars_list)}개 변수")

            # ── DESCRIPTIVES VARIABLES=v1 v2 ────────────────────────────
            elif upper.startswith('DESCRIPTIVES'):
                vars_list = self._extract_variables(line)
                if vars_list:
                    from nuristat.analysis.frequencies import run_analysis as _desc
                    spec = {"variables": {"target": vars_list}, "options": {}}
                    try:
                        from nuristat.analysis.descriptives import run_analysis as _desc2
                        result = _desc2(self._dataset, spec)
                    except ImportError:
                        result = _desc(self._dataset, spec)
                    self.analysis_ready.emit(result)
                    results.append(f"기술통계: {len(vars_list)}개 변수")

            # ── T-TEST PAIRS=var1 WITH var2 (대응표본) ───────────────────
            elif upper.startswith('T-TEST') and re.search(r'\bPAIRS\b', line, re.IGNORECASE):
                m_pairs = re.search(r'PAIRS\s*=\s*(\w+)\s+WITH\s+(\w+)', line, re.IGNORECASE)
                if m_pairs:
                    v1, v2 = m_pairs.group(1), m_pairs.group(2)
                    from nuristat.analysis.ttests import run_analysis as _ttest
                    spec = {
                        "variables": {"paired": [v1, v2]},
                        "options": {},
                    }
                    result = _ttest(self._dataset, spec)
                    self.analysis_ready.emit(result)
                    results.append(f"대응표본 T검정: {v1} - {v2}")
                else:
                    results.append("T-TEST PAIRS: PAIRS=var1 WITH var2 형식 필요")

            # ── T-TEST GROUPS=var(v1 v2) /VARIABLES=dep (독립표본) ───────
            elif upper.startswith('T-TEST'):
                m_groups = re.search(r'GROUPS\s*=\s*(\w+)\s*\(\s*([^\)]+)\)', line, re.IGNORECASE)
                dep_vars = self._extract_variables(line)
                if m_groups and dep_vars:
                    grp_var = m_groups.group(1)
                    grp_vals_raw = m_groups.group(2).split()
                    grp_vals = []
                    for v in grp_vals_raw[:2]:
                        try:
                            grp_vals.append(int(v))
                        except ValueError:
                            try:
                                grp_vals.append(float(v))
                            except ValueError:
                                grp_vals.append(v)
                    from nuristat.analysis.ttests import run_analysis as _ttest
                    spec = {
                        "variables": {"dependent": dep_vars, "group": grp_var},
                        "options": {"group_values": grp_vals, "equal_var": "auto"},
                    }
                    result = _ttest(self._dataset, spec)
                    self.analysis_ready.emit(result)
                    results.append(f"독립표본 T검정: {dep_vars[0]} (집단: {grp_var})")
                else:
                    results.append("T-TEST: GROUPS=var(v1 v2) /VARIABLES=dep 형식 필요")

            # ── ONEWAY dep BY factor ─────────────────────────────────────
            elif upper.startswith('ONEWAY'):
                m = re.match(r'ONEWAY\s+(\w+)\s+BY\s+(\w+)', line, re.IGNORECASE)
                if m:
                    dep, fac = m.group(1), m.group(2)
                    from nuristat.analysis.anova import run_analysis as _anova
                    spec = {"variables": {"dependent": dep, "factor": fac}, "options": {"posthoc": "tukey"}}
                    result = _anova(self._dataset, spec)
                    self.analysis_ready.emit(result)
                    results.append(f"일원분산분석: {dep} by {fac}")
                else:
                    results.append("ONEWAY: ONEWAY dep BY factor 형식 필요")

            # ── COMPUTE new_var = expr ────────────────────────────────────
            elif upper.startswith('COMPUTE'):
                self._execute_compute(line)
                results.append("변수 계산 완료")

            # ── RECODE var (old=new) ... [INTO new_var] ──────────────────
            elif upper.startswith('RECODE'):
                msg = self._execute_recode(line)
                results.append(msg)

            # ── SELECT IF (condition) ────────────────────────────────────
            elif upper.startswith('SELECT IF') or upper.startswith('SELECT'):
                msg = self._execute_select_if(line)
                results.append(msg)

            else:
                results.append(f"알 수 없는 명령어: {line}")

        return '; '.join(results) if results else "완료"

    def _extract_variables(self, line: str) -> list[str]:
        """구문에서 변수 목록 추출."""
        import re
        # VARIABLES=var1 var2 var3.
        match = re.search(r'VARIABLES\s*=\s*([^/\.]+)', line, re.IGNORECASE)
        if match:
            return [v.strip() for v in match.group(1).split() if v.strip()]
        return []

    def _execute_compute(self, line: str):
        """COMPUTE 구문 실행."""
        import re
        # COMPUTE new_var = expression.
        match = re.match(r'COMPUTE\s+(\w+)\s*=\s*(.+)', line, re.IGNORECASE)
        if match:
            var_name = match.group(1)
            expression = match.group(2).rstrip('.')

            # 간단한 표현식 평가
            df = self._dataset.data
            safe_dict = {'df': df, 'pd': pd}
            for col in df.columns:
                safe_dict[col] = df[col]

            try:
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                if isinstance(result, pd.Series):
                    self._dataset.data[var_name] = result
                elif isinstance(result, (int, float, bool, str)):
                    # 스칼라 상수를 모든 행에 broadcast
                    self._dataset.data[var_name] = result
            except Exception:
                pass

    def _execute_recode(self, line: str) -> str:
        """RECODE var (old=new) ... [INTO new_var] 실행."""
        import re
        m_into = re.match(r'RECODE\s+(\w+)\s+(.+?)\s+INTO\s+(\w+)', line, re.IGNORECASE)
        m_plain = re.match(r'RECODE\s+(\w+)\s+(.+)', line, re.IGNORECASE)
        if not (m_into or m_plain):
            return "RECODE: 형식 오류"
        if m_into:
            src_var, rules_str, tgt_var = m_into.group(1), m_into.group(2), m_into.group(3)
        else:
            src_var = m_plain.group(1)
            rules_str = m_plain.group(2)
            tgt_var = src_var
        df = self._dataset.data
        if src_var not in df.columns:
            return f"RECODE: 변수 '{src_var}' 없음"
        pairs = re.findall(r'\(([^=]+)=([^\)]+)\)', rules_str)
        rules: dict = {}
        for old_s, new_s in pairs:
            old_s, new_s = old_s.strip(), new_s.strip()
            def _cast(s):
                try:
                    return int(s)
                except ValueError:
                    try:
                        return float(s)
                    except ValueError:
                        return s
            rules[_cast(old_s)] = _cast(new_s)
        if not rules:
            return "RECODE: 규칙 파싱 실패"
        df[tgt_var] = df[src_var].replace(rules)
        n = len(rules)
        return f"RECODE '{src_var}' → '{tgt_var}' ({n}개 규칙) 완료"

    def _execute_select_if(self, line: str) -> str:
        """SELECT IF (condition) 실행 — 조건에 맞는 케이스만 유지."""
        import re
        m = re.match(r'SELECT\s+IF\s*\((.+)\)', line, re.IGNORECASE)
        if not m:
            m = re.match(r'SELECT\s+IF\s+(.+)', line, re.IGNORECASE)
        if not m:
            return "SELECT IF: 조건 파싱 실패"
        condition = m.group(1).strip()
        df = self._dataset.data
        try:
            mask = df.eval(condition)
            before = len(df)
            self._dataset.data = df[mask].reset_index(drop=True)
            after = len(self._dataset.data)
            return f"SELECT IF: {before}→{after}개 케이스 (조건: {condition})"
        except Exception as exc:
            return f"SELECT IF 오류: {exc}"

    def _log(self, message: str):
        """로그 출력."""
        self.log.append(message)

    def _add_to_history(self, syntax: str):
        """히스토리 추가."""
        if syntax.strip():
            self._history.append(syntax.strip())
            self._history_index = len(self._history) - 1

    def _connect_editor_signals(self):
        """에디터 시그널 연결."""
        self.editor.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self):
        """텍스트 변경 시 히스토리 저장."""
        # 타이머를 사용하여 연속 입력 시 과도한 히스토리 저장 방지
        if not hasattr(self, '_change_timer'):
            from PySide6.QtCore import QTimer
            self._change_timer = QTimer(self)
            self._change_timer.setSingleShot(True)
            self._change_timer.timeout.connect(self._save_text_state)
        self._change_timer.start(1000)  # 1초 후 저장

    def _save_text_state(self):
        """현재 텍스트 상태를 히스토리에 저장."""
        text = self.editor.toPlainText()
        if not self._history or self._history[self._history_index] != text:
            # 현재 인덱스 이후의 히스토리 제거 (새 분기)
            self._history = self._history[:self._history_index + 1]
            self._history.append(text)
            # 최대 히스토리 개수 유지
            if len(self._history) > self._max_history:
                self._history.pop(0)
            else:
                self._history_index += 1

    def _undo(self):
        """실행 취소."""
        if self._history_index > 0:
            self._history_index -= 1
            self.editor.setPlainText(self._history[self._history_index])
            self._log("[실행 취소]")

    def _redo(self):
        """다시 실행."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.editor.setPlainText(self._history[self._history_index])
            self._log("[다시 실행]")

    def _save_syntax(self):
        """구문 저장."""
        path, _ = QFileDialog.getSaveFileName(
            self, "구문 저장", "", "SPSS Syntax (*.sps);;텍스트 파일 (*.txt)"
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.editor.toPlainText())
            self._log(f"[저장] {path}")

    def _load_syntax(self):
        """구문 불러오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "구문 불러오기", "", "SPSS Syntax (*.sps);;텍스트 파일 (*.txt)"
        )
        if path:
            with open(path, encoding='utf-8') as f:
                self.editor.setPlainText(f.read())
            self._log(f"[불러오기] {path}")
