"""Script Runner Dialog — R/Python 스크립트 실행 다이얼로그.

가독성과 검증 절차를 중시한 UI 설계.
"""


from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from nuristat.analysis.python_bridge import PythonBridge
from nuristat.analysis.r_bridge import RBridge
from nuristat.core.dataset import Dataset


class ScriptRunnerThread(QThread):
    """스크립트 실행 스레드."""

    finished_signal = Signal(dict)
    progress_signal = Signal(str)

    def __init__(self, engine: str, script: str, dataset: Dataset | None) -> None:
        super().__init__()
        self.engine = engine
        self.script = script
        self.dataset = dataset
        self._python_bridge = PythonBridge()
        self._r_bridge = RBridge()

    def run(self) -> None:
        """스크립트 실행."""
        self.progress_signal.emit("스크립트 실행 중...")

        try:
            if self.engine == "Python":
                result = self._python_bridge.execute(self.script, self.dataset)
            elif self.engine == "R":
                result = self._r_bridge.execute(self.script, self.dataset)
            else:
                result = {
                    "success": False,
                    "error": f"지원하지 않는 엔진: {self.engine}",
                }

            self.finished_signal.emit(result)

        except Exception as exc:
            self.finished_signal.emit({
                "success": False,
                "error": str(exc),
            })


class ScriptRunnerDialog(QDialog):
    """R/Python 스크립트 실행 다이얼로그."""

    def __init__(self, dataset: Dataset | None, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self._thread: ScriptRunnerThread | None = None

        self.setWindowTitle("🔧 스크립트 실행기")
        self.setMinimumSize(900, 700)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 상단: 엔진 선택 및 정보
        top_layout = QHBoxLayout()

        top_layout.addWidget(QLabel("실행 엔진:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("🐍 Python", "Python")
        self.engine_combo.addItem("📊 R", "R")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        top_layout.addWidget(self.engine_combo)

        top_layout.addSpacing(20)

        # 엔진 상태
        self.engine_status = QLabel("✅ Python 사용 가능")
        self.engine_status.setStyleSheet("color: #2ca02c; font-weight: bold;")
        top_layout.addWidget(self.engine_status)

        top_layout.addStretch()

        # 예제 버튼
        self.btn_example = QPushButton("📋 예제 불러오기")
        self.btn_example.clicked.connect(self._load_example)
        top_layout.addWidget(self.btn_example)

        layout.addLayout(top_layout)

        # 스플리터: 스크립트 | 결과
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 스크립트 영역
        script_group = QGroupBox("📝 스크립트")
        script_layout = QVBoxLayout(script_group)

        self.script_editor = QPlainTextEdit()
        self.script_editor.setPlaceholderText(
            "# Python 스크립트를 입력하세요.\n"
            "# 사용 가능한 변수:\n"
            "#   df - 현재 데이터셋의 DataFrame\n"
            "#   pd - pandas\n"
            "#   np - numpy\n"
            "#   plt - matplotlib.pyplot\n"
            "#   save_plot(name) - 그림 저장 함수\n\n"
            "# 예제:\n"
            "summary = df.describe()\n"
            "print(summary)\n\n"
            "plt.figure(figsize=(10, 6))\n"
            "df.hist(bins=30)\n"
            "save_plot('histogram.png')\n"
        )
        font = QFont("Consolas", 11)
        if not QFontDatabase.hasFamily("Consolas"):
            font = QFont("Courier New", 11)
        self.script_editor.setFont(font)
        script_layout.addWidget(self.script_editor)

        # 실행 버튼
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ 실행")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #2ca02c; color: white; "
            "font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self._run_script)
        btn_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹ 중지")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_script)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()

        self.btn_save = QPushButton("💾 스크립트 저장")
        self.btn_save.clicked.connect(self._save_script)
        btn_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton("📂 스크립트 불러오기")
        self.btn_load.clicked.connect(self._load_script)
        btn_layout.addWidget(self.btn_load)

        script_layout.addLayout(btn_layout)
        splitter.addWidget(script_group)

        # 결과 영역
        result_group = QGroupBox("📊 실행 결과")
        result_layout = QVBoxLayout(result_group)

        # 진행 표시줄
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        result_layout.addWidget(self.progress_bar)

        # 결과 탭
        self.result_tabs = QTabWidget()

        # 출력 탭
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet(
            "background-color: #1a1a2e; color: #e8e8f0; "
            "font-family: Consolas, Courier New; font-size: 11px;"
        )
        self.result_tabs.addTab(self.output_text, "🖥️ 출력")

        # 그림 탭
        self.plot_text = QTextEdit()
        self.plot_text.setReadOnly(True)
        self.result_tabs.addTab(self.plot_text, "📈 그림")

        # 변수 탭
        self.vars_text = QTextEdit()
        self.vars_text.setReadOnly(True)
        self.result_tabs.addTab(self.vars_text, "🔢 변수")

        result_layout.addWidget(self.result_tabs)
        splitter.addWidget(result_group)

        splitter.setSizes([400, 300])
        layout.addWidget(splitter)

        # 하단: 검증 정보
        self.validation_label = QLabel("✅ 스크립트를 입력하고 실행하세요")
        self.validation_label.setStyleSheet(
            "color: #3a5068; padding: 6px; background-color: #f1f3f4; "
            "border-radius: 4px;"
        )
        layout.addWidget(self.validation_label)

    def _on_engine_changed(self) -> None:
        """엔진 변경 시."""
        engine = self.engine_combo.currentData()

        if engine == "R":
            bridge = RBridge()
            if bridge.is_available():
                self.engine_status.setText("✅ R 사용 가능")
                self.engine_status.setStyleSheet("color: #2ca02c; font-weight: bold;")
            else:
                self.engine_status.setText("❌ R 미설치 (R을 설치하세요)")
                self.engine_status.setStyleSheet("color: #d62728; font-weight: bold;")

            self.script_editor.setPlaceholderText(
                "# R 스크립트를 입력하세요.\n"
                "# 사용 가능한 변수:\n"
                "#   df - 현재 데이터셋의 데이터프레임\n\n"
                "# 예제:\n"
                "summary(df)\n"
                "hist(df$age)\n"
            )
        else:
            self.engine_status.setText("✅ Python 사용 가능")
            self.engine_status.setStyleSheet("color: #2ca02c; font-weight: bold;")

    def _run_script(self) -> None:
        """스크립트 실행."""
        script = self.script_editor.toPlainText().strip()
        if not script:
            QMessageBox.warning(self, "경고", "스크립트를 입력하세요")
            return

        engine = self.engine_combo.currentData()

        # UI 상태 변경
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.output_text.clear()
        self.plot_text.clear()
        self.vars_text.clear()

        # 스레드 실행
        self._thread = ScriptRunnerThread(engine, script, self.dataset)
        self._thread.progress_signal.connect(self._on_progress)
        self._thread.finished_signal.connect(self._on_finished)
        self._thread.start()

    def _stop_script(self) -> None:
        """스크립트 중지."""
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
            self._on_progress("실행이 중지되었습니다.")
            self._reset_ui()

    def _on_progress(self, message: str) -> None:
        """진행 상황 업데이트."""
        self.output_text.append(message)

    def _on_finished(self, result: dict) -> None:
        """실행 완료."""
        self._reset_ui()

        if result.get("success"):
            self.validation_label.setText("✅ 실행 완료")
            self.validation_label.setStyleSheet(
                "color: #2ca02c; padding: 6px; background-color: #e8f5e9; "
                "border-radius: 4px;"
            )

            # 출력 표시
            stdout = result.get("stdout", "")
            if stdout:
                self.output_text.append(f"[표준 출력]\n{stdout}")

            stderr = result.get("stderr", "")
            if stderr:
                self.output_text.append(f"[표준 오류]\n{stderr}")

            # 그림 표시
            plots = result.get("plots", [])
            if plots:
                self.plot_text.append(f"생성된 그림: {len(plots)}개\n")
                for plot_path in plots:
                    self.plot_text.append(f"  📊 {plot_path}\n")
            else:
                self.plot_text.append("생성된 그림이 없습니다.")

            # 변수 표시
            variables = result.get("variables", {})
            if variables:
                self.vars_text.append("생성된 변수:\n")
                for name, info in variables.items():
                    var_type = info.get("type", "unknown")
                    if var_type == "DataFrame":
                        shape = info.get("shape", (0, 0))
                        self.vars_text.append(f"  📋 {name}: DataFrame {shape}\n")
                    else:
                        value = info.get("value", "")
                        self.vars_text.append(f"  🔢 {name}: {var_type} = {value}\n")
            else:
                self.vars_text.append("생성된 변수가 없습니다.")

        else:
            error = result.get("error", "알 수 없는 오류")
            self.validation_label.setText(f"❌ 오류: {error}")
            self.validation_label.setStyleSheet(
                "color: #d62728; padding: 6px; background-color: #ffebee; "
                "border-radius: 4px;"
            )

            self.output_text.append(f"[오류]\n{error}")

            traceback_str = result.get("traceback", "")
            if traceback_str:
                self.output_text.append(f"\n[추적]\n{traceback_str}")

    def _reset_ui(self) -> None:
        """UI 상태 초기화."""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _load_example(self) -> None:
        """예제 스크립트 불러오기."""
        engine = self.engine_combo.currentData()

        if engine == "Python":
            example = '''# 데이터 요약
print("=== 데이터 요약 ===")
print(df.describe())

# 상관관계 분석
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    print("\\n=== 상관관계 행렬 ===")
    print(corr)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# 첫 번째 숫자형 변수의 히스토그램
if len(numeric_cols) > 0:
    df[numeric_cols[0]].hist(bins=30, edgecolor='black')
    plt.title(f'{numeric_cols[0]} 분포')
    plt.xlabel(numeric_cols[0])
    plt.ylabel('빈도')
    save_plot('histogram.png')
    print("\\n히스토그램이 생성되었습니다.")
'''
        else:
            example = '''# 데이터 요약
summary(df)

# 빈도분석
if(ncol(df) > 0) {
  table(df[,1])
}

# 히스토그램
if(ncol(df) > 0) {
  hist(df[,1], main="분포", xlab=names(df)[1])
}
'''

        self.script_editor.setPlainText(example)

    def _save_script(self) -> None:
        """스크립트 저장."""
        path, _ = QFileDialog.getSaveFileName(
            self, "스크립트 저장", "", "Python (*.py);;R (*.r);;모든 파일 (*.*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.script_editor.toPlainText())

    def _load_script(self) -> None:
        """스크립트 불러오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "스크립트 불러오기", "", "Python (*.py);;R (*.r);;모든 파일 (*.*)"
        )
        if path:
            with open(path, encoding="utf-8") as f:
                self.script_editor.setPlainText(f.read())

    def closeEvent(self, event) -> None:
        """종료 시 스레드 정리."""
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
        event.accept()
