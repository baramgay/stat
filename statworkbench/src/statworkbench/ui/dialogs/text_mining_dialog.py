"""텍스트 마이닝 대화상자."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from statworkbench.analysis.result import AnalysisResult
from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    all_vars,
    display_label,
    measure_icon,
)


class TextMiningDialog(QDialog):
    """텍스트 마이닝 — 단어 빈도, N-gram, TF-IDF, 워드클라우드."""

    analysis_run = Signal(AnalysisResult)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("텍스트 마이닝 — Text Mining")
        self.setMinimumSize(520, 560)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 텍스트 컬럼 선택
        col_group = QGroupBox("텍스트 컬럼 선택")
        col_layout = QVBoxLayout(col_group)
        self.col_combo = QComboBox()
        for var in all_vars(self._dataset):
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            self.col_combo.addItem(f"{icon} {label}" if icon else label, userData=var)
        col_layout.addWidget(self.col_combo)
        layout.addWidget(col_group)

        # 분석 옵션
        opt_group = QGroupBox("분석 옵션")
        opt_layout = QVBoxLayout(opt_group)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("언어:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["한국어 (ko)", "영어 (en)", "한국어+영어 (both)"])
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        opt_layout.addLayout(lang_row)

        topn_row = QHBoxLayout()
        topn_row.addWidget(QLabel("상위 단어 수 (Top-N):"))
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(5, 200)
        self.topn_spin.setValue(30)
        topn_row.addWidget(self.topn_spin)
        topn_row.addStretch()
        opt_layout.addLayout(topn_row)

        minlen_row = QHBoxLayout()
        minlen_row.addWidget(QLabel("최소 단어 길이:"))
        self.minlen_spin = QSpinBox()
        self.minlen_spin.setRange(1, 10)
        self.minlen_spin.setValue(2)
        minlen_row.addWidget(self.minlen_spin)
        minlen_row.addStretch()
        opt_layout.addLayout(minlen_row)

        ngram_row = QHBoxLayout()
        ngram_row.addWidget(QLabel("N-gram:"))
        self.ngram_combo = QComboBox()
        self.ngram_combo.addItems(["바이그램 (bigram)", "트라이그램 (trigram)", "없음 (none)"])
        ngram_row.addWidget(self.ngram_combo)
        ngram_row.addStretch()
        opt_layout.addLayout(ngram_row)

        self.chk_tfidf = QCheckBox("TF-IDF 분석 포함")
        self.chk_tfidf.setChecked(False)
        opt_layout.addWidget(self.chk_tfidf)

        self.chk_wordcloud = QCheckBox("워드클라우드 생성")
        self.chk_wordcloud.setChecked(True)
        opt_layout.addWidget(self.chk_wordcloud)

        wc_row = QHBoxLayout()
        wc_row.addWidget(QLabel("  최대 단어 수:"))
        self.wc_maxwords_spin = QSpinBox()
        self.wc_maxwords_spin.setRange(10, 500)
        self.wc_maxwords_spin.setValue(100)
        wc_row.addWidget(self.wc_maxwords_spin)
        wc_row.addStretch()
        opt_layout.addLayout(wc_row)
        layout.addWidget(opt_group)

        # 추가 불용어
        stop_group = QGroupBox("추가 불용어 (쉼표로 구분)")
        stop_layout = QVBoxLayout(stop_group)
        self.stopwords_edit = QLineEdit()
        self.stopwords_edit.setPlaceholderText("예: 때문에, 관련, the, and")
        stop_layout.addWidget(self.stopwords_edit)
        layout.addWidget(stop_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._run)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _run(self) -> None:
        text_col = self.col_combo.currentData()
        if not text_col:
            QMessageBox.warning(self, "경고", "텍스트 컬럼을 선택하세요.")
            return

        lang_map = {"한국어 (ko)": "ko", "영어 (en)": "en", "한국어+영어 (both)": "both"}
        language = lang_map.get(self.lang_combo.currentText(), "ko")

        ngram_map = {
            "바이그램 (bigram)": "bigram",
            "트라이그램 (trigram)": "trigram",
            "없음 (none)": "none",
        }
        ngram = ngram_map.get(self.ngram_combo.currentText(), "bigram")

        extra_sw = [
            s.strip() for s in self.stopwords_edit.text().split(",") if s.strip()
        ]

        try:
            from statworkbench.analysis.text_mining import run_analysis
            spec = {
                "variables": {"text_column": text_col},
                "options": {
                    "top_n": self.topn_spin.value(),
                    "min_word_len": self.minlen_spin.value(),
                    "ngram": ngram,
                    "tfidf": self.chk_tfidf.isChecked(),
                    "wordcloud": self.chk_wordcloud.isChecked(),
                    "stopwords": extra_sw,
                    "language": language,
                    "wc_max_words": self.wc_maxwords_spin.value(),
                },
            }
            result = run_analysis(self._dataset, spec)
            self.analysis_run.emit(result)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실패:\n{exc}")
