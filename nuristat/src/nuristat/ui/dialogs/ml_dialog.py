"""ML Dialog — 기계학습 다이얼로그.

K-Means, 의사결정나무, 선형회귀 ML 기능을 제공합니다.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nuristat.analysis.ml_engine import (
    decision_tree_classifier,
    kmeans_clustering,
    linear_regression_ml,
)
from nuristat.core.dataset import Dataset
from nuristat.ui.analysis_worker import run_analysis_async


class MLDialog(QDialog):
    """기계학습 다이얼로그."""

    analysis_complete = Signal(str)

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset
        self._analysis_worker = None

        self.setWindowTitle("🤖 기계학습")
        self.setMinimumSize(700, 600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 알고리즘 선택
        algo_group = QGroupBox("📊 알고리즘 선택")
        algo_layout = QHBoxLayout(algo_group)

        algo_layout.addWidget(QLabel("방법:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("🔷 K-Means 군집화", "kmeans")
        self.algo_combo.addItem("🌳 의사결정나무 분류", "decision_tree")
        self.algo_combo.addItem("📈 선형 회귀 (ML)", "linear_regression")
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addStretch()

        layout.addWidget(algo_group)

        # 변수 선택
        var_group = QGroupBox("📋 변수 선택")
        var_layout = QVBoxLayout(var_group)

        # 특성 변수
        feat_layout = QHBoxLayout()
        feat_layout.addWidget(QLabel("특성 변수 (X):"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        self.feature_list.setMaximumHeight(120)
        for col in self.dataset.data.columns:
            item = QListWidgetItem(col)
            self.feature_list.addItem(item)
        feat_layout.addWidget(self.feature_list)
        var_layout.addLayout(feat_layout)

        # 목표 변수
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("목표 변수 (Y):"))
        self.target_combo = QComboBox()
        self.target_combo.addItem("(없음)")
        self.target_combo.addItems(self.dataset.data.columns)
        target_layout.addWidget(self.target_combo)
        var_layout.addLayout(target_layout)

        layout.addWidget(var_group)

        # 파라미터 설정
        param_group = QGroupBox("⚙️ 파라미터")
        param_layout = QHBoxLayout(param_group)

        # K-Means 파라미터
        self.kmeans_widget = QWidget()
        kmeans_layout = QHBoxLayout(self.kmeans_widget)
        kmeans_layout.addWidget(QLabel("군집 수 (K):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 20)
        self.k_spin.setValue(3)
        kmeans_layout.addWidget(self.k_spin)

        # 의사결정나무/회귀 파라미터
        self.ml_widget = QWidget()
        ml_layout = QHBoxLayout(self.ml_widget)
        ml_layout.addWidget(QLabel("테스트 비율:"))
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.1, 0.5)
        self.test_ratio_spin.setValue(0.2)
        self.test_ratio_spin.setSingleStep(0.05)
        ml_layout.addWidget(self.test_ratio_spin)

        param_layout.addWidget(self.kmeans_widget)
        param_layout.addWidget(self.ml_widget)
        param_layout.addStretch()

        layout.addWidget(param_group)

        # 실행 버튼
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("▶️ 실행")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #9b59b6; color: white; "
            "font-weight: bold; padding: 10px 30px; font-size: 13px; }"
        )
        self.run_btn.clicked.connect(self._run_analysis)
        btn_layout.addWidget(self.run_btn)

        self.close_btn = QPushButton("❌ 닫기")
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 결과 탭
        self.result_tabs = QTabWidget()

        # 요약 탭
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["지표", "값"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        summary_layout.addWidget(self.summary_table)
        self.result_tabs.addTab(self.summary_tab, "📊 요약")

        # 상세 탭
        self.detail_tab = QWidget()
        detail_layout = QVBoxLayout(self.detail_tab)
        self.detail_text = QLabel("실행 결과가 여기에 표시됩니다")
        self.detail_text.setWordWrap(True)
        self.detail_text.setStyleSheet("padding: 10px; background-color: #f8f9fa;")
        detail_layout.addWidget(self.detail_text)
        self.result_tabs.addTab(self.detail_tab, "📝 상세")

        layout.addWidget(self.result_tabs)

        # 초기 상태
        self._on_algo_changed()

    def _on_algo_changed(self) -> None:
        """알고리즘 변경 시."""
        algo = self.algo_combo.currentData()

        if algo == "kmeans":
            self.kmeans_widget.setVisible(True)
            self.ml_widget.setVisible(False)
            self.target_combo.setEnabled(False)
        else:
            self.kmeans_widget.setVisible(False)
            self.ml_widget.setVisible(True)
            self.target_combo.setEnabled(True)

    def _run_analysis(self) -> None:
        """분석 실행 (P2-1: 백그라운드 스레드에서 실행, GUI 스레드 블로킹 없음)."""
        algo = self.algo_combo.currentData()

        # 특성 변수 선택
        features = [item.text() for item in self.feature_list.selectedItems()]
        if not features:
            QMessageBox.warning(self, "경고", "특성 변수를 선택하세요")
            return

        target = self.target_combo.currentText()

        if algo in ("decision_tree", "linear_regression") and target == "(없음)":
            QMessageBox.warning(self, "경고", "목표 변수를 선택하세요")
            return

        spec = {
            "algo": algo,
            "features": features,
            "target": target,
            "n_clusters": self.k_spin.value(),
            "test_size": self.test_ratio_spin.value(),
        }

        self.run_btn.setEnabled(False)
        self.run_btn.setText("실행 중...")

        run_analysis_async(
            owner=self,
            run_fn=_compute_ml_result,
            dataset=self.dataset,
            spec=spec,
            on_result=self._on_analysis_result,
            on_error=self._on_analysis_error,
        )

    def _on_analysis_result(self, result: dict) -> None:
        algo = result["algo"]
        if algo == "kmeans":
            self._display_kmeans_result(result["data"])
        elif algo == "decision_tree":
            self._display_tree_result(result["data"])
        elif algo == "linear_regression":
            self._display_regression_result(result["data"])

        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 실행")

        self.analysis_complete.emit(f"ML 분석 완료: {algo}")

    def _on_analysis_error(self, message: str) -> None:
        QMessageBox.critical(self, "오류", f"분석 실패:\n{message}")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶️ 실행")

    def _display_kmeans_result(self, result: dict) -> None:
        """K-Means 결과 표시."""
        self.summary_table.setRowCount(4)

        items = [
            ("군집 수", str(len(result["centers"]))),
            ("반복 횟수", str(result["n_iter"])),
            ("관성 (Inertia)", f"{result['inertia']:.4f}"),
            ("실루엣 점수", f"{result['silhouette']:.4f}" if result["silhouette"] else "N/A"),
        ]

        for i, (metric, value) in enumerate(items):
            self.summary_table.setItem(i, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))

        self.detail_text.setText(
            f"<b>K-Means 군집화 결과</b><br><br>"
            f"• 군집 중심점: {len(result['centers'])}개<br>"
            f"• 각 군집 크기: {self._get_cluster_sizes(result['labels'])}<br>"
            f"• 수렴 여부: 반복 {result['n_iter']}회 후 수렴"
        )

    def _display_tree_result(self, result: dict) -> None:
        """의사결정나무 결과 표시."""
        self.summary_table.setRowCount(5)

        items = [
            ("정확도", f"{result['accuracy']:.4f}"),
            ("특성 수", str(result["n_features"])),
            ("학습 데이터", str(result["n_train"])),
            ("테스트 데이터", str(result["n_test"])),
            ("최대 깊이", str(result["max_depth"])),
        ]

        for i, (metric, value) in enumerate(items):
            self.summary_table.setItem(i, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))

        # 특성 중요도
        importance = "<br>".join([
            f"• {feat}: {imp:.4f}"
            for feat, imp in result["feature_importance"].items()
        ])

        self.detail_text.setText(
            f"<b>의사결정나무 분류 결과</b><br><br>"
            f"<b>특성 중요도:</b><br>{importance}"
        )

    def _display_regression_result(self, result: dict) -> None:
        """선형 회귀 결과 표시."""
        self.summary_table.setRowCount(5)

        items = [
            ("R² 점수", f"{result['r2_score']:.4f}"),
            ("MSE", f"{result['mse']:.4f}"),
            ("RMSE", f"{result['rmse']:.4f}"),
            ("학습 데이터", str(result["n_train"])),
            ("테스트 데이터", str(result["n_test"])),
        ]

        for i, (metric, value) in enumerate(items):
            self.summary_table.setItem(i, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))

        # 계수
        coeffs = "<br>".join([
            f"• {feat}: {coef:.4f}"
            for feat, coef in result["coefficients"].items()
        ])

        self.detail_text.setText(
            f"<b>선형 회귀 결과</b><br><br>"
            f"• 절편: {result['intercept']:.4f}<br><br>"
            f"<b>계수:</b><br>{coeffs}"
        )

    def _get_cluster_sizes(self, labels: list) -> str:
        """군집 크기 문자열 반환."""
        from collections import Counter
        sizes = Counter(labels)
        return ", ".join([f"군집 {k}: {v}개" for k, v in sorted(sizes.items())])


def _compute_ml_result(dataset: Dataset, spec: dict) -> dict:
    """ML 분석 계산 (백그라운드 스레드에서 실행, GUI 위젯 접근 없음)."""
    algo = spec["algo"]
    features = spec["features"]
    target = spec["target"]

    if algo == "kmeans":
        data = kmeans_clustering(
            dataset.data,
            features,
            n_clusters=spec["n_clusters"],
        )
    elif algo == "decision_tree":
        data = decision_tree_classifier(
            dataset.data,
            features,
            target,
            test_size=spec["test_size"],
        )
    elif algo == "linear_regression":
        data = linear_regression_ml(
            dataset.data,
            features,
            target,
            test_size=spec["test_size"],
        )
    else:
        raise ValueError(f"알 수 없는 알고리즘: {algo}")

    return {"algo": algo, "data": data}
