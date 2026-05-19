"""Cluster Analysis Dialog — SPSS 스타일 군집분석 다이얼로그.

K-평균 군집과 계층적 군집을 탭으로 구분하여 제공합니다.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QDialogButtonBox,
    QPushButton, QCheckBox, QSpinBox, QFormLayout, QTabWidget,
    QWidget, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from statworkbench.core.dataset import Dataset
from statworkbench.ui.dialogs._dialog_helpers import (
    scale_vars, numeric_vars, display_label, measure_icon
)


class ClusterAnalysisDialog(QDialog):
    """SPSS 스타일 군집분석 다이얼로그.

    탭 구성:
    - K-평균 군집: 군집 수, 반복 횟수, 초기값 방법
    - 계층적 군집: 연결 방법, 거리 척도
    """

    analysis_requested = Signal(str, dict)

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self._dataset = dataset
        self.setWindowTitle("군집분석")
        self.setMinimumSize(640, 560)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 변수 선택 (공통)
        var_group = QGroupBox("분석 변수 선택")
        var_layout = QHBoxLayout(var_group)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("사용 가능한 변수:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        _vars = scale_vars(self._dataset) or numeric_vars(self._dataset)
        for var in _vars:
            icon = measure_icon(self._dataset, var)
            label = display_label(self._dataset, var)
            item = QListWidgetItem(f"{icon} {label}" if icon else label)
            item.setData(0x0100, var)
            self.available_list.addItem(item)
        left_layout.addWidget(self.available_list)
        var_layout.addLayout(left_layout)

        move_layout = QVBoxLayout()
        move_layout.addStretch()
        btn_add = QPushButton(">")
        btn_add.setFixedWidth(36)
        btn_add.clicked.connect(self._add_vars)
        btn_remove = QPushButton("<")
        btn_remove.setFixedWidth(36)
        btn_remove.clicked.connect(self._remove_vars)
        btn_add_all = QPushButton(">>")
        btn_add_all.setFixedWidth(36)
        btn_add_all.clicked.connect(self._add_all_vars)
        btn_remove_all = QPushButton("<<")
        btn_remove_all.setFixedWidth(36)
        btn_remove_all.clicked.connect(self._remove_all_vars)
        move_layout.addWidget(btn_add_all)
        move_layout.addWidget(btn_add)
        move_layout.addWidget(btn_remove)
        move_layout.addWidget(btn_remove_all)
        move_layout.addStretch()
        var_layout.addLayout(move_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("선택된 변수:"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        right_layout.addWidget(self.selected_list)
        var_layout.addLayout(right_layout)

        layout.addWidget(var_group)

        # 탭 위젯
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._build_kmeans_tab(), "K-평균 군집")
        self.tab_widget.addTab(self._build_hierarchical_tab(), "계층적 군집")
        layout.addWidget(self.tab_widget)

        # 버튼
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _build_kmeans_tab(self) -> QWidget:
        """K-평균 군집 탭 구성."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        form = QFormLayout()

        # 군집 수
        self.kmeans_k_spin = QSpinBox()
        self.kmeans_k_spin.setRange(2, 50)
        self.kmeans_k_spin.setValue(3)
        form.addRow("군집 수 (k):", self.kmeans_k_spin)

        # 최대 반복 횟수
        self.kmeans_iter_spin = QSpinBox()
        self.kmeans_iter_spin.setRange(1, 1000)
        self.kmeans_iter_spin.setValue(100)
        form.addRow("최대 반복 횟수:", self.kmeans_iter_spin)

        # 초기값 방법
        self.kmeans_init_combo = QComboBox()
        self.kmeans_init_combo.addItem("k-means++ (권장)", "k-means++")
        self.kmeans_init_combo.addItem("무작위 (Random)", "random")
        form.addRow("초기값 방법:", self.kmeans_init_combo)

        # 난수 시드
        self.kmeans_seed_spin = QSpinBox()
        self.kmeans_seed_spin.setRange(0, 99999)
        self.kmeans_seed_spin.setValue(42)
        form.addRow("난수 시드:", self.kmeans_seed_spin)

        layout.addLayout(form)

        # 출력 옵션
        output_group = QGroupBox("출력 옵션")
        output_layout = QVBoxLayout(output_group)
        self.kmeans_show_centers = QCheckBox("최종 군집 중심 표시")
        self.kmeans_show_centers.setChecked(True)
        self.kmeans_show_anova = QCheckBox("분산분석 표 표시")
        output_layout.addWidget(self.kmeans_show_centers)
        output_layout.addWidget(self.kmeans_show_anova)
        layout.addWidget(output_group)

        layout.addStretch()
        return widget

    def _build_hierarchical_tab(self) -> QWidget:
        """계층적 군집 탭 구성."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        form = QFormLayout()

        # 연결 방법
        self.hier_linkage_combo = QComboBox()
        self.hier_linkage_combo.addItem("Ward (와드)", "ward")
        self.hier_linkage_combo.addItem("완전 연결 (Complete)", "complete")
        self.hier_linkage_combo.addItem("평균 연결 (Average)", "average")
        self.hier_linkage_combo.addItem("단일 연결 (Single)", "single")
        form.addRow("연결 방법:", self.hier_linkage_combo)

        # 거리 척도
        self.hier_dist_combo = QComboBox()
        self.hier_dist_combo.addItem("유클리드 (Euclidean)", "euclidean")
        self.hier_dist_combo.addItem("맨해튼 (Manhattan)", "manhattan")
        self.hier_dist_combo.addItem("코사인 (Cosine)", "cosine")
        form.addRow("거리 척도:", self.hier_dist_combo)

        # 군집 수
        self.hier_k_spin = QSpinBox()
        self.hier_k_spin.setRange(2, 30)
        self.hier_k_spin.setValue(3)
        form.addRow("표시 군집 수:", self.hier_k_spin)

        layout.addLayout(form)

        # 출력 옵션
        output_group = QGroupBox("출력 옵션")
        output_layout = QVBoxLayout(output_group)
        self.hier_show_dendrogram = QCheckBox("덴드로그램 표시")
        self.hier_show_dendrogram.setChecked(True)
        self.hier_show_schedule = QCheckBox("병합 일정표 (Agglomeration Schedule)")
        output_layout.addWidget(self.hier_show_dendrogram)
        output_layout.addWidget(self.hier_show_schedule)
        layout.addWidget(output_group)

        layout.addStretch()
        return widget

    def _clone_item(self, source: QListWidgetItem) -> QListWidgetItem:
        new = QListWidgetItem(source.text())
        new.setData(0x0100, source.data(0x0100))
        return new

    def _add_vars(self):
        already = {self.selected_list.item(i).data(0x0100) for i in range(self.selected_list.count())}
        for item in self.available_list.selectedItems():
            if item.data(0x0100) not in already:
                self.selected_list.addItem(self._clone_item(item))

    def _remove_vars(self):
        for item in self.selected_list.selectedItems():
            self.selected_list.takeItem(self.selected_list.row(item))

    def _add_all_vars(self):
        already = {self.selected_list.item(i).data(0x0100) for i in range(self.selected_list.count())}
        for i in range(self.available_list.count()):
            src = self.available_list.item(i)
            if src.data(0x0100) not in already:
                self.selected_list.addItem(self._clone_item(src))

    def _remove_all_vars(self):
        self.selected_list.clear()

    def get_spec(self) -> dict:
        """현재 탭에 따라 분석 스펙 반환."""
        variables = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        current_tab = self.tab_widget.currentIndex()

        if current_tab == 0:
            # K-평균
            return {
                "analysis_id": "cluster_kmeans",
                "variables": variables,
                "method": "kmeans",
                "k": self.kmeans_k_spin.value(),
                "max_iter": self.kmeans_iter_spin.value(),
                "init": self.kmeans_init_combo.currentData(),
                "random_state": self.kmeans_seed_spin.value(),
                "display": {
                    "cluster_centers": self.kmeans_show_centers.isChecked(),
                    "anova_table": self.kmeans_show_anova.isChecked(),
                },
            }
        else:
            # 계층적
            return {
                "analysis_id": "cluster_hierarchical",
                "variables": variables,
                "method": "hierarchical",
                "linkage": self.hier_linkage_combo.currentData(),
                "distance": self.hier_dist_combo.currentData(),
                "k": self.hier_k_spin.value(),
                "display": {
                    "dendrogram": self.hier_show_dendrogram.isChecked(),
                    "agglomeration_schedule": self.hier_show_schedule.isChecked(),
                },
            }

    def _on_ok(self):
        variables = [
            self.selected_list.item(i).data(0x0100) or self.selected_list.item(i).text()
            for i in range(self.selected_list.count())
        ]
        if len(variables) < 2:
            QMessageBox.warning(self, "경고", "분석 변수를 2개 이상 선택하세요.")
            return

        spec = self.get_spec()
        analysis_id = spec["analysis_id"]
        self.analysis_requested.emit(analysis_id, spec)
        self.accept()
