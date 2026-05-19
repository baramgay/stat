"""Main application window for StatWorkbench.

SPSS 스타일 메뉴 구조:
파일, 편집, 보기, 데이터, 변환, 분석, 차트, 유틸리티, 창, 도움말

레이아웃:
- 중앙: 데이터 뷰 (전체 화면)
- 하단: 탭 (데이터 보기, 변수 보기, 구문 편집기)
- 결과: 독립 창 (누적 출력)
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QLabel,
    QStatusBar,
    QToolBar,
    QMenuBar,
    QMenu,
    QFileDialog,
    QMessageBox,
    QApplication,
    QDialog,
    QProgressDialog,
)
from PySide6.QtCore import Qt, QSettings, QTimer, QSize
from PySide6.QtGui import QAction, QKeySequence, QFont, QFontDatabase
from typing import Optional

from statworkbench.core.project import Project
from statworkbench.core.dataset import Dataset
from statworkbench.core.settings import SettingsManager
from statworkbench.ui.data_view import DataView
from statworkbench.ui.variable_view import VariableView
from statworkbench.ui.output_window import OutputWindow
from statworkbench.ui.syntax_editor import SyntaxEditor
from statworkbench.ui.theme import ThemeManager, ThemeMode, get_application_stylesheet
from statworkbench.ui.icons import Icons
from statworkbench.ui.dialogs.manual_data_dialog import ManualDataDialog
from statworkbench.io.csv_reader import read_csv
from statworkbench.io.excel_reader import read_excel
from statworkbench.io.spss_reader import read_sav
from statworkbench.io.project_store import save_project, load_project


class MainWindow(QMainWindow):
    """StatWorkbench 메인 윈도우."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StatWorkbench")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # 프로젝트 상태
        self.project: Optional[Project] = None
        self.current_dataset: Optional[Dataset] = None

        # 테마 설정
        self._theme_manager = ThemeManager()
        self._dark_mode = False

        # 설정 관리자
        self._settings = SettingsManager()

        # 결과 창 (단일 인스턴스)
        self._output_window: Optional[OutputWindow] = None

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_theme()
        self._load_settings()

        # 빈 프로젝트로 시작
        self._new_project()

    def _setup_ui(self) -> None:
        """UI 구성.

        레이아웃:
        - 중앙: 데이터 뷰 (전체)
        - 하단: 탭바 (데이터/변수/구문)
        """
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 메인 스플리터 (수직: 데이터 뷰 + 하단 탭)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # 상단: 데이터 뷰 영역
        self.data_area = QWidget()
        data_layout = QVBoxLayout(self.data_area)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(0)

        # 데이터 뷰 (기본)
        self.data_view = DataView()
        self.data_view.dataset_changed.connect(self._on_dataset_changed)
        data_layout.addWidget(self.data_view)

        # 변수 뷰 (숨김)
        self.variable_view = VariableView()
        self.variable_view.dataset_changed.connect(self._on_dataset_changed)
        self.variable_view.hide()
        data_layout.addWidget(self.variable_view)

        # 구문 편집기 (숨김)
        self.syntax_editor = SyntaxEditor()
        self.syntax_editor.syntax_executed.connect(self._on_syntax_executed)
        self.syntax_editor.hide()
        data_layout.addWidget(self.syntax_editor)

        self.main_splitter.addWidget(self.data_area)

        # 하단: 탭바 (SPSS 스타일 South 탭)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.currentChanged.connect(self._on_tab_changed)

        # 탭 추가 (아이콘 + 텍스트)
        self.bottom_tabs.addTab(QWidget(), "🔢  데이터 보기")
        self.bottom_tabs.addTab(QWidget(), "📋  변수 보기")
        self.bottom_tabs.addTab(QWidget(), "📝  구문 편집기")

        self.main_splitter.addWidget(self.bottom_tabs)
        self.main_splitter.setSizes([750, 30])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)

        layout.addWidget(self.main_splitter)

    def _on_tab_changed(self, index: int) -> None:
        """하단 탭 변경 시 뷰 전환."""
        # 모든 뷰 숨기기
        self.data_view.hide()
        self.variable_view.hide()
        self.syntax_editor.hide()

        # 선택된 뷰 표시
        if index == 0:
            self.data_view.show()
        elif index == 1:
            self.variable_view.show()
        elif index == 2:
            self.syntax_editor.show()

    def _setup_menus(self) -> None:
        """SPSS 스타일 메뉴 구성."""
        menubar = self.menuBar()

        # 1. 파일 메뉴
        file_menu = menubar.addMenu("📁 파일(&F)")

        new_action = QAction("🆕 새로 만들기", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setToolTip("새 프로젝트를 만듭니다 (Ctrl+N)")
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        open_action = QAction("📂 열기...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setToolTip("기존 프로젝트를 엽니다 (Ctrl+O)")
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)

        save_action = QAction("💾 저장", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setToolTip("프로젝트를 저장합니다 (Ctrl+S)")
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("💾 다른 이름으로 저장...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.setToolTip("프로젝트를 다른 이름으로 저장합니다 (Ctrl+Shift+S)")
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # 가져오기 서브메뉴
        import_menu = file_menu.addMenu("📥 가져오기(&I)")

        import_csv_action = QAction("📄 CSV 파일...", self)
        import_csv_action.triggered.connect(self._import_csv)
        import_menu.addAction(import_csv_action)

        import_excel_action = QAction("📊 Excel 파일...", self)
        import_excel_action.triggered.connect(self._import_excel)
        import_menu.addAction(import_excel_action)

        import_sav_action = QAction("📋 SPSS 파일 (.sav)...", self)
        import_sav_action.triggered.connect(self._import_sav)
        import_menu.addAction(import_sav_action)

        import_clipboard_action = QAction("📋 클립보드...", self)
        import_clipboard_action.triggered.connect(self._import_clipboard)
        import_menu.addAction(import_clipboard_action)

        # 납비 서브메뉴
        export_menu = file_menu.addMenu("📤 납비(&X)")

        export_csv_action = QAction("📄 CSV 파일...", self)
        export_csv_action.triggered.connect(self._export_csv)
        export_menu.addAction(export_csv_action)

        export_excel_action = QAction("📊 Excel 파일...", self)
        export_excel_action.triggered.connect(self._export_excel)
        export_menu.addAction(export_excel_action)

        export_sav_action = QAction("📋 SPSS 파일 (.sav)...", self)
        export_sav_action.triggered.connect(self._export_sav)
        export_menu.addAction(export_sav_action)

        file_menu.addSeparator()

        exit_action = QAction("🚪 끝내기", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setToolTip("프로그램을 종료합니다 (Ctrl+Q)")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 2. 편집 메뉴
        edit_menu = menubar.addMenu("✏️ 편집(&E)")

        undo_action = QAction("↩️ 실행 취소", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.setToolTip("마지막 작업을 취소합니다 (Ctrl+Z)")
        undo_action.triggered.connect(self._edit_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("↪️ 다시 실행", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.setToolTip("취소한 작업을 다시 실행합니다 (Ctrl+Shift+Z)")
        redo_action.triggered.connect(self._edit_redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction("✂️ 잘라내기", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.setToolTip("선택한 내용을 잘라냅니다 (Ctrl+X)")
        cut_action.triggered.connect(self._edit_cut)
        edit_menu.addAction(cut_action)

        copy_action = QAction("📋 복사", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setToolTip("선택한 내용을 복사합니다 (Ctrl+C)")
        copy_action.triggered.connect(self._edit_copy)
        edit_menu.addAction(copy_action)

        paste_action = QAction("📋 붙여넣기", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.setToolTip("클립보드 내용을 붙여넣습니다 (Ctrl+V)")
        paste_action.triggered.connect(self._edit_paste)
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        select_all_action = QAction("☑️ 모두 선택", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.setToolTip("모든 내용을 선택합니다 (Ctrl+A)")
        select_all_action.triggered.connect(self._edit_select_all)
        edit_menu.addAction(select_all_action)

        # 3. 보기 메뉴
        view_menu = menubar.addMenu("👁️ 보기(&V)")

        self._theme_action = QAction("🌙 다크 모드", self)
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(False)
        self._theme_action.setShortcut("Ctrl+Shift+D")
        self._theme_action.setToolTip("다크 모드를 전환합니다 (Ctrl+Shift+D)")
        self._theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._theme_action)

        view_menu.addSeparator()

        data_view_action = QAction("🔢 데이터 보기", self)
        data_view_action.setShortcut("Ctrl+1")
        data_view_action.setToolTip("데이터 보기 탭으로 전환합니다 (Ctrl+1)")
        data_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(0))
        view_menu.addAction(data_view_action)

        var_view_action = QAction("📋 변수 보기", self)
        var_view_action.setShortcut("Ctrl+2")
        var_view_action.setToolTip("변수 보기 탭으로 전환합니다 (Ctrl+2)")
        var_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(1))
        view_menu.addAction(var_view_action)

        syntax_view_action = QAction("📝 구문 편집기", self)
        syntax_view_action.setShortcut("Ctrl+3")
        syntax_view_action.setToolTip("구문 편집기 탭으로 전환합니다 (Ctrl+3)")
        syntax_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(2))
        view_menu.addAction(syntax_view_action)

        view_menu.addSeparator()

        show_output_action = QAction("📊 결과 창 보기", self)
        show_output_action.setShortcut("Ctrl+Shift+O")
        show_output_action.setToolTip("결과 창을 표시합니다 (Ctrl+Shift+O)")
        show_output_action.triggered.connect(self._show_output_window)
        view_menu.addAction(show_output_action)

        # 4. 데이터 메뉴
        data_menu = menubar.addMenu("데이터(&D)")

        select_cases_action = QAction("🔍 케이스 선택...", self)
        select_cases_action.triggered.connect(self._open_select_cases)
        data_menu.addAction(select_cases_action)

        weight_cases_action = QAction("⚖️ 케이스 가중치...", self)
        weight_cases_action.triggered.connect(self._open_weight_cases)
        data_menu.addAction(weight_cases_action)

        data_menu.addSeparator()

        sort_cases_action = QAction("🔀 케이스 정렬...", self)
        sort_cases_action.triggered.connect(self._open_sort_dialog)
        data_menu.addAction(sort_cases_action)

        transpose_action = QAction("행렬 전치...", self)
        data_menu.addAction(transpose_action)

        merge_files_action = QAction("🔗 파일 병합...", self)
        merge_files_action.triggered.connect(self._open_merge_files)
        data_menu.addAction(merge_files_action)

        aggregate_action = QAction("📊 피벗 테이블...", self)
        aggregate_action.triggered.connect(self._open_pivot_table)
        data_menu.addAction(aggregate_action)

        # 5. 변환 메뉴
        transform_menu = menubar.addMenu("변환(&T)")

        compute_var_action = QAction("🔢 변수 계산...", self)
        compute_var_action.triggered.connect(self._open_compute_variable)
        transform_menu.addAction(compute_var_action)

        recode_action = QAction("🔄 변수 재코딩...", self)
        recode_action.triggered.connect(self._open_recode)
        transform_menu.addAction(recode_action)

        visual_binning_action = QAction("📊 시각적 구간화...", self)
        visual_binning_action.triggered.connect(self._open_binning)
        transform_menu.addAction(visual_binning_action)

        rank_cases_action = QAction("🏆 순위 계산...", self)
        rank_cases_action.triggered.connect(self._open_rank)
        transform_menu.addAction(rank_cases_action)

        # 6. 분석 메뉴
        analyze_menu = menubar.addMenu("📊 분석(&A)")

        # 스크립트 실행
        script_action = QAction("🔧 스크립트 실행...", self)
        script_action.setShortcut("Ctrl+Shift+R")
        script_action.triggered.connect(self._open_script_runner)
        analyze_menu.addAction(script_action)

        analyze_menu.addSeparator()

        # 기술통계
        desc_menu = analyze_menu.addMenu("📈 기술통계(&R)")

        freq_action = QAction("📊 빈도...", self)
        freq_action.setShortcut("Ctrl+Shift+F")
        freq_action.triggered.connect(self._run_frequencies)
        desc_menu.addAction(freq_action)

        desc_action = QAction("📈 기술통계량...", self)
        desc_action.setShortcut("Ctrl+Shift+D")
        desc_action.triggered.connect(self._run_descriptives)
        desc_menu.addAction(desc_action)

        explore_action = QAction("🔍 탐색...", self)
        desc_menu.addAction(explore_action)

        crosstab_action = QAction("📊 교차분석...", self)
        crosstab_action.triggered.connect(self._run_crosstabs)
        desc_menu.addAction(crosstab_action)

        # 평균 비교
        compare_menu = analyze_menu.addMenu("🔄 평균 비교(&M)")

        means_action = QAction("📊 평균...", self)
        compare_menu.addAction(means_action)

        one_sample_t_action = QAction("1️⃣ 단일표본 T 검정...", self)
        one_sample_t_action.triggered.connect(self._run_one_sample_ttest)
        compare_menu.addAction(one_sample_t_action)

        ind_t_action = QAction("2️⃣ 독립표본 T 검정...", self)
        ind_t_action.setShortcut("Ctrl+Shift+T")
        ind_t_action.triggered.connect(self._run_independent_ttest)
        compare_menu.addAction(ind_t_action)

        paired_t_action = QAction("🔗 대응표본 T 검정...", self)
        paired_t_action.triggered.connect(self._run_paired_ttest)
        compare_menu.addAction(paired_t_action)

        anova_action = QAction("📊 일원분산분석...", self)
        anova_action.setShortcut("Ctrl+Shift+A")
        anova_action.triggered.connect(self._run_anova)
        compare_menu.addAction(anova_action)

        # 상관/회귀
        correlate_menu = analyze_menu.addMenu("🔗 상관(&C)")

        bivariate_corr_action = QAction("🔗 상관분석...", self)
        bivariate_corr_action.triggered.connect(self._run_correlation)
        correlate_menu.addAction(bivariate_corr_action)

        partial_corr_action = QAction("🔗 편상관...", self)
        correlate_menu.addAction(partial_corr_action)

        regression_menu = analyze_menu.addMenu("📈 회귀(&R)")

        linear_action = QAction("📈 선형...", self)
        linear_action.setShortcut("Ctrl+Shift+L")
        linear_action.triggered.connect(self._run_regression)
        regression_menu.addAction(linear_action)

        logistic_action = QAction("📊 로지스틱...", self)
        logistic_action.triggered.connect(self._run_logistic_regression)
        regression_menu.addAction(logistic_action)

        # 차원 축소
        dim_reduce_menu = analyze_menu.addMenu("📉 차원 축소(&D)")
        factor_action = QAction("📉 요인분석...", self)
        factor_action.triggered.connect(self._run_factor_analysis)
        dim_reduce_menu.addAction(factor_action)

        # 군집
        cluster_menu = analyze_menu.addMenu("🔵 군집(&K)")
        kmeans_action = QAction("🔵 K-평균 군집...", self)
        kmeans_action.triggered.connect(self._run_cluster_analysis)
        cluster_menu.addAction(kmeans_action)
        hierarchical_action = QAction("🔵 계층적 군집...", self)
        hierarchical_action.triggered.connect(self._run_cluster_analysis)
        cluster_menu.addAction(hierarchical_action)

        # 생존
        survival_menu = analyze_menu.addMenu("생존분석(&S)")
        km_action = QAction("📈 Kaplan-Meier...", self)
        km_action.triggered.connect(self._run_survival_analysis)
        survival_menu.addAction(km_action)

        # 판별
        classify_menu = analyze_menu.addMenu("🔷 판별분석(&I)")
        lda_action = QAction("🔷 판별분석...", self)
        lda_action.triggered.connect(self._run_discriminant_analysis)
        classify_menu.addAction(lda_action)

        # 비모수 검정
        nonparam_menu = analyze_menu.addMenu("🧪 비모수 검정(&N)")

        nonparam_action = QAction("🧪 비모수 검정...", self)
        nonparam_action.triggered.connect(self._run_nonparametric)
        nonparam_menu.addAction(nonparam_action)

        analyze_menu.addSeparator()

        # 기계학습
        ml_action = QAction("🤖 기계학습...", self)
        ml_action.triggered.connect(self._open_ml_dialog)
        analyze_menu.addAction(ml_action)

        # 7. 차트 메뉴
        graphs_menu = menubar.addMenu("차트(&G)")

        chart_builder_action = QAction("📊 고급 시각화...", self)
        chart_builder_action.setShortcut("Ctrl+Shift+V")
        chart_builder_action.triggered.connect(self._open_visualization)
        graphs_menu.addAction(chart_builder_action)

        graphs_menu.addSeparator()

        chart_builder_legacy_action = QAction("차트 빌더...", self)
        chart_builder_legacy_action.triggered.connect(self._open_chart_builder)
        graphs_menu.addAction(chart_builder_legacy_action)

        legacy_graphs_menu = graphs_menu.addMenu("기존 대화상자(&L)")

        bar_action = QAction("막대...", self)
        legacy_graphs_menu.addAction(bar_action)

        line_action = QAction("선...", self)
        legacy_graphs_menu.addAction(line_action)

        scatter_action = QAction("산점도...", self)
        legacy_graphs_menu.addAction(scatter_action)

        histogram_action = QAction("히스토그램...", self)
        legacy_graphs_menu.addAction(histogram_action)

        boxplot_action = QAction("상자 그림...", self)
        legacy_graphs_menu.addAction(boxplot_action)

        # 8. 유틸리티 메뉴
        utilities_menu = menubar.addMenu("유틸리티(&U)")

        data_quality_action = QAction("🔍 데이터 품질 진단...", self)
        data_quality_action.triggered.connect(self._open_data_quality)
        utilities_menu.addAction(data_quality_action)

        report_action = QAction("📄 보고서 생성...", self)
        report_action.triggered.connect(self._open_report_generator)
        utilities_menu.addAction(report_action)

        utilities_menu.addSeparator()

        var_info_action = QAction("변수 정보...", self)
        utilities_menu.addAction(var_info_action)

        file_info_action = QAction("파일 정보...", self)
        utilities_menu.addAction(file_info_action)

        define_sets_action = QAction("집합 정의...", self)
        utilities_menu.addAction(define_sets_action)

        # 9. 창 메뉴
        window_menu = menubar.addMenu("창(&W)")

        # 10. 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")

        about_action = QAction("프로그램 정보", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """도구 모음 설정."""
        toolbar = QToolBar("메인 도구 모음")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 새로 만들기
        new_btn = QAction("🆕 새로 만들기", self)
        new_btn.triggered.connect(self._new_project)
        toolbar.addAction(new_btn)

        # 열기
        open_btn = QAction("📂 열기", self)
        open_btn.triggered.connect(self._open_project)
        toolbar.addAction(open_btn)

        # 저장
        save_btn = QAction("💾 저장", self)
        save_btn.triggered.connect(self._save_project)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        # 데이터 보기
        data_view_btn = QAction("🔢 데이터", self)
        data_view_btn.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(0))
        toolbar.addAction(data_view_btn)

        # 변수 보기
        var_view_btn = QAction("📋 변수", self)
        var_view_btn.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(1))
        toolbar.addAction(var_view_btn)

        toolbar.addSeparator()

        # 분석 버튼들
        freq_btn = QAction("📊 빈도", self)
        freq_btn.triggered.connect(self._run_frequencies)
        toolbar.addAction(freq_btn)

        desc_btn = QAction("📈 기술통계", self)
        desc_btn.triggered.connect(self._run_descriptives)
        toolbar.addAction(desc_btn)

        ttest_btn = QAction("📉 T 검정", self)
        ttest_btn.triggered.connect(self._run_independent_ttest)
        toolbar.addAction(ttest_btn)

        reg_btn = QAction("📉 회귀", self)
        reg_btn.triggered.connect(self._run_regression)
        toolbar.addAction(reg_btn)

    def _setup_statusbar(self) -> None:
        """상태 표시줄 설정."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 상태 메시지
        self.statusbar.showMessage("준비됨")

        # 데이터셋 정보
        self.dataset_info_label = QLabel("데이터셋: 없음")
        self.statusbar.addPermanentWidget(self.dataset_info_label)

    def _update_statusbar(self) -> None:
        """상태 표시줄 업데이트."""
        if self.current_dataset:
            rows = len(self.current_dataset.data)
            cols = len(self.current_dataset.data.columns)
            self.dataset_info_label.setText(
                f"데이터셋: {self.current_dataset.name}  |  "
                f"행: {rows:,}  |  열: {cols}"
            )
        else:
            self.dataset_info_label.setText("데이터셋: 없음")

    def _apply_theme(self) -> None:
        """테마 적용."""
        mode = ThemeMode.DARK if self._dark_mode else ThemeMode.LIGHT
        stylesheet = get_application_stylesheet(mode)
        QApplication.instance().setStyleSheet(stylesheet)

    def _toggle_theme(self) -> None:
        """테마 전환."""
        self._dark_mode = not self._dark_mode
        self._theme_action.setChecked(self._dark_mode)
        self._apply_theme()

    def _load_settings(self) -> None:
        """저장된 설정 불러오기."""
        # 윈도우 크기/위치
        size = self._settings.load_window_size()
        pos = self._settings.load_window_position()
        maximized = self._settings.load_window_maximized()

        self.resize(size)
        self.move(pos)
        if maximized:
            self.showMaximized()

        # 테마
        dark_mode = self._settings.load_theme()
        if dark_mode != self._dark_mode:
            self._toggle_theme()

    def _save_settings(self) -> None:
        """현재 설정 저장."""
        # 윈도우 상태
        if self.isMaximized():
            self._settings.save_window_maximized(True)
        else:
            self._settings.save_window_geometry(self.size(), self.pos())

        # 테마
        self._settings.save_theme(self._dark_mode)

    def closeEvent(self, event) -> None:
        """종료 시 설정 저장."""
        self._save_settings()

        # 결과 창도 닫기
        if self._output_window is not None:
            self._output_window.close()
            self._output_window = None

        if self.project and self.project.is_dirty():
            reply = QMessageBox.question(
                self,
                "StatWorkbench",
                "저장하지 않은 변경 사항이 있습니다. 저장하시겠습니까?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            if reply == QMessageBox.StandardButton.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ── 결과 창 관리 ────────────────────────────────────────────────────────

    def _show_output_window(self) -> None:
        """결과 창 표시 (단일 인스턴스)."""
        if self._output_window is None or not self._output_window.isVisible():
            self._output_window = OutputWindow(self)
            self._output_window.setWindowTitle("📊 StatWorkbench 결과")
            self._output_window.resize(800, 600)
            self._output_window.show()
        else:
            self._output_window.raise_()
            self._output_window.activateWindow()

    def _get_output(self) -> object:
        """결과 출력 대상 반환."""
        if self._output_window is None or not self._output_window.isVisible():
            self._show_output_window()
        return self._output_window

    # ── 프로젝트 관리 ───────────────────────────────────────────────────────

    def _open_sort_dialog(self) -> None:
        """정렬 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.sort_dialog import SortDialog
        dialog = SortDialog(self.current_dataset, self)
        dialog.sort_applied.connect(self._on_sort_applied)
        dialog.exec()

    def _on_sort_applied(self) -> None:
        """정렬 적용 시."""
        self._on_dataset_changed(self.current_dataset)
        output = self._get_output()
        output.add_output("🔀 데이터 정렬이 적용되었습니다.", "success")

    def _export_sav(self) -> None:
        """SPSS .sav 납비."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "납비 데이터가 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "SPSS 저장", "", "SPSS 파일 (*.sav)"
        )
        if path:
            try:
                from statworkbench.io.spss_writer import write_sav
                write_sav(self.current_dataset, path)
                self.statusbar.showMessage(f"SPSS 저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"SPSS 저장 실패:\n{exc}")

    def _open_ml_dialog(self) -> None:
        """기계학습 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.ml_dialog import MLDialog
        dialog = MLDialog(self.current_dataset, self)
        dialog.analysis_complete.connect(
            lambda msg: self._get_output().add_output(msg, "success")
        )
        dialog.exec()

    def _new_project(self) -> None:
        """새 프로젝트."""
        import pandas as pd

        self.project = Project(name="Untitled")

        # 기본 빈 데이터셋
        df = pd.DataFrame()
        self.current_dataset = Dataset(name="DataSet1", data=df)
        self.project.add_dataset(self.current_dataset)

        self.data_view.set_dataset(self.current_dataset)
        self.variable_view.set_dataset(self.current_dataset)
        self.syntax_editor.set_dataset(self.current_dataset)

        self._update_statusbar()
        self.statusbar.showMessage("새 프로젝트가 생성되었습니다.")

    def _open_project(self) -> None:
        """프로젝트 열기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "StatWorkbench (*.swb);;모든 파일 (*.*)"
        )
        if path:
            try:
                self.project = load_project(path)
                if self.project.datasets:
                    self.current_dataset = self.project.datasets[0]
                    self.data_view.set_dataset(self.current_dataset)
                    self.variable_view.set_dataset(self.current_dataset)
                    self.syntax_editor.set_dataset(self.current_dataset)
                self._update_statusbar()
                self.statusbar.showMessage(f"프로젝트를 열었습니다: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"프로젝트 열기 실패:\n{exc}")

    def _save_project(self) -> None:
        """프로젝트 저장."""
        if self.project is None:
            return

        if self.project.file_path:
            try:
                save_project(self.project, self.project.file_path)
                self.project.clear_dirty()
                self.statusbar.showMessage(f"저장되었습니다: {self.project.file_path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        """다른 이름으로 저장."""
        if self.project is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "프로젝트 저장", "", "StatWorkbench (*.swb)"
        )
        if path:
            if not path.endswith(".swb"):
                path += ".swb"
            try:
                save_project(self.project, path)
                self.project.file_path = path
                self.project.clear_dirty()
                self.statusbar.showMessage(f"저장되었습니다: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")

    def _on_dataset_changed(self) -> None:
        """데이터셋 변경 시 호출됩니다."""
        if self.current_dataset is not None:
            # Variable View ↔ Data View 양방향 동기화
            if hasattr(self, 'variable_view') and self.variable_view:
                self.variable_view.set_dataset(self.current_dataset)
            # Data View 메타데이터 반영 (decimals, measure 등 변경 즉시 적용)
            if hasattr(self, 'data_view') and self.data_view:
                model = self.data_view.model() if hasattr(self.data_view, 'model') else None
                if model is not None and hasattr(model, 'layoutChanged'):
                    model.layoutChanged.emit()
            # 구문 편집기에도 반영
            if hasattr(self, 'syntax_editor') and self.syntax_editor:
                self.syntax_editor.set_dataset(self.current_dataset)
            # 프로젝트를 더티 상태로 표시
            if self.project is not None:
                self.project.mark_dirty()

        self._update_statusbar()

    def _on_syntax_executed(self, code: str) -> None:
        """구문 실행 완료 시."""
        self.statusbar.showMessage("구문이 실행되었습니다.")

    # ── 파일 가져오기/납비 ────────────────────────────────────────────────

    def _import_csv(self) -> None:
        """CSV 가져오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 열기", "", "CSV 파일 (*.csv)"
        )
        if path:
            try:
                dataset = read_csv(path)
                self.current_dataset = dataset
                self.data_view.set_dataset(dataset)
                self.variable_view.set_dataset(dataset)
                self.syntax_editor.set_dataset(dataset)
                self._update_statusbar()
                self.statusbar.showMessage(f"CSV 가져오기 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"CSV 가져오기 실패:\n{exc}")

    def _import_excel(self) -> None:
        """Excel 가져오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel 파일 열기", "", "Excel 파일 (*.xlsx *.xls)"
        )
        if path:
            try:
                dataset = read_excel(path)
                self.current_dataset = dataset
                self._on_dataset_changed(dataset)
                self.statusbar.showMessage(f"Excel 가져오기 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"Excel 가져오기 실패:\n{exc}")

    def _import_sav(self) -> None:
        """SPSS .sav 가져오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "SPSS 파일 열기", "", "SPSS 파일 (*.sav)"
        )
        if path:
            try:
                dataset = read_sav(path)
                self.current_dataset = dataset
                self._on_dataset_changed(dataset)
                self.statusbar.showMessage(f"SPSS 가져오기 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"SPSS 가져오기 실패:\n{exc}")

    def _import_clipboard(self) -> None:
        """클립보드 가져오기."""
        import pandas as pd

        try:
            df = pd.read_clipboard()
            if df.empty:
                QMessageBox.warning(self, "경고", "클립보드에 데이터가 없습니다.")
                return
            dataset = Dataset(name="Clipboard", data=df)
            self.current_dataset = dataset
            self._on_dataset_changed(dataset)
            self.statusbar.showMessage("클립보드 가져오기 완료")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"클립보드 가져오기 실패:\n{exc}")

    def _export_csv(self) -> None:
        """CSV 납비."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "납비 데이터가 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 저장", "", "CSV 파일 (*.csv)"
        )
        if path:
            try:
                self.current_dataset.data.to_csv(path, index=False)
                self.statusbar.showMessage(f"CSV 저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"CSV 저장 실패:\n{exc}")

    def _export_excel(self) -> None:
        """Excel 납비."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "납비 데이터가 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Excel 저장", "", "Excel 파일 (*.xlsx)"
        )
        if path:
            try:
                self.current_dataset.data.to_excel(path, index=False)
                self.statusbar.showMessage(f"Excel 저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"Excel 저장 실패:\n{exc}")

    # ── 편집 메뉴 ───────────────────────────────────────────────────────────

    def _edit_undo(self) -> None:
        """실행 취소."""
        current_widget = self.data_area.layout().currentWidget()
        if current_widget == self.syntax_editor:
            self.syntax_editor.undo()

    def _edit_redo(self) -> None:
        """다시 실행."""
        current_widget = self.data_area.layout().currentWidget()
        if current_widget == self.syntax_editor:
            self.syntax_editor.redo()

    def _edit_cut(self) -> None:
        """잘라내기."""
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'cut'):
            current_widget.cut()

    def _edit_copy(self) -> None:
        """복사."""
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'copy'):
            current_widget.copy()

    def _edit_paste(self) -> None:
        """붙여넣기."""
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'paste'):
            current_widget.paste()

    def _edit_select_all(self) -> None:
        """모두 선택."""
        current_widget = self.data_area.layout().currentWidget()
        if current_widget == self.data_view:
            self.data_view.table.selectAll()
        elif current_widget == self.syntax_editor:
            self.syntax_editor.selectAll()

    # ── 데이터 메뉴 ─────────────────────────────────────────────────────────

    def _open_select_cases(self) -> None:
        """케이스 선택 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.select_cases_dialog import SelectCasesDialog
        dialog = SelectCasesDialog(self.current_dataset, self)
        dialog.cases_selected.connect(self._on_cases_selected)
        dialog.exec()

    def _on_cases_selected(self, selection_type: str, condition: object) -> None:
        """케이스 선택 완료 시."""
        output = self._get_output()
        output.add_output(f"🔍 케이스 선택 적용: {selection_type}", "success")

    def _open_weight_cases(self) -> None:
        """가중치 적용 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.weight_cases_dialog import WeightCasesDialog
        dialog = WeightCasesDialog(self.current_dataset, self)
        dialog.weight_applied.connect(self._on_weight_applied)
        dialog.weight_cleared.connect(self._on_weight_cleared)
        dialog.exec()

    def _on_weight_applied(self, weight_var: str) -> None:
        """가중치 적용 시."""
        output = self._get_output()
        output.add_output(f"⚖️ 가중치 적용: {weight_var}", "success")

    def _on_weight_cleared(self) -> None:
        """가중치 해제 시."""
        output = self._get_output()
        output.add_output("⚖️ 가중치가 해제되었습니다.", "success")

    def _open_merge_files(self) -> None:
        """파일 병합 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.merge_dialog import MergeDialog
        dialog = MergeDialog(self.current_dataset, self)
        dialog.merge_completed.connect(self._on_merge_completed)
        dialog.exec()

    def _on_merge_completed(self, dataset: object) -> None:
        """병합 완료 시."""
        self.current_dataset = dataset
        self._on_dataset_changed(dataset)
        output = self._get_output()
        output.add_output(f"🔗 파일 병합 완료: {dataset.name} ({len(dataset.data)}행)", "success")

    def _open_pivot_table(self) -> None:
        """피벗 테이블 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.pivot_dialog import PivotDialog
        dialog = PivotDialog(self.current_dataset, self)
        dialog.pivot_created.connect(self._on_pivot_created)
        dialog.exec()

    def _on_pivot_created(self, pivot_table: object) -> None:
        """피벗 테이블 생성 완료 시."""
        output = self._get_output()
        output.add_output("📊 피벗 테이블이 생성되었습니다.", "success")

    # ── 변환 메뉴 ──────────────────────────────────────────────────────────

    def _open_compute_variable(self) -> None:
        """변수 계산 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
        dialog = ComputeVariableDialog(self.current_dataset, self)
        dialog.variable_computed.connect(self._on_variable_computed)
        dialog.exec()

    def _on_variable_computed(self, var_name: str) -> None:
        """변수 계산 완료 시."""
        self._on_dataset_changed(self.current_dataset)
        output = self._get_output()
        output.add_output(f"🔢 변수 '{var_name}'가 계산되었습니다.", "success")

    def _open_recode(self) -> None:
        """변수 재코딩 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.recode_dialog import RecodeDialog
        dialog = RecodeDialog(self.current_dataset, self)
        dialog.recode_applied.connect(self._on_recode_applied)
        dialog.exec()

    def _on_recode_applied(self, source_var: str, target_var: str, rules: dict) -> None:
        """재코딩 적용 시."""
        # 실제 데이터에 재코딩 적용
        try:
            series = self.current_dataset.data[source_var].copy()
            new_series = series.replace(rules)
            self.current_dataset.data[target_var] = new_series
        except Exception:
            pass
        self._on_dataset_changed()
        output = self._get_output()
        output.add_output(f"🔄 변수 '{source_var}' -> '{target_var}' 재코딩 완료 ({len(rules)}개 규칙)", "success")

    def _open_binning(self) -> None:
        """시각적 구간화 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.binning_dialog import BinningDialog
        dialog = BinningDialog(self.current_dataset, self)
        dialog.binning_applied.connect(self._on_bins_created)
        dialog.exec()

    def _on_bins_created(self, source_var: str, target_var: str, cut_points: list, labels: list) -> None:
        """구간화 완료 시."""
        import pandas as pd
        import numpy as np
        try:
            series = self.current_dataset.data[source_var]
            binned = pd.cut(series, bins=cut_points, labels=labels, include_lowest=True)
            self.current_dataset.data[target_var] = binned
        except Exception:
            pass
        self._on_dataset_changed()
        output = self._get_output()
        output.add_output(f"📊 구간화 변수 '{target_var}'가 생성되었습니다. (구간 수: {len(labels)})", "success")

    def _open_rank(self) -> None:
        """순위 계산 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.rank_dialog import RankDialog
        dialog = RankDialog(self.current_dataset, self)
        dialog.rank_created.connect(self._on_rank_created)
        dialog.exec()

    def _on_rank_created(self, var_name: str) -> None:
        """순위 계산 완료 시."""
        self._on_dataset_changed(self.current_dataset)
        output = self._get_output()
        output.add_output(f"🏆 순위 변수 '{var_name}'가 생성되었습니다.", "success")

    # ── 분석 메뉴 ──────────────────────────────────────────────────────────

    def _open_script_runner(self) -> None:
        """스크립트 실행 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.script_runner_dialog import ScriptRunnerDialog
        dialog = ScriptRunnerDialog(self.current_dataset, self)
        dialog.script_executed.connect(self._on_script_executed)
        dialog.exec()

    def _on_script_executed(self, result: str) -> None:
        """스크립트 실행 완료 시."""
        output = self._get_output()
        output.add_output(f"🔧 스크립트 실행 완료:\n{result}", "success")

    def _run_frequencies(self) -> None:
        """빈도 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.frequencies_dialog import FrequenciesDialog
        dialog = FrequenciesDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_descriptives(self) -> None:
        """기술통계량 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.descriptives_dialog import DescriptivesDialog
        dialog = DescriptivesDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_crosstabs(self) -> None:
        """교차분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.crosstab_dialog import CrosstabDialog
        dialog = CrosstabDialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_crosstab_completed)
        dialog.exec()

    def _run_independent_ttest(self) -> None:
        """독립표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.ttest_dialog import IndependentTTestDialog
        dialog = IndependentTTestDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_ttest_result)
        dialog.exec()

    def _on_ttest_result(self, result) -> None:
        """T 검정 결과 처리."""
        self._on_analysis_result(result)
        self.statusbar.showMessage("T 검정 완료")

    def _on_analysis_result(self, result) -> None:
        """AnalysisResult 시그널 공통 처리."""
        self._ensure_output_window()
        try:
            self._output_window.add_output(result.to_html(), "analysis")
        except Exception:
            self._output_window.add_output(str(result), "analysis")
        self.statusbar.showMessage("분석 완료")

    def _on_legacy_analysis_completed(self, spec: dict) -> None:
        """분석 완료(dict) 시그널 처리 — 다이얼로그 내부에서 이미 계산된 결과 표시."""
        self._ensure_output_window()
        analysis_type = spec.get("type", "분석")
        result_text = spec.get("result", "")
        if result_text:
            self._output_window.add_output(f"<pre>{result_text}</pre>", "analysis")
        else:
            import json
            displayable = {k: v for k, v in spec.items() if k not in ("correlation_matrix",)}
            self._output_window.add_output(f"<pre>{json.dumps(displayable, ensure_ascii=False, indent=2)}</pre>", "analysis")
        self.statusbar.showMessage(f"분석 완료: {analysis_type}")

    def _on_crosstab_completed(self, spec: dict) -> None:
        """교차분석 다이얼로그 완료 — spec을 받아 crosstab.run_analysis로 실행."""
        self._ensure_output_window()
        try:
            from statworkbench.analysis.crosstab import run_analysis
            result = run_analysis(self.current_dataset, spec)
            self._output_window.add_output(result.to_html(), "analysis")
            self.statusbar.showMessage("교차분석 완료")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"교차분석 실행 실패:\n{exc}")

    def _run_paired_ttest(self) -> None:
        """대응표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.ttest_dialog import PairedTTestDialog
        dialog = PairedTTestDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_ttest_result)
        dialog.exec()

    def _run_anova(self) -> None:
        """분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.anova_dialog import ANOVADialog
        dialog = ANOVADialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_legacy_analysis_completed)
        dialog.exec()

    def _run_correlation(self) -> None:
        """상관분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.correlation_dialog import CorrelationDialog
        dialog = CorrelationDialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_legacy_analysis_completed)
        dialog.exec()

    def _run_regression(self) -> None:
        """회귀분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.regression_dialog import RegressionDialog
        dialog = RegressionDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_ttest_result)
        dialog.exec()

    def _run_nonparametric(self) -> None:
        """비모수 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.nonparametric_dialog import NonparametricDialog
        dialog = NonparametricDialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_legacy_analysis_completed)
        dialog.exec()

    def _ensure_output_window(self) -> None:
        """결과 창이 없으면 새로 만들어 표시."""
        if self._output_window is None or not self._output_window.isVisible():
            self._show_output_window()

    def _run_one_sample_ttest(self) -> None:
        """단일표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.one_sample_ttest_dialog import OneSampleTTestDialog
        dialog = OneSampleTTestDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis.ttests import run_one_sample_ttest
                result = run_one_sample_ttest(
                    self.current_dataset.data,
                    spec["variable"],
                    spec["test_value"],
                )
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("단일표본 T 검정 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _run_logistic_regression(self) -> None:
        """로지스틱 회귀 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.logistic_regression_dialog import LogisticRegressionDialog
        dialog = LogisticRegressionDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis import logistic_regression
                result = logistic_regression.run_analysis(self.current_dataset, spec)
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("로지스틱 회귀 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _run_factor_analysis(self) -> None:
        """요인분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.factor_analysis_dialog import FactorAnalysisDialog
        dialog = FactorAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis import factor_analysis
                result = factor_analysis.run_analysis(self.current_dataset, spec)
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("요인분석 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _run_cluster_analysis(self) -> None:
        """군집분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.cluster_analysis_dialog import ClusterAnalysisDialog
        dialog = ClusterAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis import cluster_analysis
                result = cluster_analysis.run_analysis(self.current_dataset, spec)
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("군집분석 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _run_survival_analysis(self) -> None:
        """생존분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.survival_analysis_dialog import SurvivalAnalysisDialog
        dialog = SurvivalAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis import survival_analysis
                result = survival_analysis.run_analysis(self.current_dataset, spec)
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("생존분석 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _run_discriminant_analysis(self) -> None:
        """판별분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.discriminant_analysis_dialog import DiscriminantAnalysisDialog
        dialog = DiscriminantAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            try:
                from statworkbench.analysis import discriminant_analysis
                result = discriminant_analysis.run_analysis(self.current_dataset, spec)
                self._ensure_output_window()
                self._output_window.add_output(result.to_html(), "analysis")
                self.statusbar.showMessage("판별분석 완료")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    def _on_analysis_requested(self, analysis_type: str, params: dict) -> None:
        """분석 요청 처리."""
        from statworkbench.analysis.registry import AnalysisRegistry

        try:
            registry = AnalysisRegistry()
            result = registry.execute(analysis_type, self.current_dataset, params)

            output = self._get_output()
            output.add_output(result.to_html(), "analysis")

            self.statusbar.showMessage(f"분석 완료: {analysis_type}")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    # ── 차트 메뉴 ──────────────────────────────────────────────────────────

    def _open_visualization(self) -> None:
        """고급 시각화 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.visualization_dialog import VisualizationDialog
        dialog = VisualizationDialog(self.current_dataset, self)
        dialog.chart_created.connect(self._on_chart_created)
        dialog.exec()

    def _on_chart_created(self, chart_path: str) -> None:
        """차트 생성 완료 시."""
        output = self._get_output()
        output.add_output(f"📊 차트가 생성되었습니다: {chart_path}", "success")

    def _open_chart_builder(self) -> None:
        """차트 빌더 열기."""
        QMessageBox.information(self, "차트 빌더", "차트 빌더가 곧 제공됩니다.")

    # ── 유틸리티 메뉴 ──────────────────────────────────────────────────────

    def _open_data_quality(self) -> None:
        """데이터 품질 진단 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.data_quality_dialog import DataQualityDialog
        dialog = DataQualityDialog(self.current_dataset, self)
        dialog.quality_report_generated.connect(self._on_quality_report_generated)
        dialog.exec()

    def _on_quality_report_generated(self, path: str) -> None:
        """품질 보고서 생성 완료 시."""
        output = self._get_output()
        output.add_output(f"📄 데이터 품질 보고서가 생성되었습니다: {path}", "success")

    def _open_report_generator(self) -> None:
        """보고서 생성 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from statworkbench.ui.dialogs.report_dialog import ReportDialog
        dialog = ReportDialog(self.current_dataset, [], self)
        dialog.report_generated.connect(self._on_report_generated)
        dialog.exec()

    def _on_report_generated(self, path: str) -> None:
        """보고서 생성 완료 시."""
        output = self._get_output()
        output.add_output(f"📄 보고서가 생성되었습니다: {path}", "success")

    def _show_about(self) -> None:
        """프로그램 정보."""
        QMessageBox.about(
            self,
            "StatWorkbench 정보",
            "<h2>StatWorkbench</h2>"
            "<p>버전: 1.0.0</p>"
            "<p>SPSS 스타일 통계 분석 패키지</p>"
            "<p>Python + PySide6 기반</p>"
        )
