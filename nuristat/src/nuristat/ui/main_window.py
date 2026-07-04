"""Main application window for NuriStat.

SPSS 스타일 메뉴 구조:
파일, 편집, 보기, 데이터, 변환, 분석, 차트, 유틸리티, 창, 도움말

레이아웃:
- 중앙: 데이터 뷰 (전체 화면)
- 하단: 탭 (데이터 보기, 변수 보기, 구문 편집기)
- 결과: 독립 창 (누적 출력)
"""


from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from nuristat.core.dataset import Dataset
from nuristat.core.i18n import t
from nuristat.core.project import Project
from nuristat.core.settings import SettingsManager
from nuristat.ui.data_view import DataView
from nuristat.ui.output_window import OutputWindow
from nuristat.ui.syntax_editor import SyntaxEditor
from nuristat.ui.theme import ThemeManager, ThemeMode, get_application_stylesheet
from nuristat.ui.variable_view import VariableView


class MainWindow(QMainWindow):
    """NuriStat 메인 윈도우."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("누리스탯")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # 프로젝트 상태
        self.project: Project | None = None
        self._current_dataset: Dataset | None = None

        # 테마 설정
        self._theme_manager = ThemeManager()
        self._dark_mode = False

        # 설정 관리자
        self._settings = SettingsManager()

        # Language: single setting controls both UI chrome and analysis output (default "en")
        from nuristat.core import i18n as _i18n
        _i18n.set_language(self._settings.load_language())

        # 결과 창 (단일 인스턴스)
        self._output_window: OutputWindow | None = None

        # 활성 케이스 필터·가중치 상태
        self._active_weight_var: str | None = None

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_theme()
        self._load_settings()

        # 빈 프로젝트로 시작
        self._new_project()

    @property
    def current_dataset(self) -> Dataset | None:
        """현재 데이터셋을 반환합니다(지연 동기화 확정 후, P1-2)."""
        data_view = getattr(self, "data_view", None)
        if data_view is not None and self._current_dataset is not None:
            data_view.sync_dataset()
        return self._current_dataset

    @current_dataset.setter
    def current_dataset(self, value: Dataset | None) -> None:
        self._current_dataset = value

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
        self.data_view.selection_info_changed.connect(self._on_selection_info)
        data_layout.addWidget(self.data_view)

        # 변수 뷰 (숨김)
        self.variable_view = VariableView()
        self.variable_view.dataset_changed.connect(self._on_dataset_changed)
        self.variable_view.hide()
        data_layout.addWidget(self.variable_view)

        # 구문 편집기 (숨김)
        self.syntax_editor = SyntaxEditor()
        self.syntax_editor.syntax_executed.connect(self._on_syntax_executed)
        self.syntax_editor.analysis_ready.connect(self._on_analysis_result)
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
        """SPSS-style menu bar."""
        menubar = self.menuBar()

        # 1. File menu
        file_menu = menubar.addMenu(t("File(&F)"))

        new_action = QAction(t("🆕 New Project"), self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        open_action = QAction(t("📂 Open Project..."), self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)

        save_action = QAction(t("💾 Save Project"), self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction(t("💾 Save Project As..."), self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        import_menu = file_menu.addMenu(t("📥 Import(&I)"))

        import_csv_action = QAction(t("📄 CSV / Text..."), self)
        import_csv_action.triggered.connect(self._import_csv)
        import_menu.addAction(import_csv_action)

        import_excel_action = QAction(t("📊 Excel File..."), self)
        import_excel_action.triggered.connect(self._import_excel)
        import_menu.addAction(import_excel_action)

        import_sav_action = QAction(t("📋 SPSS File (.sav)..."), self)
        import_sav_action.triggered.connect(self._import_sav)
        import_menu.addAction(import_sav_action)

        import_clipboard_action = QAction(t("📋 Clipboard..."), self)
        import_clipboard_action.triggered.connect(self._import_clipboard)
        import_menu.addAction(import_clipboard_action)

        export_menu = file_menu.addMenu(t("📤 Export(&X)"))

        export_csv_action = QAction(t("📄 CSV File..."), self)
        export_csv_action.triggered.connect(self._export_csv)
        export_menu.addAction(export_csv_action)

        export_excel_action = QAction(t("📊 Excel File..."), self)
        export_excel_action.triggered.connect(self._export_excel)
        export_menu.addAction(export_excel_action)

        export_sav_action = QAction(t("📋 SPSS File (.sav)..."), self)
        export_sav_action.triggered.connect(self._export_sav)
        export_menu.addAction(export_sav_action)

        file_menu.addSeparator()

        self._recent_menu = file_menu.addMenu(t("🕘 Recent Files(&R)"))
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction(t("🚪 Exit"), self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 2. Edit menu
        edit_menu = menubar.addMenu(t("✏️ Edit(&E)"))

        undo_action = QAction(t("↩️ Undo"), self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self._edit_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction(t("↪️ Redo"), self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self._edit_redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction(t("✂️ Cut"), self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self._edit_cut)
        edit_menu.addAction(cut_action)

        copy_action = QAction(t("📋 Copy"), self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self._edit_copy)
        edit_menu.addAction(copy_action)

        paste_action = QAction(t("📋 Paste"), self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self._edit_paste)
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        select_all_action = QAction(t("☑️ Select All"), self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        select_all_action.triggered.connect(self._edit_select_all)
        edit_menu.addAction(select_all_action)

        # 3. View menu
        view_menu = menubar.addMenu(t("👁️ View(&V)"))

        self._theme_action = QAction(t("🌙 Dark Mode"), self)
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(False)
        self._theme_action.setShortcut("Ctrl+Shift+D")
        self._theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._theme_action)

        # Language switcher (UI + output)
        from PySide6.QtGui import QActionGroup
        lang_menu = view_menu.addMenu(t("🌐 UI Language"))
        self._lang_group = QActionGroup(self)
        cur_lang = self._settings.load_language()
        self._lang_ko_action = QAction("한국어", self, checkable=True)
        self._lang_en_action = QAction("English", self, checkable=True)
        self._lang_ko_action.setChecked(cur_lang == "ko")
        self._lang_en_action.setChecked(cur_lang != "ko")
        self._lang_ko_action.triggered.connect(lambda: self._set_output_language("ko"))
        self._lang_en_action.triggered.connect(lambda: self._set_output_language("en"))
        for a in (self._lang_en_action, self._lang_ko_action):
            self._lang_group.addAction(a)
            lang_menu.addAction(a)

        view_menu.addSeparator()

        data_view_action = QAction(t("🔢 Data View"), self)
        data_view_action.setShortcut("Ctrl+1")
        data_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(0))
        view_menu.addAction(data_view_action)

        var_view_action = QAction(t("📋 Variable View"), self)
        var_view_action.setShortcut("Ctrl+2")
        var_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(1))
        view_menu.addAction(var_view_action)

        syntax_view_action = QAction(t("📝 Syntax Editor"), self)
        syntax_view_action.setShortcut("Ctrl+3")
        syntax_view_action.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(2))
        view_menu.addAction(syntax_view_action)

        view_menu.addSeparator()

        self._value_labels_action = QAction(t("🏷️ Show Value Labels"), self)
        self._value_labels_action.setCheckable(True)
        self._value_labels_action.setChecked(False)
        self._value_labels_action.setShortcut("Ctrl+L")
        self._value_labels_action.triggered.connect(self._toggle_value_labels)
        view_menu.addAction(self._value_labels_action)

        view_menu.addSeparator()

        show_output_action = QAction(t("📊 Show Output Window"), self)
        show_output_action.setShortcut("Ctrl+Shift+O")
        show_output_action.triggered.connect(self._show_output_window)
        view_menu.addAction(show_output_action)

        # 4. Data menu
        data_menu = menubar.addMenu(t("Data(&D)"))

        select_cases_action = QAction(t("🔍 Select Cases..."), self)
        select_cases_action.triggered.connect(self._open_select_cases)
        data_menu.addAction(select_cases_action)

        weight_cases_action = QAction(t("⚖️ Weight Cases..."), self)
        weight_cases_action.triggered.connect(self._open_weight_cases)
        data_menu.addAction(weight_cases_action)

        data_menu.addSeparator()

        sort_cases_action = QAction(t("🔀 Sort Cases..."), self)
        sort_cases_action.triggered.connect(self._open_sort_dialog)
        data_menu.addAction(sort_cases_action)

        transpose_action = QAction(t("↔️ Transpose..."), self)
        transpose_action.triggered.connect(self._transpose_dataset)
        data_menu.addAction(transpose_action)

        merge_files_action = QAction(t("🔗 Merge Files..."), self)
        merge_files_action.triggered.connect(self._open_merge_files)
        data_menu.addAction(merge_files_action)

        aggregate_action = QAction(t("📊 Pivot Table..."), self)
        aggregate_action.triggered.connect(self._open_pivot_table)
        data_menu.addAction(aggregate_action)

        # 5. Transform menu
        transform_menu = menubar.addMenu(t("Transform(&T)"))

        compute_var_action = QAction(t("🔢 Compute Variable..."), self)
        compute_var_action.triggered.connect(self._open_compute_variable)
        transform_menu.addAction(compute_var_action)

        recode_action = QAction(t("🔄 Recode Variable..."), self)
        recode_action.triggered.connect(self._open_recode)
        transform_menu.addAction(recode_action)

        visual_binning_action = QAction(t("📊 Visual Binning..."), self)
        visual_binning_action.triggered.connect(self._open_binning)
        transform_menu.addAction(visual_binning_action)

        rank_cases_action = QAction(t("🏆 Rank Cases..."), self)
        rank_cases_action.triggered.connect(self._open_rank)
        transform_menu.addAction(rank_cases_action)

        # 6. Analyze menu
        analyze_menu = menubar.addMenu(t("📊 Analyze(&A)"))

        script_action = QAction(t("🔧 Run Script..."), self)
        script_action.setShortcut("Ctrl+Shift+R")
        script_action.triggered.connect(self._open_script_runner)
        analyze_menu.addAction(script_action)

        analyze_menu.addSeparator()

        desc_menu = analyze_menu.addMenu(t("📈 Descriptive Statistics(&R)"))

        freq_action = QAction(t("📊 Frequencies..."), self)
        freq_action.setShortcut("Ctrl+Shift+F")
        freq_action.triggered.connect(self._run_frequencies)
        desc_menu.addAction(freq_action)

        desc_action = QAction(t("📈 Descriptives..."), self)
        desc_action.setShortcut("Ctrl+Shift+U")
        desc_action.triggered.connect(self._run_descriptives)
        desc_menu.addAction(desc_action)

        explore_action = QAction(t("🔍 Explore..."), self)
        explore_action.triggered.connect(self._run_explore)
        desc_menu.addAction(explore_action)

        crosstab_action = QAction(t("📊 Crosstabulation..."), self)
        crosstab_action.triggered.connect(self._run_crosstabs)
        desc_menu.addAction(crosstab_action)

        normality_action = QAction(t("📐 Normality Test (Shapiro-Wilk)..."), self)
        normality_action.triggered.connect(self._run_normality)
        desc_menu.addAction(normality_action)

        compare_menu = analyze_menu.addMenu(t("🔄 Compare Means(&M)"))

        one_sample_t_action = QAction(t("1️⃣ One-Sample T Test..."), self)
        one_sample_t_action.triggered.connect(self._run_one_sample_ttest)
        compare_menu.addAction(one_sample_t_action)

        ind_t_action = QAction(t("2️⃣ Independent-Samples T Test..."), self)
        ind_t_action.setShortcut("Ctrl+Shift+T")
        ind_t_action.triggered.connect(self._run_independent_ttest)
        compare_menu.addAction(ind_t_action)

        paired_t_action = QAction(t("🔗 Paired-Samples T Test..."), self)
        paired_t_action.triggered.connect(self._run_paired_ttest)
        compare_menu.addAction(paired_t_action)

        anova_action = QAction(t("📊 One-Way ANOVA..."), self)
        anova_action.setShortcut("Ctrl+Shift+A")
        anova_action.triggered.connect(self._run_anova)
        compare_menu.addAction(anova_action)

        glm_menu = analyze_menu.addMenu(t("📊 General Linear Model(&G)"))
        two_way_action = QAction(t("📊 Two-Way ANOVA (Univariate)..."), self)
        two_way_action.triggered.connect(self._run_two_way_anova)
        glm_menu.addAction(two_way_action)
        rm_action = QAction(t("🔄 Repeated Measures..."), self)
        rm_action.triggered.connect(self._run_repeated_measures_anova)
        glm_menu.addAction(rm_action)
        ancova_action = QAction(t("📊 ANCOVA..."), self)
        ancova_action.triggered.connect(self._run_ancova)
        glm_menu.addAction(ancova_action)
        mixed_anova_action = QAction(t("🔀 Mixed ANOVA..."), self)
        mixed_anova_action.triggered.connect(self._run_mixed_anova)
        glm_menu.addAction(mixed_anova_action)
        manova_action = QAction(t("📊 MANOVA..."), self)
        manova_action.triggered.connect(self._run_manova)
        glm_menu.addAction(manova_action)

        correlate_menu = analyze_menu.addMenu(t("🔗 Correlate(&C)"))

        bivariate_corr_action = QAction(t("🔗 Bivariate Correlation..."), self)
        bivariate_corr_action.triggered.connect(self._run_correlation)
        correlate_menu.addAction(bivariate_corr_action)

        partial_corr_action = QAction(t("🔗 Partial Correlation..."), self)
        partial_corr_action.triggered.connect(self._run_partial_correlation)
        correlate_menu.addAction(partial_corr_action)

        regression_menu = analyze_menu.addMenu(t("📈 Regression(&R)"))

        linear_action = QAction(t("📈 Linear..."), self)
        linear_action.setShortcut("Ctrl+Shift+L")
        linear_action.triggered.connect(self._run_regression)
        regression_menu.addAction(linear_action)

        logistic_action = QAction(t("📊 Logistic..."), self)
        logistic_action.triggered.connect(self._run_logistic_regression)
        regression_menu.addAction(logistic_action)

        multinomial_action = QAction(t("📊 Multinomial Logistic..."), self)
        multinomial_action.triggered.connect(self._run_multinomial_logistic)
        regression_menu.addAction(multinomial_action)

        dim_reduce_menu = analyze_menu.addMenu(t("📉 Dimension Reduction(&D)"))
        factor_action = QAction(t("📉 Factor Analysis..."), self)
        factor_action.triggered.connect(self._run_factor_analysis)
        dim_reduce_menu.addAction(factor_action)
        pca_action = QAction(t("📉 Principal Component Analysis (PCA)..."), self)
        pca_action.triggered.connect(self._run_pca)
        dim_reduce_menu.addAction(pca_action)

        cluster_menu = analyze_menu.addMenu(t("🔵 Cluster(&K)"))
        kmeans_action = QAction(t("🔵 K-Means Cluster..."), self)
        kmeans_action.triggered.connect(self._run_cluster_analysis)
        cluster_menu.addAction(kmeans_action)
        hierarchical_action = QAction(t("🔵 Hierarchical Cluster..."), self)
        hierarchical_action.triggered.connect(self._run_cluster_analysis)
        cluster_menu.addAction(hierarchical_action)

        survival_menu = analyze_menu.addMenu(t("Survival Analysis(&S)"))
        km_action = QAction("📈 Kaplan-Meier...", self)
        km_action.triggered.connect(self._run_kaplan_meier)
        survival_menu.addAction(km_action)
        cox_action = QAction(t("📉 Cox Proportional Hazards Regression..."), self)
        cox_action.triggered.connect(self._run_cox_regression)
        survival_menu.addAction(cox_action)

        classify_menu = analyze_menu.addMenu(t("🔷 Discriminant Analysis(&I)"))
        lda_action = QAction(t("🔷 Discriminant Analysis..."), self)
        lda_action.triggered.connect(self._run_discriminant_analysis)
        classify_menu.addAction(lda_action)

        nonparam_menu = analyze_menu.addMenu(t("🧪 Nonparametric Tests(&N)"))

        nonparam_action = QAction(t("🧪 Nonparametric Tests..."), self)
        nonparam_action.triggered.connect(self._run_nonparametric)
        nonparam_menu.addAction(nonparam_action)

        chi_gof_action = QAction(t("🧮 Chi-Square Goodness-of-Fit..."), self)
        chi_gof_action.triggered.connect(self._run_chi_square_gof)
        nonparam_menu.addAction(chi_gof_action)

        diagnostic_menu = analyze_menu.addMenu(t("🔬 Diagnostic Tests(&T)"))

        roc_action = QAction(t("📈 ROC Analysis..."), self)
        roc_action.triggered.connect(self._run_roc_analysis)
        diagnostic_menu.addAction(roc_action)

        agreement_menu = analyze_menu.addMenu(t("✅ Agreement Analysis(&G)"))

        kappa_action = QAction("κ Cohen's Kappa...", self)
        kappa_action.triggered.connect(self._run_cohens_kappa)
        agreement_menu.addAction(kappa_action)

        icc_action = QAction(t("📊 ICC (Intraclass Correlation)..."), self)
        icc_action.triggered.connect(self._run_icc)
        agreement_menu.addAction(icc_action)

        ba_action = QAction("📉 Bland-Altman...", self)
        ba_action.triggered.connect(self._run_bland_altman)
        agreement_menu.addAction(ba_action)

        scale_menu = analyze_menu.addMenu(t("📐 Scale Analysis(&S)"))

        reliability_action = QAction(t("🔁 Reliability Analysis (Cronbach α)..."), self)
        reliability_action.triggered.connect(self._run_reliability)
        scale_menu.addAction(reliability_action)

        text_menu = analyze_menu.addMenu(t("📝 Text Mining(&X)"))
        text_mining_action = QAction(t("📝 Text Mining (Word Cloud)..."), self)
        text_mining_action.triggered.connect(self._run_text_mining)
        text_menu.addAction(text_mining_action)

        analyze_menu.addSeparator()

        ml_action = QAction(t("🤖 Machine Learning..."), self)
        ml_action.triggered.connect(self._open_ml_dialog)
        analyze_menu.addAction(ml_action)

        # 7. Graphs menu
        graphs_menu = menubar.addMenu(t("Graphs(&G)"))

        chart_builder_action = QAction(t("📊 Advanced Visualization..."), self)
        chart_builder_action.setShortcut("Ctrl+Shift+V")
        chart_builder_action.triggered.connect(self._open_visualization)
        graphs_menu.addAction(chart_builder_action)

        graphs_menu.addSeparator()

        chart_builder_legacy_action = QAction(t("Chart Builder..."), self)
        chart_builder_legacy_action.triggered.connect(self._open_chart_builder)
        graphs_menu.addAction(chart_builder_legacy_action)

        legacy_graphs_menu = graphs_menu.addMenu(t("Legacy Dialogs(&L)"))

        bar_action = QAction(t("Bar..."), self)
        bar_action.triggered.connect(lambda: self._open_legacy_chart("bar"))
        legacy_graphs_menu.addAction(bar_action)

        line_action = QAction(t("Line..."), self)
        line_action.triggered.connect(lambda: self._open_legacy_chart("line"))
        legacy_graphs_menu.addAction(line_action)

        scatter_action = QAction(t("Scatter..."), self)
        scatter_action.triggered.connect(lambda: self._open_legacy_chart("scatter"))
        legacy_graphs_menu.addAction(scatter_action)

        histogram_action = QAction(t("Histogram..."), self)
        histogram_action.triggered.connect(lambda: self._open_legacy_chart("hist"))
        legacy_graphs_menu.addAction(histogram_action)

        boxplot_action = QAction(t("Box Plot..."), self)
        boxplot_action.triggered.connect(lambda: self._open_legacy_chart("box"))
        legacy_graphs_menu.addAction(boxplot_action)

        # 8. Utilities menu
        utilities_menu = menubar.addMenu(t("Utilities(&U)"))

        data_quality_action = QAction(t("🔍 Data Quality Diagnosis..."), self)
        data_quality_action.triggered.connect(self._open_data_quality)
        utilities_menu.addAction(data_quality_action)

        report_action = QAction(t("📄 Report Generator..."), self)
        report_action.triggered.connect(self._open_report_generator)
        utilities_menu.addAction(report_action)

        utilities_menu.addSeparator()

        var_info_action = QAction(t("📋 Variable Information..."), self)
        var_info_action.triggered.connect(self._show_variable_info)
        utilities_menu.addAction(var_info_action)

        file_info_action = QAction(t("📁 File Information..."), self)
        file_info_action.triggered.connect(self._show_file_info)
        utilities_menu.addAction(file_info_action)

        # 9. Window menu
        menubar.addMenu(t("Window(&W)"))

        # 10. Help menu
        help_menu = menubar.addMenu(t("Help(&H)"))

        about_action = QAction(t("About NuriStat"), self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self) -> None:
        """Main toolbar."""
        toolbar = QToolBar(t("Main Toolbar"))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_btn = QAction(t("🆕 New Project"), self)
        new_btn.triggered.connect(self._new_project)
        toolbar.addAction(new_btn)

        open_btn = QAction(t("📂 Open"), self)
        open_btn.triggered.connect(self._open_project)
        toolbar.addAction(open_btn)

        save_btn = QAction(t("💾 Save"), self)
        save_btn.triggered.connect(self._save_project)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        data_view_btn = QAction(t("🔢 Data"), self)
        data_view_btn.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(0))
        toolbar.addAction(data_view_btn)

        var_view_btn = QAction(t("📋 Variables"), self)
        var_view_btn.triggered.connect(lambda: self.bottom_tabs.setCurrentIndex(1))
        toolbar.addAction(var_view_btn)

        toolbar.addSeparator()

        freq_btn = QAction(t("📊 Frequencies"), self)
        freq_btn.triggered.connect(self._run_frequencies)
        toolbar.addAction(freq_btn)

        desc_btn = QAction(t("📈 Descriptives"), self)
        desc_btn.triggered.connect(self._run_descriptives)
        toolbar.addAction(desc_btn)

        ttest_btn = QAction(t("📉 T Test"), self)
        ttest_btn.triggered.connect(self._run_independent_ttest)
        toolbar.addAction(ttest_btn)

        reg_btn = QAction(t("📉 Regression"), self)
        reg_btn.triggered.connect(self._run_regression)
        toolbar.addAction(reg_btn)

    def _setup_statusbar(self) -> None:
        """상태 표시줄 설정."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 상태 메시지
        self.statusbar.showMessage(t("Ready"))

        # 마지막 분석 이름 (오른쪽 끝)
        self._last_analysis_label = QLabel("")
        self._last_analysis_label.setStyleSheet("color: gray; margin-right: 8px;")
        self.statusbar.addPermanentWidget(self._last_analysis_label)

        # 선택 셀 정보 (다중 선택 시 "N행 × M열 선택" 표시)
        self._selection_info_label = QLabel("")
        self._selection_info_label.setStyleSheet("color: #555; margin-right: 8px;")
        self.statusbar.addPermanentWidget(self._selection_info_label)

        # 필터·가중치 상태 표시
        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet("color: #e67e22; font-weight: bold; margin-right: 8px;")
        self.statusbar.addPermanentWidget(self._filter_label)

        self._weight_label = QLabel("")
        self._weight_label.setStyleSheet("color: #8e44ad; font-weight: bold; margin-right: 8px;")
        self.statusbar.addPermanentWidget(self._weight_label)

        # 데이터셋 정보 (N=행 변수=열 형식)
        self.dataset_info_label = QLabel(f"N=0  {t('Variables')}=0")
        self.statusbar.addPermanentWidget(self.dataset_info_label)

    def _update_statusbar(self) -> None:
        """Update status bar."""
        # P1-2: 행/열 개수는 셀 편집만으로 바뀌지 않으므로 지연 동기화 강제 없이
        # self._current_dataset을 직접 참조한다.
        dataset = self._current_dataset
        if dataset and dataset.data is not None:
            rows = len(dataset.data)
            cols = len(dataset.data.columns)
            self.dataset_info_label.setText(f"N={rows:,}  {t('Variables')}={cols}")
        else:
            self.dataset_info_label.setText(f"N=0  {t('Variables')}=0")

    def _on_selection_info(self, info: str) -> None:
        """데이터 보기 다중 선택 정보를 상태바에 표시."""
        self._selection_info_label.setText(info)

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

    def _toggle_value_labels(self) -> None:
        """데이터 보기 값 라벨 표시 전환 (SPSS: 보기 > 값 라벨)."""
        if not hasattr(self, "data_view"):
            return
        shown = self.data_view.toggle_value_labels()
        self._value_labels_action.setChecked(shown)
        self.statusbar.showMessage(
            "값 라벨 표시: 켜짐 (코드 대신 라벨 표시)" if shown
            else "값 라벨 표시: 꺼짐 (코드값 표시)"
        )

    def _set_output_language(self, lang: str) -> None:
        """Switch UI + output language — saves setting; UI chrome applies on next launch."""
        from nuristat.core import i18n
        i18n.set_language(lang)
        self._settings.save_language(lang)
        self._settings.save_ui_language(lang)
        if lang == "ko":
            self.statusbar.showMessage(
                "언어: 한국어 — 분석 결과 즉시 적용, UI 메뉴는 다음 시작 시 적용"
            )
        else:
            self.statusbar.showMessage(
                "Language: English — analysis output applies immediately, UI menus on next launch"
            )

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
                "NuriStat",
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
        if self._output_window is None:
            self._output_window = OutputWindow(self)
            self._output_window.setWindowTitle("📊 누리스탯 결과")
            self._output_window.resize(800, 600)
        self._output_window.show()
        self._output_window.raise_()
        self._output_window.activateWindow()

    def _get_output(self) -> object:
        """결과 출력 대상 반환."""
        if self._output_window is None or not self._output_window.isVisible():
            self._show_output_window()
        return self._output_window

    # ── 데이터 유틸리티 ─────────────────────────────────────────────────────

    def _transpose_dataset(self) -> None:
        """행렬 전치 — 행↔열 교환. 전치 후 새 데이터셋으로 로드."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.core.dataset import Dataset
        df = self.current_dataset.data
        if df.empty:
            QMessageBox.warning(self, "경고", "데이터가 비어 있습니다.")
            return
        transposed = df.T.reset_index()
        transposed.columns = [f"VAR{i+1:05d}" for i in range(len(transposed.columns))]
        new_ds = Dataset(name=f"{self.current_dataset.name}_전치", data=transposed)
        self._load_dataset(new_ds)
        self.statusbar.showMessage(f"행렬 전치 완료: {df.shape[0]}행×{df.shape[1]}열 → {transposed.shape[0]}행×{transposed.shape[1]}열")

    def _show_variable_info(self) -> None:
        """변수 정보 요약 표시 (이름·유형·측도·라벨 목록)."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        lines = [f"{'변수명':<20} {'유형':<8} {'측도':<8} {'라벨'}"]
        lines.append("-" * 60)
        for name, var in self.current_dataset.variables.items():
            st = var.storage_type.value if hasattr(var.storage_type, "value") else str(var.storage_type)
            ms = var.measure.value if hasattr(var.measure, "value") else str(var.measure)
            label = var.label if var.label and var.label != name else ""
            lines.append(f"{name:<20} {st:<8} {ms:<8} {label}")
        QMessageBox.information(self, f"변수 정보 — {self.current_dataset.name}", "\n".join(lines))

    def _show_file_info(self) -> None:
        """현재 파일(프로젝트) 정보 표시."""
        ds = self.current_dataset
        if ds is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        path = self.project.file_path if self.project and self.project.file_path else "(미저장)"
        n_rows = len(ds.data) if ds.data is not None else 0
        n_cols = len(ds.data.columns) if ds.data is not None else 0
        n_vars = len(ds.variables)
        info = (
            f"데이터셋: {ds.name}\n"
            f"파일 경로: {path}\n"
            f"행(케이스): {n_rows:,}\n"
            f"열(변수): {n_cols}\n"
            f"메타데이터 변수 수: {n_vars}\n"
        )
        QMessageBox.information(self, "파일 정보", info)

    # ── 프로젝트 관리 ───────────────────────────────────────────────────────

    def _open_sort_dialog(self) -> None:
        """정렬 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.sort_dialog import SortDialog
        dialog = SortDialog(self.current_dataset, self)
        dialog.sort_applied.connect(self._on_sort_applied)
        dialog.exec()

    def _on_sort_applied(self) -> None:
        """정렬 적용 시."""
        self._on_dataset_changed(self.current_dataset)
        output = self._get_output()
        output.add_output("🔀 데이터 정렬이 적용되었습니다.", "success")

    def _export_sav(self) -> None:
        """SPSS .sav 내보내기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "내보내기 데이터가 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "SPSS 저장", "", "SPSS 파일 (*.sav)"
        )
        if path:
            try:
                from nuristat.io.spss_writer import write_sav
                write_sav(self.current_dataset, path)
                self.statusbar.showMessage(f"SPSS 저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"SPSS 저장 실패:\n{exc}")

    def _open_ml_dialog(self) -> None:
        """기계학습 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.ml_dialog import MLDialog
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

    # ── 최근 파일 ────────────────────────────────────────────────────────────

    def _rebuild_recent_menu(self) -> None:
        """최근 파일 서브메뉴를 설정에서 다시 채운다."""
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        files = self._settings.load_recent_files()
        if not files:
            empty = self._recent_menu.addAction("(없음)")
            empty.setEnabled(False)
            return
        import os
        for i, path in enumerate(files, start=1):
            name = os.path.basename(path)
            act = self._recent_menu.addAction(f"{i}. {name}")
            act.setToolTip(path)
            act.triggered.connect(lambda _checked=False, p=path: self._open_recent(p))
        self._recent_menu.addSeparator()
        clear_act = self._recent_menu.addAction("최근 파일 목록 지우기")
        clear_act.triggered.connect(self._clear_recent)

    def _remember_recent(self, path: str) -> None:
        """열기/가져오기/저장 성공 시 최근 파일에 기록하고 메뉴를 갱신."""
        self._settings.add_recent_file(path)
        self._rebuild_recent_menu()

    def _clear_recent(self) -> None:
        self._settings.clear_recent_files()
        self._rebuild_recent_menu()

    def _open_recent(self, path: str) -> None:
        """최근 파일을 확장자에 따라 적절한 로더로 다시 연다."""
        import os
        if not os.path.exists(path):
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{path}")
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".swb":
                from nuristat.io.project_store import load_project

                self.project = load_project(path)
                if self.project.datasets:
                    self.current_dataset = self.project.datasets[0]
                    self._on_dataset_changed(self.current_dataset)
            elif ext == ".csv":
                from nuristat.io.csv_reader import read_csv

                self._load_dataset(read_csv(path))
            elif ext in (".xlsx", ".xls"):
                from nuristat.io.excel_reader import read_excel

                self._load_dataset(read_excel(path))
            elif ext == ".sav":
                from nuristat.io.spss_reader import read_sav

                self._load_dataset(read_sav(path))
            else:
                QMessageBox.warning(self, "지원하지 않음", f"지원하지 않는 형식입니다: {ext}")
                return
            self._remember_recent(path)
            self.statusbar.showMessage(f"열었습니다: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"파일 열기 실패:\n{exc}")

    def _load_dataset(self, dataset) -> None:
        """가져온 데이터셋을 모든 뷰에 반영하는 공통 처리."""
        self.current_dataset = dataset
        self.data_view.set_dataset(dataset)
        self.variable_view.set_dataset(dataset)
        self.syntax_editor.set_dataset(dataset)
        self._update_statusbar()

    def _open_project(self) -> None:
        """프로젝트 열기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "프로젝트 열기", "", "누리스탯 프로젝트 (*.swb);;모든 파일 (*.*)"
        )
        if path:
            try:
                from nuristat.io.project_store import load_project

                self.project = load_project(path)
                if self.project.datasets:
                    self.current_dataset = self.project.datasets[0]
                    self.data_view.set_dataset(self.current_dataset)
                    self.variable_view.set_dataset(self.current_dataset)
                    self.syntax_editor.set_dataset(self.current_dataset)
                self._update_statusbar()
                self._remember_recent(path)
                self.statusbar.showMessage(f"프로젝트를 열었습니다: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"프로젝트 열기 실패:\n{exc}")

    def _save_project(self) -> None:
        """프로젝트 저장."""
        if self.project is None:
            return

        if self.project.file_path:
            try:
                from nuristat.io.project_store import save_project

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
            self, "프로젝트 저장", "", "누리스탯 프로젝트 (*.swb)"
        )
        if path:
            if not path.endswith(".swb"):
                path += ".swb"
            try:
                from nuristat.io.project_store import save_project

                save_project(self.project, path)
                self.project.file_path = path
                self.project.clear_dirty()
                self._remember_recent(path)
                self.statusbar.showMessage(f"저장되었습니다: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"저장 실패:\n{exc}")

    def _on_dataset_changed(self, dataset=None) -> None:
        """데이터셋 변경 시 호출됩니다.

        뷰에서 데이터가 바뀌면(편집·정렬·재코딩 등) 다른 뷰에 변경 내용을 전파한다.
        새 데이터셋을 로드할 때는 _load_dataset을 직접 호출하고 이 메서드를 경유하지 않는다.
        """
        if dataset is not None:
            self.current_dataset = dataset
        # P1-2: 변수명·개수는 셀 편집으로 바뀌지 않으므로 여기선 지연 동기화를
        # 강제하는 self.current_dataset 대신 self._current_dataset을 직접 참조한다.
        if self._current_dataset is not None:
            # Variable View: 변수 메타데이터 변경(값 라벨 등) → 데이터 보기에 즉시 반영
            if hasattr(self, 'variable_view') and self.variable_view:
                self.variable_view.set_dataset(self._current_dataset)
            # Data View: 메타데이터 변경(decimals, 측정 척도 등) → 셀 표시 갱신
            if hasattr(self, 'data_view') and self.data_view:
                self.data_view.refresh()
            # 구문 편집기 동기화
            if hasattr(self, 'syntax_editor') and self.syntax_editor:
                self.syntax_editor.set_dataset(self._current_dataset)
            # 프로젝트를 더티 상태로 표시
            if self.project is not None:
                self.project.mark_dirty()

        self._update_statusbar()

    def _on_syntax_executed(self, code: str) -> None:
        """구문 실행 완료 시 — 데이터 구조 변경(열 추가·행 삭제) 가능성 있으므로 전면 재로드."""
        self.statusbar.showMessage("구문이 실행되었습니다.")
        self._reload_all_views()

    def _reload_all_views(self) -> None:
        """데이터 구조 변경(열 추가·행 삭제) 후 모든 뷰를 전면 재로드합니다.

        _on_dataset_changed의 refresh()는 기존 셀 재그리기만 해서 새 열·행이
        모델 복사본에 반영되지 않습니다. 구조 변경 후에는 반드시 이 메서드를 사용.
        """
        if self.current_dataset is None:
            return
        if hasattr(self, 'data_view') and self.data_view:
            self.data_view.reload_data()
        if hasattr(self, 'variable_view') and self.variable_view:
            self.variable_view.set_dataset(self.current_dataset)
        if hasattr(self, 'syntax_editor') and self.syntax_editor:
            self.syntax_editor.set_dataset(self.current_dataset)
        if self.project is not None:
            self.project.mark_dirty()
        self._update_statusbar()

    # ── 파일 가져오기/내보내기 ────────────────────────────────────────────────

    def _import_csv(self) -> None:
        """CSV 가져오기."""
        path, _ = QFileDialog.getOpenFileName(
            self, "CSV 파일 열기", "", "CSV 파일 (*.csv)"
        )
        if path:
            try:
                from nuristat.io.csv_reader import read_csv

                dataset = read_csv(path)
                self._load_dataset(dataset)
                self._remember_recent(path)
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
                from nuristat.io.excel_reader import read_excel

                dataset = read_excel(path)
                self._load_dataset(dataset)
                self._remember_recent(path)
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
                from nuristat.io.spss_reader import read_sav

                dataset = read_sav(path)
                self._load_dataset(dataset)
                self._remember_recent(path)
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
        """CSV 내보내기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "내보내기 데이터가 없습니다.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 저장", "", "CSV 파일 (*.csv)"
        )
        if path:
            try:
                self.current_dataset.data.to_csv(path, index=False, encoding="utf-8-sig")
                self.statusbar.showMessage(f"CSV 저장 완료: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "오류", f"CSV 저장 실패:\n{exc}")

    def _export_excel(self) -> None:
        """Excel 내보내기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "내보내기 데이터가 없습니다.")
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

    def _grid_focused(self) -> bool:
        """현재 포커스가 데이터 뷰 그리드(또는 그 뷰포트)에 있는지."""
        if not hasattr(self, 'data_view'):
            return False
        table = getattr(self.data_view, 'table', None)
        if table is None:
            return False
        focused = QApplication.focusWidget()
        return focused is table or focused is table.viewport()

    def _edit_undo(self) -> None:
        """실행 취소 — 데이터 그리드 우선, 그 외 포커스 위젯."""
        if self._grid_focused():
            self.data_view.undo()
            return
        focused = QApplication.focusWidget()
        if focused is self.syntax_editor or (hasattr(focused, 'parent') and focused is getattr(self, 'syntax_editor', None)):
            self.syntax_editor.undo()
        elif hasattr(focused, 'undo'):
            focused.undo()

    def _edit_redo(self) -> None:
        """다시 실행 — 데이터 그리드 우선."""
        if self._grid_focused():
            self.data_view.redo()
            return
        focused = QApplication.focusWidget()
        if focused is self.syntax_editor or (hasattr(focused, 'parent') and focused is getattr(self, 'syntax_editor', None)):
            self.syntax_editor.redo()
        elif hasattr(focused, 'redo'):
            focused.redo()

    def _edit_cut(self) -> None:
        """잘라내기 — 데이터 그리드 우선."""
        if self._grid_focused():
            self.data_view.cut_selection()
            return
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'cut'):
            current_widget.cut()

    def _edit_copy(self) -> None:
        """복사 — 데이터 그리드 우선."""
        if self._grid_focused():
            self.data_view._copy_selection()
            return
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'copy'):
            current_widget.copy()

    def _edit_paste(self) -> None:
        """붙여넣기 — 데이터 그리드 우선."""
        if self._grid_focused():
            self.data_view._paste_selection()
            return
        current_widget = QApplication.focusWidget()
        if hasattr(current_widget, 'paste'):
            current_widget.paste()

    def _edit_select_all(self) -> None:
        """모두 선택."""
        focused = QApplication.focusWidget()
        if focused is self.syntax_editor:
            self.syntax_editor.selectAll()
        elif hasattr(self, 'data_view') and focused is getattr(self.data_view, 'table', None):
            self.data_view.table.selectAll()
        elif hasattr(focused, 'selectAll'):
            focused.selectAll()

    # ── 데이터 메뉴 ─────────────────────────────────────────────────────────

    def _open_select_cases(self) -> None:
        """케이스 선택 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.select_cases_dialog import SelectCasesDialog
        dialog = SelectCasesDialog(self.current_dataset, self)
        dialog.cases_selected.connect(self._on_cases_selected)
        dialog.exec()

    def _on_cases_selected(self, selection_type: str, condition: object) -> None:
        """케이스 선택 완료 시."""
        output = self._get_output()
        if selection_type == "all":
            output.add_output("🔍 케이스 선택: 모든 케이스 (필터 없음)", "success")
            self._update_filter_statusbar(active=False)
        else:
            df = self.current_dataset.data if self.current_dataset else None
            n_selected = int(df["filter_$"].sum()) if df is not None and "filter_$" in df.columns else "?"
            n_total = len(df) if df is not None else "?"
            output.add_output(
                f"🔍 케이스 선택 켜짐: {n_selected}/{n_total}개 선택 ({selection_type})", "success"
            )
            self._update_filter_statusbar(active=True, n_selected=n_selected, n_total=n_total)

    def _update_filter_statusbar(
        self, active: bool, n_selected: object = None, n_total: object = None
    ) -> None:
        """상태바에 필터 켜짐/꺼짐 표시 (SPSS 스타일)."""
        if hasattr(self, "_filter_label"):
            if active and n_selected is not None:
                self._filter_label.setText(f"필터 켜짐 ({n_selected}/{n_total})")
            else:
                self._filter_label.setText("")

    def _open_weight_cases(self) -> None:
        """가중치 적용 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.weight_cases_dialog import WeightCasesDialog
        dialog = WeightCasesDialog(self.current_dataset, self)
        dialog.weight_applied.connect(self._on_weight_applied)
        dialog.weight_cleared.connect(self._on_weight_cleared)
        dialog.exec()

    def _on_weight_applied(self, weight_var: str) -> None:
        """가중치 적용 시."""
        self._active_weight_var = weight_var
        if self.current_dataset is not None:
            self.current_dataset.active_weight_var = weight_var
        output = self._get_output()
        output.add_output(f"⚖️ 가중치 적용: {weight_var} — 이후 빈도/기술통계 분석에 반영됩니다", "success")
        if hasattr(self, "_weight_label"):
            self._weight_label.setText(f"가중치: {weight_var}")

    def _on_weight_cleared(self) -> None:
        """가중치 해제 시."""
        self._active_weight_var = None
        if self.current_dataset is not None:
            self.current_dataset.active_weight_var = None
        output = self._get_output()
        output.add_output("⚖️ 가중치가 해제되었습니다.", "success")
        if hasattr(self, "_weight_label"):
            self._weight_label.setText("")

    def _open_merge_files(self) -> None:
        """파일 병합 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.merge_dialog import MergeDialog
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

        from nuristat.ui.dialogs.pivot_dialog import PivotDialog
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

        from nuristat.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
        dialog = ComputeVariableDialog(self.current_dataset, self)
        dialog.computed.connect(self._on_variable_computed)
        dialog.exec()

    def _on_variable_computed(self, var_name: str, series) -> None:
        """변수 계산 완료 시."""
        if self.current_dataset is None:
            return
        import pandas as pd
        if isinstance(series, pd.Series):
            self.current_dataset.data[var_name] = series
        elif isinstance(series, (int, float, bool, str)):
            self.current_dataset.data[var_name] = series  # 스칼라 broadcast
        self._reload_all_views()
        self._get_output().add_output(f"변수 '{var_name}' 계산 완료", "success")

    def _open_recode(self) -> None:
        """변수 재코딩 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.recode_dialog import RecodeDialog
        dialog = RecodeDialog(self.current_dataset, self)
        dialog.recode_applied.connect(self._on_recode_applied)
        dialog.exec()

    def _on_recode_applied(self, source_var: str, target_var: str, rules: dict) -> None:
        """재코딩 적용 시."""
        if self.current_dataset is None:
            return
        try:
            series = self.current_dataset.data[source_var].copy()
            self.current_dataset.data[target_var] = series.replace(rules)
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"재코딩 실패:\n{exc}")
            return
        self._reload_all_views()
        self._get_output().add_output(
            f"재코딩 완료: '{source_var}' → '{target_var}' ({len(rules)}개 규칙)", "success"
        )

    def _open_binning(self) -> None:
        """시각적 구간화 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.binning_dialog import BinningDialog
        dialog = BinningDialog(self.current_dataset, self)
        dialog.binning_applied.connect(self._on_bins_created)
        dialog.exec()

    def _on_bins_created(self, source_var: str, target_var: str, cut_points: list, labels: list) -> None:
        """구간화 완료 시."""
        if self.current_dataset is None:
            return
        import pandas as pd
        try:
            series = self.current_dataset.data[source_var]
            self.current_dataset.data[target_var] = pd.cut(
                series, bins=cut_points, labels=labels, include_lowest=True
            )
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"구간화 실패:\n{exc}")
            return
        self._reload_all_views()
        self._get_output().add_output(
            f"구간화 완료: '{target_var}' 생성 (구간 수: {len(labels)})", "success"
        )

    def _open_rank(self) -> None:
        """순위 계산 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.rank_dialog import RankDialog
        dialog = RankDialog(self.current_dataset, self)
        dialog.rank_applied.connect(self._on_rank_created)
        dialog.exec()

    def _on_rank_created(self, source_var: str, target_var: str, method: str) -> None:
        """순위 계산 완료 시."""
        if self.current_dataset is None:
            return
        try:
            series = self.current_dataset.data[source_var]
            self.current_dataset.data[target_var] = series.rank(
                pct=(method == "pct"), method=method if method != "pct" else "average"
            )
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"순위 계산 실패:\n{exc}")
            return
        self._reload_all_views()
        output = self._get_output()
        output.add_output(f"🏆 순위 변수 '{target_var}'가 생성되었습니다.", "success")

    # ── 분석 메뉴 ──────────────────────────────────────────────────────────

    def _open_script_runner(self) -> None:
        """스크립트 실행 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.script_runner_dialog import ScriptRunnerDialog
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

        from nuristat.ui.dialogs.frequencies_dialog import FrequenciesDialog
        dialog = FrequenciesDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_descriptives(self) -> None:
        """기술통계량 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.descriptives_dialog import DescriptivesDialog
        dialog = DescriptivesDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_crosstabs(self) -> None:
        """교차분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.crosstab_dialog import CrosstabDialog
        dialog = CrosstabDialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_crosstab_completed)
        dialog.exec()

    def _run_normality(self) -> None:
        """정규성 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.normality_dialog import NormalityDialog
        dialog = NormalityDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_independent_ttest(self) -> None:
        """독립표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.ttest_dialog import IndependentTTestDialog
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
        self._output_window.add_analysis_result(result)
        analysis_name = getattr(result, "title", "")
        if analysis_name:
            self._last_analysis_label.setText(f"최근 분석: {analysis_name}")
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
        self._last_analysis_label.setText(f"최근 분석: {analysis_type}")
        self.statusbar.showMessage(f"분석 완료: {analysis_type}")

    def _on_crosstab_completed(self, spec: dict) -> None:
        """교차분석 다이얼로그 완료 — spec을 받아 crosstab.run_analysis로 실행."""
        ds = self.current_dataset

        def _run():
            from nuristat.analysis.crosstab import run_analysis
            return run_analysis(ds, spec)

        self._run_async(_run, "교차분석")

    def _run_paired_ttest(self) -> None:
        """대응표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.ttest_dialog import PairedTTestDialog
        dialog = PairedTTestDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_ttest_result)
        dialog.exec()

    def _run_anova(self) -> None:
        """분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.anova_dialog import ANOVADialog
        dialog = ANOVADialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_two_way_anova(self) -> None:
        """이원분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.two_way_anova_dialog import TwoWayAnovaDialog
        dialog = TwoWayAnovaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_repeated_measures_anova(self) -> None:
        """반복측정 ANOVA 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.repeated_measures_dialog import RepeatedMeasuresDialog
        dialog = RepeatedMeasuresDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_mixed_anova(self) -> None:
        """혼합 분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.mixed_anova_dialog import MixedAnovaDialog
        dialog = MixedAnovaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_pca(self) -> None:
        """주성분분석(PCA) 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.pca_dialog import PcaDialog
        dialog = PcaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_multinomial_logistic(self) -> None:
        """다항 로지스틱 회귀 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.multinomial_logistic_dialog import MultinomialLogisticDialog
        dialog = MultinomialLogisticDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_manova(self) -> None:
        """MANOVA 다변량 분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.manova_dialog import ManovaDialog
        dialog = ManovaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_text_mining(self) -> None:
        """텍스트 마이닝 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.text_mining_dialog import TextMiningDialog
        dialog = TextMiningDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_ancova(self) -> None:
        """ANCOVA 공분산분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.dialogs.ancova_dialog import AncovaDialog
        dialog = AncovaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_correlation(self) -> None:
        """상관분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.correlation_dialog import CorrelationDialog
        dialog = CorrelationDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_regression(self) -> None:
        """회귀분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.regression_dialog import RegressionDialog
        dialog = RegressionDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_nonparametric(self) -> None:
        """비모수 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.nonparametric_dialog import NonparametricDialog
        dialog = NonparametricDialog(self.current_dataset, self)
        dialog.analysis_completed.connect(self._on_legacy_analysis_completed)
        dialog.exec()

    def _ensure_output_window(self) -> None:
        """결과 창이 없으면 새로 만들어 표시."""
        if self._output_window is None or not self._output_window.isVisible():
            self._show_output_window()

    # ------------------------------------------------------------------
    # 비동기 분석 헬퍼 (QThread)
    # ------------------------------------------------------------------

    def _run_async(self, run_fn, label: str = "분석") -> None:
        """run_fn()을 백그라운드에서 실행해 UI 프리징을 방지한다."""
        from nuristat.ui.analysis_worker import AnalysisWorker

        if not hasattr(self, "_analysis_workers"):
            self._analysis_workers: list = []

        worker = AnalysisWorker(run_fn, parent=self)
        self._analysis_workers.append(worker)

        def _on_done(result) -> None:
            self._ensure_output_window()
            self._output_window.add_analysis_result(result)
            self.statusbar.showMessage(f"{label} 완료")
            if worker in self._analysis_workers:
                self._analysis_workers.remove(worker)

        def _on_err(msg: str) -> None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "오류", f"{label} 실패:\n{msg}")
            self.statusbar.showMessage(f"{label} 실패")
            if worker in self._analysis_workers:
                self._analysis_workers.remove(worker)

        worker.result_ready.connect(_on_done)
        worker.error_occurred.connect(_on_err)
        self.statusbar.showMessage(f"{label} 중...")
        worker.start()

    def _run_one_sample_ttest(self) -> None:
        """단일표본 T 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.one_sample_ttest_dialog import OneSampleTTestDialog
        dialog = OneSampleTTestDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            ds = self.current_dataset

            def _run():
                from nuristat.analysis.ttests import run_one_sample_ttest
                return run_one_sample_ttest(ds.data, spec["variable"], spec["test_value"])

            self._run_async(_run, "단일표본 T 검정")

    def _run_logistic_regression(self) -> None:
        """로지스틱 회귀 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.logistic_regression_dialog import LogisticRegressionDialog
        dialog = LogisticRegressionDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            ds = self.current_dataset

            def _run():
                from nuristat.analysis import logistic_regression
                return logistic_regression.run_analysis(ds, spec)

            self._run_async(_run, "로지스틱 회귀")

    def _run_factor_analysis(self) -> None:
        """요인분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.factor_analysis_dialog import FactorAnalysisDialog
        dialog = FactorAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            ds = self.current_dataset

            def _run():
                from nuristat.analysis import factor_analysis
                return factor_analysis.run_analysis(ds, spec)

            self._run_async(_run, "요인분석")

    def _run_cluster_analysis(self) -> None:
        """군집분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.cluster_analysis_dialog import ClusterAnalysisDialog
        dialog = ClusterAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            ds = self.current_dataset

            def _run():
                from nuristat.analysis import cluster_analysis
                return cluster_analysis.run_analysis(ds, spec)

            self._run_async(_run, "군집분석")

    def _run_kaplan_meier(self) -> None:
        """Kaplan-Meier 생존분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.kaplan_meier_dialog import KaplanMeierDialog
        dialog = KaplanMeierDialog(self.current_dataset, parent=self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_cox_regression(self) -> None:
        """Cox 비례위험 회귀 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.cox_regression_dialog import CoxRegressionDialog
        dialog = CoxRegressionDialog(self.current_dataset, parent=self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_discriminant_analysis(self) -> None:
        """판별분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.discriminant_analysis_dialog import DiscriminantAnalysisDialog
        dialog = DiscriminantAnalysisDialog(self.current_dataset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec = dialog.get_spec()
            ds = self.current_dataset

            def _run():
                from nuristat.analysis import discriminant_analysis
                return discriminant_analysis.run_analysis(ds, spec)

            self._run_async(_run, "판별분석")

    def _on_analysis_requested(self, analysis_type: str, params: dict) -> None:
        """분석 요청 처리."""
        from nuristat.analysis.registry import AnalysisRegistry

        try:
            registry = AnalysisRegistry()
            result = registry.execute(analysis_type, self.current_dataset, params)

            output = self._get_output()
            output.add_analysis_result(result)

            self._last_analysis_label.setText(f"최근 분석: {analysis_type}")
            self.statusbar.showMessage(f"분석 완료: {analysis_type}")
        except Exception as exc:
            QMessageBox.critical(self, "오류", f"분석 실행 실패:\n{exc}")

    # ── 차트 메뉴 ──────────────────────────────────────────────────────────

    def _open_visualization(self) -> None:
        """고급 시각화 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        dialog = VisualizationDialog(self.current_dataset, self)
        dialog.chart_created.connect(self._on_chart_created)
        dialog.exec()

    def _on_chart_created(self, chart_path: str) -> None:
        """차트 생성 완료 시."""
        output = self._get_output()
        output.add_output(f"📊 차트가 생성되었습니다: {chart_path}", "success")

    def _open_chart_builder(self) -> None:
        """차트 빌더 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return
        from nuristat.ui.chart_builder import ChartBuilderDialog
        dialog = ChartBuilderDialog(self.current_dataset, self)
        dialog.chart_saved.connect(self._on_chart_created)
        dialog.exec()

    def _open_legacy_chart(self, chart_type: str) -> None:
        """기존 대화상자 형태로 특정 차트 유형을 사전 선택해 시각화 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.visualization_dialog import VisualizationDialog
        dialog = VisualizationDialog(self.current_dataset, self, preset_chart_type=chart_type)
        dialog.chart_created.connect(self._on_chart_created)
        dialog.exec()

    # ── 유틸리티 메뉴 ──────────────────────────────────────────────────────

    def _open_data_quality(self) -> None:
        """데이터 품질 진단 다이얼로그 열기."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.data_quality_dialog import DataQualityDialog
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

        from nuristat.ui.dialogs.report_dialog import ReportDialog
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
            "누리스탯 정보",
            "<h2>NuriStat</h2>"
            "<p>버전: 1.0.0</p>"
            "<p>SPSS 스타일 통계 분석 패키지</p>"
            "<p>Python + PySide6 기반</p>"
        )

    def _run_explore(self) -> None:
        """탐색 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.explore_dialog import ExploreDialog
        dialog = ExploreDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_partial_correlation(self) -> None:
        """편상관 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.partial_correlation_dialog import PartialCorrelationDialog
        dialog = PartialCorrelationDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_chi_square_gof(self) -> None:
        """카이제곱 적합도 검정 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.chi_square_gof_dialog import ChiSquareGOFDialog
        dialog = ChiSquareGOFDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_roc_analysis(self) -> None:
        """ROC 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.roc_dialog import ROCDialog
        dialog = ROCDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_cohens_kappa(self) -> None:
        """Cohen's Kappa 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.agreement_dialog import KappaDialog
        dialog = KappaDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_icc(self) -> None:
        """급내 상관계수(ICC) 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.agreement_dialog import ICCDialog
        dialog = ICCDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_bland_altman(self) -> None:
        """Bland-Altman 분석 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.agreement_dialog import BlandAltmanDialog
        dialog = BlandAltmanDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()

    def _run_reliability(self) -> None:
        """신뢰도 분석(Cronbach α) 실행."""
        if self.current_dataset is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 불러오세요")
            return

        from nuristat.ui.dialogs.reliability_dialog import ReliabilityDialog
        dialog = ReliabilityDialog(self.current_dataset, self)
        dialog.analysis_run.connect(self._on_analysis_result)
        dialog.exec()
