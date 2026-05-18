"""Analysis dialogs for StatWorkbench UI."""

from statworkbench.ui.dialogs.analysis_dialog import AnalysisDialog, VariableSelectorDialog
from statworkbench.ui.dialogs.descriptives_dialog import DescriptivesDialog
from statworkbench.ui.dialogs.frequencies_dialog import FrequenciesDialog
from statworkbench.ui.dialogs.ttest_dialog import IndependentTTestDialog, PairedTTestDialog
from statworkbench.ui.dialogs.regression_dialog import RegressionDialog
from statworkbench.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
from statworkbench.ui.dialogs.manual_data_dialog import ManualDataDialog

__all__ = [
    "AnalysisDialog",
    "VariableSelectorDialog",
    "DescriptivesDialog",
    "FrequenciesDialog",
    "IndependentTTestDialog",
    "PairedTTestDialog",
    "RegressionDialog",
    "ComputeVariableDialog",
    "ManualDataDialog",
]
