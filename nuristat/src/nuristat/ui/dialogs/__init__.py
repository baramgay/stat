"""Analysis dialogs for NuriStat UI."""

from nuristat.ui.dialogs.analysis_dialog import AnalysisDialog, VariableSelectorDialog
from nuristat.ui.dialogs.compute_variable_dialog import ComputeVariableDialog
from nuristat.ui.dialogs.descriptives_dialog import DescriptivesDialog
from nuristat.ui.dialogs.frequencies_dialog import FrequenciesDialog
from nuristat.ui.dialogs.manual_data_dialog import ManualDataDialog
from nuristat.ui.dialogs.regression_dialog import RegressionDialog
from nuristat.ui.dialogs.ttest_dialog import IndependentTTestDialog, PairedTTestDialog

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
