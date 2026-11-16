"""Analysis, metrics, experiment logging, plotting, and interpretability tools."""

from neurols.analysis.ablation import (
    DEFAULT_ABLATION_SUITE,
    AblationStrategySpec,
    AblationStudy,
)
from neurols.analysis.experiment import ComparisonResult, ExperimentLogger
from neurols.analysis.interpretability import (
    compute_feature_sensitivities,
    explain_decision,
)
from neurols.analysis.metrics import (
    compute_auc,
    compute_success_rate,
    compute_time_to_target,
    summarize_trajectory,
)
from neurols.analysis.plotting import (
    plot_convergence_curves,
    plot_final_fitness_boxplots,
    plot_multiagent_diversity,
    plot_neural_feature_importance,
    plot_training_convergence,
)

__all__ = [
    "AblationStudy",
    "AblationStrategySpec",
    "DEFAULT_ABLATION_SUITE",
    "compute_auc",
    "compute_success_rate",
    "compute_time_to_target",
    "summarize_trajectory",
    "ComparisonResult",
    "ExperimentLogger",
    "plot_convergence_curves",
    "plot_final_fitness_boxplots",
    "plot_multiagent_diversity",
    "plot_training_convergence",
    "plot_neural_feature_importance",
    "explain_decision",
    "compute_feature_sensitivities",
]
