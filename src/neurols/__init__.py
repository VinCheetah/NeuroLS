"""NeuroLS: A Modular Local Search and Neuroevolution Framework in PyTorch.

Enables studying local search heuristics, metaheuristics, multi-agent dynamics,
and neuroevolution (CMA-ES) on combinatorial and continuous landscapes.
"""

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
from neurols.cli.app import app, main
from neurols.core.group import GroupConfig, GroupContext, GroupMetrics
from neurols.core.problem import BaseProblem, BatchedProblem
from neurols.core.runner import BatchedRunner, TraditionalRunner
from neurols.core.trajectory import Trajectory
from neurols.problems.nk_landscape import MultiNKLandscape, NKLandscape
from neurols.problems.onemax import OneMax
from neurols.strategies.base import BaseStrategy
from neurols.strategies.classical import (
    FirstImprovement,
    HillClimber,
    RandomWalk,
    SimulatedAnnealing,
    TabuSearch,
)
from neurols.strategies.neural import (
    BaseNeuralPolicy,
    DeepSetsPolicy,
    EquivariantMLPPolicy,
    FeatureAblationLevel,
    FeatureConfig,
    FeatureExtractor,
    InvariantModule,
    InvariantNNet,
    LinearScorer,
    Net,
    NeuralLocalSearchStrategy,
    PerActionScorer,
    compute_action_peer_deltas,
    compute_normalized_ranks,
)
from neurols.strategies.tuning import HyperparameterTuner, TuningResult, TuningTrial
from neurols.training.base import BaseOptimizer, GenerationRecord, TrainResult
from neurols.training.cma_es import CMAESOptimizer
from neurols.types import (
    AnnealingSchedule,
    DeviceType,
    HyperparameterSpec,
    MoveResult,
    MoveSelectionMode,
    NeighborhoodType,
    ProblemType,
    StepRecord,
)

__version__ = "0.1.0"

__all__ = [
    # Problems
    "BaseProblem",
    "BatchedProblem",
    "NKLandscape",
    "MultiNKLandscape",
    "OneMax",
    "ProblemType",
    "NeighborhoodType",
    # Core & Multi-Agent
    "GroupConfig",
    "GroupContext",
    "GroupMetrics",
    "BatchedRunner",
    "TraditionalRunner",
    "Trajectory",
    "StepRecord",
    "MoveResult",
    "DeviceType",
    # Strategies
    "BaseStrategy",
    "HillClimber",
    "FirstImprovement",
    "RandomWalk",
    "SimulatedAnnealing",
    "TabuSearch",
    "AnnealingSchedule",
    "HyperparameterSpec",
    "HyperparameterTuner",
    "TuningResult",
    "TuningTrial",
    # Neural Strategies
    "NeuralLocalSearchStrategy",
    "FeatureConfig",
    "FeatureExtractor",
    "FeatureAblationLevel",
    "compute_action_peer_deltas",
    "compute_normalized_ranks",
    "BaseNeuralPolicy",
    "LinearScorer",
    "EquivariantMLPPolicy",
    "DeepSetsPolicy",
    "Net",
    "InvariantModule",
    "InvariantNNet",
    "PerActionScorer",
    "MoveSelectionMode",
    # Training & Neuroevolution
    "BaseOptimizer",
    "CMAESOptimizer",
    "GenerationRecord",
    "TrainResult",
    "AblationStudy",
    "AblationStrategySpec",
    "DEFAULT_ABLATION_SUITE",
    # Analysis & Plotting
    "ExperimentLogger",
    "ComparisonResult",
    "summarize_trajectory",
    "compute_auc",
    "compute_success_rate",
    "compute_time_to_target",
    "plot_convergence_curves",
    "plot_final_fitness_boxplots",
    "plot_multiagent_diversity",
    "plot_training_convergence",
    "plot_neural_feature_importance",
    "explain_decision",
    "compute_feature_sensitivities",
    # CLI
    "app",
    "main",
]
