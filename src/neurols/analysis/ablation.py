"""Feature ablation study and automated training-before-benchmark pipeline.

Allows creating and training multiple neural local search strategies with varying
feature sets (Delta only, Ranked Delta, Action-specific Peer changes, Full Context),
both per-action and full-neighborhood architectures, and comparing them on test landscapes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from neurols.analysis.experiment import ComparisonResult, ExperimentLogger
from neurols.analysis.plotting import plot_convergence_curves, plot_final_fitness_boxplots
from neurols.core.group import GroupConfig
from neurols.core.problem import BatchedProblem
from neurols.strategies.base import BaseStrategy
from neurols.strategies.classical import HillClimber, TabuSearch
from neurols.strategies.neural.features import FeatureAblationLevel, FeatureConfig
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.training.cma_es import CMAESOptimizer


@dataclass
class AblationStrategySpec:
    """Specification of a neural strategy variant for feature ablation."""

    name: str
    ablation_level: FeatureAblationLevel
    policy_type: str  # "per_action", "invariant_nnet", "linear"
    description: str


DEFAULT_ABLATION_SUITE = [
    AblationStrategySpec(
        name="NN_DeltaOnly (PerAction)",
        ablation_level=FeatureAblationLevel.DELTA_ONLY,
        policy_type="per_action",
        description="Individual action scoring using only raw delta fitness.",
    ),
    AblationStrategySpec(
        name="NN_RankedDelta (InvariantNNet)",
        ablation_level=FeatureAblationLevel.RANKED_DELTA_ONLY,
        policy_type="invariant_nnet",
        description="All actions scored jointly via InvariantNNet using normalized delta ranks.",
    ),
    AblationStrategySpec(
        name="NN_DeltaPeers (PerAction)",
        ablation_level=FeatureAblationLevel.DELTA_AND_PEERS,
        policy_type="per_action",
        description="Individual action scoring with delta and sorted peer distance changes (+/-1).",
    ),
    AblationStrategySpec(
        name="NN_DeltaPeersRank (InvariantNNet)",
        ablation_level=FeatureAblationLevel.DELTA_PEERS_RANK,
        policy_type="invariant_nnet",
        description="InvariantNNet with delta, peer distance changes, and group fitness ranks.",
    ),
    AblationStrategySpec(
        name="NN_FullContext (InvariantNNet)",
        ablation_level=FeatureAblationLevel.FULL,
        policy_type="invariant_nnet",
        description="InvariantNNet with all features (deltas, peers, ranks, stagnation, consensus).",
    ),
]


class AblationStudy:
    """Manages the full ablation pipeline: training neural variants, then benchmarking."""

    def __init__(
        self,
        train_problem: BatchedProblem,
        test_problem: BatchedProblem,
        group_config: GroupConfig | None = None,
        train_generations: int = 10,
        eval_restarts: int = 8,
        eval_steps: int = 40,
        cma_popsize: int = 8,
        test_restarts: int = 16,
        test_steps: int = 80,
        seed: int = 42,
    ) -> None:
        self.train_problem = train_problem
        self.test_problem = test_problem
        self.group_config = group_config or GroupConfig(n_agents=1)
        self.train_generations = train_generations
        self.eval_restarts = eval_restarts
        self.eval_steps = eval_steps
        self.cma_popsize = cma_popsize
        self.test_restarts = test_restarts
        self.test_steps = test_steps
        self.seed = seed

    def run(
        self,
        specs: list[AblationStrategySpec] | None = None,
        include_classical_baselines: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
        output_dir: Path | str = "experiments/ablation",
    ) -> ComparisonResult:
        """Execute training of all neural variants followed by a rigorous test benchmark."""
        specs_to_run = specs or DEFAULT_ABLATION_SUITE
        trained_strategies: list[BaseStrategy] = []
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        n_agents = self.group_config.n_agents
        dim = self.train_problem.dimension

        # 1. Train each neural strategy variant with CMA-ES
        for i, spec in enumerate(specs_to_run):
            if progress_callback:
                progress_callback(spec.name, i + 1, len(specs_to_run))

            # Instantiate strategy
            feat_cfg = FeatureConfig.from_ablation_level(spec.ablation_level, n_agents=n_agents)
            strategy = NeuralLocalSearchStrategy(
                name=spec.name,
                policy_type=spec.policy_type,  # type: ignore
                feature_config=feat_cfg,
                n_actions=dim,
                n_agents=n_agents,
                max_steps=self.eval_steps,
                seed=self.seed,
            )

            # CMA-ES training on training landscape
            optimizer = CMAESOptimizer(
                strategy=strategy,
                problem=self.train_problem,
                eval_restarts=self.eval_restarts,
                eval_steps=self.eval_steps,
                population_size=self.cma_popsize,
                group_config=self.group_config,
                seed=self.seed + i * 100,
            )

            ckpt_dir = out_path / "checkpoints"
            optimizer.train(n_generations=self.train_generations, checkpoint_dir=ckpt_dir)
            trained_strategies.append(strategy)

        # 2. Add classical baselines for fair comparison
        if include_classical_baselines:
            trained_strategies.append(HillClimber())
            trained_strategies.append(TabuSearch(tenure=5))

        # 3. Benchmark all strategies on the test problem
        logger = ExperimentLogger(base_dir=out_path)
        comp_result = logger.run_comparison(
            experiment_name="feature_ablation_study",
            problem=self.test_problem,
            strategies=trained_strategies,
            n_restarts=self.test_restarts,
            max_steps=self.test_steps,
            group_config=self.group_config,
            seed=self.seed + 999,
        )

        # 4. Generate comparison plots
        plot_convergence_curves(
            comp_result.trajectories,
            out_path / "ablation_convergence.png",
            title=f"Feature Ablation Convergence (N={dim}, Agents={n_agents})",
        )
        plot_final_fitness_boxplots(
            comp_result.trajectories,
            out_path / "ablation_boxplots.png",
            title=f"Feature Ablation Fitness Distribution (N={dim}, Agents={n_agents})",
        )

        comp_result.save(out_path / "ablation_summary.json")
        return comp_result
