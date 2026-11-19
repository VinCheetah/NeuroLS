"""Multi-agent and multi-model comparative experimental framework.

Compares neural local search architectures (InvariantNNet, Peer-Invariant action scorers,
per-action neural variants) across different agent population sizes (e.g. 1, 4, 10 agents),
with and without information sharing, under customizable stopping criteria (steps, time, budget).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurols.analysis.plotting import (
    plot_fitness_vs_agents,
    plot_multi_training_convergence,
    plot_multiagent_diversity,
    plot_sharing_gain_bars,
    plot_step_trajectories,
)
from neurols.core.group import GroupConfig
from neurols.core.problem import BatchedProblem
from neurols.core.runner import BatchedRunner
from neurols.core.trajectory import Trajectory
from neurols.strategies.base import BaseStrategy
from neurols.strategies.classical import (
    FirstImprovement,
    HillClimber,
    RandomWalk,
    SimulatedAnnealing,
    TabuSearch,
)
from neurols.strategies.neural.features import (
    FEATURE_REGISTRY,
    FeatureConfig,
)
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.training.base import GenerationRecord
from neurols.training.cma_es import CMAESOptimizer
from neurols.types import StoppingCriteria


@dataclass
class ModelSpec:
    """Specification of a strategy or neural model variant."""

    name: str
    model_type: str = "neural"  # "neural" or "classical"
    policy_type: str = "linear"  # "linear", "mlp", "deepsets", "peer_invariant", etc.
    eval_mode: str = "per_action"  # "per_action" or "all_actions"
    features: str | list[str] | None = None
    n_agents: int = 1
    description: str = ""

    def to_strategy(
        self,
        n_actions: int = 20,
        n_agents: int = 1,
        steps: int = 100,
        seed: int | None = None,
    ) -> BaseStrategy:
        """Instantiate a runnable BaseStrategy from this ModelSpec."""
        agents = max(self.n_agents, n_agents)
        if self.model_type == "classical":
            name_lower = self.name.lower()
            if "first" in name_lower:
                return FirstImprovement(order="random", seed=seed, name=self.name)
            elif "tabu" in name_lower:
                return TabuSearch(tenure=5, name=self.name)
            elif "random" in name_lower:
                return RandomWalk(exploration_rate=1.0, seed=seed, name=self.name)
            elif "simulated" in name_lower or "anneal" in name_lower:
                return SimulatedAnnealing(max_steps=steps, seed=seed, name=self.name)
            else:
                return HillClimber(name=self.name)
        else:
            feat_cfg = FeatureConfig.from_specs(self.features, mode=self.eval_mode)
            return NeuralLocalSearchStrategy(
                name=self.name,
                policy_type=self.policy_type,  # type: ignore
                eval_mode=self.eval_mode,
                feature_config=feat_cfg,
                n_actions=n_actions,
                n_agents=agents,
                max_steps=steps,
                seed=seed,
            )


# Suite: Classical Baselines Only
BASELINES_SUITE = [
    ModelSpec(name="HillClimber", model_type="classical", description="Steepest ascent greedy hill climber."),
    ModelSpec(name="FirstImprovement", model_type="classical", description="First improving neighbor local search."),
    ModelSpec(name="TabuSearch", model_type="classical", description="Tabu Search with short-term tenure memory."),
    ModelSpec(name="SimulatedAnnealing", model_type="classical", description="Simulated Annealing with geometric cooling schedule."),
    ModelSpec(name="RandomWalk", model_type="classical", description="Unbiased random walk exploration."),
]

# Preset 1: Action-Level Neural Local Search vs Baselines (User Focus)
PER_ACTION_STUDY_SUITE = [
    ModelSpec(
        name="PeerInvariant_BasicSharing (Per-Action)",
        model_type="neural",
        policy_type="peer_invariant_per_action",
        eval_mode="per_action",
        features="@action_basic,peer_deltas",
        description="Per-action scorer with DeepSets invariant module over basic +/-1 peer distance changes.",
    ),
    ModelSpec(
        name="NN_PerAction_RankedDelta (Per-Action)",
        model_type="neural",
        policy_type="per_action",
        eval_mode="per_action",
        features="ranked_delta",
        description="Per-action scorer using dual-pole normalized ranks (unshared).",
    ),
    ModelSpec(
        name="NN_PerAction_DeltaOnly (Per-Action)",
        model_type="neural",
        policy_type="per_action",
        eval_mode="per_action",
        features="delta",
        description="Per-action scorer using raw delta fitness only (unshared).",
    ),
    ModelSpec(
        name="HillClimber (Baseline)",
        model_type="classical",
        description="Classical Greedy Steepest Ascent baseline (group-agnostic).",
    ),
    ModelSpec(
        name="TabuSearch (Baseline)",
        model_type="classical",
        description="Classical Tabu Search baseline (group-agnostic).",
    ),
    ModelSpec(
        name="SimulatedAnnealing (Baseline)",
        model_type="classical",
        description="Simulated Annealing baseline.",
    ),
]

# Preset 2: All Architectures Comparison
ARCHITECTURES_SUITE = [
    ModelSpec(
        name="LinearScorer (Per-Action)",
        model_type="neural",
        policy_type="linear",
        eval_mode="per_action",
        features="@basic",
        description="Interpretable linear model combining delta and rank.",
    ),
    ModelSpec(
        name="MLP_PerAction (Per-Action)",
        model_type="neural",
        policy_type="mlp",
        eval_mode="per_action",
        features="@basic",
        description="Deep MLP evaluating candidate moves independently.",
    ),
    ModelSpec(
        name="DeepSets (All-Actions)",
        model_type="neural",
        policy_type="deepsets",
        eval_mode="all_actions",
        features="@basic",
        description="Encoder-Pool-Decoder architecture with global neighborhood context.",
    ),
    ModelSpec(
        name="PeerInvariant (Per-Action)",
        model_type="neural",
        policy_type="peer_invariant_per_action",
        eval_mode="per_action",
        features="@action_basic,peer_deltas",
        description="Per-action scorer with peer invariance over distance deltas.",
    ),
    ModelSpec(
        name="HillClimber (Baseline)",
        model_type="classical",
        description="Classical Greedy Steepest Ascent baseline.",
    ),
    ModelSpec(
        name="TabuSearch (Baseline)",
        model_type="classical",
        description="Tabu Search baseline with short-term memory tenure.",
    ),
]

# Preset 3: Full Comparison Suite
FULL_COMPARISON_SUITE = [
    ModelSpec(
        name="NN_Linear_Basic",
        model_type="neural",
        policy_type="linear",
        eval_mode="per_action",
        features="@basic",
        description="Linear policy scoring delta and rank.",
    ),
    ModelSpec(
        name="PeerInvariant_Sharing",
        model_type="neural",
        policy_type="peer_invariant_per_action",
        eval_mode="per_action",
        features="@action_basic,peer_deltas",
        description="Per-action scorer with peer invariance over distance deltas.",
    ),
    ModelSpec(
        name="NN_RankedDelta",
        model_type="neural",
        policy_type="per_action",
        eval_mode="per_action",
        features="ranked_delta",
        description="Scale-invariant normalized ranks per action.",
    ),
    ModelSpec(
        name="HillClimber (Baseline)",
        model_type="classical",
        description="Greedy Steepest Ascent baseline.",
    ),
    ModelSpec(
        name="TabuSearch (Baseline)",
        model_type="classical",
        description="Tabu Search baseline with short-term memory tenure.",
    ),
    ModelSpec(
        name="SimulatedAnnealing (Baseline)",
        model_type="classical",
        description="Simulated Annealing baseline.",
    ),
]

DEFAULT_MULTIAGENT_MODELS = [
    ModelSpec(
        name="PeerInvariant_Sharing",
        model_type="neural",
        policy_type="peer_invariant_per_action",
        eval_mode="per_action",
        features="@action_basic,peer_deltas",
        description="Peer-invariant neural strategy with distance sharing.",
    ),
    ModelSpec(
        name="NN_Linear (Unshared)",
        model_type="neural",
        policy_type="linear",
        eval_mode="per_action",
        features="@basic",
        description="Linear neural model with delta and ranked delta.",
    ),
    ModelSpec(
        name="HillClimber (Baseline)",
        model_type="classical",
        description="Greedy Steepest Ascent baseline.",
    ),
    ModelSpec(
        name="TabuSearch (Baseline)",
        model_type="classical",
        description="Tabu Search baseline with short-term memory tenure.",
    ),
]


def get_study_suite(preset: str) -> list[ModelSpec]:
    """Retrieve predefined model suites by name."""
    p = preset.lower().replace("-", "_").replace(" ", "_")
    if "base" in p:
        return BASELINES_SUITE
    elif "action" in p:
        return PER_ACTION_STUDY_SUITE
    elif "arch" in p:
        return ARCHITECTURES_SUITE
    elif "full" in p or "all" in p:
        return FULL_COMPARISON_SUITE
    elif "multi" in p:
        return DEFAULT_MULTIAGENT_MODELS
    else:
        return ARCHITECTURES_SUITE


@dataclass
class MultiAgentRunRecord:
    """Record of a single experimental condition."""

    model_name: str
    n_agents: int
    share_info: bool
    is_group_aware: bool
    mean_team_best_fitness: float
    std_team_best_fitness: float
    mean_agent_fitness: float
    mean_diversity: float
    mean_steps: float
    mean_evaluations: int
    mean_elapsed_seconds: float
    termination_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class MultiAgentStudyResult:
    """Complete results of a multi-agent comparative study."""

    experiment_name: str
    problem_dimension: int
    agent_counts: list[int]
    sharing_modes: list[bool]
    stopping_criteria: dict[str, Any]
    records: list[MultiAgentRunRecord]
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    training_histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "problem_dimension": self.problem_dimension,
            "agent_counts": self.agent_counts,
            "sharing_modes": self.sharing_modes,
            "stopping_criteria": self.stopping_criteria,
            "records": [asdict(r) for r in self.records],
            "training_histories": self.training_histories,
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class MultiAgentStudy:
    """Orchestrates comprehensive multi-model, multi-agent comparative experiments.

    Correctly identifies group-aware vs group-agnostic strategies (e.g. HillClimber),
    eliminating redundant computation when sharing has no mathematical effect.
    """

    def __init__(
        self,
        train_problem: BatchedProblem,
        test_problem: BatchedProblem,
        agent_counts: list[int] | None = None,
        sharing_modes: list[bool] | None = None,
        models: list[ModelSpec] | None = None,
        stopping_criteria: StoppingCriteria | None = None,
        train_generations: int = 8,
        test_restarts: int = 12,
        seed: int = 42,
    ) -> None:
        self.train_problem = train_problem
        self.test_problem = test_problem
        self.agent_counts = agent_counts or [1, 4, 10]
        self.sharing_modes = sharing_modes if sharing_modes is not None else [True, False]
        self.models = models or DEFAULT_MULTIAGENT_MODELS
        self.stopping_criteria = stopping_criteria or StoppingCriteria(max_steps=50)
        self.train_generations = train_generations
        self.test_restarts = test_restarts
        self.seed = seed

    def _train_model(
        self,
        spec: ModelSpec,
        n_agents: int,
        share_info: bool,
    ) -> tuple[BaseStrategy, list[GenerationRecord]]:
        """Instantiate and optionally pre-train the model with CMA-ES."""
        if spec.model_type == "classical":
            name_lower = spec.name.lower()
            if "first" in name_lower:
                return FirstImprovement(), []
            elif "random" in name_lower:
                return RandomWalk(), []
            elif "tabu" in name_lower:
                return TabuSearch(tenure=5), []
            else:
                return HillClimber(), []

        # Neural Model
        group_cfg = GroupConfig(
            n_agents=n_agents,
            share_distances=share_info,
            share_delta_scores=share_info,
            share_ranks=share_info,
            share_approachs=share_info,
        )
        feat_cfg = FeatureConfig.from_specs(spec.features, mode=spec.eval_mode)

        strategy = NeuralLocalSearchStrategy(
            name=spec.name,
            policy_type=spec.policy_type,  # type: ignore
            eval_mode=spec.eval_mode,
            feature_config=feat_cfg,
            n_actions=self.train_problem.dimension,
            n_agents=n_agents,
            seed=self.seed,
        )

        history: list[GenerationRecord] = []
        if self.train_generations > 0:
            optimizer = CMAESOptimizer(
                strategy=strategy,
                problem=self.train_problem,
                eval_restarts=4,
                eval_steps=min(self.stopping_criteria.max_steps or 30, 30),
                population_size=6,
                group_config=group_cfg,
                seed=self.seed,
            )
            res = optimizer.train(n_generations=self.train_generations)
            history = res.history

        return strategy, history

    def run(
        self,
        output_dir: Path | str = "experiments/multi_agent_study",
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> MultiAgentStudyResult:
        """Run the full comparative study across all conditions."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        records: list[MultiAgentRunRecord] = []
        trajectories: dict[str, Trajectory] = {}
        training_histories: dict[str, list[dict[str, Any]]] = {}
        raw_training_histories: dict[str, list[GenerationRecord]] = {}

        # 1. Pre-calculate active conditions, taking into account group awareness
        conditions: list[tuple[ModelSpec, int, bool, bool]] = []
        for spec in self.models:
            for m in self.agent_counts:
                # Test whether model at m is group-aware
                feat_cfg = FeatureConfig.from_specs(spec.features, mode=spec.eval_mode)

                is_group_aware = (
                    spec.model_type == "neural"
                    and m > 1
                    and any(
                        FEATURE_REGISTRY[f].is_multi_agent
                        for f in feat_cfg.features
                        if f in FEATURE_REGISTRY
                    )
                )

                if not is_group_aware:
                    # Group-agnostic (or M=1): only run once, sharing is irrelevant
                    conditions.append((spec, m, False, False))
                else:
                    for share in self.sharing_modes:
                        conditions.append((spec, m, share, True))

        total_conditions = len(conditions)

        for cond_idx, (spec, m, share, is_group_aware) in enumerate(conditions, 1):
            cond_desc = (
                f"{spec.name} | M={m} | Sharing={'✓' if share else '✗'}"
                if is_group_aware
                else f"{spec.name} | M={m} | (Agnostic)"
            )
            if progress_callback:
                progress_callback(cond_desc, cond_idx, total_conditions)

            # 2. Train / instantiate strategy
            strategy, train_hist = self._train_model(spec, n_agents=m, share_info=share)
            if train_hist:
                hist_key = f"{spec.name}_M{m}_{'sharing' if share else 'noshare'}"
                training_histories[hist_key] = [asdict(h) for h in train_hist]
                raw_training_histories[hist_key] = train_hist

            # 3. Run rollout on test problem
            group_cfg = GroupConfig(
                n_agents=m,
                share_distances=share,
                share_delta_scores=share,
                share_ranks=share,
                share_approachs=share,
            )

            runner = BatchedRunner(
                problem=self.test_problem,
                strategy=strategy,
                stopping_criteria=self.stopping_criteria,
                group_config=group_cfg,
                record_history=True,
            )

            traj = runner.run(n_restarts=self.test_restarts, seed=self.seed + cond_idx)
            traj_key = (
                f"{spec.name}_M{m}_{'sharing' if share else 'noshare'}"
                if is_group_aware
                else f"{spec.name}_M{m}"
            )
            trajectories[traj_key] = traj

            # 4. Compute group / team performance metrics
            n_groups = self.test_restarts
            best_fits = traj.best_fitness.view(n_groups, m)
            team_best_per_group = best_fits.max(dim=1).values.detach().cpu().numpy()

            mean_team_best = float(np.mean(team_best_per_group))
            std_team_best = float(np.std(team_best_per_group))
            mean_agent_fit = float(traj.final_mean_fitness)
            mean_div = float(np.mean(traj.diversity_history)) if traj.diversity_history else 0.0
            mean_steps_val = float(traj.steps_taken.float().mean().item())

            m_label = f"{re.sub(r'_M\d+$', '', spec.name)}_M{m}"
            rec = MultiAgentRunRecord(
                model_name=m_label,
                n_agents=m,
                share_info=share,
                is_group_aware=is_group_aware,
                mean_team_best_fitness=mean_team_best,
                std_team_best_fitness=std_team_best,
                mean_agent_fitness=mean_agent_fit,
                mean_diversity=mean_div,
                mean_steps=mean_steps_val,
                mean_evaluations=traj.total_evaluations,
                mean_elapsed_seconds=traj.elapsed_seconds,
                termination_reasons={traj.termination_reason: 1},
            )
            records.append(rec)

        result = MultiAgentStudyResult(
            experiment_name="MultiAgent_Architecture_Comparison",
            problem_dimension=self.test_problem.dimension,
            agent_counts=self.agent_counts,
            sharing_modes=self.sharing_modes,
            stopping_criteria=asdict(self.stopping_criteria),
            records=records,
            trajectories=trajectories,
            training_histories=training_histories,
        )

        result.save(out_path / "study_results.json")

        # 5. Generate comprehensive comparison plots
        plot_fitness_vs_agents(
            records=records,
            save_path=out_path / "fitness_vs_agents.png",
            title=f"Team Best Fitness vs. Agent Count (N={self.test_problem.dimension})",
        )
        plot_sharing_gain_bars(
            records=records,
            save_path=out_path / "sharing_gain.png",
            title="Fitness Boost from Information Sharing Across Group Sizes",
        )
        plot_step_trajectories(
            trajectories=trajectories,
            save_path=out_path / "step_convergence.png",
            title="Step-by-Step Convergence Across Conditions",
        )
        plot_multiagent_diversity(
            trajectories=trajectories,
            save_path=out_path / "diversity_dynamics.png",
            title="Multi-Agent Diversity Evolution Across Models",
        )
        if raw_training_histories:
            plot_multi_training_convergence(
                histories=raw_training_histories,
                save_path=out_path / "training_curves.png",
                title="CMA-ES Neuroevolution Training Progression",
            )

        return result
