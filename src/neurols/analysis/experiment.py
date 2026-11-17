"""Structured experiment tracking, persistence, and multi-method comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neurols.analysis.metrics import summarize_trajectory
from neurols.core.group import GroupConfig
from neurols.core.problem import BatchedProblem
from neurols.core.runner import BatchedRunner
from neurols.core.trajectory import Trajectory
from neurols.strategies.base import BaseStrategy
from neurols.types import StoppingCriteria


@dataclass
class ComparisonResult:
    """Summary of a multi-strategy benchmark comparison."""

    experiment_name: str
    problem_info: dict[str, Any]
    trajectories: dict[str, Trajectory]
    summary_table: list[dict[str, Any]]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "problem_info": self.problem_info,
            "timestamp": self.timestamp,
            "summary_table": self.summary_table,
            "trajectories": {
                name: {
                    "mean_fitness_history": traj.mean_fitness_history,
                    "best_fitness_history": traj.best_fitness_history,
                    "diversity_history": traj.diversity_history,
                    "summary": summarize_trajectory(traj),
                }
                for name, traj in self.trajectories.items()
            },
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentLogger:
    """Orchestrates and saves structured experiments to disk."""

    def __init__(self, base_dir: str | Path = "experiments") -> None:
        self.base_dir = Path(base_dir)

    def create_run_dir(self, experiment_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run_comparison(
        self,
        experiment_name: str,
        problem: BatchedProblem,
        strategies: list[BaseStrategy],
        n_restarts: int = 16,
        max_steps: int | None = 100,
        stopping_criteria: StoppingCriteria | None = None,
        group_config: GroupConfig | None = None,
        seed: int = 42,
    ) -> ComparisonResult:
        """Execute a controlled comparison across multiple local search strategies."""
        trajectories: dict[str, Trajectory] = {}
        summary_rows: list[dict[str, Any]] = []

        stopping = stopping_criteria or StoppingCriteria(max_steps=max_steps)

        # Fix starting states across all strategies for fair evaluation
        B = n_restarts * (group_config.n_agents if group_config else 1)
        import torch

        gen = torch.Generator().manual_seed(seed)
        fixed_initial_states = problem.sample_initial_states((B,), generator=gen)

        for strat in strategies:
            runner = BatchedRunner(
                problem=problem,
                strategy=strat,
                stopping_criteria=stopping,
                group_config=group_config,
                record_history=True,
            )
            traj = runner.run(
                n_restarts=n_restarts,
                initial_states=fixed_initial_states.clone(),
                seed=seed,
            )
            trajectories[strat.name] = traj

            summary = summarize_trajectory(traj)
            summary_rows.append({"strategy": strat.name, **summary})

        # Sort summary rows by mean best fitness descending
        summary_rows.sort(key=lambda r: r["mean_best_fitness"], reverse=True)

        return ComparisonResult(
            experiment_name=experiment_name,
            problem_info={"dimension": problem.dimension, "type": type(problem).__name__},
            trajectories=trajectories,
            summary_table=summary_rows,
            timestamp=datetime.now().isoformat(),
        )
