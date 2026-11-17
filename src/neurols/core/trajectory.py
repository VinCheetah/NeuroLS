"""Trajectory tracking and run results for Local Search executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from neurols.types import StepRecord


@dataclass
class Trajectory:
    """Complete record of a local search rollout across a batch of agents.

    Attributes:
        batch_size: Total number of agents in the batch B.
        dimension: Search space dimensionality N.
        max_steps: Total allocated step budget.
        initial_states: Starting states tensor, shape (B, N).
        final_states: Final states tensor upon completion, shape (B, N).
        best_states: Best states found by each agent, shape (B, N).
        initial_fitness: Fitness at step 0, shape (B,).
        final_fitness: Fitness at final step, shape (B,).
        best_fitness: Best fitness achieved across the run, shape (B,).
        steps_taken: Actual number of steps completed per agent, shape (B,).
        step_history: Optional detailed per-step records.
        mean_fitness_history: Evolution of batch mean fitness across steps.
        best_fitness_history: Evolution of batch best fitness across steps.
        diversity_history: Evolution of multi-agent diversity across steps.
        metadata: Additional diagnostic info (strategy name, hyperparameters, timing).
    """

    batch_size: int
    dimension: int
    max_steps: int
    initial_states: torch.Tensor
    final_states: torch.Tensor
    best_states: torch.Tensor
    initial_fitness: torch.Tensor
    final_fitness: torch.Tensor
    best_fitness: torch.Tensor
    steps_taken: torch.Tensor
    step_history: list[StepRecord] = field(default_factory=list[StepRecord])
    mean_fitness_history: list[float] = field(default_factory=list[float])
    best_fitness_history: list[float] = field(default_factory=list[float])
    diversity_history: list[float] = field(default_factory=list[float])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def final_mean_fitness(self) -> float:
        """Average final fitness across all agents."""
        return float(self.final_fitness.mean().item())

    @property
    def best_overall_fitness(self) -> float:
        """Highest fitness reached by any agent in the batch."""
        return float(self.best_fitness.max().item())

    @property
    def average_best_fitness(self) -> float:
        """Average of each agent's best-so-far fitness."""
        return float(self.best_fitness.mean().item())

    def get_summary(self) -> dict[str, float]:
        """Compute key summary statistics."""
        best_arr = self.best_fitness.detach().cpu().numpy()
        steps_arr = self.steps_taken.detach().cpu().numpy()
        return {
            "mean_best_fitness": float(np.mean(best_arr)),
            "std_best_fitness": float(np.std(best_arr)),
            "max_best_fitness": float(np.max(best_arr)),
            "min_best_fitness": float(np.min(best_arr)),
            "median_best_fitness": float(np.median(best_arr)),
            "mean_steps": float(np.mean(steps_arr)),
        }
