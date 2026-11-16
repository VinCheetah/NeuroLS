"""Hyperparameter optimization and tuning for Local Search strategies.

Allows automated exploration and tuning of hyperparameters (e.g. Tabu tenure,
Simulated Annealing cooling rate and temperature) via Grid Search, Random Search,
and CMA-ES parameter optimization.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurols.core.problem import BatchedProblem
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec


@dataclass
class TuningTrial:
    """Record of an evaluated hyperparameter configuration."""

    params: dict[str, Any]
    score: float
    std_score: float
    metadata: dict[str, Any]


@dataclass
class TuningResult:
    """Summary of hyperparameter optimization."""

    strategy_name: str
    best_params: dict[str, Any]
    best_score: float
    trials: list[TuningTrial]

    def to_summary_table(self) -> list[dict[str, Any]]:
        """Format trials into a sorted table list."""
        sorted_trials = sorted(self.trials, key=lambda t: t.score, reverse=True)
        return [
            {"rank": i + 1, "score": t.score, "std": t.std_score, **t.params}
            for i, t in enumerate(sorted_trials)
        ]


class HyperparameterTuner:
    """Tuner engine for optimizing strategy hyperparameters."""

    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        problem: BatchedProblem,
        eval_fn: Callable[[BaseStrategy, BatchedProblem], tuple[float, float]],
    ) -> None:
        """Initialize tuner.

        Args:
            strategy_class: Class of the strategy to tune.
            problem: BatchedProblem benchmark instance.
            eval_fn: Function taking (strategy, problem) -> (mean_score, std_score).
        """
        self.strategy_class = strategy_class
        self.problem = problem
        self.eval_fn = eval_fn
        # Create a dummy instance to inspect specs
        dummy = strategy_class()
        self.specs: list[HyperparameterSpec] = dummy.hyperparameter_specs

    def random_search(
        self,
        n_trials: int = 20,
        fixed_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> TuningResult:
        """Perform Random Search over tunable hyperparameters."""
        rng = random.Random(seed)
        fixed = fixed_params or {}
        tunable_specs = [s for s in self.specs if s.name not in fixed]

        trials: list[TuningTrial] = []
        best_score = -float("inf")
        best_params: dict[str, Any] = {}

        for _ in range(n_trials):
            sampled_params: dict[str, Any] = dict(fixed)
            for spec in tunable_specs:
                if spec.choices is not None:
                    sampled_params[spec.name] = rng.choice(spec.choices)
                elif spec.param_type == "int":
                    low = int(spec.low if spec.low is not None else 1)
                    high = int(spec.high if spec.high is not None else 20)
                    sampled_params[spec.name] = rng.randint(low, high)
                elif spec.param_type == "float":
                    low = float(spec.low if spec.low is not None else 0.0)
                    high = float(spec.high if spec.high is not None else 1.0)
                    if spec.log_scale and low > 0:
                        val = 10 ** rng.uniform(np.log10(low), np.log10(high))
                    else:
                        val = rng.uniform(low, high)
                    sampled_params[spec.name] = round(val, 5)
                elif spec.param_type == "bool":
                    sampled_params[spec.name] = rng.choice([True, False])

            # Instantiate strategy
            strategy = self.strategy_class(**sampled_params)
            mean_score, std_score = self.eval_fn(strategy, self.problem)

            trial = TuningTrial(
                params=sampled_params,
                score=mean_score,
                std_score=std_score,
                metadata={},
            )
            trials.append(trial)

            if mean_score > best_score:
                best_score = mean_score
                best_params = sampled_params

        return TuningResult(
            strategy_name=self.strategy_class.__name__,
            best_params=best_params,
            best_score=best_score,
            trials=trials,
        )

    def grid_search(
        self,
        grid_values: dict[str, list[Any]],
        fixed_params: dict[str, Any] | None = None,
    ) -> TuningResult:
        """Perform exhaustive Grid Search over specified hyperparameter values."""
        fixed = fixed_params or {}
        keys = list(grid_values.keys())
        combinations = list(itertools.product(*[grid_values[k] for k in keys]))

        trials: list[TuningTrial] = []
        best_score = -float("inf")
        best_params: dict[str, Any] = {}

        for combo in combinations:
            params = dict(fixed)
            params.update(dict(zip(keys, combo, strict=True)))

            strategy = self.strategy_class(**params)
            mean_score, std_score = self.eval_fn(strategy, self.problem)

            trial = TuningTrial(
                params=params,
                score=mean_score,
                std_score=std_score,
                metadata={},
            )
            trials.append(trial)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        return TuningResult(
            strategy_name=self.strategy_class.__name__,
            best_params=best_params,
            best_score=best_score,
            trials=trials,
        )
