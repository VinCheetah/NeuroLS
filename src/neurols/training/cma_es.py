"""CMA-ES Neuroevolution Optimizer for Neural Local Search.

Optimizes neural network policy parameters using Covariance Matrix Adaptation
Evolution Strategy (Hansen, 2016), leveraging vectorized batched rollouts for high throughput.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cma
import numpy as np
import torch

from neurols.core.group import GroupConfig
from neurols.core.problem import BatchedProblem
from neurols.core.runner import BatchedRunner
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.training.base import BaseOptimizer, GenerationRecord, TrainResult


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES parameter optimizer for neural local search policies."""

    def __init__(
        self,
        strategy: NeuralLocalSearchStrategy,
        problem: BatchedProblem,
        eval_restarts: int = 8,
        eval_steps: int = 50,
        population_size: int | None = None,
        sigma0: float = 0.5,
        group_config: GroupConfig | None = None,
        use_common_random_numbers: bool = True,
        seed: int | None = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            strategy=strategy,
            problem=problem,
            eval_restarts=eval_restarts,
            eval_steps=eval_steps,
            group_config=group_config,
            device=device,
        )
        self.sigma0 = sigma0
        self.seed = seed
        self.use_common_random_numbers = use_common_random_numbers

        # Initial parameter vector from policy
        self.x0 = self.strategy.get_weights_vector()
        self.dimension = len(self.x0)

        # Configure CMA options
        cma_opts: dict[str, Any] = {
            "verb_disp": 0,
            "verb_log": 0,
            "seed": seed if seed is not None else 0,
        }
        if population_size is not None:
            cma_opts["popsize"] = population_size

        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, cma_opts)
        self.population_size = self.es.popsize

        # Generation counter & best tracker
        self.generation = 0
        self.best_fitness = -float("inf")
        self.best_weights: np.ndarray = self.x0.copy()
        self.total_evaluations = 0

        # Fixed starting states if using Common Random Numbers (CRN)
        self._fixed_init_states: torch.Tensor | None = None
        if self.use_common_random_numbers:
            B = self.eval_restarts * self.group_config.n_agents
            g = torch.Generator(device=self.device)
            if seed is not None:
                g.manual_seed(seed)
            self._fixed_init_states = self.problem.sample_initial_states((B,), generator=g)

    def evaluate_candidate(self, candidate_weights: np.ndarray, seed: int | None = None) -> float:
        """Evaluate a single candidate parameter vector on benchmark landscape.

        Returns:
            Scalar objective score (average best fitness across restarts).
        """
        self.strategy.set_weights_vector(candidate_weights)
        runner = BatchedRunner(
            problem=self.problem,
            strategy=self.strategy,
            max_steps=self.eval_steps,
            group_config=self.group_config,
            record_history=False,
            device=self.device,
        )

        init_s = self._fixed_init_states if self.use_common_random_numbers else None
        traj = runner.run(
            n_restarts=self.eval_restarts,
            initial_states=init_s,
            seed=seed,
        )
        return traj.average_best_fitness

    def step(self) -> GenerationRecord:
        """Execute one generation of CMA-ES."""
        t0 = time.perf_counter()
        candidates = self.es.ask()  # List of candidate vectors
        scores: list[float] = []

        # Evaluate candidate population
        for cand in candidates:
            score = self.evaluate_candidate(cand)
            scores.append(score)
            self.total_evaluations += 1

            if score > self.best_fitness:
                self.best_fitness = score
                self.best_weights = cand.copy()

        # Tell CMA-ES: minimizes objective, so pass negative fitness
        losses = [-s for s in scores]
        self.es.tell(candidates, losses)

        elapsed = time.perf_counter() - t0
        scores_arr = np.array(scores, dtype=np.float32)

        record = GenerationRecord(
            generation=self.generation,
            best_fitness=self.best_fitness,
            gen_best_fitness=float(np.max(scores_arr)),
            gen_mean_fitness=float(np.mean(scores_arr)),
            gen_std_fitness=float(np.std(scores_arr)),
            sigma=float(self.es.sigma),
            elapsed_time=elapsed,
            metadata={"popsize": self.population_size},
        )
        self.generation += 1
        return record

    def train(
        self,
        n_generations: int,
        checkpoint_dir: str | Path | None = None,
        callback: Callable[[GenerationRecord], None] | None = None,
    ) -> TrainResult:
        """Run CMA-ES neuroevolution for n_generations."""
        start_time = time.perf_counter()
        history: list[GenerationRecord] = []

        ckpt_path: Path | None = None
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"cma_checkpoint_gen{n_generations}.json"

        for _ in range(n_generations):
            rec = self.step()
            history.append(rec)
            if callback is not None:
                callback(rec)

        total_elapsed = time.perf_counter() - start_time

        # Update strategy with best discovered parameters
        self.strategy.set_weights_vector(self.best_weights)

        result = TrainResult(
            best_weights=self.best_weights,
            best_fitness=self.best_fitness,
            history=history,
            total_generations=n_generations,
            total_evaluations=self.total_evaluations,
            total_time_seconds=total_elapsed,
            checkpoint_path=ckpt_path,
        )

        if ckpt_path is not None:
            result.save(ckpt_path)

        return result
