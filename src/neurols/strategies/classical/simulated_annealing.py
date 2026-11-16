"""Simulated Annealing Local Search strategy.

Probabilistically accepts non-improving moves using the Metropolis acceptance criterion
P(accept) = exp(delta_f / T) to escape local optima, cooling temperature over time.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.types import AnnealingSchedule, HyperparameterSpec, MoveResult, SeedType


class SimulatedAnnealing(BaseStrategy):
    """Simulated Annealing Metaheuristic.

    At each step, samples a random candidate move from the neighborhood.
    Always accepts improving moves (delta > 0). Accepts deteriorating moves with
    probability exp(delta / T).
    """

    def __init__(
        self,
        initial_temperature: float = 0.1,
        cooling_rate: float = 0.98,
        min_temperature: float = 1e-4,
        schedule: AnnealingSchedule | str = AnnealingSchedule.GEOMETRIC,
        max_steps: int = 500,
        state_dependent_deterministic: bool = False,
        seed: SeedType = None,
        name: str = "SimulatedAnnealing",
    ) -> None:
        super().__init__(
            name=name,
            state_dependent_deterministic=state_dependent_deterministic,
            seed=seed,
        )
        if initial_temperature <= 0:
            raise ValueError(f"initial_temperature must be > 0, got {initial_temperature}")
        if not (0 < cooling_rate < 1.0):
            raise ValueError(f"cooling_rate must be in (0, 1), got {cooling_rate}")

        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.schedule = AnnealingSchedule(schedule)
        self.max_steps = max_steps
        self.current_temperature = initial_temperature

    @property
    def is_deterministic(self) -> bool:
        return self.state_dependent_deterministic

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="initial_temperature",
                param_type="float",
                default=0.1,
                low=1e-3,
                high=10.0,
                log_scale=True,
                description="Starting temperature T_0.",
            ),
            HyperparameterSpec(
                name="cooling_rate",
                param_type="float",
                default=0.98,
                low=0.8,
                high=0.999,
                description="Geometric cooling factor alpha.",
            ),
            HyperparameterSpec(
                name="min_temperature",
                param_type="float",
                default=1e-4,
                low=1e-6,
                high=1e-2,
                log_scale=True,
                description="Minimum temperature floor.",
            ),
            HyperparameterSpec(
                name="schedule",
                param_type="str",
                default="geometric",
                choices=["geometric", "linear", "cosine"],
                description="Temperature decay schedule.",
            ),
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "min_temperature": self.min_temperature,
            "schedule": self.schedule.value,
        }

    def reset(self, batch_size: int, device: torch.device, seed: SeedType = None) -> None:
        super().reset(batch_size=batch_size, device=device, seed=seed)
        self.current_temperature = self.initial_temperature

    def _get_temperature(self, step: int) -> float:
        """Compute current temperature according to schedule."""
        if self.schedule == AnnealingSchedule.GEOMETRIC:
            temp = self.initial_temperature * (self.cooling_rate**step)
        elif self.schedule == AnnealingSchedule.LINEAR:
            frac = max(0.0, 1.0 - (step / max(self.max_steps, 1)))
            temp = self.min_temperature + frac * (self.initial_temperature - self.min_temperature)
        elif self.schedule == AnnealingSchedule.COSINE:
            progress = min(1.0, step / max(self.max_steps, 1))
            temp = self.min_temperature + 0.5 * (
                self.initial_temperature - self.min_temperature
            ) * (1.0 + math.cos(math.pi * progress))
        else:
            temp = self.initial_temperature
        return max(temp, self.min_temperature)

    def select_move(
        self,
        states: torch.Tensor,
        deltas: torch.Tensor,
        current_fitness: torch.Tensor,
        best_fitness: torch.Tensor,
        group_context: GroupContext | None = None,
        group_metrics: GroupMetrics | None = None,
        step: int = 0,
    ) -> MoveResult:
        B, N = deltas.shape
        device = deltas.device
        temp = self._get_temperature(step)
        self.current_temperature = temp

        # Candidate move selection: pick candidate move per agent
        if self.state_dependent_deterministic:
            row_seeds = self._get_row_seeds(states, step)
            candidate_moves = (row_seeds % N).to(device)
            unif_samples = ((row_seeds // N) % 100000).float() / 100000.0
        else:
            candidate_moves = torch.randint(0, N, (B,), device=device, generator=self._generator)
            unif_samples = torch.rand((B,), device=device, generator=self._generator)

        # Delta of candidate moves: (B,)
        cand_deltas = torch.gather(deltas, dim=1, index=candidate_moves.unsqueeze(1)).squeeze(1)

        # Metropolis acceptance criterion:
        # If delta > 0: accept (prob 1.0)
        # Else: accept with prob exp(delta / temp)
        accept_probs = torch.where(
            cand_deltas > 0,
            torch.ones_like(cand_deltas),
            torch.exp(cand_deltas / temp).clamp(max=1.0),
        )

        accepted = unif_samples < accept_probs
        # If rejected, move is -1 (stay)
        chosen = torch.where(accepted, candidate_moves, torch.tensor(-1, device=device))
        is_improving = accepted & (cand_deltas > 0)

        return MoveResult(
            chosen_moves=chosen,
            is_improving=is_improving,
            stop_mask=torch.zeros(B, dtype=torch.bool, device=device),
            metadata={"temperature": temp, "accept_rate": accepted.float().mean().item()},
        )
