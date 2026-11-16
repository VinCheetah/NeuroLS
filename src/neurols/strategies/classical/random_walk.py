"""Random Walk Local Search strategy.

Explores the neighborhood by picking neighbor moves uniformly at random, or
interpolating between random exploration and greedy hill climbing.
"""

from __future__ import annotations

from typing import Any

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec, MoveResult, SeedType


class RandomWalk(BaseStrategy):
    """Random Walk Local Search strategy.

    With probability exploration_rate, selects a neighbor uniformly at random.
    With probability 1 - exploration_rate, selects the best greedy move.
    """

    def __init__(
        self,
        exploration_rate: float = 1.0,
        state_dependent_deterministic: bool = False,
        seed: SeedType = None,
        name: str = "RandomWalk",
    ) -> None:
        super().__init__(
            name=name,
            state_dependent_deterministic=state_dependent_deterministic,
            seed=seed,
        )
        if not (0.0 <= exploration_rate <= 1.0):
            raise ValueError(f"exploration_rate must be in [0, 1], got {exploration_rate}")
        self.exploration_rate = exploration_rate

    @property
    def is_deterministic(self) -> bool:
        return (self.exploration_rate == 0.0) or self.state_dependent_deterministic

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="exploration_rate",
                param_type="float",
                default=1.0,
                low=0.0,
                high=1.0,
                description="Probability of selecting a random move rather than the best greedy move.",
            )
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"exploration_rate": self.exploration_rate}

    def reset(self, batch_size: int, device: torch.device, seed: SeedType = None) -> None:
        super().reset(batch_size=batch_size, device=device, seed=seed)

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

        if self.state_dependent_deterministic:
            row_seeds = self._get_row_seeds(states, step)
            random_moves = (row_seeds % N).to(device)
            random_gate = ((row_seeds // N) % 10000).float() / 10000.0
            do_explore = random_gate < self.exploration_rate
        else:
            random_moves = torch.randint(0, N, (B,), device=device, generator=self._generator)
            gate = torch.rand((B,), device=device, generator=self._generator)
            do_explore = gate < self.exploration_rate

        # Best greedy move
        _, best_moves = torch.max(deltas, dim=-1)
        chosen = torch.where(do_explore, random_moves, best_moves)

        # Check if chosen move is improving
        chosen_deltas = torch.gather(deltas, dim=1, index=chosen.unsqueeze(1)).squeeze(1)
        is_improving = chosen_deltas > 0.0

        return MoveResult(
            chosen_moves=chosen,
            is_improving=is_improving,
            stop_mask=torch.zeros(B, dtype=torch.bool, device=device),
        )
