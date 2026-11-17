"""Tabu Search Local Search metaheuristic.

Tracks recently modified variables in a short-term tabu memory to prevent cycling
and escape local optima, allowing non-improving moves unless prohibited by tabu tenure.
"""

from __future__ import annotations

from typing import Any

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec, MoveResult, SeedType


class TabuSearch(BaseStrategy):
    """Tabu Search Metaheuristic.

    Maintains a tenure matrix tracking which variables/bits cannot be flipped.
    Overrides tabu status if a move satisfies the aspiration criterion (improves the global best).
    """

    def __init__(
        self,
        tenure: int = 8,
        aspiration: bool = True,
        name: str = "TabuSearch",
    ) -> None:
        super().__init__(name=name, state_dependent_deterministic=False, seed=None)
        if tenure < 1:
            raise ValueError(f"tenure must be >= 1, got {tenure}")
        self.tenure = tenure
        self.aspiration = aspiration
        self._tabu_expiration: torch.Tensor | None = None

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="tenure",
                param_type="int",
                default=5,
                low=1,
                high=50,
                description="Number of iterations for which a flipped variable remains tabu.",
            ),
            HyperparameterSpec(
                name="aspiration",
                param_type="bool",
                default=True,
                description="Whether to override tabu status if a move beats the global best fitness.",
            ),
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"tenure": self.tenure, "aspiration": self.aspiration}

    def reset(self, batch_size: int, device: torch.device, seed: SeedType = None) -> None:
        super().reset(batch_size=batch_size, device=device, seed=seed)
        # Tabu expiration matrix: initialized to -1 (no moves tabu initially)
        self._tabu_expiration = None

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

        # Initialize tabu table if first step or batch shape changed
        if self._tabu_expiration is None or self._tabu_expiration.shape != (B, N):
            self._tabu_expiration = torch.full((B, N), -1, dtype=torch.int64, device=device)

        # Tabu mask: True if move j is currently tabu
        is_tabu = self._tabu_expiration > step  # (B, N)

        # Aspiration criterion: allows tabu moves that achieve a new best-so-far fitness
        if self.aspiration:
            prospective_fitness = current_fitness.unsqueeze(1) + deltas  # (B, N)
            meets_aspiration = prospective_fitness > best_fitness.unsqueeze(1)
            effective_tabu = is_tabu & (~meets_aspiration)
        else:
            effective_tabu = is_tabu

        # Mask tabu moves with -infinity so they aren't chosen
        scores = deltas.clone()
        scores[effective_tabu] = -float("inf")

        # Pick best allowed move
        best_allowed_scores, chosen = torch.max(scores, dim=-1)  # (B,)

        # In rare cases where all moves are tabu, fallback to least recently modified
        all_tabu_mask = torch.isneginf(best_allowed_scores)
        if all_tabu_mask.any():
            # Pick move with smallest expiration time
            fallback_moves = torch.argmin(self._tabu_expiration[all_tabu_mask], dim=-1)
            chosen[all_tabu_mask] = fallback_moves

        # Update tabu expiration for chosen moves
        expiration_step = step + self.tenure
        row_indices = torch.arange(B, device=device)
        self._tabu_expiration[row_indices, chosen] = expiration_step

        # Check whether chosen move improves current fitness
        chosen_deltas = torch.gather(deltas, dim=1, index=chosen.unsqueeze(1)).squeeze(1)
        is_improving = chosen_deltas > 0.0

        return MoveResult(
            chosen_moves=chosen,
            is_improving=is_improving,
            stop_mask=torch.zeros(B, dtype=torch.bool, device=device),
            metadata={
                "tenure": self.tenure,
                "tabu_count": is_tabu.sum(dim=-1).float().mean().item(),
            },
        )
