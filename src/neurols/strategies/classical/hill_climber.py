"""Hill Climbing / Steepest Ascent Local Search strategy.

Evaluates the full neighborhood and greedily selects the neighbor move that yields
the maximum positive fitness improvement. Terminates or stays when no strictly
improving move exists (local optimum reached).
"""

from __future__ import annotations

from typing import Any

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec, MoveResult, SeedType


class HillClimber(BaseStrategy):
    """Steepest Ascent Hill Climber.

    Picks the move with highest positive delta: argmax_{j} delta_f(x, j).
    Strictly deterministic with deterministic tie-breaking (first index).
    """

    def __init__(
        self,
        stop_at_local_optimum: bool = True,
        name: str = "HillClimber",
    ) -> None:
        super().__init__(name=name, state_dependent_deterministic=False, seed=None)
        self.stop_at_local_optimum = stop_at_local_optimum

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="stop_at_local_optimum",
                param_type="bool",
                default=True,
                description="Whether to signal stop when at a local optimum (no positive delta).",
            )
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"stop_at_local_optimum": self.stop_at_local_optimum}

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
        # deltas: shape (B, neighborhood_size)
        best_deltas, best_moves = torch.max(deltas, dim=-1)  # (B,), (B,)
        is_improving = best_deltas > 0.0  # (B,)

        # If stop_at_local_optimum, agents with no improving moves stop
        if self.stop_at_local_optimum:
            stop_mask = ~is_improving
            # For non-improving agents, set move to -1 (no-op)
            chosen_moves = torch.where(
                is_improving, best_moves, torch.tensor(-1, device=deltas.device)
            )
        else:
            stop_mask = torch.zeros_like(is_improving)
            chosen_moves = best_moves

        return MoveResult(
            chosen_moves=chosen_moves,
            is_improving=is_improving,
            stop_mask=stop_mask,
            metadata={"max_delta": best_deltas},
        )
