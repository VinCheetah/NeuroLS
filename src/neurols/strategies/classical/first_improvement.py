"""First Improvement Local Search strategy.

Scans the candidate moves in the neighborhood and immediately accepts the first
move that strictly improves the objective function (delta_f > 0).
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec, MoveResult, SeedType


class FirstImprovement(BaseStrategy):
    """First Improvement Hill Climber.

    Accepts the first encountered move with delta > 0.
    Can scan in lexicographic order (deterministic) or random permutation order.
    """

    def __init__(
        self,
        order: Literal["lexicographic", "random"] = "random",
        stop_at_local_optimum: bool = True,
        state_dependent_deterministic: bool = False,
        seed: SeedType = None,
        name: str = "FirstImprovement",
    ) -> None:
        super().__init__(
            name=name,
            state_dependent_deterministic=state_dependent_deterministic,
            seed=seed,
        )
        self.order = order
        self.stop_at_local_optimum = stop_at_local_optimum

    @property
    def is_deterministic(self) -> bool:
        return self.order == "lexicographic" or self.state_dependent_deterministic

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="order",
                param_type="str",
                default="random",
                choices=["lexicographic", "random"],
                description="Neighborhood scanning sequence order.",
            ),
            HyperparameterSpec(
                name="stop_at_local_optimum",
                param_type="bool",
                default=True,
                description="Whether to signal stop when no improving move is found.",
            ),
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "stop_at_local_optimum": self.stop_at_local_optimum,
        }

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

        if self.order == "lexicographic":
            # Lexicographic scan: order is 0, 1, ..., N-1
            # Positive mask: (B, N)
            pos_mask = deltas > 0.0
            # Find first index along dim 1 where pos_mask is True
            has_pos, first_idx = torch.max(pos_mask.long(), dim=-1)
            is_improving = has_pos > 0
            chosen = torch.where(is_improving, first_idx, torch.tensor(-1, device=device))
        else:
            # Random order scan
            if self.state_dependent_deterministic:
                # Deterministic pseudo-random permutations per row from state hash
                row_seeds = self._get_row_seeds(states, step)
                perms: list[torch.Tensor] = []
                for b in range(B):
                    g = torch.Generator(device="cpu").manual_seed(int(row_seeds[b].item()))
                    perms.append(torch.randperm(N, generator=g, device=device))
                order_indices = torch.stack(perms, dim=0)  # (B, N)
            else:
                # Stochastic permutation across batch
                # Sort random keys
                keys = torch.rand((B, N), device=device, generator=self._generator)
                order_indices = torch.argsort(keys, dim=-1)  # (B, N)

            # Reorder deltas according to permutation:
            permuted_deltas = torch.gather(deltas, dim=1, index=order_indices)
            pos_mask = permuted_deltas > 0.0
            has_pos, first_pos_idx = torch.max(pos_mask.long(), dim=-1)  # index into permuted
            is_improving = has_pos > 0

            # Gather actual move index from order_indices
            gathered_move = torch.gather(
                order_indices, dim=1, index=first_pos_idx.unsqueeze(-1)
            ).squeeze(-1)
            chosen = torch.where(is_improving, gathered_move, torch.tensor(-1, device=device))

        stop_mask = ~is_improving if self.stop_at_local_optimum else torch.zeros_like(is_improving)

        return MoveResult(
            chosen_moves=chosen,
            is_improving=is_improving,
            stop_mask=stop_mask,
        )
