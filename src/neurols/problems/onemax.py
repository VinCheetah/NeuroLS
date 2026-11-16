"""OneMax Problem implementation.

A simple unimodal binary optimization problem where the objective is to maximize
the number of 1s in a bitstring: f(x) = sum(x_i) / N.
Useful for validation, benchmarking, and proving generality.
"""

from __future__ import annotations

import torch

from neurols.core.problem import BatchedProblem
from neurols.types import DeviceType


class OneMax(BatchedProblem):
    """OneMax binary optimization problem: f(x) = count(1s) / N."""

    def __init__(self, n: int = 50, device: DeviceType = "cpu") -> None:
        super().__init__(device=device)
        self.n = n

    @property
    def dimension(self) -> int:
        return self.n

    @property
    def neighborhood_size(self) -> int:
        return self.n

    def to(self, device: DeviceType) -> OneMax:
        return OneMax(n=self.n, device=device)

    def sample_initial_states(
        self,
        batch_shape: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        total_shape = (*batch_shape, self.n)
        probs = torch.full(total_shape, 0.5, device=self.device)
        return torch.bernoulli(probs, generator=generator).to(torch.int8)

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate OneMax fitness: fraction of bits set to 1."""
        return states.float().mean(dim=-1)

    def get_neighborhood_deltas(self, states: torch.Tensor) -> torch.Tensor:
        """Delta for flipping bit j:

        If bit j is 0 -> flips to 1, delta is +1/N.
        If bit j is 1 -> flips to 0, delta is -1/N.
        """
        # states: (*shape, N) in {0, 1}
        # (1 - 2*states) / N: if 0 -> +1/N, if 1 -> -1/N
        return (1.0 - 2.0 * states.float()) / float(self.n)

    def apply_moves(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        orig_shape = states.shape
        states_flat = states.view(-1, self.n).clone()
        moves_flat = moves.view(-1).long()

        one_hot = torch.zeros_like(states_flat)
        valid = (moves_flat >= 0) & (moves_flat < self.n)
        if valid.any():
            one_hot[valid, moves_flat[valid]] = 1

        new_states = states_flat ^ one_hot
        return new_states.view(orig_shape)

    def compute_distances(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        return (states1 != states2).sum(dim=-1).float()
