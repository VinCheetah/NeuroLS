"""Base problem classes and interfaces for Local Search in NeuroLS.

Provides abstract base classes for generic problems and specialized high-performance
batched tensor problems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch

from neurols.types import DeviceType, MoveType, StateType


class BaseProblem(ABC, Generic[StateType, MoveType]):
    """Abstract base class for all optimization problems."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Search space dimensionality (e.g., number of variables or bits)."""
        pass

    @property
    @abstractmethod
    def neighborhood_size(self) -> int:
        """Number of candidate moves evaluated per state in the local neighborhood."""
        pass

    @abstractmethod
    def evaluate(self, states: StateType) -> torch.Tensor:
        """Evaluate the objective / fitness function for given states.

        Args:
            states: Current problem states.

        Returns:
            Tensor of shape (B,) containing fitness values (higher is better).
        """
        pass

    @abstractmethod
    def sample_initial_states(
        self,
        batch_shape: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> StateType:
        """Generate random initial states for local search trajectories.

        Args:
            batch_shape: Shape tuple, e.g. (n_instances, n_restarts, n_agents) or (B,).
            generator: Optional PyTorch random generator for reproducibility.

        Returns:
            Initial states object/tensor.
        """
        pass

    @abstractmethod
    def get_neighborhood_deltas(self, states: StateType) -> torch.Tensor:
        """Compute the fitness change for each move in the neighborhood.

        Delta is defined as: delta_f(x, m) = fitness(apply_move(x, m)) - fitness(x).
        Positive values indicate improving moves.

        Args:
            states: Current problem states.

        Returns:
            Tensor of shape (B, neighborhood_size) of fitness deltas.
        """
        pass

    @abstractmethod
    def apply_moves(self, states: StateType, moves: MoveType) -> StateType:
        """Apply chosen moves to states to produce new states.

        Args:
            states: Current problem states.
            moves: Selected move indices/descriptors.

        Returns:
            Updated states after applying moves.
        """
        pass

    @abstractmethod
    def compute_distances(self, states1: StateType, states2: StateType) -> torch.Tensor:
        """Compute distance (e.g. Hamming or Euclidean) between pairs of states.

        Args:
            states1: First batch of states.
            states2: Second batch of states.

        Returns:
            Tensor of pairwise or broadcasted distances.
        """
        pass


class BatchedProblem(BaseProblem[torch.Tensor, torch.Tensor]):
    """Base class for problems with vectorized PyTorch tensor states and moves.

    States are represented as tensors of shape `(B, dimension)` or higher dimensional.
    Moves are integer tensors of shape `(B,)` indexing candidate neighborhood transitions.
    """

    def __init__(self, device: DeviceType = "cpu") -> None:
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        """PyTorch computation device (CPU or CUDA)."""
        return self._device

    @abstractmethod
    def to(self, device: DeviceType) -> BatchedProblem:
        """Move problem tensors and lookups to the specified target device."""
        pass

    def is_local_optimum(
        self,
        states: torch.Tensor,
        deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Check whether each state in the batch is at a local optimum.

        A state is a local optimum if no move in its neighborhood strictly improves
        the objective function (i.e. all deltas <= 0).

        Args:
            states: Current states tensor of shape (B, dimension).
            deltas: Optional precomputed neighborhood deltas tensor of shape (B, neighborhood_size).

        Returns:
            Boolean tensor of shape (B,) where True indicates a local optimum.
        """
        if deltas is None:
            deltas = self.get_neighborhood_deltas(states)
        max_deltas, _ = torch.max(deltas, dim=-1)
        return max_deltas <= 0.0
