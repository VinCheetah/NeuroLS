"""Base classes and contracts for Local Search strategies.

Defines the strategy lifecycle, move selection interface, determinism guarantees,
and hyperparameter introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from neurols.core.group import GroupContext, GroupMetrics
from neurols.types import HyperparameterSpec, MoveResult, SeedType


class BaseStrategy(ABC):
    """Abstract base class for all Local Search strategies.

    Subclasses implement specific move selection heuristics, ranging from classical
    hill climbing and metaheuristics (Tabu, SA) to neural policies.

    Determinism Contract:
    - If `is_deterministic` is True: Strategy is strictly deterministic.
    - If `is_deterministic` is False and `state_dependent_deterministic` is True:
      Pseudo-random choices are seeded strictly from the current state bits and step index.
      This guarantees that any identical state at the same step yields identical behavior.
    - If `state_dependent_deterministic` is False: Uses the strategy's internal PyTorch generator.
    """

    def __init__(
        self,
        name: str,
        state_dependent_deterministic: bool = False,
        seed: SeedType = None,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self.state_dependent_deterministic = state_dependent_deterministic
        self.seed = seed
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    @property
    def name(self) -> str:
        """Name of the strategy."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Whether the strategy is inherently deterministic (no stochastic sampling)."""
        pass

    @property
    def uses_group_information(self) -> bool:
        """Whether this strategy actively uses group context/metrics in decision making.

        Classical single-agent heuristics (HillClimber, FirstImprovement, TabuSearch,
        SimulatedAnnealing, RandomWalk) ignore group context and return False.
        Strategies returning False behave identically regardless of information sharing flags.
        """
        return False

    @property
    @abstractmethod
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        """Specification of strategy hyperparameters for tuning and reflection."""
        pass

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Current values of strategy hyperparameters."""
        pass

    @abstractmethod
    def reset(self, batch_size: int, device: torch.device, seed: SeedType = None) -> None:
        """Reset internal search memory, tabu lists, step counters, and RNG state.

        Args:
            batch_size: Total batch size B = (n_instances * n_restarts * n_agents).
            device: Computation device.
            seed: Optional seed to override RNG state.
        """
        if seed is not None:
            self.seed = seed
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(seed)
        elif self.seed is not None and self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)

    @abstractmethod
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
        """Select a neighbor move for each agent in the batch.

        Args:
            states: Current states tensor, shape (B, dimension).
            deltas: Precomputed fitness deltas for all candidate moves, shape (B, neighborhood_size).
            current_fitness: Current objective values, shape (B,).
            best_fitness: Best objective values found so far, shape (B,).
            group_context: Optional multi-agent group configuration and topology.
            group_metrics: Optional computed multi-agent group metrics (distances, ranks, ...).
            step: Current search step index.

        Returns:
            MoveResult with selected move index per agent and step metadata.
        """
        pass

    def _get_row_seeds(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """Generate deterministic 64-bit integer seeds from state bits and step count.

        Used when `state_dependent_deterministic=True` to ensure state-reproducible randomness.
        """
        # Linear congruential hash over state rows
        B, D = states.shape
        # Weight each bit with a prime power
        primes = torch.tensor(
            [(1000003 + i * 9973) % 2147483647 for i in range(D)],
            device=states.device,
            dtype=torch.int64,
        )
        states_int64 = states.to(torch.int64)
        hash_vals = (states_int64 * primes).sum(dim=-1) + (step * 2654435761)
        return hash_vals.abs()
