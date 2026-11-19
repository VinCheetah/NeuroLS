"""Core type definitions, protocols, and data structures for NeuroLS.

This module establishes strict typing across the entire library using PEP 585/604/695
constructs, protocols, and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import torch

# Type aliases
DeviceType = str | torch.device
DType = torch.dtype
SeedType = int | None
BatchShape = tuple[int, int, int]  # (n_instances, n_restarts, n_agents)


class ProblemType(StrEnum):
    """Supported problem categories."""

    NK_LANDSCAPE = "nk_landscape"
    BINARY = "binary"
    CONTINUOUS = "continuous"
    CUSTOM = "custom"


class NeighborhoodType(StrEnum):
    """Epistatic interaction models for NK Landscapes."""

    ADJACENT = "adjacent"
    RANDOM = "random"


class MoveSelectionMode(StrEnum):
    """Action selection policy mode."""

    ARGMAX = "argmax"
    SAMPLE = "sample"
    GREEDY_FIRST = "greedy_first"
    TABU = "tabu"


class AnnealingSchedule(StrEnum):
    """Cooling schedule types for Simulated Annealing."""

    GEOMETRIC = "geometric"
    LINEAR = "linear"
    COSINE = "cosine"


class TerminationReason(StrEnum):
    """Reason for search trajectory termination."""

    MAX_STEPS = "max_steps"
    EVALUATION_BUDGET = "evaluation_budget"
    TIME_LIMIT = "time_limit"
    TARGET_FITNESS = "target_fitness"
    STAGNATION = "stagnation"
    LOCAL_OPTIMUM = "local_optimum"
    STRATEGY_STOP = "strategy_stop"


@dataclass
class StoppingCriteria:
    """Configurable termination criteria for Local Search.

    Supports combinations of step limits, evaluation budgets, wall-clock time,
    target fitness thresholds, and stagnation limits.
    Stops as soon as ANY active criterion is satisfied (OR logic).

    Attributes:
        max_steps: Maximum iterations/steps (t).
        max_evaluations: Maximum function / neighborhood evaluation count.
        max_time_seconds: Maximum wall-clock time in seconds.
        target_fitness: Stop early if best fitness reaches this threshold.
        max_stagnation_steps: Stop early if no improvement for K steps.
    """

    max_steps: int | None = None
    max_evaluations: int | None = None
    max_time_seconds: float | None = None
    target_fitness: float | None = None
    max_stagnation_steps: int | None = None

    def __post_init__(self) -> None:
        if not any(
            [
                self.max_steps,
                self.max_evaluations,
                self.max_time_seconds,
                self.target_fitness,
                self.max_stagnation_steps,
            ]
        ):
            self.max_steps = 100
            print("Warning: No stopping criteria specified. Defaulting to max_steps=100.")

    def check(
        self,
        step: int,
        evaluations: int,
        elapsed_seconds: float,
        best_fitness: float,
        stagnation_steps: int,
    ) -> tuple[bool, TerminationReason | None]:
        """Check if any stopping condition is met."""
        if self.max_steps is not None and step >= self.max_steps:
            return True, TerminationReason.MAX_STEPS
        if self.max_evaluations is not None and evaluations >= self.max_evaluations:
            return True, TerminationReason.EVALUATION_BUDGET
        if self.max_time_seconds is not None and elapsed_seconds >= self.max_time_seconds:
            return True, TerminationReason.TIME_LIMIT
        if self.target_fitness is not None and best_fitness >= self.target_fitness:
            return True, TerminationReason.TARGET_FITNESS
        if self.max_stagnation_steps is not None and stagnation_steps >= self.max_stagnation_steps:
            return True, TerminationReason.STAGNATION
        return False, None


class Device(StrEnum):
    """Computation device types."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass(frozen=True)
class MoveResult:
    """Encapsulates the decision made by a local search strategy at a given step.

    Attributes:
        chosen_moves: Tensor of chosen move indices per batch element, shape (B,).
        is_improving: Boolean tensor indicating if each chosen move strictly improves fitness, shape (B,).
        probabilities: Optional tensor of action probabilities over neighborhood, shape (B, N).
        logits: Optional tensor of raw policy logits over neighborhood, shape (B, N).
        stop_mask: Boolean tensor indicating if each agent decided to terminate / converged, shape (B,).
        metadata: Strategy-specific step diagnostics and intermediate features.
    """

    chosen_moves: torch.Tensor
    is_improving: torch.Tensor
    probabilities: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    stop_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    """Trajectory snapshot at a discrete search step.

    Attributes:
        step: Current iteration step index (0-based).
        current_fitness: Current fitness per batch element, shape (B,).
        best_fitness: Best fitness achieved so far per batch element, shape (B,).
        chosen_moves: Selected move indices at this step, shape (B,).
        group_diversity: Optional average Hamming / Euclidean distance within each multi-agent group.
        delta_fitness: Change in fitness achieved by this step, shape (B,).
    """

    step: int
    current_fitness: torch.Tensor
    best_fitness: torch.Tensor
    chosen_moves: torch.Tensor
    delta_fitness: torch.Tensor
    group_diversity: torch.Tensor | None = None
    states: torch.Tensor | None = None
    action_ranks: torch.Tensor | None = None


@dataclass
class HyperparameterSpec:
    """Specification of a strategy hyperparameter for documentation and automated tuning.

    Attributes:
        name: Unique hyperparameter identifier.
        param_type: Python type ('float', 'int', 'bool', 'str').
        default: Default value.
        low: Optional lower bound (for numerical parameters).
        high: Optional upper bound (for numerical parameters).
        choices: Optional discrete list of choices (for categorical parameters).
        log_scale: Whether the parameter is typically searched on a logarithmic scale.
        description: Human-readable explanation of the parameter's role.
    """

    name: str
    param_type: Literal["float", "int", "bool", "str"]
    default: Any
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    description: str = ""


# Generic Type Variables
StateType = TypeVar("StateType")
MoveType = TypeVar("MoveType")


@runtime_checkable
class ProblemProtocol(Protocol[StateType, MoveType]):
    """Structural protocol defining the contract for any problem in NeuroLS."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the search space (e.g., bitstring length N)."""
        ...

    def evaluate(self, states: StateType) -> torch.Tensor:
        """Evaluate the objective / fitness function for given states."""
        ...

    def sample_initial_states(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> StateType:
        """Sample random starting states for local search trajectories."""
        ...

    def get_neighborhood_deltas(self, states: StateType) -> torch.Tensor:
        """Compute objective function changes for all neighbor moves.

        Returns:
            Tensor of shape (B, num_neighbors) containing fitness deltas.
        """
        ...

    def apply_moves(self, states: StateType, moves: MoveType) -> StateType:
        """Apply selected moves to the current states, returning updated states."""
        ...
