"""Base classes and contracts for Neuroevolution and Policy Optimization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurols.core.group import GroupConfig
from neurols.core.problem import BatchedProblem
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy


@dataclass
class GenerationRecord:
    """Metrics recorded at the end of each training generation."""

    generation: int
    best_fitness: float
    gen_best_fitness: float
    gen_mean_fitness: float
    gen_std_fitness: float
    sigma: float
    elapsed_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Complete summary of a neuroevolution training run."""

    best_weights: np.ndarray
    best_fitness: float
    history: list[GenerationRecord]
    total_generations: int
    total_evaluations: int
    total_time_seconds: float
    checkpoint_path: Path | None = None

    def save(self, filepath: str | Path) -> None:
        """Save training summary and best weights to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "best_fitness": self.best_fitness,
            "total_generations": self.total_generations,
            "total_evaluations": self.total_evaluations,
            "total_time_seconds": self.total_time_seconds,
            "best_weights": self.best_weights.tolist(),
            "history": [asdict(rec) for rec in self.history],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> TrainResult:
        """Load training result from disk."""
        path = Path(filepath)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        history = [GenerationRecord(**rec) for rec in data["history"]]
        return cls(
            best_weights=np.array(data["best_weights"], dtype=np.float32),
            best_fitness=data["best_fitness"],
            history=history,
            total_generations=data["total_generations"],
            total_evaluations=data["total_evaluations"],
            total_time_seconds=data["total_time_seconds"],
            checkpoint_path=path,
        )


class BaseOptimizer(ABC):
    """Abstract base class for policy parameter optimizers (e.g. CMA-ES, Genetic Algorithms)."""

    def __init__(
        self,
        strategy: NeuralLocalSearchStrategy,
        problem: BatchedProblem,
        eval_restarts: int = 4,
        eval_steps: int = 50,
        group_config: GroupConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.strategy = strategy
        self.problem = problem.to(torch.device(device))
        self.eval_restarts = eval_restarts
        self.eval_steps = eval_steps
        self.group_config = group_config or GroupConfig(n_agents=1)
        self.device = torch.device(device)

    @abstractmethod
    def step(self) -> GenerationRecord:
        """Execute one generation of evolutionary optimization."""
        pass

    @abstractmethod
    def train(
        self,
        n_generations: int,
        checkpoint_dir: str | Path | None = None,
        callback: Callable[[GenerationRecord], None] | None = None,
    ) -> TrainResult:
        """Execute full evolutionary optimization run."""
        pass
