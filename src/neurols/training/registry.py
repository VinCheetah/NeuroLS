"""Model Checkpoint Registry and Experiment Storage for NeuroLS.

Provides structured saving, loading, listing, and inspection of trained neural
local search strategies, ensuring complete reproducibility of network architectures,
features, hyperparameters, and training histories.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.training.base import GenerationRecord, TrainResult

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")


@dataclass
class ModelManifest:
    """Full reproducibility manifest for a trained neural strategy."""

    experiment_name: str
    model_name: str
    created_at: str
    problem_config: dict[str, Any]
    strategy_config: dict[str, Any]
    training_config: dict[str, Any]
    metrics: dict[str, Any]
    weights: list[float]
    training_history: list[dict[str, Any]] = field(default_factory=list)
    file_path: Path | None = None

    @property
    def best_fitness(self) -> float:
        return float(self.metrics.get("best_fitness", 0.0))

    @property
    def policy_type(self) -> str:
        return str(self.strategy_config.get("policy_type", "unknown"))

    @property
    def eval_mode(self) -> str:
        return str(self.strategy_config.get("eval_mode", "per_action"))

    @property
    def features_summary(self) -> str:
        feats = self.strategy_config.get("features")
        if isinstance(feats, list):
            return ",".join(feats)
        return str(feats or "default")

    @property
    def n(self) -> int:
        return int(self.problem_config.get("n", self.strategy_config.get("n_actions", 0)))

    @property
    def k(self) -> int:
        return int(self.problem_config.get("k", 0))

    @property
    def n_agents(self) -> int:
        return int(self.strategy_config.get("n_agents", 1))

    @property
    def stopping_conditions(self) -> dict[str, Any]:
        return self.strategy_config.get("stopping_conditions", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": "1.0",
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "problem_config": self.problem_config,
            "strategy_config": self.strategy_config,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "weights": self.weights,
            "training_history": self.training_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], file_path: Path | None = None) -> ModelManifest:
        # Backward compatibility for flat TrainResult checkpoints
        if "manifest_version" not in data and "best_weights" in data:
            weights = data["best_weights"]
            history = data.get("history", [])
            best_fit = data.get("best_fitness", 0.0)
            exp_name = file_path.parent.name if file_path else "legacy"
            model_name = file_path.stem if file_path else "legacy_model"
            return cls(
                experiment_name=exp_name,
                model_name=model_name,
                created_at=datetime.datetime.now().isoformat(),
                problem_config={"n": len(weights), "k": 0},
                strategy_config={
                    "policy_type": "linear",
                    "eval_mode": "per_action",
                    "features": None,
                },
                training_config={
                    "optimizer": "cma_es",
                    "generations": data.get("total_generations", len(history)),
                },
                metrics={
                    "best_fitness": best_fit,
                    "total_generations": data.get("total_generations", len(history)),
                },
                weights=weights,
                training_history=history,
                file_path=file_path,
            )

        return cls(
            experiment_name=data.get("experiment_name", "default"),
            model_name=data.get("model_name", "unnamed_model"),
            created_at=data.get("created_at", datetime.datetime.now().isoformat()),
            problem_config=data.get("problem_config", {}),
            strategy_config=data.get("strategy_config", {}),
            training_config=data.get("training_config", {}),
            metrics=data.get("metrics", {}),
            weights=data.get("weights", []),
            training_history=data.get("training_history", []),
            file_path=file_path,
        )


def save_model_checkpoint(
    strategy: NeuralLocalSearchStrategy,
    train_result: TrainResult,
    experiment_name: str = "default_experiment",
    model_name: str | None = None,
    problem_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    base_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    """Save a fully reproducible model checkpoint and metadata to the local registry.

    Args:
        strategy: Trained NeuralLocalSearchStrategy.
        train_result: TrainResult containing best weights, fitness, and generation history.
        experiment_name: Identifier for the experiment suite / folder.
        model_name: Name for the model file (defaults to strategy name).
        problem_config: Problem parameters (N, K, topology, etc.).
        training_config: Optimizer hyperparameters (popsize, generations, etc.).
        base_dir: Root directory for checkpoints.

    Returns:
        Path to the saved JSON checkpoint.
    """
    base_path = Path(base_dir)
    exp_dir = base_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    name = model_name or strategy.name.replace(" ", "_")
    target_path = exp_dir / f"{name}.json"

    # Extract strategy architecture configuration
    stopping_cond = {
        "max_steps": getattr(strategy, "max_steps", None),
        "stop_at_local_optimum": getattr(strategy, "stop_at_local_optimum", False),
    }
    if training_config and "eval_steps" in training_config:
        stopping_cond["eval_steps"] = training_config["eval_steps"]

    strat_cfg: dict[str, Any] = {
        "strategy_type": "neural",
        "policy_type": strategy.policy_type,
        "eval_mode": strategy.eval_mode.value,
        "features": strategy.feature_config.feature_names,
        "selection_mode": strategy.selection_mode.value,
        "n_actions": strategy.n_actions,
        "n_agents": strategy.n_agents,
        "stopping_conditions": stopping_cond,
        "remix": getattr(strategy, "remix", True),
        "rescale": getattr(strategy, "rescale", False),
        "temperature": strategy.temperature,
    }

    # Extract problem configuration if available
    prob_cfg = problem_config or {
        "problem_type": "nk_landscape",
        "n": strategy.n_actions,
        "k": 0,
    }

    train_cfg = training_config or {
        "optimizer": "cma_es",
        "total_generations": train_result.total_generations,
        "total_evaluations": train_result.total_evaluations,
    }

    metrics = {
        "best_fitness": float(train_result.best_fitness),
        "total_generations": train_result.total_generations,
        "total_evaluations": train_result.total_evaluations,
        "total_time_seconds": float(train_result.total_time_seconds),
    }

    history_records = [
        asdict(rec) if isinstance(rec, GenerationRecord) else rec for rec in train_result.history
    ]

    manifest = ModelManifest(
        experiment_name=experiment_name,
        model_name=name,
        created_at=datetime.datetime.now().isoformat(),
        problem_config=prob_cfg,
        strategy_config=strat_cfg,
        training_config=train_cfg,
        metrics=metrics,
        weights=train_result.best_weights.tolist(),
        training_history=history_records,
        file_path=target_path,
    )

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return target_path


def resolve_model_path(
    query: str | Path,
    base_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    """Resolve a query string (path, 'exp:model', 'exp/model', or 'model') into a valid checkpoint path."""
    p = Path(query)
    if p.exists() and p.is_file():
        return p

    base_path = Path(base_dir)

    # Check "exp:model" syntax
    query_str = str(query)
    if ":" in query_str:
        exp, model = query_str.split(":", 1)
        target = base_path / exp / f"{model}.json"
        if target.exists():
            return target
        target_noext = base_path / exp / model
        if target_noext.exists():
            return target_noext

    # Check "exp/model" syntax
    if "/" in query_str:
        target = base_path / f"{query_str}.json"
        if target.exists():
            return target
        target_direct = base_path / query_str
        if target_direct.exists():
            return target_direct

    # Search for model_name.json in any subfolder under base_dir
    matches = list(base_path.glob(f"**/{query_str}.json"))
    if matches:
        return matches[0]

    matches_raw = list(base_path.glob(f"**/{query_str}"))
    if matches_raw and matches_raw[0].is_file():
        return matches_raw[0]

    raise FileNotFoundError(
        f"Could not find model checkpoint for '{query}'. "
        f"Searched in '{base_path.resolve()}' using patterns: 'exp:model', 'exp/model', or direct file path."
    )


def load_model_checkpoint(
    query_or_path: str | Path,
    base_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
    override_n_actions: int | None = None,
    override_n_agents: int | None = None,
) -> tuple[NeuralLocalSearchStrategy, ModelManifest]:
    """Load a trained model by experiment/model name or path, recreating the strategy with exact weights.

    Args:
        query_or_path: Checkpoint file path, 'experiment:model', 'experiment/model', or model name.
        base_dir: Root checkpoints directory.
        override_n_actions: Optional override for N (if evaluating on different problem dimension).
        override_n_agents: Optional override for agent count.

    Returns:
        tuple of (instantiated NeuralLocalSearchStrategy with loaded weights, ModelManifest metadata).
    """
    path = resolve_model_path(query_or_path, base_dir=base_dir)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    manifest = ModelManifest.from_dict(data, file_path=path)

    strat_cfg = manifest.strategy_config
    n_actions = override_n_actions or strat_cfg.get("n_actions", manifest.n)
    n_agents = override_n_agents or strat_cfg.get("n_agents", 1)

    policy_type = strat_cfg.get("policy_type", "linear")
    eval_mode = strat_cfg.get("eval_mode", "per_action")
    features = strat_cfg.get("features")
    selection_mode = strat_cfg.get("selection_mode", "argmax")
    remix = strat_cfg.get("remix", True)
    rescale = strat_cfg.get("rescale", False)
    temperature = strat_cfg.get("temperature", 1.0)

    strategy = NeuralLocalSearchStrategy(
        policy_type=policy_type,
        eval_mode=eval_mode,
        features=features,
        n_actions=n_actions,
        n_agents=n_agents,
        remix=remix,
        rescale=rescale,
        temperature=temperature,
        selection_mode=selection_mode,
        name=manifest.model_name,
    )

    weights = np.array(manifest.weights, dtype=np.float32)
    strategy.set_weights_vector(weights)

    return strategy, manifest


def list_saved_models(
    base_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
    experiment_name: str | None = None,
) -> list[ModelManifest]:
    """List all saved model checkpoints in the registry, optionally filtered by experiment."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    manifests: list[ModelManifest] = []
    pattern = f"{experiment_name}/*.json" if experiment_name else "**/*.json"

    for json_file in base_path.glob(pattern):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if "weights" in data or "best_weights" in data:
                manifests.append(ModelManifest.from_dict(data, file_path=json_file))
        except Exception:
            continue

    # Sort manifests by creation date descending
    manifests.sort(key=lambda m: m.created_at, reverse=True)
    return manifests


def delete_saved_model(
    query_or_path: str | Path,
    base_dir: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> bool:
    """Delete a saved model checkpoint from disk."""
    try:
        path = resolve_model_path(query_or_path, base_dir=base_dir)
        path.unlink()
        return True
    except FileNotFoundError:
        return False
