"""Neural Local Search Strategy.

Integrates neural policy networks with feature extraction and local search loops,
supporting both individual per-action scoring and joint all-actions permutation-equivariant evaluation.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from neurols.core.group import GroupContext, GroupMetrics
from neurols.strategies.base import BaseStrategy
from neurols.strategies.neural.features import (
    EvaluationMode,
    FeatureAblationLevel,
    FeatureConfig,
    FeatureExtractor,
)
from neurols.strategies.neural.policy import (
    BaseNeuralPolicy,
    DeepSetsPolicy,
    DenseNeighborhoodPolicy,
    InvariantNNet,
    LinearScorer,
    PeerInvariantActionScorer,
    PerActionScorer,
)
from neurols.types import HyperparameterSpec, MoveResult, MoveSelectionMode, SeedType


class NeuralLocalSearchStrategy(BaseStrategy):
    """Local Search strategy driven by a neural network policy.

    Supports:
    - Individual Action Scoring (eval_mode='per_action'): scores each candidate action
      independently using its action-level features and group context, then chooses a move.
    - All-Actions Evaluation (eval_mode='all_actions'): processes all N actions
      simultaneously, either via permutation-equivariant DeepSets architectures
      (InvariantNNet, DeepSetsPolicy) or full-neighborhood dense networks (DenseNeighborhoodPolicy).
    - Modular Feature Specification: specify exact feature channels or group keywords
      ('@action_basic', '@scores', '@distances', '@group', '@all').
    """

    def __init__(
        self,
        policy: BaseNeuralPolicy | None = None,
        policy_type: Literal[
            "linear",
            "mlp",
            "deepsets",
            "invariant_nnet",
            "per_action",
            "peer_invariant_per_action",
            "dense",
            "dense_neighborhood",
        ] = "linear",
        eval_mode: EvaluationMode | str = EvaluationMode.PER_ACTION,
        features: str | list[str] | None = None,
        ablation_level: FeatureAblationLevel | str = FeatureAblationLevel.FULL,
        feature_config: FeatureConfig | None = None,
        selection_mode: MoveSelectionMode | str = MoveSelectionMode.ARGMAX,
        n_actions: int = 20,
        n_agents: int = 1,
        hidden_dim: int = 16,
        hidden_layers: tuple[int, ...] = (16, 16),
        remix: bool = True,
        rescale: bool = False,
        temperature: float = 1.0,
        max_steps: int | None = 500,
        stop_at_local_optimum: bool = False,
        state_dependent_deterministic: bool = False,
        seed: SeedType = None,
        name: str | None = None,
    ) -> None:
        self.eval_mode = EvaluationMode(eval_mode)
        strategy_name = name or f"Neural_{self.eval_mode.value}_{policy_type}"
        super().__init__(
            name=strategy_name,
            state_dependent_deterministic=state_dependent_deterministic,
            seed=seed,
        )
        self.policy_type = policy_type
        self.ablation_level = (
            FeatureAblationLevel(ablation_level)
            if isinstance(ablation_level, (FeatureAblationLevel, str))
            else FeatureAblationLevel.FULL
        )
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.remix = remix
        self.rescale = rescale

        if feature_config is not None:
            self.feature_config = feature_config
        elif features is not None:
            self.feature_config = FeatureConfig.from_specs(
                spec=features,
                mode=self.eval_mode,
            )
        else:
            self.feature_config = FeatureConfig.from_ablation_level(
                level=self.ablation_level,
                n_agents=self.n_agents,
                mode=self.eval_mode,
            )

        self.extractor = FeatureExtractor(self.feature_config, n_agents=self.n_agents)
        self.selection_mode = MoveSelectionMode(selection_mode)
        self.temperature = max(1e-4, temperature)
        self.max_steps = max_steps
        self.stop_at_local_optimum = stop_at_local_optimum

        # Construct policy if not passed explicitly
        if policy is not None:
            self.policy = policy
        else:
            num_feats = self.feature_config.get_num_features(self.n_agents)
            if policy_type == "linear":
                self.policy = LinearScorer(num_feats)
            elif policy_type in ("mlp", "per_action"):
                self.policy = PerActionScorer(
                    num_features=num_feats,
                    hidden_layers=hidden_layers,
                )
            elif policy_type == "deepsets":
                self.policy = DeepSetsPolicy(num_feats, hidden_dim=hidden_dim)
            elif policy_type == "invariant_nnet":
                layers_size = [num_feats, *hidden_layers, 1]
                self.policy = InvariantNNet(
                    n_actions=self.n_actions,
                    layers_size=layers_size,
                    remix=self.remix,
                    rescale=self.rescale,
                )
            elif policy_type == "peer_invariant_per_action":
                self.policy = PeerInvariantActionScorer(
                    num_base_features=1,
                    n_agents=self.n_agents,
                    peer_feature_dim=1,
                    hidden_dim=hidden_dim,
                )
            elif policy_type in ("dense", "dense_neighborhood"):
                self.policy = DenseNeighborhoodPolicy(
                    n_actions=self.n_actions,
                    num_features=num_feats,
                    hidden_dim=hidden_dim,
                )
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

        # Internal search state
        self._stagnation_counter: torch.Tensor | None = None
        self._prev_best_fitness: torch.Tensor | None = None

    @property
    def is_deterministic(self) -> bool:
        return self.selection_mode == MoveSelectionMode.ARGMAX or self.state_dependent_deterministic

    @property
    def uses_group_information(self) -> bool:
        """Whether this neural strategy takes peer or group features as inputs."""
        if self.n_agents <= 1:
            return False
        cfg = self.feature_config
        return bool(
            cfg.use_action_peer_deltas
            or cfg.use_net_peer_delta
            or cfg.use_group_scores
            or cfg.use_group_distances
        )

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="temperature",
                param_type="float",
                default=1.0,
                low=0.01,
                high=10.0,
                log_scale=True,
                description="Softmax temperature for move sampling.",
            ),
            HyperparameterSpec(
                name="selection_mode",
                param_type="str",
                default="argmax",
                choices=["argmax", "sample"],
                description="Move decision rule (argmax or probabilistic sampling).",
            ),
            HyperparameterSpec(
                name="stop_at_local_optimum",
                param_type="bool",
                default=False,
                description="Whether to terminate when no improving neighbor exists.",
            ),
        ]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "eval_mode": self.eval_mode.value,
            "policy_type": self.policy_type,
            "features": self.feature_config.features,
            "ablation_level": self.ablation_level.value,
            "temperature": self.temperature,
            "selection_mode": self.selection_mode.value,
            "stop_at_local_optimum": self.stop_at_local_optimum,
        }

    def reset(self, batch_size: int, device: torch.device, seed: SeedType = None) -> None:
        super().reset(batch_size=batch_size, device=device, seed=seed)
        self.policy.to(device)
        self._stagnation_counter = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self._prev_best_fitness = None

    def get_weights_vector(self) -> np.ndarray:
        """Export flat policy parameters for CMA-ES optimizer."""
        return self.policy.get_flat_weights()

    def set_weights_vector(self, weights: np.ndarray) -> None:
        """Import flat parameter vector into neural policy."""
        self.policy.set_flat_weights(weights)

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

        # Maintain stagnation counter
        if self._stagnation_counter is None or self._stagnation_counter.shape[0] != B:
            self._stagnation_counter = torch.zeros(B, dtype=torch.int64, device=device)
            self._prev_best_fitness = best_fitness.clone()

        if self._prev_best_fitness is not None:
            improved = best_fitness > self._prev_best_fitness
            self._stagnation_counter[improved] = 0
            self._stagnation_counter[~improved] += 1
            self._prev_best_fitness = best_fitness.clone()

        # Update n_agents if group_context is active
        if group_context is not None and group_context.n_agents != self.n_agents:
            self.n_agents = group_context.n_agents
            self.extractor.n_agents = self.n_agents

        # 1. Feature extraction: shape (B, N, F)
        features = self.extractor.extract(
            states=states,
            deltas=deltas,
            current_fitness=current_fitness,
            best_fitness=best_fitness,
            group_ctx=group_context,
            group_metrics=group_metrics,
            step=step,
            max_steps=self.max_steps if self.max_steps is not None else 0,
            stagnation_counter=self._stagnation_counter,
        )

        # 2. Forward pass through policy network: shape (B, N)
        with torch.no_grad():
            logits = self.policy(features)

        # 3. Action selection
        probs: torch.Tensor | None = None
        if self.selection_mode == MoveSelectionMode.ARGMAX:
            chosen = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits / self.temperature, dim=-1)
        else:
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)

            if self.state_dependent_deterministic:
                row_seeds = self._get_row_seeds(states, step)
                u = (
                    ((row_seeds.unsqueeze(1) * 1103515245 + 12345) % 2147483647).float()
                    / 2147483647.0
                ).clamp(1e-6, 1.0 - 1e-6)
                multipliers = torch.arange(1, N + 1, device=device).float().unsqueeze(0)
                u_n = ((u * multipliers) % 1.0).clamp(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u_n))
                chosen = torch.argmax(scaled_logits + gumbel, dim=-1)
            else:
                chosen = torch.multinomial(probs, num_samples=1, generator=self._generator).squeeze(
                    -1
                )

        # Check if chosen move is improving
        chosen_deltas = torch.gather(deltas, dim=1, index=chosen.unsqueeze(1)).squeeze(1)
        is_improving = chosen_deltas > 0.0

        if self.stop_at_local_optimum:
            is_optimum = (deltas <= 0.0).all(dim=-1)
            stop_mask = is_optimum
            chosen = torch.where(stop_mask, torch.tensor(-1, device=device), chosen)
        else:
            stop_mask = torch.zeros(B, dtype=torch.bool, device=device)

        return MoveResult(
            chosen_moves=chosen,
            is_improving=is_improving,
            probabilities=probs,
            logits=logits,
            stop_mask=stop_mask,
            metadata={
                "features": features,
                "stagnation": self._stagnation_counter.clone(),
            },
        )
