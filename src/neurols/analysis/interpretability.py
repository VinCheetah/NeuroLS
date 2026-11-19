"""Interpretability, diagnostic, and inspection tools for Neural Local Search policies.

Provides utilities to understand what features (delta fitness, agent distance,
group consensus, stagnation) the neural network utilizes to choose its search transitions,
interactively test single inputs, compute sensitivity curves, and evaluate in real NK conditions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from neurols.core.group import GroupConfig, GroupContext, GroupMetrics
from neurols.core.problem import BatchedProblem
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.neural.features import (
    FEATURE_CANONICAL,
    FEATURE_REGISTRY,
    EvaluationMode,
)
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.strategies.neural.policy import LinearScorer
from neurols.training.registry import ModelManifest, load_model_checkpoint

# ============================================================================
# 1. Feature Inspection Metadata & Bounds
# ============================================================================


@dataclass(frozen=True)
class FeatureInspectionInfo:
    """Detailed metadata and valid dynamic bounds for an individual feature channel."""

    name: str
    canonical_name: str
    category: Literal["action", "agent", "context"]
    description: str
    formula: str
    default_val: float
    min_val: float
    max_val: float
    step_val: float
    is_binary: bool = False
    is_discrete: bool = False


def get_feature_metadata(channel_name: str, n_agents: int = 1) -> FeatureInspectionInfo:
    """Retrieve inspection metadata, physical bounds, and UI controls for a feature channel."""
    # Strip channel index suffix if present (e.g. peer_deltas_0 -> peer_deltas)
    base_name = channel_name
    for prefix in (
        "peer_deltas",
        "peer_distances",
        "peer_scores",
        "peer_delta_scores",
    ):
        if channel_name.startswith(f"{prefix}_"):
            base_name = prefix
            break

    canonical = FEATURE_CANONICAL.get(base_name, base_name)
    desc = FEATURE_REGISTRY.get(canonical)

    category: Literal["action", "agent", "context"] = desc.category if desc else "action"
    description: str = desc.description if desc else f"Feature channel {channel_name}"
    formula: str = desc.formula if desc else channel_name

    # Determine domain bounds, step, and defaults based on canonical feature semantics
    if canonical in ("delta", "delta_fitness"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.08,
            min_val=-1.0,
            max_val=1.0,
            step_val=0.01,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical == "ranked_delta":
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.5,
            min_val=-1.0,
            max_val=1.0,
            step_val=0.05,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical == "peer_deltas":
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=-1.0,
            min_val=-1.0,
            max_val=1.0,
            step_val=1.0,
            is_binary=False,
            is_discrete=True,
        )
    elif canonical == "net_peer_delta":
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.0,
            min_val=-1.0,
            max_val=1.0,
            step_val=0.05,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical in ("current_locus", "current_bit", "best_locus"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.0,
            min_val=0.0,
            max_val=1.0,
            step_val=1.0,
            is_binary=True,
            is_discrete=True,
        )
    elif canonical in ("group_mean_locus", "peers_mean_locus"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.5,
            min_val=0.0,
            max_val=1.0,
            step_val=0.05,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical in ("current_fitness", "fitness", "peer_scores"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.65,
            min_val=0.0,
            max_val=1.0,
            step_val=0.01,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical in ("dist_group_barycentre", "dist_others_barycentre", "peer_distances"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.35,
            min_val=0.0,
            max_val=1.0,
            step_val=0.05,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical in ("delta_mean_score", "peer_delta_scores"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.0,
            min_val=-1.0,
            max_val=1.0,
            step_val=0.01,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical == "normalized_rank":
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.5,
            min_val=0.0,
            max_val=1.0,
            step_val=0.05,
            is_binary=False,
            is_discrete=False,
        )
    elif canonical in ("fitness_gap_to_best", "step_progress", "stagnation_ratio"):
        return FeatureInspectionInfo(
            name=channel_name,
            canonical_name=canonical,
            category=category,
            description=description,
            formula=formula,
            default_val=0.1,
            min_val=0.0,
            max_val=1.0,
            step_val=0.01,
            is_binary=False,
            is_discrete=False,
        )

    # General fallback
    return FeatureInspectionInfo(
        name=channel_name,
        canonical_name=canonical,
        category=category,
        description=description,
        formula=formula,
        default_val=0.0,
        min_val=-1.0,
        max_val=1.0,
        step_val=0.05,
        is_binary=False,
        is_discrete=False,
    )


def sample_feature_value(channel_name: str, seed: int | None = None) -> float:
    """Generate a realistic random value within the physical domain of the feature."""
    rng = random.Random(seed) if seed is not None else random
    info = get_feature_metadata(channel_name)

    if info.is_binary:
        return float(rng.choice([0.0, 1.0]))
    if info.is_discrete:
        return float(rng.choice([-1.0, 1.0]))
    if info.canonical_name in ("delta", "delta_fitness"):
        # Roughly normal around 0 with realistic spread
        val = rng.gauss(0.0, 0.15)
        return float(np.clip(round(val, 4), info.min_val, info.max_val))
    if info.canonical_name == "ranked_delta":
        return float(round(rng.uniform(-1.0, 1.0), 3))
    if info.min_val >= 0.0 and info.max_val <= 1.0:
        return float(round(rng.uniform(info.min_val, info.max_val), 3))

    return float(round(rng.uniform(info.min_val, info.max_val), 3))


# ============================================================================
# 2. Model Inspector Engine
# ============================================================================


@dataclass
class SingleActionEvaluation:
    """Result of evaluating a neural policy on an isolated candidate move."""

    score: float
    is_positive: bool
    linear_breakdown: dict[str, float] | None
    linear_bias: float | None
    feature_sensitivities: dict[str, float]
    raw_features: dict[str, float]
    confidence_probability: float | None = None


@dataclass
class SensitivityCurveResult:
    """1D sensitivity curve showing score f(x) as a function of one feature."""

    feature_name: str
    x_values: list[float]
    scores: list[float]
    current_x: float
    current_score: float
    zero_crossings: list[float]
    local_gradient: float
    min_score: float
    max_score: float
    is_monotonic: bool
    frozen_features: dict[str, float]


@dataclass
class SensitivitySurfaceResult:
    """2D sensitivity surface showing score f(x1, x2) as a function of two features."""

    feat1: str
    feat2: str
    grid_x: list[float]
    grid_y: list[float]
    scores_grid: list[list[float]]
    current_point: tuple[float, float, float]
    frozen_features: dict[str, float]


class ModelInspector:
    """Comprehensive interpretability inspector for Neural Local Search strategies.

    Enables:
    - Dynamic discovery of feature channels required by the model.
    - Isolated single-action evaluation with manual feature adjustments.
    - Sensitivity sweep plotting f(x) for any feature.
    - 2D interaction surface plotting f(x1, x2).
    - Linear weight decomposition and autograd saliency gradients.
    """

    def __init__(
        self,
        strategy: NeuralLocalSearchStrategy,
        manifest: ModelManifest | None = None,
        n_agents: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.manifest = manifest
        self.n_agents = n_agents or strategy.n_agents
        self.feature_names: list[str] = strategy.feature_config.get_feature_names(self.n_agents)
        self.feature_infos: list[FeatureInspectionInfo] = [
            get_feature_metadata(name, self.n_agents) for name in self.feature_names
        ]
        self.is_per_action: bool = strategy.eval_mode == EvaluationMode.PER_ACTION
        self.is_linear: bool = isinstance(strategy.policy, LinearScorer)

    @classmethod
    def from_checkpoint(
        cls,
        query_or_path: str | Path,
        base_dir: str | Path = "checkpoints",
        override_n_actions: int | None = None,
        override_n_agents: int | None = None,
    ) -> ModelInspector:
        """Instantiate an inspector from a saved JSON checkpoint."""
        strategy, manifest = load_model_checkpoint(
            query_or_path=query_or_path,
            base_dir=base_dir,
            override_n_actions=override_n_actions,
            override_n_agents=override_n_agents,
        )
        return cls(strategy=strategy, manifest=manifest, n_agents=override_n_agents)

    def get_feature_info(self, name: str) -> FeatureInspectionInfo:
        """Find metadata for a given feature name."""
        for info in self.feature_infos:
            if info.name == name:
                return info
        return get_feature_metadata(name, self.n_agents)

    def get_default_feature_vector(self) -> dict[str, float]:
        """Return the default feature values for all channels."""
        return {info.name: info.default_val for info in self.feature_infos}

    def sample_random_feature_vector(self, seed: int | None = None) -> dict[str, float]:
        """Generate a random feature vector across all channels respecting physical bounds."""
        rng = random.Random(seed) if seed is not None else random
        return {
            info.name: sample_feature_value(
                info.name, seed=rng.randint(0, 1000000) if seed is not None else None
            )
            for info in self.feature_infos
        }

    def evaluate_single_action(
        self,
        features_dict: dict[str, float],
    ) -> SingleActionEvaluation:
        """Compute the network score and feature attribution for a single candidate action.

        Args:
            features_dict: Mapping of feature channel names to scalar values.

        Returns:
            SingleActionEvaluation with score, breakdown, sensitivities, etc.
        """
        # Vectorize features in deterministic order
        vec = [float(features_dict.get(name, 0.0)) for name in self.feature_names]
        feat_tensor = torch.tensor([[vec]], dtype=torch.float32)  # shape (1, 1, F)

        # 1. Forward pass
        linear_breakdown: dict[str, float] | None = None
        linear_bias: float | None = None

        if self.is_linear and isinstance(self.strategy.policy, LinearScorer):
            weights = (
                self.strategy.policy.linear.weight.squeeze(0).detach().cpu().numpy().tolist()
            )
            bias = float(self.strategy.policy.linear.bias.item())
            linear_bias = bias
            linear_breakdown = {}
            score = bias
            for name, val, w in zip(self.feature_names, vec, weights, strict=False):
                contrib = float(w * val)
                linear_breakdown[name] = contrib
                score += contrib
        else:
            with torch.no_grad():
                if self.is_per_action:
                    out = self.strategy.policy(feat_tensor)  # (1, 1)
                    score = float(out[0, 0].item())
                else:
                    # For all-actions networks, candidate actions are evaluated in neighborhood of size N.
                    # We broadcast this feature vector across N actions to obtain a representative score.
                    N = max(self.strategy.n_actions, 1)
                    feat_rep = feat_tensor.repeat(1, N, 1)
                    out = self.strategy.policy(feat_rep)  # (1, N)
                    score = float(out[0, 0].item())

        # 2. Gradient-based sensitivity computation via autograd
        feat_for_grad = feat_tensor.clone().detach().requires_grad_(True)
        if self.is_per_action:
            out_grad = self.strategy.policy(feat_for_grad)
            scalar_loss = out_grad[0, 0]
        else:
            N = max(self.strategy.n_actions, 1)
            feat_rep = feat_for_grad.repeat(1, N, 1)
            out_grad = self.strategy.policy(feat_rep)
            scalar_loss = out_grad[0, 0]

        scalar_loss.backward()
        grads = feat_for_grad.grad[0, 0].detach().cpu().numpy()  # (F,)
        sensitivities = {
            name: float(abs(g)) for name, g in zip(self.feature_names, grads, strict=False)
        }

        # Sigmoid probability equivalence for standalone score visualization
        prob = float(torch.sigmoid(torch.tensor(score)).item())

        return SingleActionEvaluation(
            score=score,
            is_positive=score > 0.0,
            linear_breakdown=linear_breakdown,
            linear_bias=linear_bias,
            feature_sensitivities=sensitivities,
            raw_features=dict(features_dict),
            confidence_probability=prob,
        )

    def compute_sensitivity_curve(
        self,
        target_feature: str,
        base_features: dict[str, float],
        min_val: float | None = None,
        max_val: float | None = None,
        num_points: int = 100,
    ) -> SensitivityCurveResult:
        """Compute the response curve S(x) by sweeping target_feature over [min_val, max_val].

        Args:
            target_feature: Feature channel to vary.
            base_features: Fixed baseline values for all other features.
            min_val: Sweep minimum (defaults to feature info min).
            max_val: Sweep maximum (defaults to feature info max).
            num_points: Evaluation resolution.

        Returns:
            SensitivityCurveResult with values, scores, zero crossings, and local gradient.
        """
        info = self.get_feature_info(target_feature)
        x_min = min_val if min_val is not None else info.min_val
        x_max = max_val if max_val is not None else info.max_val
        if x_min >= x_max:
            x_max = x_min + 1.0

        xs = np.linspace(x_min, x_max, num_points).tolist()
        scores: list[float] = []

        feat_copy = dict(base_features)
        for x in xs:
            feat_copy[target_feature] = x
            res = self.evaluate_single_action(feat_copy)
            scores.append(res.score)

        current_x = float(base_features.get(target_feature, info.default_val))
        feat_copy[target_feature] = current_x
        current_res = self.evaluate_single_action(feat_copy)
        current_score = current_res.score

        # Detect zero crossings
        zero_crossings: list[float] = []
        for i in range(len(scores) - 1):
            s1, s2 = scores[i], scores[i + 1]
            if (s1 < 0.0 and s2 >= 0.0) or (s1 > 0.0 and s2 <= 0.0):
                if abs(s2 - s1) > 1e-7:
                    root = xs[i] - s1 * (xs[i + 1] - xs[i]) / (s2 - s1)
                else:
                    root = xs[i]
                if not any(abs(root - z) < 1e-4 for z in zero_crossings):
                    zero_crossings.append(float(root))
            elif abs(s1) < 1e-7 and not any(abs(xs[i] - z) < 1e-4 for z in zero_crossings):
                zero_crossings.append(float(xs[i]))

        # Finite difference local gradient at current operating point
        eps = 1e-4
        feat_copy[target_feature] = current_x + eps
        s_plus = self.evaluate_single_action(feat_copy).score
        feat_copy[target_feature] = current_x - eps
        s_minus = self.evaluate_single_action(feat_copy).score
        local_grad = float((s_plus - s_minus) / (2.0 * eps))

        diffs = np.diff(scores)
        is_monotonic = bool(np.all(diffs >= -1e-5) or np.all(diffs <= 1e-5))

        frozen = {k: v for k, v in base_features.items() if k != target_feature}

        return SensitivityCurveResult(
            feature_name=target_feature,
            x_values=[float(x) for x in xs],
            scores=scores,
            current_x=current_x,
            current_score=current_score,
            zero_crossings=zero_crossings,
            local_gradient=local_grad,
            min_score=float(min(scores)),
            max_score=float(max(scores)),
            is_monotonic=is_monotonic,
            frozen_features=frozen,
        )

    def compute_sensitivity_surface(
        self,
        feat1: str,
        feat2: str,
        base_features: dict[str, float],
        range1: tuple[float, float] | None = None,
        range2: tuple[float, float] | None = None,
        grid_size: int = 25,
    ) -> SensitivitySurfaceResult:
        """Compute a 2D interaction grid of scores S(feat1, feat2)."""
        info1 = self.get_feature_info(feat1)
        info2 = self.get_feature_info(feat2)

        x1_min, x1_max = range1 if range1 else (info1.min_val, info1.max_val)
        x2_min, x2_max = range2 if range2 else (info2.min_val, info2.max_val)

        grid_x = np.linspace(x1_min, x1_max, grid_size).tolist()
        grid_y = np.linspace(x2_min, x2_max, grid_size).tolist()

        feat_copy = dict(base_features)
        scores_grid: list[list[float]] = []

        for y in grid_y:
            row: list[float] = []
            feat_copy[feat2] = y
            for x in grid_x:
                feat_copy[feat1] = x
                s = self.evaluate_single_action(feat_copy).score
                row.append(s)
            scores_grid.append(row)

        curr_x = float(base_features.get(feat1, info1.default_val))
        curr_y = float(base_features.get(feat2, info2.default_val))
        feat_copy[feat1] = curr_x
        feat_copy[feat2] = curr_y
        curr_z = self.evaluate_single_action(feat_copy).score

        frozen = {k: v for k, v in base_features.items() if k not in (feat1, feat2)}

        return SensitivitySurfaceResult(
            feat1=feat1,
            feat2=feat2,
            grid_x=[float(x) for x in grid_x],
            grid_y=[float(y) for y in grid_y],
            scores_grid=scores_grid,
            current_point=(curr_x, curr_y, curr_z),
            frozen_features=frozen,
        )


# ============================================================================
# 3. Real NK Landscape Interactive Session
# ============================================================================


class RealNKSession:
    """Simulates and tracks local search trajectories on an exact NK Landscape instance.

    Allows the user to inspect real extracted features, visualize the model's choices,
    and step through the search space either by following the model's choice or making manual moves.
    """

    def __init__(
        self,
        strategy: NeuralLocalSearchStrategy,
        n: int = 12,
        k: int = 2,
        interaction_type: str = "adjacent",
        n_agents: int = 1,
        seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.n = n
        self.k = min(k, n - 1)
        self.interaction_type = interaction_type
        self.n_agents = max(n_agents, 1)
        self.seed = seed

        self.problem = NKLandscape(
            n=self.n,
            k=self.k,
            interaction_type=self.interaction_type,
            seed=self.seed,
            device="cpu",
        )

        self.group_config = GroupConfig(n_agents=self.n_agents)
        self.group_context: GroupContext | None = (
            GroupContext(self.group_config, n_instances=1, n_restarts=1, device="cpu")
            if self.n_agents > 1
            else None
        )

        self.step_count = 0
        self.states: torch.Tensor = torch.zeros((self.n_agents, self.n), dtype=torch.int8)
        self.best_states: torch.Tensor = torch.zeros((self.n_agents, self.n), dtype=torch.int8)
        self.best_fitness: torch.Tensor = torch.zeros(self.n_agents, dtype=torch.float32)
        self.stagnation_counter: torch.Tensor = torch.zeros(self.n_agents, dtype=torch.int64)

        self.trajectory_history: list[dict[str, Any]] = []
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Sample fresh initial random states for all agents and reset trajectory."""
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        self.states = self.problem.sample_initial_states((self.n_agents,), generator=gen)
        self.best_states = self.states.clone()
        init_fit = self.problem.evaluate(self.states)
        self.best_fitness = init_fit.clone()
        self.step_count = 0
        self.stagnation_counter = torch.zeros(self.n_agents, dtype=torch.int64)

        self.trajectory_history = [
            {
                "step": 0,
                "fitness": init_fit.cpu().numpy().tolist(),
                "best_fitness": self.best_fitness.cpu().numpy().tolist(),
                "action_taken": [-1] * self.n_agents,
                "action_type": ["init"] * self.n_agents,
                "delta": [0.0] * self.n_agents,
            }
        ]

    def evaluate_current_state(self) -> dict[str, Any]:
        """Compute full neighborhood deltas, features, logits, and choices on current states."""
        curr_fit = self.problem.evaluate(self.states)  # (M,)
        deltas = self.problem.get_neighborhood_deltas(self.states)  # (M, N)

        # Update best fitness
        improved = curr_fit > self.best_fitness
        self.best_fitness[improved] = curr_fit[improved]
        self.best_states[improved] = self.states[improved]
        self.stagnation_counter[improved] = 0
        self.stagnation_counter[~improved] += 1

        # Multi-agent group metrics
        group_metrics: GroupMetrics | None = None
        if self.group_context is not None:
            group_metrics = self.group_context.update(
                states=self.states,
                current_fitness=curr_fit,
                best_fitness=self.best_fitness,
                best_states=self.best_states,
            )

        # Extract features for all candidate moves
        features = self.strategy.extractor.extract(
            states=self.states,
            deltas=deltas,
            current_fitness=curr_fit,
            best_fitness=self.best_fitness,
            group_ctx=self.group_context,
            group_metrics=group_metrics,
            step=self.step_count,
            max_steps=500,
            stagnation_counter=self.stagnation_counter,
        )  # (M, N, F)

        # Model forward pass
        with torch.no_grad():
            if self.strategy.eval_mode == EvaluationMode.ALL_ACTIONS:
                _, sort_indices = torch.sort(deltas, dim=-1, descending=True)
                gather_idx = sort_indices.unsqueeze(-1).expand(
                    self.n_agents, self.n, features.shape[-1]
                )
                features_sorted = torch.gather(features, dim=1, index=gather_idx)
                sorted_logits = self.strategy.policy(features_sorted)
                logits = torch.empty_like(sorted_logits).scatter_(
                    dim=-1, index=sort_indices, src=sorted_logits
                )
            else:
                logits = self.strategy.policy(features)  # (M, N)

            probs = F.softmax(logits / max(self.strategy.temperature, 1e-4), dim=-1)
            chosen_argmax = torch.argmax(logits, dim=-1)  # (M,)

        feature_names = self.strategy.feature_config.get_feature_names(self.n_agents)

        # Build candidate action records for each agent
        agents_data: list[dict[str, Any]] = []
        for a_idx in range(self.n_agents):
            actions_list: list[dict[str, Any]] = []
            for bit in range(self.n):
                d_val = float(deltas[a_idx, bit].item())
                l_val = float(logits[a_idx, bit].item())
                p_val = float(probs[a_idx, bit].item())
                feat_vals = {
                    name: float(features[a_idx, bit, f_idx].item())
                    for f_idx, name in enumerate(feature_names)
                }
                actions_list.append(
                    {
                        "bit_index": bit,
                        "delta_fitness": d_val,
                        "is_improving": d_val > 0.0,
                        "logit": l_val,
                        "probability": p_val,
                        "is_chosen": bool(bit == chosen_argmax[a_idx].item()),
                        "features": feat_vals,
                    }
                )

            # Pairwise distances to other peers
            peer_dists: list[float] = []
            if self.n_agents > 1:
                cur_state = self.states[a_idx]
                for p_idx in range(self.n_agents):
                    if p_idx != a_idx:
                        dist = float((cur_state != self.states[p_idx]).sum().item())
                        peer_dists.append(dist)

            agents_data.append(
                {
                    "agent_index": a_idx,
                    "current_state": self.states[a_idx].cpu().numpy().tolist(),
                    "current_fitness": float(curr_fit[a_idx].item()),
                    "best_fitness": float(self.best_fitness[a_idx].item()),
                    "stagnation": int(self.stagnation_counter[a_idx].item()),
                    "chosen_action": int(chosen_argmax[a_idx].item()),
                    "chosen_delta": float(deltas[a_idx, chosen_argmax[a_idx]].item()),
                    "peer_distances": peer_dists,
                    "candidate_actions": actions_list,
                }
            )

        diversity = 0.0
        if self.n_agents > 1 and group_metrics is not None:
            if isinstance(group_metrics.group_diversity, torch.Tensor):
                diversity = float(group_metrics.group_diversity.mean().item())
            else:
                diversity = float(group_metrics.group_diversity)

        return {
            "step": self.step_count,
            "n_agents": self.n_agents,
            "dimension": self.n,
            "agents": agents_data,
            "diversity": diversity,
        }

    def step_model(self, agent_idx: int = 0) -> dict[str, Any]:
        """Advance the search by applying the model's chosen action for agent_idx."""
        diag = self.evaluate_current_state()
        chosen = diag["agents"][agent_idx]["chosen_action"]
        delta = diag["agents"][agent_idx]["chosen_delta"]

        move_tensor = torch.tensor([chosen], dtype=torch.int64)
        agent_state = self.states[agent_idx : agent_idx + 1]
        new_state = self.problem.apply_moves(agent_state, move_tensor)
        self.states[agent_idx : agent_idx + 1] = new_state

        self.step_count += 1
        curr_fit = self.problem.evaluate(self.states)
        self.trajectory_history.append(
            {
                "step": self.step_count,
                "fitness": curr_fit.cpu().numpy().tolist(),
                "best_fitness": self.best_fitness.cpu().numpy().tolist(),
                "action_taken": [chosen if i == agent_idx else -1 for i in range(self.n_agents)],
                "action_type": ["model" if i == agent_idx else "idle" for i in range(self.n_agents)],
                "delta": [delta if i == agent_idx else 0.0 for i in range(self.n_agents)],
            }
        )
        return {
            "agent_index": agent_idx,
            "chosen_bit": chosen,
            "delta": delta,
            "new_fitness": float(curr_fit[agent_idx].item()),
        }

    def step_manual(self, bit_idx: int, agent_idx: int = 0) -> dict[str, Any]:
        """Advance the search by applying a user-selected manual bit flip to agent_idx."""
        deltas = self.problem.get_neighborhood_deltas(self.states)
        delta = float(deltas[agent_idx, bit_idx].item())

        move_tensor = torch.tensor([bit_idx], dtype=torch.int64)
        agent_state = self.states[agent_idx : agent_idx + 1]
        new_state = self.problem.apply_moves(agent_state, move_tensor)
        self.states[agent_idx : agent_idx + 1] = new_state

        self.step_count += 1
        curr_fit = self.problem.evaluate(self.states)
        self.trajectory_history.append(
            {
                "step": self.step_count,
                "fitness": curr_fit.cpu().numpy().tolist(),
                "best_fitness": self.best_fitness.cpu().numpy().tolist(),
                "action_taken": [bit_idx if i == agent_idx else -1 for i in range(self.n_agents)],
                "action_type": ["manual" if i == agent_idx else "idle" for i in range(self.n_agents)],
                "delta": [delta if i == agent_idx else 0.0 for i in range(self.n_agents)],
            }
        )
        return {
            "agent_index": agent_idx,
            "chosen_bit": bit_idx,
            "delta": delta,
            "new_fitness": float(curr_fit[agent_idx].item()),
        }

    def step_all_agents_model(self) -> list[dict[str, Any]]:
        """Simultaneously advance all M agents according to their respective model decisions."""
        diag = self.evaluate_current_state()
        chosen_moves = [diag["agents"][i]["chosen_action"] for i in range(self.n_agents)]
        deltas = [diag["agents"][i]["chosen_delta"] for i in range(self.n_agents)]

        moves_tensor = torch.tensor(chosen_moves, dtype=torch.int64)
        self.states = self.problem.apply_moves(self.states, moves_tensor)

        self.step_count += 1
        curr_fit = self.problem.evaluate(self.states)
        self.trajectory_history.append(
            {
                "step": self.step_count,
                "fitness": curr_fit.cpu().numpy().tolist(),
                "best_fitness": self.best_fitness.cpu().numpy().tolist(),
                "action_taken": list(chosen_moves),
                "action_type": ["model"] * self.n_agents,
                "delta": list(deltas),
            }
        )

        return [
            {
                "agent_index": i,
                "chosen_bit": chosen_moves[i],
                "delta": deltas[i],
                "new_fitness": float(curr_fit[i].item()),
            }
            for i in range(self.n_agents)
        ]

    def toggle_bit(self, agent_idx: int, bit_idx: int) -> None:
        """Directly toggle a bit in agent's current state without recording trajectory step."""
        self.states[agent_idx, bit_idx] ^= 1

    def get_action_features(self, agent_idx: int, action_idx: int) -> dict[str, float]:
        """Extract the exact feature dictionary of a candidate move to transfer to sliders."""
        diag = self.evaluate_current_state()
        actions = diag["agents"][agent_idx]["candidate_actions"]
        return dict(actions[action_idx]["features"])


# ============================================================================
# 4. Backward-Compatible Functions
# ============================================================================


def explain_decision(
    strategy: NeuralLocalSearchStrategy,
    states: torch.Tensor,
    problem: BatchedProblem,
    step: int = 0,
) -> dict[str, Any]:
    """Provide a transparent diagnostic breakdown of the strategy's decision on a given state."""
    deltas = problem.get_neighborhood_deltas(states)
    curr_fit = problem.evaluate(states)

    move_res = strategy.select_move(
        states=states,
        deltas=deltas,
        current_fitness=curr_fit,
        best_fitness=curr_fit,
        step=step,
    )

    chosen_idx = int(move_res.chosen_moves[0].item())
    probs = (
        move_res.probabilities[0].detach().cpu().numpy()
        if move_res.probabilities is not None
        else None
    )
    logits = move_res.logits[0].detach().cpu().numpy() if move_res.logits is not None else None

    feature_names = strategy.feature_config.feature_names
    chosen_features = (
        move_res.metadata["features"][0, chosen_idx].detach().cpu().numpy()
        if "features" in move_res.metadata
        else None
    )

    feature_breakdown = {}
    if chosen_features is not None:
        for name, val in zip(feature_names, chosen_features, strict=False):
            feature_breakdown[name] = float(val)

    return {
        "chosen_move": chosen_idx,
        "is_improving": bool(move_res.is_improving[0].item()),
        "chosen_delta": float(deltas[0, chosen_idx].item()) if chosen_idx >= 0 else 0.0,
        "features_of_chosen_move": feature_breakdown,
        "probabilities": probs.tolist() if probs is not None else None,
        "logits": logits.tolist() if logits is not None else None,
    }


def compute_feature_sensitivities(
    strategy: NeuralLocalSearchStrategy,
    states: torch.Tensor,
    problem: BatchedProblem,
) -> dict[str, float]:
    """Compute average gradient-based saliency / sensitivity of each input feature."""
    deltas = problem.get_neighborhood_deltas(states)
    curr_fit = problem.evaluate(states)

    if isinstance(strategy.policy, LinearScorer):
        w_dict = strategy.policy.get_feature_weights(strategy.feature_config.feature_names)
        return {k: abs(v) for k, v in w_dict.items()}

    features = strategy.extractor.extract(
        states=states,
        deltas=deltas,
        current_fitness=curr_fit,
        best_fitness=curr_fit,
        group_metrics=None,
        step=0,
        max_steps=100,
        stagnation_counter=torch.zeros(states.shape[0], dtype=torch.int64, device=states.device),
    )

    features.requires_grad_(True)
    logits = strategy.policy(features)
    loss = logits.sum()
    loss.backward()

    grads = features.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    feat_names = strategy.feature_config.feature_names
    return {name: float(g) for name, g in zip(feat_names, grads, strict=False)}
