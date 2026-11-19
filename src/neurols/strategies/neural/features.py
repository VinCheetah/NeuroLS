"""Feature extraction and specification DSL for Neural Local Search.

Translates problem states, candidate move deltas, multi-agent group metrics, and
action-specific peer distance transitions into structured tensor representations,
with explicit support for both:
1. Per-Action Evaluation: scoring each move independently with action-level signals
   and group context.
2. All-Actions Joint Evaluation: scoring the entire neighborhood at once (either via
   permutation-equivariant DeepSets architectures or dense neighborhood networks).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import torch

from neurols.core.group import GroupContext, GroupMetrics


class EvaluationMode(StrEnum):
    """Decision evaluation paradigm."""

    PER_ACTION = "per_action"
    ALL_ACTIONS = "all_actions"


class FeatureAblationLevel(StrEnum):
    """Standardized feature ablation tiers for backward compatibility and quick benchmarks."""

    DELTA_ONLY = "delta_only"
    RANKED_DELTA_ONLY = "ranked_delta_only"
    DELTA_AND_PEERS = "delta_and_peers"
    DELTA_PEERS_RANK = "delta_peers_rank"
    FULL = "full"


def compute_normalized_ranks(delta_fitness: torch.Tensor) -> torch.Tensor:
    """Compute dual-pole normalized ranks for delta fitness across candidate actions.

    Negative deltas are normalized to [-1, 0] and positive deltas to (0, 1].
    Provides scale-invariance across different fitness ranges.

    Args:
        delta_fitness: Tensor of shape (B, N).

    Returns:
        Tensor of shape (B, N) containing normalized ranks in [-1.0, 1.0].
    """
    B, N = delta_fitness.shape
    device = delta_fitness.device

    number_positive = (delta_fitness > 0).sum(dim=1, keepdim=True)
    number_negative = N - number_positive

    test_range = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    rank = test_range.unsqueeze(0) - number_negative  # (B, N)
    pos_denom = number_positive.clamp_min(1)
    neg_denom = number_negative.clamp_min(1)

    normalized_rank = torch.where(
        rank <= 0,
        (rank - 1) / neg_denom,
        rank / pos_denom,
    )

    _, sorted_action = delta_fitness.sort(dim=-1, descending=False)
    ranked_delta = torch.zeros_like(delta_fitness)
    ranked_delta.scatter_(1, sorted_action, normalized_rank)
    return ranked_delta


def compute_action_peer_deltas(
    states: torch.Tensor,
    group_ctx: GroupContext | None,
    sort_peers: bool = True,
    group_metrics: GroupMetrics | None = None,
    current_fitness: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute action-specific peer distance changes (+1 / -1) for each candidate 1-bit flip.

    For each candidate bit flip j in {0, ..., N-1} and peer m:
    - Flipping bit j increases distance to peer m by +1 if x_m[j] == x_self[j].
    - Flipping bit j decreases distance to peer m by -1 if x_m[j] != x_self[j].

    Peers are canonically ordered by descending current fitness (best peer first) to provide
    a consistent, permutation-invariant semantic representation.

    Args:
        states: Current states tensor, shape (B, N).
        group_ctx: Multi-agent group context.
        sort_peers: Whether to ensure peers are ordered by fitness.
        group_metrics: Precomputed GroupMetrics (optional, for speed).
        current_fitness: Current fitness tensor of agents, shape (B,).

    Returns:
        peer_deltas: Tensor of shape (B, N, max(M-1, 1)) in {-1.0, +1.0}.
        net_peer_delta: Tensor of shape (B, N, 1) in [-1.0, +1.0] (mean change across peers).
    """
    B, N = states.shape
    device = states.device

    if (
        group_ctx is None
        or group_ctx.n_agents <= 1
        or (group_ctx.config is not None and not group_ctx.config.share_distances)
    ):
        n_peers = max(group_ctx.n_agents - 1 if group_ctx is not None else 1, 1)
        zeros_peers = torch.zeros((B, N, n_peers), device=device, dtype=torch.float32)
        zeros_net = torch.zeros((B, N, 1), device=device, dtype=torch.float32)
        return zeros_peers, zeros_net

    G = group_ctx.n_groups
    M = group_ctx.n_agents
    D = N
    states_grouped = states.view(G, M, D)

    eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)

    if group_metrics is not None and group_metrics.ordered_peer_indices.shape[-1] == M - 1:
        ordered_peer_idx = group_metrics.ordered_peer_indices.view(G, M, M - 1)
    elif current_fitness is not None and sort_peers:
        # Canonical ordering: sort peers by descending current fitness
        fitness_grouped = current_fitness.view(G, M)
        peer_fit_matrix = fitness_grouped.unsqueeze(1).expand(G, M, M).clone()
        peer_fit_masked = peer_fit_matrix.masked_fill(eye, -float("inf"))
        _, sorted_peer_idx = peer_fit_masked.sort(dim=-1, descending=True)
        ordered_peer_idx = sorted_peer_idx[:, :, : M - 1]
    else:
        # Default fallback: mask self on diagonal
        ordered_peer_idx = (
            torch.arange(M, device=device).expand(G, M, M).masked_select(~eye).view(G, M, M - 1)
        )

    # Gather ordered peer states: shape (G, M, M - 1, D)
    gather_idx = ordered_peer_idx.unsqueeze(-1).expand(G, M, M - 1, D)
    ordered_peer_states = torch.gather(
        states_grouped.unsqueeze(1).expand(G, M, M, D), dim=2, index=gather_idx
    )

    # diff is True if x_self[j] != x_peer[j] -> flip reduces distance (-1)
    diff = states_grouped.unsqueeze(2) != ordered_peer_states  # (G, M, M - 1, D)
    peer_deltas_raw = torch.where(diff, -1.0, 1.0)  # (G, M, M - 1, D)

    # Permute to (G, M, D, M - 1) and reshape to (B, N, M - 1)
    peer_deltas = peer_deltas_raw.permute(0, 1, 3, 2).reshape(B, N, M - 1)

    # When no explicit fitness or metric ordering is supplied, sort deltas directly
    if group_metrics is None and current_fitness is None and sort_peers and (M - 1) > 1:
        peer_deltas, _ = peer_deltas.sort(dim=-1, descending=False)

    net_peer_delta = peer_deltas.mean(dim=-1, keepdim=True)  # (B, N, 1)

    return peer_deltas, net_peer_delta


# ============================================================================
# Feature Registry & Descriptors
# ============================================================================


@dataclass(frozen=True)
class FeatureDescriptor:
    """Metadata describing an individual feature channel."""

    name: str
    category: Literal["action", "agent", "context"]
    description: str
    formula: str
    is_multi_agent: bool
    dim_fn: Callable[[int], int]

    def get_dimension(self, n_agents: int) -> int:
        if self.is_multi_agent and n_agents <= 1:
            return 0
        return self.dim_fn(n_agents)


# Complete registry of all supported features
FEATURE_REGISTRY: dict[str, FeatureDescriptor] = {
    # --- Action-Specific Features ---
    "delta": FeatureDescriptor(
        name="delta",
        category="action",
        description="Raw fitness change delta_f(x, j) for flipping candidate bit j",
        formula="f(x ^ e_j) - f(x)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "delta_fitness": FeatureDescriptor(
        name="delta_fitness",
        category="action",
        description="Alias for delta (raw fitness change for candidate bit flip)",
        formula="f(x ^ e_j) - f(x)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "ranked_delta": FeatureDescriptor(
        name="ranked_delta",
        category="action",
        description="Dual-pole normalized rank of delta_f(x, j) across all N candidate moves in [-1, 1]",
        formula="dual_pole_rank(delta)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "peer_deltas": FeatureDescriptor(
        name="peer_deltas",
        category="action",
        description="(M - 1) distance changes in {-1, +1} towards each peer (ordered by peer fitness)",
        formula="-1 if x_i[j] != x_peer[j] else +1",
        is_multi_agent=True,
        dim_fn=lambda m: max(m - 1, 0),
    ),
    "net_peer_delta": FeatureDescriptor(
        name="net_peer_delta",
        category="action",
        description="Mean distance change across all (M - 1) peers for candidate bit flip j in [-1, 1]",
        formula="mean_peer(peer_deltas)",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "group_mean_locus": FeatureDescriptor(
        name="group_mean_locus",
        category="action",
        description="Mean bit value across all agents in group for locus j in [0, 1]",
        formula="(1 / M) * sum_m x_m[j]",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "peers_mean_locus": FeatureDescriptor(
        name="peers_mean_locus",
        category="action",
        description="Mean bit value across other peers in group for locus j in [0, 1]",
        formula="(1 / (M - 1)) * sum_{m != i} x_m[j]",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "best_locus": FeatureDescriptor(
        name="best_locus",
        category="action",
        description="Bit value of current group best-performing agent for locus j in {0, 1}",
        formula="x_{group_best}[j]",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "current_locus": FeatureDescriptor(
        name="current_locus",
        category="action",
        description="Current state bit value x_i[j] in {0, 1}",
        formula="x_i[j]",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "current_bit": FeatureDescriptor(
        name="current_bit",
        category="action",
        description="Alias for current_locus",
        formula="x_i[j]",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    # --- Agent / Group-Level Features (Valable pour les deux) ---
    "current_fitness": FeatureDescriptor(
        name="current_fitness",
        category="agent",
        description="Current agent objective fitness value f(x_i)",
        formula="f(x_i)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "fitness": FeatureDescriptor(
        name="fitness",
        category="agent",
        description="Alias for current_fitness",
        formula="f(x_i)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "dist_group_barycentre": FeatureDescriptor(
        name="dist_group_barycentre",
        category="agent",
        description="Normalized Hamming / L1 distance to group barycentre (centroid) in [0, 1]",
        formula="(1 / N) * sum_j |x_i[j] - x_group_mean[j]|",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "dist_others_barycentre": FeatureDescriptor(
        name="dist_others_barycentre",
        category="agent",
        description="Normalized Hamming distance to barycentre of others (mean peer distance / N) in [0, 1]",
        formula="(1 / N) * sum_j |x_i[j] - x_peers_mean[j]|",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "peer_distances": FeatureDescriptor(
        name="peer_distances",
        category="agent",
        description="(M - 1) normalized Hamming distances to each peer (ordered by peer fitness)",
        formula="dist(x_i, x_peer_k) / N",
        is_multi_agent=True,
        dim_fn=lambda m: max(m - 1, 0),
    ),
    "delta_mean_score": FeatureDescriptor(
        name="delta_mean_score",
        category="agent",
        description="Fitness gap between current agent and group mean fitness",
        formula="f(x_i) - mean(f_group)",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    "peer_scores": FeatureDescriptor(
        name="peer_scores",
        category="agent",
        description="(M - 1) fitness scores of each peer (ordered by peer fitness)",
        formula="f(x_peer_k)",
        is_multi_agent=True,
        dim_fn=lambda m: max(m - 1, 0),
    ),
    "peer_delta_scores": FeatureDescriptor(
        name="peer_delta_scores",
        category="agent",
        description="(M - 1) delta fitness scores relative to each peer: f(self) - f(peer_k)",
        formula="f(x_i) - f(x_peer_k)",
        is_multi_agent=True,
        dim_fn=lambda m: max(m - 1, 0),
    ),
    "normalized_rank": FeatureDescriptor(
        name="normalized_rank",
        category="agent",
        description="Normalized fitness rank in group in [0.0, 1.0] (1.0 = best agent)",
        formula="rank / (M - 1)",
        is_multi_agent=True,
        dim_fn=lambda m: 1 if m > 1 else 0,
    ),
    # --- Context / Trajectory Features ---
    "fitness_gap_to_best": FeatureDescriptor(
        name="fitness_gap_to_best",
        category="context",
        description="Fitness gap between historical best so far and current fitness",
        formula="f_best - f(x_i)",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "step_progress": FeatureDescriptor(
        name="step_progress",
        category="context",
        description="Fraction of search budget elapsed: step / max_steps",
        formula="t / T",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
    "stagnation_ratio": FeatureDescriptor(
        name="stagnation_ratio",
        category="context",
        description="Normalized stagnation counter: steps_without_improvement / max_steps",
        formula="stagnation / T",
        is_multi_agent=False,
        dim_fn=lambda m: 1,
    ),
}

# Synonyms and Canonical Mapping
FEATURE_CANONICAL: dict[str, str] = {
    "delta_fitness": "delta",
    "current_bit": "current_locus",
    "fitness": "current_fitness",
    "rank": "normalized_rank",
    "group_rank": "normalized_rank",
    "barycentre_group": "dist_group_barycentre",
    "barycentre_others": "dist_others_barycentre",
    "mean_peer_dist": "dist_others_barycentre",
    "peer_deltas_fitness_ordered": "peer_deltas",
}

# Group Keywords
FEATURE_GROUPS: dict[str, list[str]] = {
    "basic": ["delta", "ranked_delta"],
    "basic_action": ["delta", "ranked_delta"],
    "action_basic": ["delta", "ranked_delta"],
    "group_action": ["peer_deltas", "net_peer_delta", "group_mean_locus", "best_locus"],
    "action_all": [
        "delta",
        "ranked_delta",
        "peer_deltas",
        "net_peer_delta",
        "group_mean_locus",
        "peers_mean_locus",
        "best_locus",
        "current_locus",
    ],
    "scores": ["current_fitness", "delta_mean_score", "peer_delta_scores", "normalized_rank"],
    "distances": ["dist_group_barycentre", "dist_others_barycentre", "peer_distances"],
    "group_agent": [
        "current_fitness",
        "dist_group_barycentre",
        "dist_others_barycentre",
        "peer_distances",
        "delta_mean_score",
        "peer_delta_scores",
        "normalized_rank",
    ],
    "group": [
        "peer_deltas",
        "net_peer_delta",
        "group_mean_locus",
        "best_locus",
        "current_fitness",
        "dist_group_barycentre",
        "dist_others_barycentre",
        "peer_distances",
        "delta_mean_score",
        "peer_delta_scores",
        "normalized_rank",
    ],
    "group_all": [
        "peer_deltas",
        "net_peer_delta",
        "group_mean_locus",
        "peers_mean_locus",
        "best_locus",
        "current_fitness",
        "dist_group_barycentre",
        "dist_others_barycentre",
        "peer_distances",
        "delta_mean_score",
        "peer_scores",
        "peer_delta_scores",
        "normalized_rank",
    ],
    "context": ["fitness_gap_to_best", "step_progress", "stagnation_ratio"],
    "minimal": ["delta"],
    "default": [
        "delta",
        "ranked_delta",
        "peer_deltas",
        "best_locus",
        "dist_others_barycentre",
        "normalized_rank",
    ],
    "all": [
        "delta",
        "ranked_delta",
        "peer_deltas",
        "net_peer_delta",
        "group_mean_locus",
        "peers_mean_locus",
        "best_locus",
        "current_locus",
        "current_fitness",
        "dist_group_barycentre",
        "dist_others_barycentre",
        "peer_distances",
        "delta_mean_score",
        "peer_scores",
        "peer_delta_scores",
        "normalized_rank",
        "fitness_gap_to_best",
        "step_progress",
        "stagnation_ratio",
    ],
}


def resolve_feature_names(
    spec: str | list[str] | None,
    mode: EvaluationMode | str = EvaluationMode.PER_ACTION,
) -> list[str]:
    """Parse a flexible feature specification string or list into canonical feature names.

    Supports:
    - Group keywords prefixed with '@' (e.g. '@action_basic', '@scores', '@all')
    - Comma-separated names: "delta,ranked_delta,best_locus"
    - Exclusions with '-': "@all,-current_locus,-peer_scores"
    - Automatic canonical alias resolution.

    Args:
        spec: Feature string or list of tokens.
        mode: Target evaluation mode ('per_action' or 'all_actions').

    Returns:
        List of unique canonical feature names in deterministic order.
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return list(FEATURE_GROUPS["default"])

    if isinstance(spec, str):
        # Split by comma or semicolon
        tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in spec if str(t).strip()]

    included: list[str] = []
    excluded: set[str] = set()

    for token in tokens:
        token_clean = token.lower()
        is_exclude = False
        if token_clean.startswith("-") or token_clean.startswith("no-"):
            is_exclude = True
            token_clean = token_clean.lstrip("-").removeprefix("no-")

        # Strip optional @ prefix
        key = token_clean.lstrip("@")

        # Check if group keyword
        if key in FEATURE_GROUPS:
            items = FEATURE_GROUPS[key]
            if is_exclude:
                excluded.update(items)
            else:
                for item in items:
                    if item not in included:
                        included.append(item)
        else:
            # Map canonical name
            canonical = FEATURE_CANONICAL.get(key, key)
            if canonical not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown feature or keyword '{token}'. "
                    f"Available features: {list(FEATURE_REGISTRY.keys())}, "
                    f"Available groups: {['@' + g for g in FEATURE_GROUPS.keys()]}"
                )
            if is_exclude:
                excluded.add(canonical)
            else:
                if canonical not in included:
                    included.append(canonical)

    final_features = [f for f in included if f not in excluded]
    if not final_features:
        final_features = ["delta"]

    return final_features


# ============================================================================
# Feature Configuration Dataclass
# ============================================================================


@dataclass(frozen=True)
class FeatureConfig:
    """Configures which feature channels are extracted for neural policies.

    Attributes:
        features: Resolved list of canonical feature names.
        mode: Evaluation mode ('per_action' or 'all_actions').
        ablation_level: Ablation tier for backward compatibility.
    """

    features: list[str] = field(default_factory=lambda: list(FEATURE_GROUPS["default"]))
    mode: EvaluationMode | str = EvaluationMode.PER_ACTION
    ablation_level: FeatureAblationLevel | str = FeatureAblationLevel.FULL

    @classmethod
    def from_specs(
        cls,
        spec: str | list[str] | None = None,
        mode: EvaluationMode | str = EvaluationMode.PER_ACTION,
    ) -> FeatureConfig:
        """Create FeatureConfig by parsing a user-defined feature string or keyword list."""
        resolved = resolve_feature_names(spec, mode=mode)
        return cls(features=resolved, mode=EvaluationMode(mode))

    @classmethod
    def from_ablation_level(
        cls,
        level: FeatureAblationLevel | str,
        n_agents: int = 1,
        mode: EvaluationMode | str = EvaluationMode.PER_ACTION,
    ) -> FeatureConfig:
        """Factory creating standardized ablation configurations for backward compatibility."""
        lvl = FeatureAblationLevel(level)
        if lvl == FeatureAblationLevel.DELTA_ONLY:
            feats = ["delta"]
        elif lvl == FeatureAblationLevel.RANKED_DELTA_ONLY:
            feats = ["ranked_delta"]
        elif lvl == FeatureAblationLevel.DELTA_AND_PEERS:
            feats = ["delta", "peer_deltas", "net_peer_delta"] if n_agents > 1 else ["delta"]
        elif lvl == FeatureAblationLevel.DELTA_PEERS_RANK:
            feats = (
                ["delta", "ranked_delta", "peer_deltas", "net_peer_delta", "normalized_rank"]
                if n_agents > 1
                else ["delta", "ranked_delta"]
            )
        else:  # FULL
            feats = list(FEATURE_GROUPS["default"])

        return cls(features=feats, mode=EvaluationMode(mode), ablation_level=lvl)

    def get_feature_names(self, n_agents: int = 1) -> list[str]:
        """Ordered list of human-readable feature channel names given agent count."""
        names: list[str] = []
        for feat in self.features:
            desc = FEATURE_REGISTRY.get(feat)
            if desc is None:
                continue
            dim = desc.get_dimension(n_agents)
            if dim == 0:
                continue
            elif dim == 1:
                names.append(feat)
            else:
                for k in range(dim):
                    names.append(f"{feat}_{k}")
        return names if names else ["delta"]

    def get_num_features(self, n_agents: int = 1) -> int:
        """Total number of scalar feature values per action given agent count."""
        return len(self.get_feature_names(n_agents))

    # Compatibility properties with previous FeatureConfig
    @property
    def feature_names(self) -> list[str]:
        return self.get_feature_names(n_agents=1)

    @property
    def num_features(self) -> int:
        return self.get_num_features(n_agents=1)

    @property
    def use_deltas(self) -> bool:
        return "delta" in self.features or "delta_fitness" in self.features

    @property
    def use_ranked_deltas(self) -> bool:
        return "ranked_delta" in self.features

    @property
    def use_action_peer_deltas(self) -> bool:
        return "peer_deltas" in self.features

    @property
    def use_net_peer_delta(self) -> bool:
        return "net_peer_delta" in self.features

    @property
    def use_group_scores(self) -> bool:
        return (
            "normalized_rank" in self.features
            or "delta_mean_score" in self.features
            or "peer_delta_scores" in self.features
        )

    @property
    def use_group_distances(self) -> bool:
        return (
            "dist_group_barycentre" in self.features
            or "dist_others_barycentre" in self.features
            or "peer_distances" in self.features
        )

    @property
    def use_trajectory_context(self) -> bool:
        return (
            "fitness_gap_to_best" in self.features
            or "step_progress" in self.features
            or "stagnation_ratio" in self.features
        )


# ============================================================================
# Feature Extractor Implementation
# ============================================================================


class FeatureExtractor:
    """Extracts candidate move feature tensors for neural network consumption."""

    def __init__(self, config: FeatureConfig | None = None, n_agents: int = 1) -> None:
        self.config = config or FeatureConfig()
        self.n_agents = n_agents

    def extract(
        self,
        states: torch.Tensor,
        deltas: torch.Tensor,
        current_fitness: torch.Tensor,
        best_fitness: torch.Tensor,
        group_ctx: GroupContext | None,
        group_metrics: GroupMetrics | None,
        step: int,
        max_steps: int,
        stagnation_counter: torch.Tensor,
    ) -> torch.Tensor:
        """Extract a structured feature tensor of shape (B, N, num_features).

        Args:
            states: Current states tensor, shape (B, N).
            deltas: Candidate move deltas, shape (B, N).
            current_fitness: Objective values, shape (B,).
            best_fitness: Best-so-far objective values, shape (B,).
            group_ctx: Multi-agent group context, if available.
            group_metrics: Multi-agent group metrics, if available.
            step: Current search step index.
            max_steps: Maximum planned search steps.
            stagnation_counter: Steps without improvement per agent, shape (B,).

        Returns:
            Tensor of shape (B, N, F) where F = self.config.get_num_features(self.n_agents).
        """
        B, N = deltas.shape
        device = deltas.device
        M = self.n_agents
        is_multi = M > 1 and group_ctx is not None and group_metrics is not None

        feature_tensors: list[torch.Tensor] = []

        # Cache precomputations
        cached_ranks: torch.Tensor | None = None
        cached_peer_deltas: torch.Tensor | None = None
        cached_net_peer_delta: torch.Tensor | None = None

        for feat in self.config.features:
            desc = FEATURE_REGISTRY.get(feat)
            if desc is None:
                continue

            dim = desc.get_dimension(M)
            if dim == 0:
                continue

            # 1. Action-level features: shape (B, N, dim)
            if feat in ("delta", "delta_fitness"):
                feature_tensors.append(deltas.unsqueeze(-1).float())

            elif feat == "ranked_delta":
                if cached_ranks is None:
                    cached_ranks = compute_normalized_ranks(deltas).unsqueeze(-1)
                feature_tensors.append(cached_ranks)

            elif feat in ("peer_deltas", "net_peer_delta"):
                if cached_peer_deltas is None and is_multi:
                    cached_peer_deltas, cached_net_peer_delta = compute_action_peer_deltas(
                        states=states,
                        group_ctx=group_ctx,
                        sort_peers=True,
                        group_metrics=group_metrics,
                        current_fitness=current_fitness,
                    )
                if feat == "peer_deltas":
                    if is_multi and cached_peer_deltas is not None:
                        feature_tensors.append(cached_peer_deltas)
                elif feat == "net_peer_delta":
                    if is_multi and cached_net_peer_delta is not None:
                        feature_tensors.append(cached_net_peer_delta)

            elif feat == "group_mean_locus":
                if is_multi and group_metrics is not None:
                    # group_barycentre: (B, N) -> (B, N, 1)
                    feature_tensors.append(group_metrics.group_barycentre.unsqueeze(-1))

            elif feat == "peers_mean_locus":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(group_metrics.others_barycentre.unsqueeze(-1))

            elif feat == "best_locus":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.current_best_agent_states.float().unsqueeze(-1)
                    )
                else:
                    feature_tensors.append(states.float().unsqueeze(-1))

            elif feat in ("current_locus", "current_bit"):
                feature_tensors.append(states.float().unsqueeze(-1))

            # 2. Agent / Group-level features: broadcast to shape (B, N, dim)
            elif feat in ("current_fitness", "fitness"):
                feature_tensors.append(
                    current_fitness.unsqueeze(1).unsqueeze(-1).expand(B, N, 1).float()
                )

            elif feat == "dist_group_barycentre":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.dist_group_barycentre.unsqueeze(1)
                        .unsqueeze(-1)
                        .expand(B, N, 1)
                        .float()
                    )

            elif feat == "dist_others_barycentre":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.dist_others_barycentre.unsqueeze(1)
                        .unsqueeze(-1)
                        .expand(B, N, 1)
                        .float()
                    )

            elif feat == "peer_distances":
                if is_multi and group_metrics is not None:
                    # ordered_peer_distances is (B, M - 1) -> expand to (B, N, M - 1)
                    feature_tensors.append(
                        group_metrics.ordered_peer_distances.unsqueeze(1)
                        .expand(B, N, M - 1)
                        .float()
                    )

            elif feat == "delta_mean_score":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.delta_mean_score.unsqueeze(1)
                        .unsqueeze(-1)
                        .expand(B, N, 1)
                        .float()
                    )

            elif feat == "peer_scores":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.ordered_peer_scores.unsqueeze(1).expand(B, N, M - 1).float()
                    )

            elif feat == "peer_delta_scores":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.ordered_peer_delta_scores.unsqueeze(1)
                        .expand(B, N, M - 1)
                        .float()
                    )

            elif feat == "normalized_rank":
                if is_multi and group_metrics is not None:
                    feature_tensors.append(
                        group_metrics.ranks.unsqueeze(1).unsqueeze(-1).expand(B, N, 1).float()
                    )

            # 3. Contextual features: broadcast to shape (B, N, 1)
            elif feat == "fitness_gap_to_best":
                fit_gap = (
                    (best_fitness - current_fitness).unsqueeze(1).unsqueeze(-1).expand(B, N, 1)
                )
                feature_tensors.append(fit_gap.float())

            elif feat == "step_progress":
                prog = torch.full(
                    (B, N, 1), step / max(max_steps, 1), device=device, dtype=torch.float32
                )
                feature_tensors.append(prog)

            elif feat == "stagnation_ratio":
                stag = (
                    (stagnation_counter.float() / max(max_steps, 1))
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(B, N, 1)
                )
                feature_tensors.append(stag)

        if not feature_tensors:
            feature_tensors.append(deltas.unsqueeze(-1).float())

        return torch.cat(feature_tensors, dim=-1)
