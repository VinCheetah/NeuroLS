"""Feature extraction and ablation pipeline for Neural Local Search strategies.

Translates problem states, candidate move deltas, multi-agent group metrics, and
action-specific peer distance transitions into structured tensor representations,
with explicit support for permutation-invariance (via sorting or pooling).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from neurols.core.group import GroupContext, GroupMetrics


class FeatureAblationLevel(StrEnum):
    """Standardized feature ablation tiers to test which signals drive performance."""

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
        Tensor of shape (B, N) containing normalized ranks.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute action-specific peer distance changes for each candidate 1-bit flip.

    For each candidate bit flip j in {0, ..., N-1}:
    - Flipping bit j increases distance to peer m by +1 if x_m[j] == x_self[j].
    - Flipping bit j decreases distance to peer m by -1 if x_m[j] != x_self[j].

    To ensure permutation invariance across the (M - 1) peers:
    - The peer deltas are sorted in ascending order (sort_peers=True).
    - Summary metrics (mean peer delta) are computed.

    Args:
        states: Current states tensor, shape (B, N).
        group_ctx: Multi-agent group context.
        sort_peers: Whether to sort peer deltas to guarantee permutation invariance.

    Returns:
        peer_deltas: Tensor of shape (B, N, max(M-1, 1)) in {-1, +1}.
        net_peer_delta: Tensor of shape (B, N, 1) in [-1, +1] (mean change across peers).
    """
    B, N = states.shape
    device = states.device

    if group_ctx is None or group_ctx.n_agents <= 1:
        # Solo agent: trivial zero change
        zeros = torch.zeros((B, N, 1), device=device, dtype=torch.float32)
        return zeros, zeros

    G = group_ctx.n_groups
    M = group_ctx.n_agents
    D = N

    # Reshape states to (G, M, D)
    states_grouped = states.view(G, M, D)

    # For each agent a, compute difference with all peers m:
    # diff: (G, M, M, D) where diff[g, a, m, j] is True if state[g, a, j] != state[g, m, j]
    diff = states_grouped.unsqueeze(2) != states_grouped.unsqueeze(1)  # (G, M, M, D)

    # If diff is True, flipping bit j makes them equal -> delta distance is -1.
    # If diff is False (equal), flipping bit j makes them differ -> delta distance is +1.
    # delta_d = -1 if diff else +1  =>  1 - 2 * diff
    delta_dist_all = torch.where(diff, -1.0, 1.0)  # (G, M, M, D)

    # Extract non-self peers: for each agent a, exclude m = a
    # Create mask of peers: (M, M) with False on diagonal
    peer_mask = ~torch.eye(M, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(
        -1
    )  # (1, M, M, 1)
    peer_mask = peer_mask.expand(G, M, M, D)

    # Gather peer deltas: shape (G, M, M - 1, D)
    # Transpose D and peers to (G, M, D, M - 1)
    delta_dist_flat = delta_dist_all.masked_select(peer_mask).view(G, M, M - 1, D)
    delta_dist_perm = delta_dist_flat.permute(0, 1, 3, 2).reshape(B, N, M - 1)  # (B, N, M - 1)

    if sort_peers and (M - 1) > 1:
        # Sort across peer dimension to enforce strict permutation invariance
        sorted_deltas, _ = delta_dist_perm.sort(dim=-1, descending=False)
        peer_deltas = sorted_deltas
    else:
        peer_deltas = delta_dist_perm

    # Net peer delta: mean change across peers in [-1.0, 1.0]
    net_peer_delta = delta_dist_perm.mean(dim=-1, keepdim=True)  # (B, N, 1)

    return peer_deltas, net_peer_delta


@dataclass(frozen=True)
class FeatureConfig:
    """Configures which feature channels are extracted for neural policies.

    Attributes:
        ablation_level: Predefined ablation tier (DELTA_ONLY, RANKED_DELTA_ONLY, etc.).
        use_deltas: Include raw fitness change delta_f(x, j) per candidate move.
        use_ranked_deltas: Include dual-pole normalized rank of delta_f(x, j).
        use_action_peer_deltas: Include (+/-1) distance changes to each peer.
        use_net_peer_delta: Include mean distance change across peers.
        use_current_bit: Include current bit value x_j.
        use_group_scores: Include agent rank and fitness gap to group best.
        use_group_distances: Include mean and min peer Hamming distances.
        use_trajectory_context: Include step fraction, fitness gap, stagnation counter.
        sort_peer_deltas: Sort peer deltas for strict permutation invariance.
    """

    ablation_level: FeatureAblationLevel | str = FeatureAblationLevel.FULL
    use_deltas: bool = True
    use_ranked_deltas: bool = False
    use_action_peer_deltas: bool = True
    use_net_peer_delta: bool = True
    use_current_bit: bool = True
    use_group_scores: bool = True
    use_group_distances: bool = True
    use_trajectory_context: bool = True
    sort_peer_deltas: bool = True

    @classmethod
    def from_ablation_level(
        cls,
        level: FeatureAblationLevel | str,
        n_agents: int = 1,
    ) -> FeatureConfig:
        """Factory creating standardized ablation configurations."""
        lvl = FeatureAblationLevel(level)
        if lvl == FeatureAblationLevel.DELTA_ONLY:
            return cls(
                ablation_level=lvl,
                use_deltas=True,
                use_ranked_deltas=False,
                use_action_peer_deltas=False,
                use_net_peer_delta=False,
                use_current_bit=False,
                use_group_scores=False,
                use_group_distances=False,
                use_trajectory_context=False,
            )
        elif lvl == FeatureAblationLevel.RANKED_DELTA_ONLY:
            return cls(
                ablation_level=lvl,
                use_deltas=False,
                use_ranked_deltas=True,
                use_action_peer_deltas=False,
                use_net_peer_delta=False,
                use_current_bit=False,
                use_group_scores=False,
                use_group_distances=False,
                use_trajectory_context=False,
            )
        elif lvl == FeatureAblationLevel.DELTA_AND_PEERS:
            return cls(
                ablation_level=lvl,
                use_deltas=True,
                use_ranked_deltas=False,
                use_action_peer_deltas=n_agents > 1,
                use_net_peer_delta=n_agents > 1,
                use_current_bit=False,
                use_group_scores=False,
                use_group_distances=False,
                use_trajectory_context=False,
            )
        elif lvl == FeatureAblationLevel.DELTA_PEERS_RANK:
            return cls(
                ablation_level=lvl,
                use_deltas=True,
                use_ranked_deltas=True,
                use_action_peer_deltas=n_agents > 1,
                use_net_peer_delta=n_agents > 1,
                use_current_bit=False,
                use_group_scores=n_agents > 1,
                use_group_distances=n_agents > 1,
                use_trajectory_context=False,
            )
        else:  # FULL
            return cls(
                ablation_level=lvl,
                use_deltas=True,
                use_ranked_deltas=True,
                use_action_peer_deltas=n_agents > 1,
                use_net_peer_delta=n_agents > 1,
                use_current_bit=True,
                use_group_scores=True,
                use_group_distances=True,
                use_trajectory_context=True,
            )

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of human-readable feature channel names (for single-agent default)."""
        return self.get_feature_names(n_agents=1)

    @property
    def num_features(self) -> int:
        """Total number of scalar features per candidate move (for single-agent default)."""
        return len(self.feature_names)

    def get_feature_names(self, n_agents: int = 1) -> list[str]:
        """Ordered list of human-readable feature channel names."""
        names: list[str] = []
        if self.use_deltas:
            names.append("delta_fitness")
        if self.use_ranked_deltas:
            names.append("ranked_delta")
        if self.use_action_peer_deltas and n_agents > 1:
            for p in range(n_agents - 1):
                names.append(f"peer_delta_{p}")
        if self.use_net_peer_delta and n_agents > 1:
            names.append("net_peer_delta")
        if self.use_current_bit:
            names.append("current_bit")
        if self.use_group_scores:
            names.extend(["group_rank", "group_fitness_gap"])
        if self.use_group_distances:
            names.extend(["mean_peer_dist", "min_peer_dist"])
        if self.use_trajectory_context:
            names.extend(["fitness_gap_to_best", "step_progress", "stagnation_ratio"])
        return names

    def get_num_features(self, n_agents: int = 1) -> int:
        """Total number of scalar features per candidate move."""
        return len(self.get_feature_names(n_agents))


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
        feature_tensors: list[torch.Tensor] = []

        # 1. Raw Delta Fitness: shape (B, N, 1)
        if self.config.use_deltas:
            feature_tensors.append(deltas.unsqueeze(-1).float())

        # 2. Dual-pole Normalized Ranked Deltas: shape (B, N, 1)
        if self.config.use_ranked_deltas:
            ranked = compute_normalized_ranks(deltas).unsqueeze(-1)
            feature_tensors.append(ranked)

        # 3. Action-Specific Peer Distance Changes: (+/- 1 for each peer)
        if (
            self.config.use_action_peer_deltas or self.config.use_net_peer_delta
        ) and self.n_agents > 1:
            peer_deltas, net_peer_delta = compute_action_peer_deltas(
                states=states,
                group_ctx=group_ctx,
                sort_peers=self.config.sort_peer_deltas,
            )
            if self.config.use_action_peer_deltas:
                feature_tensors.append(peer_deltas)
            if self.config.use_net_peer_delta:
                feature_tensors.append(net_peer_delta)

        # 4. Current Bit value: shape (B, N, 1)
        if self.config.use_current_bit:
            feature_tensors.append(states.float().unsqueeze(-1))

        # 6. Group scores & ranks: shape (B, N, 1)
        if self.config.use_group_scores:
            if group_metrics is not None:
                rank_feat = group_metrics.ranks.unsqueeze(1).unsqueeze(-1).expand(B, N, 1)
                grp_fit_gap = (
                    group_metrics.fitness_diff_to_group_best.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(B, N, 1)
                )
            else:
                rank_feat = torch.ones((B, N, 1), device=device)
                grp_fit_gap = torch.zeros((B, N, 1), device=device)
            feature_tensors.extend([rank_feat, grp_fit_gap])

        # 7. Group distances: shape (B, N, 1)
        if self.config.use_group_distances:
            if group_metrics is not None:
                mean_dist = (
                    (group_metrics.mean_peer_distance / float(N))
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(B, N, 1)
                )
                min_dist = (
                    (group_metrics.min_peer_distance / float(N))
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(B, N, 1)
                )
            else:
                mean_dist = torch.zeros((B, N, 1), device=device)
                min_dist = torch.zeros((B, N, 1), device=device)
            feature_tensors.extend([mean_dist, min_dist])

        # 8. Trajectory context: shape (B, N, 1)
        if self.config.use_trajectory_context:
            fit_gap = (best_fitness - current_fitness).unsqueeze(1).unsqueeze(-1).expand(B, N, 1)
            step_prog = torch.full(
                (B, N, 1), step / max(max_steps, 1), device=device, dtype=torch.float32
            )
            stag_ratio = (
                (stagnation_counter.float() / max(max_steps, 1))
                .unsqueeze(1)
                .unsqueeze(-1)
                .expand(B, N, 1)
            )
            feature_tensors.extend([fit_gap, step_prog, stag_ratio])

        # If no features selected, fallback to deltas
        if not feature_tensors:
            feature_tensors.append(deltas.unsqueeze(-1).float())

        return torch.cat(feature_tensors, dim=-1)
