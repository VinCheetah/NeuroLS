"""Multi-agent coordination and group context for Local Search.

Provides structured information sharing across groups of agents exploring
the same problem landscape instance, including pairwise distances, group rankings,
ordered peer dynamics, barycentre statistics, and diversity tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neurols.types import DeviceType


@dataclass
class GroupConfig:
    """Configuration governing information sharing among agents within a group.

    Attributes:
        n_agents: Number of agents in each group (M >= 1). If 1, runs in single-agent mode.
        share_distances: Whether agents have access to peer distance metrics.
        share_distance_mean: Whether agents have access to the mean distance to peers.
        share_delta_scores: Whether agents have access to peer fitness scores and deltas to group best.
        share_delta_mean_score: Whether agents have access to the mean delta score of the group.
        share_ranks: Whether agents have access to their fitness ranking within the group.
        share_approachs: Whether agents know if action `i` reduces distance to peers.
        share_mean_variable: Whether agents know the mean variable / locus of the group.
        share_best_variable: Whether agents know the variable / locus of the best agent.
    """

    n_agents: int = 1
    share_distances: bool = True
    share_distance_mean: bool = True
    share_delta_scores: bool = True
    share_delta_mean_score: bool = True
    share_ranks: bool = True
    share_approachs: bool = True
    share_mean_variable: bool = True
    share_best_variable: bool = True

    # Backwards-compatible aliases
    share_distance: bool | None = None
    share_scores: bool | None = None
    share_consensus: bool | None = None
    share_group_best: bool | None = None

    def __post_init__(self) -> None:
        if self.share_distance is not None:
            object.__setattr__(self, "share_distances", self.share_distance)
            object.__setattr__(self, "share_distance_mean", self.share_distance)
        if self.share_scores is not None:
            object.__setattr__(self, "share_delta_scores", self.share_scores)
            object.__setattr__(self, "share_delta_mean_score", self.share_scores)
        if self.share_consensus is not None:
            object.__setattr__(self, "share_mean_variable", self.share_consensus)
        if self.share_group_best is not None:
            object.__setattr__(self, "share_best_variable", self.share_group_best)

    @property
    def is_multi_agent(self) -> bool:
        """True if more than 1 agent operates per group."""
        return self.n_agents > 1

    def __getattr__(self, name: str) -> Any:
        if name == "share_distance":
            return self.share_distances
        if name == "share_scores":
            return self.share_delta_scores
        if name == "share_consensus":
            return self.share_mean_variable
        if name == "share_group_best":
            return self.share_best_variable
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class GroupContext:
    """Manages multi-agent group dynamics, metrics, and shared state across a batch.

    The batch is assumed to consist of G groups, each containing M agents, for a total
    batch size of B = G * M. Agents within the same group explore the same problem
    instance and restart, and can share metrics according to the GroupConfig.
    """

    def __init__(
        self,
        config: GroupConfig,
        n_instances: int,
        n_restarts: int,
        device: DeviceType = "cpu",
    ) -> None:
        self.config = config
        self.n_instances = n_instances
        self.n_restarts = n_restarts
        self.n_groups = n_instances * n_restarts
        self.n_agents = config.n_agents
        self.total_batch_size = self.n_groups * self.n_agents
        self.device = torch.device(device)

        # Preallocate group index mappings
        self.group_ids = torch.repeat_interleave(
            torch.arange(self.n_groups, device=self.device), self.n_agents
        )
        self.agent_in_group_ids = torch.arange(self.n_agents, device=self.device).repeat(
            self.n_groups
        )

    def update(
        self,
        states: torch.Tensor,
        current_fitness: torch.Tensor,
        best_fitness: torch.Tensor,
        best_states: torch.Tensor,
    ) -> GroupMetrics:
        """Compute all group-level metrics and shared information for the current step.

        Args:
            states: Current states tensor, shape (B, dimension).
            current_fitness: Current fitness tensor, shape (B,).
            best_fitness: Best fitness achieved so far by each agent, shape (B,).
            best_states: Best states achieved so far by each agent, shape (B, dimension).

        Returns:
            GroupMetrics containing extracted relational features and diversity measures.
        """
        B, D = states.shape
        G = self.n_groups
        M = self.n_agents

        if M == 1 or not self.config.is_multi_agent:
            zeros_b = torch.zeros(B, device=self.device, dtype=torch.float32)
            zeros_bm = torch.zeros((B, 0), device=self.device, dtype=torch.float32)
            zeros_idx = torch.zeros((B, 0), device=self.device, dtype=torch.int64)
            return GroupMetrics(
                pairwise_distances=torch.zeros((B, 1), device=self.device),
                mean_peer_distance=zeros_b,
                min_peer_distance=zeros_b,
                group_diversity=0.0,
                group_best_fitness=best_fitness.clone(),
                group_best_states=best_states.clone(),
                fitness_diff_to_group_best=zeros_b,
                ranks=torch.ones(B, device=self.device, dtype=torch.float32),
                group_mean_fitness=current_fitness.clone(),
                delta_mean_score=zeros_b,
                group_barycentre=states.float(),
                dist_group_barycentre=zeros_b,
                others_barycentre=states.float(),
                dist_others_barycentre=zeros_b,
                ordered_peer_indices=zeros_idx,
                ordered_peer_distances=zeros_bm,
                ordered_peer_scores=zeros_bm,
                ordered_peer_delta_scores=zeros_bm,
                current_best_agent_states=states.clone(),
            )

        # Reshape to (G, M, D) and (G, M)
        states_grouped = states.view(G, M, D)
        fitness_grouped = current_fitness.view(G, M)
        best_fitness_grouped = best_fitness.view(G, M)
        best_states_grouped = best_states.view(G, M, D)

        # 1. Pairwise Hamming Distances within each group: shape (G, M, M)
        diffs = states_grouped.unsqueeze(2) != states_grouped.unsqueeze(1)  # (G, M, M, D)
        pairwise_dist = diffs.sum(dim=-1).float()  # (G, M, M)
        pairwise_dist_flat = pairwise_dist.view(B, M)

        # Mean peer distance (excluding self distance of 0)
        sum_dist = pairwise_dist.sum(dim=-1)  # (G, M)
        mean_peer_dist = (sum_dist / max(M - 1, 1)).view(B)

        # Min peer distance (excluding self distance)
        eye = torch.eye(M, device=self.device, dtype=torch.bool).unsqueeze(0)
        dist_no_self = pairwise_dist.masked_fill(eye, float("inf"))
        min_peer_dist, _ = dist_no_self.min(dim=-1)
        min_peer_dist = min_peer_dist.view(B)

        # Group diversity: average pairwise distance over all distinct pairs
        distinct_sum = pairwise_dist.sum(dim=(-1, -2)) / 2.0  # (G,)
        n_pairs = (M * (M - 1)) / 2.0
        group_diversity_val = (distinct_sum / n_pairs).mean().item()

        # 2. Historical Group Best Fitness & States
        max_best_fit, best_agent_idx = best_fitness_grouped.max(dim=1)  # (G,)
        group_best_fit = torch.repeat_interleave(max_best_fit, M)  # (B,)
        gather_idx = best_agent_idx.view(G, 1, 1).expand(G, 1, D)
        group_best_states_per_group = best_states_grouped.gather(1, gather_idx).squeeze(1)
        group_best_states = torch.repeat_interleave(group_best_states_per_group, M, dim=0)
        fitness_diff = group_best_fit - current_fitness

        # 3. Current Best Agent in Group
        max_curr_fit, curr_best_agent_idx = fitness_grouped.max(dim=1)
        curr_gather_idx = curr_best_agent_idx.view(G, 1, 1).expand(G, 1, D)
        curr_best_states_per_group = states_grouped.gather(1, curr_gather_idx).squeeze(1)
        current_best_agent_states = torch.repeat_interleave(curr_best_states_per_group, M, dim=0)

        # 4. Ranks within group (normalized [0, 1], where 1.0 is highest fitness)
        sorted_indices = torch.argsort(fitness_grouped, dim=1)
        ranks_in_group = torch.empty_like(fitness_grouped, dtype=torch.float32)
        ranks_in_group.scatter_(
            1, sorted_indices, torch.arange(M, device=self.device, dtype=torch.float32).expand(G, M)
        )
        ranks_normalized = (ranks_in_group / (M - 1)).view(B)

        # 5. Group Mean Fitness & Delta score to average
        group_mean_fit = fitness_grouped.mean(dim=1, keepdim=True).expand(G, M).reshape(B)
        delta_mean_score = current_fitness - group_mean_fit

        # 6. Group Barycentre and Others' Barycentre
        states_float = states_grouped.float()
        group_barycentre_g = states_float.mean(dim=1)  # (G, D)
        group_barycentre = group_barycentre_g.unsqueeze(1).expand(G, M, D).reshape(B, D)

        # Hamming / L1 distance to group barycentre normalized by D
        dist_group_barycentre = (
            torch.abs(states_float - group_barycentre_g.unsqueeze(1)).mean(dim=-1).view(B)
        )

        # Barycentre of others (excluding self)
        sum_states = states_float.sum(dim=1, keepdim=True)  # (G, 1, D)
        others_barycentre_g = (sum_states - states_float) / (M - 1)  # (G, M, D)
        others_barycentre = others_barycentre_g.reshape(B, D)

        # Distance to others' barycentre normalized by D
        dist_others_barycentre = torch.abs(states_float - others_barycentre_g).mean(dim=-1).view(B)

        # 7. Ordered Peers by Fitness (Highest fitness peer first)
        # Mask out self by setting diagonal to -infinity
        peer_fit_matrix = fitness_grouped.unsqueeze(1).expand(G, M, M).clone()
        peer_fit_masked = peer_fit_matrix.masked_fill(eye, -float("inf"))
        sorted_peer_fit, sorted_peer_idx = peer_fit_masked.sort(dim=-1, descending=True)
        ordered_peer_indices = sorted_peer_idx[:, :, : M - 1]  # (G, M, M - 1)
        ordered_peer_scores = sorted_peer_fit[:, :, : M - 1]  # (G, M, M - 1)

        # Gather peer states ordered by fitness
        ordered_gather_idx = ordered_peer_indices.unsqueeze(-1).expand(G, M, M - 1, D)
        ordered_peer_states = torch.gather(
            states_grouped.unsqueeze(1).expand(G, M, M, D), dim=2, index=ordered_gather_idx
        )  # (G, M, M - 1, D)

        # Ordered peer distances normalized by D
        ordered_diff = states_grouped.unsqueeze(2) != ordered_peer_states
        ordered_peer_distances = ordered_diff.sum(dim=-1).float() / float(D)  # (G, M, M - 1)

        # Ordered peer delta scores: f(self) - f(peer)
        self_fit_expanded = fitness_grouped.unsqueeze(2).expand(G, M, M - 1)
        ordered_peer_delta_scores = self_fit_expanded - ordered_peer_scores  # (G, M, M - 1)

        return GroupMetrics(
            pairwise_distances=pairwise_dist_flat,
            mean_peer_distance=mean_peer_dist,
            min_peer_distance=min_peer_dist,
            group_diversity=group_diversity_val,
            group_best_fitness=group_best_fit,
            group_best_states=group_best_states,
            fitness_diff_to_group_best=fitness_diff,
            ranks=ranks_normalized,
            group_mean_fitness=group_mean_fit,
            delta_mean_score=delta_mean_score,
            group_barycentre=group_barycentre,
            dist_group_barycentre=dist_group_barycentre,
            others_barycentre=others_barycentre,
            dist_others_barycentre=dist_others_barycentre,
            ordered_peer_indices=ordered_peer_indices.reshape(B, M - 1),
            ordered_peer_distances=ordered_peer_distances.reshape(B, M - 1),
            ordered_peer_scores=ordered_peer_scores.reshape(B, M - 1),
            ordered_peer_delta_scores=ordered_peer_delta_scores.reshape(B, M - 1),
            current_best_agent_states=current_best_agent_states,
        )


@dataclass(frozen=True)
class GroupMetrics:
    """Computed multi-agent metrics at a given local search step.

    Attributes:
        pairwise_distances: Tensor of shape (B, M) with Hamming distances to group members.
        mean_peer_distance: Average distance to peers per agent, shape (B,).
        min_peer_distance: Distance to the closest peer per agent, shape (B,).
        group_diversity: Scalar mean diversity (average pairwise Hamming distance) across groups.
        group_best_fitness: Highest fitness found so far within the agent's group, shape (B,).
        group_best_states: Best state found so far within the agent's group, shape (B, D).
        fitness_diff_to_group_best: Difference between group best fitness and current fitness, shape (B,).
        ranks: Normalized fitness rank within group in [0.0, 1.0], shape (B,).
        group_mean_fitness: Average fitness of agents in the group, shape (B,).
        delta_mean_score: Current fitness minus group mean fitness, shape (B,).
        group_barycentre: Centroid of all agents in the group, shape (B, D).
        dist_group_barycentre: Normalized Hamming distance to group barycentre, shape (B,).
        others_barycentre: Centroid of other peers in the group, shape (B, D).
        dist_others_barycentre: Normalized Hamming distance to others' barycentre, shape (B,).
        ordered_peer_indices: Indices of peers sorted by descending fitness, shape (B, M - 1).
        ordered_peer_distances: Normalized Hamming distance to peers sorted by fitness, shape (B, M - 1).
        ordered_peer_scores: Fitness scores of peers sorted by descending fitness, shape (B, M - 1).
        ordered_peer_delta_scores: Score delta (self - peer) sorted by peer fitness, shape (B, M - 1).
        current_best_agent_states: State of the best-performing agent in current step, shape (B, D).
    """

    pairwise_distances: torch.Tensor
    mean_peer_distance: torch.Tensor
    min_peer_distance: torch.Tensor
    group_diversity: float
    group_best_fitness: torch.Tensor
    group_best_states: torch.Tensor
    fitness_diff_to_group_best: torch.Tensor
    ranks: torch.Tensor
    group_mean_fitness: torch.Tensor
    delta_mean_score: torch.Tensor
    group_barycentre: torch.Tensor
    dist_group_barycentre: torch.Tensor
    others_barycentre: torch.Tensor
    dist_others_barycentre: torch.Tensor
    ordered_peer_indices: torch.Tensor
    ordered_peer_distances: torch.Tensor
    ordered_peer_scores: torch.Tensor
    ordered_peer_delta_scores: torch.Tensor
    current_best_agent_states: torch.Tensor

    def extract_agent_features(self, config: GroupConfig) -> torch.Tensor:
        """Extract a structured feature tensor per agent based on active sharing permissions.

        Returns:
            Tensor of shape (B, num_active_features).
        """
        features: list[torch.Tensor] = []
        B = self.mean_peer_distance.shape[0]

        if config.share_ranks:
            features.append(self.ranks.unsqueeze(1))
        if config.share_delta_scores:
            features.append(self.fitness_diff_to_group_best.unsqueeze(1))
        if config.share_distances:
            features.append(self.mean_peer_distance.unsqueeze(1))
            features.append(self.min_peer_distance.unsqueeze(1))

        if not features:
            return torch.zeros((B, 1), device=self.mean_peer_distance.device)

        return torch.cat(features, dim=-1)
