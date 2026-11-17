"""Multi-agent coordination and group context for Local Search.

Provides structured information sharing across groups of agents exploring
the same problem landscape instance, including pairwise distances, group rankings,
consensus statistics, and diversity tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurols.types import DeviceType


@dataclass(frozen=True)
class GroupConfig:
    """Configuration governing information sharing among agents within a group.

    Attributes:
        n_agents: Number of agents in each group (M >= 1). If 1, runs in single-agent mode.
        share_distance: Whether agents have access to peer distance metrics.
        share_scores: Whether agents have access to peer fitness scores and deltas to group best.
        share_ranks: Whether agents have access to their fitness ranking within the group.
        share_consensus: Whether agents have access to group-wide bit/variable consensus frequencies.
        share_group_best: Whether agents know the overall best state and fitness in their group.
    """

    n_agents: int = 1
    share_distance: bool = True
    share_scores: bool = True
    share_ranks: bool = True
    share_consensus: bool = True
    share_group_best: bool = True

    @property
    def is_multi_agent(self) -> bool:
        """True if more than 1 agent operates per group."""
        return self.n_agents > 1


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
        # group_ids: shape (B,) mapping each agent index to its group index in [0, n_groups-1]
        self.group_ids = torch.repeat_interleave(
            torch.arange(self.n_groups, device=self.device), self.n_agents
        )
        # agent_in_group_ids: shape (B,) mapping each agent to its rank in [0, M-1]
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
            # Single-agent fast path: trivial group metrics
            zeros_b = torch.zeros(B, device=self.device, dtype=torch.float32)
            return GroupMetrics(
                pairwise_distances=torch.zeros((B, 1), device=self.device),
                mean_peer_distance=zeros_b,
                min_peer_distance=zeros_b,
                group_diversity=0.0,
                group_best_fitness=best_fitness.clone(),
                group_best_states=best_states.clone(),
                fitness_diff_to_group_best=zeros_b,
                ranks=torch.ones(B, device=self.device, dtype=torch.float32),
            )

        # Reshape to (G, M, D) and (G, M)
        states_grouped = states.view(G, M, D)
        fitness_grouped = current_fitness.view(G, M)
        best_fitness_grouped = best_fitness.view(G, M)
        best_states_grouped = best_states.view(G, M, D)

        # 1. Pairwise Hamming Distances within each group: shape (G, M, M)
        # Using broadcasting: |x_i - x_j| summed over dimension
        diffs = states_grouped.unsqueeze(2) != states_grouped.unsqueeze(1)  # (G, M, M, D)
        pairwise_dist = diffs.sum(dim=-1).float()  # (G, M, M)

        # Flatten pairwise distance for each agent: (B, M)
        pairwise_dist_flat = pairwise_dist.view(B, M)

        # Mean peer distance (excluding self distance of 0)
        # sum of row minus self-dist (which is 0) divided by (M - 1)
        sum_dist = pairwise_dist.sum(dim=-1)  # (G, M)
        mean_peer_dist = (sum_dist / max(M - 1, 1)).view(B)

        # Min peer distance (excluding self distance: set diagonal to infinity first)
        eye = torch.eye(M, device=self.device, dtype=torch.bool).unsqueeze(0)
        dist_no_self = pairwise_dist.masked_fill(eye, float("inf"))
        min_peer_dist, _ = dist_no_self.min(dim=-1)
        min_peer_dist = min_peer_dist.view(B)

        # Group diversity: average pairwise distance over all distinct pairs in group
        if M > 1:
            # Number of distinct pairs = M * (M - 1) / 2
            distinct_sum = pairwise_dist.sum(dim=(-1, -2)) / 2.0  # (G,)
            n_pairs = (M * (M - 1)) / 2.0
            group_diversity_val = (distinct_sum / n_pairs).mean().item()
        else:
            group_diversity_val = 0.0

        # 2. Group Best Fitness & States
        max_best_fit, best_agent_idx = best_fitness_grouped.max(dim=1)  # (G,)
        group_best_fit = torch.repeat_interleave(max_best_fit, M)  # (B,)

        # Extract group best state: (G, D) then repeat
        gather_idx = best_agent_idx.view(G, 1, 1).expand(G, 1, D)
        group_best_states_per_group = best_states_grouped.gather(1, gather_idx).squeeze(1)  # (G, D)
        group_best_states = torch.repeat_interleave(group_best_states_per_group, M, dim=0)  # (B, D)

        # Fitness diff to group best: >= 0 (0 means this agent is at the group best)
        fitness_diff = group_best_fit - current_fitness

        # 3. Ranks within group (normalized [0, 1], where 1.0 is highest fitness)
        # argsort gives rank indices
        sorted_indices = torch.argsort(fitness_grouped, dim=1)
        ranks_in_group = torch.empty_like(fitness_grouped, dtype=torch.float32)
        ranks_in_group.scatter_(
            1, sorted_indices, torch.arange(M, device=self.device, dtype=torch.float32).expand(G, M)
        )
        if M > 1:
            ranks_normalized = (ranks_in_group / (M - 1)).view(B)
        else:
            ranks_normalized = torch.ones(B, device=self.device)


        return GroupMetrics(
            pairwise_distances=pairwise_dist_flat,
            mean_peer_distance=mean_peer_dist,
            min_peer_distance=min_peer_dist,
            group_diversity=group_diversity_val,
            group_best_fitness=group_best_fit,
            group_best_states=group_best_states,
            fitness_diff_to_group_best=fitness_diff,
            ranks=ranks_normalized,
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
    """

    pairwise_distances: torch.Tensor
    mean_peer_distance: torch.Tensor
    min_peer_distance: torch.Tensor
    group_diversity: float
    group_best_fitness: torch.Tensor
    group_best_states: torch.Tensor
    fitness_diff_to_group_best: torch.Tensor
    ranks: torch.Tensor

    def extract_agent_features(self, config: GroupConfig) -> torch.Tensor:
        """Extract a structured feature tensor per agent based on active sharing permissions.

        Returns:
            Tensor of shape (B, num_active_features).
        """
        features: list[torch.Tensor] = []
        B = self.mean_peer_distance.shape[0]

        if config.share_ranks:
            features.append(self.ranks.unsqueeze(1))
        if config.share_scores:
            features.append(self.fitness_diff_to_group_best.unsqueeze(1))
        if config.share_distance:
            features.append(self.mean_peer_distance.unsqueeze(1))
            features.append(self.min_peer_distance.unsqueeze(1))

        if not features:
            return torch.zeros((B, 1), device=self.mean_peer_distance.device)

        return torch.cat(features, dim=-1)
