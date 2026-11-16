"""Vectorized NK Landscape implementation in PyTorch.

The NK Landscape (Kauffman, 1989) is a canonical tunable rugged fitness landscape
model widely used to study combinatorial search, ruggedness, and epistatic interactions.

Mathematical Formulation:
- N: Number of binary bits (variables), x in {0, 1}^N.
- K: Number of epistatic interactions per bit (0 <= K < N).
- Total fitness: f(x) = (1 / N) * sum_{i=0}^{N-1} f_i(x_i, x_{i_1}, ..., x_{i_K})
  where each f_i: {0, 1}^{K+1} -> [0, 1) is a random contribution table.
"""

from __future__ import annotations

import torch

from neurols.core.problem import BatchedProblem
from neurols.types import DeviceType, NeighborhoodType, SeedType


class NKLandscape(BatchedProblem):
    """Vectorized, GPU/CPU-accelerated NK Landscape model.

    Supports both adjacent and random epistatic interaction graphs, with fast
    batched state evaluation and exact 1-flip neighborhood delta computation.
    """

    def __init__(
        self,
        n: int,
        k: int,
        interaction_type: NeighborhoodType | str = NeighborhoodType.ADJACENT,
        tables: torch.Tensor | None = None,
        interactions: torch.Tensor | None = None,
        seed: SeedType = None,
        device: DeviceType = "cpu",
    ) -> None:
        """Initialize an NK Landscape instance.

        Args:
            n: Number of bits (dimension N >= 1).
            k: Epistatic interaction degree (0 <= K < N).
            interaction_type: Epistatic topology ('adjacent' or 'random').
            tables: Optional pre-existing contribution tables of shape (N, 2^(K+1)).
            interactions: Optional pre-existing interaction index matrix of shape (N, K+1).
            seed: Optional seed for generating tables and interactions reproducibly.
            device: Computation device ('cpu' or 'cuda').
        """
        super().__init__(device=device)

        if not (1 <= n):
            raise ValueError(f"N must be >= 1, got {n}")
        if not (0 <= k < n):
            raise ValueError(f"K must satisfy 0 <= K < N ({n}), got {k}")

        self.n = n
        self.k = k
        self.interaction_type = NeighborhoodType(interaction_type)
        self.seed = seed

        # Generator for reproducible instance creation
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        # 1. Epistatic Interactions: shape (N, K+1)
        # Entry (i, 0) is always bit i itself; entries (i, 1..K) are interacting bits.
        if interactions is not None:
            if interactions.shape != (n, k + 1):
                raise ValueError(
                    f"Expected interactions of shape ({n}, {k + 1}), got {interactions.shape}"
                )
            self._interactions = interactions.to(self.device).long()
        else:
            self._interactions = self._build_interactions(n, k, self.interaction_type, gen).to(
                self.device
            )

        # 2. Fitness Contribution Tables: shape (N, 2^(K+1))
        num_configs = 1 << (k + 1)
        if tables is not None:
            if tables.shape != (n, num_configs):
                raise ValueError(
                    f"Expected tables of shape ({n}, {num_configs}), got {tables.shape}"
                )
            self._tables = tables.to(self.device).float()
        else:
            # Random uniform contributions in [0, 1)
            self._tables = torch.rand(
                (n, num_configs), generator=gen, dtype=torch.float32, device="cpu"
            ).to(self.device)

        # Precompute binary powers for index encoding: shape (K+1,)
        # Powers: [2^K, 2^(K-1), ..., 2^1, 2^0]
        powers = 1 << torch.arange(k, -1, -1, device=self.device, dtype=torch.int64)
        self._powers = powers.unsqueeze(-1)  # shape (K+1, 1)

        # Precompute reverse dependencies for fast incremental delta updates:
        # For each bit j in [0, N-1], which functions i depend on it, and at what bit position?
        self._reverse_deps = self._build_reverse_dependencies()

    @property
    def dimension(self) -> int:
        """Dimensionality N (number of bits)."""
        return self.n

    @property
    def neighborhood_size(self) -> int:
        """Number of 1-flip neighbor moves (N)."""
        return self.n

    @property
    def tables(self) -> torch.Tensor:
        """Contribution tables tensor of shape (N, 2^(K+1))."""
        return self._tables

    @property
    def interactions(self) -> torch.Tensor:
        """Epistatic interaction index tensor of shape (N, K+1)."""
        return self._interactions

    def to(self, device: DeviceType) -> NKLandscape:
        """Move problem tensors to the designated device."""
        target = torch.device(device)
        if target == self.device:
            return self
        new_instance = NKLandscape(
            n=self.n,
            k=self.k,
            interaction_type=self.interaction_type,
            tables=self._tables.to(target),
            interactions=self._interactions.to(target),
            seed=self.seed,
            device=target,
        )
        return new_instance

    @staticmethod
    def _build_interactions(
        n: int,
        k: int,
        interaction_type: NeighborhoodType,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Construct the epistatic interaction graph."""
        interactions = torch.zeros((n, k + 1), dtype=torch.int64)

        for i in range(n):
            interactions[i, 0] = i
            if k > 0:
                if interaction_type == NeighborhoodType.ADJACENT:
                    # Circular adjacent bits: (i+1)%N, ..., (i+K)%N
                    for m in range(1, k + 1):
                        interactions[i, m] = (i + m) % n
                elif interaction_type == NeighborhoodType.RANDOM:
                    # K distinct other bits sampled uniformly without replacement
                    candidates = [j for j in range(n) if j != i]
                    # Deterministic permutation using generator
                    perm = torch.randperm(len(candidates), generator=generator)
                    selected = [candidates[idx] for idx in perm[:k].tolist()]
                    interactions[i, 1:] = torch.tensor(selected, dtype=torch.int64)

        return interactions

    def _build_reverse_dependencies(self) -> list[list[tuple[int, int, int]]]:
        """Precomputes reverse lookup for each bit j.

        Returns a list of length N, where entry j contains a list of tuples:
        (func_i, bit_position_in_func, bit_weight)
        where bit_weight = 2^(K - bit_position).
        """
        interactions_cpu = self._interactions.cpu()
        deps: list[list[tuple[int, int, int]]] = [[] for _ in range(self.n)]
        for i in range(self.n):
            for pos in range(self.k + 1):
                j = int(interactions_cpu[i, pos].item())
                weight = 1 << (self.k - pos)
                deps[j].append((i, pos, weight))
        return deps

    def sample_initial_states(
        self,
        batch_shape: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample random binary bitstrings uniformly in {0, 1}^N.

        Args:
            batch_shape: Shape tuple of batch dimensions, e.g. (B,) or (n_inst, n_rest, n_agent).
            generator: Optional PyTorch random generator.

        Returns:
            Binary int8 tensor of shape (*batch_shape, N).
        """
        total_shape = (*batch_shape, self.n)
        probs = torch.full(total_shape, 0.5, device=self.device)
        return torch.bernoulli(probs, generator=generator).to(torch.int8)

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        """Vectorized fitness evaluation for a batch of binary states.

        f(x) = (1 / N) * sum_{i=0}^{N-1} table_i[ config_idx_i(x) ]

        Args:
            states: Binary tensor of shape (B, N) or (*batch_shape, N).

        Returns:
            Fitness tensor of shape (*batch_shape,) with values in [0, 1].
        """
        orig_shape = states.shape[:-1]
        states_flat = states.view(-1, self.n).long()  # (B, N)
        B = states_flat.shape[0]

        # Gather interaction bits for each function i:
        # states_flat: (B, N)
        # self._interactions: (N, K+1)
        # gathered: (B, N, K+1)
        gathered = states_flat[:, self._interactions]

        # Compute integer config index for each bit i:
        # (B, N, K+1) @ (K+1, 1) -> (B, N, 1) -> squeeze to (B, N)
        config_indices = torch.matmul(gathered, self._powers).squeeze(-1)  # (B, N)

        # Lookup contributions from self._tables: shape (N, 2^(K+1))
        # tables_expanded: (1, N, 2^(K+1)) -> expand to (B, N, 2^(K+1))
        tables_expanded = self._tables.unsqueeze(0).expand(B, -1, -1)
        contributions = torch.gather(
            tables_expanded, dim=2, index=config_indices.unsqueeze(-1)
        ).squeeze(-1)  # (B, N)

        # Mean fitness over all N functions
        fitness = contributions.mean(dim=-1)  # (B,)

        if orig_shape:
            return fitness.view(orig_shape)
        return fitness

    def get_neighborhood_deltas(self, states: torch.Tensor) -> torch.Tensor:
        """Compute fitness deltas for all N 1-flip neighbor moves.

        delta_f(x, j) = f(x ^ e_j) - f(x).

        For high throughput and vectorization, this evaluates all 1-bit flips
        efficiently using PyTorch.

        Args:
            states: Binary tensor of shape (B, N) or (*batch_shape, N).

        Returns:
            Tensor of shape (*batch_shape, N) containing fitness deltas for flipping bit j.
        """
        orig_shape = states.shape[:-1]
        states_flat = states.view(-1, self.n).to(torch.int8)  # (B, N)
        B = states_flat.shape[0]

        # Current base fitness: (B,)
        base_fitness = self.evaluate(states_flat)  # (B,)

        # Vectorized neighborhood generation:
        # Create all N single-bit flips for each state in batch
        # flipped_states: shape (B, N, N)
        # where flipped_states[b, j, :] is states_flat[b] with bit j flipped
        flipped = states_flat.unsqueeze(1).repeat(1, self.n, 1)  # (B, N, N)
        eye = torch.eye(self.n, device=self.device, dtype=torch.int8).unsqueeze(0)
        flipped = flipped ^ eye  # Invert bit j in row j

        # Evaluate all B * N flipped states in a single batch pass
        flipped_flat = flipped.view(B * self.n, self.n)
        flipped_fitness = self.evaluate(flipped_flat).view(B, self.n)  # (B, N)

        # Deltas: f(x ^ e_j) - f(x)
        deltas = flipped_fitness - base_fitness.unsqueeze(1)  # (B, N)

        if orig_shape:
            return deltas.view(*orig_shape, self.n)
        return deltas

    def apply_moves(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """Apply chosen 1-bit flips to states.

        Args:
            states: Binary states tensor of shape (B, N) or (*batch_shape, N).
            moves: Move indices to flip, shape (B,) or (*batch_shape,). Values in [0, N-1].

        Returns:
            Updated states tensor with the designated bit flipped.
        """
        orig_shape = states.shape
        states_flat = states.view(-1, self.n).clone()
        moves_flat = moves.view(-1).long()

        # Create one-hot mask for flipped bits
        one_hot = torch.zeros_like(states_flat)
        valid_mask = (moves_flat >= 0) & (moves_flat < self.n)
        if valid_mask.any():
            one_hot[valid_mask, moves_flat[valid_mask]] = 1

        new_states = states_flat ^ one_hot
        return new_states.view(orig_shape)

    def compute_distances(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """Compute Hamming distance between binary states.

        Args:
            states1: First state tensor (*shape, N).
            states2: Second state tensor (*shape, N).

        Returns:
            Hamming distance tensor (*shape,).
        """
        return (states1 != states2).sum(dim=-1).float()

    def analyze_ruggedness(self, n_samples: int = 2000) -> dict[str, float]:
        """Estimate landscape ruggedness, density of local optima, and autocorrelation length.

        Returns:
            Dictionary containing landscape statistics:
                - mean_fitness: Expected fitness of random states.
                - std_fitness: Variance of fitness across state space.
                - local_optima_density: Estimated fraction of states that are local optima.
                - avg_improving_moves: Average number of improving 1-flip moves from random state.
        """
        sample_states = self.sample_initial_states((n_samples,))
        fitnesses = self.evaluate(sample_states)
        deltas = self.get_neighborhood_deltas(sample_states)  # (n_samples, N)

        is_optima = (deltas <= 0.0).all(dim=-1)
        improving_moves_per_state = (deltas > 0.0).float().sum(dim=-1)

        return {
            "n": float(self.n),
            "k": float(self.k),
            "mean_fitness": float(fitnesses.mean().item()),
            "std_fitness": float(fitnesses.std().item()),
            "min_fitness": float(fitnesses.min().item()),
            "max_fitness": float(fitnesses.max().item()),
            "local_optima_density": float(is_optima.float().mean().item()),
            "avg_improving_moves": float(improving_moves_per_state.mean().item()),
        }


class MultiNKLandscape(BatchedProblem):
    """Collection of multiple independent NK Landscape instances evaluated in parallel.

    Supports instances of identical (N, K) dimensions where each problem instance in
    a batch uses its own random landscape tables and interaction graph.
    """

    def __init__(
        self,
        n_instances: int,
        n: int,
        k: int,
        interaction_type: NeighborhoodType | str = NeighborhoodType.ADJACENT,
        base_seed: SeedType = 42,
        device: DeviceType = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.n_instances = n_instances
        self.n = n
        self.k = k
        self.interaction_type = NeighborhoodType(interaction_type)
        self.base_seed = base_seed

        # Instantiate each independent NKLandscape
        self.landscapes: list[NKLandscape] = []
        for i in range(n_instances):
            inst_seed = (base_seed + i * 1000) if base_seed is not None else None
            ls = NKLandscape(
                n=n,
                k=k,
                interaction_type=self.interaction_type,
                seed=inst_seed,
                device=device,
            )
            self.landscapes.append(ls)

    @property
    def dimension(self) -> int:
        return self.n

    @property
    def neighborhood_size(self) -> int:
        return self.n

    def to(self, device: DeviceType) -> MultiNKLandscape:
        target = torch.device(device)
        if target == self.device:
            return self
        new_multi = MultiNKLandscape(
            n_instances=self.n_instances,
            n=self.n,
            k=self.k,
            interaction_type=self.interaction_type,
            base_seed=self.base_seed,
            device=target,
        )
        return new_multi

    def sample_initial_states(
        self,
        batch_shape: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample initial states for all instances."""
        total_shape = (*batch_shape, self.n)
        probs = torch.full(total_shape, 0.5, device=self.device)
        return torch.bernoulli(probs, generator=generator).to(torch.int8)

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate states partitioned across instances.

        Assumes states tensor has leading dimension divisible by n_instances.
        Shape: (n_instances, restarts_and_agents, N) -> returns (n_instances, restarts_and_agents).
        """
        orig_shape = states.shape[:-1]
        B = states.view(-1, self.n).shape[0]
        per_inst = B // self.n_instances
        states_reshaped = states.view(self.n_instances, per_inst, self.n)

        fitness_list: list[torch.Tensor] = []
        for i, ls in enumerate(self.landscapes):
            fit_i = ls.evaluate(states_reshaped[i])  # (per_inst,)
            fitness_list.append(fit_i)

        all_fitness = torch.stack(fitness_list, dim=0)  # (n_instances, per_inst)
        return all_fitness.view(orig_shape)

    def get_neighborhood_deltas(self, states: torch.Tensor) -> torch.Tensor:
        """Compute deltas across all instances."""
        orig_shape = states.shape[:-1]
        B = states.view(-1, self.n).shape[0]
        per_inst = B // self.n_instances
        states_reshaped = states.view(self.n_instances, per_inst, self.n)

        deltas_list: list[torch.Tensor] = []
        for i, ls in enumerate(self.landscapes):
            deltas_i = ls.get_neighborhood_deltas(states_reshaped[i])  # (per_inst, N)
            deltas_list.append(deltas_i)

        all_deltas = torch.stack(deltas_list, dim=0)  # (n_instances, per_inst, N)
        return all_deltas.view(*orig_shape, self.n)

    def apply_moves(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        return self.landscapes[0].apply_moves(states, moves)

    def compute_distances(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        return (states1 != states2).sum(dim=-1).float()
