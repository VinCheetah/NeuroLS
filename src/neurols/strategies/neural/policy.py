"""Neural network policy architectures and action scorers for Local Search.

This module unifies all neural scoring models into two clear operational paradigms:
1. Per-Action Scorers (Local Evaluation):
   - LinearScorer: Interpretable linear combination of features for an action.
   - PerActionScorer: Deep MLP evaluating an action independently, using
     permutation-invariant multi-agent features (such as sorted peer distance deltas).
2. All-Actions Scorers (Full-Neighborhood Joint Evaluation):
   - InvariantNNet: DeepSets-style permutation-equivariant network using InvariantModule
     (mixing individual action features and neighborhood averages at every layer).
   - EquivariantMLPPolicy: Independent move MLP without inter-move mixing.
   - DeepSetsPolicy: Encoder-Pool-Decoder architecture.

Core Building Blocks:
   - Net: Multi-layer perceptron with Tanh activations.
   - InvariantModule: Meta-module implementing permutation-equivariant linear layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseNeuralPolicy(nn.Module, ABC):
    """Abstract base class for all neural local search policies and scorers."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute action logits / scores for candidate moves.

        Args:
            features: Tensor of shape (B, N, num_features).

        Returns:
            Logits tensor of shape (B, N).
        """
        pass

    def get_flat_weights(self) -> np.ndarray:
        """Flatten all trainable parameters into a 1D NumPy array for CMA-ES."""
        params: list[np.ndarray] = []
        for p in self.parameters():
            params.append(p.detach().cpu().numpy().flatten())
        return np.concatenate(params) if params else np.empty(0, dtype=np.float32)

    def set_flat_weights(self, flat_weights: np.ndarray) -> None:
        """Load flat parameter vector back into PyTorch parameters."""
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                slice_arr = flat_weights[offset : offset + numel]
                p.copy_(torch.from_numpy(slice_arr).view_as(p).to(p.device, dtype=p.dtype))
                offset += numel

    @property
    def num_parameters(self) -> int:
        """Total number of scalar parameters in the policy."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Core Building Blocks
# ============================================================================


class Net(nn.Module):
    """Standard Multi-Layer Perceptron with Tanh activations.

    Used as the feed-forward backbone for scoring individual actions.
    """

    def __init__(self, layers_size: list[int] | tuple[int, ...]) -> None:
        super().__init__()
        self.layers_size = list(layers_size)
        layers: list[nn.Module] = []

        for i in range(len(self.layers_size) - 1):
            layers.append(nn.Linear(self.layers_size[i], self.layers_size[i + 1]))
            if i != len(self.layers_size) - 2:
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def InvariantModule(
    module_class: type[nn.Module],
    n_elements: int,
    remix: bool = True,
    rescale: bool = False,
) -> type[nn.Module]:
    """Higher-order meta-module that wraps a PyTorch module to mix elements across dimension 1.

    For input tensor of shape (batch, N, features):
    - Computes individual branch: ind_module(x)
    - Computes mean branch: mean_module(mean(x, dim=1, keepdim=True))
    - Combines them:
        if rescale: (N / (N + 1)) * ind + (1 / (N + 1)) * mean
        else: ind + mean

    This satisfies exact permutation equivariance over the N elements (DeepSets).
    """

    class _InvariantModuleClass(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self._ind_module = module_class(*args, **kwargs)
            self._remix = remix
            self._rescale = rescale
            self._n = n_elements
            if remix:
                self._mean_module = module_class(*args, **kwargs)

        def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
            if self._remix:
                red_inputs = [torch.mean(x, dim=1, keepdim=True) for x in inputs]
                ind_out = self._ind_module(*inputs)
                mean_out = self._mean_module(*red_inputs)

                if self._rescale:
                    return (self._n * ind_out / (self._n + 1)) + (mean_out / (self._n + 1))
                else:
                    return ind_out + mean_out
            else:
                return self._ind_module(*inputs)

        def reset_parameters(self) -> None:
            if hasattr(self._ind_module, "reset_parameters"):
                self._ind_module.reset_parameters()
            if self._remix and hasattr(self._mean_module, "reset_parameters"):
                self._mean_module.reset_parameters()

    return _InvariantModuleClass


# ============================================================================
# Paradigm 1: Per-Action Scorers (Local Evaluation)
# ============================================================================


class LinearScorer(BaseNeuralPolicy):
    """Interpretable linear policy scoring candidate moves.

    Logit for move j is a direct linear combination of its features:
    z_j = sum_{f} w_f * x_{j, f} + bias.
    Directly inspectable to analyze learned heuristic rules.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            self.linear.weight[0, 0] = 5.0
            nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.float()).squeeze(-1)

    def get_feature_weights(self, feature_names: list[str]) -> dict[str, float]:
        """Return a mapping of feature names to their learned linear weights."""
        weights = self.linear.weight.squeeze(0).detach().cpu().numpy()
        return {name: float(w) for name, w in zip(feature_names, weights, strict=False)}


class PerActionScorer(BaseNeuralPolicy):
    """Scores each candidate action individually with an MLP.

    Given a tensor of shape (B, N, action_features), applies an MLP to each
    candidate action independently: score(b, j) = MLP(features(b, j)).
    Action-level multi-agent features (such as delta distance to peers) must be
    made permutation-invariant before being passed to this scorer (e.g. by sorting).
    """

    def __init__(
        self,
        num_features: int,
        hidden_layers: list[int] | tuple[int, ...] = (16, 16),
        init_heuristic: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        layers_size = [num_features, *hidden_layers, 1]
        self.net = Net(layers_size)
        if init_heuristic:
            self.init_heuristic_weights()

    def init_heuristic_weights(self) -> None:
        """Initialize weights so that candidate actions with positive delta get positive scores initially."""
        with torch.no_grad():
            for layer in self.net.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.zero_()
                    if layer.bias is not None:
                        layer.bias.data.zero_()
            # Feature 0 (delta_fitness) forward propagation path
            first = self.net.layers[0]
            if isinstance(first, nn.Linear):
                first.weight.data[:, 0] = 1.0
            for layer in self.net.layers[1:-1]:
                if isinstance(layer, nn.Linear):
                    dim = min(layer.in_features, layer.out_features)
                    layer.weight.data[:dim, :dim] = torch.eye(dim)
            last = self.net.layers[-1]
            if isinstance(last, nn.Linear):
                last.weight.data.fill_(1.0 / max(last.in_features, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)

    def reset_parameters(self) -> None:
        self.net.reset_parameters()


class PeerInvariantActionScorer(BaseNeuralPolicy):
    """Scores candidate actions using an InvariantModule / DeepSets layer over peer features.

    For each candidate action j in {0, ..., N-1}:
    - Action features are partitioned into:
        1. Base action features (e.g. delta fitness, ranked delta, current bit)
        2. Peer features (e.g. distance changes +/-1 to each of the M - 1 peers)
    - The M - 1 peer tokens are passed through an InvariantModule (mixing individual
      peer deltas and the group mean delta) and pooled.
    - The pooled group context is merged with the base action features to produce
      the final score for action j.

    Guarantees strict permutation-invariance over the peers in the group!
    When M=1 (solo agent), peer processing is skipped and scoring relies purely on base features.
    """

    def __init__(
        self,
        num_base_features: int = 1,
        n_agents: int = 1,
        peer_feature_dim: int = 1,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_base_features = num_base_features
        self.n_agents = n_agents
        self.n_peers = max(n_agents - 1, 0)
        self.peer_feature_dim = peer_feature_dim
        self.hidden_dim = hidden_dim

        if self.n_peers > 0:
            InvLinear = InvariantModule(
                nn.Linear, n_elements=self.n_peers, remix=True, rescale=True
            )
            self.peer_layer = InvLinear(peer_feature_dim, hidden_dim)
            self.peer_act = nn.Tanh()
            head_in_dim = num_base_features + hidden_dim
        else:
            self.peer_layer = None
            self.peer_act = None
            head_in_dim = num_base_features

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, F = features.shape

        if self.n_peers > 0 and F >= self.num_base_features + self.n_peers:
            base_feats = features[:, :, : self.num_base_features]
            peer_feats = features[
                :, :, self.num_base_features : self.num_base_features + self.n_peers
            ]
            peer_tokens = peer_feats.reshape(B * N, self.n_peers, 1)
            h_peers = self.peer_act(self.peer_layer(peer_tokens))
            pooled_peers = h_peers.mean(dim=1).view(B, N, self.hidden_dim)
            combined = torch.cat([base_feats, pooled_peers], dim=-1)
            scores = self.head(combined).squeeze(-1)
        else:
            base_feats = features[:, :, : self.num_base_features]
            scores = self.head(base_feats).squeeze(-1)

        return scores

    def reset_parameters(self) -> None:
        if self.peer_layer is not None and hasattr(self.peer_layer, "reset_parameters"):
            self.peer_layer.reset_parameters()
        for layer in self.head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# ============================================================================
# Paradigm 2: All-Actions Scorers (Full-Neighborhood Joint Evaluation)
# ============================================================================


class InvariantNNet(BaseNeuralPolicy):
    """Permutation-equivariant neural network over all candidate actions.

    Chains multiple InvariantModule layers with Tanh activations, operating on
    tensors of shape (batch, N, dimInput) and outputting action scores of shape
    (batch, N). Permuting input actions permutes output scores identically.
    """

    def __init__(
        self,
        n_actions: int,
        layers_size: list[int] | tuple[int, ...],
        remix: bool = True,
        rescale: bool = False,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.remix = remix
        self.rescale = rescale
        self.layers_size = list(layers_size)

        InvariantLinear = InvariantModule(nn.Linear, self.n_actions, self.remix, self.rescale)
        layers: list[nn.Module] = []

        for i in range(len(self.layers_size) - 1):
            layers.append(InvariantLinear(self.layers_size[i], self.layers_size[i + 1]))
            if i != len(self.layers_size) - 2:
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.layers(features.float())
        return out.squeeze(-1)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class EquivariantMLPPolicy(BaseNeuralPolicy):
    """Permutation-equivariant MLP applied independently to each candidate move.

    z_j = MLP(f_j).
    """

    def __init__(self, num_features: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


class DeepSetsPolicy(BaseNeuralPolicy):
    """DeepSets-style permutation-equivariant policy with global neighborhood pooling.

    h_j = enc(f_j), g = mean_j(h_j), z_j = dec([h_j, g]).
    """

    def __init__(self, num_features: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, _ = features.shape
        h = self.encoder(features.float())
        g = h.mean(dim=1, keepdim=True).expand(B, N, self.hidden_dim)
        combined = torch.cat([h, g], dim=-1)
        return self.decoder(combined).squeeze(-1)


class DenseNeighborhoodPolicy(BaseNeuralPolicy):
    """Dense all-actions joint neighborhood policy.

    Flattens all N actions' features into a single vector of size N * num_features,
    evaluates the entire neighborhood simultaneously through an MLP, and directly
    outputs logits for all N actions in a single forward pass.
    """

    def __init__(
        self,
        n_actions: int,
        num_features: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.num_features = num_features
        in_dim = n_actions * num_features

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, F = features.shape
        flat = features.float().reshape(B, N * F)
        return self.net(flat)
