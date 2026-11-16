"""Tests for InvariantModule, InvariantNNet, PerActionScorer, and AblationStudy."""

import torch

from neurols.analysis.ablation import AblationStrategySpec, AblationStudy
from neurols.core.group import GroupConfig, GroupContext
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.neural.architectures import (
    InvariantModule,
    InvariantNNet,
    PerActionScorer,
)
from neurols.strategies.neural.features import (
    FeatureAblationLevel,
    compute_action_peer_deltas,
    compute_normalized_ranks,
)


def test_invariant_module_permutation_equivariance() -> None:
    # Batch=2, N=5 actions, Features=3
    B, N, F = 2, 5, 3
    x = torch.randn(B, N, F)

    # Permute actions (e.g. reverse order)
    perm = torch.tensor([4, 3, 2, 1, 0])
    x_perm = x[:, perm, :]

    # Invariant linear module with remix and rescale
    InvLinear = InvariantModule(torch.nn.Linear, n_elements=N, remix=True, rescale=True)
    layer = InvLinear(F, 4)

    y = layer(x)
    y_perm = layer(x_perm)

    # Permuting input must permute the output by the exact same permutation!
    torch.testing.assert_close(y[:, perm, :], y_perm, atol=1e-5, rtol=1e-5)


def test_invariant_nnet_forward_and_params() -> None:
    N = 8
    net = InvariantNNet(n_actions=N, layers_size=[2, 16, 1], remix=True, rescale=False)
    x = torch.randn(3, N, 2)
    out = net(x)
    assert out.shape == (3, N)

    # Flat weights roundtrip
    w = net.get_flat_weights()
    assert len(w) == net.num_parameters
    net.set_flat_weights(w + 0.1)


def test_per_action_scorer() -> None:
    scorer = PerActionScorer(num_features=4, hidden_layers=[8, 8])
    x = torch.randn(2, 6, 4)  # 2 batch, 6 actions, 4 features
    scores = scorer(x)
    assert scores.shape == (2, 6)


def test_normalized_ranks_scale_invariance() -> None:
    # 5 actions with deltas
    deltas = torch.tensor([[-0.5, -0.1, 0.0, 0.2, 0.8]])
    ranks = compute_normalized_ranks(deltas)
    assert ranks.shape == (1, 5)

    # Negative deltas must be <= 0, positive deltas must be > 0
    assert ranks[0, 0] <= 0
    assert ranks[0, 1] <= 0
    assert ranks[0, 3] > 0
    assert ranks[0, 4] > 0

    # Scale invariance: multiplying deltas by 100 must yield identical ranks!
    scaled_deltas = deltas * 100.0
    scaled_ranks = compute_normalized_ranks(scaled_deltas)
    torch.testing.assert_close(ranks, scaled_ranks)


def test_action_peer_deltas_invariance() -> None:
    # 1 group of 3 agents on N=4 bits
    group_cfg = GroupConfig(n_agents=3)
    group_ctx = GroupContext(config=group_cfg, n_instances=1, n_restarts=1)

    # Agent 0: [0, 0, 0, 0]
    # Agent 1: [0, 1, 0, 0]
    # Agent 2: [1, 0, 0, 0]
    states = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int8)

    peer_deltas, net_delta = compute_action_peer_deltas(states, group_ctx, sort_peers=True)
    assert peer_deltas.shape == (3, 4, 2)  # 3 agents, 4 actions, 2 peers
    assert net_delta.shape == (3, 4, 1)

    # For Agent 0 flipping bit 0:
    # Agent 1 has bit 0 = 0 (same) -> delta is +1
    # Agent 2 has bit 0 = 1 (diff) -> delta is -1
    # Sorted peer deltas: must be [-1.0, 1.0]
    torch.testing.assert_close(peer_deltas[0, 0], torch.tensor([-1.0, 1.0]))


def test_ablation_study_execution() -> None:
    train_prob = NKLandscape(n=8, k=2, seed=11)
    test_prob = NKLandscape(n=8, k=2, seed=22)
    group_cfg = GroupConfig(n_agents=2)

    specs = [
        AblationStrategySpec(
            name="Test_PerAction",
            ablation_level=FeatureAblationLevel.DELTA_ONLY,
            policy_type="per_action",
            description="test",
        ),
        AblationStrategySpec(
            name="Test_InvariantNNet",
            ablation_level=FeatureAblationLevel.RANKED_DELTA_ONLY,
            policy_type="invariant_nnet",
            description="test",
        ),
    ]

    study = AblationStudy(
        train_problem=train_prob,
        test_problem=test_prob,
        group_config=group_cfg,
        train_generations=2,
        eval_restarts=2,
        eval_steps=10,
        cma_popsize=4,
        test_restarts=4,
        test_steps=15,
        seed=42,
    )

    res = study.run(specs=specs, include_classical_baselines=True)
    assert len(res.trajectories) == 4  # 2 neural + HillClimber + Tabu
    assert len(res.summary_table) == 4
