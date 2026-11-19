"""Tests for InvariantModule, InvariantNNet, PerActionScorer, and neural architectures."""

import torch

from neurols.analysis.multi_agent_study import ModelSpec, MultiAgentStudy
from neurols.core.group import GroupConfig, GroupContext
from neurols.core.runner import BatchedRunner
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.classical import RandomWalk
from neurols.strategies.neural.architectures import (
    InvariantModule,
    InvariantNNet,
    PeerInvariantActionScorer,
    PerActionScorer,
)
from neurols.strategies.neural.features import (
    compute_action_peer_deltas,
    compute_normalized_ranks,
)
from neurols.types import StoppingCriteria, TerminationReason


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

    # When distance sharing is disabled:
    group_no_share = GroupConfig(n_agents=3, share_distance=False)
    ctx_no_share = GroupContext(config=group_no_share, n_instances=1, n_restarts=1)
    p_zeros, n_zeros = compute_action_peer_deltas(states, ctx_no_share, sort_peers=True)
    assert (p_zeros == 0).all()
    assert (n_zeros == 0).all()




def test_peer_invariant_action_scorer_invariance() -> None:
    # 2 batch, 5 actions, base_features=3, 4 peers (n_agents=5)
    B, N, F_base, n_agents = 2, 5, 3, 5
    n_peers = n_agents - 1
    scorer = PeerInvariantActionScorer(
        num_base_features=F_base,
        n_agents=n_agents,
        peer_feature_dim=1,
        hidden_dim=16,
    )

    x_base = torch.randn(B, N, F_base)
    # Peer features: (B, N, n_peers)
    x_peers = torch.randn(B, N, n_peers)
    features = torch.cat([x_base, x_peers], dim=-1)

    scores = scorer(features)
    assert scores.shape == (B, N)

    # Permuting peer order must yield identical action scores!
    perm = torch.tensor([3, 0, 2, 1])
    x_peers_perm = x_peers[:, :, perm]
    features_perm = torch.cat([x_base, x_peers_perm], dim=-1)
    scores_perm = scorer(features_perm)

    torch.testing.assert_close(scores, scores_perm, atol=1e-5, rtol=1e-5)


def test_stopping_criteria_combination() -> None:
    problem = NKLandscape(n=10, k=2, seed=42)
    strat = RandomWalk(exploration_rate=1.0)

    # 1. Stop by budget
    crit_budget = StoppingCriteria(max_steps=100, max_evaluations=30)
    runner_budget = BatchedRunner(problem=problem, strategy=strat, stopping_criteria=crit_budget)
    traj_budget = runner_budget.run(n_restarts=1)
    assert traj_budget.termination_reason == TerminationReason.EVALUATION_BUDGET
    assert traj_budget.total_evaluations >= 30

    # 2. Stop by steps
    crit_steps = StoppingCriteria(max_steps=5, max_evaluations=10000)
    runner_steps = BatchedRunner(
        problem=problem,
        strategy=strat,
        stopping_criteria=crit_steps,
    )
    traj_steps = runner_steps.run(n_restarts=1)
    assert traj_steps.termination_reason == TerminationReason.MAX_STEPS
    assert traj_steps.steps_taken.max().item() == 5


def test_multi_agent_study_run(tmp_path) -> None:
    train_prob = NKLandscape(n=6, k=1, seed=1)
    test_prob = NKLandscape(n=6, k=1, seed=2)

    models = [
        ModelSpec(
            name="Test_HC",
            model_type="classical",
        ),
        ModelSpec(
            name="Test_PeerInv",
            model_type="neural",
            policy_type="peer_invariant_per_action",
            features="@action_basic,peer_deltas",
        ),
    ]

    study = MultiAgentStudy(
        train_problem=train_prob,
        test_problem=test_prob,
        agent_counts=[1, 2],
        sharing_modes=[True, False],
        models=models,
        stopping_criteria=StoppingCriteria(max_steps=4),
        train_generations=1,
        test_restarts=2,
        seed=42,
    )

    out_dir = tmp_path / "study"
    res = study.run(output_dir=out_dir)
    assert len(res.records) > 0
    assert (out_dir / "study_results.json").exists()
    assert (out_dir / "fitness_vs_agents.png").exists()
    assert (out_dir / "sharing_gain.png").exists()
    assert (out_dir / "step_convergence.png").exists()
    assert (out_dir / "diversity_dynamics.png").exists()
    assert (out_dir / "training_curves.png").exists()
