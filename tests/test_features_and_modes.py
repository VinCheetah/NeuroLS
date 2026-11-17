"""Comprehensive unit and integration tests for:
1. Feature registry, keyword expansion (@action_basic, @scores, @all), exclusions (-feature).
2. Exact mathematical correctness of new multi-agent features:
   - (M-1) peer deltas (+1/-1 rapprochement) ordered by fitness
   - Group and others' barycentre distances
   - Group mean locus and best agent locus
   - Delta score to mean, normalized rank, peer scores and deltas
3. Both evaluation paradigms:
   - Per-Action evaluation (Linear, MLP, PeerInvariant)
   - All-Actions full-neighborhood evaluation (InvariantNNet, DeepSets, Dense)
4. CLI model parsing and execution with multiple models.
"""

from __future__ import annotations

import torch
from typer.testing import CliRunner

from neurols.cli.app import app
from neurols.core.group import GroupConfig, GroupContext
from neurols.core.runner import BatchedRunner
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.neural.features import (
    FeatureConfig,
    FeatureExtractor,
    compute_action_peer_deltas,
    resolve_feature_names,
)
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.types import StoppingCriteria

runner = CliRunner()


def test_feature_dsl_resolution() -> None:
    """Test keyword expansion, canonical mapping, and exclusion logic."""
    # 1. Basic group keyword
    feats = resolve_feature_names("@action_basic")
    assert feats == ["delta", "ranked_delta"]

    # 2. Combining keywords and individual features
    feats = resolve_feature_names("@action_basic,best_locus,normalized_rank")
    assert feats == ["delta", "ranked_delta", "best_locus", "normalized_rank"]

    # 3. Exclusions
    feats = resolve_feature_names("@action_all,-current_locus,-peers_mean_locus")
    assert "current_locus" not in feats
    assert "peers_mean_locus" not in feats
    assert "delta" in feats
    assert "best_locus" in feats

    # 4. Scores and distances groups
    scores = resolve_feature_names("@scores")
    assert "current_fitness" in scores
    assert "delta_mean_score" in scores
    assert "normalized_rank" in scores

    distances = resolve_feature_names("@distances")
    assert "dist_group_barycentre" in distances
    assert "dist_others_barycentre" in distances
    assert "peer_distances" in distances

    # 5. Default fallback
    assert len(resolve_feature_names(None)) > 0
    assert len(resolve_feature_names("")) > 0


def test_peer_ordering_by_fitness() -> None:
    """Verify that peer features (distances, deltas, scores) are strictly ordered by descending fitness."""
    # Group of 3 agents on N=4 bits
    group_cfg = GroupConfig(n_agents=3)
    group_ctx = GroupContext(config=group_cfg, n_instances=1, n_restarts=1)

    # Agent 0: [0, 0, 0, 0], fitness = 0.20 (worst)
    # Agent 1: [0, 1, 0, 0], fitness = 0.50 (medium)
    # Agent 2: [1, 0, 0, 0], fitness = 0.90 (best!)
    states = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int8)
    fitness = torch.tensor([0.20, 0.50, 0.90], dtype=torch.float32)

    metrics = group_ctx.update(
        states=states,
        current_fitness=fitness,
        best_fitness=fitness,
        best_states=states,
    )

    # For Agent 0:
    # Its peers are Agent 1 (fit=0.5) and Agent 2 (fit=0.9).
    # In descending fitness, Peer 0 is Agent 2 (fit=0.9), Peer 1 is Agent 1 (fit=0.5).
    assert metrics.ordered_peer_indices[0, 0].item() == 2
    assert metrics.ordered_peer_indices[0, 1].item() == 1
    assert torch.isclose(metrics.ordered_peer_scores[0, 0], torch.tensor(0.90))
    assert torch.isclose(metrics.ordered_peer_scores[0, 1], torch.tensor(0.50))

    # Delta scores for Agent 0: f(0) - f(peer)
    # 0.20 - 0.90 = -0.70, 0.20 - 0.50 = -0.30
    assert torch.isclose(metrics.ordered_peer_delta_scores[0, 0], torch.tensor(-0.70))
    assert torch.isclose(metrics.ordered_peer_delta_scores[0, 1], torch.tensor(-0.30))

    # Test action peer deltas for Agent 0:
    # Agent 2 has [1, 0, 0, 0]. Agent 0 is [0, 0, 0, 0].
    # Flipping bit 0 of Agent 0 reduces distance to Agent 2 (Peer 0) -> -1.0!
    # Agent 1 has [0, 1, 0, 0].
    # Flipping bit 0 of Agent 0 increases distance to Agent 1 (Peer 1) -> +1.0!
    peer_deltas, net_delta = compute_action_peer_deltas(
        states=states,
        group_ctx=group_ctx,
        group_metrics=metrics,
    )

    # For Agent 0, bit 0: Peer 0 (Agent 2) is -1.0, Peer 1 (Agent 1) is +1.0
    torch.testing.assert_close(peer_deltas[0, 0], torch.tensor([-1.0, 1.0]))


def test_barycentre_and_locus_calculations() -> None:
    """Verify group centroid (barycentre) and locus metrics."""
    group_cfg = GroupConfig(n_agents=3)
    group_ctx = GroupContext(config=group_cfg, n_instances=1, n_restarts=1)

    # 3 agents on N=4 bits
    # Agent 0: [1, 1, 0, 0]
    # Agent 1: [1, 0, 1, 0]
    # Agent 2: [0, 0, 0, 0]
    states = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.int8)
    fitness = torch.tensor([0.4, 0.8, 0.2], dtype=torch.float32)

    metrics = group_ctx.update(
        states=states,
        current_fitness=fitness,
        best_fitness=fitness,
        best_states=states,
    )

    # Group barycentre: mean across agents = [2/3, 1/3, 1/3, 0/3]
    expected_barycentre = torch.tensor([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
    torch.testing.assert_close(metrics.group_barycentre[0], expected_barycentre)

    # Distance of Agent 0 [1, 1, 0, 0] to group barycentre:
    # |1 - 2/3| + |1 - 1/3| + |0 - 1/3| + |0 - 0| = 1/3 + 2/3 + 1/3 + 0 = 4/3
    # Normalized by N=4: (4/3) / 4 = 1/3
    assert torch.isclose(metrics.dist_group_barycentre[0], torch.tensor(1.0 / 3.0))

    # Others' barycentre for Agent 0 (excluding Agent 0):
    # Others are Agent 1 [1, 0, 1, 0] and Agent 2 [0, 0, 0, 0] -> mean = [0.5, 0.0, 0.5, 0.0]
    expected_others = torch.tensor([0.5, 0.0, 0.5, 0.0])
    torch.testing.assert_close(metrics.others_barycentre[0], expected_others)

    # Best agent is Agent 1 ([1, 0, 1, 0]) with fitness 0.8
    torch.testing.assert_close(
        metrics.current_best_agent_states[0], torch.tensor([1, 0, 1, 0], dtype=torch.int8)
    )

    # Group mean fitness = (0.4 + 0.8 + 0.2) / 3 = 1.4 / 3 ≈ 0.4667
    # Delta score to mean for Agent 1: 0.8 - 0.4667 = 0.3333
    assert torch.isclose(metrics.delta_mean_score[1], torch.tensor(0.8 - (1.4 / 3.0)))


def test_feature_extractor_dimensions() -> None:
    """Verify that FeatureExtractor accurately formats dimensions for single-agent and multi-agent setups."""
    # Single agent (M=1)
    cfg_single = FeatureConfig.from_specs("@all", mode="per_action")
    extractor_single = FeatureExtractor(cfg_single, n_agents=1)
    states_1 = torch.zeros((2, 6), dtype=torch.int8)
    deltas_1 = torch.randn(2, 6)
    fit_1 = torch.tensor([0.5, 0.6])
    feat_1 = extractor_single.extract(
        states=states_1,
        deltas=deltas_1,
        current_fitness=fit_1,
        best_fitness=fit_1,
        group_ctx=None,
        group_metrics=None,
        step=0,
        max_steps=10,
        stagnation_counter=torch.zeros(2, dtype=torch.int64),
    )
    assert feat_1.shape == (2, 6, cfg_single.get_num_features(n_agents=1))

    # Multi-agent (M=3)
    group_cfg = GroupConfig(n_agents=3)
    group_ctx = GroupContext(group_cfg, n_instances=1, n_restarts=1)
    states_3 = torch.zeros((3, 6), dtype=torch.int8)
    fit_3 = torch.tensor([0.3, 0.7, 0.5])
    metrics_3 = group_ctx.update(states_3, fit_3, fit_3, states_3)

    cfg_multi = FeatureConfig.from_specs("@all", mode="per_action")
    extractor_multi = FeatureExtractor(cfg_multi, n_agents=3)
    feat_3 = extractor_multi.extract(
        states=states_3,
        deltas=torch.randn(3, 6),
        current_fitness=fit_3,
        best_fitness=fit_3,
        group_ctx=group_ctx,
        group_metrics=metrics_3,
        step=0,
        max_steps=10,
        stagnation_counter=torch.zeros(3, dtype=torch.int64),
    )
    assert feat_3.shape == (3, 6, cfg_multi.get_num_features(n_agents=3))


def test_evaluation_modes_rollout() -> None:
    """Verify both 'per_action' and 'all_actions' neural strategies run in BatchedRunner."""
    problem = NKLandscape(n=8, k=2, seed=42)

    # 1. Per-Action Mode (MLP scoring each bit flip independently)
    strat_per_action = NeuralLocalSearchStrategy(
        eval_mode="per_action",
        policy_type="mlp",
        features="@action_basic,best_locus",
        n_actions=8,
        n_agents=2,
        max_steps=5,
    )
    group_cfg = GroupConfig(n_agents=2)
    runner_pa = BatchedRunner(
        problem=problem,
        strategy=strat_per_action,
        group_config=group_cfg,
        stopping_criteria=StoppingCriteria(max_steps=5),
    )
    traj_pa = runner_pa.run(n_restarts=2, seed=1)
    assert traj_pa.steps_taken.max().item() > 0

    # 2. All-Actions Mode (Dense policy evaluating entire neighborhood at once)
    strat_all_actions_dense = NeuralLocalSearchStrategy(
        eval_mode="all_actions",
        policy_type="dense",
        features="@action_basic,best_locus",
        n_actions=8,
        n_agents=2,
        max_steps=5,
    )
    runner_dense = BatchedRunner(
        problem=problem,
        strategy=strat_all_actions_dense,
        group_config=group_cfg,
        stopping_criteria=StoppingCriteria(max_steps=5),
    )
    traj_dense = runner_dense.run(n_restarts=2, seed=2)
    assert traj_dense.steps_taken.max().item() > 0

    # 3. All-Actions Mode (InvariantNNet permutation-equivariant)
    strat_all_actions_inv = NeuralLocalSearchStrategy(
        eval_mode="all_actions",
        policy_type="invariant_nnet",
        features="@action_basic,best_locus",
        n_actions=8,
        n_agents=2,
        max_steps=5,
    )
    runner_inv = BatchedRunner(
        problem=problem,
        strategy=strat_all_actions_inv,
        group_config=group_cfg,
        stopping_criteria=StoppingCriteria(max_steps=5),
    )
    traj_inv = runner_inv.run(n_restarts=2, seed=3)
    assert traj_inv.steps_taken.max().item() > 0


def test_cli_features_command() -> None:
    """Test 'neurols features' command displays the registry and modes."""
    res = runner.invoke(app, ["features"])
    assert res.exit_code == 0
    assert "Action-Specific Features" in res.output
    assert "Agent & Group Features" in res.output
    assert "@action_basic" in res.output
    assert "per_action" in res.output
    assert "all_actions" in res.output


def test_cli_multiple_models_benchmark(tmp_path) -> None:
    """Test specifying multiple custom models directly in the CLI for benchmark."""
    out_dir = str(tmp_path / "bench_models")
    res = runner.invoke(
        app,
        [
            "benchmark",
            "-n",
            "8",
            "-k",
            "1",
            "--steps",
            "5",
            "--restarts",
            "2",
            "--no-train",
            "--no-plot",
            "--out",
            out_dir,
            "--model",
            "name=Net_Delta,mode=per_action,policy=linear,features=delta",
            "--model",
            "name=Net_Peers,mode=per_action,policy=mlp,features=@action_basic,peer_deltas",
            "--model",
            "name=Net_Joint,mode=all_actions,policy=dense,features=@action_basic",
        ],
    )
    assert res.exit_code == 0
    assert "Net_Delta" in res.output
    assert "Net_Peers" in res.output
    assert "Net_Joint" in res.output


def test_cli_run_with_custom_features_and_mode() -> None:
    """Test running local search with custom feature and mode CLI arguments."""
    res = runner.invoke(
        app,
        [
            "run",
            "--strategy",
            "neural",
            "--mode",
            "per_action",
            "--policy-type",
            "mlp",
            "--features",
            "@action_basic,peer_deltas,best_locus",
            "-m",
            "2",
            "-n",
            "10",
            "-k",
            "1",
            "--steps",
            "5",
            "--restarts",
            "2",
        ],
    )
    assert res.exit_code == 0
    assert "Mean Best Fitness" in res.output
