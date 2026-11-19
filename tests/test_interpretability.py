"""Unit tests for NeuroLS Interpretability, ModelInspector, and RealNKSession."""

from typer.testing import CliRunner

from neurols.analysis.interpretability import (
    ModelInspector,
    RealNKSession,
    get_feature_metadata,
    sample_feature_value,
)
from neurols.cli.app import app
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy


def test_feature_metadata_and_sampling() -> None:
    """Test metadata resolution and physical bounds sampling."""
    meta_delta = get_feature_metadata("delta")
    assert meta_delta.category == "action"
    assert meta_delta.min_val == -1.0
    assert meta_delta.max_val == 1.0

    meta_bit = get_feature_metadata("current_locus")
    assert meta_bit.is_binary is True
    assert meta_bit.min_val == 0.0
    assert meta_bit.max_val == 1.0

    meta_peer = get_feature_metadata("peer_deltas_0")
    assert meta_peer.is_discrete is True
    assert meta_peer.min_val == -1.0
    assert meta_peer.max_val == 1.0

    val_delta = sample_feature_value("delta", seed=42)
    assert -1.0 <= val_delta <= 1.0

    val_bit = sample_feature_value("current_locus", seed=42)
    assert val_bit in (0.0, 1.0)

    val_peer = sample_feature_value("peer_deltas_0", seed=42)
    assert val_peer in (-1.0, 1.0)


def test_model_inspector_linear() -> None:
    """Test linear model inspection, feature breakdown, and sensitivities."""
    strat = NeuralLocalSearchStrategy(
        policy_type="linear",
        features="delta,ranked_delta",
        n_actions=10,
    )
    inspector = ModelInspector(strat)

    assert inspector.is_linear is True
    assert inspector.is_per_action is True
    assert inspector.feature_names == ["delta", "ranked_delta"]

    defs = inspector.get_default_feature_vector()
    eval_res = inspector.evaluate_single_action(defs)

    assert isinstance(eval_res.score, float)
    assert eval_res.linear_breakdown is not None
    assert "delta" in eval_res.linear_breakdown
    assert "ranked_delta" in eval_res.linear_breakdown
    assert "delta" in eval_res.feature_sensitivities
    assert eval_res.feature_sensitivities["delta"] > 0.0


def test_model_inspector_mlp() -> None:
    """Test MLP per-action policy inspection and autograd sensitivity."""
    strat = NeuralLocalSearchStrategy(
        policy_type="mlp",
        features="delta,ranked_delta",
        n_actions=10,
    )
    inspector = ModelInspector(strat)

    assert inspector.is_linear is False
    assert inspector.is_per_action is True

    defs = inspector.get_default_feature_vector()
    eval_res = inspector.evaluate_single_action(defs)

    assert isinstance(eval_res.score, float)
    assert eval_res.linear_breakdown is None
    assert "delta" in eval_res.feature_sensitivities
    assert eval_res.feature_sensitivities["delta"] >= 0.0


def test_sensitivity_curve() -> None:
    """Test 1D sensitivity response curve computation and threshold detection."""
    strat = NeuralLocalSearchStrategy(
        policy_type="linear",
        features="delta,ranked_delta",
        n_actions=10,
    )
    inspector = ModelInspector(strat)
    defs = inspector.get_default_feature_vector()

    curve = inspector.compute_sensitivity_curve(
        target_feature="delta",
        base_features=defs,
        min_val=-1.0,
        max_val=1.0,
        num_points=21,
    )

    assert curve.feature_name == "delta"
    assert len(curve.x_values) == 21
    assert len(curve.scores) == 21
    assert curve.min_score < curve.max_score
    assert len(curve.zero_crossings) >= 1
    assert abs(curve.local_gradient - 5.0) < 0.1  # Linear weight initialized to 5.0


def test_sensitivity_surface() -> None:
    """Test 2D sensitivity interaction grid computation."""
    strat = NeuralLocalSearchStrategy(
        policy_type="linear",
        features="delta,ranked_delta",
        n_actions=10,
    )
    inspector = ModelInspector(strat)
    defs = inspector.get_default_feature_vector()

    surf = inspector.compute_sensitivity_surface(
        feat1="delta",
        feat2="ranked_delta",
        base_features=defs,
        grid_size=5,
    )

    assert surf.feat1 == "delta"
    assert surf.feat2 == "ranked_delta"
    assert len(surf.grid_x) == 5
    assert len(surf.grid_y) == 5
    assert len(surf.scores_grid) == 5
    assert len(surf.scores_grid[0]) == 5


def test_real_nk_session_single_agent() -> None:
    """Test real NK landscape session with stepping and candidate moves."""
    strat = NeuralLocalSearchStrategy(
        policy_type="linear",
        features="delta,ranked_delta",
        n_actions=8,
    )
    session = RealNKSession(strategy=strat, n=8, k=1, n_agents=1, seed=42)

    diag = session.evaluate_current_state()
    assert diag["step"] == 0
    assert diag["n_agents"] == 1
    assert len(diag["agents"]) == 1

    agent = diag["agents"][0]
    assert len(agent["current_state"]) == 8
    assert len(agent["candidate_actions"]) == 8
    assert 0 <= agent["chosen_action"] < 8

    # Step according to model choice
    step_model_res = session.step_model(agent_idx=0)
    assert step_model_res["chosen_bit"] == agent["chosen_action"]
    assert session.step_count == 1
    assert len(session.trajectory_history) == 2

    # Step manual choice
    manual_bit = (agent["chosen_action"] + 1) % 8
    step_man_res = session.step_manual(bit_idx=manual_bit, agent_idx=0)
    assert step_man_res["chosen_bit"] == manual_bit
    assert session.step_count == 2
    assert len(session.trajectory_history) == 3


def test_real_nk_session_multi_agent() -> None:
    """Test real NK landscape multi-agent session with group metrics."""
    strat = NeuralLocalSearchStrategy(
        policy_type="linear",
        features="@action_basic,peer_deltas,dist_others_barycentre",
        n_actions=8,
        n_agents=3,
    )
    session = RealNKSession(strategy=strat, n=8, k=1, n_agents=3, seed=42)

    diag = session.evaluate_current_state()
    assert diag["n_agents"] == 3
    assert len(diag["agents"]) == 3
    assert len(diag["agents"][0]["peer_distances"]) == 2  # 2 peers

    # Step all agents simultaneously
    step_all_res = session.step_all_agents_model()
    assert len(step_all_res) == 3
    assert session.step_count == 1
    assert len(session.trajectory_history) == 2


def test_cli_interpret_command() -> None:
    """Test neurols interpret CLI command in terminal mode."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "interpret",
            "--cli",
            "-m",
            "CUSTOM:linear:@action_basic:8",
            "-n",
            "8",
            "-k",
            "1",
            "--agents",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "NeuroLS Model Interpretability" in result.output
    assert "Model Input Features" in result.output
    assert "Default Inputs Score" in result.output
    assert "Real NK Landscape" in result.output
