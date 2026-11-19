"""Tests for CMA-ES neuroevolution training, metrics, and plotting."""

import tempfile
from pathlib import Path

import numpy as np

from neurols.analysis.experiment import ExperimentLogger
from neurols.analysis.interpretability import compute_feature_sensitivities, explain_decision
from neurols.analysis.plotting import (
    plot_convergence_curves,
    plot_final_fitness_boxplots,
)
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.classical import HillClimber, TabuSearch
from neurols.strategies.neural import NeuralLocalSearchStrategy
from neurols.training.cma_es import CMAESOptimizer


def test_cma_es_optimization_run() -> None:
    problem = NKLandscape(n=10, k=2, seed=123)
    strategy = NeuralLocalSearchStrategy(
        policy_type="linear",
        max_steps=20,
    )

    optimizer = CMAESOptimizer(
        strategy=strategy,
        problem=problem,
        eval_restarts=4,
        eval_steps=15,
        population_size=6,
        sigma0=0.3,
        seed=42,
    )

    result = optimizer.train(n_generations=2)
    assert result.total_generations == 2
    assert len(result.history) == 2
    assert len(result.best_weights) == strategy.policy.num_parameters
    assert result.best_fitness >= 0.0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "test_result.json"
        result.save(save_file)
        loaded = result.load(save_file)
        assert loaded.best_fitness == result.best_fitness
        np.testing.assert_allclose(loaded.best_weights, result.best_weights)


def test_experiment_comparison_and_plotting() -> None:
    problem = NKLandscape(n=12, k=2, seed=456)
    hc = HillClimber()
    tabu = TabuSearch(tenure=4)

    logger = ExperimentLogger()
    comp_result = logger.run_comparison(
        experiment_name="test_cmp",
        problem=problem,
        strategies=[hc, tabu],
        n_restarts=4,
        max_steps=15,
    )

    assert len(comp_result.trajectories) == 2
    assert len(comp_result.summary_table) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1 = plot_convergence_curves(comp_result.trajectories, tmp_path / "conv.png")
        p2 = plot_final_fitness_boxplots(comp_result.trajectories, tmp_path / "box.png")
        assert p1.exists()
        assert p2.exists()


def test_interpretability_and_saliency() -> None:
    problem = NKLandscape(n=10, k=2, seed=789)
    strat = NeuralLocalSearchStrategy(policy_type="linear")
    states = problem.sample_initial_states((2,))

    exp = explain_decision(strat, states, problem)
    assert "chosen_move" in exp
    assert "features_of_chosen_move" in exp
    assert exp["chosen_move"] >= 0

    sens = compute_feature_sensitivities(strat, states, problem)
    assert len(sens) == strat.feature_config.num_features
    assert all(v >= 0 for v in sens.values())


def test_model_registry_reproducibility(tmp_path) -> None:
    """Test saving model with full metadata and loading it back ready to run."""
    from neurols.training.base import GenerationRecord, TrainResult
    from neurols.training.registry import (
        list_saved_models,
        load_model_checkpoint,
        save_model_checkpoint,
    )

    strat = NeuralLocalSearchStrategy(
        policy_type="mlp",
        eval_mode="per_action",
        features="@action_basic",
        n_actions=16,
    )
    weights = np.random.randn(strat.policy.num_parameters).astype(np.float32)
    strat.set_weights_vector(weights)

    train_res = TrainResult(
        best_weights=weights,
        best_fitness=0.8123,
        history=[
            GenerationRecord(
                generation=0,
                best_fitness=0.8123,
                gen_best_fitness=0.8123,
                gen_mean_fitness=0.71,
                gen_std_fitness=0.05,
                sigma=0.5,
                elapsed_time=0.2,
            )
        ],
        total_generations=1,
        total_evaluations=10,
        total_time_seconds=0.2,
    )

    saved_file = save_model_checkpoint(
        strategy=strat,
        train_result=train_res,
        experiment_name="repro_exp",
        model_name="repro_model",
        problem_config={"n": 16, "k": 2},
        base_dir=tmp_path,
    )
    assert saved_file.exists()

    # Verify listing
    models = list_saved_models(base_dir=tmp_path)
    assert len(models) == 1
    assert models[0].model_name == "repro_model"
    assert models[0].experiment_name == "repro_exp"

    # Load back
    loaded_strat, manifest = load_model_checkpoint("repro_exp:repro_model", base_dir=tmp_path)
    assert loaded_strat.name == "repro_model"
    assert loaded_strat.policy_type == "mlp"
    assert loaded_strat.eval_mode.value == "per_action"
    np.testing.assert_allclose(loaded_strat.get_weights_vector(), weights, atol=1e-5)


def test_plot_selector_and_projections() -> None:
    """Test plot keyword resolution and 2D projections (PCA and MDS)."""
    from neurols.analysis.plotting import (
        project_2d_mds,
        project_2d_pca,
        resolve_plot_names,
    )

    # 1. Keyword resolution
    explore_plots = resolve_plot_names("@explore")
    assert "exploration_dashboard" in explore_plots
    assert "trajectory_2d" in explore_plots
    assert "bitstring_evolution" in explore_plots

    bench_plots = resolve_plot_names("@benchmark")
    assert "dashboard" in bench_plots
    assert "convergence" in bench_plots

    custom_plots = resolve_plot_names("trajectory_2d,action_ranks,-move_deltas", context="explore")
    assert custom_plots == ["trajectory_2d", "action_ranks"]

    # 2. 2D Projections
    T, N = 20, 16
    bitstrings = np.random.randint(0, 2, size=(T, N))
    pca_coords = project_2d_pca(bitstrings)
    assert pca_coords.shape == (T, 2)

    mds_coords = project_2d_mds(bitstrings)
    assert mds_coords.shape == (T, 2)
