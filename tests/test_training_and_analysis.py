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
