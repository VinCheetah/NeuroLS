"""Tests for classical and neural strategies, determinism, and BatchedRunner."""

import torch

from neurols.core.group import GroupConfig
from neurols.core.runner import BatchedRunner, TraditionalRunner
from neurols.problems.nk_landscape import NKLandscape
from neurols.problems.onemax import OneMax
from neurols.strategies.classical import (
    HillClimber,
    RandomWalk,
    TabuSearch,
)
from neurols.strategies.neural import (
    NeuralLocalSearchStrategy,
)
from neurols.types import MoveSelectionMode


def test_hill_climber_onemax() -> None:
    problem = OneMax(n=20)
    strategy = HillClimber()
    runner = BatchedRunner(problem=problem, strategy=strategy, max_steps=30)
    traj = runner.run(n_instances=1, n_restarts=5, seed=42)

    # OneMax should be easily solved by HillClimber
    assert traj.average_best_fitness == 1.0
    assert traj.final_states.sum() == 5 * 20


def test_strategy_determinism_contract() -> None:
    problem = NKLandscape(n=15, k=3, seed=123)

    # 1. HillClimber is strictly deterministic
    hc1 = HillClimber()
    hc2 = HillClimber()
    init_state = problem.sample_initial_states((2,), generator=torch.Generator().manual_seed(99))

    runner1 = BatchedRunner(problem=problem, strategy=hc1, max_steps=15)
    runner2 = BatchedRunner(problem=problem, strategy=hc2, max_steps=15)

    traj1 = runner1.run(initial_states=init_state.clone())
    traj2 = runner2.run(initial_states=init_state.clone())
    torch.testing.assert_close(traj1.final_fitness, traj2.final_fitness)
    torch.testing.assert_close(traj1.final_states, traj2.final_states)

    # 2. RandomWalk with state_dependent_deterministic=True must be identical given same states!
    rw1 = RandomWalk(exploration_rate=0.5, state_dependent_deterministic=True)
    rw2 = RandomWalk(exploration_rate=0.5, state_dependent_deterministic=True)
    runner_rw1 = BatchedRunner(problem=problem, strategy=rw1, max_steps=10)
    runner_rw2 = BatchedRunner(problem=problem, strategy=rw2, max_steps=10)

    traj_rw1 = runner_rw1.run(initial_states=init_state.clone())
    traj_rw2 = runner_rw2.run(initial_states=init_state.clone())
    torch.testing.assert_close(traj_rw1.final_states, traj_rw2.final_states)


def test_tabu_search_escapes_local_optimum() -> None:
    problem = NKLandscape(n=16, k=4, seed=456)
    tabu = TabuSearch(tenure=5)
    runner = BatchedRunner(problem=problem, strategy=tabu, max_steps=40)
    traj = runner.run(n_restarts=4, seed=789)

    assert traj.steps_taken.min().item() == 40  # Tabu doesn't stop prematurely at local optima
    assert traj.best_overall_fitness >= traj.initial_fitness.max().item()


def test_multi_agent_information_sharing() -> None:
    problem = NKLandscape(n=20, k=3, seed=777)
    strategy = HillClimber()
    # 2 groups of 4 agents = total batch 8
    group_cfg = GroupConfig(
        n_agents=4, share_distance=True, share_scores=True, share_consensus=True
    )
    runner = BatchedRunner(problem=problem, strategy=strategy, max_steps=20, group_config=group_cfg)
    traj = runner.run(n_instances=1, n_restarts=2, seed=101)

    assert traj.batch_size == 8
    assert len(traj.diversity_history) > 0


def test_neural_strategy_architectures() -> None:
    problem = NKLandscape(n=12, k=2, seed=888)

    # Test Linear, MLP, and DeepSets policies
    for p_type in ["linear", "mlp", "deepsets"]:
        strat = NeuralLocalSearchStrategy(
            policy_type=p_type,  # type: ignore
            selection_mode=MoveSelectionMode.ARGMAX,
            max_steps=15,
        )
        runner = BatchedRunner(problem=problem, strategy=strat, max_steps=15)
        traj = runner.run(n_restarts=3, seed=42)
        assert traj.batch_size == 3
        assert traj.average_best_fitness >= 0.0

        # Verify weights export and import
        weights = strat.get_weights_vector()
        assert len(weights) > 0
        strat.set_weights_vector(weights + 0.01)


def test_traditional_runner() -> None:
    problem = OneMax(n=10)
    strategy = HillClimber()
    trad = TraditionalRunner(problem=problem, strategy=strategy)
    fit0 = trad.reset(seed=42)
    assert 0.0 <= fit0 <= 1.0

    for _ in range(15):
        record = trad.step()
        assert record.step >= 0
    assert trad.best_fitness >= fit0
