"""Unit tests for problem representations and NK Landscape."""

import pytest
import torch

from neurols.problems.nk_landscape import MultiNKLandscape, NKLandscape
from neurols.problems.onemax import OneMax
from neurols.types import NeighborhoodType


def test_onemax_evaluation() -> None:
    problem = OneMax(n=10)
    # State with 5 ones and 5 zeros
    state = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.int8)
    fit = problem.evaluate(state)
    assert pytest.approx(fit.item()) == 0.5

    deltas = problem.get_neighborhood_deltas(state)
    assert deltas.shape == (1, 10)
    # Flipping 1 should give -0.1
    assert pytest.approx(deltas[0, 0].item()) == -0.1
    # Flipping 0 should give +0.1
    assert pytest.approx(deltas[0, 5].item()) == 0.1

    # Apply move at index 5
    new_state = problem.apply_moves(state, torch.tensor([5]))
    assert new_state[0, 5].item() == 1
    new_fit = problem.evaluate(new_state)
    assert pytest.approx(new_fit.item()) == 0.6


def test_nk_landscape_evaluation_adjacent() -> None:
    n, k = 10, 2
    landscape = NKLandscape(n=n, k=k, interaction_type=NeighborhoodType.ADJACENT, seed=42)

    # Sample batch
    states = landscape.sample_initial_states((4, 3))
    assert states.shape == (4, 3, n)
    assert states.dtype == torch.int8

    fitness = landscape.evaluate(states)
    assert fitness.shape == (4, 3)
    assert (fitness >= 0.0).all() and (fitness <= 1.0).all()


def test_nk_landscape_neighborhood_deltas() -> None:
    n, k = 12, 3
    landscape = NKLandscape(n=n, k=k, interaction_type=NeighborhoodType.RANDOM, seed=123)

    states = landscape.sample_initial_states((5,))
    base_fitness = landscape.evaluate(states)

    deltas = landscape.get_neighborhood_deltas(states)
    assert deltas.shape == (5, n)

    # Verify deltas match manual 1-bit flip evaluations
    for move_idx in range(n):
        moves = torch.full((5,), move_idx, dtype=torch.long)
        flipped_states = landscape.apply_moves(states, moves)
        new_fitness = landscape.evaluate(flipped_states)
        expected_deltas = new_fitness - base_fitness
        torch.testing.assert_close(deltas[:, move_idx], expected_deltas, atol=1e-5, rtol=1e-5)


def test_multi_nk_landscape() -> None:
    n_inst, n, k = 3, 10, 2
    multi = MultiNKLandscape(n_instances=n_inst, n=n, k=k, base_seed=99)
    states = multi.sample_initial_states((n_inst, 4))
    assert states.shape == (n_inst, 4, n)

    fit = multi.evaluate(states)
    assert fit.shape == (n_inst, 4)

    deltas = multi.get_neighborhood_deltas(states)
    assert deltas.shape == (n_inst, 4, n)
