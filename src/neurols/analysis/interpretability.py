"""Interpretability and diagnostic tools for Neural Local Search policies.

Provides utilities to understand what features (delta fitness, agent distance,
group consensus, stagnation) the neural network utilizes to choose its search transitions.
"""

from __future__ import annotations

from typing import Any

import torch

from neurols.core.problem import BatchedProblem
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.strategies.neural.policy import LinearScorer


def explain_decision(
    strategy: NeuralLocalSearchStrategy,
    states: torch.Tensor,
    problem: BatchedProblem,
    step: int = 0,
) -> dict[str, Any]:
    """Provide a transparent diagnostic breakdown of the strategy's decision on a given state."""
    deltas = problem.get_neighborhood_deltas(states)
    curr_fit = problem.evaluate(states)

    move_res = strategy.select_move(
        states=states,
        deltas=deltas,
        current_fitness=curr_fit,
        best_fitness=curr_fit,
        step=step,
    )

    chosen_idx = int(move_res.chosen_moves[0].item())
    probs = (
        move_res.probabilities[0].detach().cpu().numpy()
        if move_res.probabilities is not None
        else None
    )
    logits = move_res.logits[0].detach().cpu().numpy() if move_res.logits is not None else None

    feature_names = strategy.feature_config.feature_names
    chosen_features = (
        move_res.metadata["features"][0, chosen_idx].detach().cpu().numpy()
        if "features" in move_res.metadata
        else None
    )

    feature_breakdown = {}
    if chosen_features is not None:
        for name, val in zip(feature_names, chosen_features, strict=False):
            feature_breakdown[name] = float(val)

    return {
        "chosen_move": chosen_idx,
        "is_improving": bool(move_res.is_improving[0].item()),
        "chosen_delta": float(deltas[0, chosen_idx].item()) if chosen_idx >= 0 else 0.0,
        "features_of_chosen_move": feature_breakdown,
        "probabilities": probs.tolist() if probs is not None else None,
        "logits": logits.tolist() if logits is not None else None,
    }


def compute_feature_sensitivities(
    strategy: NeuralLocalSearchStrategy,
    states: torch.Tensor,
    problem: BatchedProblem,
) -> dict[str, float]:
    """Compute average gradient-based saliency / sensitivity of each input feature.

    Sensitivity is defined as the mean absolute gradient of the output logits with
    respect to each input feature channel: E[|d(logit)/d(feature)|].
    """
    deltas = problem.get_neighborhood_deltas(states)
    curr_fit = problem.evaluate(states)

    # If linear policy, weights directly give sensitivities
    if isinstance(strategy.policy, LinearScorer):
        w_dict = strategy.policy.get_feature_weights(strategy.feature_config.feature_names)
        return {k: abs(v) for k, v in w_dict.items()}

    # General gradient-based sensitivity
    features = strategy.extractor.extract(
        states=states,
        deltas=deltas,
        current_fitness=curr_fit,
        best_fitness=curr_fit,
        group_metrics=None,
        step=0,
        max_steps=100,
        stagnation_counter=torch.zeros(states.shape[0], dtype=torch.int64, device=states.device),
    )

    features.requires_grad_(True)
    logits = strategy.policy(features)
    # Sum of logits as scalar objective for gradient
    loss = logits.sum()
    loss.backward()

    grads = features.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()  # (F,)
    feat_names = strategy.feature_config.feature_names
    return {name: float(g) for name, g in zip(feat_names, grads, strict=False)}
