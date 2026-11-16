"""Publication-grade visualization utilities for Local Search and Neuroevolution."""

from __future__ import annotations

import os
from pathlib import Path

# Ensure writable matplotlib config in sandboxed / container environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from neurols.core.trajectory import Trajectory
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.strategies.neural.policy import LinearScorer
from neurols.training.base import GenerationRecord

# Color palette for clear multi-method comparison
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def plot_convergence_curves(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Local Search Convergence Comparison",
) -> Path:
    """Plot mean fitness trajectories across search steps for each strategy."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        steps = np.arange(len(traj.mean_fitness_history))
        ax.plot(steps, traj.mean_fitness_history, label=name, color=color, linewidth=2)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Steps", fontsize=11)
    ax.set_ylabel("Mean Population Fitness", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_final_fitness_boxplots(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Final Best Fitness Distribution Across Restarts",
) -> Path:
    """Plot boxplots comparing the distribution of best fitness found across runs."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    names: list[str] = []
    data: list[np.ndarray] = []

    for name, traj in trajectories.items():
        names.append(name)
        data.append(traj.best_fitness.detach().cpu().numpy())

    bplot = ax.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )

    for patch, color in zip(bplot["boxes"], PALETTE[: len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Best Discovered Fitness", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=20, ha="right", fontsize=10)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multiagent_diversity(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Multi-Agent Population Diversity",
) -> Path:
    """Plot multi-agent diversity (mean pairwise Hamming distance) evolution."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for i, (name, traj) in enumerate(trajectories.items()):
        if traj.diversity_history:
            color = PALETTE[i % len(PALETTE)]
            steps = np.arange(len(traj.diversity_history))
            ax.plot(steps, traj.diversity_history, label=name, color=color, linewidth=2)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Steps", fontsize=11)
    ax.set_ylabel("Mean Pairwise Hamming Distance", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training_convergence(
    history: list[GenerationRecord],
    save_path: str | Path,
    title: str = "CMA-ES Neuroevolution Convergence",
) -> Path:
    """Plot generation-by-generation evolution of best/mean fitness and sigma."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gens = [rec.generation for rec in history]
    best_fits = [rec.best_fitness for rec in history]
    mean_fits = [rec.gen_mean_fitness for rec in history]
    sigmas = [rec.sigma for rec in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True)

    # Fitness curve
    ax1.plot(gens, best_fits, label="Overall Best Fitness", color="#1f77b4", linewidth=2.5)
    ax1.plot(gens, mean_fits, label="Generation Mean Fitness", color="#ff7f0e", linestyle="--")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Fitness (Higher is Better)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Sigma curve
    ax2.plot(gens, sigmas, label="CMA-ES Step Size (Sigma)", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Sigma", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_neural_feature_importance(
    strategy: NeuralLocalSearchStrategy,
    save_path: str | Path,
    title: str = "Learned Policy Feature Weights / Attributions",
) -> Path:
    """Plot bar chart showing the weights or relative attributions learned by the NN policy."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    feat_names = strategy.feature_config.feature_names

    if isinstance(strategy.policy, LinearScorer):
        weights_dict = strategy.policy.get_feature_weights(feat_names)
        names = list(weights_dict.keys())
        values = list(weights_dict.values())
    else:
        # For non-linear networks, use parameter norm or magnitude proxy
        names = feat_names
        # Approximate with first layer weight norms per feature input
        first_layer = list(strategy.policy.parameters())[0]
        values = first_layer.abs().mean(dim=0).detach().cpu().numpy().tolist()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    ax.barh(names, values, color=colors, alpha=0.8)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Learned Weight / Saliency (Positive = Favors Move)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
