"""Publication-grade visualization utilities for Local Search and Neuroevolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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

# Publication-grade distinct color palette
PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


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


def plot_step_trajectories(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Step-by-Step Search Dynamics",
    show_diversity: bool = True,
) -> Path:
    """Plot detailed step-by-step evolution of both Best-so-far and Mean fitness across steps.

    Top Subplot: Monotonically increasing Best-so-far fitness found by the agents.
    Middle Subplot: Mean current fitness of active agents at each step.
    Bottom Subplot (optional): Hamming diversity among agents in the group.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_diversity = show_diversity and any(
        bool(traj.diversity_history) for traj in trajectories.values()
    )
    n_plots = 3 if has_diversity else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 3.5 * n_plots), dpi=150, sharex=True)
    if n_plots == 1:
        axes = [axes]

    ax_best = axes[0]
    ax_mean = axes[1]
    ax_div = axes[2] if has_diversity else None

    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]

        # 1. Best fitness curve
        if traj.best_fitness_history:
            steps_b = np.arange(len(traj.best_fitness_history))
            ax_best.plot(
                steps_b,
                traj.best_fitness_history,
                label=name,
                color=color,
                linewidth=2.2,
            )

        # 2. Mean fitness curve
        if traj.mean_fitness_history:
            steps_m = np.arange(len(traj.mean_fitness_history))
            ax_mean.plot(
                steps_m,
                traj.mean_fitness_history,
                label=name,
                color=color,
                linewidth=1.8,
                linestyle="--",
            )

        # 3. Diversity curve
        if ax_div is not None and traj.diversity_history:
            steps_d = np.arange(len(traj.diversity_history))
            ax_div.plot(steps_d, traj.diversity_history, label=name, color=color, linewidth=1.8)

    ax_best.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax_best.set_ylabel("Best-so-Far Fitness", fontsize=11)
    ax_best.grid(True, linestyle="--", alpha=0.5)
    ax_best.legend(frameon=True, fontsize=9, loc="lower right")

    ax_mean.set_ylabel("Mean Agent Fitness", fontsize=11)
    ax_mean.grid(True, linestyle="--", alpha=0.5)
    ax_mean.legend(frameon=True, fontsize=9, loc="lower right")

    if ax_div is not None:
        ax_div.set_ylabel("Hamming Diversity (bits)", fontsize=11)
        ax_div.set_xlabel("Search Steps", fontsize=11)
        ax_div.grid(True, linestyle="--", alpha=0.5)
        ax_div.legend(frameon=True, fontsize=9, loc="upper right")
    else:
        ax_mean.set_xlabel("Search Steps", fontsize=11)

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
    title: str = "CMA-ES Neuroevolution Training Progression",
) -> Path:
    """Plot generation-by-generation evolution of best/mean fitness, variance, and sigma."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gens = np.array([rec.generation for rec in history])
    best_fits = np.array([rec.best_fitness for rec in history])
    gen_bests = np.array([rec.gen_best_fitness for rec in history])
    mean_fits = np.array([rec.gen_mean_fitness for rec in history])
    std_fits = np.array([rec.gen_std_fitness for rec in history])
    sigmas = np.array([rec.sigma for rec in history])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 8.5), dpi=150, sharex=True)

    # 1. Fitness curves with generation std band
    ax1.plot(gens, best_fits, label="Overall Best Fitness", color="#1f77b4", linewidth=2.5)
    ax1.plot(
        gens, gen_bests, label="Gen Best Fitness", color="#2ca02c", linestyle=":", linewidth=1.8
    )
    ax1.plot(
        gens, mean_fits, label="Gen Mean Fitness", color="#ff7f0e", linestyle="--", linewidth=1.8
    )
    ax1.fill_between(
        gens,
        mean_fits - std_fits,
        mean_fits + std_fits,
        color="#ff7f0e",
        alpha=0.15,
        label="Gen Mean ± 1 Std",
    )
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Fitness (Higher is Better)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Fitness Improvement Deltas
    deltas = np.diff(best_fits, prepend=best_fits[0])
    bar_colors = ["#2ca02c" if d > 1e-6 else "#bdc3c7" for d in deltas]
    ax2.bar(gens, deltas, color=bar_colors, width=0.6, alpha=0.85)
    ax2.set_ylabel("Δ Best Fitness", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 3. Sigma step size
    ax3.plot(gens, sigmas, label="CMA-ES Step Size (Sigma)", color="#9467bd", linewidth=2.0)
    ax3.set_xlabel("Training Generation", fontsize=11)
    ax3.set_ylabel("Sigma (Step Size)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multi_training_convergence(
    histories: dict[str, list[GenerationRecord]],
    save_path: str | Path,
    title: str = "Multi-Model Neuroevolution Training Comparison",
) -> Path:
    """Plot and compare the CMA-ES training curves of multiple neural models."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), dpi=150, sharex=True)

    for i, (name, history) in enumerate(histories.items()):
        if not history:
            continue
        color = PALETTE[i % len(PALETTE)]
        gens = [rec.generation for rec in history]
        bests = [rec.best_fitness for rec in history]
        means = [rec.gen_mean_fitness for rec in history]

        ax1.plot(gens, bests, label=name, color=color, linewidth=2.2)
        ax2.plot(gens, means, label=name, color=color, linestyle="--", linewidth=1.8)

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Overall Best Fitness", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_xlabel("Training Generation", fontsize=11)
    ax2.set_ylabel("Generation Mean Fitness", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

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


def plot_fitness_vs_agents(
    records: list[Any],
    save_path: str | Path,
    title: str = "Team Best Fitness vs. Agent Count (M)",
) -> Path:
    """Plot Best Discovered Fitness as a function of Agent Count (e.g. 1, 4, 10).

    Correctly handles group-aware vs group-agnostic strategies:
    - Group-agnostic strategies (e.g. HillClimber, TabuSearch, NN_DeltaOnly) are plotted
      as a single clean reference baseline curve (dotted line).
    - Group-aware strategies (e.g. PeerInvariant, InvariantNNet) are plotted with
      solid lines (Sharing) and dashed lines (No Sharing) to highlight the collaboration gain.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    # Group records by model
    model_names = sorted(list({r.model_name for r in records}))

    for i, model in enumerate(model_names):
        color = PALETTE[i % len(PALETTE)]
        mod_records = [r for r in records if r.model_name == model]

        # Check if model is group-aware
        is_group_aware = any(getattr(r, "is_group_aware", True) for r in mod_records)

        if not is_group_aware:
            # Single reference baseline curve: ignore sharing distinction
            sorted_recs = sorted(mod_records, key=lambda r: r.n_agents)
            # Remove any duplicates on n_agents
            seen_m: set[int] = set()
            unique_recs = []
            for r in sorted_recs:
                if r.n_agents not in seen_m:
                    seen_m.add(r.n_agents)
                    unique_recs.append(r)

            x = [r.n_agents for r in unique_recs]
            y = [r.mean_team_best_fitness for r in unique_recs]
            ax.plot(
                x,
                y,
                label=f"{model} (Agnostic)",
                color=color,
                marker="s",
                linestyle=":",
                linewidth=2.0,
                alpha=0.85,
            )
        else:
            # Group-aware: Plot Sharing vs No Sharing
            sharing_recs = sorted(
                [r for r in mod_records if r.share_info],
                key=lambda r: r.n_agents,
            )
            if sharing_recs:
                x_s = [r.n_agents for r in sharing_recs]
                y_s = [r.mean_team_best_fitness for r in sharing_recs]
                ax.plot(
                    x_s,
                    y_s,
                    label=f"{model} (Sharing)",
                    color=color,
                    marker="o",
                    linewidth=2.2,
                )

            noshar_recs = sorted(
                [r for r in mod_records if not r.share_info or r.n_agents == 1],
                key=lambda r: r.n_agents,
            )
            # Deduplicate by n_agents
            seen_m_ns: set[int] = set()
            unique_ns = []
            for r in noshar_recs:
                if r.n_agents not in seen_m_ns:
                    seen_m_ns.add(r.n_agents)
                    unique_ns.append(r)

            if len(unique_ns) > 1:
                x_ns = [r.n_agents for r in unique_ns]
                y_ns = [r.mean_team_best_fitness for r in unique_ns]
                ax.plot(
                    x_ns,
                    y_ns,
                    label=f"{model} (No Sharing)",
                    color=color,
                    linestyle="--",
                    marker="x",
                    linewidth=1.8,
                    alpha=0.85,
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Agents in Group (M)", fontsize=11)
    ax.set_ylabel("Mean Team Best Fitness", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sharing_gain_bars(
    records: list[Any],
    save_path: str | Path,
    title: str = "Collaboration Gain from Information Sharing",
) -> Path:
    """Plot bar chart of empirical performance gain: Δ = Fitness(Sharing) - Fitness(No Sharing)."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Filter group-aware models with M > 1
    models = sorted(
        list(
            {r.model_name for r in records if getattr(r, "is_group_aware", True) and r.n_agents > 1}
        )
    )
    agent_counts = sorted(list({r.n_agents for r in records if r.n_agents > 1}))

    if not models or not agent_counts:
        # Create empty placeholder figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No multi-agent sharing pairs to compare (M > 1)",
            ha="center",
            va="center",
        )
        fig.savefig(path)
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    width = 0.8 / max(len(models), 1)
    x_indices = np.arange(len(agent_counts))

    for i, model in enumerate(models):
        gains: list[float] = []
        for m in agent_counts:
            r_s = next(
                (r for r in records if r.model_name == model and r.n_agents == m and r.share_info),
                None,
            )
            r_ns = next(
                (
                    r
                    for r in records
                    if r.model_name == model and r.n_agents == m and not r.share_info
                ),
                None,
            )
            if r_s and r_ns:
                gain = r_s.mean_team_best_fitness - r_ns.mean_team_best_fitness
            else:
                gain = 0.0
            gains.append(gain)

        offset = (i - len(models) / 2 + 0.5) * width
        color = PALETTE[i % len(PALETTE)]
        ax.bar(x_indices + offset, gains, width=width, label=model, color=color, alpha=0.85)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Agents (M)", fontsize=11)
    ax.set_ylabel("Fitness Gain (Sharing - No Sharing)", fontsize=11)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"M={m}" for m in agent_counts])
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_benchmark_dashboard(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "NeuroLS Benchmark Executive Dashboard",
) -> Path:
    """Generate a comprehensive 4-panel publication-grade dashboard for benchmark comparisons.

    - Panel 1: Step Convergence (Best-so-far fitness curves over steps)
    - Panel 2: Mean Population Fitness trajectories over steps
    - Panel 3: Final Fitness Distribution (Boxplots)
    - Panel 4: Diversity Evolution / Step Efficiency
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Panel 1: Best-so-far Convergence
    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        if traj.best_fitness_history:
            steps = np.arange(len(traj.best_fitness_history))
            ax1.plot(steps, traj.best_fitness_history, label=name, color=color, linewidth=2.0)
    ax1.set_title("A. Best-so-Far Fitness Trajectory", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Steps", fontsize=10)
    ax1.set_ylabel("Best Fitness", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 2: Mean Fitness Dynamics
    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        if traj.mean_fitness_history:
            steps = np.arange(len(traj.mean_fitness_history))
            ax2.plot(
                steps,
                traj.mean_fitness_history,
                label=name,
                color=color,
                linewidth=1.8,
                linestyle="--",
            )
    ax2.set_title("B. Population Mean Fitness Dynamics", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Mean Fitness", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    # Panel 3: Final Boxplots
    names = list(trajectories.keys())
    data = [traj.best_fitness.detach().cpu().numpy() for traj in trajectories.values()]
    bplot = ax3.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bplot["boxes"], PALETTE[: len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title("C. Final Fitness Distribution Across Restarts", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Final Best Fitness", fontsize=10)
    ax3.tick_params(axis="x", rotation=25, labelsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 4: Diversity Evolution
    any_diversity = False
    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        if traj.diversity_history:
            any_diversity = True
            steps = np.arange(len(traj.diversity_history))
            ax4.plot(steps, traj.diversity_history, label=name, color=color, linewidth=1.8)

    if any_diversity:
        ax4.set_title("D. Multi-Agent Group Diversity Evolution", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Steps", fontsize=10)
        ax4.set_ylabel("Mean Hamming Distance (bits)", fontsize=10)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(fontsize=8, loc="upper right")
    else:
        # Fallback to evaluations bar chart
        evals = [traj.total_evaluations for traj in trajectories.values()]
        ax4.barh(names, evals, color=PALETTE[: len(names)], alpha=0.8)
        ax4.set_title("D. Computational Evaluations", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Total Evaluations", fontsize=10)
        ax4.tick_params(axis="y", labelsize=8)
        ax4.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
