"""Publication-grade visualization utilities for Local Search and Neuroevolution."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
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


# ============================================================================
# Dimensionality Reduction Utilities for State Trajectories
# ============================================================================


def project_2d_pca(states: np.ndarray) -> np.ndarray:
    """Project high-dimensional bitstrings / states of shape (T, N) to 2D using SVD-based PCA."""
    if len(states) < 2:
        return np.zeros((len(states), 2))

    arr = states.astype(np.float64)
    centered = arr - arr.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.zeros((len(states), 2))

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 2:
        coords = np.zeros((len(states), 2))
        coords[:, 0] = centered @ vt[0]
        return coords

    return centered @ vt[:2].T


def project_2d_mds(states: np.ndarray) -> np.ndarray:
    """Project high-dimensional binary states of shape (T, N) to 2D using Classical MDS on Hamming distances."""
    t = len(states)
    if t < 2:
        return np.zeros((t, 2))

    arr = states.astype(np.float64)
    # Pairwise Hamming distances
    diff = np.abs(arr[:, None, :] - arr[None, :, :])
    d = np.sum(diff, axis=-1)

    # Double centering: B = -0.5 * H * D^2 * H
    h = np.eye(t) - np.ones((t, t)) / t
    b = -0.5 * (h @ (d**2) @ h)

    eigvals, eigvecs = np.linalg.eigh(b)
    # Sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    top_vals = np.maximum(eigvals[idx[:2]], 0.0)
    top_vecs = eigvecs[:, idx[:2]]

    coords = top_vecs * np.sqrt(top_vals)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((t, 2 - coords.shape[1]))])
    return coords


# ============================================================================
# Exploration & Diagnostic Visualizations (Single Strategy Deep Dive)
# ============================================================================


def plot_fitness_evolution(
    trajectory: Trajectory,
    save_path: str | Path,
    title: str = "Search Trajectory Fitness Evolution",
    agent_idx: int = 0,
) -> Path:
    """Plot step-by-step current and best-so-far fitness for a single representative agent run."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    curr_fit, best_fit = trajectory.get_agent_fitness_history(agent_idx=agent_idx)
    steps = np.arange(len(curr_fit))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.plot(
        steps, best_fit, label="Best-so-Far Fitness", color="#2ca02c", linewidth=2.4, linestyle="-"
    )
    ax.plot(
        steps,
        curr_fit,
        label="Current Fitness",
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
        linestyle="-",
    )

    # Highlight improvement points
    if len(best_fit) > 1:
        improved_mask = np.diff(best_fit, prepend=best_fit[0]) > 1e-6
        if improved_mask.any():
            ax.scatter(
                steps[improved_mask],
                best_fit[improved_mask],
                color="#ff7f0e",
                s=40,
                zorder=5,
                label="New Best Found",
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Search Steps", fontsize=10)
    ax.set_ylabel("Fitness Score", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bitstring_evolution(
    states: np.ndarray,
    chosen_moves: np.ndarray | None,
    save_path: str | Path,
    title: str = "Bitstring Configuration & Loci Flips Over Time",
) -> Path:
    """Plot a heatmap matrix of bit values {0, 1} across search steps, highlighting flipped loci."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t, n = states.shape
    fig, (ax_heat, ax_hist) = plt.subplots(
        1, 2, figsize=(10, max(4.5, min(8.0, 0.15 * t))), dpi=150, width_ratios=[4, 1], sharey=True
    )

    # Heatmap of bitstring values (0 = dark blue, 1 = gold/yellow)
    ax_heat.imshow(states, aspect="auto", cmap="Blues_r", origin="upper", interpolation="nearest")
    ax_heat.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax_heat.set_xlabel("Bit Locus (0 to N-1)", fontsize=10)
    ax_heat.set_ylabel("Search Step (t)", fontsize=10)

    # Overlay red markers on loci that flipped
    flip_counts = np.zeros(n, dtype=int)
    if chosen_moves is not None and len(chosen_moves) > 0:
        for step_idx, locus in enumerate(chosen_moves):
            if 0 <= locus < n and step_idx < t:
                ax_heat.scatter([locus], [step_idx], color="#d62728", s=18, marker="x", zorder=5)
                flip_counts[locus] += 1
    elif t > 1:
        flips = states[1:] != states[:-1]
        for step_idx in range(t - 1):
            for locus in np.where(flips[step_idx])[0]:
                ax_heat.scatter(
                    [locus], [step_idx + 1], color="#d62728", s=18, marker="x", zorder=5
                )
                flip_counts[locus] += 1

    # Right subplot: Total flip frequency per locus
    ax_hist.barh(np.arange(n), flip_counts, color="#1f77b4", alpha=0.8)
    ax_hist.set_title("Flip Count", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Flips", fontsize=9)
    ax_hist.grid(True, linestyle="--", alpha=0.4, axis="x")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_search_trajectory_2d(
    states: np.ndarray,
    fitness: np.ndarray,
    save_path: str | Path,
    method: str = "pca",
    title: str = "Search Trajectory in 2D Latent Space",
) -> Path:
    """Project the trajectory of visited solutions into 2D (PCA or MDS) with fitness color coding."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    method_clean = method.lower().strip()
    if method_clean == "mds":
        coords = project_2d_mds(states)
        method_label = "Classical MDS (Hamming Distance)"
    else:
        coords = project_2d_pca(states)
        method_label = "Principal Component Analysis (PCA)"

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)
    t = len(coords)

    # Plot continuous trajectory path
    if t > 1:
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color="#7f7f7f",
            linewidth=1.2,
            linestyle="-",
            alpha=0.6,
            zorder=2,
        )
        # Add light directional arrows
        skip = max(1, t // 15)
        for i in range(0, t - 1, skip):
            dx = coords[i + 1, 0] - coords[i, 0]
            dy = coords[i + 1, 1] - coords[i, 1]
            if abs(dx) + abs(dy) > 1e-4:
                ax.annotate(
                    "",
                    xy=(coords[i + 1, 0], coords[i + 1, 1]),
                    xytext=(coords[i, 0], coords[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0, alpha=0.7),
                    zorder=3,
                )

    # Scatter points colored by fitness
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=fitness,
        cmap="viridis",
        s=36,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Fitness Score", fontsize=10)

    # Mark Start, End, and Best
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        color="#2ca02c",
        s=130,
        marker="o",
        edgecolor="black",
        label="Start (t=0)",
        zorder=6,
    )
    ax.scatter(
        coords[-1, 0],
        coords[-1, 1],
        color="#d62728",
        s=130,
        marker="s",
        edgecolor="black",
        label=f"End (t={t - 1})",
        zorder=6,
    )

    best_idx = int(np.argmax(fitness))
    ax.scatter(
        coords[best_idx, 0],
        coords[best_idx, 1],
        color="#ffd700",
        s=180,
        marker="*",
        edgecolor="black",
        label=f"Best Found ({fitness[best_idx]:.4f})",
        zorder=7,
    )

    ax.set_title(f"{title}\n({method_label})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Component 1", fontsize=10)
    ax.set_ylabel("Component 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_action_ranks(
    ranks: np.ndarray,
    save_path: str | Path,
    title: str = "Neighbor Action Selection Rank Dynamics",
) -> Path:
    """Plot the distribution and step timeline of chosen action rank among all neighbors.

    Rank 0 = Best improving move in the 1-flip neighborhood (steepest ascent).
    Rank > 0 = Suboptimal / exploratory move choice.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, width_ratios=[1, 1.4])

    if len(ranks) == 0:
        ax1.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        ax2.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        fig.savefig(path)
        plt.close(fig)
        return path

    # 1. Bar Chart: Breakdown of ranks chosen
    rank0_count = int(np.sum(ranks == 0))
    rank1_count = int(np.sum(ranks == 1))
    rank2_count = int(np.sum(ranks == 2))
    rank_ge3_count = int(np.sum(ranks >= 3))
    total_moves = len(ranks)

    categories = [
        "Rank 0\n(Steepest)",
        "Rank 1\n(2nd Best)",
        "Rank 2\n(3rd Best)",
        "Rank ≥ 3\n(Exploratory)",
    ]
    counts = [rank0_count, rank1_count, rank2_count, rank_ge3_count]
    pcts = [100.0 * c / max(1, total_moves) for c in counts]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    bars = ax1.bar(categories, pcts, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for bar, pct, cnt in zip(bars, pcts, counts, strict=False):
        if cnt > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f}%\n({cnt})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    ax1.set_ylim(0, max(pcts, default=10) * 1.25 + 5)
    ax1.set_title("A. Action Rank Distribution", fontsize=11, fontweight="bold")
    ax1.set_ylabel("% of Chosen Moves", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # 2. Timeline: Rank chosen at each step
    steps = np.arange(len(ranks))
    ax2.plot(steps, ranks, color="#1f77b4", alpha=0.7, linewidth=1.5)
    ax2.scatter(steps, ranks, color="#1f77b4", s=18, alpha=0.8)
    ax2.axhline(0, color="#2ca02c", linestyle="--", linewidth=1.2, label="Rank 0 (Greedy Optimum)")

    ax2.set_title("B. Chosen Rank Across Steps", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Search Step", fontsize=10)
    ax2.set_ylabel("Rank in Neighborhood (0 = Best)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_move_deltas(
    deltas: np.ndarray,
    save_path: str | Path,
    title: str = "Step Fitness Deltas (Exploration vs Exploitation)",
) -> Path:
    """Plot delta fitness of chosen moves over steps, distinguishing improving vs deteriorating moves."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    steps = np.arange(len(deltas))

    colors = ["#2ca02c" if d > 1e-6 else ("#d62728" if d < -1e-6 else "#7f7f7f") for d in deltas]
    ax.bar(steps, deltas, color=colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)

    improving = int(np.sum(deltas > 1e-6))
    neutral = int(np.sum(np.abs(deltas) <= 1e-6))
    worsening = int(np.sum(deltas < -1e-6))

    ax.set_title(
        f"{title}\n(Improving: {improving} | Neutral: {neutral} | Worsening: {worsening})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Search Step", fontsize=10)
    ax.set_ylabel("Fitness Change (Δf)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_exploration_dashboard(
    trajectory: Trajectory,
    save_path: str | Path,
    agent_idx: int = 0,
    method: str = "pca",
    title: str = "Strategy In-Depth Exploration Dashboard",
) -> Path:
    """Generate a multi-panel visual diagnostic dashboard dissecting strategy behavior."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    states = trajectory.get_states_matrix(agent_idx=agent_idx)
    curr_fit, best_fit = trajectory.get_agent_fitness_history(agent_idx=agent_idx)
    ranks = trajectory.get_action_ranks(agent_idx=agent_idx)
    deltas = trajectory.get_chosen_deltas(agent_idx=agent_idx)
    chosen_moves = trajectory.get_chosen_moves(agent_idx=agent_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # 1. Fitness Evolution
    ax1 = axes[0, 0]
    steps_fit = np.arange(len(curr_fit))
    ax1.plot(steps_fit, best_fit, label="Best Found", color="#2ca02c", linewidth=2.2, linestyle="-")
    ax1.plot(
        steps_fit, curr_fit, label="Current State", color="#1f77b4", linestyle="-", linewidth=1.8
    )
    ax1.set_title("A. Fitness Evolution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Steps", fontsize=9)
    ax1.set_ylabel("Fitness", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)

    # 2. 2D Search Trajectory
    ax2 = axes[0, 1]
    coords = project_2d_pca(states) if method.lower() != "mds" else project_2d_mds(states)
    if len(coords) > 1:
        ax2.plot(coords[:, 0], coords[:, 1], color="#888888", linewidth=1.0, alpha=0.6)
    sc = ax2.scatter(
        coords[:, 0], coords[:, 1], c=curr_fit[: len(coords)], cmap="viridis", s=28, alpha=0.9
    )
    ax2.scatter(
        coords[0, 0], coords[0, 1], color="#2ca02c", s=100, marker="o", label="Start", zorder=5
    )
    ax2.scatter(
        coords[-1, 0], coords[-1, 1], color="#d62728", s=100, marker="s", label="End", zorder=5
    )
    best_step = int(np.argmax(curr_fit[: len(coords)]))
    ax2.scatter(
        coords[best_step, 0],
        coords[best_step, 1],
        color="#ffd700",
        s=140,
        marker="*",
        label="Best",
        zorder=6,
    )
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Fitness", fontsize=8)
    ax2.set_title(f"B. 2D Trajectory ({method.upper()})", fontsize=11, fontweight="bold")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Bitstring Heatmap
    ax3 = axes[1, 0]
    ax3.imshow(states, aspect="auto", cmap="Blues_r", origin="upper", interpolation="nearest")
    if len(chosen_moves) > 0:
        for s_idx, m_idx in enumerate(chosen_moves):
            if 0 <= m_idx < states.shape[1] and s_idx < len(states):
                ax3.scatter([m_idx], [s_idx], color="#d62728", s=12, marker="x")
    ax3.set_title("C. Bitstring Evolution & Loci Flips (Red X)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Bit Locus", fontsize=9)
    ax3.set_ylabel("Step", fontsize=9)

    # 4. Action Rank Timeline & Deltas
    ax4 = axes[1, 1]
    if len(ranks) > 0:
        steps_r = np.arange(len(ranks))
        ax4.plot(steps_r, ranks, color="#1f77b4", linewidth=1.6, label="Chosen Rank")
        ax4.axhline(0, color="#2ca02c", linestyle="--", linewidth=1.2, label="Rank 0 (Best)")
        ax4.set_title("D. Chosen Action Rank (0 = Best Neighbor)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Step", fontsize=9)
        ax4.set_ylabel("Rank in Neighborhood", fontsize=9)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", fontsize=8)
    elif len(deltas) > 0:
        steps_d = np.arange(len(deltas))
        ax4.bar(steps_d, deltas, color="#2ca02c", alpha=0.8)
        ax4.set_title("D. Step Fitness Changes (Δf)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Step", fontsize=9)
        ax4.set_ylabel("Δf", fontsize=9)
        ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Benchmark & Comparative Visualizations
# ============================================================================


def plot_convergence_curves(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Local Search Convergence Comparison",
) -> Path:
    """Plot mean fitness trajectories across search steps with shaded variance bands (±1 std)."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        if (
            traj.step_history
            and len(traj.step_history) > 0
            and getattr(traj.step_history[0], "current_fitness", None) is not None
            and traj.step_history[0].current_fitness.ndim > 0
        ):
            step_means = np.array([rec.current_fitness.mean().item() for rec in traj.step_history])
            step_stds = np.array([rec.current_fitness.std().item() for rec in traj.step_history])
            steps = np.arange(len(step_means))
            ax.plot(steps, step_means, label=name, color=color, linewidth=2.2, linestyle="-")
            ax.fill_between(
                steps,
                step_means - step_stds,
                step_means + step_stds,
                color=color,
                alpha=0.15,
            )
        else:
            steps = np.arange(len(traj.mean_fitness_history))
            ax.plot(
                steps,
                traj.mean_fitness_history,
                label=name,
                color=color,
                linewidth=2.2,
                linestyle="-",
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Steps", fontsize=11)
    ax.set_ylabel("Mean Population Fitness (±1 Std)", fontsize=11)
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
    """Plot detailed step-by-step evolution of both Best-so-far and Mean fitness across steps."""
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
                linestyle="-",
            )

        # 2. Mean fitness curve
        if traj.mean_fitness_history:
            steps_m = np.arange(len(traj.mean_fitness_history))
            ax_mean.plot(
                steps_m,
                traj.mean_fitness_history,
                label=name,
                color=color,
                linewidth=2.0,
                linestyle="-",
            )

        # 3. Diversity curve (if applicable)
        if ax_div is not None and traj.diversity_history:
            steps_d = np.arange(len(traj.diversity_history))
            ax_div.plot(
                steps_d,
                traj.diversity_history,
                label=name,
                color=color,
                linewidth=2.0,
                linestyle="-",
            )

    ax_best.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax_best.set_ylabel("Best Fitness Found", fontsize=11)
    ax_best.grid(True, linestyle="--", alpha=0.5)
    ax_best.legend(frameon=True, fontsize=9, loc="lower right")

    ax_mean.set_ylabel("Mean Active Fitness", fontsize=11)
    ax_mean.grid(True, linestyle="--", alpha=0.5)
    ax_mean.legend(frameon=True, fontsize=9, loc="lower right")

    if ax_div is not None:
        ax_div.set_ylabel("Hamming Diversity", fontsize=11)
        ax_div.grid(True, linestyle="--", alpha=0.5)
        ax_div.legend(frameon=True, fontsize=9, loc="upper right")
        ax_div.set_xlabel("Search Steps", fontsize=11)
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
    """Plot boxplots comparing final best fitness distributions across restarts with jittered points and mean diamonds."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

    names = list(trajectories.keys())
    data = [traj.best_fitness.detach().cpu().numpy() for traj in trajectories.values()]

    bplot = ax.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
    )

    for patch, color in zip(bplot["boxes"], PALETTE[: len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Jittered scatter points overlay to clearly depict variance across individual restarts
    rng = np.random.default_rng(42)
    for idx, d in enumerate(data, 1):
        jitter = rng.normal(idx, 0.04, size=len(d))
        ax.scatter(jitter, d, alpha=0.45, color="black", s=18, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Best Fitness Achieved", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multiagent_diversity(
    trajectories: Trajectory | dict[str, Trajectory],
    save_path: str | Path,
    title: str = "Multi-Agent Population Diversity Over Steps",
) -> Path:
    """Plot average Hamming distance across steps for multi-agent configurations."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    traj_dict: dict[str, Trajectory] = (
        {"Population": trajectories} if isinstance(trajectories, Trajectory) else trajectories
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    has_data = False
    for i, (name, traj) in enumerate(traj_dict.items()):
        if traj.diversity_history:
            has_data = True
            color = PALETTE[i % len(PALETTE)]
            steps = np.arange(len(traj.diversity_history))
            ax.plot(steps, traj.diversity_history, label=name, color=color, linewidth=2)

    if not has_data:
        ax.text(0.5, 0.5, "No multi-agent diversity records available", ha="center", va="center")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Steps", fontsize=11)
    ax.set_ylabel("Mean Pairwise Hamming Distance", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multiagent_agent_trajectories(
    trajectory: Trajectory,
    save_path: str | Path,
    n_agents: int | None = None,
    title: str = "Multi-Agent Search Trajectories & Group Best",
) -> Path:
    """Plot per-agent fitness evolution alongside the group best-so-far envelope."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    B = trajectory.batch_size
    m = n_agents if n_agents is not None else min(B, 10)
    m = max(m, 1)

    steps_count = len(trajectory.step_history)
    if steps_count > 0:
        steps = np.arange(steps_count + 1)
        agent_curves: list[list[float]] = []
        for i in range(m):
            init_val = float(trajectory.initial_fitness[i].item())
            curve = [init_val]
            for rec in trajectory.step_history:
                curve.append(float(rec.current_fitness[i].item()))
            agent_curves.append(curve)

        for i, curve in enumerate(agent_curves):
            color = PALETTE[i % len(PALETTE)]
            ax.plot(steps, curve, label=f"Agent {i + 1}", color=color, alpha=0.55, linewidth=1.5)

        fit_matrix = np.array(agent_curves)
        group_best = np.maximum.accumulate(np.max(fit_matrix, axis=0))
        group_mean = np.mean(fit_matrix, axis=0)

        ax.plot(steps, group_best, label="Group Best", color="gold", linewidth=2.8, linestyle="--")
        ax.plot(steps, group_mean, label="Group Mean", color="white", linewidth=1.8, linestyle=":")
    else:
        ax.text(0.5, 0.5, "No step history available in trajectory", ha="center", va="center")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Steps", fontsize=11)
    ax.set_ylabel("Objective Fitness f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="lower right", ncol=min(m + 2, 4))

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multiagent_dispersion_heatmap(
    trajectory: Trajectory,
    save_path: str | Path,
    n_agents: int | None = None,
    title: str = "Multi-Agent Pairwise Dispersion Heatmap",
) -> Path:
    """Plot pairwise Hamming distance heatmaps between agents at start and completion."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    B = trajectory.batch_size
    N = trajectory.dimension
    m = n_agents if n_agents is not None else min(B, 16)
    m = max(m, 2)

    init_states = trajectory.initial_states[:m].float().cpu().numpy()
    final_states = trajectory.final_states[:m].float().cpu().numpy()

    dist_init = np.zeros((m, m), dtype=np.float32)
    dist_final = np.zeros((m, m), dtype=np.float32)

    for i in range(m):
        for j in range(m):
            dist_init[i, j] = float(np.mean(np.abs(init_states[i] - init_states[j])))
            dist_final[i, j] = float(np.mean(np.abs(final_states[i] - final_states[j])))

    triu_init = dist_init[np.triu_indices(m, 1)]
    mean_init = float(triu_init.mean()) if len(triu_init) > 0 else 0.0
    im1 = axes[0].imshow(dist_init, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Initial State Distances (Mean: {mean_init:.3f})", fontsize=11)
    axes[0].set_xlabel("Agent Index", fontsize=10)
    axes[0].set_ylabel("Agent Index", fontsize=10)
    axes[0].set_xticks(range(m))
    axes[0].set_yticks(range(m))
    axes[0].set_xticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
    axes[0].set_yticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Hamming Dist / N")

    triu_final = dist_final[np.triu_indices(m, 1)]
    mean_fin = float(triu_final.mean()) if len(triu_final) > 0 else 0.0
    im2 = axes[1].imshow(dist_final, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Final State Distances (Mean: {mean_fin:.3f})", fontsize=11)
    axes[1].set_xlabel("Agent Index", fontsize=10)
    axes[1].set_ylabel("Agent Index", fontsize=10)
    axes[1].set_xticks(range(m))
    axes[1].set_yticks(range(m))
    axes[1].set_xticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
    axes[1].set_yticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Hamming Dist / N")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_multiagent_dashboard(
    trajectory: Trajectory,
    save_path: str | Path,
    n_agents: int | None = None,
    title: str = "Multi-Agent System Comprehensive Dashboard",
) -> Path:
    """Generate a comprehensive 4-panel dashboard for multi-agent rollouts."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    B = trajectory.batch_size
    N = trajectory.dimension
    m = n_agents if n_agents is not None else min(B, 10)
    m = max(m, 1)

    steps_count = len(trajectory.step_history)
    # Panel 1: Agent curves + group best
    if steps_count > 0:
        steps = np.arange(steps_count + 1)
        agent_curves: list[list[float]] = []
        for i in range(m):
            init_val = float(trajectory.initial_fitness[i].item())
            curve = [init_val]
            for rec in trajectory.step_history:
                curve.append(float(rec.current_fitness[i].item()))
            agent_curves.append(curve)

        for i, curve in enumerate(agent_curves):
            color = PALETTE[i % len(PALETTE)]
            ax1.plot(steps, curve, label=f"Agent {i+1}", color=color, alpha=0.55, linewidth=1.5)

        fit_matrix = np.array(agent_curves)
        group_best = np.maximum.accumulate(np.max(fit_matrix, axis=0))
        ax1.plot(steps, group_best, label="Group Best", color="gold", linewidth=2.8, linestyle="--")
        ax1.set_title("Individual Agent Trajectories & Group Best", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Steps", fontsize=10)
        ax1.set_ylabel("Fitness", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True, fontsize=8, loc="lower right", ncol=min(m + 1, 4))
    else:
        ax1.text(0.5, 0.5, "No steps recorded", ha="center", va="center")

    # Panel 2: Diversity dynamics
    if trajectory.diversity_history:
        steps_div = np.arange(len(trajectory.diversity_history))
        ax2.plot(steps_div, trajectory.diversity_history, color="#00bcd4", linewidth=2.2, label="Mean Pairwise Dist")
        ax2.set_title("Population Diversity (Hamming Distance)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Steps", fontsize=10)
        ax2.set_ylabel("Mean Pairwise Hamming Distance", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, fontsize=9, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No diversity history", ha="center", va="center")

    # Panel 3: Individual best fitness contribution
    bests_per_agent = [float(trajectory.best_fitness[i].item()) for i in range(m)]
    ax3.bar(range(1, m + 1), bests_per_agent, color=PALETTE[:m], alpha=0.85, edgecolor="white", linewidth=0.8)
    overall_best = max(bests_per_agent)
    ax3.axhline(overall_best, color="gold", linestyle="--", linewidth=2, label=f"Team Best ({overall_best:.4f})")
    ax3.set_title("Best Fitness Found by Each Agent", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Agent Index", fontsize=10)
    ax3.set_ylabel("Best Fitness Found", fontsize=10)
    ax3.set_xticks(range(1, m + 1))
    ax3.set_xticklabels([f"Agent {i}" for i in range(1, m + 1)], fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel 4: Final Distance Heatmap
    if m >= 2:
        final_states = trajectory.final_states[:m].float().cpu().numpy()
        dist_final = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(m):
                dist_final[i, j] = float(np.mean(np.abs(final_states[i] - final_states[j])))
        im4 = ax4.imshow(dist_final, cmap="magma", vmin=0.0, vmax=1.0)
        ax4.set_title(f"Final Agent Pairwise Distance Matrix (N={N})", fontsize=11, fontweight="bold")
        ax4.set_xticks(range(m))
        ax4.set_yticks(range(m))
        ax4.set_xticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
        ax4.set_yticklabels([f"A{i+1}" for i in range(m)], fontsize=8)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Hamming Dist / N")
    else:
        ax4.text(0.5, 0.5, "Single agent (no peer distance matrix)", ha="center", va="center")

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
    ax1.plot(
        gens, best_fits, label="Overall Best Fitness", color="#1f77b4", linewidth=2.5, linestyle="-"
    )
    ax1.plot(
        gens, gen_bests, label="Gen Best Fitness", color="#2ca02c", linestyle="-", linewidth=1.8
    )
    ax1.plot(
        gens, mean_fits, label="Gen Mean Fitness", color="#ff7f0e", linestyle="-", linewidth=1.8
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

        ax1.plot(gens, bests, label=name, color=color, linewidth=2.2, linestyle="-")
        ax2.plot(gens, means, label=name, color=color, linestyle="-", linewidth=2.0)

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
    """Plot Best Discovered Fitness as a function of Agent Count."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    model_names = sorted(list({r.model_name for r in records}))

    for i, model in enumerate(model_names):
        color = PALETTE[i % len(PALETTE)]
        mod_records = [r for r in records if r.model_name == model]
        is_group_aware = any(getattr(r, "is_group_aware", True) for r in mod_records)

        if not is_group_aware:
            sorted_recs = sorted(mod_records, key=lambda r: r.n_agents)
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
                linestyle="-",
                linewidth=2.2,
                alpha=0.9,
            )
        else:
            sharing_recs = sorted(
                [r for r in mod_records if r.share_info], key=lambda r: r.n_agents
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
                    linestyle="-",
                    linewidth=2.2,
                )

            noshar_recs = sorted(
                [r for r in mod_records if not r.share_info or r.n_agents == 1],
                key=lambda r: r.n_agents,
            )
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
                    linestyle="-",
                    marker="x",
                    linewidth=2.0,
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
    title: str = "Information Sharing Collaboration Gain (Δ Fitness)",
) -> Path:
    """Plot bar chart of sharing gain."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    agent_counts = sorted(list({r.n_agents for r in records if r.n_agents > 1}))
    models = sorted(list({r.model_name for r in records if getattr(r, "is_group_aware", True)}))

    if not agent_counts or not models:
        ax.text(0.5, 0.5, "No multi-agent sharing records available", ha="center", va="center")
        fig.savefig(path)
        plt.close(fig)
        return path

    x = np.arange(len(agent_counts))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        gains = []
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
            gain = (
                (r_s.mean_team_best_fitness - r_ns.mean_team_best_fitness)
                if (r_s and r_ns)
                else 0.0
            )
            gains.append(gain)

        color = PALETTE[i % len(PALETTE)]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, gains, width, label=model, color=color, alpha=0.85)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Agents (M)", fontsize=11)
    ax.set_ylabel("Collaboration Gain: Fitness(Sharing) - Fitness(No Sharing)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M = {m}" for m in agent_counts])
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_benchmark_dashboard(
    trajectories: dict[str, Trajectory],
    save_path: str | Path,
    title: str = "NeuroLS Benchmark Executive Dashboard",
) -> Path:
    """Generate a comprehensive 4-panel publication-grade dashboard for benchmark comparisons."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Panel 1: Best-so-far Convergence with variance band
    for i, (name, traj) in enumerate(trajectories.items()):
        color = PALETTE[i % len(PALETTE)]
        if (
            traj.step_history
            and len(traj.step_history) > 0
            and getattr(traj.step_history[0], "best_fitness", None) is not None
            and traj.step_history[0].best_fitness.ndim > 0
        ):
            b_means = np.array([rec.best_fitness.mean().item() for rec in traj.step_history])
            b_stds = np.array([rec.best_fitness.std().item() for rec in traj.step_history])
            steps = np.arange(len(b_means))
            ax1.plot(steps, b_means, label=name, color=color, linewidth=2.0, linestyle="-")
            ax1.fill_between(steps, b_means - b_stds, b_means + b_stds, color=color, alpha=0.12)
        elif traj.best_fitness_history:
            steps = np.arange(len(traj.best_fitness_history))
            ax1.plot(
                steps,
                traj.best_fitness_history,
                label=name,
                color=color,
                linewidth=2.0,
                linestyle="-",
            )
    ax1.set_title("A. Best-so-Far Fitness Trajectory (±1 Std)", fontsize=11, fontweight="bold")
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
                linewidth=2.0,
                linestyle="-",
            )
    ax2.set_title("B. Population Mean Fitness Dynamics", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Mean Fitness", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    # Panel 3: Final Boxplots with Mean Diamond and Jittered Scatter
    names = list(trajectories.keys())
    data = [traj.best_fitness.detach().cpu().numpy() for traj in trajectories.values()]
    bplot = ax3.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white", markersize=5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bplot["boxes"], PALETTE[: len(data)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for idx, d in enumerate(data, 1):
        jitter = rng.normal(idx, 0.04, size=len(d))
        ax3.scatter(jitter, d, alpha=0.45, color="black", s=14, zorder=3)

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
            ax4.plot(
                steps, traj.diversity_history, label=name, color=color, linewidth=2.0, linestyle="-"
            )

    if any_diversity:
        ax4.set_title("D. Multi-Agent Group Diversity Evolution", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Steps", fontsize=10)
        ax4.set_ylabel("Mean Hamming Distance (bits)", fontsize=10)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(fontsize=8, loc="upper right")
    else:
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


# ============================================================================
# Unified Plot Registry & CLI Selector
# ============================================================================


@dataclass
class PlotDefinition:
    """Specification of an available visualization."""

    name: str
    tags: list[str]
    description: str
    category: str  # "benchmark", "training", "explore"
    default_filename: str
    render_fn: Callable[..., Path]


PLOT_REGISTRY: dict[str, PlotDefinition] = {
    # Benchmark Category
    "dashboard": PlotDefinition(
        name="dashboard",
        tags=["benchmark", "summary", "default"],
        description="4-panel composite dashboard (convergence, dynamics, boxplots, diversity/evaluations)",
        category="benchmark",
        default_filename="benchmark_dashboard.png",
        render_fn=plot_benchmark_dashboard,
    ),
    "boxplots": PlotDefinition(
        name="boxplots",
        tags=["benchmark", "distribution"],
        description="Final best fitness boxplots comparing strategies across restarts",
        category="benchmark",
        default_filename="boxplot_comparison.png",
        render_fn=plot_final_fitness_boxplots,
    ),
    "convergence": PlotDefinition(
        name="convergence",
        tags=["benchmark", "dynamics"],
        description="Mean fitness convergence curves over steps for all strategies",
        category="benchmark",
        default_filename="convergence_comparison.png",
        render_fn=plot_convergence_curves,
    ),
    "step_trajectories": PlotDefinition(
        name="step_trajectories",
        tags=["benchmark", "dynamics"],
        description="Detailed best-so-far and mean fitness trajectories over search steps",
        category="benchmark",
        default_filename="step_trajectories.png",
        render_fn=plot_step_trajectories,
    ),
    "diversity": PlotDefinition(
        name="diversity",
        tags=["benchmark", "multiagent"],
        description="Mean pairwise Hamming distance among agents across search steps",
        category="benchmark",
        default_filename="diversity_dynamics.png",
        render_fn=plot_multiagent_diversity,
    ),
    "multiagent_trajectories": PlotDefinition(
        name="multiagent_trajectories",
        tags=["multiagent", "explore", "dynamics"],
        description="Individual agent fitness curves alongside the group best-so-far envelope",
        category="multiagent",
        default_filename="multiagent_trajectories.png",
        render_fn=plot_multiagent_agent_trajectories,
    ),
    "multiagent_dispersion": PlotDefinition(
        name="multiagent_dispersion",
        tags=["multiagent", "explore", "heatmap"],
        description="Initial and final pairwise Hamming distance heatmaps among agents",
        category="multiagent",
        default_filename="multiagent_dispersion.png",
        render_fn=plot_multiagent_dispersion_heatmap,
    ),
    "multiagent_dashboard": PlotDefinition(
        name="multiagent_dashboard",
        tags=["multiagent", "summary", "default"],
        description="Composite 4-panel dashboard of multi-agent dynamics, diversity, and dispersion",
        category="multiagent",
        default_filename="multiagent_dashboard.png",
        render_fn=plot_multiagent_dashboard,
    ),
    "fitness_vs_agents": PlotDefinition(
        name="fitness_vs_agents",
        tags=["benchmark", "multiagent"],
        description="Best fitness as a function of agent population size M",
        category="benchmark",
        default_filename="fitness_vs_agents.png",
        render_fn=plot_fitness_vs_agents,
    ),
    "sharing_gain": PlotDefinition(
        name="sharing_gain",
        tags=["benchmark", "multiagent"],
        description="Collaboration gain from information sharing across agent counts",
        category="benchmark",
        default_filename="sharing_gain.png",
        render_fn=plot_sharing_gain_bars,
    ),
    # Training Category
    "training": PlotDefinition(
        name="training",
        tags=["training", "default"],
        description="CMA-ES training convergence (best fitness, mean fitness, sigma, improvement deltas)",
        category="training",
        default_filename="training_convergence.png",
        render_fn=plot_training_convergence,
    ),
    "feature_importance": PlotDefinition(
        name="feature_importance",
        tags=["training", "model"],
        description="Learned neural policy feature weights or attribution magnitudes",
        category="training",
        default_filename="feature_importance.png",
        render_fn=plot_neural_feature_importance,
    ),
    # Exploration Category
    "exploration_dashboard": PlotDefinition(
        name="exploration_dashboard",
        tags=["explore", "summary", "default"],
        description="Composite diagnostic dashboard (fitness, 2D trajectory, bitstring heatmap, action ranks)",
        category="explore",
        default_filename="exploration_dashboard.png",
        render_fn=plot_exploration_dashboard,
    ),
    "fitness_evolution": PlotDefinition(
        name="fitness_evolution",
        tags=["explore", "dynamics"],
        description="Step-by-step current and best fitness curves for a representative run",
        category="explore",
        default_filename="fitness_evolution.png",
        render_fn=plot_fitness_evolution,
    ),
    "bitstring_evolution": PlotDefinition(
        name="bitstring_evolution",
        tags=["explore", "diagnostics"],
        description="Heatmap of binary configuration over steps with flipped loci and locus frequency",
        category="explore",
        default_filename="bitstring_evolution.png",
        render_fn=plot_bitstring_evolution,
    ),
    "trajectory_2d": PlotDefinition(
        name="trajectory_2d",
        tags=["explore", "diagnostics"],
        description="Dimensionality-reduced 2D projection (PCA or MDS) of the search path with fitness gradient",
        category="explore",
        default_filename="search_trajectory_2d.png",
        render_fn=plot_search_trajectory_2d,
    ),
    "action_ranks": PlotDefinition(
        name="action_ranks",
        tags=["explore", "diagnostics"],
        description="Distribution of chosen neighbor ranks and step-by-step rank timeline",
        category="explore",
        default_filename="action_ranks.png",
        render_fn=plot_action_ranks,
    ),
    "move_deltas": PlotDefinition(
        name="move_deltas",
        tags=["explore", "diagnostics"],
        description="Per-step fitness deltas of chosen moves (improving vs neutral vs worsening)",
        category="explore",
        default_filename="move_deltas.png",
        render_fn=plot_move_deltas,
    ),
}

PLOT_GROUPS: dict[str, list[str]] = {
    "default_benchmark": ["dashboard", "boxplots", "convergence", "step_trajectories"],
    "default_multiagent": [
        "multiagent_dashboard",
        "diversity",
        "multiagent_trajectories",
        "multiagent_dispersion",
    ],
    "default_training": ["training", "feature_importance"],
    "default_explore": [
        "exploration_dashboard",
        "trajectory_2d",
        "bitstring_evolution",
        "action_ranks",
        "fitness_evolution",
        "move_deltas",
    ],
    "all_benchmark": ["dashboard", "boxplots", "convergence", "step_trajectories", "diversity"],
    "all_multiagent": [
        "multiagent_dashboard",
        "diversity",
        "multiagent_trajectories",
        "multiagent_dispersion",
        "fitness_vs_agents",
        "sharing_gain",
    ],
    "all_explore": [
        "exploration_dashboard",
        "fitness_evolution",
        "bitstring_evolution",
        "trajectory_2d",
        "action_ranks",
        "move_deltas",
    ],
    "all_training": ["training", "feature_importance"],
    "all": list(PLOT_REGISTRY.keys()),
}


def resolve_plot_names(
    specs: str | list[str] | None,
    context: str = "benchmark",
) -> list[str]:
    """Resolve a flexible plot specification into canonical plot identifiers.

    Supports:
    - Group keywords prefixed with '@': '@default', '@all', '@benchmark', '@explore', '@training', '@multiagent'
    - Comma-separated names: 'convergence,boxplots,trajectory_2d'
    - Exclusion with '-': '@all,-move_deltas'

    Args:
        specs: Specification string, list of tokens, or None.
        context: Context of caller ('benchmark', 'training', or 'explore').

    Returns:
        List of unique canonical plot names in deterministic order.
    """
    if specs is None:
        if context == "explore":
            return list(PLOT_GROUPS["default_explore"])
        elif context == "training":
            return list(PLOT_GROUPS["default_training"])
        return list(PLOT_GROUPS["default_benchmark"])

    if isinstance(specs, str):
        tokens = [t.strip() for t in specs.replace(";", ",").split(",") if t.strip()]
    else:
        tokens = list(specs)

    included: list[str] = []
    excluded: set[str] = set()

    for token in tokens:
        is_exclude = token.startswith("-")
        clean = token.lstrip("-@").strip()

        # Group expansion
        if clean in ("default", f"default_{context}"):
            group_key = f"default_{context}"
            items = PLOT_GROUPS.get(group_key, PLOT_GROUPS["default_benchmark"])
        elif clean in ("all", f"all_{context}"):
            group_key = f"all_{context}" if f"all_{context}" in PLOT_GROUPS else "all"
            items = PLOT_GROUPS.get(group_key, list(PLOT_REGISTRY.keys()))
        elif clean in ("benchmark", "explore", "training", "multiagent"):
            items = [
                name for name, p in PLOT_REGISTRY.items() if clean in p.tags or p.category == clean
            ]
        elif clean in PLOT_GROUPS:
            items = PLOT_GROUPS[clean]
        elif clean in PLOT_REGISTRY:
            items = [clean]
        else:
            # Match by tag or substring
            matches = [
                name for name, p in PLOT_REGISTRY.items() if clean in p.tags or clean == name
            ]
            if matches:
                items = matches
            else:
                raise ValueError(
                    f"Unknown plot name or keyword '{token}'. "
                    f"Available plots: {list(PLOT_REGISTRY.keys())}, "
                    f"Available groups: ['@default', '@all', '@benchmark', '@explore', '@training', '@multiagent']"
                )

        if is_exclude:
            excluded.update(items)
        else:
            for item in items:
                if item not in included:
                    included.append(item)

    return [name for name in included if name not in excluded and name in PLOT_REGISTRY]
