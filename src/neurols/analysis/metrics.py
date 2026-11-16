"""Performance metrics and statistical aggregators for Local Search trajectories."""

from __future__ import annotations

import numpy as np

from neurols.core.trajectory import Trajectory


def compute_auc(fitness_curve: list[float] | np.ndarray) -> float:
    """Compute normalized Area Under the Convergence Curve (higher is faster & better)."""
    arr = np.asarray(fitness_curve, dtype=np.float64)
    if len(arr) < 2:
        return float(arr[0]) if len(arr) == 1 else 0.0
    # Trapezoidal rule normalized by number of intervals (NumPy 2.0+ uses trapezoid)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(arr) / (len(arr) - 1))
    return float(np.mean(arr))


def compute_time_to_target(
    trajectories: list[float] | np.ndarray,
    target_fitness: float,
) -> int | None:
    """Find the first step at which target fitness is reached or exceeded."""
    arr = np.asarray(trajectories)
    indices = np.where(arr >= target_fitness)[0]
    return int(indices[0]) if len(indices) > 0 else None


def compute_success_rate(
    final_fitness_values: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of runs achieving final fitness >= threshold."""
    return float(np.mean(final_fitness_values >= threshold))


def summarize_trajectory(traj: Trajectory) -> dict[str, float]:
    """Compute aggregate benchmark metrics for a single trajectory run."""
    best_arr = traj.best_fitness.detach().cpu().numpy()
    steps_arr = traj.steps_taken.detach().cpu().numpy()
    auc_val = compute_auc(traj.mean_fitness_history)

    return {
        "mean_best_fitness": float(np.mean(best_arr)),
        "std_best_fitness": float(np.std(best_arr)),
        "median_best_fitness": float(np.median(best_arr)),
        "iqr_best_fitness": float(np.percentile(best_arr, 75) - np.percentile(best_arr, 25)),
        "max_best_fitness": float(np.max(best_arr)),
        "min_best_fitness": float(np.min(best_arr)),
        "auc_fitness": auc_val,
        "mean_steps": float(np.mean(steps_arr)),
        "final_mean_diversity": (
            float(traj.diversity_history[-1]) if traj.diversity_history else 0.0
        ),
    }
