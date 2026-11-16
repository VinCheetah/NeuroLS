"""Execution engines for Local Search in NeuroLS.

Provides BatchedRunner (high-throughput vectorized GPU/CPU execution supporting
multiple instances, restarts, and multi-agent groups) and TraditionalRunner
(step-by-step single-agent inspection and debugging).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from neurols.core.group import GroupConfig, GroupContext
from neurols.core.problem import BatchedProblem
from neurols.core.trajectory import Trajectory
from neurols.strategies.base import BaseStrategy
from neurols.types import DeviceType, SeedType, StepRecord


class BatchedRunner:
    """High-throughput vectorized local search runner.

    Coordinates batched rollouts across arbitrary combinations of:
    - Multiple problem instances (MultiNKLandscape or single instance)
    - Multiple independent restarts (different initial search states)
    - Multi-agent groups (groups of M agents sharing context on each instance & restart)
    """

    def __init__(
        self,
        problem: BatchedProblem,
        strategy: BaseStrategy,
        max_steps: int = 200,
        group_config: GroupConfig | None = None,
        record_history: bool = True,
        device: DeviceType = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.problem = problem.to(self.device)
        self.strategy = strategy
        self.max_steps = max_steps
        self.group_config = group_config or GroupConfig(n_agents=1)
        self.record_history = record_history

    def run(
        self,
        n_instances: int = 1,
        n_restarts: int = 1,
        initial_states: torch.Tensor | None = None,
        seed: SeedType = None,
        callback: Callable[[int, float, float, float], None] | None = None,
    ) -> Trajectory:
        """Execute batched local search rollout.

        Total batch size: B = n_instances * n_restarts * n_agents.

        Args:
            n_instances: Number of distinct problem instances in parallel.
            n_restarts: Number of independent restarts per instance.
            initial_states: Optional pre-sampled starting states of shape (B, N).
            seed: Optional seed for RNG and trajectory sampling.
            callback: Optional hook called per step: callback(step, mean_fit, best_fit, diversity).

        Returns:
            Trajectory containing initial, final, and best states, fitness curves, and diagnostics.
        """
        start_time = time.perf_counter()
        n_agents = self.group_config.n_agents
        total_batch_size = n_instances * n_restarts * n_agents
        N = self.problem.dimension

        # RNG generator
        gen: torch.Generator | None = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)

        # 1. Sample or validate initial states
        if initial_states is not None:
            states = initial_states.to(self.device)
            if states.shape[0] != total_batch_size:
                # Attempt to infer n_restarts from provided initial states
                per_inst = n_instances * n_agents
                if states.shape[0] % per_inst == 0:
                    n_restarts = states.shape[0] // per_inst
                    total_batch_size = states.shape[0]
                else:
                    raise ValueError(
                        f"Expected initial_states shape ({total_batch_size}, {N}), got {states.shape}"
                    )
            if states.shape != (total_batch_size, N):
                raise ValueError(
                    f"Expected initial_states shape ({total_batch_size}, {N}), got {states.shape}"
                )
        else:
            states = self.problem.sample_initial_states((total_batch_size,), generator=gen)

        # 2. Evaluate starting fitness
        current_fitness = self.problem.evaluate(states)  # (B,)
        best_fitness = current_fitness.clone()
        best_states = states.clone()

        # 3. Setup multi-agent group context
        group_ctx = GroupContext(
            config=self.group_config,
            n_instances=n_instances,
            n_restarts=n_restarts,
            device=self.device,
        )

        # Reset strategy
        self.strategy.reset(batch_size=total_batch_size, device=self.device, seed=seed)

        # Tracking arrays
        steps_taken = torch.zeros(total_batch_size, dtype=torch.int64, device=self.device)
        active_mask = torch.ones(total_batch_size, dtype=torch.bool, device=self.device)

        mean_fit_history: list[float] = [float(current_fitness.mean().item())]
        best_fit_history: list[float] = [float(best_fitness.max().item())]
        diversity_history: list[float] = []
        step_records: list[StepRecord] = []

        initial_states_saved = states.clone()
        initial_fitness_saved = current_fitness.clone()

        # 4. Local Search Iteration Loop
        for step in range(self.max_steps):
            if not active_mask.any():
                break

            # Compute group metrics
            group_metrics = group_ctx.update(
                states=states,
                current_fitness=current_fitness,
                best_fitness=best_fitness,
                best_states=best_states,
            )
            diversity_history.append(group_metrics.group_diversity)

            # Compute neighborhood deltas
            deltas = self.problem.get_neighborhood_deltas(states)  # (B, N)

            # Strategy selects move
            move_res = self.strategy.select_move(
                states=states,
                deltas=deltas,
                current_fitness=current_fitness,
                best_fitness=best_fitness,
                group_context=group_ctx,
                group_metrics=group_metrics,
                step=step,
            )

            chosen_moves = move_res.chosen_moves  # (B,)
            if move_res.stop_mask is not None:
                active_mask &= ~move_res.stop_mask

            # Apply moves only for active agents with valid move index (>= 0)
            valid_moves = (chosen_moves >= 0) & active_mask
            prev_fitness = current_fitness.clone()

            if valid_moves.any():
                new_states = self.problem.apply_moves(states, chosen_moves)
                # Update states only where valid_moves is True
                states = torch.where(valid_moves.unsqueeze(1), new_states, states)

                # Re-evaluate fitness
                new_fitness = self.problem.evaluate(states)
                current_fitness = torch.where(valid_moves, new_fitness, current_fitness)

                # Update best so far
                improved = current_fitness > best_fitness
                if improved.any():
                    best_fitness = torch.where(improved, current_fitness, best_fitness)
                    best_states = torch.where(improved.unsqueeze(1), states, best_states)

            # Steps taken tracking
            steps_taken[active_mask] += 1

            # Metric recording
            delta_fit = current_fitness - prev_fitness
            mean_fit = float(current_fitness.mean().item())
            best_fit = float(best_fitness.max().item())
            mean_fit_history.append(mean_fit)
            best_fit_history.append(best_fit)

            if self.record_history:
                step_records.append(
                    StepRecord(
                        step=step,
                        current_fitness=current_fitness.clone(),
                        best_fitness=best_fitness.clone(),
                        chosen_moves=chosen_moves.clone(),
                        delta_fitness=delta_fit.clone(),
                        group_diversity=torch.tensor(group_metrics.group_diversity),
                    )
                )

            if callback is not None:
                callback(step, mean_fit, best_fit, group_metrics.group_diversity)

        elapsed = time.perf_counter() - start_time

        return Trajectory(
            batch_size=total_batch_size,
            dimension=N,
            max_steps=self.max_steps,
            initial_states=initial_states_saved,
            final_states=states,
            best_states=best_states,
            initial_fitness=initial_fitness_saved,
            final_fitness=current_fitness,
            best_fitness=best_fitness,
            steps_taken=steps_taken,
            step_history=step_records,
            mean_fitness_history=mean_fit_history,
            best_fitness_history=best_fit_history,
            diversity_history=diversity_history,
            metadata={
                "strategy": self.strategy.name,
                "hyperparameters": self.strategy.hyperparameters,
                "elapsed_seconds": elapsed,
                "steps_per_second": (total_batch_size * max(steps_taken.float().mean().item(), 1.0))
                / max(elapsed, 1e-6),
            },
        )


class TraditionalRunner:
    """Sequential step-by-step local search runner for single instances and debugging.

    Provides a clean, transparent interface without batching overhead, ideal
    for inspecting individual trajectory steps, intermediate state transitions,
    and testing custom heuristics.
    """

    def __init__(
        self,
        problem: BatchedProblem,
        strategy: BaseStrategy,
        device: DeviceType = "cpu",
    ) -> None:
        self.problem = problem.to(torch.device(device))
        self.strategy = strategy
        self.device = torch.device(device)
        self.state: torch.Tensor | None = None
        self.fitness: float = 0.0
        self.best_fitness: float = 0.0
        self.best_state: torch.Tensor | None = None
        self.step_idx: int = 0

    def reset(self, initial_state: torch.Tensor | None = None, seed: SeedType = None) -> float:
        """Reset runner to initial state."""
        self.strategy.reset(batch_size=1, device=self.device, seed=seed)
        if initial_state is not None:
            self.state = initial_state.to(self.device).view(1, -1)
        else:
            self.state = self.problem.sample_initial_states((1,))
        fit_tensor = self.problem.evaluate(self.state)
        self.fitness = float(fit_tensor.item())
        self.best_fitness = self.fitness
        self.best_state = self.state.clone()
        self.step_idx = 0
        return self.fitness

    def step(self) -> StepRecord:
        """Execute a single local search transition."""
        if self.state is None:
            self.reset()
        assert self.state is not None

        deltas = self.problem.get_neighborhood_deltas(self.state)
        curr_fit_t = torch.tensor([self.fitness], device=self.device)
        best_fit_t = torch.tensor([self.best_fitness], device=self.device)

        move_res = self.strategy.select_move(
            states=self.state,
            deltas=deltas,
            current_fitness=curr_fit_t,
            best_fitness=best_fit_t,
            step=self.step_idx,
        )

        chosen = int(move_res.chosen_moves[0].item())
        delta_val = float(deltas[0, chosen].item()) if chosen >= 0 else 0.0

        if chosen >= 0 and not (move_res.stop_mask is not None and move_res.stop_mask[0].item()):
            self.state = self.problem.apply_moves(self.state, move_res.chosen_moves)
            self.fitness = float(self.problem.evaluate(self.state).item())
            if self.fitness > self.best_fitness:
                self.best_fitness = self.fitness
                self.best_state = self.state.clone()

        record = StepRecord(
            step=self.step_idx,
            current_fitness=torch.tensor([self.fitness]),
            best_fitness=torch.tensor([self.best_fitness]),
            chosen_moves=move_res.chosen_moves,
            delta_fitness=torch.tensor([delta_val]),
        )
        self.step_idx += 1
        return record
