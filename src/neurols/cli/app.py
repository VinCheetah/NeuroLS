"""Command Line Interface for NeuroLS using Typer and Rich.

Provides commands to run local search trajectories, benchmark classical and neural
strategies, train policies with CMA-ES, inspect neural feature attributions, and analyze NK landscapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from neurols.analysis.ablation import (
    DEFAULT_ABLATION_SUITE,
    AblationStudy,
)
from neurols.analysis.experiment import ExperimentLogger
from neurols.analysis.interpretability import compute_feature_sensitivities, explain_decision
from neurols.analysis.metrics import summarize_trajectory
from neurols.analysis.multi_agent_study import ModelSpec, MultiAgentStudy, get_study_suite
from neurols.analysis.plotting import (
    plot_benchmark_dashboard,
    plot_convergence_curves,
    plot_final_fitness_boxplots,
    plot_neural_feature_importance,
    plot_step_trajectories,
    plot_training_convergence,
)
from neurols.cli.args import (
    KOption,
    NbAgentsOption,
    NbBudgetOption,
    NbGenerationsOption,
    NbIterationsOption,
    NbRestartsOption,
    NbTimeOption,
    NOption,
    OutputDirOption,
    PlotOption,
    PopulationSizeOption,
    SeedOption,
    SharedInfoOption,
    StrategyOption,
    TrainingOption,
)
from neurols.core.group import GroupConfig
from neurols.core.runner import BatchedRunner
from neurols.problems.nk_landscape import NKLandscape
from neurols.strategies.classical import (
    FirstImprovement,
    HillClimber,
    RandomWalk,
    SimulatedAnnealing,
    TabuSearch,
)
from neurols.strategies.neural import NeuralLocalSearchStrategy
from neurols.strategies.neural.features import FeatureAblationLevel, FeatureConfig
from neurols.training.base import GenerationRecord
from neurols.training.cma_es import CMAESOptimizer
from neurols.types import StoppingCriteria

app = typer.Typer(
    name="neurols",
    help="NeuroLS: Modular Local Search & Neuroevolution Framework",
    add_completion=True,
)
console = Console()


def _get_strategy(name: str, **kwargs) -> Any:
    """Helper factory for strategies."""
    name_lower = name.lower()
    if "hill" in name_lower or "steepest" in name_lower:
        return HillClimber()
    elif "first" in name_lower:
        return FirstImprovement(order=kwargs.get("order", "random"))
    elif "tabu" in name_lower:
        return TabuSearch(tenure=kwargs.get("tenure", 8))
    elif "random" in name_lower:
        return RandomWalk(exploration_rate=kwargs.get("exploration_rate", 1.0))
    elif "neural" in name_lower or "nn" in name_lower:
        return NeuralLocalSearchStrategy(
            policy_type=kwargs.get("policy_type", "linear"),
            max_steps=kwargs.get("steps", 100),
        )
    else:
        raise typer.BadParameter(f"Unknown strategy: '{name}'")


@app.command()
def run(
    strategy: StrategyOption = "HillClimber",
    n: NOption = 32,
    k: KOption = 4,
    steps: NbIterationsOption = None,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    restarts: NbRestartsOption = 10,
    agents: NbAgentsOption = 1,
    share_info: SharedInfoOption = True,
    plot: PlotOption = False,
    output_dir: OutputDirOption = Path("experiments/run"),
    seed: SeedOption = 42,
) -> None:
    """Execute a local search rollout with live progress and metrics display."""
    crit_info = []
    if steps is not None:
        crit_info.append(f"Steps: [magenta]{steps}[/magenta]")
    if budget is not None:
        crit_info.append(f"Budget: [magenta]{budget:,}[/magenta]")
    if max_time is not None:
        crit_info.append(f"Max Time: [magenta]{max_time}s[/magenta]")
    crit_str = " | ".join(crit_info) if crit_info else "Steps: [magenta]100[/magenta]"

    console.print(
        Panel.fit(
            f"[bold cyan]NeuroLS Run[/bold cyan]\n"
            f"Strategy: [yellow]{strategy}[/yellow] | Problem: [green]NK Landscape (N={n}, K={k})[/green]\n"
            f"Restarts: [magenta]{restarts}[/magenta] | Agents per group: [magenta]{agents}[/magenta] | {crit_str}",
            border_style="cyan",
        )
    )

    problem = NKLandscape(n=n, k=k, seed=seed)
    group_cfg = GroupConfig(
        n_agents=agents,
        share_distance=share_info,
        share_scores=share_info,
        share_ranks=share_info,
        share_consensus=share_info,
    )
    strat = _get_strategy(strategy, steps=steps)
    stopping_criteria = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )
    runner = BatchedRunner(
        problem=problem,
        strategy=strat,
        stopping_criteria=stopping_criteria,
        group_config=group_cfg,
    )

    total_progress_steps = steps or 100
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running local search...", total=total_progress_steps)

        def step_callback(step: int, mean_fit: float, best_fit: float, diversity: float) -> None:
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Step {step + 1} | Best: [bold green]{best_fit:.4f}[/bold green] | Mean: [yellow]{mean_fit:.4f}[/yellow]",
            )

        traj = runner.run(n_restarts=restarts, seed=seed, callback=step_callback)
        progress.update(task, completed=total_progress_steps)

    summary = summarize_trajectory(traj)

    # Display Rich summary table
    table = Table(title="Execution Summary", border_style="bright_blue")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Mean Best Fitness", f"{summary['mean_best_fitness']:.4f}")
    table.add_row("Median Best Fitness", f"{summary['median_best_fitness']:.4f}")
    table.add_row("Std Best Fitness", f"{summary['std_best_fitness']:.4f}")
    table.add_row("Max Overall Fitness", f"{summary['max_best_fitness']:.4f}")
    table.add_row("Min Fitness", f"{summary['min_best_fitness']:.4f}")
    table.add_row("AUC Fitness", f"{summary['auc_fitness']:.4f}")
    table.add_row("Mean Steps Taken", f"{summary['mean_steps']:.1f}")
    if traj.termination_reason:
        reason_str = (
            traj.termination_reason.value
            if hasattr(traj.termination_reason, "value")
            else str(traj.termination_reason)
        )
        table.add_row("Termination Reason", f"[bold yellow]{reason_str}[/bold yellow]")
    if traj.total_evaluations > 0:
        table.add_row("Total Evaluations", f"{traj.total_evaluations:,}")
    if agents > 1:
        table.add_row("Final Group Diversity", f"{summary['final_mean_diversity']:.2f} bits")
    table.add_row("Runtime", f"{traj.elapsed_seconds:.3f}s")
    table.add_row("Throughput", f"{traj.metadata.get('steps_per_second', 0.0):,.0f} agent-steps/s")

    console.print(table)

    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_path = output_dir / "step_dynamics.png"
        plot_step_trajectories(
            {strategy: traj},
            traj_path,
            title=f"Trajectory Dynamics: {strategy} on NK(N={n}, K={k})",
        )
        console.print(
            f"[bold green]✓ Saved trajectory dynamics plot to:[/bold green] [cyan]{traj_path}[/cyan]"
        )


@app.command()
def benchmark(
    n: NOption = 32,
    k: KOption = 4,
    preset: str = typer.Option(
        "standard",
        "--preset",
        "-p",
        help="Benchmark suite preset: 'standard', 'per-action', 'architectures', 'all'",
    ),
    policy_type: str = typer.Option(
        "invariant_nnet",
        "--policy-type",
        help="Policy architecture for standard preset ('peer_invariant_per_action', 'invariant_nnet', 'per_action')",
    ),
    ablation_level: str = typer.Option(
        "full",
        "--ablation-level",
        help="Feature ablation level ('delta_only', 'ranked_delta_only', 'delta_and_peers', 'full')",
    ),
    steps: NbIterationsOption = 100,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    restarts: NbRestartsOption = 20,
    agents: NbAgentsOption = 1,
    train_nn: TrainingOption = True,
    train_generations: NbGenerationsOption = 10,
    plot: PlotOption = True,
    output_dir: OutputDirOption = Path("experiments/benchmark"),
    seed: SeedOption = 42,
) -> None:
    """Benchmark multiple local search strategies side-by-side with pre-trained neural policies."""
    crit_info = []
    if steps is not None:
        crit_info.append(f"Steps: [yellow]{steps}[/yellow]")
    if budget is not None:
        crit_info.append(f"Budget: [yellow]{budget:,}[/yellow]")
    if max_time is not None:
        crit_info.append(f"Max Time: [yellow]{max_time}s[/yellow]")
    crit_str = " | ".join(crit_info) if crit_info else "Steps: [yellow]100[/yellow]"

    console.print(
        Panel.fit(
            f"[bold green]NeuroLS Multi-Strategy Benchmark[/bold green]\n"
            f"Landscape: [cyan]NK(N={n}, K={k})[/cyan] | Preset: [yellow]{preset}[/yellow] | Restarts: [yellow]{restarts}[/yellow] | {crit_str}\n"
            f"Pre-train NN: [magenta]{train_nn} ({train_generations} gens)[/magenta] | Agents per group: [magenta]{agents}[/magenta]",
            border_style="green",
        )
    )

    problem = NKLandscape(n=n, k=k, seed=seed)
    group_cfg = GroupConfig(n_agents=agents)
    stopping_criteria = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )

    # Determine strategies based on preset
    strategies: list[Any] = []
    if preset in ("per-action", "per_action"):
        suite = get_study_suite("per-action")
    elif preset in ("architectures", "arch"):
        suite = get_study_suite("architectures")
    elif preset in ("all", "full"):
        suite = get_study_suite("all")
    else:
        # Standard: user-selected neural model + classical baselines
        suite = [
            ModelSpec(
                name=f"Neural_{policy_type}",
                model_type="neural",
                policy_type=policy_type,
                ablation_level=FeatureAblationLevel(ablation_level),
                description=f"Neural {policy_type} policy",
            ),
            ModelSpec(name="HillClimber", model_type="classical"),
            ModelSpec(name="FirstImprovement", model_type="classical"),
            ModelSpec(name="TabuSearch", model_type="classical"),
        ]

    for spec in suite:
        if spec.model_type == "classical":
            name_lower = spec.name.lower()
            if "first" in name_lower:
                strategies.append(FirstImprovement(order="random", seed=seed))
            elif "tabu" in name_lower:
                strategies.append(TabuSearch(tenure=5))
            else:
                strategies.append(HillClimber())
        else:
            feat_cfg = FeatureConfig.from_ablation_level(spec.ablation_level, n_agents=agents)
            n_strat = NeuralLocalSearchStrategy(
                name=spec.name,
                policy_type=spec.policy_type,  # type: ignore
                feature_config=feat_cfg,
                n_actions=n,
                n_agents=agents,
                max_steps=steps or 100,
                seed=seed,
            )
            if train_nn and train_generations > 0:
                with console.status(f"[bold magenta]Pre-training {spec.name} via CMA-ES..."):
                    train_problem = NKLandscape(n=n, k=k, seed=seed + 1000)
                    optimizer = CMAESOptimizer(
                        strategy=n_strat,
                        problem=train_problem,
                        eval_restarts=min(restarts, 12),
                        eval_steps=min(steps or 30, 30),
                        population_size=8,
                        group_config=group_cfg,
                        seed=seed,
                    )
                    optimizer.train(n_generations=train_generations)
            strategies.append(n_strat)

    logger = ExperimentLogger(base_dir=output_dir)
    with console.status("[bold cyan]Benchmarking strategies in parallel..."):
        result = logger.run_comparison(
            experiment_name=f"benchmark_N{n}_K{k}_{preset}",
            problem=problem,
            strategies=strategies,
            n_restarts=restarts,
            max_steps=steps,
            stopping_criteria=stopping_criteria,
            group_config=group_cfg,
            seed=seed,
        )

    # Render Leaderboard Table
    table = Table(
        title=f"Benchmark Results: NK(N={n}, K={k}, Preset={preset})", border_style="green"
    )
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Strategy", style="bold white")
    table.add_column("Mean Best Fit", justify="right", style="cyan")
    table.add_column("Std Fit", justify="right", style="dim")
    table.add_column("Max Fit", justify="right", style="green")
    table.add_column("AUC", justify="right", style="magenta")
    table.add_column("Mean Steps", justify="right", style="white")

    for rank, row in enumerate(result.summary_table, 1):
        table.add_row(
            str(rank),
            row["strategy"],
            f"{row['mean_best_fitness']:.4f}",
            f"{row['std_best_fitness']:.4f}",
            f"{row['max_best_fitness']:.4f}",
            f"{row['auc_fitness']:.4f}",
            f"{row['mean_steps']:.1f}",
        )

    console.print(table)

    # Save plots
    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        conv_path = output_dir / "convergence_comparison.png"
        step_path = output_dir / "step_trajectories.png"
        box_path = output_dir / "boxplot_comparison.png"
        dash_path = output_dir / "benchmark_dashboard.png"

        plot_convergence_curves(
            result.trajectories, conv_path, title=f"Convergence: NK(N={n}, K={k})"
        )
        plot_step_trajectories(
            result.trajectories, step_path, title=f"Step Trajectories: NK(N={n}, K={k})"
        )
        plot_final_fitness_boxplots(
            result.trajectories, box_path, title=f"Fitness Distribution: NK(N={n}, K={k})"
        )
        plot_benchmark_dashboard(
            result.trajectories, dash_path, title=f"Benchmark Dashboard: NK(N={n}, K={k})"
        )
        console.print(
            f"[bold green]✓ Saved benchmark plots and dashboard to:[/bold green] [cyan]{output_dir}[/cyan]"
        )
        console.print(f"[bold green]✓ Saved plots to:[/bold green] [cyan]{output_dir}[/cyan]")


@app.command()
def train(
    n: NOption = 32,
    k: KOption = 4,
    generations: NbGenerationsOption = 20,
    popsize: PopulationSizeOption = 10,
    policy_type: str = typer.Option(
        "linear", "--policy-type", help="Policy architecture (linear, mlp, deepsets)"
    ),
    eval_restarts: int = typer.Option(
        8, "--eval-restarts", "-r", help="Restarts per evaluation candidate"
    ),
    eval_steps: int = typer.Option(40, "--eval-steps", help="Steps per evaluation rollout"),
    agents: NbAgentsOption = 1,
    checkpoint_dir: Path = typer.Option(
        Path("checkpoints"), "--ckpt-dir", help="Directory for saving checkpoints"
    ),
    plot: PlotOption = True,
    seed: SeedOption = 42,
) -> None:
    """Train a Neural Local Search policy using CMA-ES neuroevolution."""
    console.print(
        Panel.fit(
            f"[bold magenta]CMA-ES Neuroevolution Training[/bold magenta]\n"
            f"Policy: [cyan]{policy_type}[/cyan] | Landscape: [green]NK(N={n}, K={k})[/green]\n"
            f"Generations: [yellow]{generations}[/yellow] | Eval Restarts: [yellow]{eval_restarts}[/yellow] | Eval Steps: [yellow]{eval_steps}[/yellow]",
            border_style="magenta",
        )
    )

    problem = NKLandscape(n=n, k=k, seed=seed)
    group_cfg = GroupConfig(n_agents=agents)
    strategy = NeuralLocalSearchStrategy(
        policy_type=policy_type,  # type: ignore
        max_steps=eval_steps,
        seed=seed,
    )

    optimizer = CMAESOptimizer(
        strategy=strategy,
        problem=problem,
        eval_restarts=eval_restarts,
        eval_steps=eval_steps,
        population_size=popsize,
        group_config=group_cfg,
        seed=seed,
    )

    table = Table(title="CMA-ES Generation Progress", border_style="magenta")
    table.add_column("Gen", justify="center", style="bold yellow")
    table.add_column("Best Fitness", justify="right", style="bold green")
    table.add_column("Gen Best", justify="right", style="cyan")
    table.add_column("Gen Mean", justify="right", style="white")
    table.add_column("Gen Std", justify="right", style="dim")
    table.add_column("Sigma", justify="right", style="magenta")
    table.add_column("Time", justify="right", style="dim")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Evolving policy...", total=generations)

        def gen_callback(rec: GenerationRecord) -> None:
            progress.update(
                task,
                advance=1,
                description=f"[magenta]Gen {rec.generation + 1}/{generations} | Best: [bold green]{rec.best_fitness:.4f}[/bold green]",
            )
            table.add_row(
                str(rec.generation + 1),
                f"{rec.best_fitness:.4f}",
                f"{rec.gen_best_fitness:.4f}",
                f"{rec.gen_mean_fitness:.4f}",
                f"{rec.gen_std_fitness:.4f}",
                f"{rec.sigma:.3f}",
                f"{rec.elapsed_time:.2f}s",
            )

        result = optimizer.train(
            n_generations=generations,
            checkpoint_dir=checkpoint_dir,
            callback=gen_callback,
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Training Complete![/bold green] Best Fitness: [bold cyan]{result.best_fitness:.4f}[/bold cyan]"
    )
    if result.checkpoint_path:
        console.print(
            f"[bold green]✓ Saved checkpoint to:[/bold green] [cyan]{result.checkpoint_path}[/cyan]"
        )

    if plot:
        fig_dir = checkpoint_dir / "plots"
        fig_dir.mkdir(parents=True, exist_ok=True)
        conv_plot = fig_dir / "training_convergence.png"
        feat_plot = fig_dir / "feature_importance.png"
        plot_training_convergence(result.history, conv_plot)
        plot_neural_feature_importance(strategy, feat_plot)
        console.print(f"[bold green]✓ Saved training plots to:[/bold green] [cyan]{fig_dir}[/cyan]")


@app.command()
def inspect_nn(
    checkpoint: Path | None = typer.Option(None, "--ckpt", help="Path to trained checkpoint JSON"),
    policy_type: str = typer.Option(
        "linear", "--policy-type", help="Policy type (if no checkpoint)"
    ),
    n: int = typer.Option(10, "-n", help="NK dimension N for diagnostic sample"),
    k: int = typer.Option(2, "-k", help="NK epistasis K"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Save feature importance plot"),
    output_dir: Path = typer.Option(Path("diagnostics"), "--out", help="Output directory"),
) -> None:
    """Inspect what inputs a Neural Policy attends to (weights, sensitivities, explanations)."""
    console.print(
        Panel.fit(
            "[bold cyan]Neural Local Search Policy Inspection[/bold cyan]", border_style="cyan"
        )
    )

    strategy = NeuralLocalSearchStrategy(policy_type=policy_type)  # type: ignore

    if checkpoint is not None and checkpoint.exists():
        from neurols.training.base import TrainResult

        res = TrainResult.load(checkpoint)
        strategy.set_weights_vector(res.best_weights)
        console.print(
            f"[green]Loaded checkpoint from:[/green] [cyan]{checkpoint}[/cyan] (Best Fitness: {res.best_fitness:.4f})"
        )

    problem = NKLandscape(n=n, k=k, seed=123)
    sample_state = problem.sample_initial_states((1,))

    # 1. Feature Weights / Sensitivities Table
    sensitivities = compute_feature_sensitivities(strategy, sample_state, problem)
    table = Table(title="Policy Input Feature Saliency / Weights", border_style="cyan")
    table.add_column("Feature Channel", style="bold white")
    table.add_column("Sensitivity / Weight", justify="right", style="cyan")

    for feat, sens in sorted(sensitivities.items(), key=lambda x: x[1], reverse=True):
        table.add_row(feat, f"{sens:+.4f}")
    console.print(table)

    # 2. Decision Explanation on Sample State
    decision = explain_decision(strategy, sample_state, problem)
    console.print("\n[bold yellow]Sample Decision Analysis:[/bold yellow]")
    console.print(
        f"  • Selected candidate move: [bold cyan]bit {decision['chosen_move']}[/bold cyan]"
    )
    console.print(
        f"  • Is improving move: [{'green' if decision['is_improving'] else 'red'}]{decision['is_improving']}[/]"
    )
    console.print(f"  • Expected Delta Fitness: [cyan]{decision['chosen_delta']:+.4f}[/cyan]")
    console.print("  • Feature breakdown of chosen move:")
    for k_feat, v_feat in decision["features_of_chosen_move"].items():
        console.print(f"      - {k_feat:<22}: [bold]{v_feat:+.4f}[/bold]")

    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        p = plot_neural_feature_importance(strategy, output_dir / "policy_features.png")
        console.print(
            f"\n[bold green]✓ Saved feature importance plot to:[/bold green] [cyan]{p}[/cyan]"
        )


@app.command()
def inspect_nk(
    n: int = typer.Option(20, "-n", help="Number of bits N"),
    k: int = typer.Option(3, "-k", help="Epistatic interactions K"),
    topology: str = typer.Option(
        "adjacent", "--topology", help="Topology ('adjacent' or 'random')"
    ),
    samples: int = typer.Option(2000, "--samples", help="Monte Carlo sample size"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Analyze NK Landscape ruggedness, density of local optima, and properties."""
    console.print(
        Panel.fit(
            f"[bold green]NK Landscape Analyzer: N={n}, K={k}[/bold green]", border_style="green"
        )
    )

    with console.status("[cyan]Estimating landscape ruggedness via Monte Carlo sampling..."):
        landscape = NKLandscape(n=n, k=k, interaction_type=topology, seed=seed)
        stats = landscape.analyze_ruggedness(n_samples=samples)

    table = Table(
        title=f"Landscape Properties (N={n}, K={k}, topology={topology})", border_style="green"
    )
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Mean Random Fitness", f"{stats['mean_fitness']:.4f}")
    table.add_row("Fitness Variance / Std", f"{stats['std_fitness']:.4f}")
    table.add_row("Min Sampled Fitness", f"{stats['min_fitness']:.4f}")
    table.add_row("Max Sampled Fitness", f"{stats['max_fitness']:.4f}")
    table.add_row("Estimated Local Optima Density", f"{stats['local_optima_density'] * 100:.2f}%")
    table.add_row(
        "Average Improving Moves from Random State", f"{stats['avg_improving_moves']:.1f} / {n}"
    )

    console.print(table)


@app.command()
def ablation(
    n: int = typer.Option(20, "-n", help="NK dimension N"),
    k: int = typer.Option(3, "-k", help="NK epistasis K"),
    generations: int = typer.Option(
        8, "--generations", "-g", help="CMA-ES training generations per variant"
    ),
    agents: int = typer.Option(4, "--agents", "-m", help="Agents per group (multi-agent)"),
    restarts: int = typer.Option(12, "--restarts", "-r", help="Evaluation restarts"),
    steps: int = typer.Option(50, "--steps", help="Steps per rollout"),
    output_dir: Path = typer.Option(Path("experiments/ablation"), "--out", help="Output directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run an automated feature ablation study: pre-train neural variants with CMA-ES, then benchmark."""
    console.print(
        Panel.fit(
            f"[bold cyan]NeuroLS Feature Ablation Study[/bold cyan]\n"
            f"Landscape: [green]NK(N={n}, K={k})[/green] | Agents per group: [yellow]{agents}[/yellow]\n"
            f"Train Generations: [magenta]{generations}[/magenta] | Test Restarts: [yellow]{restarts}[/yellow]",
            border_style="cyan",
        )
    )

    train_problem = NKLandscape(n=n, k=k, seed=seed)
    test_problem = NKLandscape(n=n, k=k, seed=seed + 500)
    group_cfg = GroupConfig(n_agents=agents)

    study = AblationStudy(
        train_problem=train_problem,
        test_problem=test_problem,
        group_config=group_cfg,
        train_generations=generations,
        eval_restarts=6,
        eval_steps=min(steps, 30),
        cma_popsize=8,
        test_restarts=restarts,
        test_steps=steps,
        seed=seed,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Evolving & evaluating feature variants...", total=len(DEFAULT_ABLATION_SUITE)
        )

        def ablate_cb(name: str, current: int, total: int) -> None:
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Training [bold yellow]{name}[/bold yellow] ({current}/{total})...",
            )

        result = study.run(
            include_classical_baselines=True,
            progress_callback=ablate_cb,
            output_dir=output_dir,
        )

    # Render Leaderboard Table
    table = Table(title=f"Ablation Results: NK(N={n}, K={k}, Agents={agents})", border_style="cyan")
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Strategy / Feature Variant", style="bold white")
    table.add_column("Mean Best Fit", justify="right", style="green")
    table.add_column("Std Fit", justify="right", style="dim")
    table.add_column("Max Fit", justify="right", style="cyan")
    table.add_column("AUC", justify="right", style="magenta")

    for rank, row in enumerate(result.summary_table, 1):
        table.add_row(
            str(rank),
            row["strategy"],
            f"{row['mean_best_fitness']:.4f}",
            f"{row['std_best_fitness']:.4f}",
            f"{row['max_best_fitness']:.4f}",
            f"{row['auc_fitness']:.4f}",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Ablation plots and JSON saved to:[/bold green] [cyan]{output_dir}[/cyan]"
    )


@app.command(name="multi-agent-study")
def multi_agent_study(
    n: NOption = 32,
    k: KOption = 4,
    agents: str = typer.Option(
        "1,4,10", "--agents", "-m", help="Comma-separated agent counts (e.g. '1,4,10')"
    ),
    preset: str = typer.Option(
        "per-action",
        "--preset",
        "-p",
        help="Study suite preset: 'per-action' (action scorers with/without basic sharing), 'architectures', 'all'",
    ),
    steps: NbIterationsOption = None,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    train_generations: NbGenerationsOption = 10,
    restarts: NbRestartsOption = 10,
    output_dir: OutputDirOption = Path("experiments/multi_agent_study"),
    seed: SeedOption = 42,
) -> None:
    """Run a comparative multi-agent study across models, agent counts, and sharing modes."""
    agent_counts = [int(a.strip()) for a in agents.split(",") if a.strip()]
    stopping = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )
    console.print(
        Panel.fit(
            f"[bold cyan]NeuroLS Multi-Agent & Multi-Model Comparative Study[/bold cyan]\n"
            f"Landscape: [green]NK(N={n}, K={k})[/green] | Preset: [yellow]{preset}[/yellow] | Agent Counts: [yellow]{agent_counts}[/yellow]\n"
            f"Stopping: [magenta]{stopping}[/magenta] | Train Generations: [magenta]{train_generations}[/magenta] | Restarts: [yellow]{restarts}[/yellow]",
            border_style="cyan",
        )
    )

    train_problem = NKLandscape(n=n, k=k, seed=seed)
    test_problem = NKLandscape(n=n, k=k, seed=seed + 500)
    models = get_study_suite(preset)

    study = MultiAgentStudy(
        train_problem=train_problem,
        test_problem=test_problem,
        agent_counts=agent_counts,
        sharing_modes=[True],
        models=None,
        stopping_criteria=stopping,
        train_generations=train_generations,
        test_restarts=restarts,
        seed=seed,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running comparative conditions...", total=None)

        def cb(cond_name: str, current: int, total: int) -> None:
            progress.update(
                task,
                total=total,
                completed=current,
                description=f"[cyan]Evaluating [bold yellow]{cond_name}[/bold yellow] ({current}/{total})...",
            )

        result = study.run(output_dir=output_dir, progress_callback=cb)

    # Render Leaderboard Table
    table = Table(
        title=f"Multi-Agent Study Leaderboard: NK(N={n}, K={k}, Suite={preset})",
        border_style="cyan",
    )
    table.add_column("Model / Architecture", style="bold white")
    table.add_column("M", justify="center", style="yellow")
    table.add_column("Sharing", justify="center", style="cyan")
    table.add_column("Team Best Fit", justify="right", style="bold green")
    table.add_column("Std Fit", justify="right", style="dim")
    table.add_column("Diversity", justify="right", style="magenta")
    table.add_column("Steps", justify="right", style="white")
    table.add_column("Evals", justify="right", style="cyan")
    table.add_column("Time (s)", justify="right", style="dim")

    sorted_records = sorted(result.records, key=lambda r: r.mean_team_best_fitness, reverse=True)
    for rec in sorted_records:
        sharing_label = (
            "—"
            if not rec.is_group_aware
            else ("[green]✓[/green]" if rec.share_info else "[red]✗[/red]")
        )
        table.add_row(
            rec.model_name,
            str(rec.n_agents),
            sharing_label,
            f"{rec.mean_team_best_fitness:.4f}",
            f"{rec.std_team_best_fitness:.4f}",
            f"{rec.mean_diversity:.2f}",
            f"{rec.mean_steps:.1f}",
            f"{rec.mean_evaluations:,}",
            f"{rec.mean_elapsed_seconds:.2f}",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Multi-agent study completed![/bold green] Results and plots saved to: [cyan]{output_dir}[/cyan]"
    )
    console.print("  • Team Best Fitness vs. Agent Count : [cyan]fitness_vs_agents.png[/cyan]")
    console.print("  • Sharing Collaboration Gain Bar    : [cyan]sharing_gain.png[/cyan]")
    console.print("  • Step-by-Step Search Trajectories : [cyan]step_convergence.png[/cyan]")
    console.print("  • Multi-Agent Diversity Dynamics   : [cyan]diversity_dynamics.png[/cyan]")
    if (output_dir / "training_curves.png").exists():
        console.print("  • Neuroevolution Training Curves    : [cyan]training_curves.png[/cyan]")


def main() -> None:
    """Entry point for CLI script."""
    app()


if __name__ == "__main__":
    main()
