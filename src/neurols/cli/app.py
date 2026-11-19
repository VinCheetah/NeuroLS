"""Command Line Interface for NeuroLS using Typer and Rich.

Provides a unified, streamlined set of commands to:
- run: Fast execution of single strategies or saved models
- train: Evolve neural policies with CMA-ES, register checkpoints locally with full reproducibility manifests, and plot training curves
- benchmark: Unified comparative benchmark for classical baselines, ablated neural variants, model suites, or multi-agent sweeps
- explore: In-depth strategy diagnostics (trajectory 2D projection, bitstring evolution heatmap, neighbor rank distributions, statistical multi-restarts)
- models: Inspect, list, and manage saved local model checkpoints
- features: Browse available neural feature channels and group keywords
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from neurols.analysis.experiment import ExperimentLogger
from neurols.analysis.multi_agent_study import (
    ARCHITECTURES_SUITE,
    BASELINES_SUITE,
    FULL_COMPARISON_SUITE,
    PER_ACTION_STUDY_SUITE,
    ModelSpec,
    MultiAgentStudy,
)
from neurols.analysis.plotting import (
    plot_action_ranks,
    plot_benchmark_dashboard,
    plot_bitstring_evolution,
    plot_convergence_curves,
    plot_exploration_dashboard,
    plot_final_fitness_boxplots,
    plot_fitness_evolution,
    plot_fitness_vs_agents,
    plot_move_deltas,
    plot_multi_training_convergence,
    plot_multiagent_agent_trajectories,
    plot_multiagent_dashboard,
    plot_multiagent_dispersion_heatmap,
    plot_multiagent_diversity,
    plot_neural_feature_importance,
    plot_search_trajectory_2d,
    plot_sharing_gain_bars,
    plot_step_trajectories,
    plot_training_convergence,
    resolve_plot_names,
)
from neurols.cli.args import (
    EvalModeOption,
    EvalRestartsOption,
    EvalStepsOption,
    FeaturesOption,
    KOption,
    ModelsOption,
    NbAgentsOption,
    NbBudgetOption,
    NbGenerationsOption,
    NbIterationsOption,
    NbRestartsOption,
    NbTimeOption,
    NOption,
    OutputDirOption,
    PlotOption,
    PlotsOption,
    PopulationSizeOption,
    SeedOption,
    SharedInfoOption,
    Sigma0Option,
    StrategiesOption,
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
from neurols.strategies.neural import LinearScorer, NeuralLocalSearchStrategy
from neurols.strategies.neural.features import (
    FEATURE_GROUPS,
    FEATURE_REGISTRY,
    FeatureConfig,
)
from neurols.training.base import GenerationRecord
from neurols.training.cma_es import CMAESOptimizer
from neurols.training.registry import (
    DEFAULT_CHECKPOINT_ROOT,
    delete_saved_model,
    list_saved_models,
    load_model_checkpoint,
    save_model_checkpoint,
)
from neurols.types import StoppingCriteria

app = typer.Typer(
    name="neurols",
    help="NeuroLS: Unified Modular Local Search & Neuroevolution Framework",
    add_completion=True,
)
console = Console()


# ============================================================================
# Helpers: Model & Strategy Resolution
# ============================================================================


def make_group_config(n_agents: int = 1, shared_info: bool = True) -> GroupConfig:
    """Construct GroupConfig with proper sharing flags."""
    if not shared_info:
        return GroupConfig(
            n_agents=n_agents,
            share_distances=False,
            share_distance_mean=False,
            share_delta_scores=False,
            share_delta_mean_score=False,
            share_ranks=False,
            share_approachs=False,
            share_mean_variable=False,
            share_best_variable=False,
        )
    return GroupConfig(n_agents=n_agents)


def format_strategy_name(name: str, n_agents: int) -> str:
    """Format strategy name to strictly follow '{strategy_name}_M{n_agents}'."""
    clean = re.sub(r"_M\d+$", "", name.strip())
    return f"{clean}_M{n_agents}"


def parse_model_spec(spec_str: str, default_name_idx: int = 1) -> ModelSpec:
    """Parse a flexible CLI model specification string into a ModelSpec."""
    spec_str = spec_str.strip()

    # 1. Key-value format: "name=M1,mode=per_action,policy=mlp,features=@action_basic,agents=4"
    if "=" in spec_str:
        tokens = [t.strip() for t in spec_str.replace(";", ",").split(",") if t.strip()]
        params: dict[str, str] = {}
        features_list: list[str] = []
        for token in tokens:
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("feature", "features", "feat"):
                    features_list.append(v)
                else:
                    params[k] = v
            else:
                features_list.append(token)

        name = params.get("name", f"Model_{default_name_idx}")
        agents_str = params.get("agents", params.get("m", params.get("n_agents", "1")))
        try:
            model_agents = int(agents_str)
        except ValueError:
            model_agents = 1

        m_type = params.get("type", params.get("model_type", "neural")).lower()
        if m_type == "classical":
            return ModelSpec(
                name=name,
                model_type="classical",
                n_agents=model_agents,
                description=params.get("description", ""),
            )

        eval_mode = params.get("mode", params.get("eval_mode", "per_action"))
        default_policy = "mlp" if eval_mode == "per_action" else "invariant_nnet"
        policy_type = params.get(
            "policy", params.get("policy_type", params.get("arch", default_policy))
        )
        features_str = ",".join(features_list) if features_list else None
        return ModelSpec(
            name=name,
            model_type="neural",
            policy_type=policy_type,
            eval_mode=eval_mode,
            features=features_str,
            n_agents=model_agents,
            description=params.get("description", f"{eval_mode} {policy_type} policy"),
        )

    # 2. Shorthand colon format: "per_action:mlp:@action_basic" or "exp:model"
    parts = [p.strip() for p in spec_str.split(":") if p.strip()]
    if len(parts) == 1:
        token_lower = parts[0].lower()
        if any(cls_name in token_lower for cls_name in ("hill", "first", "tabu", "simulated", "anneal", "random", "steepest")):
            return ModelSpec(name=parts[0], model_type="classical", description="Classical baseline")
        elif parts[0] in ("per_action", "all_actions"):
            eval_mode = parts[0]
            policy = "mlp" if eval_mode == "per_action" else "invariant_nnet"
            return ModelSpec(
                name=f"Model_{default_name_idx}",
                model_type="neural",
                policy_type=policy,
                eval_mode=eval_mode,
            )
        elif parts[0].endswith(".json"):
            return ModelSpec(
                name=Path(parts[0]).stem,
                model_type="neural",
                description=f"Saved model path: {parts[0]}",
            )
        else:
            return ModelSpec(
                name=f"Model_{default_name_idx}",
                model_type="neural",
                policy_type="linear",
                eval_mode="per_action",
                features=parts[0],
            )
    elif len(parts) == 2:
        if parts[0] in ("per_action", "all_actions"):
            eval_mode = parts[0]
            policy = parts[1]
            return ModelSpec(
                name=f"Model_{default_name_idx}",
                model_type="neural",
                policy_type=policy,
                eval_mode=eval_mode,
            )
        elif parts[1] == "classical":
            return ModelSpec(name=parts[0], model_type="classical")
        else:
            # Check if this could be an exp:model saved checkpoint reference
            return ModelSpec(
                name=parts[1],
                model_type="neural",
                policy_type="mlp",
                eval_mode="per_action",
                description=f"Saved model: {spec_str}",
            )
    else:
        name = parts[0]
        eval_mode = parts[1]
        policy = parts[2]
        feat = ":".join(parts[3:]) if len(parts) > 3 else None
        return ModelSpec(
            name=name,
            model_type="neural",
            policy_type=policy,
            eval_mode=eval_mode,
            features=feat,
        )


def _get_strategy(name: str, **kwargs: Any) -> Any:
    """Resolve a strategy by classical name, saved registry query, or neural specification."""
    name_clean = name.strip()
    name_lower = name_clean.lower()

    # 1. Classical strategies
    if "hill" in name_lower or "steepest" in name_lower:
        return HillClimber()
    elif "first" in name_lower:
        return FirstImprovement(order=kwargs.get("order", "random"), seed=kwargs.get("seed", None))
    elif "tabu" in name_lower:
        return TabuSearch(tenure=kwargs.get("tenure", 8))
    elif "simulated" in name_lower or "anneal" in name_lower:
        return SimulatedAnnealing(
            initial_temperature=kwargs.get("temperature", 0.1),
            cooling_rate=kwargs.get("cooling_rate", 0.98),
            max_steps=kwargs.get("steps", 500) or 500,
            seed=kwargs.get("seed", None),
        )
    elif "random" in name_lower:
        return RandomWalk(
            exploration_rate=kwargs.get("exploration_rate", 1.0), seed=kwargs.get("seed", None)
        )

    # 2. Check if name refers to a locally saved checkpoint in registry
    ckpt_dir = kwargs.get("checkpoint_dir", DEFAULT_CHECKPOINT_ROOT)
    try:
        strat, _ = load_model_checkpoint(
            name_clean,
            base_dir=ckpt_dir,
            override_n_actions=kwargs.get("n_actions", None),
            override_n_agents=kwargs.get("n_agents", None),
        )
        return strat
    except FileNotFoundError:
        pass

    # 3. Ad-hoc Neural strategy
    if "neural" in name_lower or "nn" in name_lower:
        eval_mode = kwargs.get("eval_mode", "per_action")
        policy_type = kwargs.get("policy_type", "linear")
        features = kwargs.get("features", None)
        n_actions = kwargs.get("n_actions", 20)
        n_agents = kwargs.get("n_agents", 1)
        steps = kwargs.get("steps", 100)
        seed = kwargs.get("seed", None)
        return NeuralLocalSearchStrategy(
            eval_mode=eval_mode,
            policy_type=policy_type,
            features=features,
            n_actions=n_actions,
            n_agents=n_agents,
            max_steps=steps,
            seed=seed,
            name=name_clean,
        )

    raise typer.BadParameter(f"Unknown strategy or saved model: '{name}'")


# ============================================================================
# COMMAND 1: RUN
# ============================================================================


@app.command()
def run(
    strategy: StrategyOption = "HillClimber",
    model: str | None = typer.Option(
        None,
        "-M",
        "--model",
        help="Saved model query ('exp:model', path.json) or inline spec",
    ),
    mode: EvalModeOption = "per_action",
    policy_type: str = typer.Option(
        "linear",
        "--policy-type",
        help="Policy architecture (linear, mlp, invariant_nnet, deepsets, dense)",
    ),
    features: FeaturesOption = None,
    n: NOption = 32,
    k: KOption = 4,
    steps: NbIterationsOption = None,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    restarts: NbRestartsOption = 10,
    agents: NbAgentsOption = 1,
    shared_info: SharedInfoOption = True,
    train: bool = typer.Option(
        False,
        "--train/--no-train",
        "--train-nn/--no-train-nn",
        help="Train the neural strategy with CMA-ES before running rollout",
    ),
    train_generations: int = typer.Option(15, "--train-generations", "-g", help="CMA-ES generations if training"),
    train_popsize: int = typer.Option(10, "--train-popsize", "-p", help="CMA-ES population size if training"),
    train_eval_steps: int = typer.Option(30, "--train-eval-steps", help="Rollout steps per training evaluation"),
    train_eval_restarts: int = typer.Option(6, "--train-eval-restarts", help="Restarts per training candidate"),
    train_sigma0: float = typer.Option(0.3, "--train-sigma0", help="CMA-ES initial mutation step size"),
    plot: PlotOption = False,
    output_dir: OutputDirOption = Path("experiments/run"),
    seed: SeedOption = 42,
) -> None:
    """Execute a local search rollout on an NK Landscape and display summary statistics."""
    crit_info = []
    if steps is not None:
        crit_info.append(f"Steps: [yellow]{steps}[/yellow]")
    if budget is not None:
        crit_info.append(f"Budget: [yellow]{budget:,}[/yellow]")
    if max_time is not None:
        crit_info.append(f"Max Time: [yellow]{max_time}s[/yellow]")
    crit_str = " | ".join(crit_info) if crit_info else "Steps: [yellow]100[/yellow]"

    strat_label = model if model else strategy
    train_label = f" | Train: [magenta]{train_generations} gens[/magenta]" if train else ""
    console.print(
        Panel.fit(
            f"[bold green]NeuroLS Local Search Rollout[/bold green]\n"
            f"Strategy: [cyan]{strat_label}[/cyan] | Landscape: [cyan]NK(N={n}, K={k})[/cyan]\n"
            f"Restarts: [yellow]{restarts}[/yellow] | Agents: [magenta]{agents}[/magenta] (Sharing: {shared_info}){train_label}\n"
            f"Stopping: {crit_str} | Seed: [dim]{seed}[/dim]",
            border_style="green",
        )
    )

    problem = NKLandscape(n=n, k=k, seed=seed)
    group_cfg = make_group_config(n_agents=agents, shared_info=shared_info)

    # Resolve strategy
    strat_query = model if model else strategy
    strat_instance = _get_strategy(
        strat_query,
        eval_mode=mode,
        policy_type=policy_type,
        features=features,
        n_actions=n,
        n_agents=agents,
        steps=steps or 100,
        seed=seed,
    )
    strat_instance.name = format_strategy_name(strat_instance.name, agents)

    # Optional pre-training for neural strategies
    if train and isinstance(strat_instance, NeuralLocalSearchStrategy) and train_generations > 0:
        console.print(f"[bold magenta]Pre-training {strat_instance.name} using CMA-ES ({train_generations} gens)...[/bold magenta]")
        train_problem = NKLandscape(n=n, k=k, seed=seed + 1000)
        optimizer = CMAESOptimizer(
            strategy=strat_instance,
            problem=train_problem,
            eval_restarts=train_eval_restarts,
            eval_steps=train_eval_steps,
            population_size=train_popsize,
            sigma0=train_sigma0,
            group_config=group_cfg,
            seed=seed,
        )
        res = optimizer.train(n_generations=train_generations)
        console.print(f"[bold green]✓ Pre-training complete![/bold green] Best fitness: [bold cyan]{res.best_fitness:.4f}[/bold cyan]")

    stopping_criteria = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )

    runner = BatchedRunner(
        problem=problem,
        strategy=strat_instance,
        stopping_criteria=stopping_criteria,
        group_config=group_cfg,
        record_history=True,
        record_states=True,
    )

    with console.status("[bold cyan]Running trajectory rollouts..."):
        trajectory = runner.run(n_instances=1, n_restarts=restarts, seed=seed)

    summary = trajectory.get_summary()

    table = Table(title="Execution Summary", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Mean Best Fitness", f"{summary['mean_best_fitness']:.4f}")
    table.add_row("Std Best Fitness", f"{summary['std_best_fitness']:.4f}")
    table.add_row("Max Best Fitness", f"{summary['max_best_fitness']:.4f}")
    table.add_row("Min Best Fitness", f"{summary['min_best_fitness']:.4f}")
    table.add_row("Mean Steps Taken", f"{summary['mean_steps']:.1f}")
    table.add_row("Total Evaluations", f"{summary['total_evaluations']:,}")
    table.add_row("Elapsed Seconds", f"{summary['elapsed_seconds']:.3f}s")
    table.add_row("Termination Reason", summary["termination_reason"])

    console.print(table)

    # Visualizations
    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_exploration_dashboard(
            trajectory,
            output_dir / "exploration_dashboard.png",
            title=f"{strat_instance.name} Rollout Dashboard",
        )
        if agents > 1:
            plot_multiagent_dashboard(
                trajectory,
                output_dir / "multiagent_dashboard.png",
                n_agents=agents,
                title=f"{strat_instance.name} Multi-Agent Dashboard",
            )
            plot_multiagent_agent_trajectories(
                trajectory,
                output_dir / "multiagent_trajectories.png",
                n_agents=agents,
                title=f"{strat_instance.name} Agent Trajectories",
            )
            plot_multiagent_dispersion_heatmap(
                trajectory,
                output_dir / "multiagent_dispersion.png",
                n_agents=agents,
                title=f"{strat_instance.name} Dispersion Heatmap",
            )
            plot_multiagent_diversity(
                trajectory,
                output_dir / "diversity_dynamics.png",
                title=f"{strat_instance.name} Population Diversity",
            )
        console.print(f"[bold green]✓ Saved rollout plots to:[/bold green] [cyan]{output_dir}[/cyan]")


# ============================================================================
# COMMAND 2: TRAIN
# ============================================================================


@app.command()
def train(
    mode: EvalModeOption = "per_action",
    policy_type: str = typer.Option(
        "linear",
        "--policy-type",
        help="Policy architecture (linear, mlp, invariant_nnet, deepsets, peer_invariant, dense)",
    ),
    features: FeaturesOption = None,
    n: NOption = 32,
    k: KOption = 4,
    generations: NbGenerationsOption = 20,
    popsize: PopulationSizeOption = 10,
    eval_restarts: int = typer.Option(
        8, "--eval-restarts", "-r", help="Restarts per evaluation candidate"
    ),
    eval_steps: int = typer.Option(40, "--eval-steps", help="Steps per evaluation rollout"),
    sigma0: float = typer.Option(0.3, "--sigma0", help="Initial mutation step size for CMA-ES"),
    agents: NbAgentsOption = 1,
    shared_info: SharedInfoOption = True,
    exp_name: str = typer.Option(
        "default_experiment",
        "--exp-name",
        "--experiment",
        "-e",
        help="Experiment suite name in local registry",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Unique model name (defaults to auto-generated from architecture)",
    ),
    checkpoint_dir: Path = typer.Option(
        Path("checkpoints"), "--ckpt-dir", help="Directory for saving checkpoints"
    ),
    plots: PlotsOption = None,
    plot: PlotOption = True,
    seed: SeedOption = 42,
) -> None:
    """Train a Neural Local Search policy using CMA-ES and register checkpoint with full reproducibility manifest."""
    feat_desc = f"Features: [yellow]{features}[/yellow]" if features else "Default Features"
    assigned_name = format_strategy_name(model_name or f"{policy_type}_{mode}_N{n}_K{k}", agents)

    problem = NKLandscape(n=n, k=k, seed=seed)
    group_cfg = make_group_config(n_agents=agents, shared_info=shared_info)
    strategy = NeuralLocalSearchStrategy(
        eval_mode=mode,
        policy_type=policy_type,  # type: ignore
        features=features,
        n_actions=n,
        n_agents=agents,
        max_steps=eval_steps,
        seed=seed,
        name=assigned_name,
    )

    # Evaluate baseline HillClimber on the same instance for comparison
    hc_runner = BatchedRunner(
        problem=problem,
        strategy=HillClimber(),
        max_steps=eval_steps,
        group_config=group_cfg,
    )
    hc_baseline = hc_runner.run(n_restarts=eval_restarts, seed=seed).average_best_fitness

    console.print(
        Panel.fit(
            f"[bold magenta]CMA-ES Neuroevolution Training & Model Registration[/bold magenta]\n"
            f"Experiment: [cyan]{exp_name}[/cyan] | Model: [bold white]{assigned_name}[/bold white]\n"
            f"Mode: [cyan]{mode}[/cyan] | Policy: [cyan]{policy_type}[/cyan] ({strategy.policy.num_parameters} params) | {feat_desc}\n"
            f"Landscape: [green]NK(N={n}, K={k})[/green] | Agents: [magenta]{agents}[/magenta] (Sharing: {shared_info})\n"
            f"Generations: [yellow]{generations}[/yellow] | Popsize: [yellow]{popsize}[/yellow] | Eval Restarts: [yellow]{eval_restarts}[/yellow] | Sigma0: [yellow]{sigma0}[/yellow]\n"
            f"Baseline HillClimber Benchmark: [bold green]{hc_baseline:.4f}[/bold green]",
            border_style="magenta",
        )
    )

    optimizer = CMAESOptimizer(
        strategy=strategy,
        problem=problem,
        eval_restarts=eval_restarts,
        eval_steps=eval_steps,
        population_size=popsize,
        sigma0=sigma0,
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
    gain_over_hc = result.best_fitness - hc_baseline
    console.print(
        f"\n[bold green]✓ Training Complete![/bold green] Best Fitness: [bold cyan]{result.best_fitness:.4f}[/bold cyan] "
        f"([bold {'green' if gain_over_hc >= 0 else 'red'}]{gain_over_hc:+.4f}[/bold {'green' if gain_over_hc >= 0 else 'red'}] vs HillClimber)"
    )

    # Print learned weights for interpretable linear models
    if isinstance(strategy.policy, LinearScorer):
        feat_names = strategy.feature_config.get_feature_names(agents)
        weights_map = strategy.policy.get_feature_weights(feat_names)
        w_table = Table(title="Learned Feature Weights (LinearScorer)", border_style="cyan")
        w_table.add_column("Feature Channel", style="bold white")
        w_table.add_column("Learned Weight", justify="right", style="bold cyan")
        for f_name, w_val in weights_map.items():
            w_table.add_row(f_name, f"{w_val:+.4f}")
        console.print(w_table)

    # Save complete reproducibility manifest to Model Registry
    saved_path = save_model_checkpoint(
        strategy=strategy,
        train_result=result,
        experiment_name=exp_name,
        model_name=assigned_name,
        problem_config={"problem_type": "nk_landscape", "n": n, "k": k, "seed": seed},
        training_config={
            "optimizer": "cma_es",
            "generations": generations,
            "popsize": popsize,
            "eval_restarts": eval_restarts,
            "eval_steps": eval_steps,
            "sigma0": sigma0,
            "seed": seed,
            "baseline_fitness": hc_baseline,
        },
        base_dir=checkpoint_dir,
    )
    console.print(f"[bold green]✓ Registered model to:[/bold green] [cyan]{saved_path}[/cyan]")

    # Training Visualizations
    if plot:
        fig_dir = checkpoint_dir / exp_name / "plots"
        fig_dir.mkdir(parents=True, exist_ok=True)
        conv_plot = fig_dir / "training_convergence.png"
        feat_plot = fig_dir / "feature_importance.png"

        plot_training_convergence(result.history, conv_plot)
        plot_neural_feature_importance(strategy, feat_plot)
        console.print(f"[bold green]✓ Saved training plots to:[/bold green] [cyan]{fig_dir}[/cyan]")


# ============================================================================
# COMMAND 3: BENCHMARK (UNIFIED)
# ============================================================================


@app.command()
def benchmark(
    strategies: StrategiesOption = None,
    models: ModelsOption = None,
    mode: EvalModeOption = "per_action",
    features: FeaturesOption = None,
    n: NOption = 32,
    k: KOption = 4,
    preset: str = typer.Option(
        "standard",
        "--preset",
        "-p",
        help="Benchmark suite preset: 'standard', 'baselines', 'architectures', 'per-action', 'multiagent'",
    ),
    policy_type: str = typer.Option(
        "linear",
        "--policy-type",
        help="Policy architecture for standard preset ('linear', 'mlp', 'deepsets', 'peer_invariant', 'dense')",
    ),
    steps: NbIterationsOption = 100,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    restarts: NbRestartsOption = 20,
    agents: str = typer.Option(
        "1",
        "--agents",
        "-m",
        help="Agent count or comma-separated list of agent counts to sweep (e.g. '1', '1,4,10')",
    ),
    shared_info: bool = typer.Option(
        True,
        "--shared-info/--no-shared-info",
        help="Whether multi-agent groups share peer information",
    ),
    train_nn: TrainingOption = True,
    train_generations: int = typer.Option(15, "--train-generations", "-g", help="CMA-ES training generations"),
    train_popsize: int = typer.Option(10, "--train-popsize", "-p", help="CMA-ES population size"),
    train_eval_steps: int = typer.Option(30, "--train-eval-steps", help="Rollout steps per candidate during training"),
    train_eval_restarts: int = typer.Option(6, "--train-eval-restarts", help="Restarts per candidate during training"),
    train_sigma0: float = typer.Option(0.3, "--train-sigma0", help="CMA-ES initial mutation step size"),
    plots: PlotsOption = None,
    plot: PlotOption = True,
    explore_sample: bool = typer.Option(
        True,
        "--explore-sample/--no-explore-sample",
        help="Generate per-model representative trajectory exploration diagnostics and plots in output_dir/exploration/{model_name}/",
    ),
    output_dir: OutputDirOption = Path("experiments/benchmark"),
    seed: SeedOption = 42,
) -> None:
    """Unified benchmark comparing classical strategies, saved neural models, and multi-agent groups."""
    crit_info = []
    if steps is not None:
        crit_info.append(f"Steps: [yellow]{steps}[/yellow]")
    if budget is not None:
        crit_info.append(f"Budget: [yellow]{budget:,}[/yellow]")
    if max_time is not None:
        crit_info.append(f"Max Time: [yellow]{max_time}s[/yellow]")
    crit_str = " | ".join(crit_info) if crit_info else "Steps: [yellow]100[/yellow]"

    # Parse agent counts
    agent_counts = [int(a.strip()) for a in str(agents).split(",") if a.strip()]
    is_multiagent_sweep = len(agent_counts) > 1

    custom_items = (strategies or []) + (models or [])
    suite_desc = f"{len(custom_items)} custom strategies/models" if custom_items else f"Preset: [yellow]{preset}[/yellow]"
    console.print(
        Panel.fit(
            f"[bold green]NeuroLS Unified Benchmark[/bold green]\n"
            f"Landscape: [cyan]NK(N={n}, K={k})[/cyan] | {suite_desc} | Restarts: [yellow]{restarts}[/yellow] | {crit_str}\n"
            f"Pre-train NN: [magenta]{train_nn} ({train_generations} gens)[/magenta] | Agents: [magenta]{agents}[/magenta] (Sharing: {shared_info})\n"
            f"Explore Sample Runs: [cyan]{explore_sample}[/cyan]",
            border_style="green",
        )
    )

    problem = NKLandscape(n=n, k=k, seed=seed)
    stopping_criteria = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )

    # Determine model suite
    suite: list[ModelSpec] = []
    if custom_items:
        suite = [parse_model_spec(m, i) for i, m in enumerate(custom_items, 1)]
    elif preset in ("baselines", "baseline"):
        suite = list(BASELINES_SUITE)
    elif preset in ("per-action", "per_action"):
        suite = list(PER_ACTION_STUDY_SUITE)
    elif preset in ("architectures", "arch"):
        suite = list(ARCHITECTURES_SUITE)
    elif preset in ("multiagent", "multi"):
        suite = list(FULL_COMPARISON_SUITE)
    elif preset in ("all", "full"):
        suite = list(FULL_COMPARISON_SUITE)
    else:  # standard
        suite = [
            ModelSpec(
                name=f"Neural_{policy_type}",
                model_type="neural",
                policy_type=policy_type,
                eval_mode=mode,
                features=features,
                description=f"Neural {mode} {policy_type} policy",
            ),
            ModelSpec(name="HillClimber", model_type="classical", description="Greedy Steepest Ascent"),
            ModelSpec(name="FirstImprovement", model_type="classical", description="First Improving Move"),
            ModelSpec(name="TabuSearch", model_type="classical", description="Tabu Search"),
            ModelSpec(name="SimulatedAnnealing", model_type="classical", description="Simulated Annealing"),
        ]

    # Instantiate strategies & pre-train neural variants if requested
    primary_agents = agent_counts[0]
    group_cfg = make_group_config(n_agents=primary_agents, shared_info=shared_info)
    strategies: list[Any] = []
    training_histories: dict[str, list[GenerationRecord]] = {}

    for spec_idx, spec in enumerate(suite, 1):
        m_type = getattr(spec, "model_type", "neural")
        spec_name = spec.name
        strat_formatted_name = format_strategy_name(spec_name, primary_agents)

        if m_type == "classical":
            strat = spec.to_strategy(n_actions=n, n_agents=primary_agents, steps=steps or 100, seed=seed)
            strat.name = strat_formatted_name
            strategies.append(strat)
        else:
            # Check if spec_name or description is a saved checkpoint
            loaded_strat = None
            try:
                loaded_strat, _ = load_model_checkpoint(
                    spec_name, override_n_actions=n, override_n_agents=primary_agents
                )
            except Exception:
                pass

            if loaded_strat is not None:
                loaded_strat.name = strat_formatted_name
                strategies.append(loaded_strat)
            else:
                eval_mode = getattr(spec, "eval_mode", mode)
                pol_type = getattr(spec, "policy_type", policy_type)
                feat_spec = getattr(spec, "features", features)

                feat_cfg = FeatureConfig.from_specs(feat_spec, mode=eval_mode)

                n_strat = NeuralLocalSearchStrategy(
                    name=strat_formatted_name,
                    policy_type=pol_type,  # type: ignore
                    eval_mode=eval_mode,
                    feature_config=feat_cfg,
                    n_actions=n,
                    n_agents=primary_agents,
                    max_steps=steps or 100,
                    seed=seed,
                )

                if train_nn and train_generations > 0:
                    train_problem = NKLandscape(n=n, k=k, seed=seed + 1000)
                    optimizer = CMAESOptimizer(
                        strategy=n_strat,
                        problem=train_problem,
                        eval_restarts=train_eval_restarts,
                        eval_steps=train_eval_steps,
                        population_size=train_popsize,
                        sigma0=train_sigma0,
                        group_config=group_cfg,
                        seed=seed,
                    )

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                        console=console,
                    ) as prog:
                        t_task = prog.add_task(
                            f"[bold magenta]Training policy [{spec_idx}/{len(suite)}]: {strat_formatted_name}...[/bold magenta]",
                            total=train_generations,
                        )

                        curr_task = t_task
                        curr_idx = spec_idx
                        curr_name = strat_formatted_name

                        def gen_cb(
                            rec: GenerationRecord,
                            task_id=curr_task,
                            idx=curr_idx,
                            name=curr_name,
                        ) -> None:
                            prog.update(
                                task_id,
                                advance=1,
                                description=f"[bold magenta]Training [{idx}/{len(suite)}]: {name} | Gen {rec.generation + 1}/{train_generations} (Best: [bold green]{rec.best_fitness:.4f}[/bold green])[/bold magenta]",
                            )

                        train_res = optimizer.train(
                            n_generations=train_generations, callback=gen_cb
                        )

                    training_histories[strat_formatted_name] = train_res.history
                    console.print(
                        f"  [bold green]✓ Trained {strat_formatted_name}:[/bold green] Best Fitness = [bold cyan]{train_res.best_fitness:.4f}[/bold cyan]"
                    )

                    # Save checkpoint into registry
                    save_model_checkpoint(
                        strategy=n_strat,
                        train_result=train_res,
                        experiment_name="benchmark_trained",
                        model_name=strat_formatted_name,
                        problem_config={"n": n, "k": k, "seed": seed},
                        training_config={
                            "optimizer": "cma_es",
                            "eval_steps": train_eval_steps,
                            "eval_restarts": train_eval_restarts,
                            "generations": train_generations,
                            "popsize": train_popsize,
                            "sigma0": train_sigma0,
                            "seed": seed,
                        },
                        base_dir=DEFAULT_CHECKPOINT_ROOT,
                    )

                strategies.append(n_strat)

    # Multi-Agent Sweep handling if multiple agent counts given
    if is_multiagent_sweep:
        train_prob = NKLandscape(n=n, k=k, seed=seed)
        test_prob = NKLandscape(n=n, k=k, seed=seed + 500)
        ma_study = MultiAgentStudy(
            train_problem=train_prob,
            test_problem=test_prob,
            models=suite,
            agent_counts=agent_counts,
            sharing_modes=[True] if shared_info else [False],
            test_restarts=restarts,
            stopping_criteria=stopping_criteria,
            train_generations=train_generations if train_nn else 0,
            seed=seed,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as prog_ma:
            ma_task = prog_ma.add_task(
                "[bold cyan]Running Multi-Agent Comparative Sweep...", total=100
            )

            def ma_cb(desc: str, cur: int, total: int) -> None:
                prog_ma.update(
                    ma_task,
                    total=total,
                    completed=cur,
                    description=f"[bold cyan]Multi-Agent Condition [{cur}/{total}]: {desc}[/bold cyan]",
                )

            study_result = ma_study.run(output_dir=output_dir, progress_callback=ma_cb)

        # Leaderboard Table
        table = Table(title="Multi-Agent Study Leaderboard", border_style="green")
        table.add_column("Model", style="bold white")
        table.add_column("Agents (M)", justify="center", style="yellow")
        table.add_column("Sharing", justify="center", style="cyan")
        table.add_column("Mean Team Best", justify="right", style="bold green")
        table.add_column("Std Best", justify="right", style="dim")
        table.add_column("Mean Steps", justify="right", style="white")

        for r in study_result.records:
            table.add_row(
                r.model_name,
                str(r.n_agents),
                "✓" if r.share_info else "✗",
                f"{r.mean_team_best_fitness:.4f}",
                f"{r.std_team_best_fitness:.4f}",
                f"{r.mean_steps:.1f}",
            )
        console.print(table)

        if plot:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_fitness_vs_agents(study_result.records, output_dir / "fitness_vs_agents.png")
            plot_sharing_gain_bars(study_result.records, output_dir / "sharing_gain.png")
            console.print(
                f"[bold green]✓ Saved multi-agent plots to:[/bold green] [cyan]{output_dir}[/cyan]"
            )
        return

    # Standard Single-Agent or Single-M Benchmark Execution
    logger = ExperimentLogger(base_dir=output_dir)
    console.print(
        f"\n[bold cyan]Benchmarking {len(strategies)} strategies ({restarts} restarts each)...[/bold cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bench_task = progress.add_task("[cyan]Benchmarking models...", total=len(strategies))

        def bench_cb(
            strat_name: str, current_idx: int, total_count: int, summary: dict[str, Any]
        ) -> None:
            if not summary:
                progress.update(
                    bench_task,
                    description=f"[bold cyan]Benchmarking [{current_idx}/{total_count}]: {strat_name} ({restarts} restarts)...[/bold cyan]",
                )
            else:
                progress.update(bench_task, advance=1)
                m_fit = summary.get("mean_best_fitness", 0.0)
                s_fit = summary.get("std_best_fitness", 0.0)
                m_stp = summary.get("mean_steps", 0.0)
                sec = summary.get("elapsed_seconds", 0.0)
                console.print(
                    f"  [bold green]✓[/bold green] [bold white]{strat_name}[/bold white]: "
                    f"Mean Best = [bold cyan]{m_fit:.4f}[/bold cyan] (±{s_fit:.4f}) | Steps = {m_stp:.1f} in {sec:.2f}s"
                )

        result = logger.run_comparison(
            experiment_name=f"benchmark_N{n}_K{k}_{preset}",
            problem=problem,
            strategies=strategies,
            n_restarts=restarts,
            max_steps=steps,
            stopping_criteria=stopping_criteria,
            group_config=group_cfg,
            progress_callback=bench_cb,
            seed=seed,
        )

    # Leaderboard Table
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

    # Produce Selected Plots
    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = resolve_plot_names(plots, context="benchmark")

        if "convergence" in selected:
            plot_convergence_curves(
                result.trajectories,
                output_dir / "convergence_comparison.png",
                title=f"Convergence: NK(N={n}, K={k})",
            )
        if "step_trajectories" in selected:
            plot_step_trajectories(
                result.trajectories,
                output_dir / "step_trajectories.png",
                title=f"Step Trajectories: NK(N={n}, K={k})",
            )
        if "boxplots" in selected:
            plot_final_fitness_boxplots(
                result.trajectories,
                output_dir / "boxplot_comparison.png",
                title=f"Fitness Distribution: NK(N={n}, K={k})",
            )
        if "dashboard" in selected:
            plot_benchmark_dashboard(
                result.trajectories,
                output_dir / "benchmark_dashboard.png",
                title=f"Benchmark Dashboard: NK(N={n}, K={k})",
            )
        if "diversity" in selected or primary_agents > 1:
            plot_multiagent_diversity(
                result.trajectories,
                output_dir / "diversity_dynamics.png",
                title=f"Group Diversity: NK(N={n}, K={k})",
            )
        if primary_agents > 1 and result.summary_table:
            best_strat_name = result.summary_table[0]["strategy"]
            best_traj = result.trajectories.get(best_strat_name)
            if best_traj is not None:
                plot_multiagent_dashboard(
                    best_traj,
                    output_dir / "multiagent_dashboard.png",
                    n_agents=primary_agents,
                    title=f"{best_strat_name} Multi-Agent Dashboard",
                )
        if training_histories and "training" in selected:
            if len(training_histories) == 1:
                hist = next(iter(training_histories.values()))
                plot_training_convergence(hist, output_dir / "training_convergence.png")
            else:
                plot_multi_training_convergence(
                    training_histories, output_dir / "training_convergence.png"
                )

        console.print(
            f"[bold green]✓ Generated {len(selected)} benchmark plot(s) in:[/bold green] [cyan]{output_dir}[/cyan]"
        )

    # Sample Run Exploration per Model if requested
    if explore_sample:
        console.print(
            "\n[bold cyan]Tracing representative sample runs for exploration diagnostics...[/bold cyan]"
        )
        explore_base = output_dir / "exploration"
        for strat in strategies:
            strat_explore_dir = explore_base / strat.name
            strat_explore_dir.mkdir(parents=True, exist_ok=True)

            runner_trace = BatchedRunner(
                problem=problem,
                strategy=strat,
                stopping_criteria=stopping_criteria,
                group_config=group_cfg,
                record_history=True,
                record_states=True,
            )
            trace = runner_trace.run(n_instances=1, n_restarts=1, seed=seed)
            s_matrix = trace.get_states_matrix(agent_idx=0)
            c_fit, _ = trace.get_agent_fitness_history(agent_idx=0)
            r_arr = trace.get_action_ranks(agent_idx=0)
            d_arr = trace.get_chosen_deltas(agent_idx=0)
            m_arr = trace.get_chosen_moves(agent_idx=0)

            plot_exploration_dashboard(
                trace,
                strat_explore_dir / "exploration_dashboard.png",
                title=f"{strat.name} Diagnostic Dashboard",
            )
            plot_search_trajectory_2d(
                s_matrix,
                c_fit,
                strat_explore_dir / "search_trajectory_2d.png",
                title=f"{strat.name} 2D Trajectory Path",
            )
            plot_bitstring_evolution(
                s_matrix,
                m_arr,
                strat_explore_dir / "bitstring_evolution.png",
                title=f"{strat.name} Bitstring Dynamics",
            )
            plot_action_ranks(
                r_arr,
                strat_explore_dir / "action_ranks.png",
                title=f"{strat.name} Action Ranks Selection",
            )
            plot_move_deltas(
                d_arr,
                strat_explore_dir / "move_deltas.png",
                title=f"{strat.name} Step Deltas",
            )
            plot_fitness_evolution(
                trace,
                strat_explore_dir / "fitness_evolution.png",
                title=f"{strat.name} Fitness Progression",
            )
            if primary_agents > 1:
                plot_multiagent_dashboard(
                    trace,
                    strat_explore_dir / "multiagent_dashboard.png",
                    n_agents=primary_agents,
                    title=f"{strat.name} Multi-Agent Dashboard",
                )
                plot_multiagent_agent_trajectories(
                    trace,
                    strat_explore_dir / "multiagent_trajectories.png",
                    n_agents=primary_agents,
                    title=f"{strat.name} Agent Trajectories",
                )
                plot_multiagent_dispersion_heatmap(
                    trace,
                    strat_explore_dir / "multiagent_dispersion.png",
                    n_agents=primary_agents,
                    title=f"{strat.name} Dispersion Heatmap",
                )

        console.print(
            f"[bold green]✓ Generated in-depth sample exploration diagnostics for each model in:[/bold green] [cyan]{explore_base}[/cyan]"
        )


# ============================================================================
# COMMAND 4: EXPLORE (IN-DEPTH STRATEGY DEEP DIVE)
# ============================================================================


@app.command()
def explore(
    strategy: StrategyOption = "HillClimber",
    model: str | None = typer.Option(
        None,
        "-M",
        "--model",
        help="Saved model name ('exp:model', 'exp/model', 'path.json') or custom spec",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--ckpt",
        help="Path to trained checkpoint JSON",
    ),
    policy_type: str = typer.Option(
        "linear",
        "--policy-type",
        help="Policy architecture if using ad-hoc neural model (linear, mlp, invariant_nnet, deepsets, dense)",
    ),
    mode: EvalModeOption = "per_action",
    features: FeaturesOption = None,
    n: NOption = 20,
    k: KOption = 3,
    topology: str = typer.Option(
        "adjacent", "--topology", help="NK topology ('adjacent' or 'random')"
    ),
    steps: NbIterationsOption = 60,
    budget: NbBudgetOption = None,
    max_time: NbTimeOption = None,
    restarts: NbRestartsOption = 16,
    agents: NbAgentsOption = 1,
    reduction: str = typer.Option(
        "pca", "--reduction", help="Dimensionality reduction method ('pca' or 'mds')"
    ),
    plots: PlotsOption = None,
    plot: PlotOption = True,
    output_dir: OutputDirOption = Path("diagnostics/explore"),
    checkpoint_dir: Path = typer.Option(
        Path("checkpoints"),
        "--ckpt-dir",
        help="Directory where checkpoints are stored",
    ),
    seed: SeedOption = 42,
) -> None:
    """In-depth strategy exploration: inspect trajectory dynamics, bitstring heatmaps, 2D projections, neighbor action ranks, and average statistical performance."""
    strat_query = str(checkpoint) if checkpoint else (model if model else strategy)

    # 1. Resolve strategy instance
    manifest_meta = None
    if checkpoint is not None:
        strat_instance, manifest_meta = load_model_checkpoint(
            checkpoint, base_dir=checkpoint_dir, override_n_actions=n, override_n_agents=agents
        )
    else:
        try:
            strat_instance, manifest_meta = load_model_checkpoint(
                strat_query, base_dir=checkpoint_dir, override_n_actions=n, override_n_agents=agents
            )
        except Exception:
            strat_instance = _get_strategy(
                strat_query,
                eval_mode=mode,
                policy_type=policy_type,
                features=features,
                n_actions=n,
                n_agents=agents,
                steps=steps or 100,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
            )

    console.print(
        Panel.fit(
            f"[bold cyan]NeuroLS In-Depth Strategy Exploration & Diagnostics[/bold cyan]\n"
            f"Target Strategy: [bold white]{strat_instance.name}[/bold white] | Problem: [green]NK(N={n}, K={k}, Topology={topology})[/green]\n"
            f"Run Trace Steps: [yellow]{steps}[/yellow] | Evaluation Restarts: [yellow]{restarts}[/yellow] | 2D Method: [magenta]{reduction.upper()}[/magenta]",
            border_style="cyan",
        )
    )

    problem = NKLandscape(n=n, k=k, interaction_type=topology, seed=seed)
    group_cfg = GroupConfig(n_agents=agents)
    stopping_criteria = StoppingCriteria(
        max_steps=steps,
        max_evaluations=budget,
        max_time_seconds=max_time,
    )

    # 2. Detailed Single Rollout with State & Rank Recording
    runner_trace = BatchedRunner(
        problem=problem,
        strategy=strat_instance,
        stopping_criteria=stopping_criteria,
        group_config=group_cfg,
        record_history=True,
        record_states=True,
    )

    with console.status("[bold cyan]Tracing representative trajectory & neighbor rankings..."):
        traj_trace = runner_trace.run(n_instances=1, n_restarts=1, seed=seed)

    # Extract single run details
    states = traj_trace.get_states_matrix(agent_idx=0)
    curr_fit, best_fit = traj_trace.get_agent_fitness_history(agent_idx=0)
    ranks = traj_trace.get_action_ranks(agent_idx=0)
    deltas = traj_trace.get_chosen_deltas(agent_idx=0)
    chosen_moves = traj_trace.get_chosen_moves(agent_idx=0)
    run_steps = len(ranks)

    # 3. Multi-restart statistical evaluation
    runner_stat = BatchedRunner(
        problem=problem,
        strategy=strat_instance,
        stopping_criteria=stopping_criteria,
        group_config=group_cfg,
        record_history=False,
    )
    with console.status(
        f"[bold cyan]Evaluating statistical performance across {restarts} restarts..."
    ):
        traj_stat = runner_stat.run(n_instances=1, n_restarts=restarts, seed=seed + 100)

    best_fits_arr = traj_stat.best_fitness.detach().cpu().numpy()
    steps_arr = traj_stat.steps_taken.detach().cpu().numpy()

    # Behavioral Metrics
    greedy_moves = int(np.sum(ranks == 0)) if len(ranks) > 0 else 0
    subopt_moves = int(np.sum(ranks > 0)) if len(ranks) > 0 else 0
    greedy_pct = (100.0 * greedy_moves / max(1, len(ranks))) if len(ranks) > 0 else 100.0

    improving_moves = int(np.sum(deltas > 1e-6)) if len(deltas) > 0 else 0
    neutral_moves = int(np.sum(np.abs(deltas) <= 1e-6)) if len(deltas) > 0 else 0
    worsening_moves = int(np.sum(deltas < -1e-6)) if len(deltas) > 0 else 0

    unique_flipped = len(np.unique(chosen_moves[chosen_moves >= 0])) if len(chosen_moves) > 0 else 0

    # Classify behavior
    if greedy_pct >= 95.0:
        behavior_type = "Strict Greedy Steepest Ascent (Always exploits optimal local gradient)"
    elif worsening_moves > 0:
        behavior_type = (
            "Stochastic / Non-Monotonic Search (Willing to accept downhill steps to escape optima)"
        )
    elif subopt_moves > 0:
        behavior_type = (
            "Stochastic Ascent (Chooses positive moves with diversity / non-greedy exploration)"
        )
    else:
        behavior_type = "Greedy Hill Climbing with Neutral Drift"

    # Display Tables
    trace_table = Table(title="Representative Run Profile (Seed Instance)", border_style="cyan")
    trace_table.add_column("Diagnostic Feature", style="bold white")
    trace_table.add_column("Observation", style="cyan")

    trace_table.add_row("Starting State Fitness", f"{curr_fit[0]:.4f}")
    trace_table.add_row("Final Reached Fitness", f"{curr_fit[-1]:.4f}")
    trace_table.add_row("Peak Discovered Fitness", f"{best_fit[-1]:.4f}")
    trace_table.add_row("Steps Completed", f"{run_steps}")
    trace_table.add_row("Unique Bits Flipped", f"{unique_flipped} / {n} loci")
    trace_table.add_row("Steepest Moves (Rank 0)", f"{greedy_moves} ({greedy_pct:.1f}%)")
    trace_table.add_row("Suboptimal Moves (Rank > 0)", f"{subopt_moves} ({100 - greedy_pct:.1f}%)")
    trace_table.add_row("Improving Moves (Δf > 0)", f"{improving_moves} / {max(1, len(deltas))}")
    trace_table.add_row("Neutral Moves (Δf == 0)", f"{neutral_moves} / {max(1, len(deltas))}")
    trace_table.add_row("Worsening Moves (Δf < 0)", f"{worsening_moves} / {max(1, len(deltas))}")
    trace_table.add_row("Behavioral Classification", f"[bold green]{behavior_type}[/bold green]")

    console.print(trace_table)

    stat_table = Table(
        title=f"Statistical Aggregate Performance ({restarts} Independent Restarts)",
        border_style="green",
    )
    stat_table.add_column("Metric", style="bold white")
    stat_table.add_column("Value", style="green")

    stat_table.add_row("Mean Best Fitness", f"{np.mean(best_fits_arr):.4f}")
    stat_table.add_row("Std Best Fitness", f"{np.std(best_fits_arr):.4f}")
    stat_table.add_row("Median Best Fitness", f"{np.median(best_fits_arr):.4f}")
    stat_table.add_row("Max Best Fitness", f"{np.max(best_fits_arr):.4f}")
    stat_table.add_row("Min Best Fitness", f"{np.min(best_fits_arr):.4f}")
    stat_table.add_row("Average Search Steps", f"{np.mean(steps_arr):.1f}")
    stat_table.add_row("Termination Reason", traj_stat.termination_reason)

    console.print(stat_table)

    # 4. Generate Diagnostic Visualizations
    if plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_plots = resolve_plot_names(plots, context="explore")

        if "exploration_dashboard" in selected_plots:
            plot_exploration_dashboard(
                traj_trace,
                output_dir / "exploration_dashboard.png",
                method=reduction,
                title=f"{strat_instance.name} Deep Dive Dashboard",
            )
        if "trajectory_2d" in selected_plots:
            plot_search_trajectory_2d(
                states,
                curr_fit,
                output_dir / "search_trajectory_2d.png",
                method=reduction,
                title=f"{strat_instance.name} Trajectory Path",
            )
        if "bitstring_evolution" in selected_plots:
            plot_bitstring_evolution(
                states,
                chosen_moves,
                output_dir / "bitstring_evolution.png",
                title=f"{strat_instance.name} Bitstring Dynamics",
            )
        if "action_ranks" in selected_plots:
            plot_action_ranks(
                ranks,
                output_dir / "action_ranks.png",
                title=f"{strat_instance.name} Action Rank Selection",
            )
        if "fitness_evolution" in selected_plots:
            plot_fitness_evolution(
                traj_trace,
                output_dir / "fitness_evolution.png",
                title=f"{strat_instance.name} Fitness Progression",
            )
        if "move_deltas" in selected_plots:
            plot_move_deltas(
                deltas,
                output_dir / "move_deltas.png",
                title=f"{strat_instance.name} Step Deltas",
            )
        if agents > 1:
            plot_multiagent_dashboard(
                traj_trace,
                output_dir / "multiagent_dashboard.png",
                n_agents=agents,
                title=f"{strat_instance.name} Multi-Agent Dashboard",
            )
            plot_multiagent_agent_trajectories(
                traj_trace,
                output_dir / "multiagent_trajectories.png",
                n_agents=agents,
                title=f"{strat_instance.name} Agent Trajectories",
            )
            plot_multiagent_dispersion_heatmap(
                traj_trace,
                output_dir / "multiagent_dispersion.png",
                n_agents=agents,
                title=f"{strat_instance.name} Dispersion Heatmap",
            )
            plot_multiagent_diversity(
                traj_trace,
                output_dir / "diversity_dynamics.png",
                title=f"{strat_instance.name} Population Diversity",
            )

        # Save machine-readable JSON summary
        summary_payload = {
            "strategy": strat_instance.name,
            "problem": {"n": n, "k": k, "topology": topology, "seed": seed},
            "representative_run": {
                "initial_fitness": float(curr_fit[0]),
                "final_fitness": float(curr_fit[-1]),
                "best_fitness": float(best_fit[-1]),
                "steps": run_steps,
                "greedy_ratio": float(greedy_pct),
                "improving_moves": improving_moves,
                "worsening_moves": worsening_moves,
                "behavior_type": behavior_type,
            },
            "statistical_aggregate": {
                "mean_best_fitness": float(np.mean(best_fits_arr)),
                "std_best_fitness": float(np.std(best_fits_arr)),
                "median_best_fitness": float(np.median(best_fits_arr)),
                "max_best_fitness": float(np.max(best_fits_arr)),
                "min_best_fitness": float(np.min(best_fits_arr)),
                "mean_steps": float(np.mean(steps_arr)),
                "restarts": restarts,
            },
        }
        with open(output_dir / "explore_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        console.print(
            f"\n[bold green]✓ Generated {len(selected_plots)} diagnostic exploration plot(s) in:[/bold green] [cyan]{output_dir}[/cyan]"
        )


# ============================================================================
# COMMAND 5: MODELS (MODEL REGISTRY & LOCAL CHECKPOINT MANAGER)
# ============================================================================


@app.command()
def models(
    ckpt_dir: Path = typer.Option(
        Path("checkpoints"),
        "--ckpt-dir",
        "-d",
        help="Checkpoints root directory",
    ),
    experiment: str | None = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Filter models by experiment name",
    ),
    inspect: str | None = typer.Option(
        None,
        "--inspect",
        "-i",
        help="Inspect a specific model's architecture, features, and training history",
    ),
    delete: str | None = typer.Option(
        None,
        "--delete",
        help="Delete a specific model checkpoint from the local registry",
    ),
) -> None:
    """List, inspect, and manage locally saved neural models and checkpoints in the registry."""
    if delete is not None:
        success = delete_saved_model(delete, base_dir=ckpt_dir)
        if success:
            console.print(
                f"[bold green]✓ Successfully deleted model checkpoint:[/bold green] [red]{delete}[/red]"
            )
        else:
            console.print(f"[bold red]✗ Model not found:[/bold red] [yellow]{delete}[/yellow]")
        return

    if inspect is not None:
        try:
            strat, manifest = load_model_checkpoint(inspect, base_dir=ckpt_dir)
            console.print(
                Panel.fit(
                    f"[bold cyan]Model Manifest Inspection: {manifest.model_name}[/bold cyan]\n"
                    f"Experiment: [yellow]{manifest.experiment_name}[/yellow] | Created: [dim]{manifest.created_at}[/dim]\n"
                    f"File: [cyan]{manifest.file_path}[/cyan]",
                    border_style="cyan",
                )
            )

            table = Table(title="Model Specifications", border_style="cyan")
            table.add_column("Property", style="bold white")
            table.add_column("Value", style="cyan")

            table.add_row("Policy Architecture", manifest.policy_type)
            table.add_row("Evaluation Mode", manifest.eval_mode)
            table.add_row("Input Features", manifest.features_summary)
            table.add_row("Trained Problem (N, K)", f"N={manifest.n}, K={manifest.k}")
            table.add_row("Agent Count (M)", str(manifest.n_agents))
            table.add_row(
                "Stopping Conditions",
                json.dumps(manifest.stopping_conditions)
                if manifest.stopping_conditions
                else "None recorded",
            )
            table.add_row("Best Training Fitness", f"{manifest.best_fitness:.4f}")
            num_params = (
                getattr(strat.policy, "num_parameters", len(manifest.weights))
                if hasattr(strat, "policy")
                else len(manifest.weights)
            )
            table.add_row("Total Parameters", f"{num_params:,}")
            table.add_row(
                "Training Generations",
                str(manifest.training_config.get("generations", len(manifest.training_history))),
            )

            console.print(table)
            return
        except Exception as e:
            console.print(f"[bold red]Error loading model '{inspect}':[/bold red] {e}")
            return

    # List all models
    saved_list = list_saved_models(base_dir=ckpt_dir, experiment_name=experiment)
    if not saved_list:
        console.print(
            Panel.fit(
                f"[yellow]No saved models found in '{ckpt_dir.resolve()}'.[/yellow]\n"
                f"Train a model with [bold green]neurols train --exp-name <exp> --model-name <name>[/bold green] to save it.",
                border_style="yellow",
            )
        )
        return

    table = Table(title="Locally Registered Neural Models", border_style="cyan")
    table.add_column("Experiment", style="yellow", no_wrap=True)
    table.add_column("Model Name", style="bold white", no_wrap=True)
    table.add_column("Agents (M)", style="magenta", justify="center", no_wrap=True)
    table.add_column("Policy", style="cyan")
    table.add_column("Mode", style="white")
    table.add_column("Features", style="dim")
    table.add_column("N / K", style="green", justify="center")
    table.add_column("Best Fit", style="bold green", justify="right")
    table.add_column("Date", style="dim")

    for m in saved_list:
        table.add_row(
            m.experiment_name,
            m.model_name,
            str(m.n_agents),
            m.policy_type,
            m.eval_mode,
            m.features_summary[:28] + ("..." if len(m.features_summary) > 28 else ""),
            f"{m.n} / {m.k}",
            f"{m.best_fitness:.4f}",
            m.created_at[:10],
        )

    console.print(table)
    console.print(
        "\n[dim]To explore a saved model: [bold green]neurols explore --model <experiment>:<model_name>[/bold green][/dim]"
    )


# ============================================================================
# COMMAND 6: FEATURES
# ============================================================================


@app.command(name="features")
def features_list() -> None:
    """List all available neural feature channels, group keywords, and evaluation modes."""
    console.print(
        Panel.fit(
            "[bold cyan]NeuroLS Feature Registry & Evaluation Modes[/bold cyan]\n"
            "Easily configure precise inputs for per-action scoring or joint full-neighborhood evaluation.",
            border_style="cyan",
        )
    )

    # 1. Action-level features
    t_action = Table(
        title="1. Action-Specific Features (Par Action - candidate move j)", border_style="cyan"
    )
    t_action.add_column("Feature Key", style="bold yellow")
    t_action.add_column("Dim (M agents)", justify="center", style="magenta")
    t_action.add_column("Formula / Meaning", style="green")
    t_action.add_column("Description", style="white")

    for k, desc in FEATURE_REGISTRY.items():
        if getattr(desc, "category", "") == "action" and k not in ("delta_fitness", "current_bit"):
            dim_str = (
                "M - 1"
                if desc.name == "peer_deltas"
                else ("1 (if M>1)" if desc.is_multi_agent else "1")
            )
            t_action.add_row(
                k, dim_str, getattr(desc, "formula", ""), getattr(desc, "description", "")
            )
    console.print(t_action)
    console.print()

    # 2. Agent-level features
    t_agent = Table(
        title="2. Agent & Group Features (Valable pour les deux - broadcasted or joint)",
        border_style="green",
    )
    t_agent.add_column("Feature Key", style="bold yellow")
    t_agent.add_column("Dim (M agents)", justify="center", style="magenta")
    t_agent.add_column("Formula / Meaning", style="green")
    t_agent.add_column("Description", style="white")

    for k, desc in FEATURE_REGISTRY.items():
        if getattr(desc, "category", "") in ("agent", "context") and k not in ("fitness",):
            dim_str = (
                "M - 1"
                if "peer_" in desc.name or "peer_delta" in desc.name
                else ("1 (if M>1)" if desc.is_multi_agent else "1")
            )
            t_agent.add_row(
                k, dim_str, getattr(desc, "formula", ""), getattr(desc, "description", "")
            )
    console.print(t_agent)
    console.print()

    # 3. Group keywords
    t_groups = Table(
        title="3. Feature Group Keywords (Combine with '@' in CLI)", border_style="magenta"
    )
    t_groups.add_column("Keyword", style="bold yellow")
    t_groups.add_column("Included Features", style="cyan")

    for grp_name, feats in FEATURE_GROUPS.items():
        t_groups.add_row(f"@{grp_name}", ", ".join(feats))
    console.print(t_groups)
    console.print()

    # 4. Evaluation modes
    t_modes = Table(
        title="4. Evaluation Modes: 'per_action' vs 'all_actions'", border_style="yellow"
    )
    t_modes.add_column("Mode", style="bold cyan")
    t_modes.add_column("Description", style="white")
    t_modes.add_column("Compatible Policies", style="green")

    t_modes.add_row(
        "per_action",
        "Evaluates candidate moves individually. Model scores move j based on action features + agent context, then takes argmax.",
        "linear, mlp (per_action), peer_invariant_per_action",
    )
    t_modes.add_row(
        "all_actions",
        "Evaluates the entire neighborhood at once. Receives full neighborhood information in one go.",
        "invariant_nnet, deepsets, dense (dense_neighborhood)",
    )
    console.print(t_modes)


# ============================================================================
# COMMAND 7: INTERPRET (MODEL REASONING, SENSITIVITY & INTERACTIVE DASHBOARD)
# ============================================================================


@app.command(name="interpret")
@app.command(name="inspect-model")
def interpret_cmd(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Path or name of trained model checkpoint (e.g. 'benchmark_trained/Neural_linear_M1')",
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Port for the interactive Streamlit dashboard"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Automatically open in browser"),
    cli: bool = typer.Option(
        False, "--cli", help="Run non-interactive terminal diagnostic report instead of Web GUI"
    ),
    n: int = typer.Option(12, "-n", help="NK dimension N for real evaluation"),
    k: int = typer.Option(2, "-k", help="NK epistasis degree K"),
    agents: int = typer.Option(1, "--agents", "-M", help="Number of agents M"),
    seed: int = typer.Option(42, "--seed", help="Random seed for NK landscape"),
) -> None:
    """Launch the interactive Neural Policy Reason & Interpretability Dashboard or print terminal diagnostics."""
    from neurols.analysis.interpretability import ModelInspector, RealNKSession
    from neurols.training.registry import list_saved_models

    if cli:
        # Terminal-based diagnostic report
        console.print(
            Panel.fit(
                "[bold cyan]🧠 NeuroLS Model Interpretability & Reasoning Diagnostic[/bold cyan]",
                border_style="cyan",
            )
        )
        target_model = model
        if not target_model:
            saved = list_saved_models()
            if saved:
                target_model = str(saved[0].file_path or f"{saved[0].experiment_name}:{saved[0].model_name}")
            else:
                target_model = "CUSTOM:linear:@action_basic:12"

        console.print(f"[bold green]Loading Model:[/bold green] [yellow]{target_model}[/yellow]")
        if target_model.startswith("CUSTOM:"):
            parts = target_model.split(":")
            strat = NeuralLocalSearchStrategy(
                policy_type=parts[1],  # type: ignore
                features=parts[2],
                n_actions=int(parts[3]),
                n_agents=agents,
            )
            inspector = ModelInspector(strat, n_agents=agents)
        else:
            inspector = ModelInspector.from_checkpoint(
                target_model, override_n_agents=agents if agents > 1 else None
            )

        # 1. Architecture & Features Table
        t_feats = Table(
            title=f"1. Model Input Features ({len(inspector.feature_names)} channels)",
            border_style="cyan",
        )
        t_feats.add_column("Channel Name", style="bold yellow")
        t_feats.add_column("Category", style="magenta")
        t_feats.add_column("Valid Range", style="green")
        t_feats.add_column("Formula / Meaning", style="white")

        for info in inspector.feature_infos:
            t_feats.add_row(info.name, info.category, f"[{info.min_val}, {info.max_val}]", info.formula)
        console.print(t_feats)
        console.print()

        # 2. Single Action Evaluation on Default Inputs
        defs = inspector.get_default_feature_vector()
        eval_res = inspector.evaluate_single_action(defs)

        score_style = "bold green" if eval_res.is_positive else "bold red"
        console.print(
            f"[bold]Default Inputs Score (Logit):[/bold] [{score_style}]{eval_res.score:+.4f}[/{score_style}] (Positive: {eval_res.is_positive})"
        )

        if eval_res.linear_breakdown is not None:
            t_lin = Table(title="Linear Feature Breakdown (Contribution = w_i * x_i)", border_style="green")
            t_lin.add_column("Feature", style="bold yellow")
            t_lin.add_column("Value x_i", style="white")
            t_lin.add_column("Contribution", style="cyan")
            for k_name, v in eval_res.linear_breakdown.items():
                t_lin.add_row(k_name, f"{defs.get(k_name, 0.0):.3f}", f"{v:+.4f}")
            if eval_res.linear_bias is not None:
                t_lin.add_row("Bias", "1.0", f"{eval_res.linear_bias:+.4f}")
            console.print(t_lin)
            console.print()

        # Sensitivities
        t_sens = Table(title="Autograd Feature Saliencies |d(Score)/d(Feature)|", border_style="magenta")
        t_sens.add_column("Feature", style="bold yellow")
        t_sens.add_column("Sensitivity", style="cyan")
        for k_name, v in sorted(eval_res.feature_sensitivities.items(), key=lambda x: x[1], reverse=True):
            t_sens.add_row(k_name, f"{v:.4f}")
        console.print(t_sens)
        console.print()

        # 3. 1D Sensitivity Curve summary
        primary_feat = inspector.feature_names[0]
        curve = inspector.compute_sensitivity_curve(primary_feat, defs, num_points=21)
        console.print(f"[bold cyan]Sensitivity Curve on '{primary_feat}':[/bold cyan]")
        console.print(f"  - Score Range: [{curve.min_score:+.4f}, {curve.max_score:+.4f}]")
        console.print(f"  - Local Gradient dS/dX: {curve.local_gradient:+.4f}")
        console.print(f"  - Zero Crossings (Thresholds): {curve.zero_crossings if curve.zero_crossings else 'None'}")
        console.print()

        # 4. Real NK Landscape Step Evaluation
        session = RealNKSession(inspector.strategy, n=n, k=k, n_agents=agents, seed=seed)
        diag = session.evaluate_current_state()
        agent0 = diag["agents"][0]
        console.print(f"[bold cyan]Real NK Landscape (N={n}, K={k}, Seed={seed}):[/bold cyan]")
        console.print(f"  - Current Bitstring: {''.join(str(b) for b in agent0['current_state'])}")
        console.print(f"  - Current Fitness: {agent0['current_fitness']:.4f}")
        console.print(f"  - Model's Chosen Move: Bit {agent0['chosen_action']} (Delta: {agent0['chosen_delta']:+.4f})")
        console.print()
        return

    # Web Dashboard mode
    dash_path = Path(__file__).resolve().parents[1] / "vis" / "dashboard.py"
    console.print(
        Panel.fit(
            f"[bold green]🚀 Launching NeuroLS Interpretability & Reasoning Dashboard...[/bold green]\n\n"
            f"[cyan]Dashboard URL:[/cyan] http://localhost:{port}\n"
            f"[dim]Press Ctrl+C to terminate the web server.[/dim]",
            border_style="green",
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dash_path),
        "--server.port",
        str(port),
    ]
    if not browser:
        cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


main = app

if __name__ == "__main__":
    app()
