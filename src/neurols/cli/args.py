from pathlib import Path
from typing import Annotated

import typer

from neurols.types import Device

StrategyOption = Annotated[
    str,
    typer.Option(
        "--strategy",
        "-s",
        help="The strategy to use for the experiment.",
    ),
]

StrategiesOption = Annotated[
    list[str],
    typer.Option(
        "--strategy",
        "-s",
        help="The strategies to use for the experiment.",
    ),
]

ModelsOption = Annotated[
    list[str] | None,
    typer.Option(
        "--model",
        "-M",
        help="Specify one or more model definitions (e.g. 'name=M1,mode=per_action,policy=mlp,features=@action_basic'). Repeat to compare multiple models.",
    ),
]

FeaturesOption = Annotated[
    str | None,
    typer.Option(
        "--features",
        "-f",
        help="Features to include: comma-separated names or group keywords (e.g. '@action_basic,peer_deltas', '@all', '@scores').",
    ),
]

EvalModeOption = Annotated[
    str,
    typer.Option(
        "--mode",
        help="Evaluation mode: 'per_action' (evaluate moves individually) or 'all_actions' (evaluate entire neighborhood jointly).",
    ),
]

VisualisationsOption = Annotated[
    list[str],
    typer.Option(
        "--visualisation",
        "-v",
        help="Visualisation names (repeat the option to display several charts).",
    ),
]

LiveOption = Annotated[
    bool,
    typer.Option("--live", help="Update live-compatible visualisations during execution."),
]

NOption = Annotated[
    int,
    typer.Option(
        "--n",
        "-n",
        "--N",
        "-N",
        help="The number of loci in the NK landscape.",
        min=1,
    ),
]

KOption = Annotated[
    int,
    typer.Option(
        "--k",
        "-k",
        "--K",
        "-K",
        help="The number of interactions in the NK landscape.",
        min=0,
    ),
]


def device_callback(value: str | Device) -> Device:
    return value if isinstance(value, Device) else Device(value)


DeviceOption = Annotated[
    Device,
    typer.Option(
        "--device",
        "-d",
        help="The device to use for the experiment (cpu or gpu).",
        callback=device_callback,
    ),
]

NbInstancesOption = Annotated[
    int,
    typer.Option(
        "--nb-instances",
        "-i",
        help="The number of instances to run.",
        min=1,
        max=10,
    ),
]

NbRestartsOption = Annotated[
    int,
    typer.Option(
        "--restarts",
        "--nb-restarts",
        "-r",
        help="The number of restarts per instance.",
        min=1,
    ),
]

PopulationSizeOption = Annotated[
    int,
    typer.Option(
        "--population-size",
        "--popsize",
        "-p",
        help="Number of agents per population (1 for a single agent).",
        min=1,
    ),
]

NbAgentsOption = Annotated[
    int,
    typer.Option(
        "--agents",
        "--nb-agents",
        "-m",
        help="Number of agents per group (1 for a single agent).",
        min=1,
    ),
]

SharedInfoOption = Annotated[
    bool,
    typer.Option(
        "--shared-info/--no-shared-info",
        help="Whether agents share information within a group.",
    ),
]

SeedOption = Annotated[
    int,
    typer.Option(
        "--seed",
        help="The random seed for the experiment.",
    ),
]

NbGenerationsOption = Annotated[
    int,
    typer.Option(
        "--generations",
        "-g",
        help="The number of generations for training.",
        min=1,
    ),
]

Sigma0Option = Annotated[
    float,
    typer.Option(
        "--sigma0",
        help="Initial mutation step-size (standard deviation) for CMA-ES.",
        min=0.01,
        max=5.0,
    ),
]

EvalRestartsOption = Annotated[
    int,
    typer.Option(
        "--eval-restarts",
        help="Number of rollouts/restarts per candidate during training evaluation.",
        min=1,
    ),
]

EvalStepsOption = Annotated[
    int,
    typer.Option(
        "--eval-steps",
        help="Number of search steps per rollout during training evaluation.",
        min=1,
    ),
]

PlotOption = Annotated[
    bool,
    typer.Option(
        "--plot/--no-plot",
        help="Whether to generate plots for the experiment.",
    ),
]

PlotsOption = Annotated[
    str | None,
    typer.Option(
        "--plots",
        help="Plots to generate: comma-separated names or keywords ('@default', '@all', '@benchmark', '@explore', '@training', etc.).",
    ),
]

TrainingOption = Annotated[
    bool,
    typer.Option(
        "--train/--no-train",
        "--train-nn/--no-train-nn",
        help="Whether to train the neural strategy before benchmarking.",
    ),
]


# Callback to transform into Path from pathlib
def path_callback(value: str) -> Path:
    return Path(value)


OutputDirOption = Annotated[
    Path,
    typer.Option(
        "--output-dir",
        "--out",
        "-o",
        help="The output directory.",
    ),
    path_callback,
]

NbIterationsOption = Annotated[
    int | None,
    typer.Option(
        "--nb-iterations",
        "--steps",
        "-t",
        help="The maximum number of iterations / steps.",
        min=1,
    ),
]

NbBudgetOption = Annotated[
    int | None,
    typer.Option(
        "--nb-budget",
        "--budget",
        "-b",
        help="Maximum function / evaluation budget.",
        min=1,
    ),
]

NbTimeOption = Annotated[
    float | None,
    typer.Option(
        "--nb-time",
        "--time",
        help="Maximum wall-clock time limit in seconds.",
        min=0.01,
    ),
]
