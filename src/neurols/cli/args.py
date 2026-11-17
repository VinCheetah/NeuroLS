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
        "--N",
        "-N",
        help="The number of loci in the NK landscape.",
        min=1,
    ),
]

KOption = Annotated[
    int,
    typer.Option(
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
        "-p",
        help="Number of agents per population (1 for a single agent).",
        min=1,
    ),
]

NbAgentsOption = Annotated[
    int,
    typer.Option(
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
        "-s",
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

PlotOption = Annotated[
    bool,
    typer.Option(
        "--plot/--no-plot",
        help="Whether to generate plots for the experiment.",
    ),
]

TrainingOption = Annotated[
    bool,
    typer.Option(
        "--train/--no-train",
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
        "-o",
        help="The output directory.",
    ),
    path_callback,
]

NbIterationsOption = Annotated[
    int,
    typer.Option(
        "--nb-iterations",
        "-t",
        help="The number of iterations per restart.",
        min=1,
    ),
]

NbBudgetOption = Annotated[
    int,
    typer.Option(
        "--nb-budget",
        "-b",
        help="The budget for the experiment.",
        min=1,
    ),
]

NbTimeOption = Annotated[
    int,
    typer.Option(
        "--nb-time",
        "--time",
        help="The time limit for the experiment.",
        min=1,
    ),
]
