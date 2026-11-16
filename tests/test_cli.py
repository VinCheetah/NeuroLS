"""Tests for Typer CLI commands."""

from typer.testing import CliRunner

from neurols.cli.app import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "NeuroLS" in result.output


def test_cli_inspect_nk() -> None:
    result = runner.invoke(app, ["inspect-nk", "-n", "10", "-k", "2", "--samples", "100"])
    assert result.exit_code == 0
    assert "Landscape Properties" in result.output
    assert "Mean Random Fitness" in result.output


def test_cli_run() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--strategy",
            "HillClimber",
            "-n",
            "12",
            "-k",
            "2",
            "--steps",
            "15",
            "--restarts",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Execution Summary" in result.output
    assert "Mean Best Fitness" in result.output


def test_cli_benchmark(tmp_path) -> None:
    out_dir = str(tmp_path / "bench")
    result = runner.invoke(
        app,
        ["benchmark", "-n", "10", "-k", "2", "--steps", "10", "--restarts", "2", "--out", out_dir],
    )
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output
    assert "HillClimber" in result.output


def test_cli_train(tmp_path) -> None:
    ckpt_dir = str(tmp_path / "ckpt")
    result = runner.invoke(
        app,
        [
            "train",
            "-n",
            "10",
            "-k",
            "2",
            "-g",
            "2",
            "--popsize",
            "4",
            "--eval-restarts",
            "2",
            "--eval-steps",
            "10",
            "--ckpt-dir",
            ckpt_dir,
            "--no-plot",
        ],
    )
    assert result.exit_code == 0
    assert "CMA-ES Generation Progress" in result.output
    assert "Training Complete" in result.output


def test_cli_inspect_nn(tmp_path) -> None:
    out_dir = str(tmp_path / "inspect")
    result = runner.invoke(
        app,
        ["inspect-nn", "-n", "10", "-k", "2", "--out", out_dir, "--no-plot"],
    )
    assert result.exit_code == 0
    assert "Policy Input Feature Saliency" in result.output
    assert "Sample Decision Analysis" in result.output
