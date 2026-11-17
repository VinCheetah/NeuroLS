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


def test_cli_run_with_budget_and_time() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--strategy",
            "HillClimber",
            "-n",
            "10",
            "-k",
            "1",
            "--budget",
            "150",
            "--time",
            "5.0",
            "--restarts",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Execution Summary" in result.output
    assert "Termination Reason" in result.output
    assert "Total Evaluations" in result.output


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


def test_cli_multi_agent_study(tmp_path) -> None:
    out_dir = str(tmp_path / "ma_study")
    result = runner.invoke(
        app,
        [
            "multi-agent-study",
            "-n",
            "8",
            "-k",
            "1",
            "--agents",
            "1,2",
            "--steps",
            "4",
            "--budget",
            "200",
            "-g",
            "1",
            "-r",
            "2",
            "--out",
            out_dir,
        ],
    )
    assert result.exit_code == 0
    assert "Multi-Agent Study Leaderboard" in result.output
    assert "study_results.json" in str(list(tmp_path.glob("**/*")))


def test_cli_benchmark_per_action(tmp_path) -> None:
    out_dir = str(tmp_path / "bench_pa")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "-n",
            "8",
            "-k",
            "1",
            "--preset",
            "per-action",
            "--steps",
            "5",
            "--restarts",
            "2",
            "-g",
            "1",
            "--out",
            out_dir,
        ],
    )
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output
    assert (tmp_path / "bench_pa" / "benchmark_dashboard.png").exists()
