"""Tests for Typer CLI commands."""

from typer.testing import CliRunner

from neurols.cli.app import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "NeuroLS" in result.output
    # Check that legacy commands are not listed
    assert "ablation" not in result.output
    assert "inspect-nk" not in result.output


def test_cli_features() -> None:
    result = runner.invoke(app, ["features"])
    assert result.exit_code == 0
    assert "Action-Specific Features" in result.output
    assert "peer_deltas" in result.output
    assert "@action_basic" in result.output


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
        [
            "benchmark",
            "-n",
            "10",
            "-k",
            "2",
            "--steps",
            "10",
            "--restarts",
            "2",
            "--no-train-nn",
            "--out",
            out_dir,
        ],
    )
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output
    # Check strict strategy naming format: {strategy}_M{agents}
    assert "HillClimber_M1" in result.output
    assert "FirstImprovement_M1" in result.output
    # Check default plots created
    assert (tmp_path / "bench" / "benchmark_dashboard.png").exists()
    assert (tmp_path / "bench" / "convergence_comparison.png").exists()
    assert (tmp_path / "bench" / "boxplot_comparison.png").exists()
    assert (tmp_path / "bench" / "step_trajectories.png").exists()
    # Check sample exploration created for models
    assert (
        tmp_path / "bench" / "exploration" / "HillClimber_M1" / "exploration_dashboard.png"
    ).exists()
    assert (
        tmp_path / "bench" / "exploration" / "HillClimber_M1" / "search_trajectory_2d.png"
    ).exists()


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
            "--agents",
            "2",
            "--ckpt-dir",
            ckpt_dir,
            "--no-plot",
        ],
    )
    assert result.exit_code == 0
    assert "CMA-ES Generation Progress" in result.output
    assert "Training Complete" in result.output


def test_cli_models_agents_and_stopping_conditions(tmp_path) -> None:
    ckpt_dir = str(tmp_path / "models_ckpt")
    # Train model with 4 agents
    train_res = runner.invoke(
        app,
        [
            "train",
            "-n",
            "10",
            "-k",
            "1",
            "-g",
            "2",
            "--popsize",
            "4",
            "--eval-restarts",
            "2",
            "--eval-steps",
            "8",
            "--agents",
            "4",
            "--exp-name",
            "exp_agents",
            "--model-name",
            "net_m4",
            "--ckpt-dir",
            ckpt_dir,
            "--no-plot",
        ],
    )
    assert train_res.exit_code == 0

    # Test models listing displays Agents (M)
    list_res = runner.invoke(app, ["models", "--ckpt-dir", ckpt_dir])
    assert list_res.exit_code == 0
    assert "Agents (M)" in list_res.output
    assert "4" in list_res.output

    # Test inspect displays Agent Count and Stopping Conditions
    inspect_res = runner.invoke(
        app,
        ["models", "--inspect", "exp_agents:net_m4_M4", "--ckpt-dir", ckpt_dir],
    )
    assert inspect_res.exit_code == 0
    assert "Agent Count (M)" in inspect_res.output
    assert "Stopping Conditions" in inspect_res.output


def test_cli_benchmark_multi_agent_sweep_and_naming(tmp_path) -> None:
    out_dir = str(tmp_path / "bench_sweep")
    result = runner.invoke(
        app,
        [
            "benchmark",
            "-n",
            "8",
            "-k",
            "1",
            "--agents",
            "1,2",
            "--steps",
            "5",
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
    # Check that model names have _M1 and _M2
    assert "_M1" in result.output
    assert "_M2" in result.output


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


def test_cli_explore_classical(tmp_path) -> None:
    """Test deep-dive explore command with HillClimber and diagnostic plots."""
    out_dir = str(tmp_path / "explore_hc")
    res = runner.invoke(
        app,
        [
            "explore",
            "-s",
            "HillClimber",
            "-n",
            "12",
            "-k",
            "2",
            "--steps",
            "10",
            "--restarts",
            "3",
            "--out",
            out_dir,
        ],
    )
    assert res.exit_code == 0
    assert "In-Depth Strategy Exploration" in res.output
    assert "Representative Run Profile" in res.output
    assert "Statistical Aggregate Performance" in res.output
    assert (tmp_path / "explore_hc" / "exploration_dashboard.png").exists()
    assert (tmp_path / "explore_hc" / "search_trajectory_2d.png").exists()
    assert (tmp_path / "explore_hc" / "bitstring_evolution.png").exists()
    assert (tmp_path / "explore_hc" / "action_ranks.png").exists()
    assert (tmp_path / "explore_hc" / "explore_summary.json").exists()


def test_cli_explore_plot_selection_and_mds(tmp_path) -> None:
    """Test explore command with MDS 2D reduction and selective plot keywords."""
    out_dir = str(tmp_path / "explore_mds")
    res = runner.invoke(
        app,
        [
            "explore",
            "-s",
            "TabuSearch",
            "-n",
            "10",
            "-k",
            "1",
            "--steps",
            "8",
            "--restarts",
            "2",
            "--reduction",
            "mds",
            "--plots",
            "trajectory_2d,move_deltas",
            "--out",
            out_dir,
        ],
    )
    assert res.exit_code == 0
    assert (tmp_path / "explore_mds" / "search_trajectory_2d.png").exists()
    assert (tmp_path / "explore_mds" / "move_deltas.png").exists()
    assert not (tmp_path / "explore_mds" / "action_ranks.png").exists()


def test_cli_train_registry_and_explore_saved_model(tmp_path) -> None:
    """Test full workflow: train model, verify registry, inspect it, and explore it."""
    ckpt_dir = str(tmp_path / "ckpts")
    exp_out = str(tmp_path / "saved_explore")

    # 1. Train and register
    train_res = runner.invoke(
        app,
        [
            "train",
            "-n",
            "10",
            "-k",
            "1",
            "-g",
            "2",
            "--popsize",
            "4",
            "--eval-restarts",
            "2",
            "--eval-steps",
            "8",
            "--exp-name",
            "test_exp",
            "--model-name",
            "my_net",
            "--ckpt-dir",
            ckpt_dir,
            "--no-plot",
        ],
    )
    assert train_res.exit_code == 0
    assert "Registered model to:" in train_res.output
    assert (tmp_path / "ckpts" / "test_exp" / "my_net_M1.json").exists()

    # 2. List in models
    models_res = runner.invoke(
        app,
        ["models", "--ckpt-dir", ckpt_dir],
    )
    assert models_res.exit_code == 0
    assert "my_net_M1" in models_res.output

    # 3. Inspect model
    inspect_res = runner.invoke(
        app,
        ["models", "--inspect", "test_exp:my_net_M1", "--ckpt-dir", ckpt_dir],
    )
    assert inspect_res.exit_code == 0
    assert "Model Specifications" in inspect_res.output
    assert "my_net_M1" in inspect_res.output
    assert "test_exp" in inspect_res.output

    # 4. Explore the saved registered model
    explore_res = runner.invoke(
        app,
        [
            "explore",
            "-M",
            "test_exp:my_net_M1",
            "-n",
            "10",
            "-k",
            "1",
            "--steps",
            "8",
            "--restarts",
            "2",
            "--ckpt-dir",
            ckpt_dir,
            "--out",
            exp_out,
        ],
    )
    assert explore_res.exit_code == 0
    assert "Target Strategy: my_net_M1" in explore_res.output
    assert (tmp_path / "saved_explore" / "exploration_dashboard.png").exists()
