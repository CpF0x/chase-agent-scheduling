"""Command-line interface for the CHASE experiment package."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chase-exp")
    parser.add_argument(
        "command",
        nargs="?",
        default="main",
        choices=["main", "high-pressure", "timecost", "cross-dataset", "sensitivity", "fig1"],
        help="Experiment to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny smoke workflow instead of the full experiment.",
    )
    args = parser.parse_args(argv)

    if args.command == "main":
        from project.experiments import main_experiment

        if args.quick:
            main_experiment.run_quick_smoke()
            return
        data = main_experiment.run_full_experiment()
        main_experiment.plot_results(data)
        main_experiment.export_statistics_tables(data)
        return

    if args.command == "high-pressure":
        from project.experiments import main_experiment

        main_experiment.run_high_pressure_experiment(
            num_trials=5 if args.quick else main_experiment.HIGH_PRESSURE_TRIALS
        )
        return

    if args.command == "timecost":
        from project.experiments.timecost import main as run
    elif args.command == "cross-dataset":
        from project.experiments.cross_dataset import main as run
    elif args.command == "sensitivity":
        from project.experiments.sensitivity import main as run
    else:
        from project.experiments.fig1 import main as run

    run()


if __name__ == "__main__":
    main()
