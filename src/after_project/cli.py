"""Command-line interface for the CHASE experiment package."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chase-exp")
    parser.add_argument(
        "command",
        nargs="?",
        default="main",
        choices=["main", "timecost", "cross-dataset", "sensitivity", "fig1"],
        help="Experiment to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny smoke workflow instead of the full experiment.",
    )
    args = parser.parse_args(argv)

    if args.command == "main":
        from after_project.experiments import main_experiment

        if args.quick:
            main_experiment.run_quick_smoke()
            return
        data = main_experiment.run_full_experiment()
        main_experiment.plot_results(data)
        main_experiment.export_statistics_tables(data)
        return

    if args.command == "timecost":
        from after_project.experiments.timecost import main as run
    elif args.command == "cross-dataset":
        from after_project.experiments.cross_dataset import main as run
    elif args.command == "sensitivity":
        from after_project.experiments.sensitivity import main as run
    else:
        from after_project.experiments.fig1 import main as run

    run()


if __name__ == "__main__":
    main()
