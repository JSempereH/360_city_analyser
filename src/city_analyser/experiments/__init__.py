"""city-analyser-experiment: run attribute providers against reference data and compare them (see docs/experiments.md)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment file's runs and score them against the reference data.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("run", "run every run (resumable), then write the report"), ("report", "score the runs already done")):
        command = commands.add_parser(name, help=text)
        command.add_argument("config", type=Path, help="experiment file, e.g. experiments/valencia-pilot.toml")
        command.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "run":
            command.add_argument("--model-root", type=Path, default=Path("models"), help="weights for transformers runs")
            command.add_argument("--only", help="comma-separated run ids to run (default: all)")
    args = parser.parse_args()
    from .report import write_report
    from .runner import run_experiment

    try:
        config = load_config(args.config)
        if args.command == "run":
            run_experiment(args.data_root, config, model_root=args.model_root, only=set(args.only.split(",")) if args.only else None)
        folder = write_report(args.data_root, config)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Report: {folder / 'report.html'} (also report.md and report.json)")


if __name__ == "__main__":
    main()
