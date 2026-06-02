import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.lg_regression.core import full_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression for 15m timeframe")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=1800, help="Optuna timeout in seconds")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning")
    args = parser.parse_args()

    full_training_pipeline(
        timeframe="15m",
        n_trials=args.trials,
        timeout=args.timeout,
        skip_tuning=args.skip_tuning,
    )
