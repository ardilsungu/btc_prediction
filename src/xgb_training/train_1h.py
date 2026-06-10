import argparse
import sys
from pathlib import Path

# src/ dizinini path'e ekle (xgb_training/ alt klasorunden calistigimiz icin)
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xgb_training.core import full_training_pipeline


TIMEFRAME = '1h'


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f'BTC XGBoost Egitimi - {TIMEFRAME.upper()} Timeframe'
    )
    parser.add_argument(
        '--trials', type=int, default=50,
        help='Optuna trial sayisi (varsayilan: 50)'
    )
    parser.add_argument(
        '--timeout', type=int, default=1800,
        help='Tuning timeout, saniye (varsayilan: 1800 = 30 dk)'
    )
    parser.add_argument(
        '--skip-tuning', action='store_true',
        help='Optuna tuning atla, sadece baseline parametreleriyle egit'
    )
    args = parser.parse_args()

    full_training_pipeline(
        timeframe=TIMEFRAME,
        n_trials=args.trials,
        timeout=args.timeout,
        skip_tuning=args.skip_tuning,
    )


if __name__ == '__main__':
    main()
