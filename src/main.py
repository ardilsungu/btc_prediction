"""
BTC Prediction - Ana Giris Noktasi
====================================
Calistirma:
    python main.py                        # Mevcut dosyalar varsa atla
    python main.py --force                # Her seyi sifirdan hesapla
    python main.py --timeframe 1h         # Sadece 1h timeframe
    python main.py --skip-train           # Model egitimini atla
    python main.py --trials 30            # Optuna trial sayisi
"""
import argparse
from pathlib import Path

from data_processing import DataProcessor, TIMEFRAME_CONFIG
from feature_engineering import process_timeframe, TIMEFRAME_CONFIGS
from time_split import split_timeframe, TIMEFRAMES

# Proje koku: src/ klasorunun bir ustu
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ham veri dosyalari
RAW_FILES = {
    '15m': PROJECT_ROOT / 'data/raw/btc_15m_data_2018_to_2025.csv',
    '1h':  PROJECT_ROOT / 'data/raw/btc_1h_data_2018_to_2025.csv',
    '4h':  PROJECT_ROOT / 'data/raw/btc_4h_data_2018_to_2025.csv',
    '1d':  PROJECT_ROOT / 'data/raw/btc_1d_data_2018_to_2025.csv',
}

PROCESSING_DIR     = PROJECT_ROOT / 'data/processing'
FEATURE_ENG_DIR    = PROJECT_ROOT / 'data/feature_engineering'


# --------------------------------------------------------------
# ADIM 1 - VERI ISLEME  (data_processing.py)
# --------------------------------------------------------------
def run_processing(force: bool = False, timeframes: list = None) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 1 / 4 - VERI ISLEME')
    print(f'{"#"*70}')

    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    targets = timeframes or list(RAW_FILES.keys())

    for tf in targets:
        raw_path = RAW_FILES.get(tf)
        if raw_path is None:
            print(f'[{tf}] Ham veri dosyasi tanimli degil, atlaniyor.')
            continue

        print(f'\n{"="*70}')
        print(f'  {tf.upper()} TIMEFRAME')
        print(f'{"="*70}')

        out_path = PROCESSING_DIR / f'btc_{tf}_processed.csv'

        if out_path.exists() and not force:
            print(f'[{tf}] Zaten mevcut, atlaniyor -> {out_path}')
            print(f'  (Yeniden islemek icin --force kullanin)')
            continue

        if out_path.exists() and force:
            out_path.unlink()
            print(f'[{tf}] --force aktif, mevcut dosya silindi.')

        processor = DataProcessor(
            file_path=str(raw_path),
            timeframe=tf,
        )
        processor.run_pipeline(save=True, output_dir=str(PROCESSING_DIR))


# --------------------------------------------------------------
# ADIM 2 - FEATURE ENGINEERING  (feature_engineering.py)
# --------------------------------------------------------------
def run_feature_engineering(force: bool = False, timeframes: list = None) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 2 / 4 - FEATURE ENGINEERING')
    print(f'{"#"*70}')

    FEATURE_ENG_DIR.mkdir(parents=True, exist_ok=True)

    targets = timeframes or list(TIMEFRAME_CONFIGS.keys())

    for tf_name in targets:
        cfg = TIMEFRAME_CONFIGS.get(tf_name)
        if cfg is None:
            print(f'[{tf_name}] Timeframe config bulunamadi, atlaniyor.')
            continue
        process_timeframe(tf_name, cfg, FEATURE_ENG_DIR, force=force)


# --------------------------------------------------------------
# ADIM 3 - TARIH BAZLI SPLIT  (time_split.py)
# --------------------------------------------------------------
def run_split(force: bool = False, timeframes: list = None) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 3 / 4 - TARIH BAZLI SPLIT')
    print(f'{"#"*70}')

    targets = timeframes or TIMEFRAMES

    for tf in targets:
        split_timeframe(tf, force=force)


# --------------------------------------------------------------
# ADIM 4 - XGBOOST EGITIM  (xgb_training/core.py)
# --------------------------------------------------------------
def run_training(
    timeframes: list = None,
    n_trials: int = 50,
    skip_tuning: bool = False,
) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 4 / 4 - XGBOOST EGITIM')
    print(f'{"#"*70}')

    from xgb_training.core import full_training_pipeline

    targets = timeframes or TIMEFRAMES

    for tf in targets:
        full_training_pipeline(
            timeframe=tf,
            n_trials=n_trials,
            skip_tuning=skip_tuning,
        )

def run_lr_training(
    timeframes: list = None,
    n_trials: int = 50,
    skip_tuning: bool = False,
) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 4b / 4 - LOGISTIC REGRESSION EGITIM')
    print(f'{"#"*70}')

    from lg_regression.core import full_training_pipeline

    targets = timeframes or TIMEFRAMES

    for tf in targets:
        full_training_pipeline(
            timeframe=tf,
            n_trials=n_trials,
            skip_tuning=skip_tuning,
        )

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='BTC Prediction Pipeline - Processing + Features + Split + Training'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Mevcut dosyalari sil ve tum adimlari sifirdan calistir'
    )
    parser.add_argument(
        '--timeframe', '-t',
        choices=TIMEFRAMES,
        default=None,
        help='Sadece belirtilen timeframe icin calistir (orn: 1h)'
    )
    parser.add_argument(
        '--skip-train', action='store_true',
        help='Model egitimini atla (sadece veri hazirligi yap)'
    )
    parser.add_argument(
        '--skip-tuning', action='store_true',
        help='Optuna tuning atla, baseline parametreleriyle egit'
    )
    parser.add_argument(
        '--trials', type=int, default=50,
        help='Optuna trial sayisi (varsayilan: 50)'
    )
    args = parser.parse_args()

    timeframes = [args.timeframe] if args.timeframe else None

    # Adim 1-3: Veri hazirligi
    run_processing(force=args.force, timeframes=timeframes)
    run_feature_engineering(force=args.force, timeframes=timeframes)
    run_split(force=args.force, timeframes=timeframes)

    # Adim 4: Model egitimi
    if not args.skip_train:
        run_training(
            timeframes=timeframes,
            n_trials=args.trials,
            skip_tuning=args.skip_tuning,
        )
        run_lr_training(
            timeframes=timeframes,
            n_trials=args.trials,
            skip_tuning=args.skip_tuning,
        )
    else:
        print(f'\n{"#"*70}')
        print('  ADIM 4 / 4 - EGITIM ATLANDI (--skip-train)')
        print(f'{"#"*70}')

    print(f'\n{"#"*70}')
    print('  TUM ADIMLAR TAMAMLANDI')
    print(f'{"#"*70}\n')


if __name__ == '__main__':
    main()