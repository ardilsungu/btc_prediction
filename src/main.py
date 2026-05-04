"""
BTC Prediction — Ana Giriş Noktası
====================================
Çalıştırma:
    python main.py            # Mevcut dosyalar varsa atla
    python main.py --force    # Her şeyi sıfırdan hesapla
"""
import argparse
from pathlib import Path

from data_processing import DataProcessor, TIMEFRAME_CONFIG
from feature_engineering import process_timeframe, TIMEFRAME_CONFIGS

# Proje koku: src/ klasörünün bir üstü
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ham veri dosyaları
RAW_FILES = {
    '15m': PROJECT_ROOT / 'data/raw/btc_15m_data_2018_to_2025.csv',
    '1h':  PROJECT_ROOT / 'data/raw/btc_1h_data_2018_to_2025.csv',
    '4h':  PROJECT_ROOT / 'data/raw/btc_4h_data_2018_to_2025.csv',
    '1d':  PROJECT_ROOT / 'data/raw/btc_1d_data_2018_to_2025.csv',
}

PROCESSING_DIR     = PROJECT_ROOT / 'data/processing'
FEATURE_ENG_DIR    = PROJECT_ROOT / 'data/feature_engineering'


# ──────────────────────────────────────────────────────────────
# ADIM 1 — VERİ İŞLEME  (data_processing.py)
# ──────────────────────────────────────────────────────────────
def run_processing(force: bool = False) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 1 / 2 — VERİ İŞLEME')
    print(f'{"#"*70}')

    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    for tf, raw_path in RAW_FILES.items():
        print(f'\n{"="*70}')
        print(f'  {tf.upper()} TIMEFRAME')
        print(f'{"="*70}')

        out_path = PROCESSING_DIR / f'btc_{tf}_processed.csv'

        # force olmadan mevcutsa atla — DataProcessor.run_pipeline() da kontrol
        # eder ama burada erken çıkarak gereksiz obje oluşturmuyoruz
        if out_path.exists() and not force:
            print(f'[{tf}] Zaten mevcut, atlanıyor → {out_path}')
            print(f'  (Yeniden işlemek için --force kullanın)')
            continue

        if out_path.exists() and force:
            out_path.unlink()          # eski dosyayı sil; run_pipeline tekrar yazsın
            print(f'[{tf}] --force aktif, mevcut dosya silindi.')

        processor = DataProcessor(
            file_path=str(raw_path),
            timeframe=tf,
        )
        processor.run_pipeline(save=True, output_dir=str(PROCESSING_DIR))


# ──────────────────────────────────────────────────────────────
# ADIM 2 — FEATURE ENGINEERING  (feature_engineering.py)
# ──────────────────────────────────────────────────────────────
def run_feature_engineering(force: bool = False) -> None:
    print(f'\n{"#"*70}')
    print('  ADIM 2 / 2 — FEATURE ENGINEERING')
    print(f'{"#"*70}')

    FEATURE_ENG_DIR.mkdir(parents=True, exist_ok=True)

    for tf_name, cfg in TIMEFRAME_CONFIGS.items():
        process_timeframe(tf_name, cfg, FEATURE_ENG_DIR, force=force)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description='BTC Prediction Pipeline — Processing + Feature Engineering'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Mevcut dosyaları sil ve tüm adımları sıfırdan çalıştır'
    )
    args = parser.parse_args()

    run_processing(force=args.force)
    run_feature_engineering(force=args.force)

    print(f'\n{"#"*70}')
    print('  TÜM ADIMLAR TAMAMLANDI')
    print(f'{"#"*70}\n')


if __name__ == '__main__':
    main()