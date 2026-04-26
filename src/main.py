from data_processing import DataProcessor

# ──────────────────────────────────────────────────────────────
# İşlenecek zaman dilimleri ve karşılık gelen ham veri dosyaları
# ──────────────────────────────────────────────────────────────
JOBS = [
    {'timeframe': '15m', 'file_path': 'data/raw/btc_15m_data_2018_to_2025.csv'},
    {'timeframe': '1h',  'file_path': 'data/raw/btc_1h_data_2018_to_2025.csv'},
    {'timeframe': '4h',  'file_path': 'data/raw/btc_4h_data_2018_to_2025.csv'},
    {'timeframe': '1d',  'file_path': 'data/raw/btc_1d_data_2018_to_2025.csv'},
]

# ──────────────────────────────────────────────────────────────
# Pipeline'ı çalıştır
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = {}

    for job in JOBS:
        print(f"\n{'='*55}")
        print(f"  Zaman dilimi: {job['timeframe'].upper()}")
        print(f"{'='*55}")

        processor = DataProcessor(
            file_path=job['file_path'],
            timeframe=job['timeframe'],
        )

        df = processor.run_pipeline(save=True, output_dir='data/processing')