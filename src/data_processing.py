import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Her zaman dilimine göre threshold varsayılanları
# ──────────────────────────────────────────────────────────────
TIMEFRAME_CONFIG = {
    '15m': {'threshold': 0.15},
    '1h':  {'threshold': 0.30},
    '4h':  {'threshold': 0.60},
    '1d':  {'threshold': 1.50},
}


class DataProcessor:
    def __init__(self, file_path: str, timeframe: str, threshold: float = None):
        """
        file_path : İşlenecek CSV dosyasının yolu.
        timeframe : '15m' | '1h' | '4h' | '1d'
        threshold : (Opsiyonel) Otomatik değeri ezmek istersen gir.
        """
        if timeframe not in TIMEFRAME_CONFIG:
            raise ValueError(
                f"Geçersiz timeframe: '{timeframe}'. "
                f"Geçerli değerler: {list(TIMEFRAME_CONFIG.keys())}"
            )

        self.file_path = file_path
        self.timeframe = timeframe
        self.threshold = threshold if threshold is not None else TIMEFRAME_CONFIG[timeframe]['threshold']
        self.df        = None

    # ──────────────────────────────────────────────────────────
    # 1. VERİ YÜKLEME & TEMİZLEME
    # ──────────────────────────────────────────────────────────
    def load_and_clean_data(self) -> pd.DataFrame:
        """Ham veriyi yükler, tiplerini düzeltir ve gereksiz kolonları atar."""
        self.df = pd.read_csv(self.file_path)

        self.df['Open time'] = pd.to_datetime(self.df['Open time'])
        self.df = self.df[~self.df['Open time'].duplicated(keep='first')]
        self.df.set_index('Open time', inplace=True)
        self.df.sort_index(inplace=True)

        cols_to_drop = ['Close time', 'Ignore']
        self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns],
            inplace=True
        )

        numeric_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume'
        ]
        existing = [c for c in numeric_cols if c in self.df.columns]
        self.df[existing] = self.df[existing].apply(pd.to_numeric, errors='coerce')

        print(f"[{self.timeframe}] Veri yüklendi — {len(self.df):,} satır "
              f"({self.df.index[0].date()} → {self.df.index[-1].date()})")
        return self.df

    # ──────────────────────────────────────────────────────────
    # 2. HEDEF DEĞİŞKEN
    # ──────────────────────────────────────────────────────────
    def create_target(self) -> pd.DataFrame:
        """
        3 sınıflı hedef değişken oluşturur:
          1  : UP      (fiyat >= +threshold % artacak)
         -1  : DOWN    (fiyat <= -threshold % düşecek)
          0  : NEUTRAL (threshold içinde kalacak)
        """
        if self.df is None:
            raise ValueError("Önce load_and_clean_data() çalıştırılmalı!")

        next_close        = self.df['Close'].shift(-1)
        future_return_pct = ((next_close - self.df['Close']) / self.df['Close']) * 100

        conditions = [
            future_return_pct >=  self.threshold,
            future_return_pct <= -self.threshold,
        ]
        self.df['Target'] = np.select(conditions, [1, -1], default=0)

        dist  = self.df['Target'].value_counts().sort_index()
        total = len(self.df)
        print(f"[{self.timeframe}] Target oluşturuldu (eşik: ±%{self.threshold})")
        print(f"  DOWN(-1): {dist.get(-1, 0):>7,}  ({dist.get(-1, 0) / total * 100:.1f}%)")
        print(f"  NEUTRAL : {dist.get( 0, 0):>7,}  ({dist.get( 0, 0) / total * 100:.1f}%)")
        print(f"  UP  (+1): {dist.get( 1, 0):>7,}  ({dist.get( 1, 0) / total * 100:.1f}%)")

        return self.df

    # ──────────────────────────────────────────────────────────
    # 3. TAM PIPELINE
    # ──────────────────────────────────────────────────────────
    def run_pipeline(self, save: bool = True, output_dir: str = 'data/processing') -> pd.DataFrame:
        """
        Dosya zaten işlenmişse diskten okur, yoksa işleyip kaydeder.
        """
        out_path = Path(output_dir) / f"btc_{self.timeframe}_processed.csv"

        # Zaten varsa işleme, direkt oku
        if out_path.exists():
            print(f"[{self.timeframe}] Zaten mevcut, okunuyor → {out_path}")
            self.df = pd.read_csv(out_path, index_col='Open time', parse_dates=True)
            return self.df

        # Yoksa işle ve kaydet
        self.load_and_clean_data()
        self.create_target()

        before = len(self.df)
        self.df.dropna(inplace=True)
        after  = len(self.df)
        print(f"[{self.timeframe}] NaN temizlendi: {before:,} → {after:,} satır "
              f"(düşen: {before - after:,})")

        if save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.df.to_csv(out_path)
            print(f"[{self.timeframe}] Kaydedildi → {out_path}")

        self._print_summary()
        return self.df

    def _print_summary(self):
        """İşlenmiş verinin özet tablosunu basar."""
        total   = len(self.df)
        down    = (self.df['Target'] == -1).sum()
        neutral = (self.df['Target'] ==  0).sum()
        up      = (self.df['Target'] ==  1).sum()

        print(f"\n  {'Satır':<10} {'DOWN':>8} {'NEUTRAL':>10} {'UP':>8}")
        print(f"  {'-'*40}")
        print(f"  {total:<10,} {down:>8,} {neutral:>10,} {up:>8,}\n")