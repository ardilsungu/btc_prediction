"""
BTC Prediction - Tarih Bazli Train / Validation / Test Split
=============================================================
Zaman serisi verisinde random shuffle YAPILMAZ.
Kronolojik sira korunarak %70 / %15 / %15 oraninda bolunur.

Kullanim:
    python time_split.py                 # Tum timeframe'ler
    python time_split.py --timeframe 1h  # Sadece 1h
    python time_split.py --force         # Mevcut dosyalarin uzerine yaz
"""
import argparse
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = PROJECT_ROOT / 'data' / 'feature_engineering'
SPLIT_DIR   = PROJECT_ROOT / 'data' / 'splits'


TIMEFRAMES = ['15m', '1h', '4h', '1d']


# --------------------------------------------------------------
# 
# --------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    purge_rows: int    = 5,
) -> tuple:
    """
    Kronolojik sirayi koruyarak veriyi 3 parcaya boler.

    Parameters
    ----------
    df          : Feature engineering ciktisi (Open time kolonu dahil).
    train_ratio : Train setinin orani (varsayilan %70).
    val_ratio   : Validation setinin orani (varsayilan %15).
                  Test orani = 1 - train_ratio - val_ratio.
    purge_rows  : Train-Val ve Val-Test sinirlarinda silinecek satir sayisi.
                  Lag feature'larin (lag 1-5) yarattigi bilgi sizintisini onler.

    Returns
    -------
    (train_df, val_df, test_df) : Uc ayri DataFrame.
    """
    
    df = df.sort_values('Open time').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

  
    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    
    if purge_rows > 0:
      
        train_df = train_df.iloc[:-purge_rows].copy()
       
        val_df = val_df.iloc[:-purge_rows].copy()

    
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


# --------------------------------------------------------------
# RAPOR
# --------------------------------------------------------------
def _print_split_report(name: str, df: pd.DataFrame) -> None:
    """Bir split setinin ozetini basar."""
    n = len(df)
    target_dist = df['Target'].value_counts().sort_index()

    down    = target_dist.get(-1, 0)
    neutral = target_dist.get(0, 0)
    up      = target_dist.get(1, 0)

    date_start = df['Open time'].iloc[0]
    date_end   = df['Open time'].iloc[-1]

    print(f"  {name:12s} | {n:>8,} satir | {date_start} -> {date_end}")
    print(f"  {' ':12s} | DOWN: {down:>6,} ({down/n*100:5.1f}%) | "
          f"NEUTRAL: {neutral:>6,} ({neutral/n*100:5.1f}%) | "
          f"UP: {up:>6,} ({up/n*100:5.1f}%)")


def print_full_report(
    timeframe: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """3 setin tam raporunu basar."""
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n{'-'*90}")
    print(f"  [{timeframe.upper()}] SPLIT RAPORU  (toplam: {total:,} satir, purge sonrasi)")
    print(f"{'-'*90}")
    _print_split_report('TRAIN',      train_df)
    _print_split_report('VALIDATION', val_df)
    _print_split_report('TEST',       test_df)
    print(f"{'-'*90}")

    
    train_end = pd.to_datetime(train_df['Open time'].iloc[-1])
    val_start = pd.to_datetime(val_df['Open time'].iloc[0])
    val_end   = pd.to_datetime(val_df['Open time'].iloc[-1])
    test_start = pd.to_datetime(test_df['Open time'].iloc[0])

    gap_train_val = val_start - train_end
    gap_val_test  = test_start - val_end

    print(f"  Train->Val gap  : {gap_train_val}")
    print(f"  Val->Test gap   : {gap_val_test}")

    if train_end < val_start and val_end < test_start:
        print("  [OK] Data leakage yok - tarih sirasi dogru")
    else:
        print("  [HATA] UYARI: Tarih cakismasi tespit edildi!")
    print()


# --------------------------------------------------------------
# TEK TIMEFRAME ISLEME
# --------------------------------------------------------------
def split_timeframe(timeframe: str, force: bool = False) -> None:
    """Bir timeframe icin split yapar ve CSV olarak kaydeder."""
    input_path = FEATURE_DIR / f'{timeframe}_features.csv'

    if not input_path.exists():
        print(f"[{timeframe}] Feature dosyasi bulunamadi: {input_path}")
        print(f"  Once feature engineering calistirin.")
        return

   
    out_train = SPLIT_DIR / f'{timeframe}_train.csv'
    out_val   = SPLIT_DIR / f'{timeframe}_val.csv'
    out_test  = SPLIT_DIR / f'{timeframe}_test.csv'

   
    if all(p.exists() for p in [out_train, out_val, out_test]) and not force:
        print(f"[{timeframe}] Split dosyalari zaten mevcut, atlaniyor.")
        print(f"  (Yeniden bolmek icin --force kullanin)")
        return

   
    print(f"\n[{timeframe}] Feature dosyasi okunuyor: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[{timeframe}] Toplam satir: {len(df):,}")

   
    train_df, val_df, test_df = time_based_split(df)

    
    print_full_report(timeframe, train_df, val_df, test_df)

    
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"[{timeframe}] Kaydedildi:")
    print(f"  Train      -> {out_train}")
    print(f"  Validation -> {out_val}")
    print(f"  Test       -> {out_test}")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='BTC Prediction - Tarih Bazli Split'
    )
    parser.add_argument(
        '--timeframe', '-t',
        choices=TIMEFRAMES,
        default=None,
        help='Sadece belirtilen timeframe icin split yap (orn: 1h)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Mevcut split dosyalarinin uzerine yaz'
    )
    args = parser.parse_args()

    print(f'\n{"="*70}')
    print('  TARIH BAZLI SPLIT')
    print(f'{"="*70}')

    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES

    for tf in timeframes:
        split_timeframe(tf, force=args.force)

    print(f'\n{"="*70}')
    print('  SPLIT TAMAMLANDI')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
