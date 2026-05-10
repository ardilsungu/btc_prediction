import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Proje koku: bu dosyanin bulundugu klasorun bir ust dizini (src/ -> BTC_PREDICTION/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIMEFRAME_CONFIGS = {
    '15m': {
        'input_path': PROJECT_ROOT / 'data/processing/btc_15m_processed.csv',
        'rolling_short':   48,    # 12 saat
        'rolling_med':     96,    # 1 gun
        'rolling_long':    288,   # 3 gun
        'long_ema_period': 672,   # 1 hafta
        'volume_window':   96,    # 1 gun
        'taker_buy_ma':    48,    # 12 saat
        'obv_change_lag':  48,    # 12 saat
        'include_hour':    True,
    },
    '1h': {
        'input_path': PROJECT_ROOT / 'data/processing/btc_1h_processed.csv',
        'rolling_short':   12,    # 12 saat
        'rolling_med':     24,    # 1 gun
        'rolling_long':    72,    # 3 gun
        'long_ema_period': 200,   # ~8.3 gun
        'volume_window':   24,
        'taker_buy_ma':    12,
        'obv_change_lag':  12,
        'include_hour':    True,
    },
    '4h': {
        'input_path': PROJECT_ROOT / 'data/processing/btc_4h_processed.csv',
        'rolling_short':   12,    # 2 gun
        'rolling_med':     24,    # 4 gun
        'rolling_long':    72,    # 12 gun
        'long_ema_period': 200,   # ~33 gun
        'volume_window':   24,
        'taker_buy_ma':    12,
        'obv_change_lag':  12,
        'include_hour':    True,
    },
    '1d': {
        'input_path': PROJECT_ROOT / 'data/processing/btc_1d_processed.csv',
        'rolling_short':   12,    # 12 gun
        'rolling_med':     24,    # ~3 hafta
        'rolling_long':    72,    # ~2.5 ay
        'long_ema_period': 100,   # ~3.3 ay
        'volume_window':   24,
        'taker_buy_ma':    12,
        'obv_change_lag':  12,
        'include_hour':    False,
    },
}


# ──────────────────────────────────────────────────────────────
# 1) VERİ YÜKLEME
# ──────────────────────────────────────────────────────────────
def load_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Open time'] = pd.to_datetime(df['Open time'], errors='coerce', utc=True)
    df = df.dropna(subset=['Open time']).reset_index(drop=True)
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# 2) RETURN TABANLI ÖZELLİKLER
# ──────────────────────────────────────────────────────────────
def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    for n in [1, 3, 6, 12, 24]:
        df[f'log_return_{n}'] = np.log(df['Close'] / df['Close'].shift(n))
    return df


# ──────────────────────────────────────────────────────────────
# 3) VOLATİLİTE ÖZELLİKLERİ
# ──────────────────────────────────────────────────────────────
def add_volatility_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s, m, l = cfg['rolling_short'], cfg['rolling_med'], cfg['rolling_long']

    df['rolling_std_short'] = df['log_return_1'].rolling(s).std()
    df['rolling_std_med']   = df['log_return_1'].rolling(m).std()
    df['rolling_std_long']  = df['log_return_1'].rolling(l).std()
    df['realized_vol_med']  = np.sqrt((df['log_return_1'] ** 2).rolling(m).sum())

    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low']  - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean() / df['Close']

    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ──────────────────────────────────────────────────────────────
# 4) TEKNİK İNDİKATÖRLER
# ──────────────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI_14'] = _rsi(df['Close'], 14)

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_line']      = (ema_12 - ema_26) / df['Close']
    df['MACD_signal']    = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_position'] = (df['Close'] - bb_mid) / (2 * bb_std)

    return df


# ──────────────────────────────────────────────────────────────
# 5) HAREKETLİ ORTALAMA SAPMALARI
# ──────────────────────────────────────────────────────────────
def add_ma_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sma_20   = df['Close'].rolling(20).mean()
    sma_50   = df['Close'].rolling(50).mean()
    ema_9    = df['Close'].ewm(span=9,  adjust=False).mean()
    ema_21   = df['Close'].ewm(span=21, adjust=False).mean()
    ema_long = df['Close'].ewm(span=cfg['long_ema_period'], adjust=False).mean()

    df['close_to_sma20']    = df['Close'] / sma_20 - 1
    df['close_to_sma50']    = df['Close'] / sma_50 - 1
    df['close_to_ema_long'] = df['Close'] / ema_long - 1
    df['ema9_to_ema21']     = ema_9 / ema_21 - 1
    return df


# ──────────────────────────────────────────────────────────────
# 6) HACİM VE ORDER FLOW ÖZELLİKLERİ
# ──────────────────────────────────────────────────────────────
def add_volume_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vw = cfg['volume_window']

    vol_mean = df['Volume'].rolling(vw).mean()
    vol_std  = df['Volume'].rolling(vw).std()
    df['volume_zscore'] = (df['Volume'] - vol_mean) / vol_std
    df['volume_ratio']  = df['Volume'] / vol_mean

    df['taker_buy_ratio']    = df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)
    df['taker_buy_ratio_ma'] = df['taker_buy_ratio'].rolling(cfg['taker_buy_ma']).mean()

    nt_mean = df['Number of trades'].rolling(vw).mean()
    nt_std  = df['Number of trades'].rolling(vw).std()
    df['num_trades_zscore'] = (df['Number of trades'] - nt_mean) / nt_std

    direction = np.sign(df['Close'].diff()).fillna(0)
    obv = (direction * df['Volume']).cumsum()
    df['OBV_change'] = obv.pct_change(periods=cfg['obv_change_lag']).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


# ──────────────────────────────────────────────────────────────
# 
# ──────────────────────────────────────────────────────────────
def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    candle_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['body_size']        = (df['Close'] - df['Open']).abs() / df['Open']
    df['upper_wick_ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / candle_range
    df['lower_wick_ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / candle_range
    return df


# ──────────────────────────────────────────────────────────────
#
# ──────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 3, 4, 5]:
        df[f'log_return_lag_{lag}'] = df['log_return_1'].shift(lag)
    return df


# ──────────────────────────────────────────────────────────────
# 9) ZAMAN ÖZELLİKLERİ
# ──────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame, include_hour: bool) -> pd.DataFrame:
    dow = df['Open time'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * dow / 7)

    if include_hour:
        hour = df['Open time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return df


# ──────────────────────────────────────────────────────────────
# ANA PIPELINE
# ──────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_volatility_features(df, cfg)
    df = add_technical_indicators(df)
    df = add_ma_features(df, cfg)
    df = add_volume_features(df, cfg)
    df = add_candle_features(df)
    df = add_lag_features(df)
    df = add_time_features(df, cfg['include_hour'])
    return df


def get_feature_columns(cfg: dict) -> list:
    cols = [
        'log_return_1', 'log_return_3', 'log_return_6', 'log_return_12', 'log_return_24',
        'rolling_std_short', 'rolling_std_med', 'rolling_std_long',
        'realized_vol_med', 'ATR_14',
        'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_histogram', 'BB_position',
        'close_to_sma20', 'close_to_sma50', 'close_to_ema_long', 'ema9_to_ema21',
        'volume_zscore', 'volume_ratio', 'taker_buy_ratio', 'taker_buy_ratio_ma',
        'num_trades_zscore', 'OBV_change',
        'body_size', 'upper_wick_ratio', 'lower_wick_ratio',
        'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3',
        'log_return_lag_4', 'log_return_lag_5',
        'dayofweek_sin', 'dayofweek_cos',
    ]
    if cfg['include_hour']:
        cols += ['hour_sin', 'hour_cos']
    return cols


# ──────────────────────────────────────────────────────────────
# TEK TIMEFRAME İŞLEM
# ──────────────────────────────────────────────────────────────
def process_timeframe(tf_name: str, cfg: dict, output_dir: Path, force: bool = False) -> None:

    print(f'\n{"="*70}')
    print(f'  {tf_name.upper()} TIMEFRAME')
    print(f'{"="*70}')

    output_path = output_dir / f'{tf_name}_features.csv'

    # Dosya varsa ve --force verilmediyse atla
    if output_path.exists() and not force:
        print(f'[{tf_name}] Zaten mevcut, atlanıyor → {output_path}')
        print(f'  (Yeniden hesaplamak için --force kullanın)')
        return

    if output_path.exists() and force:
        print(f'[{tf_name}] --force aktif, mevcut dosyanın üzerine yazılıyor.')

    # Yukleme
    df = load_and_clean(cfg['input_path'])
    print(f'Yuklenen satir    : {len(df):,}')
    print(f'Tarih araligi     : {df["Open time"].iloc[0].date()} → {df["Open time"].iloc[-1].date()}')
    print(f'Pencere ayarlari  : short={cfg["rolling_short"]}, med={cfg["rolling_med"]}, '
          f'long={cfg["rolling_long"]}, EMA-long={cfg["long_ema_period"]}')

    # Feature engineering
    df = engineer_features(df, cfg)
    feature_cols = get_feature_columns(cfg)
    print(f'Uretilen feature  : {len(feature_cols)}')

    # Kolon secimi
    keep_cols = ['Open time'] + feature_cols + ['Target']
    df_clean = df[keep_cols].copy()

    # NaN / inf temizligi
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(df_clean)
    df_clean = df_clean.dropna().reset_index(drop=True)
    print(f'Warm-up dusen     : {n_before - len(df_clean):,}')
    print(f'Final satir       : {len(df_clean):,}')

    # Kaydet
    df_clean.to_csv(output_path, index=False)
    print(f'Kaydedildi        → {output_path}')


# ──────────────────────────────────────────────────────────────
# DOĞRUDAN ÇALIŞTIRMA  →  python feature_engineering.py [--force]
# ──────────────────────────────────────────────────────────────
def main(force: bool = False) -> None:
    output_dir = PROJECT_ROOT / 'data/feature_engineering'
    output_dir.mkdir(parents=True, exist_ok=True)

    for tf_name, cfg in TIMEFRAME_CONFIGS.items():
        process_timeframe(tf_name, cfg, output_dir, force=force)

    print(f'\n{"="*70}')
    print('  FEATURE ENGINEERING TAMAMLANDI')
    print(f'{"="*70}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTC Feature Engineering')
    parser.add_argument(
        '--force', action='store_true',
        help='Mevcut feature dosyalarının üzerine yaz'
    )
    args = parser.parse_args()
    main(force=args.force)