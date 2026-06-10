"""
Binance Client — Fetches full OHLCV + taker buy data via Binance REST API.

Since ccxt returns only 6 columns, we use the Binance klines endpoint directly.
This endpoint does not require an API key (public).

Binance klines response format (12 fields):
[
  0:  open time (ms)
  1:  open
  2:  high
  3:  low
  4:  close
  5:  volume
  6:  close time (ms)
  7:  quote asset volume
  8:  number of trades
  9:  taker buy base asset volume
  10: taker buy quote asset volume
  11: ignore
]
"""
from __future__ import annotations

import logging
from typing import Dict

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"

# Number of candles to fetch per timeframe (for warm-up)
FETCH_LIMIT: Dict[str, int] = {
    "15m": 1500,
    "1h":  500,
    "4h":  300,
    "1d":  200,
}

BINANCE_INTERVAL: Dict[str, str] = {
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}


def fetch_ohlcv(timeframe: str) -> pd.DataFrame:
    """
    Fetches full OHLCV + trade data from Binance klines endpoint.

    Returned column names (fully compatible with feature_engineering.py):
        Open time, Open, High, Low, Close, Volume,
        Quote asset volume, Number of trades,
        Taker buy base asset volume, Taker buy quote asset volume

    Parameters
    ----------
    timeframe : "15m" | "1h" | "4h" | "1d"

    Returns
    -------
    pd.DataFrame — sorted, clean data
    """
    if timeframe not in FETCH_LIMIT:
        raise ValueError(f"Invalid timeframe: {timeframe!r}")

    interval = BINANCE_INTERVAL[timeframe]
    limit    = FETCH_LIMIT[timeframe]

    # Binance returns at most 1000 candles, we make 2 requests for 1500
    all_raw = []
    end_time = None

    remaining = limit
    while remaining > 0:
        batch_size = min(remaining, 1000)
        params = {
            "symbol":   SYMBOL,
            "interval": interval,
            "limit":    batch_size,
        }
        if end_time is not None:
            params["endTime"] = end_time

        logger.info("[Binance] Requesting %d candles for %s %s...", batch_size, SYMBOL, timeframe)

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params)
                resp.raise_for_status()
                batch = resp.json()
        except httpx.NetworkError as exc:
            raise ConnectionError(f"Binance network error: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Binance HTTP error {exc.response.status_code}: {exc.response.text}") from exc

        if not batch:
            break

        all_raw = batch + all_raw   # prepend old candles
        end_time = batch[0][0] - 1  # end point of the previous batch
        remaining -= len(batch)

        if len(batch) < batch_size:
            break  # Fewer data returned, history end reached

    if not all_raw:
        raise ValueError("Empty data returned from Binance.")

    # Convert to DataFrame
    df = pd.DataFrame(all_raw, columns=[
        "Open time",
        "Open", "High", "Low", "Close", "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ])

    # Timestamp → UTC datetime
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)

    # Drop unnecessary columns
    df.drop(columns=["Close time", "Ignore"], inplace=True)

    # Type conversion
    numeric_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Sorting and cleaning
    df = df.drop_duplicates(subset=["Open time"])
    df = df.sort_values("Open time").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    logger.info(
        "[Binance] %d candles received — last: %s, price: %.2f",
        len(df),
        df["Open time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        df["Close"].iloc[-1],
    )

    return df
