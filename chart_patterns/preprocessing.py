from typing import Optional
import numpy as np
import pandas as pd

from .schemas import PatternCandle


def prepare_ohlcv_dataframe(
    candles: list[PatternCandle],
    lookback_bars: int = 180,
    atr_period: int = 14,
    as_of_date: Optional[str] = None,
    offset_days: int = 0,
) -> pd.DataFrame:
    """
    Normalize OHLCV candles into a clean, chronologically sorted DataFrame with
    ATR(14) and Volume SMA(20) computed strictly without look-ahead bias.
    """
    if not candles:
        return pd.DataFrame()

    rows = [
        {
            "date": str(c.timestamp)[:10],
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume or 0.0),
        }
        for c in candles
        if c.close > 0
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # Enforce no look-ahead bias via as_of_date or offset_days BEFORE any calculation
    if as_of_date:
        cutoff = str(as_of_date)[:10]
        df = df[df["date"] <= cutoff].reset_index(drop=True)

    if offset_days > 0 and len(df) > offset_days:
        df = df.iloc[: len(df) - offset_days].reset_index(drop=True)

    if len(df) > lookback_bars:
        df = df.iloc[-lookback_bars:].reset_index(drop=True)

    if df.empty:
        return df

    # Sanitize OHLC consistency
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    # True Range & Wilder's ATR
    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ),
    )
    df["tr"] = tr
    df["atr"] = tr.ewm(alpha=1.0 / max(1, atr_period), min_periods=1, adjust=False).mean()
    # Fallback floor for very low volatility / illiquid stocks
    min_atr = df["close"] * 0.005
    df["atr"] = np.maximum(df["atr"], min_atr)

    # Volume baseline (20-day rolling SMA, shifted by 1 on breakout bar evaluation)
    df["vol_sma20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["vol_ratio"] = np.where(df["vol_sma20"] > 0, df["volume"] / df["vol_sma20"], 1.0)

    df["idx"] = np.arange(len(df), dtype=int)
    return df
