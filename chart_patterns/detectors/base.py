from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple
import numpy as np
import pandas as pd

from ..schemas import DetectedPattern, PivotPoint


class BasePatternDetector(ABC):
    """Abstract base class for all modular classical chart pattern detectors."""

    @abstractmethod
    def detect(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
    ) -> List[DetectedPattern]:
        """Detect patterns of this family from preprocessed OHLCV + confirmed pivots."""
        raise NotImplementedError


def check_channel_containment(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    m_up: float,
    b_up: float,
    m_lo: float,
    b_lo: float,
    atr: float,
    max_violation_ratio: float = 0.08,
) -> bool:
    """
    Verify that candle closes inside the formation window [start_idx, end_idx]
    respect the upper resistance and lower support trendlines within an ATR tolerance.
    """
    if end_idx <= start_idx + 2:
        return False
    x = np.arange(start_idx, end_idx + 1, dtype=float)
    closes = df["close"].iloc[start_idx : end_idx + 1].to_numpy(dtype=float)
    tol = max(atr * 0.55, float(np.median(closes)) * 0.005)
    up_line = m_up * x + b_up
    lo_line = m_lo * x + b_lo
    violations = int(np.sum((closes > up_line + tol) | (closes < lo_line - tol)))
    return (violations / len(x)) <= max_violation_ratio


def check_volume_contraction(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    """True if average volume in the second half of formation is lower than the first half."""
    if end_idx - start_idx < 8:
        return False
    mid = (start_idx + end_idx) // 2
    v1 = float(df["volume"].iloc[start_idx:mid].mean())
    v2 = float(df["volume"].iloc[mid : end_idx + 1].mean())
    return v1 > 0 and v2 < v1 * 0.95


def evaluate_breakout_and_status(
    df: pd.DataFrame,
    end_idx: int,
    upper_level: Optional[float],
    lower_level: Optional[float],
    expected_bias: Literal["bullish", "bearish", "neutral"],
    atr: float,
) -> Tuple[Literal["forming", "confirmed", "invalidated"], Literal["bullish", "bearish", "none"], Optional[float], bool, float]:
    """
    Evaluate formation status ('forming', 'confirmed', 'invalidated'), breakout direction,
    active breakout level, and volume confirmation strictly using bars at/after `end_idx`
    up to the latest bar in `df` (zero look-ahead bias).
    """
    last_close = float(df["close"].iloc[-1])
    last_vol_ratio = float(df["vol_ratio"].iloc[-1])
    buf = max(atr * 0.25, last_close * 0.002)

    post_df = df.iloc[max(0, end_idx):]
    if post_df.empty:
        return "forming", "none", upper_level if expected_bias == "bullish" else lower_level, False, round(last_vol_ratio, 2)

    if expected_bias == "bullish":
        if upper_level is not None:
            broke_up = post_df[post_df["close"] > upper_level + buf]
            if not broke_up.empty:
                b_vol = float(broke_up["vol_ratio"].iloc[0])
                if lower_level is not None and last_close < lower_level - buf:
                    return "invalidated", "bearish", round(upper_level, 2), False, round(b_vol, 2)
                return "confirmed", "bullish", round(upper_level, 2), b_vol >= 1.30, round(b_vol, 2)
        if lower_level is not None and last_close < lower_level - buf * 1.5:
            return "invalidated", "bearish", round(lower_level, 2), False, round(last_vol_ratio, 2)
        return "forming", "none", round(upper_level, 2) if upper_level is not None else None, False, round(last_vol_ratio, 2)

    if expected_bias == "bearish":
        if lower_level is not None:
            broke_down = post_df[post_df["close"] < lower_level - buf]
            if not broke_down.empty:
                b_vol = float(broke_down["vol_ratio"].iloc[0])
                if upper_level is not None and last_close > upper_level + buf:
                    return "invalidated", "bullish", round(lower_level, 2), False, round(b_vol, 2)
                return "confirmed", "bearish", round(lower_level, 2), b_vol >= 1.20, round(b_vol, 2)
        if upper_level is not None and last_close > upper_level + buf * 1.5:
            return "invalidated", "bullish", round(upper_level, 2), False, round(last_vol_ratio, 2)
        return "forming", "none", round(lower_level, 2) if lower_level is not None else None, False, round(last_vol_ratio, 2)

    if upper_level is not None:
        broke_up = post_df[post_df["close"] > upper_level + buf]
        if not broke_up.empty:
            b_vol = float(broke_up["vol_ratio"].iloc[0])
            return "confirmed", "bullish", round(upper_level, 2), b_vol >= 1.30, round(b_vol, 2)
    if lower_level is not None:
        broke_down = post_df[post_df["close"] < lower_level - buf]
        if not broke_down.empty:
            b_vol = float(broke_down["vol_ratio"].iloc[0])
            return "confirmed", "bearish", round(lower_level, 2), b_vol >= 1.20, round(b_vol, 2)

    mid = upper_level if upper_level is not None else lower_level
    return "forming", "none", round(mid, 2) if mid is not None else None, False, round(last_vol_ratio, 2)


def compute_confidence(
    geometry_score: float,
    touches: int,
    recency_idx: int,
    total_bars: int,
    status: str,
    volume_confirmed: bool,
    volume_contraction: bool = False,
) -> float:
    """
    Score pattern quality from 0 to 100 based on geometric fit, pivot touches,
    recency in the lookback window, confirmation status, and volume behavior.
    """
    base = max(0.0, min(52.0, geometry_score * 52.0))
    touch_pts = min(14.0, max(0, touches - 2) * 4.5)
    recency_pts = 12.0 * (recency_idx / max(1, total_bars - 1))
    status_pts = 10.0 if status == "confirmed" else (4.0 if status == "forming" else -20.0)
    vol_pts = (8.0 if volume_confirmed else 0.0) + (4.0 if volume_contraction else 0.0)
    return round(float(np.clip(base + touch_pts + recency_pts + status_pts + vol_pts, 10.0, 98.0)), 2)
