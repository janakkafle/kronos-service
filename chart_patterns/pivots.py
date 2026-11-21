from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks

from .schemas import PivotPoint


def detect_swings(
    df: pd.DataFrame,
    order: int = 5,
    atr_prominence_mult: float = 0.75,
) -> Tuple[List[PivotPoint], List[PivotPoint]]:
    """
    Detect confirmed swing highs and swing lows using SciPy argrelextrema & find_peaks
    with ATR-scaled prominence filtering and ZigZag alternation enforcement.

    No look-ahead bias: a pivot at index `i` is only accepted if `i <= len(df) - 1 - order`,
    meaning all `order` right-hand confirmation bars have already closed.
    """
    n = len(df)
    if n < (2 * order + 3):
        return [], []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    dates = df["date"].tolist()
    median_atr = float(df["atr"].median()) if "atr" in df.columns else float(np.median(highs - lows))
    min_prominence = max(median_atr * atr_prominence_mult, float(np.median(closes)) * 0.01)

    max_confirmed_idx = n - 1 - order
    ext_high_idx = set(argrelextrema(highs, np.greater_equal, order=order)[0].tolist())
    ext_low_idx = set(argrelextrema(lows, np.less_equal, order=order)[0].tolist())

    pk_high_idx, _ = find_peaks(highs, distance=max(3, order), prominence=min_prominence)
    pk_low_idx, _ = find_peaks(-lows, distance=max(3, order), prominence=min_prominence)

    raw_highs = sorted(i for i in (ext_high_idx | set(pk_high_idx.tolist())) if order <= i <= max_confirmed_idx)
    raw_lows = sorted(i for i in (ext_low_idx | set(pk_low_idx.tolist())) if order <= i <= max_confirmed_idx)

    high_indices = _dedupe_adjacent_pivots(raw_highs, highs, is_high=True, min_gap=max(3, order))
    low_indices = _dedupe_adjacent_pivots(raw_lows, lows, is_high=False, min_gap=max(3, order))

    # Enforce strict ZigZag alternation: between any two highs there must be a low, and vice versa
    combined: List[Tuple[int, str, float]] = (
        [(i, "high", float(highs[i])) for i in high_indices]
        + [(i, "low", float(lows[i])) for i in low_indices]
    )
    combined.sort(key=lambda x: x[0])

    alternating: List[Tuple[int, str, float]] = []
    for idx, ptype, val in combined:
        if not alternating:
            alternating.append((idx, ptype, val))
            continue
        prev_idx, prev_type, prev_val = alternating[-1]
        if ptype == prev_type:
            if (ptype == "high" and val >= prev_val) or (ptype == "low" and val <= prev_val):
                alternating[-1] = (idx, ptype, val)
        else:
            if abs(val - prev_val) >= median_atr * 0.8 or (idx - prev_idx) >= order * 2:
                alternating.append((idx, ptype, val))

    swing_highs = [
        PivotPoint(time=dates[i], value=round(v, 2), type="high", label="H", index=int(i))
        for i, t, v in alternating
        if t == "high"
    ]
    swing_lows = [
        PivotPoint(time=dates[i], value=round(v, 2), type="low", label="L", index=int(i))
        for i, t, v in alternating
        if t == "low"
    ]
    return swing_highs, swing_lows


def _dedupe_adjacent_pivots(
    indices: List[int],
    values: np.ndarray,
    is_high: bool,
    min_gap: int,
) -> List[int]:
    if not indices:
        return []

    kept: List[int] = [indices[0]]
    for idx in indices[1:]:
        prev = kept[-1]
        if (idx - prev) < min_gap:
            if (is_high and values[idx] >= values[prev]) or (not is_high and values[idx] <= values[prev]):
                kept[-1] = idx
        else:
            kept.append(idx)
    return kept
