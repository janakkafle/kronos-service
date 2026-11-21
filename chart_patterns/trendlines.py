from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .schemas import ChartPoint, DetectedTrendline, PivotPoint


def detect_support_resistance_trendlines(
    df: pd.DataFrame,
    swing_highs: List[PivotPoint],
    swing_lows: List[PivotPoint],
    max_lines_per_side: int = 3,
) -> Tuple[List[DetectedTrendline], List[DetectedTrendline]]:
    """
    Detect support and resistance trendlines.

    Attempts `pytrendline` first (on compatible DataFrames) and augments/falls back to
    a fast scikit-learn pivot-pair + ATR-inlier regression engine so results are always
    available, deterministic, and TradingView-ready.
    """
    support_lines = _detect_via_pytrendline(df, "support")
    resistance_lines = _detect_via_pytrendline(df, "resistance")

    if not support_lines:
        support_lines = _detect_sklearn_trendlines(df, swing_lows, "support", max_lines_per_side)
    if not resistance_lines:
        resistance_lines = _detect_sklearn_trendlines(df, swing_highs, "resistance", max_lines_per_side)

    return support_lines[:max_lines_per_side], resistance_lines[:max_lines_per_side]


def _detect_via_pytrendline(df: pd.DataFrame, side: str) -> List[DetectedTrendline]:
    """
    Run pytrendline if installed and compatible with current pandas version.
    pytrendline 1.0.1 internally uses `DataFrame.append`, which was removed in pandas 2.0,
    so this safely catches compatibility errors and delegates to the scikit-learn detector.
    """
    try:
        import pytrendline  # type: ignore

        pt_df = df[["date", "open", "high", "low", "close"]].copy()
        pt_df.columns = ["Date", "Open", "High", "Low", "Close"]
        pt_df["Date"] = pd.to_datetime(pt_df["Date"])
        candle_data = pytrendline.CandlestickData(df=pt_df, time_interval="1d")
        t_type = (
            pytrendline.TrendlineTypes.SUPPORT
            if side == "support"
            else pytrendline.TrendlineTypes.RESISTANCE
        )
        res = pytrendline.detect(
            candlestick_data=candle_data,
            trend_type=t_type,
            min_points_required=3,
            ignore_breakouts=False,
        )
        key = "support_trendlines" if side == "support" else "resistance_trendlines"
        t_df = res.get(key)
        if t_df is None or t_df.empty:
            return []

        dates = df["date"].tolist()
        out: List[DetectedTrendline] = []
        for _, row in t_df.head(3).iterrows():
            s_idx = int(row["starts_at_index"])
            e_idx = int(row["ends_at_index"])
            m = float(row["m"])
            b = float(row["b"])
            out.append(
                DetectedTrendline(
                    type=side,  # type: ignore[arg-type]
                    start=ChartPoint(time=dates[s_idx], value=round(m * s_idx + b, 2)),
                    end=ChartPoint(time=dates[e_idx], value=round(m * e_idx + b, 2)),
                    slope=round(m, 6),
                    intercept=round(b, 4),
                    touches=int(row["num_points"]),
                    score=round(float(row["score"]), 2),
                    broken=bool(row.get("is_breakout", False)),
                )
            )
        return out
    except Exception:
        return []


def _detect_sklearn_trendlines(
    df: pd.DataFrame,
    pivots: List[PivotPoint],
    side: str,
    max_lines: int = 3,
) -> List[DetectedTrendline]:
    """
    Fit candidate trendlines across pivot pairs using scikit-learn LinearRegression
    over inlier pivots within an ATR-scaled tolerance band.
    """
    if len(pivots) < 2 or df.empty:
        return []

    n = len(df)
    dates = df["date"].tolist()
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    median_atr = float(df["atr"].median())
    tol = max(median_atr * 0.55, float(np.median(closes)) * 0.008)

    piv_indices = np.array([p.index for p in pivots if p.index is not None], dtype=float)
    piv_values = np.array([p.value for p in pivots if p.index is not None], dtype=float)
    if len(piv_indices) < 2:
        return []

    candidates: List[DetectedTrendline] = []
    num_piv = len(piv_indices)

    for i in range(num_piv):
        for j in range(i + 1, num_piv):
            x1, y1 = piv_indices[i], piv_values[i]
            x2, y2 = piv_indices[j], piv_values[j]
            if x2 - x1 < 4:
                continue

            slope_init = (y2 - y1) / (x2 - x1)
            intercept_init = y1 - slope_init * x1

            # Collect all pivot inliers along this line
            predicted_pivs = slope_init * piv_indices + intercept_init
            inlier_mask = np.abs(piv_values - predicted_pivs) <= tol
            touches = int(np.sum(inlier_mask))
            if touches < 2:
                continue

            # Refit with scikit-learn LinearRegression on all inliers
            X_in = piv_indices[inlier_mask].reshape(-1, 1)
            y_in = piv_values[inlier_mask]
            reg = LinearRegression().fit(X_in, y_in)
            m = float(reg.coef_[0])
            b = float(reg.intercept_)

            s_idx = int(piv_indices[inlier_mask][0])
            e_idx = n - 1  # Extend to latest bar for TradingView rendering

            # Validate line doesn't slice heavily through candle bodies inside [s_idx, e_idx]
            x_span = np.arange(s_idx, n, dtype=float)
            line_vals = m * x_span + b
            if side == "resistance":
                violations = int(np.sum(closes[s_idx:] > line_vals + tol))
            else:
                violations = int(np.sum(closes[s_idx:] < line_vals - tol))

            if violations > max(2, int(0.1 * len(x_span))):
                continue

            broken = bool(
                (side == "resistance" and closes[-1] > (m * e_idx + b + tol * 0.5))
                or (side == "support" and closes[-1] < (m * e_idx + b - tol * 0.5))
            )

            rmse = float(np.sqrt(np.mean((y_in - (m * X_in.ravel() + b)) ** 2)))
            recency_bonus = 10.0 * (float(piv_indices[inlier_mask][-1]) / max(1, n - 1))
            score = (touches * 22.0) + recency_bonus - (rmse / max(1e-6, median_atr)) * 10.0 - (violations * 5.0)

            candidates.append(
                DetectedTrendline(
                    type=side,  # type: ignore[arg-type]
                    start=ChartPoint(time=dates[s_idx], value=round(m * s_idx + b, 2)),
                    end=ChartPoint(time=dates[e_idx], value=round(m * e_idx + b, 2)),
                    slope=round(m, 6),
                    intercept=round(b, 4),
                    touches=touches,
                    score=round(max(0.0, score), 2),
                    broken=broken,
                )
            )

    # Deduplicate similar slopes & intercepts, keeping highest score
    candidates.sort(key=lambda t: t.score, reverse=True)
    selected: List[DetectedTrendline] = []
    for cand in candidates:
        is_dup = any(
            abs(cand.slope - s.slope) <= (median_atr * 0.02)
            and abs(cand.end.value - s.end.value) <= tol * 1.2
            for s in selected
        )
        if not is_dup:
            selected.append(cand)
        if len(selected) >= max_lines:
            break

    return selected


def fit_line_segment(
    indices: np.ndarray,
    values: np.ndarray,
    start_idx: int,
    end_idx: int,
    dates: List[str],
) -> Tuple[float, float, List[ChartPoint]]:
    """
    Fit an OLS trendline via scikit-learn and return (slope, intercept, [start_pt, end_pt]).
    """
    reg = LinearRegression().fit(indices.reshape(-1, 1), values)
    m = float(reg.coef_[0])
    b = float(reg.intercept_)
    s_i = max(0, min(start_idx, len(dates) - 1))
    e_i = max(0, min(end_idx, len(dates) - 1))
    pts = [
        ChartPoint(time=dates[s_i], value=round(m * s_i + b, 2)),
        ChartPoint(time=dates[e_i], value=round(m * e_i + b, 2)),
    ]
    return m, b, pts
