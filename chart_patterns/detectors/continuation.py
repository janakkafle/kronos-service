from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..schemas import ChartPoint, DetectedPattern, PatternBoundaries, PivotPoint
from .base import BasePatternDetector, compute_confidence, evaluate_breakout_and_status


class ContinuationPatternDetector(BasePatternDetector):
    """
    Detects Bullish/Bearish Flags and Bullish/Bearish Pennants with strong
    impulse-pole validation (>= 3.5 ATR) and volume contraction during consolidation.
    """

    def detect(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
    ) -> List[DetectedPattern]:
        n = len(df)
        if n < 30:
            return []

        dates = df["date"].tolist()
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)
        atr = float(df["atr"].iloc[-1])

        for cons_len in (10, 14, 18):
            if n < cons_len + 14:
                continue
            cons_start = n - cons_len
            pole_start = max(0, cons_start - 14)

            pole_move = closes[cons_start] - closes[pole_start]
            pole_abs = abs(pole_move)
            pole_pct = pole_abs / max(1e-6, closes[pole_start])
            if pole_abs < atr * 3.5 or pole_pct < 0.045:
                continue

            is_bull_pole = pole_move > 0
            cons_x = np.arange(cons_start, n, dtype=float).reshape(-1, 1)
            cons_highs = highs[cons_start:]
            cons_lows = lows[cons_start:]

            reg_up = LinearRegression().fit(cons_x, cons_highs)
            reg_lo = LinearRegression().fit(cons_x, cons_lows)
            m_up, b_up = float(reg_up.coef_[0]), float(reg_up.intercept_)
            m_lo, b_lo = float(reg_lo.coef_[0]), float(reg_lo.intercept_)

            retrace = (float(np.max(cons_highs)) - float(np.min(cons_lows))) / max(1e-6, pole_abs)
            if retrace > 0.48:
                continue

            pole_vol = float(np.mean(volumes[pole_start:cons_start]))
            cons_vol = float(np.mean(volumes[cons_start:]))
            vol_contract = pole_vol > 0 and cons_vol < pole_vol * 0.90

            spread_start = (m_up * cons_start + b_up) - (m_lo * cons_start + b_lo)
            spread_end = (m_up * (n - 1) + b_up) - (m_lo * (n - 1) + b_lo)
            if spread_start <= 0 or spread_end <= 0:
                continue

            slope_diff = abs(m_up - m_lo)
            is_converging = m_up < 0 and m_lo > 0 and spread_end < spread_start * 0.72
            is_flag_parallel = (
                slope_diff <= max(atr * 0.08, 0.01)
                and ((is_bull_pole and m_up <= 0.05 * atr) or (not is_bull_pole and m_lo >= -0.05 * atr))
            )

            if is_converging:
                name = "Bullish Pennant" if is_bull_pole else "Bearish Pennant"
            elif is_flag_parallel:
                name = "Bullish Flag" if is_bull_pole else "Bearish Flag"
            else:
                continue

            bias = "bullish" if is_bull_pole else "bearish"
            upper_end = m_up * (n - 1) + b_up
            lower_end = m_lo * (n - 1) + b_lo

            status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                df, n - 3, upper_end, lower_end, bias, atr
            )
            if status == "invalidated":
                continue

            target = round(upper_end + pole_abs * 0.85, 2) if is_bull_pole else round(lower_end - pole_abs * 0.85, 2)
            stop = round(lower_end - atr * 0.4, 2) if is_bull_pole else round(upper_end + atr * 0.4, 2)

            geom = 1.0 - min(1.0, retrace)
            conf = compute_confidence(geom, 4, n - 1, n, status, vol_ok, vol_contract)

            up_pts = [
                ChartPoint(time=dates[cons_start], value=round(m_up * cons_start + b_up, 2)),
                ChartPoint(time=dates[n - 1], value=round(upper_end, 2)),
            ]
            lo_pts = [
                ChartPoint(time=dates[cons_start], value=round(m_lo * cons_start + b_lo, 2)),
                ChartPoint(time=dates[n - 1], value=round(lower_end, 2)),
            ]
            pivs = [
                PivotPoint(time=dates[pole_start], value=round(float(closes[pole_start]), 2), type="low" if is_bull_pole else "high", label="Pole0", index=pole_start),
                PivotPoint(time=dates[cons_start], value=round(float(closes[cons_start]), 2), type="high" if is_bull_pole else "low", label="Pole1", index=cons_start),
            ]
            sub_df = df.iloc[pole_start:]

            return [
                DetectedPattern(
                    pattern_name=name,
                    pattern_category="continuation",
                    bias=bias,  # type: ignore[arg-type]
                    formation_status=status,
                    confidence_score=conf,
                    start_date=dates[pole_start],
                    end_date=dates[-1],
                    detected_date=dates[-1],
                    breakout_level=b_lvl,
                    target_price=target,
                    stop_loss=stop,
                    breakout_direction=b_dir,
                    volume_confirmed=vol_ok,
                    volume_ratio=vol_ratio,
                    atr_at_detection=round(atr, 2),
                    upper_trendline=up_pts,
                    lower_trendline=lo_pts,
                    pivot_points=pivs,
                    boundaries=PatternBoundaries(
                        start_date=dates[pole_start],
                        end_date=dates[-1],
                        min_price=round(float(sub_df["low"].min()), 2),
                        max_price=round(float(sub_df["high"].max()), 2),
                    ),
                    notes=f"{name} after {round(pole_pct * 100, 1)}% pole move ({round(retrace * 100, 1)}% retracement).",
                )
            ]

        return []
