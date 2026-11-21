from typing import List
import numpy as np
import pandas as pd

from ..schemas import DetectedPattern, PatternBoundaries, PivotPoint
from ..trendlines import fit_line_segment
from .base import (
    BasePatternDetector,
    check_channel_containment,
    check_volume_contraction,
    compute_confidence,
    evaluate_breakout_and_status,
)


class WedgePatternDetector(BasePatternDetector):
    """
    Detects Rising Wedges (bearish convergence upward) and Falling Wedges
    (bullish convergence downward) with strict containment and residual checks.
    """

    def detect(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
    ) -> List[DetectedPattern]:
        if len(swing_highs) < 2 or len(swing_lows) < 2 or df.empty:
            return []

        n = len(df)
        dates = df["date"].tolist()
        atr = float(df["atr"].iloc[-1])
        min_slope = (atr * 0.14) / 10.0

        for win in (4, 3):
            if len(swing_highs) < win or len(swing_lows) < win:
                continue
            h_slice = swing_highs[-win:]
            l_slice = swing_lows[-win:]
            s_idx = min(h_slice[0].index or 0, l_slice[0].index or 0)
            e_idx = max(h_slice[-1].index or 0, l_slice[-1].index or 0)
            if (e_idx - s_idx) < 14 or (e_idx - s_idx) > 130 or (n - 1 - e_idx) > 25:
                continue

            hx = np.array([p.index for p in h_slice], dtype=float)
            hy = np.array([p.value for p in h_slice], dtype=float)
            lx = np.array([p.index for p in l_slice], dtype=float)
            ly = np.array([p.value for p in l_slice], dtype=float)

            m_up, b_up, up_pts = fit_line_segment(hx, hy, s_idx, n - 1, dates)
            m_lo, b_lo, lo_pts = fit_line_segment(lx, ly, s_idx, n - 1, dates)

            is_rising = m_up > min_slope and m_lo > min_slope and m_lo > m_up * 1.15
            is_falling = m_up < -min_slope and m_lo < -min_slope and m_up < m_lo * 1.15
            if not (is_rising or is_falling):
                continue

            spread_start = (m_up * s_idx + b_up) - (m_lo * s_idx + b_lo)
            spread_end = (m_up * (n - 1) + b_up) - (m_lo * (n - 1) + b_lo)
            if spread_start <= atr * 1.2 or spread_end <= 0 or spread_end >= spread_start * 0.85:
                continue

            if not check_channel_containment(df, s_idx, e_idx, m_up, b_up, m_lo, b_lo, atr):
                continue

            res_up = float(np.mean(np.abs(hy - (m_up * hx + b_up))))
            res_lo = float(np.mean(np.abs(ly - (m_lo * lx + b_lo))))
            if (res_up + res_lo) > atr * 1.4:
                continue

            upper_end = m_up * (n - 1) + b_up
            lower_end = m_lo * (n - 1) + b_lo
            name = "Rising Wedge" if is_rising else "Falling Wedge"
            bias = "bearish" if is_rising else "bullish"

            status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                df, e_idx, upper_end, lower_end, bias, atr
            )
            if status == "invalidated":
                continue

            target = round(lower_end - spread_start * 0.8, 2) if is_rising else round(upper_end + spread_start * 0.8, 2)
            stop = round(upper_end + atr * 0.4, 2) if is_rising else round(lower_end - atr * 0.4, 2)

            geom = 1.0 - min(1.0, (res_up + res_lo) / max(1e-6, atr * 1.4))
            touches = len(h_slice) + len(l_slice)
            vol_contract = check_volume_contraction(df, s_idx, e_idx)
            conf = compute_confidence(geom, touches, e_idx, n, status, vol_ok, vol_contract)

            pivs = [PivotPoint(time=p.time, value=p.value, type="high", label="W-H", index=p.index) for p in h_slice] + [
                PivotPoint(time=p.time, value=p.value, type="low", label="W-L", index=p.index) for p in l_slice
            ]
            pivs.sort(key=lambda p: p.index or 0)
            sub_df = df.iloc[s_idx:]

            return [
                DetectedPattern(
                    pattern_name=name,
                    pattern_category="reversal",
                    bias=bias,  # type: ignore[arg-type]
                    formation_status=status,
                    confidence_score=conf,
                    start_date=dates[s_idx],
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
                        start_date=dates[s_idx],
                        end_date=dates[-1],
                        min_price=round(float(sub_df["low"].min()), 2),
                        max_price=round(float(sub_df["high"].max()), 2),
                    ),
                    notes=f"{name} narrowing from spread {round(spread_start, 2)} to {round(spread_end, 2)}.",
                )
            ]

        return []
