from typing import List
import numpy as np
import pandas as pd

from ..schemas import DetectedPattern, PatternBoundaries, PivotPoint
from ..trendlines import fit_line_segment
from .base import (
    BasePatternDetector,
    check_channel_containment,
    compute_confidence,
    evaluate_breakout_and_status,
)


class ChannelPatternDetector(BasePatternDetector):
    """
    Detects horizontal Rectangle Patterns and sloping Ascending / Descending Channels
    using parallel upper resistance & lower support linear regressions with containment checks.
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
        flat_slope_tol = (atr * 0.14) / 10.0

        for win in (4, 3):
            if len(swing_highs) < win or len(swing_lows) < win:
                continue
            h_slice = swing_highs[-win:]
            l_slice = swing_lows[-win:]
            s_idx = min(h_slice[0].index or 0, l_slice[0].index or 0)
            e_idx = max(h_slice[-1].index or 0, l_slice[-1].index or 0)
            if (e_idx - s_idx) < 16 or (e_idx - s_idx) > 150 or (n - 1 - e_idx) > 25:
                continue

            hx = np.array([p.index for p in h_slice], dtype=float)
            hy = np.array([p.value for p in h_slice], dtype=float)
            lx = np.array([p.index for p in l_slice], dtype=float)
            ly = np.array([p.value for p in l_slice], dtype=float)

            m_up, b_up, up_pts = fit_line_segment(hx, hy, s_idx, n - 1, dates)
            m_lo, b_lo, lo_pts = fit_line_segment(lx, ly, s_idx, n - 1, dates)

            slope_diff = abs(m_up - m_lo)
            if slope_diff > max(flat_slope_tol * 1.2, 0.015):
                continue

            avg_slope = (m_up + m_lo) / 2.0
            upper_end = m_up * (n - 1) + b_up
            lower_end = m_lo * (n - 1) + b_lo
            height = upper_end - lower_end
            if height < atr * 1.4:
                continue

            if not check_channel_containment(df, s_idx, e_idx, m_up, b_up, m_lo, b_lo, atr):
                continue

            res_up = float(np.mean(np.abs(hy - (m_up * hx + b_up))))
            res_lo = float(np.mean(np.abs(ly - (m_lo * lx + b_lo))))
            if (res_up + res_lo) > atr * 1.4:
                continue

            if abs(avg_slope) <= flat_slope_tol:
                name, cat, bias = "Rectangle", "bilateral", "neutral"
            elif avg_slope > flat_slope_tol:
                name, cat, bias = "Ascending Channel", "channel", "bullish"
            else:
                name, cat, bias = "Descending Channel", "channel", "bearish"

            status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                df, e_idx, upper_end, lower_end, bias, atr  # type: ignore[arg-type]
            )
            if status == "invalidated":
                continue

            target = None
            stop = None
            if b_dir == "bullish" or bias == "bullish":
                target = round(upper_end + height, 2)
                stop = round(lower_end - atr * 0.4, 2)
            elif b_dir == "bearish" or bias == "bearish":
                target = round(lower_end - height, 2)
                stop = round(upper_end + atr * 0.4, 2)

            geom = 1.0 - min(1.0, (res_up + res_lo) / max(1e-6, atr * 1.4))
            touches = len(h_slice) + len(l_slice)
            conf = compute_confidence(geom, touches, e_idx, n, status, vol_ok)

            pivs = [PivotPoint(time=p.time, value=p.value, type="high", label="CH", index=p.index) for p in h_slice] + [
                PivotPoint(time=p.time, value=p.value, type="low", label="CL", index=p.index) for p in l_slice
            ]
            pivs.sort(key=lambda p: p.index or 0)
            sub_df = df.iloc[s_idx:]

            return [
                DetectedPattern(
                    pattern_name=name,
                    pattern_category=cat,  # type: ignore[arg-type]
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
                    notes=f"{name} bounded between {round(lower_end, 2)} and {round(upper_end, 2)} ({touches} touches).",
                )
            ]

        return []
