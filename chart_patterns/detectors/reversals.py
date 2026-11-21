from typing import List
import numpy as np
import pandas as pd

from ..schemas import ChartPoint, DetectedPattern, PatternBoundaries, PivotPoint
from .base import BasePatternDetector, compute_confidence, evaluate_breakout_and_status, has_prior_trend


class ReversalPatternDetector(BasePatternDetector):
    """
    Detects classical reversal chart structures using confirmed swing highs/lows
    and ATR-scaled horizontal/neckline tolerances:
      - Head & Shoulders / Inverse Head & Shoulders
      - Double Top / Double Bottom
      - Triple Top / Triple Bottom
    """

    def detect(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
    ) -> List[DetectedPattern]:
        if df.empty:
            return []

        patterns: List[DetectedPattern] = []
        patterns.extend(self._detect_head_and_shoulders(df, swing_highs, swing_lows, inverse=False))
        patterns.extend(self._detect_head_and_shoulders(df, swing_highs, swing_lows, inverse=True))
        patterns.extend(self._detect_double_or_triple(df, swing_highs, swing_lows, is_top=True))
        patterns.extend(self._detect_double_or_triple(df, swing_highs, swing_lows, is_top=False))
        return patterns

    def _detect_head_and_shoulders(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
        inverse: bool = False,
    ) -> List[DetectedPattern]:
        prim = swing_lows if inverse else swing_highs
        opp = swing_highs if inverse else swing_lows
        if len(prim) < 3 or len(opp) < 2:
            return []

        n = len(df)
        dates = df["date"].tolist()
        atr = float(df["atr"].iloc[-1])
        tol = max(atr * 0.75, float(df["close"].iloc[-1]) * 0.01)
        results: List[DetectedPattern] = []

        # Scan recent 3-peak combinations (favoring recent formations)
        for i in range(max(0, len(prim) - 7), len(prim) - 2):
            p1, p2, p3 = prim[i], prim[i + 1], prim[i + 2]
            if p1.index is None or p2.index is None or p3.index is None:
                continue
            if (p3.index - p1.index) < 12 or (p3.index - p1.index) > 140:
                continue

            # Head must stand clearly beyond both shoulders
            if not inverse:
                if not (p2.value > p1.value + atr * 0.5 and p2.value > p3.value + atr * 0.5):
                    continue
            else:
                if not (p2.value < p1.value - atr * 0.5 and p2.value < p3.value - atr * 0.5):
                    continue

            # Shoulders must be roughly symmetric in price
            shoulder_diff = abs(p1.value - p3.value)
            head_prominence = abs(p2.value - (p1.value + p3.value) / 2.0)
            if shoulder_diff > max(tol * 1.6, head_prominence * 0.45):
                continue

            # Find neckline troughs (or peaks for Inverse H&S) between p1-p2 and p2-p3
            n1_cands = [q for q in opp if q.index is not None and p1.index < q.index < p2.index]
            n2_cands = [q for q in opp if q.index is not None and p2.index < q.index < p3.index]
            if not n1_cands or not n2_cands:
                continue

            n1 = max(n1_cands, key=lambda x: x.value) if inverse else min(n1_cands, key=lambda x: x.value)
            n2 = max(n2_cands, key=lambda x: x.value) if inverse else min(n2_cands, key=lambda x: x.value)
            assert n1.index is not None and n2.index is not None

            # Neckline slope & projection to latest bar
            slope = (n2.value - n1.value) / max(1, n2.index - n1.index)
            intercept = n1.value - slope * n1.index
            # Reject wildly sloped necklines
            if abs(n2.value - n1.value) > head_prominence * 0.65:
                continue

            neck_at_end = slope * (n - 1) + intercept
            neck_at_p1 = slope * p1.index + intercept
            pattern_height = abs(p2.value - (slope * p2.index + intercept))
            if pattern_height < atr * 1.2:
                continue

            bias = "bullish" if inverse else "bearish"
            upper_lvl = neck_at_end if inverse else None
            lower_lvl = None if inverse else neck_at_end

            # Strict invalidation: if price after RS breached RS/Head, pattern is invalidated
            post_rs = df.iloc[p3.index + 1 :]
            if not post_rs.empty:
                if not inverse and float(post_rs["close"].max()) > p3.value + atr * 0.4:
                    continue
                if inverse and float(post_rs["close"].min()) < p3.value - atr * 0.4:
                    continue

            status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                df, p3.index, upper_lvl, lower_lvl, bias, atr
            )
            if status == "invalidated":
                continue
            if (n - 1 - p3.index) > 28:
                continue
            if not has_prior_trend(df, p1.index, "down" if inverse else "up", atr):
                continue

            target = round(neck_at_end + pattern_height, 2) if inverse else round(neck_at_end - pattern_height, 2)
            stop = round(p3.value - atr * 0.5, 2) if inverse else round(p3.value + atr * 0.5, 2)

            symmetry = 1.0 - min(1.0, shoulder_diff / max(1e-6, head_prominence))
            time_sym = 1.0 - min(1.0, abs((p2.index - p1.index) - (p3.index - p2.index)) / max(1, p3.index - p1.index))
            geom_score = 0.6 * symmetry + 0.4 * time_sym
            conf = compute_confidence(geom_score, 5, p3.index, n, status, vol_ok)

            neckline_pts = [
                ChartPoint(time=dates[p1.index], value=round(neck_at_p1, 2)),
                ChartPoint(time=dates[n - 1], value=round(neck_at_end, 2)),
            ]
            labeled_pivots = [
                PivotPoint(time=p1.time, value=p1.value, type=p1.type, label="LS", index=p1.index),
                PivotPoint(time=n1.time, value=n1.value, type=n1.type, label="N1", index=n1.index),
                PivotPoint(time=p2.time, value=p2.value, type=p2.type, label="H", index=p2.index),
                PivotPoint(time=n2.time, value=n2.value, type=n2.type, label="N2", index=n2.index),
                PivotPoint(time=p3.time, value=p3.value, type=p3.type, label="RS", index=p3.index),
            ]

            sub_df = df.iloc[p1.index :]
            results.append(
                DetectedPattern(
                    pattern_name="Inverse Head & Shoulders" if inverse else "Head & Shoulders",
                    pattern_category="reversal",
                    bias=bias,
                    formation_status=status,
                    confidence_score=conf,
                    start_date=p1.time,
                    end_date=dates[-1] if status == "forming" else p3.time,
                    detected_date=dates[-1],
                    breakout_level=b_lvl,
                    target_price=target,
                    stop_loss=stop,
                    breakout_direction=b_dir,
                    volume_confirmed=vol_ok,
                    volume_ratio=vol_ratio,
                    atr_at_detection=round(atr, 2),
                    upper_trendline=[ChartPoint(time=p1.time, value=p1.value), ChartPoint(time=p2.time, value=p2.value), ChartPoint(time=p3.time, value=p3.value)] if not inverse else neckline_pts,
                    lower_trendline=neckline_pts if not inverse else [ChartPoint(time=p1.time, value=p1.value), ChartPoint(time=p2.time, value=p2.value), ChartPoint(time=p3.time, value=p3.value)],
                    neckline=neckline_pts,
                    pivot_points=labeled_pivots,
                    boundaries=PatternBoundaries(
                        start_date=p1.time,
                        end_date=dates[-1],
                        min_price=round(float(sub_df["low"].min()), 2),
                        max_price=round(float(sub_df["high"].max()), 2),
                    ),
                    notes=f"{'Bullish' if inverse else 'Bearish'} H&S neckline at {round(neck_at_end, 2)}, target {target}.",
                )
            )
        return results

    def _detect_double_or_triple(
        self,
        df: pd.DataFrame,
        swing_highs: List[PivotPoint],
        swing_lows: List[PivotPoint],
        is_top: bool = True,
    ) -> List[DetectedPattern]:
        prim = swing_highs if is_top else swing_lows
        opp = swing_lows if is_top else swing_highs
        if len(prim) < 2 or len(opp) < 1:
            return []

        n = len(df)
        dates = df["date"].tolist()
        atr = float(df["atr"].iloc[-1])
        tol = max(atr * 0.65, float(df["close"].iloc[-1]) * 0.012)
        results: List[DetectedPattern] = []

        # Check Triple Top / Bottom first (3 recent pivots at same level)
        used_triples = set()
        if len(prim) >= 3:
            for i in range(max(0, len(prim) - 6), len(prim) - 2):
                p1, p2, p3 = prim[i], prim[i + 1], prim[i + 2]
                if p1.index is None or p2.index is None or p3.index is None:
                    continue
                if (p3.index - p1.index) < 12 or (p3.index - p1.index) > 150:
                    continue

                vals = [p1.value, p2.value, p3.value]
                if (max(vals) - min(vals)) > tol * 1.35:
                    continue

                mids = [q for q in opp if q.index is not None and p1.index < q.index < p3.index]
                if len(mids) < 2:
                    continue

                neck_val = min(m.value for m in mids) if is_top else max(m.value for m in mids)
                avg_peak = float(np.mean(vals))
                height = abs(avg_peak - neck_val)
                if height < atr * 1.2:
                    continue

                bias = "bearish" if is_top else "bullish"
                post_p3 = df.iloc[p3.index + 1 :]
                if not post_p3.empty:
                    if is_top and float(post_p3["close"].max()) > max(vals) + atr * 0.4:
                        continue
                    if not is_top and float(post_p3["close"].min()) < min(vals) - atr * 0.4:
                        continue

                status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                    df, p3.index, None if is_top else neck_val, neck_val if is_top else None, bias, atr
                )
                if status == "invalidated":
                    continue
                if (n - 1 - p3.index) > 28:
                    continue
                if not has_prior_trend(df, p1.index, "up" if is_top else "down", atr):
                    continue

                used_triples.update([(p1.index, p2.index), (p2.index, p3.index)])
                target = round(neck_val - height, 2) if is_top else round(neck_val + height, 2)
                stop = round(max(vals) + atr * 0.5, 2) if is_top else round(min(vals) - atr * 0.5, 2)
                geom = 1.0 - min(1.0, (max(vals) - min(vals)) / max(1e-6, tol * 1.5))
                conf = compute_confidence(geom, 5, p3.index, n, status, vol_ok)

                neck_line = [
                    ChartPoint(time=p1.time, value=round(neck_val, 2)),
                    ChartPoint(time=dates[-1], value=round(neck_val, 2)),
                ]
                peak_line = [
                    ChartPoint(time=p1.time, value=round(avg_peak, 2)),
                    ChartPoint(time=dates[-1], value=round(avg_peak, 2)),
                ]
                sub_df = df.iloc[p1.index :]
                lbl = "T" if is_top else "B"
                results.append(
                    DetectedPattern(
                        pattern_name="Triple Top" if is_top else "Triple Bottom",
                        pattern_category="reversal",
                        bias=bias,
                        formation_status=status,
                        confidence_score=conf,
                        start_date=p1.time,
                        end_date=dates[-1] if status == "forming" else p3.time,
                        detected_date=dates[-1],
                        breakout_level=b_lvl,
                        target_price=target,
                        stop_loss=stop,
                        breakout_direction=b_dir,
                        volume_confirmed=vol_ok,
                        volume_ratio=vol_ratio,
                        atr_at_detection=round(atr, 2),
                        upper_trendline=peak_line if is_top else neck_line,
                        lower_trendline=neck_line if is_top else peak_line,
                        neckline=neck_line,
                        pivot_points=[
                            PivotPoint(time=p1.time, value=p1.value, type=p1.type, label=f"{lbl}1", index=p1.index),
                            PivotPoint(time=p2.time, value=p2.value, type=p2.type, label=f"{lbl}2", index=p2.index),
                            PivotPoint(time=p3.time, value=p3.value, type=p3.type, label=f"{lbl}3", index=p3.index),
                        ],
                        boundaries=PatternBoundaries(
                            start_date=p1.time,
                            end_date=dates[-1],
                            min_price=round(float(sub_df["low"].min()), 2),
                            max_price=round(float(sub_df["high"].max()), 2),
                        ),
                        notes=f"3-point {'resistance' if is_top else 'support'} near {round(avg_peak, 2)}, neckline {round(neck_val, 2)}.",
                    )
                )

        # Double Top / Double Bottom
        for i in range(max(0, len(prim) - 6), len(prim) - 1):
            p1, p2 = prim[i], prim[i + 1]
            if p1.index is None or p2.index is None:
                continue
            if (p1.index, p2.index) in used_triples:
                continue
            if (p2.index - p1.index) < 7 or (p2.index - p1.index) > 110:
                continue
            if abs(p1.value - p2.value) > tol:
                continue

            mids = [q for q in opp if q.index is not None and p1.index < q.index < p2.index]
            if not mids:
                continue
            mid = min(mids, key=lambda x: x.value) if is_top else max(mids, key=lambda x: x.value)
            avg_lvl = (p1.value + p2.value) / 2.0
            height = abs(avg_lvl - mid.value)
            if height < atr * 1.2:
                continue

            # Ensure no intermediate close breached the double top/bottom excessively
            between_df = df.iloc[p1.index + 1 : p2.index]
            if is_top and float(between_df["close"].max()) > max(p1.value, p2.value) + tol * 0.5:
                continue
            if not is_top and float(between_df["close"].min()) < min(p1.value, p2.value) - tol * 0.5:
                continue

            bias = "bearish" if is_top else "bullish"
            post_p2 = df.iloc[p2.index + 1 :]
            if not post_p2.empty:
                if is_top and float(post_p2["close"].max()) > max(p1.value, p2.value) + atr * 0.4:
                    continue
                if not is_top and float(post_p2["close"].min()) < min(p1.value, p2.value) - atr * 0.4:
                    continue

            status, b_dir, b_lvl, vol_ok, vol_ratio = evaluate_breakout_and_status(
                df, p2.index, None if is_top else mid.value, mid.value if is_top else None, bias, atr
            )
            if status == "invalidated":
                continue
            if (n - 1 - p2.index) > 28:
                continue
            if not has_prior_trend(df, p1.index, "up" if is_top else "down", atr):
                continue

            target = round(mid.value - height, 2) if is_top else round(mid.value + height, 2)
            stop = round(max(p1.value, p2.value) + atr * 0.5, 2) if is_top else round(min(p1.value, p2.value) - atr * 0.5, 2)
            geom = 1.0 - min(1.0, abs(p1.value - p2.value) / max(1e-6, tol))
            conf = compute_confidence(geom, 3, p2.index, n, status, vol_ok)

            neck_line = [
                ChartPoint(time=p1.time, value=round(mid.value, 2)),
                ChartPoint(time=dates[-1], value=round(mid.value, 2)),
            ]
            level_line = [
                ChartPoint(time=p1.time, value=round(avg_lvl, 2)),
                ChartPoint(time=dates[-1], value=round(avg_lvl, 2)),
            ]
            sub_df = df.iloc[p1.index :]
            lbl = "T" if is_top else "B"
            results.append(
                DetectedPattern(
                    pattern_name="Double Top" if is_top else "Double Bottom",
                    pattern_category="reversal",
                    bias=bias,
                    formation_status=status,
                    confidence_score=conf,
                    start_date=p1.time,
                    end_date=dates[-1] if status == "forming" else p2.time,
                    detected_date=dates[-1],
                    breakout_level=b_lvl,
                    target_price=target,
                    stop_loss=stop,
                    breakout_direction=b_dir,
                    volume_confirmed=vol_ok,
                    volume_ratio=vol_ratio,
                    atr_at_detection=round(atr, 2),
                    upper_trendline=level_line if is_top else neck_line,
                    lower_trendline=neck_line if is_top else level_line,
                    neckline=neck_line,
                    pivot_points=[
                        PivotPoint(time=p1.time, value=p1.value, type=p1.type, label=f"{lbl}1", index=p1.index),
                        PivotPoint(time=mid.time, value=mid.value, type=mid.type, label="Neck", index=mid.index),
                        PivotPoint(time=p2.time, value=p2.value, type=p2.type, label=f"{lbl}2", index=p2.index),
                    ],
                    boundaries=PatternBoundaries(
                        start_date=p1.time,
                        end_date=dates[-1],
                        min_price=round(float(sub_df["low"].min()), 2),
                        max_price=round(float(sub_df["high"].max()), 2),
                    ),
                    notes=f"2-point {'resistance' if is_top else 'support'} near {round(avg_lvl, 2)}, neckline {round(mid.value, 2)}.",
                )
            )
        return results
