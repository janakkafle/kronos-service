from typing import List
from .detectors import (
    BasePatternDetector,
    ChannelPatternDetector,
    ContinuationPatternDetector,
    ReversalPatternDetector,
    TrianglePatternDetector,
    WedgePatternDetector,
)
from .pivots import detect_swings
from .preprocessing import prepare_ohlcv_dataframe
from .schemas import ChartPatternRequest, ChartPatternResponse, DetectedPattern
from .trendlines import detect_support_resistance_trendlines


class ChartPatternEngine:
    """
    Orchestrates OHLCV preprocessing, swing high/low detection, support/resistance
    trendline discovery, and modular classical chart pattern recognition.
    """

    def __init__(self, detectors: List[BasePatternDetector] | None = None):
        self.detectors: List[BasePatternDetector] = detectors or [
            ReversalPatternDetector(),
            TrianglePatternDetector(),
            WedgePatternDetector(),
            ContinuationPatternDetector(),
            ChannelPatternDetector(),
        ]

    def analyze(self, request: ChartPatternRequest) -> ChartPatternResponse:
        df = prepare_ohlcv_dataframe(
            candles=request.candles,
            lookback_bars=request.lookback_bars,
            atr_period=request.atr_period,
            as_of_date=request.as_of_date,
            offset_days=request.offset_days,
        )

        if df.empty or len(df) < 20:
            raise ValueError(
                f"Insufficient OHLCV history for {request.symbol} (need >= 20 valid sessions, got {len(df)})."
            )

        swing_highs, swing_lows = detect_swings(df, order=request.pivot_order)
        support_lines, resistance_lines = detect_support_resistance_trendlines(
            df, swing_highs, swing_lows
        )

        raw_patterns: List[DetectedPattern] = []
        for detector in self.detectors:
            raw_patterns.extend(detector.detect(df, swing_highs, swing_lows))

        filtered = [p for p in raw_patterns if p.confidence_score >= request.min_confidence]
        deduped = self._deduplicate_patterns(filtered)
        deduped.sort(
            key=lambda p: (1 if p.formation_status == "confirmed" else 0, p.confidence_score),
            reverse=True,
        )

        return ChartPatternResponse(
            symbol=request.symbol,
            detected_date=str(df["date"].iloc[-1]),
            bars_analyzed=len(df),
            atr_value=round(float(df["atr"].iloc[-1]), 2),
            last_close=round(float(df["close"].iloc[-1]), 2),
            patterns_count=len(deduped),
            patterns=deduped,
            swing_highs=swing_highs[-12:],
            swing_lows=swing_lows[-12:],
            support_trendlines=support_lines,
            resistance_trendlines=resistance_lines,
        )

    @staticmethod
    def _deduplicate_patterns(patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Keep highest-confidence detection when same pattern family overlaps on same start_date."""
        best_by_key: dict[tuple[str, str], DetectedPattern] = {}
        for p in patterns:
            key = (p.pattern_name, p.start_date)
            existing = best_by_key.get(key)
            if existing is None or p.confidence_score > existing.confidence_score:
                best_by_key[key] = p
        return list(best_by_key.values())
