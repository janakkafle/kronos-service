import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np

KRONOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

from chart_patterns import ChartPatternEngine, ChartPatternRequest
from chart_patterns.schemas import PatternCandle


def make_candles(closes: list[float], spread: float = 2.0, volumes: list[float] | None = None) -> list[PatternCandle]:
    start = datetime(2026, 1, 1)
    out = []
    for i, c in enumerate(closes):
        dt = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        vol = volumes[i] if volumes else 10000.0
        out.append(
            PatternCandle(
                timestamp=dt,
                open=round(c - 0.3, 2),
                high=round(c + spread, 2),
                low=round(c - spread, 2),
                close=round(c, 2),
                volume=vol,
            )
        )
    return out


class TestChartPatternEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ChartPatternEngine()

    def test_double_top_detection_and_breakout(self) -> None:
        # Rise to 200, dip to 170, rise to 200, drop below 170 neckline
        seg1 = np.linspace(150, 200, 15)
        seg2 = np.linspace(198, 170, 12)
        seg3 = np.linspace(172, 200, 12)
        seg4 = np.linspace(197, 162, 12)
        closes = np.concatenate([seg1, seg2, seg3, seg4]).tolist()
        vols = [10000.0] * (len(closes) - 5) + [25000.0] * 5
        req = ChartPatternRequest(symbol="NABIL", candles=make_candles(closes, spread=1.5, volumes=vols))
        res = self.engine.analyze(req)
        names = [p.pattern_name for p in res.patterns]
        self.assertIn("Double Top", names)
        dt_pat = next(p for p in res.patterns if p.pattern_name == "Double Top")
        self.assertEqual(dt_pat.bias, "bearish")
        self.assertEqual(dt_pat.formation_status, "confirmed")
        self.assertGreaterEqual(len(dt_pat.neckline), 2)

    def test_head_and_shoulders_detection(self) -> None:
        # LS=190, N1=170, H=215, N2=170, RS=190, breakdown=162
        segs = [
            np.linspace(150, 190, 12),
            np.linspace(188, 170, 10),
            np.linspace(172, 215, 12),
            np.linspace(213, 170, 12),
            np.linspace(172, 190, 10),
            np.linspace(188, 162, 10),
        ]
        closes = np.concatenate(segs).tolist()
        req = ChartPatternRequest(symbol="NICA", candles=make_candles(closes, spread=1.5), pivot_order=4)
        res = self.engine.analyze(req)
        names = [p.pattern_name for p in res.patterns]
        self.assertIn("Head & Shoulders", names)
        hs = next(p for p in res.patterns if p.pattern_name == "Head & Shoulders")
        self.assertEqual(len(hs.pivot_points), 5)
        self.assertEqual(hs.bias, "bearish")

    def test_ascending_triangle_detection(self) -> None:
        # Flat resistance near 200, rising lows: 165 -> 178 -> 189 -> breakout 206
        segs = [
            np.linspace(165, 200, 10),
            np.linspace(198, 175, 8),
            np.linspace(177, 200, 8),
            np.linspace(198, 184, 7),
            np.linspace(186, 200, 7),
            np.linspace(198, 190, 6),
            np.linspace(192, 208, 6),
        ]
        closes = np.concatenate(segs).tolist()
        vols = [10000.0] * (len(closes) - 4) + [28000.0] * 4
        req = ChartPatternRequest(symbol="SHIVM", candles=make_candles(closes, spread=1.0, volumes=vols), pivot_order=3)
        res = self.engine.analyze(req)
        names = [p.pattern_name for p in res.patterns]
        self.assertIn("Ascending Triangle", names)

    def test_no_lookahead_bias_with_as_of_date(self) -> None:
        segs = [
            np.linspace(150, 200, 15),
            np.linspace(198, 170, 12),
            np.linspace(172, 200, 12),
            np.linspace(197, 162, 12),
        ]
        closes = np.concatenate(segs).tolist()
        candles = make_candles(closes, spread=1.5)
        cutoff = candles[35].timestamp
        res_full = self.engine.analyze(ChartPatternRequest(symbol="TEST", candles=candles[:36]))
        res_sliced = self.engine.analyze(ChartPatternRequest(symbol="TEST", candles=candles, as_of_date=cutoff))
        self.assertEqual(res_full.detected_date, res_sliced.detected_date)
        self.assertEqual(res_full.bars_analyzed, res_sliced.bars_analyzed)
        self.assertEqual([p.pattern_name for p in res_full.patterns], [p.pattern_name for p in res_sliced.patterns])


if __name__ == "__main__":
    unittest.main()
