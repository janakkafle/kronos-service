"""
Classical Chart Pattern Detection Engine for NEPSE Technical Intelligence.

Detects classical geometric price-action structures (Head & Shoulders, Double/Triple
Tops & Bottoms, Triangles, Wedges, Flags, Pennants, Rectangles, Channels) using
SciPy peak detection, NumPy/Pandas vectorization, scikit-learn regression, and
pytrendline trendline discovery with ATR-based tolerances and zero look-ahead bias.
"""

from .engine import ChartPatternEngine
from .schemas import (
    BatchChartPatternRequest,
    BatchChartPatternResponse,
    ChartPatternRequest,
    ChartPatternResponse,
    DetectedPattern,
)

__all__ = [
    "ChartPatternEngine",
    "ChartPatternRequest",
    "ChartPatternResponse",
    "BatchChartPatternRequest",
    "BatchChartPatternResponse",
    "DetectedPattern",
]
