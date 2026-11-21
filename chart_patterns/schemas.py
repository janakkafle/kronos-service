from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PatternCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class ChartPoint(BaseModel):
    time: str
    value: float


class PivotPoint(BaseModel):
    time: str
    value: float
    type: Literal["high", "low"]
    label: str = ""
    index: Optional[int] = None


class PatternBoundaries(BaseModel):
    start_date: str
    end_date: str
    min_price: float
    max_price: float


class DetectedTrendline(BaseModel):
    type: Literal["support", "resistance"]
    start: ChartPoint
    end: ChartPoint
    slope: float
    intercept: float
    touches: int
    score: float
    broken: bool = False


class DetectedPattern(BaseModel):
    pattern_name: str
    pattern_category: Literal["reversal", "continuation", "bilateral", "channel"]
    bias: Literal["bullish", "bearish", "neutral"]
    formation_status: Literal["forming", "confirmed", "invalidated"]
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    start_date: str
    end_date: str
    detected_date: str
    breakout_level: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    breakout_direction: Literal["bullish", "bearish", "none"] = "none"
    volume_confirmed: bool = False
    volume_ratio: Optional[float] = None
    atr_at_detection: Optional[float] = None
    upper_trendline: List[ChartPoint] = Field(default_factory=list)
    lower_trendline: List[ChartPoint] = Field(default_factory=list)
    neckline: List[ChartPoint] = Field(default_factory=list)
    pivot_points: List[PivotPoint] = Field(default_factory=list)
    boundaries: PatternBoundaries
    notes: str = ""


class ChartPatternRequest(BaseModel):
    symbol: str
    candles: List[PatternCandle]
    lookback_bars: int = Field(default=180, ge=30, le=1000)
    pivot_order: int = Field(default=5, ge=2, le=15)
    atr_period: int = Field(default=14, ge=5, le=50)
    as_of_date: Optional[str] = None
    offset_days: int = Field(default=0, ge=0, le=500)
    min_confidence: float = Field(default=45.0, ge=0.0, le=100.0)


class ChartPatternResponse(BaseModel):
    symbol: str
    detected_date: str
    bars_analyzed: int
    atr_value: float
    last_close: float
    patterns_count: int
    patterns: List[DetectedPattern] = Field(default_factory=list)
    swing_highs: List[PivotPoint] = Field(default_factory=list)
    swing_lows: List[PivotPoint] = Field(default_factory=list)
    support_trendlines: List[DetectedTrendline] = Field(default_factory=list)
    resistance_trendlines: List[DetectedTrendline] = Field(default_factory=list)


class BatchChartPatternRequest(BaseModel):
    items: List[ChartPatternRequest]


class BatchChartPatternResponse(BaseModel):
    success: bool
    processed_count: int
    elapsed_seconds: float
    results: List[ChartPatternResponse] = Field(default_factory=list)
    errors: Optional[Dict[str, str]] = None
