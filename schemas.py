from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CandleInput(BaseModel):
    timestamp: Any = Field(..., description="Date string (YYYY-MM-DD) or Unix timestamp in seconds")
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: Optional[float] = None


class StockPredictionRequest(BaseModel):
    symbol: str
    candles: List[CandleInput]
    pred_len: Optional[int] = 10
    sample_count: Optional[int] = 15
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.9


class BatchPredictionRequest(BaseModel):
    items: List[StockPredictionRequest]


class PredictedCandle(BaseModel):
    step: int
    date: str
    unix_timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    range_p10: float
    range_p25: Optional[float] = None
    range_p50: float
    range_p75: Optional[float] = None
    range_p90: float
    win_prob: Optional[float] = None


class TradingViewMarker(BaseModel):
    time: int
    position: str = "aboveBar"
    color: str = "#089981"
    shape: str = "arrowUp"
    text: str = "AI Target"


class TradingViewBandPoint(BaseModel):
    time: int
    upper: float
    p75: Optional[float] = None
    median: float
    p25: Optional[float] = None
    lower: float


class TradingViewCandlePoint(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class TradingViewPayload(BaseModel):
    symbol: str
    last_close: float
    last_time: int
    forecast_candles: List[TradingViewCandlePoint]
    confidence_bands: List[TradingViewBandPoint]
    target_price: float
    stop_loss: float
    trend: str
    confidence_pct: float


class StockPredictionResult(BaseModel):
    symbol: str
    prediction_date: str
    last_close: float
    predicted_target_price: float
    predicted_change_pct: float
    predicted_high: float
    predicted_low: float
    confidence_score: float
    range_p10: float
    range_p25: Optional[float] = None
    range_p50: float
    range_p75: Optional[float] = None
    range_p90: float
    predicted_trend: str
    volatility_forecast: float
    risk_reward_ratio: float
    horizon_days: int
    model_version: str = "Kronos-base"
    quant_metrics: Optional[Dict[str, Any]] = None
    predicted_candles: List[PredictedCandle]
    tradingview_payload: TradingViewPayload


class BatchPredictionResponse(BaseModel):
    success: bool
    processed_count: int
    elapsed_seconds: float
    results: List[StockPredictionResult]
    errors: Optional[Dict[str, str]] = None


class BacktestRequest(BaseModel):
    symbol: str
    candles: List[CandleInput]
    offset_days: int = 10
    sample_count: Optional[int] = 15
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.9


class BatchBacktestRequest(BaseModel):
    items: List[BacktestRequest]


class BacktestDayComparison(BaseModel):
    step: int
    date: str
    unix_timestamp: int
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float
    actual_open: float
    actual_high: float
    actual_low: float
    actual_close: float
    range_p10: float
    range_p50: float
    range_p90: float
    abs_error: float
    pct_error: float
    within_cone: bool


class BacktestMetrics(BaseModel):
    mape_pct: float
    rmse: float
    terminal_error_pct: float
    predicted_change_pct: float
    actual_change_pct: float
    directional_hit: bool
    cone_coverage_pct: float
    bias_pct: float
    evaluated_days: int


class BacktestResult(BaseModel):
    symbol: str
    cutoff_date: str
    cutoff_close: float
    prediction: StockPredictionResult
    metrics: BacktestMetrics
    comparisons: List[BacktestDayComparison]


class BatchBacktestResponse(BaseModel):
    success: bool
    processed_count: int
    elapsed_seconds: float
    results: List[BacktestResult]
    errors: Optional[Dict[str, str]] = None
