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
    sample_count: Optional[int] = 5
    temperature: Optional[float] = 0.8
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
    range_p50: float
    range_p90: float


class TradingViewMarker(BaseModel):
    time: int
    position: str = "aboveBar"
    color: str = "#089981"
    shape: str = "arrowUp"
    text: str = "AI Target"


class TradingViewBandPoint(BaseModel):
    time: int
    upper: float
    median: float
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
    range_p50: float
    range_p90: float
    predicted_trend: str
    volatility_forecast: float
    risk_reward_ratio: float
    horizon_days: int
    model_version: str = "Kronos-base"
    predicted_candles: List[PredictedCandle]
    tradingview_payload: TradingViewPayload


class BatchPredictionResponse(BaseModel):
    success: bool
    processed_count: int
    elapsed_seconds: float
    results: List[StockPredictionResult]
    errors: Optional[Dict[str, str]] = None

