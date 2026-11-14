import sys
import os
import time
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from kronos.config import (
        HOST,
        PORT,
        MODEL_NAME,
        TOKENIZER_NAME,
        DEVICE,
        DEFAULT_PREDICT_LEN,
    )
    from kronos.schemas import (
        StockPredictionRequest,
        StockPredictionResult,
        BatchPredictionRequest,
        BatchPredictionResponse,
        TradingViewPayload,
        BacktestRequest,
        BacktestResult,
        BatchBacktestRequest,
        BatchBacktestResponse,
    )
    from kronos.predictor import (
        init_kronos_model,
        execute_prediction,
        execute_backtest,
        _IS_REAL_MODEL_LOADED,
    )
except ImportError:
    from config import (
        HOST,
        PORT,
        MODEL_NAME,
        TOKENIZER_NAME,
        DEVICE,
        DEFAULT_PREDICT_LEN,
    )
    from schemas import (
        StockPredictionRequest,
        StockPredictionResult,
        BatchPredictionRequest,
        BatchPredictionResponse,
        TradingViewPayload,
        BacktestRequest,
        BacktestResult,
        BatchBacktestRequest,
        BatchBacktestResponse,
    )
    from predictor import (
        init_kronos_model,
        execute_prediction,
        execute_backtest,
        _IS_REAL_MODEL_LOADED,
    )

app = FastAPI(
    title="Kronos Financial Foundation Model API",
    description="High-performance Candlestick Time-Series Forecasting powered by Kronos-base",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SERVER_START_TIME = time.time()
_POKE_COUNT = 0

@app.on_event("startup")
def on_startup():
    init_kronos_model()


@app.get("/")
def index():
    return {
        "status": "online",
        "service": "Kronos-base Financial Time-Series Forecasting API",
        "engine": "Kronos (shiyu-coder/Kronos)",
        "model": MODEL_NAME,
        "device": DEVICE,
        "is_model_loaded": _IS_REAL_MODEL_LOADED,
    }


@app.get("/poke")
@app.get("/ping")
def poke():
    """
    Ultra-lightweight keep-alive heartbeat endpoint.
    Zero GPU load, zero database query, returns sub-millisecond response
    to keep Colab runtime & ngrok tunnel alive and active without load.
    """
    global _POKE_COUNT
    _POKE_COUNT += 1
    return {
        "status": "alive",
        "service": "kronos-inference",
        "poke_count": _POKE_COUNT,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _IS_REAL_MODEL_LOADED,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "default_horizon": DEFAULT_PREDICT_LEN,
    }


@app.post("/predict", response_model=StockPredictionResult)
def predict_stock(request: StockPredictionRequest):
    try:
        result = execute_prediction(request)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate Kronos forecast for {request.symbol}: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()
    results = []
    errors: Dict[str, str] = {}

    for item in request.items:
        try:
            pred = execute_prediction(item)
            results.append(pred)
        except Exception as exc:
            errors[item.symbol] = str(exc)

    elapsed = round(time.time() - start_time, 3)

    return BatchPredictionResponse(
        success=len(results) > 0,
        processed_count=len(results),
        elapsed_seconds=elapsed,
        results=results,
        errors=errors if errors else None,
    )


@app.post("/draw/tradingview", response_model=TradingViewPayload)
def get_tradingview_draw_payload(request: StockPredictionRequest):
    try:
        result = execute_prediction(request)
        return result.tradingview_payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to format TradingView drawing payload for {request.symbol}: {str(exc)}",
        )


@app.post("/backtest", response_model=BacktestResult)
def backtest_single_stock(request: BacktestRequest):
    try:
        return execute_backtest(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed for {request.symbol}: {str(exc)}",
        )


@app.post("/backtest/batch", response_model=BatchBacktestResponse)
def backtest_batch_stocks(request: BatchBacktestRequest):
    start_time = time.time()
    results = []
    errors: Dict[str, str] = {}

    for item in request.items:
        try:
            res = execute_backtest(item)
            results.append(res)
        except Exception as exc:
            errors[item.symbol] = str(exc)

    elapsed = round(time.time() - start_time, 3)

    return BatchBacktestResponse(
        success=len(results) > 0,
        processed_count=len(results),
        elapsed_seconds=elapsed,
        results=results,
        errors=errors if errors else None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
