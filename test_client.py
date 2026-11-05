import requests
import json
from datetime import datetime, timedelta

API_URL = "http://127.0.0.1:8001"


def test_health():
    print("Testing /health endpoint on " + API_URL + "...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
        return r.status_code == 200
    except Exception as e:
        print(f"Could not connect to {API_URL}: {e}")
        return False


def test_predict():
    print("\nTesting /predict endpoint with sample candles...")
    now = datetime.now()
    candles = []
    base_price = 450.0

    for i in range(120):
        dt = now - timedelta(days=(120 - i))
        if dt.weekday() in (4, 5):
            continue
        base_price += (i % 5 - 2) * 1.5
        candles.append({
            "timestamp": dt.strftime("%Y-%m-%d"),
            "open": round(base_price - 1.0, 2),
            "high": round(base_price + 3.0, 2),
            "low": round(base_price - 2.0, 2),
            "close": round(base_price, 2),
            "volume": 125000 + (i * 100),
        })

    payload = {
        "symbol": "NABIL",
        "candles": candles,
        "pred_len": 10,
        "sample_count": 5,
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Symbol: {data.get('symbol')}")
        print(f"Last Close: {data.get('last_close')}")
        print(f"Target Price: {data.get('predicted_target_price')} ({data.get('predicted_change_pct')}%)")
        print(f"Trend: {data.get('predicted_trend')}")
        print(f"Confidence: {data.get('confidence_score')}%")
        print(f"TradingView Candles Generated: {len(data.get('tradingview_payload', {}).get('forecast_candles', []))}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False


if __name__ == "__main__":
    if test_health():
        test_predict()
