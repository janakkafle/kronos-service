# Kronos Financial AI Inference Service

High-performance financial foundation model inference microservice powered by **Kronos-base** (`NeoQuasar/Kronos-base`). Built with FastAPI, PyTorch, and Hugging Face Transformers.

Designed for cloud GPU inference (Google Colab / AWS / RunPod / GCP) and integrates seamlessly with Laravel or any web application.

---

## ⚡ Quick Start: Google Colab GPU

The fastest way to run Kronos with free T4 GPU acceleration:

1. Open [`colab_launcher.ipynb`](colab_launcher.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Run cells (Step 1 to Step 4). Step 4 is pre-configured with your static domain:
   ```
   https://moisten-vacation-wizard.ngrok-free.dev
   ```
4. Your Laravel `.env` is already configured with:
   ```dotenv
   KRONOS_API_URL=https://moisten-vacation-wizard.ngrok-free.dev
   ```
5. **Keep-Alive (Prevent Colab Disconnect)**:
   - Run the 10-second poke daemon: `python keep_alive_poker.py` (or double-click `run_poker.bat`).
   - Or in Laravel: `php artisan kronos:keep-alive` (or automatic via Laravel schedule).

---

## 🛠 Local Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU

### Installation
```bash
# Clone the repository
git clone https://github.com/janakkafle/kronos-service.git
cd kronos-service

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally
```bash
# Windows quick start:
.\run_local.bat
# or PowerShell:
.\run_local.ps1
# or direct Python:
python main.py
```
The API server will run at `http://127.0.0.1:8000`.

---

## 📡 API Endpoints

- `GET /poke` (or `/ping`): Ultra-fast zero-load keep-alive heartbeat ping.
- `GET /health`: Health check and model readiness status.
- `POST /predict`: Generate candlestick forecasts and confidence bands for a single symbol.
- `POST /predict/batch`: Parallel batch predictions across multiple symbols.
- `POST /draw/tradingview`: Generates formatted historical + forecast candlestick series and confidence fan charts ready for TradingView Lightweight Charts.
