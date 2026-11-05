# Kronos Financial AI Inference Service

High-performance financial foundation model inference microservice powered by **Kronos-base** (`NeoQuasar/Kronos-base`). Built with FastAPI, PyTorch, and Hugging Face Transformers.

Designed for cloud GPU inference (Google Colab / AWS / RunPod / GCP) and integrates seamlessly with Laravel or any web application.

---

## ⚡ Quick Start: Google Colab GPU

The fastest way to run Kronos with free T4 GPU acceleration:

1. Open [`colab_launcher.ipynb`](colab_launcher.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Run all cells.
4. Copy the generated **Ngrok Tunnel URL** (e.g. `https://xyz.ngrok-free.app`).
5. Set `KRONOS_API_URL` in your Laravel `.env`:
   ```dotenv
   KRONOS_API_URL=https://xyz.ngrok-free.app
   ```

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

- `GET /health`: Health check and model readiness status.
- `POST /predict`: Generate candlestick forecasts and confidence bands for a single symbol.
- `POST /predict/batch`: Parallel batch predictions across multiple symbols.
- `POST /predict/tradingview`: Generates formatted historical + forecast candlestick series and confidence fan charts ready for TradingView Lightweight Charts.
