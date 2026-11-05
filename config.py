import os
import torch

HOST = os.getenv("KRONOS_HOST", "127.0.0.1")
PORT = int(os.getenv("KRONOS_PORT", "8001"))

MODEL_NAME = os.getenv("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-base")
TOKENIZER_NAME = os.getenv("KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base")

FORCE_CPU = os.getenv("KRONOS_FORCE_CPU", "false").lower() in ("true", "1", "yes")
if torch.cuda.is_available() and not FORCE_CPU:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DEFAULT_PREDICT_LEN = int(os.getenv("KRONOS_PREDICT_LEN", "10"))
MAX_LOOKBACK = int(os.getenv("KRONOS_MAX_LOOKBACK", "400"))
DEFAULT_SAMPLE_COUNT = int(os.getenv("KRONOS_SAMPLE_COUNT", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("KRONOS_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("KRONOS_TOP_P", "0.9"))
