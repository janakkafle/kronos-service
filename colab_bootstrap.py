import subprocess
import os
import sys
import time

print("=" * 60)
print("🚀 BOOTSTRAPPING KRONOS GPU INFERENCE SERVER ON COLAB")
print("=" * 60)

# Step 1: Clone Kronos Architecture
if not os.path.exists("/content/Kronos"):
    print("[1/4] Cloning official Kronos architecture...")
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", "/content/Kronos"], check=True)
else:
    print("[1/4] /content/Kronos already exists.")

# Step 2: Clone our standalone Kronos Microservice
if not os.path.exists("/content/kronos_service"):
    print("[2/4] Cloning kronos_service repo...")
    subprocess.run(["git", "clone", "https://github.com/janakkafle/kronos-service.git", "/content/kronos_service"], check=True)
else:
    print("[2/4] Pulling latest updates in /content/kronos_service...")
    subprocess.run(["git", "-C", "/content/kronos_service", "pull"], check=False)

# Step 3: Install Required Dependencies
print("[3/4] Installing Python dependencies...")
packages = [
    "fastapi", "uvicorn", "pydantic", "pyngrok", 
    "transformers", "accelerate", "safetensors", 
    "pandas", "numpy", "scipy", "scikit-learn", "pytrendline", "requests"
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)

# Step 4: Configure Ngrok with static domain
print("[4/4] Setting up Ngrok with static domain...")
NGROK_AUTH_TOKEN = "3M59t38Z8aG23qQfLsCx5hm0PKK_2UZ26SHAJhwVyniPYpV6u"
NGROK_DOMAIN = "moisten-vacation-wizard.ngrok-free.dev"

from pyngrok import ngrok
ngrok.kill()
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

try:
    tunnel = ngrok.connect(8000, domain=NGROK_DOMAIN)
except Exception as e:
    print(f"Domain bind note: {e}, attempting url param...")
    try:
        tunnel = ngrok.connect(8000, url=f"https://{NGROK_DOMAIN}")
    except Exception:
        tunnel = ngrok.connect(8000)

print("\n" + "=" * 60)
print("⚡ NGROK TUNNEL ONLINE!")
print("Public Static URL :", tunnel.public_url)
print("Keep-Alive Poke URL:", f"{tunnel.public_url}/poke")
print("=" * 60)

# Step 5: Start Uvicorn background process
print("\nStarting FastAPI Uvicorn server on port 8000...")
subprocess.run("pkill -f uvicorn", shell=True)
time.sleep(1)

env = os.environ.copy()
env["PYTHONPATH"] = f"/content/Kronos:/content/kronos_service:{env.get('PYTHONPATH', '')}"

log_file = open("/content/uvicorn.log", "w")
proc = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/content/kronos_service",
    env=env,
    stdout=log_file,
    stderr=log_file
)
print(f"Uvicorn server running in background (PID: {proc.pid})")

# Wait and verify health
import requests
for i in range(25):
    time.sleep(2)
    try:
        r = requests.get("http://localhost:8000/poke", timeout=2)
        if r.status_code == 200:
            print("✅ Server successfully responded to /poke:", r.json())
            break
    except Exception:
        pass

print("\n🚀 SETUP COMPLETE! Kronos GPU server is ready to receive requests.")

