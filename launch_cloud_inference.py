import subprocess
import time
import re
import sys
import os
import requests

print("=== Starting Kronos Inference Server on Colab GPU ===")

# Make sure Kronos official repo is in python path
kronos_official = "/content/Kronos"
kronos_service = "/content/kronos_service"

env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = f"{kronos_official}:{kronos_service}:{pythonpath}"

# Kill any existing uvicorn or cloudflared
subprocess.run("pkill -f uvicorn", shell=True)
subprocess.run("pkill -f cloudflared", shell=True)
time.sleep(1)

# Start uvicorn server in background
log_file = open("/content/uvicorn.log", "w")
uvicorn_proc = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=kronos_service,
    env=env,
    stdout=log_file,
    stderr=log_file
)
print("Uvicorn process started (PID:", uvicorn_proc.pid, ")")

# Wait for local server to be ready
ready = False
for i in range(15):
    time.sleep(1)
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            print("Local FastAPI server is HEALTHY:", r.json())
            ready = True
            break
    except Exception:
        pass

if not ready:
    print("Warning: server took longer than expected to start, checking logs:")
    with open("/content/uvicorn.log", "r") as f:
        print(f.read()[-500:])

# Start Cloudflare tunnel
print("Starting Cloudflare Tunnel...")
cf_log = open("/content/cloudflared.log", "w")
cf_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=cf_log,
    stderr=cf_log
)
print("Cloudflared process started (PID:", cf_proc.pid, ")")

# Extract public URL from cloudflared logs
public_url = None
for _ in range(20):
    time.sleep(1)
    if os.path.exists("/content/cloudflared.log"):
        with open("/content/cloudflared.log", "r") as f:
            content = f.read()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                public_url = match.group(0)
                break

if public_url:
    print("\n" + "="*60)
    print("🚀 KRONOS INFERENCE SERVER IS LIVE ON CLOUD GPU!")
    print(f"Public URL: {public_url}")
    print("="*60)
    print(f"\nUpdate your Laravel .env:")
    print(f"KRONOS_API_URL={public_url}")
    print("="*60 + "\n")
    with open("/content/kronos_public_url.txt", "w") as f:
        f.write(public_url)
else:
    print("Could not extract tunnel URL from cloudflared logs:")
    if os.path.exists("/content/cloudflared.log"):
        with open("/content/cloudflared.log", "r") as f:
            print(f.read())
