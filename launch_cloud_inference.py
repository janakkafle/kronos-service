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

# Kill any existing uvicorn, ngrok or cloudflared
subprocess.run("pkill -f uvicorn", shell=True)
subprocess.run("pkill -f cloudflared", shell=True)
subprocess.run("pkill -f ngrok", shell=True)
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
        r = requests.get("http://localhost:8000/poke", timeout=2)
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

# Ngrok Static Domain Configuration
NGROK_AUTH_TOKEN = "3M59t38Z8aG23qQfLsCx5hm0PKK_2UZ26SHAJhwVyniPYpV6u"
NGROK_DOMAIN = "moisten-vacation-wizard.ngrok-free.dev"

public_url = None

print("Starting Ngrok Static Domain Tunnel...")
try:
    from pyngrok import ngrok
    ngrok.kill()
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    try:
        tunnel = ngrok.connect(8000, domain=NGROK_DOMAIN)
    except Exception as exc:
        print(f"Direct domain bind notice ({exc}), trying with url param...")
        try:
            tunnel = ngrok.connect(8000, url=f"https://{NGROK_DOMAIN}")
        except Exception:
            tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
except Exception as e:
    print(f"pyngrok failed: {e}. Trying cloudflared fallback...")
    cf_log = open("/content/cloudflared.log", "w")
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=cf_log,
        stderr=cf_log
    )
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
    print("\n" + "="*65)
    print("🚀 KRONOS INFERENCE SERVER IS LIVE ON CLOUD GPU (STATIC DOMAIN)!")
    print(f"Public Static URL : {public_url}")
    print(f"Keep-Alive Poke URL: {public_url}/poke")
    print("="*65)
    print("\nPre-configured in your Laravel .env:")
    print(f"KRONOS_API_URL={public_url}")
    print("="*65 + "\n")
    with open("/content/kronos_public_url.txt", "w") as f:
        f.write(public_url)
else:
    print("Could not obtain a public tunnel URL. Check logs above.")
