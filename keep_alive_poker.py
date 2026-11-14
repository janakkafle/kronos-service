"""
Kronos Cloud GPU Keep-Alive Poker
Continuously pings the static ngrok /poke endpoint every 10 seconds.
Generates zero GPU load while maintaining active network traffic so Google Colab
and ngrok tunnels remain active and never enter idle timeout.
"""

import time
import sys
import datetime
import requests

TARGET_URL = "https://moisten-vacation-wizard.ngrok-free.dev/poke"
INTERVAL_SECONDS = 10

print("=" * 65)
print("💓 KRONOS COLAB GPU KEEP-ALIVE POKER")
print(f"Target URL : {TARGET_URL}")
print(f"Interval   : Every {INTERVAL_SECONDS} seconds")
print("Press Ctrl+C at any time to stop.")
print("=" * 65)

count = 0
while True:
    count += 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        start = time.time()
        resp = requests.get(TARGET_URL, timeout=8)
        elapsed_ms = round((time.time() - start) * 1000, 1)

        if resp.status_code == 200:
            data = resp.json()
            remote_pokes = data.get("poke_count", count)
            uptime = data.get("uptime_seconds", 0)
            print(f"[{timestamp}] 💓 Poke #{remote_pokes} SUCCESS ({resp.status_code}) in {elapsed_ms}ms | Colab Uptime: {uptime}s")
        else:
            print(f"[{timestamp}] ⚠️ Poke #{count} returned HTTP {resp.status_code} in {elapsed_ms}ms")
    except requests.exceptions.RequestException as e:
        print(f"[{timestamp}] ⏳ Poke #{count} unreachable ({e.__class__.__name__}). Colab might be starting...")
    except KeyboardInterrupt:
        print("\nKeep-alive poker stopped by user.")
        sys.exit(0)

    time.sleep(INTERVAL_SECONDS)

