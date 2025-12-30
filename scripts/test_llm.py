import os
import json
from pathlib import Path
import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_local_env

def main():
    load_local_env()
    endpoint = os.environ.get("LLM_ENDPOINT", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    if not endpoint:
        print("LLM endpoint not configured")
        return
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "ping"},
            {"role": "user", "content": "请只返回纯JSON: [{\"ok\": true}]"}
        ],
        "temperature": 0.0
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=20)
        print("status:", resp.status_code)
        txt = resp.text
        print(txt[:800])
    except Exception as e:
        print("error:", repr(e))

if __name__ == "__main__":
    main()
