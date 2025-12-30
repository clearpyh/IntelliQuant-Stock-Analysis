import os
from pathlib import Path

def load_local_env(files=(".env.local", ".env")):
    root = Path(__file__).resolve().parents[1]
    for name in files:
        fp = root / name
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8")
        except Exception:
            content = fp.read_text()  # fallback
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                os.environ[key] = val
