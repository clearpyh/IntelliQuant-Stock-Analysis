from pathlib import Path
import json
from typing import Optional, Dict, Any
import hashlib

def _mod_key(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def get_cache_dir(root: Path, symbol: str) -> Path:
    d = root / "export" / "cache" / symbol.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _hash_params(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def load(root: Path, symbol: str, module: str) -> Optional[Dict[str, Any]]:
    d = get_cache_dir(root, symbol)
    fp = d / f"{_mod_key(module)}.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None

def save(root: Path, symbol: str, module: str, payload: Dict[str, Any]) -> Path:
    d = get_cache_dir(root, symbol)
    fp = d / f"{_mod_key(module)}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return fp

def load_advisor(root: Path, symbol: str, module: str) -> Optional[Dict[str, Any]]:
    d = get_cache_dir(root, symbol)
    fp = d / f"{_mod_key(module)}_advisor.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_advisor(root: Path, symbol: str, module: str, payload: Dict[str, Any]) -> Path:
    d = get_cache_dir(root, symbol)
    fp = d / f"{_mod_key(module)}_advisor.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return fp

def is_stale(existing: Dict[str, Any], params: Dict[str, Any]) -> bool:
    if not existing:
        return True
    h = existing.get("params_hash")
    return h != _hash_params(params)

def build_payload(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data": data,
        "params": params,
        "params_hash": _hash_params(params),
        "timestamp": data.get("timestamp")
    }

def build_advisor_payload(text: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "advisor_text": text,
        "params": params,
        "params_hash": _hash_params(params),
        "timestamp": params.get("time_range")
    }
