from pathlib import Path
from typing import Dict, Any
from storage.local_store import load, save, load_advisor, save_advisor, is_stale as _is_stale, build_payload as _build_payload, build_advisor_payload as _build_adv_payload

def load_module_result(root: Path, symbol: str, module: str):
    return load(root, symbol, module)

def save_module_result(root: Path, symbol: str, module: str, payload: Dict[str, Any]):
    return save(root, symbol, module, payload)

def is_stale(existing: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return _is_stale(existing, params)

def build_payload(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return _build_payload(data, params)

def load_module_advisor(root: Path, symbol: str, module: str):
    return load_advisor(root, symbol, module)

def save_module_advisor(root: Path, symbol: str, module: str, payload: Dict[str, Any]):
    return save_advisor(root, symbol, module, payload)

def build_advisor_payload(text: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return _build_adv_payload(text, params)
