from pathlib import Path
from storage.repository import load_module_result, save_module_result, is_stale, build_payload

def load(root: Path, symbol: str, module: str):
    return load_module_result(root, symbol, module)

def save(root: Path, symbol: str, module: str, payload: dict):
    return save_module_result(root, symbol, module, payload)

def fresh(existing: dict, params: dict) -> bool:
    return not is_stale(existing, params)

def payload(data: dict, params: dict) -> dict:
    return build_payload(data, params)
