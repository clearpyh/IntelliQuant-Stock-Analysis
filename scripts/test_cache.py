from pathlib import Path
from src.cache import save_module_result, load_module_result, is_stale, build_payload

def run():
    root = Path(__file__).resolve().parents[1]
    symbol = "TEST.SYM"
    module = "PCA分析"
    data = {"explained": [0.5, 0.3, 0.2], "timestamp": "2025-01-01T00:00:00"}
    params = {"fetch_key": "A_B_C_D", "ma_short": 20, "ma_long": 60, "frequency": "d"}
    save_module_result(root, symbol, module, build_payload(data, params))
    loaded = load_module_result(root, symbol, module)
    assert loaded is not None
    assert not is_stale(loaded, params)
    assert is_stale(loaded, {"fetch_key": "X", "ma_short": 20, "ma_long": 60, "frequency": "d"})
    print("cache test ok")

if __name__ == "__main__":
    run()
