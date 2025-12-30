from datetime import datetime
import pandas as pd
from src.analysis.stats import compute_pca

def run(inputs: dict) -> dict:
    pivot_close: pd.DataFrame = inputs["pivot_close"]
    X = pivot_close.pct_change(fill_method=None).dropna()
    ts = datetime.now().isoformat()
    if X.empty:
        return {"timestamp": ts, "data": {}}
    _, explained = compute_pca(X)
    return {"timestamp": ts, "data": {"explained": list(explained)}}
