from datetime import datetime
import numpy as np
import pandas as pd
from src.analysis.stats import compute_kmeans

def run_risk_cluster_analysis(pivot_close: pd.DataFrame) -> dict:
    rets = pivot_close.pct_change(fill_method=None).dropna()
    ts = datetime.now().isoformat()
    if rets.empty or rets.shape[1] < 3:
        return {"timestamp": ts, "data": {}}
    feat = pd.DataFrame({"ret": rets.mean()*252, "vol": rets.std()*np.sqrt(252)}).dropna()
    if feat.shape[0] < 3:
        return {"timestamp": ts, "data": {}}
    n_clusters = min(3, max(2, feat.shape[0]//2))
    _, labels = compute_kmeans(feat, n_clusters=n_clusters)
    return {"timestamp": ts, "data": {"labels": labels.astype(int).tolist(), "index": list(feat.index), "n_clusters": int(n_clusters), "ret": feat["ret"].astype(float).tolist(), "vol": feat["vol"].astype(float).tolist()}}
