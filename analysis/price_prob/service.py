from datetime import datetime
import pandas as pd
from src.analysis.stats import compute_logistic_proba

def run_price_prob_analysis(pivot_close: pd.DataFrame, industry_symbols: list, feature_source: dict) -> dict:
    R = pivot_close.pct_change(fill_method=None).dropna().mean()*252
    thr = float(R.median())
    y_cls = (R > thr).astype(int)
    X_rows = []
    for s in industry_symbols:
        sym_df = feature_source.get(s)
        if sym_df is None or sym_df.empty:
            continue
        sym_df['date'] = pd.to_datetime(sym_df['date'])
        sym_df = sym_df.sort_values('date')
        slope_sma = float(sym_df["close"].rolling(20).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
        slope_ema = float(sym_df["close"].ewm(span=60, adjust=False).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
        X_rows.append({"symbol": s, "sma_slope": slope_sma, "ema_slope": slope_ema})
    X = pd.DataFrame(X_rows).set_index("symbol")
    X = X.loc[y_cls.index.intersection(X.index)]
    ts = datetime.now().isoformat()
    if X.empty or X.shape[0] < 5:
        return {"timestamp": ts, "data": {}}
    _, proba, auc = compute_logistic_proba(X, y_cls.loc[X.index])
    return {"timestamp": ts, "data": {"proba": list(proba), "auc": float(auc)}}
