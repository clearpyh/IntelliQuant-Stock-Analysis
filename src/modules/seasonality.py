from datetime import datetime
import pandas as pd
from src.analysis.timeseries import compute_stl, compute_acf_pacf

def run(inputs: dict) -> dict:
    df_symbol: pd.DataFrame = inputs["df_symbol"]
    date_col: str = inputs["date_col"]
    period: int = inputs.get("period", 7)
    stl_res = compute_stl(df_symbol.set_index(date_col)["close"], period=period)
    a_vals, p_vals = None, None
    try:
        a_vals, p_vals = compute_acf_pacf(df_symbol["close"].pct_change(fill_method=None).dropna())
    except Exception:
        a_vals, p_vals = None, None
    ts = datetime.now().isoformat()
    return {
        "timestamp": ts,
        "data": {
            "trend_index": stl_res.trend.index.astype(str).tolist(),
            "trend_values": stl_res.trend.astype(float).tolist(),
            "seasonal_index": stl_res.seasonal.index.astype(str).tolist(),
            "seasonal_values": stl_res.seasonal.astype(float).tolist(),
            "resid_index": stl_res.resid.index.astype(str).tolist(),
            "resid_values": stl_res.resid.astype(float).tolist(),
            "acf": a_vals.tolist() if a_vals is not None else None,
            "pacf": p_vals.tolist() if p_vals is not None else None
        }
    }
