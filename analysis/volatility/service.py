from datetime import datetime
import pandas as pd
from src.analysis.timeseries import compute_volatility, compute_garch

def run_volatility_analysis(df_symbol: pd.DataFrame, window: int = 20) -> dict:
    vol = compute_volatility(df_symbol["close"], window=window)
    sigma2 = None
    try:
        res, forecast = compute_garch(df_symbol["close"])
        sigma2 = forecast.variance.values[-1][-1] if forecast is not None else None
    except Exception:
        sigma2 = None
    ts = datetime.now().isoformat()
    return {"timestamp": ts, "data": {"vol_index": vol.index.astype(str).tolist(), "vol_values": vol.astype(float).tolist(), "sigma2": sigma2}}
