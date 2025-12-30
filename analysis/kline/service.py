from datetime import datetime
import pandas as pd
from src.analysis.timeseries import compute_indicators, compute_adx

def run_kline_analysis(df_symbol: pd.DataFrame, date_col: str, ma_short: int, ma_long: int) -> dict:
    ind = compute_indicators(df_symbol.set_index(date_col), ma_short=ma_short, ma_long=ma_long)
    adx_series = compute_adx(df_symbol)
    adx_index = df_symbol[date_col].astype(str).tolist()
    ts = datetime.now().isoformat()
    return {
        "timestamp": ts,
        "data": {
            "ind": {
                "index": ind.index.astype(str).tolist(),
                "columns": list(ind.columns),
                "data": ind.astype(float).values.tolist()
            },
            "adx": {
                "index": adx_index,
                "values": adx_series.astype(float).tolist()
            }
        }
    }
