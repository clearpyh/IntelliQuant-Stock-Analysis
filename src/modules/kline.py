from datetime import datetime
import pandas as pd
from src.analysis.timeseries import compute_indicators, compute_adx

def run(inputs: dict) -> dict:
    df_symbol: pd.DataFrame = inputs["df_symbol"]
    date_col: str = inputs["date_col"]
    ma_short: int = inputs["ma_short"]
    ma_long: int = inputs["ma_long"]
    ind = compute_indicators(df_symbol.set_index(date_col), ma_short=ma_short, ma_long=ma_long)
    adx_series = compute_adx(df_symbol)
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
                "index": adx_series.index.astype(str).tolist(),
                "values": adx_series.astype(float).tolist()
            }
        }
    }
