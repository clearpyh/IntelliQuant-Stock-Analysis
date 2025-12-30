import pandas as pd
from pathlib import Path
from typing import Optional
from src.mapping import process_stock_map, resolve_security
from src.data_io import ts_to_baostock, fetch_kline_baostock

def read_stock_map_upload(file_obj) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj)
    except Exception:
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return pd.read_csv(file_obj, encoding="gbk")
        except Exception:
            return pd.DataFrame()

def auto_load_default_map(root: Path) -> Optional[pd.DataFrame]:
    default_map = root / "stock_industry.csv"
    if default_map.exists():
        df = pd.read_csv(default_map)
        return process_stock_map(df)
    return None

def resolve_query(df_map: pd.DataFrame, query: str):
    return resolve_security(df_map, query)

def find_local_ohlcv(root: Path, symbol: str) -> Optional[str]:
    folder = root / "data" / "ohlcv"
    pattern = f"{symbol.replace('.','_')}.csv"
    found = list(folder.rglob(pattern)) if folder.exists() else []
    return str(found[0]) if found else None

def fetch_resolved_df(session_state, symbol: str, industry: str, start_date: str, end_date: str, frequency: str) -> pd.DataFrame:
    fetch_key = f"{symbol}_{start_date}_{end_date}_{frequency}"
    use_state = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        use_state = get_script_run_ctx() is not None
    except Exception:
        use_state = False
    if use_state and session_state.get("last_fetch_key") == fetch_key and "resolved_df" in session_state:
        return session_state["resolved_df"]
    bs_code = ts_to_baostock(symbol)
    df_rt = fetch_kline_baostock(bs_code, start_date, end_date, frequency)
    if not df_rt.empty:
        df_rt["symbol"] = symbol
        df_rt["industry"] = industry
        if use_state:
            session_state["resolved_df"] = df_rt
            session_state["last_fetch_key"] = fetch_key
        return df_rt
    return pd.DataFrame()
