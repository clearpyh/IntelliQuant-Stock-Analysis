from pathlib import Path
import pandas as pd
from typing import Optional
from data import fetch_resolved_df, find_local_ohlcv
from src.report import summarize_fundamentals
from src.data_io import safe_read_csv

def load_ohlcv(session_state, root: Path, symbol: str, industry: str, start_date: str, end_date: str, frequency: str) -> pd.DataFrame:
    local_path = find_local_ohlcv(root, symbol)
    if local_path:
        try:
            return safe_read_csv(local_path)
        except Exception:
            return pd.read_csv(local_path, encoding="gbk")
    return fetch_resolved_df(session_state, symbol, industry, start_date, end_date, frequency)

def load_fundamentals(root: Path, symbol: str) -> dict:
    sym_dir = Path(root) / "data" / "fundamentals" / symbol.replace(".", "_")
    return summarize_fundamentals(sym_dir)
