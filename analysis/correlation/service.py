from datetime import datetime
import pandas as pd
from src.analysis.stats import compute_corr_matrix

def run_correlation_analysis(pivot_close: pd.DataFrame) -> dict:
    corr = compute_corr_matrix(pivot_close)
    ts = datetime.now().isoformat()
    return {"timestamp": ts, "data": {"corr": corr.to_dict()}}
