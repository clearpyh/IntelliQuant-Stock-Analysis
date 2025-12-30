import pandas as pd

def slice_by_days(df: pd.DataFrame, date_col: str, end_dt, days: int) -> pd.DataFrame:
    start_dt = pd.to_datetime(end_dt) - pd.Timedelta(days=days)
    out = df[df[date_col] >= start_dt]
    if out.empty:
        return df
    return out
