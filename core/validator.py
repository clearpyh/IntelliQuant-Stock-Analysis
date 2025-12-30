import pandas as pd

def ensure_non_empty(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty
