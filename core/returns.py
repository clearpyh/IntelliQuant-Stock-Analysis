import numpy as np
import pandas as pd

def annualized_returns(pivot_close: pd.DataFrame) -> pd.Series:
    return pivot_close.pct_change(fill_method=None).dropna().mean() * 252

def annualized_volatility(pivot_close: pd.DataFrame) -> pd.Series:
    return pivot_close.pct_change(fill_method=None).dropna().std() * np.sqrt(252)
