import pandas as pd
import pandas_ta as ta
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model

def compute_indicators(df: pd.DataFrame, ma_short: int, ma_long: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["SMA"] = df["close"].rolling(ma_short).mean()
    out["EMA"] = df["close"].ewm(span=ma_long, adjust=False).mean()
    rsi = ta.rsi(df["close"]) if "close" in df.columns else None
    if rsi is not None:
        out["RSI"] = rsi.values
    return out

def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    adx = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=window)
    return adx["ADX_14"] if isinstance(adx, pd.DataFrame) and "ADX_14" in adx.columns else adx.squeeze()

def compute_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window).std() * np.sqrt(252)

def compute_garch(series: pd.Series):
    returns = series.pct_change().dropna() * 100
    model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    return res, forecast

def compute_stl(series: pd.Series, period: int = 7):
    stl = STL(series.dropna(), period=period, robust=True)
    res = stl.fit()
    return res

def compute_acf_pacf(series: pd.Series, nlags: int = 40):
    a = acf(series.dropna(), nlags=nlags, fft=True)
    p = pacf(series.dropna(), nlags=nlags)
    return np.array(a), np.array(p)

def compute_max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    cummax = s.cummax()
    dd = (s / cummax) - 1.0
    return float(dd.min()) if not dd.empty else 0.0

def compute_ann_return(series: pd.Series) -> float:
    r = series.pct_change().dropna()
    if r.empty:
        return 0.0
    return float(r.mean() * 252)

def compute_sharpe(series: pd.Series) -> float:
    r = series.pct_change().dropna()
    if r.empty:
        return 0.0
    mu = r.mean() * 252
    sig = r.std() * np.sqrt(252)
    return float(mu / sig) if sig and sig != 0 else 0.0

def compute_proximity_52w(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    last = float(s.iloc[-1])
    window = 252 if s.shape[0] >= 252 else s.shape[0]
    high = float(s.iloc[-window:].max())
    return float(last / high - 1.0) if high and high != 0 else 0.0

def compute_roc(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if s.empty or s.shape[0] <= window:
        return 0.0
    return float(s.diff(window).iloc[-1] / s.shift(window).iloc[-1]) if s.shift(window).iloc[-1] else 0.0
