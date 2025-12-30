import pandas as pd
from analysis.kline.service import run_kline_analysis

def test_run_kline_analysis():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30, freq="D"),
                       "open": range(30), "high": range(30), "low": range(30),
                       "close": range(30), "symbol": ["TEST.SYM"]*30})
    res = run_kline_analysis(df, "date", 5, 10)
    assert "data" in res
    assert "ind" in res["data"]
