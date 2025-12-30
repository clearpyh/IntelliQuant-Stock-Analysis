import os
import pandas as pd
from typing import Any

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def fetch_kline_baostock(code: str, start_date: str, end_date: str, frequency: str = "d") -> pd.DataFrame:
    import baostock as bs
    lg = bs.login()
    rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close,volume", start_date=start_date, end_date=end_date, frequency=frequency, adjustflag="3")
    data_list = []
    while rs.error_code == "0" and rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_fina_tushare(ts_code: str, start_date: str, end_date: str, fields: str=None) -> pd.DataFrame:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(token) if token else ts.pro_api()
    try:
        ts_code = normalize_ts_code(ts_code)
        df = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_balancesheet(ts_code: str, start_date: str, end_date: str, fields: str=None) -> pd.DataFrame:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(token) if token else ts.pro_api()
    try:
        ts_code = normalize_ts_code(ts_code)
        df = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_income(ts_code: str, start_date: str, end_date: str, fields: str=None) -> pd.DataFrame:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(token) if token else ts.pro_api()
    try:
        ts_code = normalize_ts_code(ts_code)
        df = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_cashflow(ts_code: str, start_date: str, end_date: str, fields: str=None) -> pd.DataFrame:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(token) if token else ts.pro_api()
    try:
        ts_code = normalize_ts_code(ts_code)
        df = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_daily_basic(ts_code: str, start_date: str, end_date: str, fields: str=None) -> pd.DataFrame:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    pro = ts.pro_api(token) if token else ts.pro_api()
    try:
        ts_code = normalize_ts_code(ts_code)
        df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        return df
    except Exception:
        return pd.DataFrame()

def ts_to_baostock(ts_code: str) -> str:
    code, exch = ts_code.split(".")
    prefix = "sh" if exch.upper() == "SH" else "sz"
    return f"{prefix}.{code}"

def normalize_ts_code(s: str) -> str:
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""
    lc = ss.lower()
    if lc.startswith("sh.") or lc.startswith("sz."):
        parts = ss.split(".")
        return f"{parts[1]}.{'SH' if lc.startswith('sh.') else 'SZ'}"
    if "." in ss and ss.split(".")[1].upper() in ("SH","SZ"):
        return ss.upper()
    if ss.isdigit() and len(ss) == 6:
        exch = "SH" if ss.startswith("6") else "SZ"
        return f"{ss}.{exch}"
    return ss
def fetch_index_quote_bs(ts_code: str) -> dict:
    try:
        import baostock as bs
        bs_code = ts_to_baostock(ts_code)
        lg = bs.login()
        rs = bs.query_current(bs_code)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if not rows:
            return {}
        df = pd.DataFrame(rows, columns=rs.fields)
        # parse fields
        for c in df.columns:
            if c in ("open","high","low","price","volume","amount","turn","preclose"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        ts = None
        if "date" in df.columns:
            try:
                if "time" in df.columns:
                    ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce").iloc[-1]
                else:
                    ts = pd.to_datetime(df["date"].astype(str), errors="coerce").iloc[-1]
            except Exception:
                ts = None
        price = float(df["price"].iloc[-1]) if "price" in df.columns else None
        preclose = float(df["preclose"].iloc[-1]) if "preclose" in df.columns else None
        change = (price - preclose) if (price is not None and preclose is not None) else None
        pct = (change / preclose * 100.0) if (change is not None and preclose and preclose != 0) else None
        return {"index_code": normalize_ts_code(ts_code), "price": price, "change": change, "pct_change": pct, "timestamp": ts}
    except Exception:
        return {}
def export_single(ts_code: str, start: str, end: str, out_path, industry: str):
    bs_code = ts_to_baostock(ts_code)
    df = fetch_kline_baostock(bs_code, start, end)
    df["symbol"] = ts_code
    df["industry"] = industry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

def export_batch(stock_map_df: pd.DataFrame, industries: list, start: str, end: str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ind in industries:
        df_ind = stock_map_df[stock_map_df["industry"] == ind]
        symbols = []
        if "symbol" in df_ind.columns:
            symbols = df_ind["symbol"].dropna().astype(str).tolist()
        elif "code" in df_ind.columns:
            codes = df_ind["code"].dropna().astype(str).tolist()
            for c in codes:
                lc = c.lower()
                if lc.startswith("sh."):
                    symbols.append(c.split(".")[1] + ".SH")
                elif lc.startswith("sz."):
                    symbols.append(c.split(".")[1] + ".SZ")
                elif "." in c and c.split(".")[1] in ("SH","SZ"):
                    symbols.append(c)
        else:
            raise KeyError("stock_industry.csv 缺少 symbol 或 code 列")
        for ts_code in symbols:
            out_path = out_dir / f"{ts_code.replace('.','_')}.csv"
            export_single(ts_code, start, end, out_path, ind)

def export_financials_single(ts_code: str, start_date: str, end_date: str, base_dir, industry: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    sym_dir = base_dir / ts_code.replace('.', '_')
    sym_dir.mkdir(parents=True, exist_ok=True)
    need_fetch = {
        "fina_indicator": not (sym_dir / "fina_indicator.csv").exists(),
        "balancesheet": not (sym_dir / "balancesheet.csv").exists(),
        "income": not (sym_dir / "income.csv").exists(),
        "cashflow": not (sym_dir / "cashflow.csv").exists(),
    }
    fi = fetch_fina_tushare(ts_code, start_date, end_date) if need_fetch["fina_indicator"] else None
    bs = fetch_balancesheet(ts_code, start_date, end_date) if need_fetch["balancesheet"] else None
    ic = fetch_income(ts_code, start_date, end_date) if need_fetch["income"] else None
    cf = fetch_cashflow(ts_code, start_date, end_date) if need_fetch["cashflow"] else None
    for name, df in [("fina_indicator", fi), ("balancesheet", bs), ("income", ic), ("cashflow", cf)]:
        if df is None or df.empty:
            continue
        df["industry"] = industry
        df.to_csv(sym_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

def export_daily_basic_batch(stock_map_df: pd.DataFrame, industry: str, start_date: str, end_date: str, base_dir):
    df_ind = stock_map_df[stock_map_df["industry"] == industry]
    symbols = []
    if "symbol" in df_ind.columns:
        symbols = df_ind["symbol"].dropna().astype(str).tolist()
    elif "code" in df_ind.columns:
        codes = df_ind["code"].dropna().astype(str).tolist()
        for c in codes:
            lc = c.lower()
            if lc.startswith("sh."):
                symbols.append(c.split(".")[1] + ".SH")
            elif lc.startswith("sz."):
                symbols.append(c.split(".")[1] + ".SZ")
            elif "." in c and c.split(".")[1] in ("SH","SZ"):
                symbols.append(c)
    else:
        raise KeyError("stock_industry.csv 缺少 symbol 或 code 列")
    base_dir.mkdir(parents=True, exist_ok=True)
    for ts_code in symbols:
        try:
            df = fetch_daily_basic(ts_code, start_date, end_date)
            if df is None or df.empty:
                continue
            sym_dir = base_dir / ts_code.replace('.', '_')
            sym_dir.mkdir(parents=True, exist_ok=True)
            df["industry"] = industry
            df.to_csv(sym_dir / "daily_basic.csv", index=False, encoding="utf-8-sig")
        except Exception:
            continue

def validate_tushare_token() -> bool:
    try:
        import tushare as ts
        token = os.environ.get("TUSHARE_TOKEN", "")
        token = token.strip() if token else ""
        if not token:
            return False
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.query('stock_basic', limit=1)
        return isinstance(df, pd.DataFrame) and not df.empty
    except Exception:
        return False

def export_financials_batch(stock_map_df: pd.DataFrame, industry: str, start_date: str, end_date: str, base_dir):
    df_ind = stock_map_df[stock_map_df["industry"] == industry]
    symbols = []
    if "symbol" in df_ind.columns:
        symbols = df_ind["symbol"].dropna().astype(str).tolist()
    elif "code" in df_ind.columns:
        codes = df_ind["code"].dropna().astype(str).tolist()
        for c in codes:
            lc = c.lower()
            if lc.startswith("sh."):
                symbols.append(c.split(".")[1] + ".SH")
            elif lc.startswith("sz."):
                symbols.append(c.split(".")[1] + ".SZ")
            elif "." in c and c.split(".")[1] in ("SH","SZ"):
                symbols.append(c)
    else:
        raise KeyError("stock_industry.csv 缺少 symbol 或 code 列")
    base_dir.mkdir(parents=True, exist_ok=True)
    for ts_code in symbols:
        export_financials_single(ts_code, start_date, end_date, base_dir, industry)

def safe_read_csv(src: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(src, engine="pyarrow")
    except Exception:
        try:
            if hasattr(src, "seek"):
                try:
                    src.seek(0)
                except Exception:
                    pass
            return pd.read_csv(src)
        except Exception:
            try:
                if hasattr(src, "seek"):
                    try:
                        src.seek(0)
                    except Exception:
                        pass
                return pd.read_csv(src, encoding="gbk")
            except Exception:
                return pd.DataFrame()
