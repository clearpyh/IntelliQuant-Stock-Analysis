import pandas as pd
from pathlib import Path

def _try_read_csv(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()

def _normalize_symbol(code_or_symbol: str) -> str:
    s = str(code_or_symbol).strip()
    if not s:
        return ""
    lc = s.lower()
    if lc.startswith("sh.") or lc.startswith("sz."):
        parts = s.split(".")
        return f"{parts[1]}.{'SH' if lc.startswith('sh.') else 'SZ'}"
    if "." in s and s.split(".")[1].upper() in ("SH","SZ"):
        return s.upper()
    if s.isdigit() and len(s) == 6:
        exch = "SH" if s.startswith("6") else "SZ"
        return f"{s}.{exch}"
    return s

def process_stock_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol","industry","name"])
    cols = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in cols:
        lc = c.lower()
        if lc in ("symbol", "代码", "code", "ts_code", "证券代码"):
            col_map[c] = "code"
        elif lc in ("industry", "行业", "板块"):
            col_map[c] = "industry"
        elif lc in ("name", "股票名称", "证券姓名", "sec_name", "stock_name", "code_name", "sname"):
            col_map[c] = "name"
        else:
            col_map[c] = lc
    df = df.rename(columns=col_map)
    symbol = None
    if "symbol" in df.columns:
        symbol = df["symbol"].astype(str)
    elif "code" in df.columns:
        symbol = df["code"].astype(str).map(_normalize_symbol)
    else:
        symbol = pd.Series("", index=df.index)
    
    name_col = None
    for c in ["name", "code_name", "stock_name", "sec_name", "sname", "股票名称", "证券姓名"]:
        if c in df.columns:
            name_col = c
            break
    
    name = df[name_col].astype(str) if name_col else pd.Series("", index=df.index)
    industry = df["industry"].astype(str) if "industry" in df.columns else pd.Series("未知", index=df.index)
    out = pd.DataFrame({"symbol": symbol, "name": name, "industry": industry})
    out = out.dropna(subset=["symbol"]).replace({"": pd.NA}).dropna(subset=["symbol"])
    return out

def load_stock_map(path: Path) -> pd.DataFrame:
    df = _try_read_csv(path)
    return process_stock_map(df)

def resolve_security(df_map: pd.DataFrame, query: str):
    if df_map is None or df_map.empty or not query:
        return None
    q = str(query).strip()
    sym_norm = _normalize_symbol(q)
    m1 = df_map[df_map["symbol"].astype(str).str.upper() == sym_norm.upper()]
    if not m1.empty:
        row = m1.iloc[0]
        return str(row["symbol"]), str(row["industry"]), str(row.get("name",""))
    m2 = df_map[df_map["symbol"].astype(str).str.contains(q, case=False, na=False)]
    if not m2.empty:
        row = m2.iloc[0]
        return str(row["symbol"]), str(row["industry"]), str(row.get("name",""))
    if "name" in df_map.columns:
        m3 = df_map[df_map["name"].astype(str).str.contains(q, case=False, na=False)]
        if not m3.empty:
            row = m3.iloc[0]
            return str(row["symbol"]), str(row["industry"]), str(row.get("name",""))
    return None
