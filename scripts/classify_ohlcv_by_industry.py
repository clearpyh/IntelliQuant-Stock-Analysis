import sys
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV_DIR = ROOT / "data" / "ohlcv"
MAP_FP = ROOT / "stock_industry.csv"

def try_read_csv(fp: Path, nrows: int = None) -> pd.DataFrame:
    try:
        return pd.read_csv(fp, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(fp, nrows=nrows, encoding="gbk")
        except Exception:
            return pd.DataFrame()

def load_map() -> pd.DataFrame:
    if MAP_FP.exists():
        return try_read_csv(MAP_FP)
    return pd.DataFrame()

def infer_industry(df: pd.DataFrame, stock_map: pd.DataFrame) -> str:
    if "industry" in df.columns:
        s = df["industry"].dropna()
        if not s.empty:
            return str(s.iloc[0])
    sym = None
    if "symbol" in df.columns:
        s = df["symbol"].dropna()
        if not s.empty:
            sym = str(s.iloc[0])
    elif "code" in df.columns:
        s = df["code"].dropna()
        if not s.empty:
            sym = str(s.iloc[0])
            lc = sym.lower()
            if lc.startswith("sh.") or lc.startswith("sz."):
                parts = sym.split(".")
                sym = f"{parts[1]}.{'SH' if lc.startswith('sh.') else 'SZ'}"
    if sym and not stock_map.empty:
        dfm = stock_map.copy()
        if "symbol" in dfm.columns:
            row = dfm[dfm["symbol"].astype(str) == sym]
            if not row.empty:
                s = row["industry"].dropna()
                if not s.empty:
                    return str(s.iloc[0])
        elif "code" in dfm.columns:
            # normalize map to symbol
            codes = dfm["code"].astype(str)
            for idx, c in codes.items():
                lc = c.lower()
                ts = None
                if lc.startswith("sh."):
                    ts = c.split(".")[1] + ".SH"
                elif lc.startswith("sz."):
                    ts = c.split(".")[1] + ".SZ"
                elif "." in c and c.split(".")[1].upper() in ("SH","SZ"):
                    ts = c
                if ts == sym:
                    val = dfm.loc[idx, "industry"]
                    if pd.notna(val):
                        return str(val)
    return "未知"

def main():
    stock_map = load_map()
    if not OHLCV_DIR.exists():
        print("ohlcv dir not found:", OHLCV_DIR)
        return
    files = [p for p in OHLCV_DIR.glob("*.csv")]
    if not files:
        print("no top-level csv to classify")
        return
    for fp in files:
        dfh = try_read_csv(fp, nrows=5)
        ind = infer_industry(dfh, stock_map)
        target_dir = OHLCV_DIR / ind
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / fp.name
        shutil.move(str(fp), str(dest))
        print(f"moved: {fp.name} -> {target_dir}")

if __name__ == "__main__":
    main()
