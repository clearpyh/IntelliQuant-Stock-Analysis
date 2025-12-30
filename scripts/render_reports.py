import sys
import os
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_local_env
from src.report import render_industry_report

def to_symbols(df):
    if "symbol" in df.columns:
        return df["symbol"].dropna().astype(str).tolist()
    elif "code" in df.columns:
        out = []
        for c in df["code"].dropna().astype(str).tolist():
            lc = c.lower()
            if lc.startswith("sh."):
                out.append(c.split(".")[1] + ".SH")
            elif lc.startswith("sz."):
                out.append(c.split(".")[1] + ".SZ")
            elif "." in c and c.split(".")[1].upper() in ("SH","SZ"):
                out.append(c)
        return out
    return []

def main():
    load_local_env()
    export_dir = ROOT / "export"
    funda_dir = ROOT / "data" / "fundamentals"
    stock_map = pd.read_csv(ROOT / "stock_industry.csv")
    inds = sys.argv[1:] if len(sys.argv) > 1 else ["汽车","家用电器","农林牧渔"]
    for ind in inds:
        df_ind = stock_map[stock_map["industry"] == ind]
        symbols = to_symbols(df_ind)
        charts = []
        ind_dir = export_dir / ind
        if ind_dir.exists():
            for fp in ind_dir.glob("*.png"):
                charts.append(str(fp))
        conclusions_paths = []
        for fp in export_dir.glob(f"*_{ind}_conclusions.json"):
            conclusions_paths.append(str(fp))
        out_fp = render_industry_report(ind, symbols, funda_dir, export_dir, charts, conclusions_paths)
        print(str(out_fp))

if __name__ == "__main__":
    main()
