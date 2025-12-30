import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_io import fetch_kline_baostock, ts_to_baostock

def read_stock_map(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")

def normalize_symbols(df: pd.DataFrame, industry: str) -> list:
    col_ind = "industry" if "industry" in df.columns else None
    if col_ind is None:
        raise ValueError("行业映射缺少 industry 列")
    dfi = df[df[col_ind] == industry]
    if "symbol" in dfi.columns:
        return dfi["symbol"].dropna().tolist()
    if "code" in dfi.columns:
        codes = dfi["code"].dropna().tolist()
        syms = []
        for c in codes:
            # 支持 like 'sh.600006' / 'sz.000001'
            if isinstance(c, str) and c.lower().startswith("sh."):
                syms.append(c.split(".")[1] + ".SH")
            elif isinstance(c, str) and c.lower().startswith("sz."):
                syms.append(c.split(".")[1] + ".SZ")
        return syms
    raise ValueError("行业映射缺少 symbol/code 列")

def ts_to_baostock(ts_code: str) -> str:
    code, exch = ts_code.split(".")
    prefix = "sh" if exch.upper() == "SH" else "sz"
    return f"{prefix}.{code}"

def export_single(ts_code: str, start: str, end: str, out_path: Path, industry: str):
    bs_code = ts_to_baostock(ts_code)
    df = fetch_kline_baostock(bs_code, start, end)
    if df.empty:
        return
    df["symbol"] = ts_code
    df["industry"] = industry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

def export_batch(stock_map_path: Path, industry: str, start: str, end: str, out_dir: Path):
    stocks = read_stock_map(stock_map_path)
    symbols = normalize_symbols(stocks, industry)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ts_code in symbols:
        out_path = out_dir / f"{ts_code.replace('.','_')}.csv"
        export_single(ts_code, start, end, out_path, industry)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", help="如 600006.SH")
    p.add_argument("--industry", default="未知")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", help="输出CSV路径")
    p.add_argument("--stock-map", help="行业映射CSV路径")
    p.add_argument("--out-dir", help="批量输出目录")
    args = p.parse_args()
    if args.symbol and args.out:
        export_single(args.symbol, args.start, args.end, Path(args.out), args.industry)
    elif args.stock_map and args.industry and args.out_dir:
        export_batch(Path(args.stock_map), args.industry, args.start, args.end, Path(args.out_dir))
    else:
        raise SystemExit("参数不足：提供 --symbol 与 --out 或提供 --stock-map 与 --industry 与 --out-dir")

if __name__ == "__main__":
    main()
