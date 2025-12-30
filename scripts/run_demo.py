import os
from pathlib import Path
import sys
import pandas as pd
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "ohlcv"
EXPORT_DIR = ROOT / "export"

from src.analysis.timeseries import compute_indicators
from src.analysis.stats import compute_corr_matrix, compute_pca
from src.visualization import (
    plot_candlestick_with_indicators,
    plot_corr_heatmap,
    plot_pca_explained,
)
from src.conclusion import build_facts, generate_conclusions_with_llm


def collect_symbols_by_industry(industry: str, limit: int = 3) -> List[Path]:
    files = sorted(DATA_DIR.glob("*_*.csv"))
    selected = []
    for fp in files:
        try:
            dfh = pd.read_csv(fp, nrows=50)
        except UnicodeDecodeError:
            dfh = pd.read_csv(fp, nrows=50, encoding="gbk")
        if "industry" in dfh.columns and dfh["industry"].astype(str).str.contains(industry).any():
            selected.append(fp)
            if len(selected) >= limit:
                break
    return selected


def load_ohlcv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def run_for_industry(industry: str, start: str = "2018-01-01", end: str = "2025-12-31") -> Tuple[List[str], Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = EXPORT_DIR / industry
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_symbols_by_industry(industry, limit=3)
    charts = []
    corr = None
    explained = None

    # Load and analyze each symbol
    merged = []
    for fp in files:
        df = load_ohlcv(fp)
        df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
        if df.empty:
            continue
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else fp.stem.replace("_",".")
        ind = compute_indicators(df.set_index("date"), ma_short=20, ma_long=60)
        fig_k = plot_candlestick_with_indicators(df, date_col="date", indicators=ind)
        k_path = out_dir / f"{symbol}_kline.png"
        fig_k.write_image(str(k_path))
        charts.append(str(k_path))
        merged.append(df[["date","close","symbol"]].rename(columns={"close": symbol}))

        facts = build_facts(df, ind, None, None, symbol, industry)
        # Try LLM first, fallback to rule-based if empty
        cons = generate_conclusions_with_llm(facts, charts)
        if not cons:
            prev = os.environ.get("LLM_ENDPOINT")
            if prev:
                os.environ["LLM_ENDPOINT"] = ""
                cons = generate_conclusions_with_llm(facts, charts)
                os.environ["LLM_ENDPOINT"] = prev
        out_json = out_dir / f"{symbol}_conclusions.json"
        with open(out_json, "w", encoding="utf-8") as f:
            import json
            json.dump(cons, f, ensure_ascii=False, indent=2)

    # Correlation & PCA over selected symbols
    if merged:
        base = merged[0][["date"]]
        for m in merged:
            base = base.merge(m.drop(columns=["symbol"]), on="date", how="outer")
        pivot_close = base.set_index("date").dropna(how="all")
        if pivot_close.shape[1] >= 2:
            corr = compute_corr_matrix(pivot_close)
            fig_corr = plot_corr_heatmap(corr)
            c_path = out_dir / f"{industry}_corr.png"
            fig_corr.write_image(str(c_path))
            charts.append(str(c_path))
        if pivot_close.shape[1] >= 2:
            X = pivot_close.dropna().pct_change().dropna()
            _, explained = compute_pca(X)
            fig_pca = plot_pca_explained(explained)
            p_path = out_dir / f"{industry}_pca.png"
            fig_pca.write_image(str(p_path))
            charts.append(str(p_path))

    return charts, out_dir


def main():
    industries = ["汽车", "家用电器", "农林牧渔"]
    for ind in industries:
        charts, out_dir = run_for_industry(ind)
        print(f"{ind} 导出完成: {out_dir}")
        for c in charts:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
