from pathlib import Path
import pandas as pd
from functools import reduce
from src.cache import load_module_result, save_module_result, is_stale, build_payload
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.modules.kline import run as run_kline
from src.modules.correlation import run as run_corr
from src.modules.pca import run as run_pca
from src.modules.volatility import run as run_vol
from src.modules.seasonality import run as run_season
from src.modules.clustering import run as run_cluster
from src.modules.factor_portrait import run as run_factor_portrait
from src.modules.probability import run as run_prob
from data import fetch_resolved_df
from core.data_loader import load_ohlcv

MODULE_RUNNERS = {
    "K线与指标": run_kline,
    "相关性分析": run_corr,
    "PCA分析": run_pca,
    "波动性分析": run_vol,
    "季节性分析": run_season,
    "风险-收益聚类分析": run_cluster,
    "基本面因子暴露分析": run_factor_portrait,
    "涨跌概率分析": run_prob
}

def build_pivot_close(session_state, industry: str, sd_local, ed_local, frequency: str) -> pd.DataFrame:
    df_map = session_state.get("stock_map")
    if df_map is None or df_map.empty or industry not in df_map["industry"].values:
        return pd.DataFrame()
    df_ind = df_map[df_map["industry"] == industry]
    symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
    cache_key = f"{industry}|{str(sd_local)}|{str(ed_local)}|{frequency}|{len(symbols_in_industry)}"
    pivot_cache = session_state.setdefault("pivot_cache", {})
    if cache_key in pivot_cache:
        cached = pivot_cache.get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
    all_dfs = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(symbols_in_industry)))) as ex:
        futures = {}
        for sym in symbols_in_industry:
            futures[ex.submit(load_ohlcv, session_state, Path(__file__).parent.parent, sym, industry, str(sd_local), str(ed_local), frequency)] = sym
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                sym_df = fut.result()
                if sym_df is None or sym_df.empty:
                    sym_df = fetch_resolved_df(session_state, sym, industry, str(sd_local), str(ed_local), frequency)
                if not sym_df.empty:
                    df_i = sym_df[['date', 'close']].rename(columns={'close': sym})
                    df_i['date'] = pd.to_datetime(df_i['date'])
                    all_dfs.append(df_i)
            except Exception:
                pass
    if not all_dfs:
        return pd.DataFrame()
    pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
    pivot_close = pivot_close.set_index('date').sort_index()
    pivot_cache[cache_key] = pivot_close
    return pivot_close

def run_selected_modules(session_state, root: Path, symbol: str, industry: str, df_symbol: pd.DataFrame, date_col: str, sd_local, ed_local, frequency: str, ma_short: int, ma_long: int, selected: list):
    status = {}
    params_base = {"fetch_key": session_state.get("last_fetch_key"), "ma_short": ma_short, "ma_long": ma_long, "frequency": frequency, "start_date": str(sd_local), "end_date": str(ed_local), "industry": industry}
    pivot_close = None
    if any(m in selected for m in ("相关性分析","PCA分析","风险-收益聚类分析","涨跌概率分析")):
        pivot_close = build_pivot_close(session_state, industry, sd_local, ed_local, frequency)
    df_map = session_state.get("stock_map")
    symbols_in_industry = []
    if df_map is not None and industry in df_map["industry"].values:
        df_ind = df_map[df_map["industry"] == industry]
        symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
    feature_source = {}
    if "涨跌概率分析" in selected and symbols_in_industry:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(symbols_in_industry)))) as ex:
            futures = {}
            for s in symbols_in_industry:
                futures[ex.submit(fetch_resolved_df, session_state, s, industry, str(sd_local), str(ed_local), frequency)] = s
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    feature_source[s] = fut.result()
                except Exception:
                    pass
    for mod in selected:
        cached = load_module_result(root, symbol, mod)
        if cached and not is_stale(cached, params_base):
            status[mod] = {"status": "已完成", "time": cached.get("timestamp"), "fresh": True}
            continue
        runner = MODULE_RUNNERS.get(mod)
        if not runner:
            status[mod] = {"status": "失败", "time": None, "fresh": None}
            continue
        if mod == "K线与指标":
            inputs = {"df_symbol": df_symbol, "date_col": date_col, "ma_short": ma_short, "ma_long": ma_long}
        elif mod in ("相关性分析","PCA分析","风险-收益聚类分析"):
            inputs = {"pivot_close": pivot_close if pivot_close is not None else pd.DataFrame()}
        elif mod == "季节性分析":
            inputs = {"df_symbol": df_symbol, "date_col": date_col}
        elif mod == "波动性分析":
            inputs = {"df_symbol": df_symbol, "window": 20}
        elif mod == "涨跌概率分析":
            inputs = {"pivot_close": pivot_close if pivot_close is not None else pd.DataFrame(), "industry_symbols": symbols_in_industry, "feature_source": feature_source}
        elif mod == "基本面因子暴露分析":
            funda_dir_default = str((root / "data" / "fundamentals").resolve())
            fundamentals_dir = Path(session_state.get("funda_dir", funda_dir_default))
            inputs = {"symbol": symbol, "fundamentals_dir": fundamentals_dir}
        else:
            inputs = {}
        res = runner(inputs)
        data_to_save = res.get("data", {})
        data_to_save["timestamp"] = res.get("timestamp")
        save_module_result(root, symbol, mod, build_payload(data_to_save, params_base))
        status[mod] = {"status": "已完成", "time": res.get("timestamp"), "fresh": True}
    return status
