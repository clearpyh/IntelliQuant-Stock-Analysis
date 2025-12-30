from pathlib import Path
import pandas as pd
from functools import reduce
from storage.repository import load_module_result, save_module_result, is_stale, build_payload
from registry.analysis_registry import REGISTRY
from core.data_loader import load_ohlcv

def build_pivot_close(session_state, root: Path, industry: str, sd_local, ed_local, frequency: str) -> pd.DataFrame:
    df_map = session_state.get("stock_map")
    if df_map is None or df_map.empty or industry not in df_map["industry"].values:
        return pd.DataFrame()
    df_ind = df_map[df_map["industry"] == industry]
    symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
    all_dfs = []
    for sym in symbols_in_industry:
        try:
            sym_df = load_ohlcv(session_state, root, sym, industry, str(sd_local), str(ed_local), frequency)
            if not sym_df.empty:
                sym_df = sym_df[['date', 'close']].rename(columns={'close': sym})
                sym_df['date'] = pd.to_datetime(sym_df['date'], errors="coerce")
                sym_df = sym_df.dropna(subset=['date'])
                all_dfs.append(sym_df)
        except Exception:
            continue
    if not all_dfs:
        return pd.DataFrame()
    pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
    pivot_close = pivot_close.set_index('date').sort_index()
    return pivot_close

def run_selected_modules(session_state, root: Path, symbol: str, industry: str, df_symbol: pd.DataFrame, date_col: str, sd_local, ed_local, frequency: str, ma_short: int, ma_long: int, selected: list):
    status = {}
    params_base = {"fetch_key": session_state.get("last_fetch_key"), "ma_short": ma_short, "ma_long": ma_long, "frequency": frequency, "start_date": str(sd_local), "end_date": str(ed_local), "industry": industry}
    pivot_close = None
    if any(m in selected for m in ("相关性分析","PCA分析","风险-收益聚类分析","涨跌概率分析")):
        pivot_close = build_pivot_close(session_state, root, industry, sd_local, ed_local, frequency)
    df_map = session_state.get("stock_map")
    symbols_in_industry = []
    if df_map is not None and industry in df_map["industry"].values:
        df_ind = df_map[df_map["industry"] == industry]
        symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
    feature_source = {}
    if "涨跌概率分析" in selected and symbols_in_industry:
        for s in symbols_in_industry:
            try:
                sym_df = load_ohlcv(session_state, root, s, industry, str(sd_local), str(ed_local), frequency)
                feature_source[s] = sym_df
            except Exception:
                continue
    for mod in selected:
        cached = load_module_result(root, symbol, mod)
        if cached and not is_stale(cached, params_base):
            status[mod] = {"status": "已完成", "time": cached.get("timestamp"), "fresh": True}
            continue
        runner = REGISTRY.get(mod)
        if not runner:
            status[mod] = {"status": "失败", "time": None, "fresh": None}
            continue
        if mod == "K线与指标":
            inputs = {"df_symbol": df_symbol, "date_col": date_col, "ma_short": ma_short, "ma_long": ma_long}
            res = runner(df_symbol, date_col, ma_short, ma_long)
        elif mod in ("相关性分析","PCA分析","风险-收益聚类分析"):
            inputs = {"pivot_close": pivot_close if pivot_close is not None else pd.DataFrame()}
            res = runner(inputs["pivot_close"])
        elif mod == "季节性分析":
            inputs = {"df_symbol": df_symbol, "date_col": date_col}
            res = runner(df_symbol, date_col)
        elif mod == "波动性分析":
            inputs = {"df_symbol": df_symbol, "window": 20}
            res = runner(df_symbol, 20)
        elif mod == "涨跌概率分析":
            inputs = {"pivot_close": pivot_close if pivot_close is not None else pd.DataFrame(), "industry_symbols": symbols_in_industry, "feature_source": feature_source}
            res = runner(inputs["pivot_close"], symbols_in_industry, feature_source)
        elif mod == "基本面因子暴露分析":
            funda_dir_default = str((root / "data" / "fundamentals").resolve())
            fundamentals_dir = Path(session_state.get("funda_dir", funda_dir_default))
            res = runner(symbol, fundamentals_dir)
        else:
            res = {"timestamp": None, "data": {}}
        data_to_save = res.get("data", {})
        data_to_save["timestamp"] = res.get("timestamp")
        save_module_result(root, symbol, mod, build_payload(data_to_save, params_base))
        status[mod] = {"status": "已完成", "time": res.get("timestamp"), "fresh": True}
    return status
