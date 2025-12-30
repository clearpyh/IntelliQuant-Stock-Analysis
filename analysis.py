import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
from src.analysis.stats import compute_corr_matrix, compute_pca, compute_kmeans, compute_factor_regression, compute_logistic_proba
from src.analysis.timeseries import compute_indicators, compute_adx, compute_volatility, compute_garch, compute_stl, compute_acf_pacf
from src.visualization import plot_corr_heatmap, plot_pca_explained, plot_stl_components, plot_acf_pacf, plot_cluster_scatter, plot_regression_coeffs
from src.report import summarize_fundamentals
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

def compute_single_indicators(df_symbol: pd.DataFrame, date_col: str, ma_short: int, ma_long: int) -> pd.DataFrame:
    return compute_indicators(df_symbol.set_index(date_col), ma_short=ma_short, ma_long=ma_long)

def run_background(session_state, modules: list, df_symbol: pd.DataFrame, date_col: str, industry: str, pivot_close: pd.DataFrame, root: Path):
    def _set_status(mod, status, data=None, error=None):
        session_state.setdefault("module_cache", {})
        session_state["module_cache"][mod] = {"status": status, "data": data, "error": error}
    def _bg_volatility():
        try:
            ser_vol = compute_volatility(df_symbol["close"], window=20)
            fig_v = px.line(x=ser_vol.index, y=ser_vol.values, labels={"x": date_col, "y": "HV(20)"})
            res, forecast = compute_garch(df_symbol["close"])
            sigma2_bg = None
            try:
                sigma2_bg = forecast.variance.values[-1][-1]
            except Exception:
                sigma2_bg = None
            _set_status("波动性分析", "ready", {"vol": ser_vol, "fig_vol": fig_v, "sigma2": sigma2_bg})
        except Exception as e:
            _set_status("波动性分析", "failed", error=str(e))
    def _bg_seasonality():
        try:
            stl_bg = compute_stl(df_symbol.set_index(date_col)["close"], period=7)
            fig_stl_bg = plot_stl_components(stl_bg.trend, stl_bg.seasonal, stl_bg.resid)
            a_vals_bg, p_vals_bg = compute_acf_pacf(df_symbol["close"].pct_change(fill_method=None).dropna())
            fig_ap_bg = plot_acf_pacf(a_vals_bg, p_vals_bg)
            _set_status("季节性分析", "ready", {"stl_res": stl_bg, "fig_stl": fig_stl_bg, "fig_ap": fig_ap_bg})
        except Exception as e:
            _set_status("季节性分析", "failed", error=str(e))
    def _bg_corr_pca_cluster_factor_prob():
        pc = pivot_close
        if pc is None or pc.empty or pc.shape[1] < 2:
            _set_status("相关性分析", "failed", error="行业矩阵不可用")
            _set_status("PCA分析", "failed", error="行业矩阵不可用")
            _set_status("风险-收益聚类分析", "failed", error="行业矩阵不可用")
            _set_status("基本面因子暴露分析", "failed", error="行业矩阵不可用")
            _set_status("涨跌概率分析", "failed", error="行业矩阵不可用")
            return
        try:
            corr_bg = compute_corr_matrix(pc)
            fig_corr_bg = plot_corr_heatmap(corr_bg)
            _set_status("相关性分析", "ready", {"corr": corr_bg, "fig_corr": fig_corr_bg})
        except Exception as e:
            _set_status("相关性分析", "failed", error=str(e))
        try:
            X_bg = pc.dropna().pct_change(fill_method=None).dropna()
            if not X_bg.empty:
                _, explained_bg = compute_pca(X_bg)
                fig_pca_bg = plot_pca_explained(explained_bg)
                _set_status("PCA分析", "ready", {"explained": explained_bg, "fig_pca": fig_pca_bg})
        except Exception as e:
            _set_status("PCA分析", "failed", error=str(e))
        try:
            rets_bg = pc.pct_change(fill_method=None).dropna()
            if not rets_bg.empty:
                feat_bg = pd.DataFrame({"ret": rets_bg.mean()*252, "vol": rets_bg.std()*np.sqrt(252)}).dropna()
                if feat_bg.shape[0] >= 3:
                    n_clusters_bg = min(3, max(2, feat_bg.shape[0]//2))
                    _, labels_bg = compute_kmeans(feat_bg, n_clusters=n_clusters_bg)
                    fig_cluster_bg = plot_cluster_scatter(feat_bg, labels_bg)
                    _set_status("风险-收益聚类分析", "ready", {"n_clusters": n_clusters_bg, "labels": labels_bg, "fig_cluster": fig_cluster_bg})
        except Exception as e:
            _set_status("风险-收益聚类分析", "failed", error=str(e))
        try:
            df_ind_map = session_state.get("stock_map")
            if df_ind_map is not None and not df_ind_map.empty:
                df_ind_map = df_ind_map[df_ind_map["industry"] == industry]
                syms_bg = df_ind_map["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind_map.columns else []
                rows_bg = []
                funda_dir_default = str((root / "data" / "fundamentals").resolve())
                for s in syms_bg:
                    m = summarize_fundamentals(Path(session_state.get("funda_dir", funda_dir_default)) / s.replace('.', '_'))
                    if m:
                        rows_bg.append({"symbol": s, **m})
                if rows_bg:
                    F_bg = pd.DataFrame(rows_bg).set_index("symbol")
                    R_bg = pc.pct_change(fill_method=None).dropna().mean()*252
                    R_bg = R_bg.loc[F_bg.index.intersection(R_bg.index)]
                    X_bg = F_bg.loc[R_bg.index]
                    if not X_bg.empty and not R_bg.empty:
                        _, coefs_bg, r2_bg = compute_factor_regression(X_bg, R_bg)
                        fig_coef_bg = plot_regression_coeffs(coefs_bg)
                        _set_status("基本面因子暴露分析", "ready", {"coefs": coefs_bg, "r2": r2_bg, "fig_coef": fig_coef_bg})
        except Exception as e:
            _set_status("基本面因子暴露分析", "failed", error=str(e))
        try:
            R_bg = pc.pct_change(fill_method=None).dropna().mean()*252
            thr_bg = float(R_bg.median())
            y_cls_bg = (R_bg > thr_bg).astype(int)
            df_ind_map = session_state.get("stock_map")
            if df_ind_map is not None and not df_ind_map.empty:
                df_ind_map = df_ind_map[df_ind_map["industry"] == industry]
                syms_bg = df_ind_map["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind_map.columns else []
                X_rows_bg = []
                for s in syms_bg:
                    pattern = f"{s.replace('.','_')}.csv"
                    found = list((root / "data" / "ohlcv").rglob(pattern))
                    if not found:
                        continue
                    fp = found[0]
                    try:
                        df_i = pd.read_csv(fp)
                    except Exception:
                        df_i = pd.read_csv(fp, encoding="gbk")
                    df_i = df_i.sort_values(date_col)
                    slope_sma_bg = float(df_i["close"].rolling(20).mean().diff().dropna().iloc[-1]) if "close" in df_i.columns else None
                    slope_ema_bg = float(df_i["close"].ewm(span=60, adjust=False).mean().diff().dropna().iloc[-1]) if "close" in df_i.columns else None
                    X_rows_bg.append({"symbol": s, "sma_slope": slope_sma_bg, "ema_slope": slope_ema_bg})
                X_bg2 = pd.DataFrame(X_rows_bg).set_index("symbol")
                X_bg2 = X_bg2.loc[y_cls_bg.index.intersection(X_bg2.index)]
                if not X_bg2.empty:
                    _, proba_bg, auc_bg = compute_logistic_proba(X_bg2, y_cls_bg.loc[X_bg2.index])
                    fig_prob_bg = plot_probability_hist(proba_bg=None) if False else None
                    _set_status("涨跌概率分析", "ready", {"proba": proba_bg, "auc": auc_bg, "fig_prob": None})
        except Exception as e:
            _set_status("涨跌概率分析", "failed", error=str(e))
    session_state.setdefault("pivot_close", None)
    if pivot_close is not None and not pivot_close.empty:
        session_state["pivot_close"] = pivot_close
    with ThreadPoolExecutor(max_workers=4) as ex:
        for mod in modules:
            if session_state.get("module_sel") == mod:
                continue
            if session_state.setdefault("module_cache", {}).get(mod, {}).get("status") in ("ready", "pending"):
                continue
            _set_status(mod, "pending")
            if mod in ("波动性分析", "季节性分析"):
                ex.submit(_bg_volatility if mod == "波动性分析" else _bg_seasonality)
            else:
                ex.submit(_bg_corr_pca_cluster_factor_prob)

@dataclass
class STLResultLite:
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series

def compute_module_result(session_state, module: str, df_symbol: pd.DataFrame, date_col: str, sd_local, ed_local, frequency: str, industry: str, root: Path) -> Dict[str, Any]:
    symbol = str(df_symbol["symbol"].dropna().iloc[0]) if "symbol" in df_symbol.columns and not df_symbol.empty else ""
    now_ts = datetime.now().isoformat()
    params = {
        "symbol": symbol,
        "industry": industry,
        "start_date": str(sd_local),
        "end_date": str(ed_local),
        "frequency": frequency
    }
    if module == "K线与指标":
        ind = compute_indicators(df_symbol.set_index(date_col), ma_short=20, ma_long=60)
        adx_series = compute_adx(df_symbol)
        adx_index = df_symbol[date_col].astype(str).tolist()
        return {
            "timestamp": now_ts,
            "params": params,
            "data": {
                "ind": {
                    "index": ind.index.astype(str).tolist(),
                    "columns": list(ind.columns),
                    "data": ind.astype(float).values.tolist()
                },
                "adx": {
                    "index": adx_index,
                    "values": adx_series.astype(float).tolist()
                }
            }
        }
    if module == "波动性分析":
        vol = compute_volatility(df_symbol["close"], window=20)
        sigma2 = None
        try:
            res, forecast = compute_garch(df_symbol["close"])
            sigma2 = forecast.variance.values[-1][-1] if forecast is not None else None
        except Exception:
            sigma2 = None
        return {"timestamp": now_ts, "params": params, "data": {"vol_index": vol.index.astype(str).tolist(), "vol_values": vol.tolist(), "sigma2": sigma2}}
    if module == "季节性分析":
        stl_res = compute_stl(df_symbol.set_index(date_col)["close"], period=7)
        a_vals, p_vals = None, None
        try:
            a_vals, p_vals = compute_acf_pacf(df_symbol["close"].pct_change(fill_method=None).dropna())
        except Exception:
            a_vals, p_vals = None, None
        return {"timestamp": now_ts, "params": params, "data": {
            "trend_index": stl_res.trend.index.astype(str).tolist(),
            "trend_values": stl_res.trend.tolist(),
            "seasonal_index": stl_res.seasonal.index.astype(str).tolist(),
            "seasonal_values": stl_res.seasonal.tolist(),
            "resid_index": stl_res.resid.index.astype(str).tolist(),
            "resid_values": stl_res.resid.tolist(),
            "acf": a_vals.tolist() if a_vals is not None else None,
            "pacf": p_vals.tolist() if p_vals is not None else None
        }}
    if module in ("相关性分析", "PCA分析", "风险-收益聚类分析", "涨跌概率分析", "基本面因子暴露分析"):
        df_map = session_state.get("stock_map")
        if df_map is None or df_map.empty or industry not in df_map["industry"].values:
            return {"timestamp": now_ts, "params": params, "data": {}}
        df_ind = df_map[df_map["industry"] == industry]
        symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
        all_dfs = []
        for sym in symbols_in_industry:
            try:
                sym_df = session_state.get("resolved_df") if session_state.get("symbol_left") == sym else None
                if sym_df is None or sym_df.empty:
                    from data import fetch_resolved_df
                    sym_df = fetch_resolved_df(session_state, sym, industry, str(sd_local), str(ed_local), frequency)
                if not sym_df.empty:
                    sym_df = sym_df[['date', 'close']].rename(columns={'close': sym})
                    sym_df['date'] = pd.to_datetime(sym_df['date'])
                    all_dfs.append(sym_df)
            except Exception:
                continue
        if not all_dfs:
            return {"timestamp": now_ts, "params": params, "data": {}}
        from functools import reduce
        pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
        pivot_close = pivot_close.set_index('date').sort_index()
        if module == "相关性分析":
            corr = compute_corr_matrix(pivot_close)
            return {"timestamp": now_ts, "params": params, "data": {"corr": corr.to_dict()}}
        if module == "PCA分析":
            X = pivot_close.pct_change(fill_method=None).dropna()
            if X.empty:
                return {"timestamp": now_ts, "params": params, "data": {}}
            _, explained = compute_pca(X)
            return {"timestamp": now_ts, "params": params, "data": {"explained": list(explained)}}
        if module == "风险-收益聚类分析":
            rets = pivot_close.pct_change(fill_method=None).dropna()
            if rets.empty or rets.shape[1] < 3:
                return {"timestamp": now_ts, "params": params, "data": {}}
            feat = pd.DataFrame({"ret": rets.mean()*252, "vol": rets.std()*np.sqrt(252)}).dropna()
            if feat.shape[0] < 3:
                return {"timestamp": now_ts, "params": params, "data": {}}
            n_clusters = min(3, max(2, feat.shape[0]//2))
            _, labels = compute_kmeans(feat, n_clusters=n_clusters)
            return {"timestamp": now_ts, "params": params, "data": {"labels": labels.tolist(), "index": list(feat.index), "n_clusters": int(n_clusters), "ret": feat["ret"].tolist(), "vol": feat["vol"].tolist()}}
        if module == "涨跌概率分析":
            R = pivot_close.pct_change(fill_method=None).dropna().mean()*252
            thr = float(R.median())
            y_cls = (R > thr).astype(int)
            X_rows = []
            for s in symbols_in_industry:
                try:
                    from data import fetch_resolved_df
                    sym_df = fetch_resolved_df(session_state, s, industry, str(sd_local), str(ed_local), frequency)
                    if not sym_df.empty:
                        sym_df['date'] = pd.to_datetime(sym_df['date'])
                        sym_df = sym_df.sort_values('date')
                        slope_sma = float(sym_df["close"].rolling(20).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
                        slope_ema = float(sym_df["close"].ewm(span=60, adjust=False).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
                        X_rows.append({"symbol": s, "sma_slope": slope_sma, "ema_slope": slope_ema})
                except Exception:
                    continue
            X = pd.DataFrame(X_rows).set_index("symbol")
            X = X.loc[y_cls.index.intersection(X.index)]
            if X.empty or X.shape[0] < 5:
                return {"timestamp": now_ts, "params": params, "data": {}}
            _, proba, auc = compute_logistic_proba(X, y_cls.loc[X.index])
            return {"timestamp": now_ts, "params": params, "data": {"proba": list(proba), "auc": float(auc)}}
        if module == "基本面因子暴露分析":
            funda_dir_default = str((root / "data" / "fundamentals").resolve())
            sym_dir = Path(session_state.get("funda_dir", funda_dir_default)) / symbol.replace('.', '_')
            fi_fp = sym_dir / "fina_indicator.csv"
            portrait_rows = []
            if fi_fp.exists():
                fi = pd.read_csv(fi_fp)
                fi = fi.sort_values(["end_date","ann_date"]) if "end_date" in fi.columns and "ann_date" in fi.columns else fi
                metrics_cols = [c for c in ["roe","roa","grossprofit_margin","debt_to_assets","oper_cash_flow","pe","pb"] if c in fi.columns]
                for m in metrics_cols:
                    ser = pd.to_numeric(fi[m], errors="coerce").dropna()
                    val = float(ser.iloc[-1]) if not ser.empty else None
                    trend = float((ser.diff().dropna().iloc[-1])) if ser.shape[0] >= 2 else 0.0
                    portrait_rows.append({"metric": m, "value": val, "trend": trend, "percentile": 0.5})
            return {"timestamp": now_ts, "params": params, "data": {"portrait": portrait_rows}}
    return {"timestamp": now_ts, "params": params, "data": {}}
