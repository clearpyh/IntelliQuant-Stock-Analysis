import os
import json
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from src.visualization import plot_candlestick_with_indicators, plot_corr_heatmap, plot_pca_explained, plot_regression_coeffs, plot_probability_hist, plot_stl_components, plot_acf_pacf, plot_cluster_scatter, plot_factor_portrait, plot_returns_scatter, plot_rolling_corr
from src.analysis.stats import compute_corr_matrix, compute_pca, compute_kmeans, compute_factor_regression, compute_logistic_proba, generate_security_tags
from src.analysis.timeseries import compute_indicators, compute_adx, compute_volatility, compute_garch, compute_stl, compute_acf_pacf, compute_max_drawdown, compute_ann_return, compute_sharpe
from src.conclusion import build_facts, generate_conclusions_with_llm
from src.conclusion import generate_conclusion_text
from src.conclusion import generate_module_advice_text
from src.explanation import (
    explain_trend_adx, explain_volatility_risk, explain_correlation_risk,
    explain_pca_structure, explain_seasonality, explain_prediction_probability,
    explain_factor_regression, explain_clustering, explain_factor_portrait
)
from src.config import load_local_env
from src.report import summarize_fundamentals
from src.report import render_industry_report
from src.report import render_single_report
from src.mapping import load_stock_map, resolve_security, process_stock_map
from src.data_io import ts_to_baostock, fetch_kline_baostock
from src.fundamentals import compute_industry_scoring, compute_symbol_metrics, generate_text_report
from src.data_io import export_financials_single, validate_tushare_token
from advisor import get_or_generate_advisor, generate_followup_reply
from ui import analysis_card, advisor_text, followup, nav_pills
from storage.repository import load_module_result, save_module_result, is_stale, build_payload
from registry.dispatcher import run_selected_modules
from data import find_local_ohlcv, fetch_resolved_df

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EXPORT_DIR = ROOT / "export"

load_local_env()

st.set_page_config(page_title="智能证券分析系统", layout="wide")

top_cols = st.columns(5)
top_cols[0].title("智能证券分析系统")
try:
    from src.data_io import fetch_index_quote_bs, fetch_index_latest_tushare
    @st.cache_data(ttl=3)
    def _load_index_overview(sd: str, ed: str, freq: str):
        index_map = [
            ("000001.SH","上证指数"),
            ("399001.SZ","深证成指"),
            ("000300.SH","沪深300"),
            ("000688.SH","科创50")
        ]
        items = []
        for code, name in index_map:
            q = fetch_index_quote_bs(code)
            if not q or q.get("price") is None:
                try:
                    bs_code = ts_to_baostock(code)
                    df_idx = fetch_kline_baostock(bs_code, sd, ed, freq)
                    if not df_idx.empty:
                        df_idx = df_idx.sort_values("date")
                        last = float(df_idx["close"].iloc[-1])
                        prev = float(df_idx["close"].iloc[-2]) if len(df_idx) > 1 else last
                        change = last - prev
                        pct = (change/prev*100.0) if prev != 0 else 0.0
                        ts = pd.to_datetime(df_idx["date"]).iloc[-1]
                        items.append({"index_code": code, "index_name": name, "price": last, "change": change, "pct_change": pct, "timestamp": ts})
                    else:
                        tq = fetch_index_latest_tushare(code, sd, ed)
                        if tq and tq.get("price") is not None:
                            items.append({"index_code": code, "index_name": name, "price": tq.get("price"), "change": tq.get("change"), "pct_change": tq.get("pct_change"), "timestamp": tq.get("timestamp")})
                        else:
                            items.append({"index_code": code, "index_name": name, "price": None, "change": None, "pct_change": None, "timestamp": None})
                except Exception:
                    tq = fetch_index_latest_tushare(code, sd, ed)
                    if tq and tq.get("price") is not None:
                        items.append({"index_code": code, "index_name": name, "price": tq.get("price"), "change": tq.get("change"), "pct_change": tq.get("pct_change"), "timestamp": tq.get("timestamp")})
                    else:
                        items.append({"index_code": code, "index_name": name, "price": None, "change": None, "pct_change": None, "timestamp": None})
            else:
                items.append({"index_code": code, "index_name": name, "price": q.get("price"), "change": q.get("change"), "pct_change": q.get("pct_change"), "timestamp": q.get("timestamp")})
        return pd.DataFrame(items)
    enable_auto = st.checkbox("顶部指数自动刷新", value=True, key="idx_auto_refresh")
    if enable_auto:
        try:
            st.experimental_autorefresh(interval=3000, key="idx_autorefresh_key")
        except Exception:
            pass
    _today = datetime.now().date()
    _sd = str(_today - timedelta(days=10))
    _ed = str(_today)
    df_idx = _load_index_overview(_sd, _ed, "d")
    for i, (_, row) in enumerate(df_idx.iterrows()):
        if i >= 4: break
        label = row["index_name"]
        val = f"{row['price']:.2f}" if pd.notna(row["price"]) else "N/A"
        delta = f"{row['pct_change']:+.2f}%" if pd.notna(row["pct_change"]) else "N/A"
        top_cols[i+1].metric(label, val, delta=delta)
    if pd.notna(df_idx["timestamp"]).any():
        try:
            ts_show = max([t for t in df_idx["timestamp"].tolist() if t is not None])
            st.caption(f"更新时间: {ts_show}")
        except Exception:
            pass
except Exception:
    pass

# 主功能选择 - 在标题下方
main_function = st.radio(
    "选择主功能模块",
    options=["单个证券分析", "证券多因子量化评分"],
    index=0,
    horizontal=True,
    key="main_function"
)

def render_analysis_card(analysis):
    analysis_card(analysis)

def render_conclusion_item(item):
    if not item: return
    icon = "💡"
    conf = item.get("confidence", 0.5)
    color = "blue"
    if conf >= 0.7: color = "green"
    elif conf <= 0.4: color = "red"
    
    with st.expander(f"{icon} {item.get('title','未命名结论')} (置信度: {conf:.2f})", expanded=True):
        st.markdown(f"**摘要**: {item.get('summary','')}")
        if item.get("advice"):
            st.info(f"建议: {item.get('advice')}")
        metrics = item.get("metrics", {})
        if metrics:
            st.json(metrics, expanded=False)
            
def render_advisor_text(txt):
    advisor_text(txt)

def render_followup(module: str, inputs: dict, advisor_text_val: str, facts: dict):
    followup(module, inputs, facts)

def get_or_generate_advisor_wrapper(module: str, inputs: dict, facts: dict, time_label: str, symbol: str, industry: str) -> str:
    inputs = dict(inputs or {})
    inputs["time_range"] = time_label
    return get_or_generate_advisor(st.session_state, module, inputs, facts)

with st.sidebar:
    st.header("1. 行业映射与配置")
    stock_file = st.file_uploader("上传行业证券映射CSV", type=["csv"], help="文件需包含 symbol, industry 列")
    
    if stock_file is not None:
        try:
            stocks_preview = pd.read_csv(stock_file)
        except Exception:
            try:
                if hasattr(stock_file, "seek"): stock_file.seek(0)
                stocks_preview = pd.read_csv(stock_file, encoding="gbk")
            except Exception:
                stocks_preview = pd.DataFrame()
        st.session_state["stock_map"] = process_stock_map(stocks_preview)
        st.success(f"已加载映射: {len(stocks_preview)} 条")
    
    if "stock_map" not in st.session_state:
        default_map = (ROOT / "stock_industry.csv")
        if default_map.exists():
            st.session_state["stock_map"] = load_stock_map(default_map)
            st.success(f"已自动加载默认映射: {default_map.name}")
            
    today = datetime.now().date()
    default_start = today - timedelta(days=365)
    start_date = str(default_start)
    end_date = str(today)
    frequency = "d"
    date_col = "date"
    ma_short = 20
    ma_long = 60
    
    st.divider()
    st.header("2. 数据源选项")
    
    with st.expander("本地文件选项"):
        ohlcv_files = st.file_uploader("上传行情CSV (批量)", type=["csv"], accept_multiple_files=True)
        scan_btn = st.button("扫描 data/ohlcv 目录")
        
        if ohlcv_files:
            items = []
            for f in ohlcv_files:
                items.append((f.name, f))
            if items:
                labels = [it[0] for it in items]
                sel = st.selectbox("选择上传CSV中用于单只分析的文件", options=labels)
                st.session_state["upload_primary_path"] = items[labels.index(sel)][1]
        
        if scan_btn:
            folder = (ROOT / "data" / "ohlcv")
            files = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.exists() else []
            items = []
            for fp in files:
                items.append((fp.name, str(fp)))
            if items:
                labels = [it[0] for it in items]
                values = [it[1] for it in items]
                sel = st.selectbox("选择本地CSV", options=labels)
                st.session_state["local_csv_path"] = values[labels.index(sel)]
    
    st.divider()
    st.header("3. 批量处理与导出")
    
    with st.expander("批量导出选项"):
        industries_all = []
        if "stock_map" in st.session_state:
            industries_all = sorted(set(st.session_state["stock_map"]["industry"].dropna().tolist()))
        
        if industries_all:
            industries_sel = st.multiselect("选择批量导出行业", options=industries_all)
            export_dir = st.text_input("导出目录 (OHLCV)", value=str((ROOT / "data" / "ohlcv").resolve()))
            export_start_date = st.date_input("导出开始日期", value=default_start, key="export_start_date")
            export_end_date = st.date_input("导出结束日期", value=today, key="export_end_date")
            custom_inds = st.text_input("自定义行业名称（逗号分隔）", value="")
            do_export = st.button("按行业批量导出行情CSV")
            if do_export and industries_sel:
                from src.data_io import export_batch
                inds_final = list(industries_sel)
                if custom_inds.strip():
                    inds_final.extend([s.strip() for s in custom_inds.split(",") if s.strip()])
                sd_exp = str(export_start_date)
                ed_exp = str(export_end_date)
                export_batch(st.session_state["stock_map"], inds_final, sd_exp, ed_exp, Path(export_dir))
                st.success(f"批量导出完成：{', '.join(inds_final)}")

        funda_dir = st.text_input("财报目录", value=str((ROOT / "data" / "fundamentals").resolve()))
        do_funda = st.button("批量采集财报(四表)")
        if do_funda and "stock_map" in st.session_state:
            from src.data_io import export_financials_single, validate_tushare_token
            st.session_state["funda_dir"] = funda_dir
            ok = validate_tushare_token()
            if not ok:
                st.error("Tushare Token 无效或权限不足")
            else:
                try:
                    df_map = st.session_state.get("stock_map")
                    industry = st.session_state.get("selected_industry", industries_all[0] if industries_all else "")
                    if df_map is not None and industry:
                        df_ind = df_map[df_map["industry"] == industry]
                        symbols = df_ind["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind.columns else []
                        if not symbols:
                            st.warning("该行业下没有证券数据")
                        else:
                            prog = st.progress(0.0)
                            stat = st.empty()
                            succ = 0; fail = 0
                            total = len(symbols)
                            sd_fmt = start_date.replace("-",""); ed_fmt = end_date.replace("-","")
                            for i, ts_code in enumerate(symbols, start=1):
                                try:
                                    stat.markdown(f"采集中: {ts_code} ({i}/{total})")
                                    export_financials_single(ts_code, sd_fmt, ed_fmt, Path(funda_dir), industry)
                                    succ += 1
                                except Exception as e:
                                    fail += 1
                                    stat.markdown(f"采集失败: {ts_code}，原因: {e}")
                                prog.progress(i/total)
                            stat.markdown(f"完成: 成功 {succ} / 失败 {fail} / 总计 {total}")
                            if fail == 0:
                                st.success("财报采集完成")
                            else:
                                st.warning("部分证券采集中出现失败，请检查Token权限或网络状态")
                    else:
                        st.error("请先选择行业")
                except Exception as e:
                    st.error(f"财报采集失败: {e}")
        
        do_daily_basic = st.button("采集日频估值(daily_basic)")
        if do_daily_basic and "stock_map" in st.session_state:
            from src.data_io import export_daily_basic_batch, validate_tushare_token
            st.session_state["funda_dir"] = funda_dir
            ok = validate_tushare_token()
            if not ok:
                st.error("Tushare Token 无效或权限不足")
            else:
                try:
                    df_map = st.session_state.get("stock_map")
                    industry = st.session_state.get("selected_industry", industries_all[0] if industries_all else "")
                    if df_map is not None and industry:
                        export_daily_basic_batch(df_map, industry, start_date.replace("-",""), end_date.replace("-",""), Path(funda_dir))
                        st.success("daily_basic 采集完成")
                    else:
                        st.error("请先选择行业")
                except Exception as e:
                    st.error(f"daily_basic 采集失败: {e}")
        
        do_validate_pepb = st.button("验证估值数据(pe/pb)")
        if do_validate_pepb and "stock_map" in st.session_state:
            industry = st.session_state.get("ind_sel_right", industries_all[0] if industries_all else "")
            if industry:
                df_map = st.session_state.get("stock_map")
                df_ind = df_map[df_map["industry"] == industry] if df_map is not None else pd.DataFrame()
                syms = df_ind["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind.columns else []
                rows = []
                for sym in syms:
                    try:
                        fp = Path(funda_dir) / sym.replace(".","_") / "fina_indicator.csv"
                        fp_db = Path(funda_dir) / sym.replace(".","_") / "daily_basic.csv"
                        pe_col = None; pb_col = None; pe_val = None; pb_val = None; src = "缺失"
                        if fp_db.exists():
                            try:
                                db = pd.read_csv(fp_db)
                            except Exception:
                                db = pd.read_csv(fp_db, encoding="gbk")
                            db = db.sort_values(["trade_date"]) if "trade_date" in db.columns else db
                            pe_cols_db = [c for c in ["pe","pe_ttm"] if c in db.columns]
                            pb_cols_db = [c for c in ["pb","pb_mrq"] if c in db.columns]
                            if pe_cols_db:
                                pe_col = pe_cols_db[0]
                                ser_pe = pd.to_numeric(db[pe_col], errors="coerce").dropna()
                                pe_val = float(ser_pe.iloc[-1]) if not ser_pe.empty else None
                            if pb_cols_db:
                                pb_col = pb_cols_db[0]
                                ser_pb = pd.to_numeric(db[pb_col], errors="coerce").dropna()
                                pb_val = float(ser_pb.iloc[-1]) if not ser_pb.empty else None
                            src = "daily_basic"
                        if (pe_val is None or pb_val is None) and fp.exists():
                            try:
                                fi = pd.read_csv(fp)
                            except Exception:
                                fi = pd.read_csv(fp, encoding="gbk")
                            fi = fi.sort_values(["end_date","ann_date"]) if "end_date" in fi.columns and "ann_date" in fi.columns else fi
                            pe_cols = [c for c in ["pe","pe_ttm","pe_basic","pe_circ"] if c in fi.columns]
                            pb_cols = [c for c in ["pb","pb_mrq"] if c in fi.columns]
                            if pe_val is None and pe_cols:
                                pe_col = pe_cols[0]
                                ser_pe = pd.to_numeric(fi[pe_col], errors="coerce").dropna()
                                pe_val = float(ser_pe.iloc[-1]) if not ser_pe.empty else None
                                src = "fina_indicator"
                            if pb_val is None and pb_cols:
                                pb_col = pb_cols[0]
                                ser_pb = pd.to_numeric(fi[pb_col], errors="coerce").dropna()
                                pb_val = float(ser_pb.iloc[-1]) if not ser_pb.empty else None
                                src = "fina_indicator"
                        rows.append({"symbol": sym, "files": "已采集" if src != "缺失" else "缺失", "source": src, "pe_col": pe_col, "pe_val": pe_val, "pb_col": pb_col, "pb_val": pb_val})
                    except Exception:
                        rows.append({"symbol": sym, "files": "异常", "source": None, "pe_col": None, "pe_val": None, "pb_col": None, "pb_val": None})
                df_check = pd.DataFrame(rows).set_index("symbol")
                st.subheader("估值数据验证")
                st.dataframe(df_check)
                n_missing = int((df_check["pe_val"].isna()).sum() + (df_check["pb_val"].isna()).sum())
                if n_missing > 0:
                    st.warning("部分标的估值缺失，建议采集 daily_basic 或在财报接口显式加入 pe_ttm/pb_mrq 字段")
        
        do_report = st.button("行业基本面概览(HTML)")
        if do_report and "stock_map" in st.session_state:
            industry = st.session_state.get("ind_sel_right", industries_all[0] if industries_all else "")
            if industry:
                charts = []
                ind_dir = EXPORT_DIR / industry
                if ind_dir.exists():
                    for fp in ind_dir.glob("*.png"):
                        charts.append(str(fp))
                conclusions_paths = []
                for fp in EXPORT_DIR.glob(f"*_{industry}_conclusions.json"):
                    conclusions_paths.append(str(fp))
                symbols = []
                if "stock_map" in st.session_state:
                    sm = st.session_state["stock_map"]
                    df_ind_map = sm[sm["industry"] == industry]
                    symbols = df_ind_map["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind_map.columns else []
                out_fp = render_industry_report(industry, symbols, Path(funda_dir), EXPORT_DIR, charts, conclusions_paths)
                st.success(f"行业报告已生成: {out_fp}")
    
    st.divider()
    st.header("4. 大模型配置")
    llm_model = st.text_input("LLM模型", value=os.environ.get("LLM_MODEL", "gpt-4"), key="llm_model")
    llm_endpoint = st.text_input("LLM端点", value=os.environ.get("LLM_ENDPOINT", ""), key="llm_endpoint")
    llm_api_key = st.text_input("API密钥", value=os.environ.get("LLM_API_KEY", ""), type="password", key="llm_api_key")
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    if llm_endpoint:
        os.environ["LLM_ENDPOINT"] = llm_endpoint
    if llm_api_key:
        os.environ["LLM_API_KEY"] = llm_api_key
    
    st.divider()
    st.header("5. 显示选项")
    st.checkbox("显示综合结论（10条）", value=st.session_state.get("show_conclusions", False), key="show_conclusions")
    
    st.divider()
    if st.button("清空会话缓存"):
        for k in ["df_source_left", "df_source_right", "resolved_df", "last_fetch_key", "module_cache", "advisor_extras_done", "pivot_close"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("已清空缓存")

# 创建布局：根据模式切换仅显示一个工作区
ratio = [1, 0.0001] if main_function == "单个证券分析" else [0.0001, 1]
left_col, right_col = st.columns(ratio)

# 左侧模块：单个证券分析
with left_col:
    if main_function == "单个证券分析":
        st.header("单个证券分析")
        # 证券查询
        query = st.text_input("输入证券代码或名称", value="", placeholder="例如: 600006 或 晋西车轴", key="query_left")
        
        col1, col2 = st.columns(2)
        with col1:
            sd_local = st.date_input("开始日期", value=datetime.now()-timedelta(days=365), key="sd_left")
        with col2:
            ed_local = st.date_input("结束日期", value=datetime.now(), key="ed_left")
        
        # 执行单个证券分析（自动触发，无需按钮）
        if query:
            df = None
            local_csv_path = st.session_state.get("local_csv_path")
            upload_primary_path = st.session_state.get("upload_primary_path")
            
            if local_csv_path:
                try:
                    df = pd.read_csv(local_csv_path)
                except Exception:
                    df = pd.read_csv(local_csv_path, encoding="gbk")
            elif upload_primary_path is not None:
                try:
                    df = pd.read_csv(upload_primary_path)
                except Exception:
                    if hasattr(upload_primary_path, "seek"):
                        upload_primary_path.seek(0)
                    df = pd.read_csv(upload_primary_path, encoding="gbk")
            else:
                # 先查本地，再用第三方接口
                df_map = st.session_state.get("stock_map")
                resolved = resolve_security(df_map, query) if df_map is not None else None
                symbol_try = None
                industry_try = "未知"
                name_try = ""
                if resolved:
                    symbol_try, industry_try, name_try = resolved
                    st.success(f"已识别: {name_try} ({symbol_try}) - {industry_try}")
                else:
                    from src.data_io import normalize_ts_code
                    symbol_try = normalize_ts_code(query)
                if symbol_try:
                    local_path = find_local_ohlcv(ROOT, symbol_try)
                    if local_path:
                        try:
                            df = pd.read_csv(local_path)
                        except Exception:
                            df = pd.read_csv(local_path, encoding="gbk")
                        st.info(f"使用本地数据: {Path(local_path).name}")
                        try:
                            latest_dt = pd.to_datetime(df[date_col], errors="coerce").max()
                            latest_txt = latest_dt.strftime("%Y-%m-%d") if pd.notna(latest_dt) else "未知"
                        except Exception:
                            latest_txt = "未知"
                        st.caption(f"本地数据最新日期: {latest_txt}")
                        do_refetch = st.button("重新采集最新行情", key=f"refetch_{symbol_try}")
                        if do_refetch:
                            with st.spinner(f"正在重新采集 {symbol_try} 行情数据..."):
                                df_rt = fetch_resolved_df(st.session_state, symbol_try, industry_try, str(sd_local), str(ed_local), frequency)
                                if not df_rt.empty:
                                    df = df_rt
                                    st.success(f"已更新行情数据 ({len(df)} 条)")
                                else:
                                    st.warning("重新采集失败或数据为空，请检查日期或代码")
                    else:
                        with st.spinner(f"正在拉取 {symbol_try} 行情数据..."):
                            df_rt = fetch_resolved_df(st.session_state, symbol_try, industry_try, str(sd_local), str(ed_local), frequency)
                            if not df_rt.empty:
                                df = df_rt
                                st.success(f"行情拉取成功 ({len(df)} 条)")
                            else:
                                st.warning("未获取到行情数据，请检查日期或代码")
                else:
                    st.warning(f"未识别到证券: {query}")
            
            if df is not None and not df.empty:
                st.session_state["df_source_left"] = df
                if resolved:
                    st.session_state["symbol_left"] = symbol_try
                    st.session_state["industry_left"] = industry_try
                    st.session_state["name_left"] = name_try
                else:
                    if "symbol" in df.columns:
                        symbol_extracted = str(df["symbol"].dropna().iloc[0])
                        st.session_state["symbol_left"] = symbol_extracted
                        st.session_state["industry_left"] = "未知"
                        st.session_state["name_left"] = query
            else:
                st.error("无法加载数据")
        
        # 如果有数据，显示分析模块
        if "df_source_left" in st.session_state and st.session_state["df_source_left"] is not None:
            df = st.session_state["df_source_left"]
            symbol = st.session_state.get("symbol_left", "")
            industry = st.session_state.get("industry_left", "")
            name = st.session_state.get("name_left", "")
            
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # 确定symbol
            if not symbol and "symbol" in df.columns:
                symbols_in_df = [str(x) for x in df["symbol"].dropna().unique().tolist()]
                if symbols_in_df:
                    symbol = symbols_in_df[0]
            
            df_symbol = df[df["symbol"] == symbol] if symbol and "symbol" in df.columns else df
            
            if not df_symbol.empty:
                modules = ["K线与指标", "相关性分析", "PCA分析", "波动性分析", "季节性分析", "风险-收益聚类分析", "基本面因子暴露分析", "涨跌概率分析"]
                sel_multi = st.multiselect("选择要执行的分析模块", options=modules, default=["K线与指标"], key="module_batch_sel_left")
                run_selected_btn = st.button("开始分析(仅选中模块)", key="run_selected_left", type="primary")
                if run_selected_btn and sel_multi:
                    st.session_state.setdefault("module_status_left", {})
                    with st.spinner("分析执行中..."):
                        status = run_selected_modules(st.session_state, ROOT, symbol, industry, df_symbol, date_col, sd_local, ed_local, frequency, ma_short, ma_long, sel_multi)
                        for mod, s in status.items():
                            st.session_state["module_status_left"][mod] = s
                    st.success("选中模块分析完成")
                params_now = {"fetch_key": st.session_state.get("last_fetch_key"), "ma_short": ma_short, "ma_long": ma_long, "frequency": frequency}
                st.session_state.setdefault("module_status_left", {})
                for mod in modules:
                    if mod not in st.session_state["module_status_left"]:
                        cached_payload = load_module_result(ROOT, symbol, mod)
                        if not cached_payload:
                            st.session_state["module_status_left"][mod] = {"status": "未分析", "time": None, "fresh": None}
                        else:
                            stale = is_stale(cached_payload, params_now)
                            st.session_state["module_status_left"][mod] = {"status": "需重新分析" if stale else "已完成", "time": cached_payload.get("timestamp"), "fresh": not stale}
                with st.expander("模块状态", expanded=False):
                    for mod in modules:
                        s = st.session_state["module_status_left"].get(mod, {"status": "未分析", "time": None, "fresh": None})
                        lbl = f"{mod} | 状态: {s['status']}"
                        if s["time"]:
                            lbl += f" | 上次分析: {s['time']}"
                        if s["status"] == "需重新分析":
                            lbl += " | 原因: 参数或数据变更"
                        st.caption(lbl)
                module_sel = st.pills("选择分析模块", modules, default="K线与指标", key="module_sel_left_pills")
                
                ind = compute_indicators(df_symbol.set_index(date_col), ma_short=ma_short, ma_long=ma_long)
                if "K线与指标" in st.session_state.get("mod_cache_left", {}):
                    ind = st.session_state["mod_cache_left"]["K线与指标"].get("ind", ind)
                    adx_prefetch = st.session_state["mod_cache_left"]["K线与指标"].get("adx", None)
                
                # 构建事实数据
                facts = build_facts(df_symbol, ind, None, None, symbol, industry)
                
                if st.session_state.get("show_conclusions", False):
                    st.divider()
                    st.subheader("综合结论（10条）")
                    try:
                        conclusions = generate_conclusions_with_llm(facts, [])
                    except Exception:
                        conclusions = []
                    if conclusions:
                        for item in conclusions[:10]:
                            render_conclusion_item(item)
                    else:
                        st.info("综合结论生成失败或数据不足")
                
                # 根据模块显示不同内容（加入模块级缓存，切换不重新计算）
                mod_cache = st.session_state.setdefault("mod_cache_left", {})
                if module_sel == "相关性分析" and "相关性分析" not in mod_cache:
                    payload = load_module_result(ROOT, symbol, "相关性分析")
                    if payload and payload.get("data", {}).get("corr"):
                        corr_df = pd.DataFrame(payload["data"]["corr"])
                        fig_corr = plot_corr_heatmap(corr_df)
                        mod_sum = f"行业相关性分析: 平均相关性: {corr_df.mean().mean():.3f}"
                        advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                        mod_cache["相关性分析"] = {"corr": corr_df, "fig_corr": fig_corr, "advisor_inputs": advisor_inputs}
                if module_sel == "K线与指标" and "K线与指标" not in mod_cache:
                    payload = load_module_result(ROOT, symbol, "K线与指标")
                    d = payload["data"] if payload else {}
                    ind_d = d.get("ind"); adx_d = d.get("adx")
                    if ind_d and adx_d:
                        ind_cached = pd.DataFrame(ind_d["data"], index=pd.to_datetime(ind_d["index"]), columns=ind_d["columns"])
                        try:
                            idx_try = pd.to_datetime(adx_d["index"], errors="coerce")
                            if hasattr(idx_try, "isna") and idx_try.isna().any():
                                tail_idx = pd.to_datetime(df_symbol[date_col]).iloc[-len(adx_d["values"]):]
                                idx = tail_idx
                            else:
                                idx = idx_try
                        except Exception:
                            idx = pd.to_datetime(df_symbol[date_col]).iloc[-len(adx_d["values"]):]
                        n = min(len(adx_d["values"]), len(idx))
                        adx_cached = pd.Series(list(adx_d["values"])[-n:], index=idx[-n:])
                        mod_cache["K线与指标"] = {"ind": ind_cached, "adx": adx_cached}
                if module_sel == "K线与指标":
                    st.subheader("K线与指标")
                    end_dt = pd.to_datetime(df_symbol[date_col]).max()
                    ranges_labels = ["一周", "一个月", "三个月", "六个月", "一年", "三年", "五年", "全部"]
                    ranges_days = {
                        "一周": 7,
                        "一个月": 30,
                        "三个月": 90,
                        "六个月": 180,
                        "一年": 365,
                        "三年": 365*3,
                        "五年": 365*5
                    }
                    
                    sel_lbl = st.selectbox("查看区间", ranges_labels, index=4)
                    if sel_lbl == "全部":
                        df_slice = df_symbol
                        ind_slice = ind
                    else:
                        start_dt = end_dt - timedelta(days=ranges_days[sel_lbl])
                        df_slice = df_symbol[df_symbol[date_col] >= start_dt]
                        if df_slice.empty:
                            df_slice = df_symbol
                        ind_slice = ind.loc[df_slice[date_col].values] if not df_slice.empty else ind
                    
                    st.subheader("三秒快览")
                    try:
                        ma_slope = float(ind_slice["SMA"].diff().dropna().iloc[-1]) if "SMA" in ind_slice.columns else None
                        ema_slope = float(ind_slice["EMA"].diff().dropna().iloc[-1]) if "EMA" in ind_slice.columns else None
                        rsi_last = float(ind_slice["RSI"].iloc[-1]) if "RSI" in ind_slice.columns else None
                    except Exception:
                        ma_slope = None; ema_slope = None; rsi_last = None
                    trend = "多头" if ((ma_slope or 0) > 0) or ((ema_slope or 0) > 0) else ("空头" if ((ma_slope or 0) < 0 and (ema_slope or 0) < 0) else "震荡")
                    rsi_tag = "超买" if (rsi_last is not None and rsi_last >= 70) else ("超卖" if (rsi_last is not None and rsi_last <= 30) else "中性")
                    try:
                        adx_series = adx_prefetch if 'adx_prefetch' in locals() and adx_prefetch is not None else compute_adx(df_symbol)
                        adx_last = float(adx_series.iloc[-1]) if adx_series is not None and not adx_series.empty else None
                    except Exception:
                        adx_last = None
                    adx_tag = "弱" if (adx_last is not None and adx_last < 20) else ("中" if (adx_last is not None and adx_last < 40) else ("强" if (adx_last is not None and adx_last < 60) else ("极强" if adx_last is not None else "N/A")))
                    try:
                        r = df_slice["close"].pct_change(fill_method=None).dropna()
                        vol_ann = float(r.std() * np.sqrt(252)) if not r.empty else None
                    except Exception:
                        vol_ann = None
                    vol_tag = "低" if (vol_ann is not None and vol_ann < 0.20) else ("中" if (vol_ann is not None and vol_ann < 0.35) else ("高" if (vol_ann is not None) else "N/A"))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("趋势", trend)
                    c2.metric("RSI", f"{int(rsi_last) if rsi_last is not None else 'N/A'} ({rsi_tag})")
                    c3.metric("ADX", f"{adx_last:.1f} ({adx_tag})" if adx_last is not None else "N/A")
                    c4.metric("波动率(年化)", f"{vol_ann:.2%} ({vol_tag})" if vol_ann is not None else "N/A")
                    
                    fig_k_current = plot_candlestick_with_indicators(df_slice, date_col=date_col, indicators=ind_slice, time_span=sel_lbl, show_text=False)
                    st.plotly_chart(fig_k_current, width="stretch")
                    
                    try:
                        adx_series = adx_prefetch if 'adx_prefetch' in locals() and adx_prefetch is not None else compute_adx(df_symbol)
                        fig_adx = px.line(x=adx_series.index, y=adx_series.values, labels={"x": date_col, "y": "ADX"})
                        st.plotly_chart(fig_adx, width="stretch")
                        mod_sum = f"证券: {name}({symbol}), 行业: {industry}, 分析区间: {sel_lbl}, 最新收盘价: {df_symbol['close'].iloc[-1]:.2f}"
                        analysis_trend = explain_trend_adx(df_slice, adx_series, ma_short, ma_long)
                        render_analysis_card(analysis_trend)
                        try:
                            vol_series = df_slice["close"].pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
                        except Exception:
                            vol_series = pd.Series(dtype=float)
                        analysis_vol = explain_volatility_risk(vol_series)
                        render_analysis_card(analysis_vol)
                        advisor_k = get_or_generate_advisor_wrapper("K线与指标", 
                            {"time_range": sel_lbl, "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}, 
                            facts, sel_lbl, symbol, industry)
                        render_advisor_text(advisor_k)
                        render_followup("K线与指标", 
                            {"time_range": sel_lbl, "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}, 
                            advisor_k, facts)
                        st.subheader("3分钟信任")
                        trust_cols = st.columns(4)
                        try:
                            from src.data_io import validate_tushare_token
                            token_ok = validate_tushare_token()
                        except Exception:
                            token_ok = False
                        trust_cols[0].metric("数据源", "Baostock/Tushare")
                        trust_cols[1].metric("Token", "有效" if token_ok else "未配置")
                        try:
                            prob_payload = load_module_result(ROOT, symbol, "涨跌概率分析")
                            auc_val = float(prob_payload.get("data", {}).get("auc")) if prob_payload else None
                        except Exception:
                            auc_val = None
                        trust_cols[2].metric("模型AUC", f"{auc_val:.2f}" if auc_val is not None else "N/A")
                        trust_cols[3].metric("区间", sel_lbl)
                    except Exception:
                        st.info("ADX计算失败")
                
                elif module_sel == "相关性分析":
                    st.subheader("相关性分析")
                    view_lbl = st.session_state.get("view_range_label", "一年")
                    rng_days = {"一周": 7, "一个月": 30, "三个月": 90, "六个月": 180, "一年": 365, "三年": 365*3, "五年": 365*5}
                    df_s = df_symbol[[date_col, "close"]].copy()
                    df_s[date_col] = pd.to_datetime(df_s[date_col])
                    df_s = df_s.sort_values(by=date_col)
                    r_raw_s = df_s["close"].divide(df_s["close"].shift(1)) - 1
                    ret_s = pd.Series(r_raw_s.values, index=df_s[date_col]).dropna()
                    bench_codes = {"上证指数": "sh.000001", "沪深300": "sh.000300", "创业板指": "sz.399006"}
                    bench_rets = {}
                    for nm, code in bench_codes.items():
                        try:
                            df_b = fetch_kline_baostock(code, str(sd_local), str(ed_local), frequency)
                            if not df_b.empty:
                                df_b = df_b.sort_values(by="date")
                                r_raw_b = df_b["close"].divide(df_b["close"].shift(1)) - 1
                                r_b = pd.Series(r_raw_b.values, index=df_b["date"]).dropna()
                                bench_rets[nm] = r_b
                        except Exception:
                            continue
                    pc = None
                    try:
                        from src.scheduler import build_pivot_close as _build_pc
                        df_map = st.session_state.get("stock_map")
                        if df_map is None or df_map.empty:
                            st.warning("行业映射未加载，无法构造行业指数")
                        elif industry not in df_map["industry"].values:
                            fallback_ind = st.session_state.get("ind_sel_right")
                            if fallback_ind and fallback_ind in df_map["industry"].values:
                                st.info(f"当前行业不在映射中，使用右侧选择行业: {fallback_ind}")
                                industry = fallback_ind
                            else:
                                st.warning(f"映射中不包含当前行业: {industry}")
                        else:
                            syms = df_map[df_map["industry"] == industry]["symbol"].dropna().astype(str).tolist()
                            st.caption(f"行业成分数: {len(syms)}")
                            pc = _build_pc(st.session_state, industry, sd_local, ed_local, frequency)
                            if pc is not None:
                                st.caption(f"行业透视形状(全部区间): {pc.shape}")
                    except Exception:
                        pc = None
                    if pc is not None and not pc.empty and view_lbl != "全部":
                        start_dt = pd.to_datetime(df_symbol[date_col]).max() - pd.Timedelta(days=rng_days.get(view_lbl, 365))
                        pc_full = pc
                        pc = pc.loc[pc.index >= start_dt]
                        if pc.empty:
                            pc = pc_full
                            st.info("所选区间内行业数据为空，已回退到全部区间")
                        else:
                            st.caption(f"行业透视形状(所选区间): {pc.shape}")
                    ind_ret = None
                    if pc is not None and not pc.empty:
                        X = pc.ffill()
                        X = X.divide(X.shift(1)) - 1
                        ind_ret = X.dropna().mean(axis=1)
                        ind_ret.name = "行业指数"
                    else:
                        st.warning("未能构造行业指数，检查映射与本地数据或网络采集")
                    cross_assets = {}
                    try:
                        import yfinance as yf
                        ca_map = {"黄金": "GC=F", "原油": "CL=F", "USDCNH": "USDCNH=X", "TLT": "TLT"}
                        for nm, tk in ca_map.items():
                            try:
                                df_ca = yf.download(tk, start=str(sd_local), end=str(ed_local), interval="1d", progress=False)
                                if not df_ca.empty and "Close" in df_ca.columns:
                                    df_ca.index = pd.to_datetime(df_ca.index)
                                    r = (df_ca["Close"].divide(df_ca["Close"].shift(1)) - 1).dropna()
                                    cross_assets[nm] = r
                            except Exception:
                                continue
                    except Exception:
                        pass
                    all_series = {"证券": ret_s}
                    for k, v in bench_rets.items():
                        all_series[k] = v
                    if ind_ret is not None:
                        all_series["行业指数"] = ind_ret
                    for k, v in cross_assets.items():
                        all_series[k] = v
                    series_norm = {}
                    for k, v in all_series.items():
                        if isinstance(v, pd.Series):
                            series_norm[k] = pd.Series(np.asarray(v.values).reshape(-1), index=v.index)
                        elif isinstance(v, pd.DataFrame) and v.shape[1] == 1:
                            col0 = v.columns[0]
                            series_norm[k] = pd.Series(np.asarray(v[col0].values).reshape(-1), index=v.index)
                        else:
                            try:
                                vv = pd.Series(v)
                                series_norm[k] = vv
                            except Exception:
                                pass
                    df_ret = pd.DataFrame(series_norm)
                    corr = df_ret.corr()
                    fig_corr = plot_corr_heatmap(corr)
                    st.plotly_chart(fig_corr, width="stretch")
                    def _lvl(r):
                        a = abs(r)
                        return "高度相关" if a >= 0.7 else ("中度相关" if a >= 0.3 else "弱相关")
                    lines = []
                    if "证券" in corr.index and "沪深300" in corr.index:
                        r_sym_hs = float(corr.loc["证券","沪深300"])
                        if np.isfinite(r_sym_hs):
                            dir_txt = "正相关" if r_sym_hs >= 0 else "负相关"
                            lines.append(f"标的证券与沪深300呈{_lvl(r_sym_hs)}（{r_sym_hs:.3f}）{dir_txt}，β敞口{'较高' if abs(r_sym_hs)>=0.4 else '可控'}。")
                        else:
                            lines.append("标的证券与沪深300相关性无法计算，可能由于样本区间内有效数据不足。")
                    if "证券" in corr.index and "创业板指" in corr.index:
                        r_sym_cyb = float(corr.loc["证券","创业板指"])
                        if np.isfinite(r_sym_cyb):
                            dir_txt = "正相关" if r_sym_cyb >= 0 else "负相关"
                            lines.append(f"标的证券与创业板指数相关性 {r_sym_cyb:.3f}（{_lvl(r_sym_cyb)}，{dir_txt}）。")
                        else:
                            lines.append("标的证券与创业板指数相关性无法计算，可能由于样本区间内有效数据不足。")
                    if "证券" in corr.index and "行业指数" in corr.index:
                        r_sym_ind = float(corr.loc["证券","行业指数"])
                        if np.isfinite(r_sym_ind):
                            dir_txt = "正相关" if r_sym_ind >= 0 else "负相关"
                            lines.append(f"标的证券与行业指数相关性 {r_sym_ind:.3f}（{_lvl(r_sym_ind)}，{dir_txt}），反映与同业联动程度。")
                        else:
                            lines.append("行业指数相关性无法计算，可能由于样本区间内有效数据不足。")
                    ca_list = ["黄金","原油","TLT","USDCNH"]
                    low_assets = []
                    per_assets = []
                    for nm in ca_list:
                        if nm in corr.index and "证券" in corr.index:
                            rv = float(corr.loc["证券", nm])
                            if np.isfinite(rv):
                                if abs(rv) < 0.1:
                                    low_assets.append(nm)
                                else:
                                    dir_txt = "正相关" if rv >= 0 else "负相关"
                                    per_assets.append(f"标的证券与{nm}{dir_txt}（{rv:.3f}，{_lvl(rv)}）")
                            else:
                                per_assets.append(f"标的证券与{nm}相关性无法计算，可能由于样本区间内有效数据不足。")
                    if low_assets:
                        lines.append(f"标的证券与{', '.join(low_assets)}相关性均低于 0.1（近似零），与避险/大宗资产关联度不高。")
                    lines.extend(per_assets)
                    hs300 = bench_rets.get("沪深300")
                    anomaly_txt = None
                    if hs300 is not None and not hs300.empty:
                        comb_an = pd.DataFrame({"证券": ret_s, "沪深300": hs300}).dropna()
                        if not comb_an.empty:
                            rc_full = float(comb_an["证券"].corr(comb_an["沪深300"]))
                            rc_roll = float(comb_an["证券"].rolling(window=60).corr(comb_an["沪深300"]).dropna().iloc[-1]) if comb_an.shape[0] >= 60 else rc_full
                            diff = rc_roll - rc_full
                            if abs(diff) >= 0.2 or (np.sign(rc_roll) != np.sign(rc_full)):
                                anomaly_txt = f"标的证券与沪深300近期滚动相关性变化显著（当前 {rc_roll:.2f}，全样本 {rc_full:.2f}），需关注关系变化。"
                    if anomaly_txt:
                        lines.append(anomaly_txt)
                    st.subheader("相关性分析结论")
                    for sline in lines:
                        st.markdown(f"- {sline}")
                    mod_sum = "；".join(lines) if lines else f"综合相关性: 平均相关性: {corr.mean().mean():.3f}"
                    advisor_inputs = {"time_range": view_lbl, "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                    hs300 = bench_rets.get("沪深300")
                    if hs300 is not None and not hs300.empty:
                        comb = pd.DataFrame({"证券": ret_s, "沪深300": hs300}).dropna()
                        fig_scatter = plot_returns_scatter(comb["沪深300"], comb["证券"], "沪深300", "证券")
                        st.plotly_chart(fig_scatter, width="stretch")
                        if comb.shape[0] >= 10:
                            r_scatter = float(comb["证券"].corr(comb["沪深300"]))
                            if np.isfinite(r_scatter):
                                lvl_txt = "高度" if abs(r_scatter) >= 0.7 else ("中度" if abs(r_scatter) >= 0.3 else "弱")
                                dir_txt = "正相关" if r_scatter >= 0 else "负相关"
                                tail_txt = "短期收益变化趋势基本一致" if r_scatter >= 0 else "短期收益变化趋势相反"
                                st.markdown(f"- 标的证券与沪深300收益率散点图显示出{lvl_txt}{dir_txt}（Pearson r={r_scatter:.3f}），{tail_txt}。")
                            else:
                                st.markdown("- 收益率散点图数据不足或缺失。")
                        else:
                            st.markdown("- 收益率散点图数据不足或缺失。")
                        roll = comb["证券"].rolling(window=60).corr(comb["沪深300"])
                        fig_roll = plot_rolling_corr(roll, "60日滚动相关系数")
                        st.plotly_chart(fig_roll, width="stretch")
                        roll_clean = roll.dropna()
                        if not roll_clean.empty:
                            last_r = float(roll_clean.iloc[-1])
                            mean_r = float(roll_clean.mean())
                            std_r = float(roll_clean.std())
                            if np.isfinite(last_r) and np.isfinite(mean_r):
                                overall = "高度正相关" if mean_r >= 0.7 else ("中度正相关" if 0.3 <= mean_r < 0.7 else ("负相关" if mean_r < 0 else "相关性弱"))
                                trend_delta = last_r - mean_r
                                trend_txt = "近期相关性上升" if trend_delta > 0.1 else ("近期相关性下降" if trend_delta < -0.1 else "近期相关性变化不明显")
                                stab_txt = "相关性不稳定，波动较大" if std_r > 0.3 else "相关性较为稳定"
                                st.markdown(f"- 标的证券与沪深30060日滚动相关系数整体呈{overall}（均值={mean_r:.2f}），{trend_txt}，{stab_txt}。")
                            else:
                                st.markdown("- 60日滚动相关系数数据不足或缺失。")
                        else:
                            st.markdown("- 60日滚动相关系数数据不足或缺失。")
                        cov = float(comb["证券"].cov(comb["沪深300"]))
                        var_b = float(comb["沪深300"].var())
                        beta = cov / var_b if var_b != 0 else np.nan
                        st.metric("β系数(相对沪深300)", f"{beta:.2f}" if np.isfinite(beta) else "N/A")
                    render_analysis_card(explain_correlation_risk(corr))
                    advisor_c = get_or_generate_advisor_wrapper("相关性分析", advisor_inputs, facts, view_lbl, symbol, industry)
                    render_advisor_text(advisor_c)
                    render_followup("相关性分析", advisor_inputs, advisor_c, facts)
                    mod_cache["相关性分析"] = {"corr": corr, "fig_corr": fig_corr, "advisor_inputs": advisor_inputs}
                
                elif module_sel == "PCA分析":
                    if "PCA分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "PCA分析")
                        if payload and payload.get("data", {}).get("explained"):
                            explained = pd.Series(payload["data"]["explained"])
                            fig_pca = plot_pca_explained(explained)
                            mod_sum = f"PCA分析: 前3个主成分累计解释方差: {sum(explained[:3]):.2%}"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["PCA分析"] = {"explained": explained, "fig_pca": fig_pca, "advisor_inputs": advisor_inputs}
                    st.subheader("PCA分析")
                    cached = mod_cache.get("PCA分析")
                    if cached:
                        st.plotly_chart(cached["fig_pca"], width="stretch")
                        render_analysis_card(explain_pca_structure(cached["explained"]))
                        advisor_p = get_or_generate_advisor_wrapper("PCA分析", cached["advisor_inputs"], facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_p)
                        render_followup("PCA分析", cached["advisor_inputs"], advisor_p, facts)
                    elif "stock_map" in st.session_state and st.session_state["stock_map"] is not None:
                        df_map = st.session_state["stock_map"]
                        if industry and industry in df_map["industry"].values:
                            df_ind = df_map[df_map["industry"] == industry]
                            symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
                            
                            # 读取行业内其他股票数据
                            all_dfs = []
                            for sym in symbols_in_industry:
                                try:
                                    sym_df = fetch_resolved_df(st.session_state, sym, industry, str(sd_local), str(ed_local), frequency)
                                    if not sym_df.empty:
                                        sym_df = sym_df[['date', 'close']].rename(columns={'close': sym})
                                        sym_df['date'] = pd.to_datetime(sym_df['date'])
                                        all_dfs.append(sym_df)
                                except:
                                    continue
                            
                            if all_dfs:
                                # 合并所有股票数据
                                from functools import reduce
                                pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
                                pivot_close = pivot_close.set_index('date').sort_index()
                                
                                # 计算PCA
                                X = pivot_close.pct_change(fill_method=None).dropna()
                                if not X.empty:
                                    pca_model, explained = compute_pca(X)
                                    fig_pca = plot_pca_explained(explained)
                                    st.plotly_chart(fig_pca, width="stretch")
                                    render_analysis_card(explain_pca_structure(explained))
                                    # 大模型分析
                                    mod_sum = f"PCA分析: 前3个主成分累计解释方差: {sum(explained[:3]):.2%}"
                                    advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                                    advisor_p = get_or_generate_advisor_wrapper("PCA分析", advisor_inputs, facts, "最近一年", symbol, industry)
                                    render_advisor_text(advisor_p)
                                    render_followup("PCA分析", advisor_inputs, advisor_p, facts)
                                    mod_cache["PCA分析"] = {"explained": explained, "fig_pca": fig_pca, "advisor_inputs": advisor_inputs}
                                else:
                                    st.warning("数据不足进行PCA分析")
                            else:
                                st.warning("无法获取行业内其他股票数据")
                        else:
                            st.warning("未找到该股票对应的行业信息")
                    else:
                        st.warning("请先上传行业映射文件")
                
                elif module_sel == "波动性分析":
                    if "波动性分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "波动性分析")
                        if payload and payload.get("data", {}).get("vol_values") is not None:
                            try:
                                idx_try = pd.to_datetime(payload["data"]["vol_index"], errors="coerce")
                                if hasattr(idx_try, "isna") and idx_try.isna().any():
                                    tail_idx = pd.to_datetime(df_symbol[date_col]).iloc[-len(payload["data"]["vol_values"]):]
                                    idx = tail_idx
                                else:
                                    idx = idx_try
                            except Exception:
                                idx = pd.to_datetime(df_symbol[date_col]).iloc[-len(payload["data"]["vol_values"]):]
                            n = min(len(payload["data"]["vol_values"]), len(idx))
                            vol = pd.Series(list(payload["data"]["vol_values"])[-n:], index=idx[-n:])
                            fig_vol = px.line(x=vol.index, y=vol.values, labels={"x": date_col, "y": "HV(20)"})
                            sigma2 = payload["data"].get("sigma2")
                            mod_sum = f"波动性分析: HV20={vol.iloc[-1]:.4f}, GARCH预测方差={sigma2 if sigma2 else 'N/A'}"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["波动性分析"] = {"vol": vol, "fig_vol": fig_vol, "sigma2": sigma2, "advisor_inputs": advisor_inputs}
                    cached = mod_cache.get("波动性分析")
                    if cached:
                        vol = cached["vol"]; fig_vol = cached["fig_vol"]; sigma2 = cached["sigma2"]; advisor_inputs = cached["advisor_inputs"]
                        st.plotly_chart(fig_vol, width="stretch")
                        if sigma2 is not None:
                            st.metric(label="GARCH预测方差(下一期)", value=f"{sigma2:.6f}")
                        render_analysis_card(explain_volatility_risk(vol, sigma2))
                        advisor_v = get_or_generate_advisor_wrapper("波动性分析", advisor_inputs, facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_v)
                        render_followup("波动性分析", advisor_inputs, advisor_v, facts)
                    else:
                        try:
                            vol = compute_volatility(df_symbol["close"], window=20)
                            fig_vol = px.line(x=vol.index, y=vol.values, labels={"x": date_col, "y": "HV(20)"})
                            st.plotly_chart(fig_vol, width="stretch")
                            
                            sigma2 = None
                            try:
                                res, forecast = compute_garch(df_symbol["close"])
                                sigma2 = forecast.variance.values[-1][-1] if forecast is not None else None
                                if sigma2:
                                    st.metric(label="GARCH预测方差(下一期)", value=f"{sigma2:.6f}")
                            except Exception:
                                pass
                            render_analysis_card(explain_volatility_risk(vol, sigma2))
                            mod_sum = f"波动性分析: HV20={vol.iloc[-1]:.4f}, GARCH预测方差={sigma2 if sigma2 else 'N/A'}"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            advisor_v = get_or_generate_advisor_wrapper("波动性分析", advisor_inputs, facts, "最近一年", symbol, industry)
                            render_advisor_text(advisor_v)
                            render_followup("波动性分析", advisor_inputs, advisor_v, facts)
                            mod_cache["波动性分析"] = {"vol": vol, "fig_vol": fig_vol, "sigma2": sigma2, "advisor_inputs": advisor_inputs}
                        except Exception as e:
                            st.info(f"波动性分析失败: {str(e)}")
                
                elif module_sel == "季节性分析":
                    if "季节性分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "季节性分析")
                        d = payload["data"] if payload else {}
                        if d.get("trend_values") and d.get("seasonal_values") and d.get("resid_values"):
                            trend = pd.Series(d["trend_values"], index=pd.to_datetime(d["trend_index"]))
                            seasonal = pd.Series(d["seasonal_values"], index=pd.to_datetime(d["seasonal_index"]))
                            resid = pd.Series(d["resid_values"], index=pd.to_datetime(d["resid_index"]))
                            fig_stl = plot_stl_components(trend, seasonal, resid)
                            fig_ap = None
                            if d.get("acf") and d.get("pacf"):
                                a_vals = pd.Series(d["acf"])
                                p_vals = pd.Series(d["pacf"])
                                fig_ap = plot_acf_pacf(a_vals, p_vals)
                            mod_sum = "季节性分解分析: 展示了趋势、季节性和残差分量"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["季节性分析"] = {"stl_res": type("STL", (), {"trend": trend, "seasonal": seasonal, "resid": resid})(), "fig_stl": fig_stl, "fig_ap": fig_ap, "advisor_inputs": advisor_inputs}
                    cached = mod_cache.get("季节性分析")
                    if cached:
                        stl_res = cached["stl_res"]; fig_stl = cached["fig_stl"]; fig_ap = cached.get("fig_ap"); advisor_inputs = cached["advisor_inputs"]
                        if fig_stl is not None: st.plotly_chart(fig_stl, width="stretch")
                        if fig_ap is not None: st.plotly_chart(fig_ap, width="stretch")
                        render_analysis_card(explain_seasonality(stl_res))
                        advisor_s = get_or_generate_advisor_wrapper("季节性分析", advisor_inputs, facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_s)
                        render_followup("季节性分析", advisor_inputs, advisor_s, facts)
                    else:
                        try:
                            stl_res = compute_stl(df_symbol.set_index(date_col)["close"], period=7)
                            fig_stl = plot_stl_components(stl_res.trend, stl_res.seasonal, stl_res.resid)
                            st.plotly_chart(fig_stl, width="stretch")
                            
                            fig_ap = None
                            try:
                                a_vals, p_vals = compute_acf_pacf(df_symbol["close"].pct_change().dropna())
                                fig_ap = plot_acf_pacf(a_vals, p_vals)
                                st.plotly_chart(fig_ap, width="stretch")
                            except Exception:
                                pass
                            render_analysis_card(explain_seasonality(stl_res))
                            mod_sum = "季节性分解分析: 展示了趋势、季节性和残差分量"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            advisor_s = get_or_generate_advisor_wrapper("季节性分析", advisor_inputs, facts, "最近一年", symbol, industry)
                            render_advisor_text(advisor_s)
                            render_followup("季节性分析", advisor_inputs, advisor_s, facts)
                            mod_cache["季节性分析"] = {"stl_res": stl_res, "fig_stl": fig_stl, "fig_ap": fig_ap, "advisor_inputs": advisor_inputs}
                        except Exception as e:
                            st.info(f"季节性分析失败: {str(e)}")
                
                elif module_sel == "风险-收益聚类分析":
                    if "风险-收益聚类分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "风险-收益聚类分析")
                        d = payload["data"] if payload else {}
                        if d.get("ret") and d.get("vol") and d.get("labels") and d.get("index"):
                            feat = pd.DataFrame({"ret": pd.Series(d["ret"], index=d["index"]), "vol": pd.Series(d["vol"], index=d["index"])})
                            labels = pd.Series(d["labels"], index=d["index"])
                            fig_cluster = plot_cluster_scatter(feat, labels)
                            mod_sum = f"聚类分析: 将{feat.shape[0]}只股票分为{int(d.get('n_clusters', 0))}类"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["风险-收益聚类分析"] = {"fig_cluster": fig_cluster, "labels": labels, "n_clusters": int(d.get("n_clusters", 0)), "advisor_inputs": advisor_inputs}
                    st.subheader("风险-收益聚类分析")
                    cached = mod_cache.get("风险-收益聚类分析")
                    if cached:
                        st.plotly_chart(cached["fig_cluster"], width="stretch")
                        render_analysis_card(explain_clustering(cached["n_clusters"], cached["labels"]))
                        advisor_km = get_or_generate_advisor_wrapper("风险-收益聚类分析", cached["advisor_inputs"], facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_km)
                        render_followup("风险-收益聚类分析", cached["advisor_inputs"], advisor_km, facts)
                    elif "stock_map" in st.session_state and st.session_state["stock_map"] is not None:
                        df_map = st.session_state["stock_map"]
                        if industry and industry in df_map["industry"].values:
                            df_ind = df_map[df_map["industry"] == industry]
                            symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
                            
                            # 读取行业内其他股票数据
                            all_dfs = []
                            for sym in symbols_in_industry:
                                try:
                                    sym_df = fetch_resolved_df(st.session_state, sym, industry, str(sd_local), str(ed_local), frequency)
                                    if not sym_df.empty:
                                        sym_df = sym_df[['date', 'close']].rename(columns={'close': sym})
                                        sym_df['date'] = pd.to_datetime(sym_df['date'])
                                        all_dfs.append(sym_df)
                                except:
                                    continue
                            
                            if all_dfs:
                                # 合并所有股票数据
                                from functools import reduce
                                pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
                                pivot_close = pivot_close.set_index('date').sort_index()
                                
                                # 计算收益率和波动率
                                rets = pivot_close.pct_change(fill_method=None).dropna()
                                if not rets.empty and rets.shape[1] >= 3:
                                    feat = pd.DataFrame({
                                        "ret": rets.mean() * 252,
                                        "vol": rets.std() * np.sqrt(252)
                                    })
                                    feat = feat.dropna()
                                    
                                    if feat.shape[0] >= 3:
                                        n_clusters = min(3, max(2, feat.shape[0]//2))
                                        km, labels = compute_kmeans(feat, n_clusters=n_clusters)
                                        fig_cluster = plot_cluster_scatter(feat, labels)
                                        st.plotly_chart(fig_cluster, width="stretch")
                                        render_analysis_card(explain_clustering(n_clusters, labels))
                                        # 大模型分析
                                        mod_sum = f"聚类分析: 将{feat.shape[0]}只股票分为{n_clusters}类"
                                        advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                                        advisor_km = get_or_generate_advisor_wrapper("风险-收益聚类分析", advisor_inputs, facts, "最近一年", symbol, industry)
                                        render_advisor_text(advisor_km)
                                        render_followup("风险-收益聚类分析", advisor_inputs, advisor_km, facts)
                                        mod_cache["风险-收益聚类分析"] = {"fig_cluster": fig_cluster, "labels": labels, "n_clusters": n_clusters, "advisor_inputs": advisor_inputs}
                                    else:
                                        st.warning("数据不足进行聚类分析")
                                else:
                                    st.warning("数据不足进行聚类分析")
                            else:
                                st.warning("无法获取行业内其他股票数据")
                        else:
                            st.warning("未找到该股票对应的行业信息")
                    else:
                        st.warning("请先上传行业映射文件")
                
                elif module_sel == "基本面因子暴露分析":
                    if "基本面因子暴露分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "基本面因子暴露分析")
                        d = payload["data"] if payload else {}
                        rows = d.get("portrait")
                        if rows:
                            portrait = pd.DataFrame(rows)
                            fig_portrait = plot_factor_portrait(portrait.fillna(0.5))
                            mod_sum = f"因子画像分析: 分析了{portrait.shape[0]}个基本面指标"
                            advisor_inputs = {"analysis_mode": "portrait", "time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["基本面因子暴露分析"] = {"fig_portrait": fig_portrait, "portrait": portrait.fillna(0.5), "advisor_inputs": advisor_inputs}
                    st.subheader("基本面因子暴露分析")
                    funda_dir_default = str((ROOT / "data" / "fundamentals").resolve())
                    funda_dir = Path(st.session_state.get("funda_dir", funda_dir_default))
                    
                    cached = mod_cache.get("基本面因子暴露分析")
                    if cached:
                        st.plotly_chart(cached["fig_portrait"], width="stretch")
                        render_analysis_card(explain_factor_portrait(cached["portrait"]))
                        advisor_fp = get_or_generate_advisor_wrapper("基本面因子暴露分析", cached["advisor_inputs"], facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_fp)
                        render_followup("基本面因子暴露分析", cached["advisor_inputs"], advisor_fp, facts)
                    else:
                        try:
                            sym_dir = funda_dir / symbol.replace('.', '_')
                            if sym_dir.exists():
                                fi_fp = sym_dir / "fina_indicator.csv"
                                portrait_rows = []
                                if fi_fp.exists():
                                    fi = pd.read_csv(fi_fp)
                                    fi = fi.sort_values(["end_date","ann_date"]) if "end_date" in fi.columns and "ann_date" in fi.columns else fi
                                    metrics_cols = [c for c in ["roe","roa","grossprofit_margin","debt_to_assets","oper_cash_flow","pe","pb"] if c in fi.columns]
                                    if metrics_cols:
                                        for m in metrics_cols:
                                            ser = pd.to_numeric(fi[m], errors="coerce").dropna()
                                            val = float(ser.iloc[-1]) if not ser.empty else None
                                            trend = float((ser.diff().dropna().iloc[-1])) if ser.shape[0] >= 2 else 0.0
                                            portrait_rows.append({"metric": m, "value": val, "trend": trend})
                                if portrait_rows:
                                    portrait = pd.DataFrame(portrait_rows)
                                    fig_portrait = plot_factor_portrait(portrait.fillna(0.5))
                                    st.plotly_chart(fig_portrait, width="stretch")
                                    render_analysis_card(explain_factor_portrait(portrait.fillna(0.5)))
                                    mod_sum = f"因子画像分析: 分析了{len(portrait_rows)}个基本面指标"
                                    advisor_inputs = {"analysis_mode": "portrait", "time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                                    advisor_fp = get_or_generate_advisor_wrapper("基本面因子暴露分析", advisor_inputs, facts, "最近一年", symbol, industry)
                                    render_advisor_text(advisor_fp)
                                    render_followup("基本面因子暴露分析", advisor_inputs, advisor_fp, facts)
                                    mod_cache["基本面因子暴露分析"] = {"fig_portrait": fig_portrait, "portrait": portrait.fillna(0.5), "advisor_inputs": advisor_inputs}
                                else:
                                    st.warning("未找到基本面数据")
                            else:
                                st.warning("未找到该股票对应的基本面数据目录")
                                st.warning("请先采集该股票的基本面数据")
                        except Exception as e:
                            st.info(f"基本面因子分析失败: {str(e)}")
                
                elif module_sel == "涨跌概率分析":
                    if "涨跌概率分析" not in mod_cache:
                        payload = load_module_result(ROOT, symbol, "涨跌概率分析")
                        d = payload["data"] if payload else {}
                        if d.get("proba") is not None and d.get("auc") is not None:
                            proba = pd.Series(d["proba"])
                            auc = float(d["auc"])
                            fig_prob = plot_probability_hist(proba)
                            mod_sum = f"涨跌概率预测: AUC={auc:.3f}, 平均上涨概率={proba.mean():.3f}"
                            advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                            mod_cache["涨跌概率分析"] = {"proba": proba, "auc": auc, "fig_prob": fig_prob, "advisor_inputs": advisor_inputs}
                    st.subheader("涨跌概率分析")
                    cached = mod_cache.get("涨跌概率分析")
                    if cached:
                        if cached["fig_prob"] is not None: st.plotly_chart(cached["fig_prob"], width="stretch")
                        if cached["auc"] is not None: st.metric(label="AUC", value=f"{cached['auc']:.3f}")
                        render_analysis_card(explain_prediction_probability(cached["proba"], cached["auc"]))
                        advisor_lr = get_or_generate_advisor_wrapper("涨跌概率分析", cached["advisor_inputs"], facts, "最近一年", symbol, industry)
                        render_advisor_text(advisor_lr)
                        render_followup("涨跌概率分析", cached["advisor_inputs"], advisor_lr, facts)
                    elif "stock_map" in st.session_state and st.session_state["stock_map"] is not None:
                        df_map = st.session_state["stock_map"]
                        if industry and industry in df_map["industry"].values:
                            df_ind = df_map[df_map["industry"] == industry]
                            symbols_in_industry = df_ind["symbol"].dropna().astype(str).tolist()
                            
                            # 读取行业内其他股票数据
                            all_dfs = []
                            for sym in symbols_in_industry:
                                try:
                                    sym_df = fetch_resolved_df(st.session_state, sym, industry, str(sd_local), str(ed_local), frequency)
                                    if not sym_df.empty:
                                        sym_df = sym_df[['date', 'close']].rename(columns={'close': sym})
                                        sym_df['date'] = pd.to_datetime(sym_df['date'])
                                        all_dfs.append(sym_df)
                                except:
                                    continue
                            
                            if all_dfs and len(all_dfs) >= 5:
                                # 合并所有股票数据
                                from functools import reduce
                                pivot_close = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), all_dfs)
                                pivot_close = pivot_close.set_index('date').sort_index()
                                
                                # 计算年化收益率
                                R = pivot_close.pct_change(fill_method=None).dropna().mean() * 252
                                thr = float(R.median())
                                y_cls = (R > thr).astype(int)
                                
                                # 构建特征
                                X_rows = []
                                for s in symbols_in_industry:
                                    try:
                                        sym_df = fetch_resolved_df(st.session_state, s, industry, str(sd_local), str(ed_local), frequency)
                                        if not sym_df.empty:
                                            sym_df['date'] = pd.to_datetime(sym_df['date'])
                                            sym_df = sym_df.sort_values('date')
                                            slope_sma = float(sym_df["close"].rolling(20).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
                                            slope_ema = float(sym_df["close"].ewm(span=60, adjust=False).mean().diff().dropna().iloc[-1]) if "close" in sym_df.columns else None
                                            X_rows.append({"symbol": s, "sma_slope": slope_sma, "ema_slope": slope_ema})
                                    except:
                                        continue
                                
                                X = pd.DataFrame(X_rows).set_index("symbol")
                                X = X.loc[y_cls.index.intersection(X.index)]
                                
                                if not X.empty and X.shape[0] >= 5:
                                    model, proba, auc = compute_logistic_proba(X, y_cls.loc[X.index])
                                    fig_prob = plot_probability_hist(proba)
                                    st.plotly_chart(fig_prob, width="stretch")
                                    st.metric(label="AUC", value=f"{auc:.3f}")
                                    render_analysis_card(explain_prediction_probability(proba, auc))
                                    # 大模型分析
                                    mod_sum = f"涨跌概率预测: AUC={auc:.3f}, 平均上涨概率={proba.mean():.3f}"
                                    advisor_inputs = {"time_range": "最近一年", "stock_name": name, "stock_code": symbol, "industry": industry, "module_data_summary": mod_sum}
                                    advisor_lr = get_or_generate_advisor_wrapper("涨跌概率分析", advisor_inputs, facts, "最近一年", symbol, industry)
                                    render_advisor_text(advisor_lr)
                                    render_followup("涨跌概率分析", advisor_inputs, advisor_lr, facts)
                                    mod_cache["涨跌概率分析"] = {"proba": proba, "auc": auc, "fig_prob": fig_prob, "advisor_inputs": advisor_inputs}
                                else:
                                    st.warning("数据不足进行逻辑回归分析")
                            else:
                                st.warning("行业内股票数据不足，至少需要5只股票")
                        else:
                            st.warning("未找到该股票对应的行业信息")
                    else:
                        st.warning("请先上传行业映射文件")
    else:
        pass

# 右侧模块：行业与股票选择
with right_col:
    if main_function == "证券多因子量化评分":
        st.header("证券多因子量化评分")
        # 行业选择
        industries_all = []
        if "stock_map" in st.session_state:
            industries_all = sorted(set(st.session_state["stock_map"]["industry"].dropna().tolist()))
        
        if industries_all:
            selected_industry = st.selectbox("选择行业", industries_all, key="ind_sel_right")
            
            # 行业内股票选择
            df_map = st.session_state.get("stock_map")
            if df_map is not None:
                df_ind = df_map[df_map["industry"] == selected_industry]
                syms_ind = df_ind["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind.columns else []
                names_ind = df_ind["name"].astype(str).tolist() if "name" in df_ind.columns else []
                
                if syms_ind:
                    # 创建股票选择界面
                    st.subheader("选择行业内证券")
                    stock_options = {}
                    for i, (sym, name) in enumerate(zip(syms_ind, names_ind)):
                        if i < len(names_ind):
                            stock_options[f"{name}({sym})"] = sym
                        else:
                            stock_options[sym] = sym
                    
                    selected_labels = st.multiselect(
                        "选择证券",
                        options=list(stock_options.keys()),
                        default=list(stock_options.keys())[:min(3, len(stock_options))],
                        key="stock_sel_right"
                    )
                    
                    selected_symbols = [stock_options[label] for label in selected_labels]
                    st.session_state["selected_symbols"] = selected_symbols
                    
                    if selected_symbols:
                        # 权重配置
                        st.subheader("权重配置")
                        with st.expander("权重配置说明", expanded=False):
                            st.caption("盈利能力权重：关注公司赚钱效率与质量（如 ROE、毛利率、净利率）。权重越高，越偏好稳健盈利的公司。")
                            st.caption("偿债能力权重：关注负债结构与现金流偿付能力（如资产负债率、流动/速动比率、经营现金流）。权重越高，越重视抗风险能力。")
                            st.caption("成长性权重：关注营收与利润的增长（如同比增速、复合增速）。权重越高，越偏好具备扩张与成长潜力的公司。")
                            st.caption("投资回报权重：关注股东回报与估值性价比（如股息、ROE≈ROI、PE/PB合理性）。权重越高，越偏好当前回报与估值更友好。")
                            st.caption("提示：四项权重会自动归一化，总和为 1；可先选预设，再用滑块微调。")
                        preset = st.radio("权重预设", ["稳健","均衡","进取"], index=1, horizontal=True, key="w_preset")
                        preset_vals = {"稳健": (0.5,0.3,0.1,0.1), "均衡": (0.4,0.3,0.2,0.1), "进取": (0.3,0.2,0.3,0.2)}
                        
                        if st.session_state.get("last_preset") != preset:
                            wp, ws, wg, wr = preset_vals[preset]
                            st.session_state["w_profit"] = wp
                            st.session_state["w_solv"] = ws
                            st.session_state["w_grow"] = wg
                            st.session_state["w_ret"] = wr
                            st.session_state["last_preset"] = preset
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            wp = st.slider("盈利能力权重", min_value=0.0, max_value=1.0, 
                                         value=float(st.session_state.get("w_profit",0.4)), step=0.05, key="w_profit_right")
                            ws = st.slider("偿债能力权重", min_value=0.0, max_value=1.0, 
                                         value=float(st.session_state.get("w_solv",0.3)), step=0.05, key="w_solv_right")
                        with col2:
                            wg = st.slider("成长性权重", min_value=0.0, max_value=1.0, 
                                         value=float(st.session_state.get("w_grow",0.2)), step=0.05, key="w_grow_right")
                            wr = st.slider("投资回报权重", min_value=0.0, max_value=1.0, 
                                         value=float(st.session_state.get("w_ret",0.1)), step=0.05, key="w_ret_right")
                        
                        funda_dir_default = str((ROOT / "data" / "fundamentals").resolve())
                        funda_dir = Path(st.session_state.get("funda_dir", funda_dir_default))
                        missing = []
                        for s in selected_symbols:
                            sym_dir = funda_dir / s.replace('.','_')
                            need = []
                            for fn in ["fina_indicator.csv","balancesheet.csv","income.csv","cashflow.csv"]:
                                if not (sym_dir / fn).exists():
                                    need.append(fn)
                            if need:
                                missing.append((s, need))
                        if missing:
                            st.warning(f"所选证券缺少财报数据：{len(missing)} 个。评分为零或报告数据缺失通常由此导致。")
                            with st.expander("缺失详情", expanded=False):
                                for s, need in missing:
                                    st.markdown(f"- {s} 缺少: {', '.join(need)}")
                            colm1, colm2 = st.columns(2)
                            with colm1:
                                sd_fi = st.date_input("财报开始日期", value=datetime.now()-timedelta(days=365*3), key="fi_sd_right")
                            with colm2:
                                ed_fi = st.date_input("财报结束日期", value=datetime.now(), key="fi_ed_right")
                            do_fetch = st.button("一键采集所选证券财报", key="fetch_fi_right")
                            if do_fetch:
                                ok = validate_tushare_token()
                                if not ok:
                                    st.error("Tushare Token 无效或权限不足，请在 .env.local 配置 TUSHARE_TOKEN")
                                else:
                                    with st.spinner("正在采集财报数据..."):
                                        for s in selected_symbols:
                                            try:
                                                export_financials_single(s, str(sd_fi).replace("-",""), str(ed_fi).replace("-",""), funda_dir, selected_industry)
                                            except Exception as e:
                                                pass
                                    st.success("财报采集完成，请再次点击“运行行业分析”")
                        
                        # 运行行业分析
                        run_industry_btn = st.button("运行行业分析", key="run_industry_right", type="primary")
                        
                        if run_industry_btn:
                            weights = {"profitability": wp, "solvency": ws, "growth": wg, "return": wr}
                            tw = sum(weights.values())
                            if abs(tw - 1.0) > 1e-6 and tw > 0:
                                for k in list(weights.keys()):
                                    weights[k] = weights[k]/tw
                            
                            funda_dir_default = str((ROOT / "data" / "fundamentals").resolve())
                            funda_dir = Path(st.session_state.get("funda_dir", funda_dir_default))
                            
                            with st.spinner("正在计算行业评分..."):
                                res = compute_industry_scoring(df_map, selected_industry, funda_dir, selected_symbols, weights)
                                
                                if res is not None and not res.empty:
                                  
                                    st.subheader("评分结果")
                                    df_show = res.reset_index()[["symbol","score_profitability","score_solvency","score_growth","score_return","composite_score"]].rename(columns={
                                        "symbol": "证券代码",
                                        "score_profitability": "盈利能力评分",
                                        "score_solvency": "偿债能力评分",
                                        "score_growth": "成长性评分",
                                        "score_return": "投资回报评分",
                                        "composite_score": "综合评分"
                                    })
                                    st.dataframe(df_show)
                                    
                                    st.subheader("行业基准与横向分析")
                                    all_dfs = []
                                    for sym in selected_symbols:
                                        try:
                                            df_i = fetch_resolved_df(st.session_state, sym, selected_industry, start_date, end_date, frequency)
                                            if not df_i.empty:
                                                df_i = df_i[["date","close"]].rename(columns={"close": sym})
                                                df_i["date"] = pd.to_datetime(df_i["date"])
                                                all_dfs.append(df_i)
                                        except Exception:
                                            pass
                                    pivot_close = None
                                    if all_dfs:
                                        from functools import reduce
                                        pivot_close = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), all_dfs)
                                        pivot_close = pivot_close.set_index("date").sort_index()
                                    core_rows = []
                                    pe_vals = []
                                    pb_vals = []
                                    if pivot_close is not None and not pivot_close.empty:
                                        R = pivot_close.pct_change(fill_method=None).dropna().mean() * 252
                                        V = pivot_close.pct_change(fill_method=None).dropna().std() * np.sqrt(252)
                                        for sym in selected_symbols:
                                            ser = pd.to_numeric(pivot_close[sym], errors="coerce") if sym in pivot_close.columns else pd.Series(dtype=float)
                                            ret = float(compute_ann_return(ser)) if not ser.empty else 0.0
                                            vol = float(ser.pct_change(fill_method=None).dropna().std() * np.sqrt(252)) if not ser.empty else 0.0
                                            mdd = float(compute_max_drawdown(ser)) if not ser.empty else 0.0
                                            shp = float(compute_sharpe(ser)) if not ser.empty else 0.0
                                            funda_dir_default = str((ROOT / "data" / "fundamentals").resolve())
                                            fi_fp = Path(st.session_state.get("funda_dir", funda_dir_default)) / sym.replace(".","_") / "fina_indicator.csv"
                                            db_fp = Path(st.session_state.get("funda_dir", funda_dir_default)) / sym.replace(".","_") / "daily_basic.csv"
                                            pe = None; pb = None
                                            db = None
                                            if db_fp.exists():
                                                try:
                                                    db = pd.read_csv(db_fp)
                                                except Exception:
                                                    db = None
                                            if db is not None:
                                                db = db.sort_values(["trade_date"]) if "trade_date" in db.columns else db
                                                pe_cols_db = [c for c in ["pe","pe_ttm"] if c in db.columns]
                                                pb_cols_db = [c for c in ["pb","pb_mrq"] if c in db.columns]
                                                if pe_cols_db:
                                                    ser_pe = pd.to_numeric(db[pe_cols_db[0]], errors="coerce").dropna()
                                                    if not ser_pe.empty:
                                                        pe = float(ser_pe.iloc[-1])
                                                if pb_cols_db:
                                                    ser_pb = pd.to_numeric(db[pb_cols_db[0]], errors="coerce").dropna()
                                                    if not ser_pb.empty:
                                                        pb = float(ser_pb.iloc[-1])
                                            if (pe is None or pb is None) and fi_fp.exists():
                                                try:
                                                    fi = pd.read_csv(fi_fp)
                                                    fi = fi.sort_values(["end_date","ann_date"]) if "end_date" in fi.columns and "ann_date" in fi.columns else fi
                                                    pe_cols = [c for c in ["pe","pe_ttm","pe_basic","pe_circ"] if c in fi.columns]
                                                    pb_cols = [c for c in ["pb","pb_mrq"] if c in fi.columns]
                                                    if pe is None and pe_cols:
                                                        ser_pe = pd.to_numeric(fi[pe_cols[0]], errors="coerce").dropna()
                                                        if not ser_pe.empty:
                                                            pe = float(ser_pe.iloc[-1])
                                                    if pb is None and pb_cols:
                                                        ser_pb = pd.to_numeric(fi[pb_cols[0]], errors="coerce").dropna()
                                                        if not ser_pb.empty:
                                                            pb = float(ser_pb.iloc[-1])
                                                except Exception:
                                                    pass
                                            s_close = pd.to_numeric(pivot_close[sym], errors="coerce").dropna() if sym in pivot_close.columns else pd.Series(dtype=float)
                                            price_last = float(s_close.iloc[-1]) if not s_close.empty else None
                                            if pe is None and fi_fp.exists() and price_last is not None:
                                                try:
                                                    fi2 = pd.read_csv(fi_fp)
                                                    fi2 = fi2.sort_values(["end_date","ann_date"]) if "end_date" in fi2.columns and "ann_date" in fi2.columns else fi2
                                                    eps_cols = [c for c in ["eps","eps_basic","basic_eps","eps_ttm","eps_diluted"] if c in fi2.columns]
                                                    if eps_cols:
                                                        ser_eps = pd.to_numeric(fi2[eps_cols[0]], errors="coerce").dropna()
                                                        if not ser_eps.empty:
                                                            eps_last = float(ser_eps.iloc[-1])
                                                            if eps_last != 0:
                                                                pe = price_last / eps_last
                                                except Exception:
                                                    pass
                                            if pb is None and price_last is not None:
                                                bps = None
                                                if fi_fp.exists():
                                                    try:
                                                        fi3 = pd.read_csv(fi_fp)
                                                        fi3 = fi3.sort_values(["end_date","ann_date"]) if "end_date" in fi3.columns and "ann_date" in fi3.columns else fi3
                                                        bps_cols = [c for c in ["bps","net_asset_ps","net_assets_ps"] if c in fi3.columns]
                                                        if bps_cols:
                                                            ser_bps = pd.to_numeric(fi3[bps_cols[0]], errors="coerce").dropna()
                                                            if not ser_bps.empty:
                                                                bps = float(ser_bps.iloc[-1])
                                                    except Exception:
                                                        bps = None
                                                if bps is not None and bps != 0:
                                                    pb = price_last / bps
                                            if pe is not None: pe_vals.append(pe)
                                            if pb is not None: pb_vals.append(pb)
                                            core_rows.append({"symbol": sym, "收益率": ret, "波动率": vol, "最大回撤": mdd, "夏普比率": shp, "PE": pe, "PB": pb})
                                        core_df = pd.DataFrame(core_rows).set_index("symbol")
                                        baseline_ret = float(R.mean()) if not R.empty else 0.0
                                        baseline_vol = float(V.mean()) if not V.empty else 0.0
                                        pe_med = float(pd.Series(pe_vals).median()) if pe_vals else None
                                        pb_med = float(pd.Series(pb_vals).median()) if pb_vals else None
                                        idx_norm = pivot_close.copy()
                                        for c in idx_norm.columns:
                                            try:
                                                s = pd.to_numeric(idx_norm[c], errors="coerce").dropna()
                                                if not s.empty and s.iloc[0] != 0:
                                                    idx_norm[c] = s / s.iloc[0]
                                            except Exception:
                                                pass
                                        ind_idx = idx_norm.mean(axis=1).dropna()
                                        mood = "震荡"
                                        try:
                                            window = ind_idx.iloc[-90:] if ind_idx.shape[0] >= 90 else ind_idx
                                            change = float(window.iloc[-1]/window.iloc[0] - 1.0) if window.iloc[0] else 0.0
                                            if change > 0.05:
                                                mood = "景气上行期"
                                            elif change < -0.05:
                                                mood = "景气下行期"
                                        except Exception:
                                            mood = "震荡"
                                        st.markdown(f"行业基准：平均年化收益率 {baseline_ret:.2%}，平均波动率 {baseline_vol:.2%}，估值中位数 PE={pe_med if pe_med is not None else 'N/A'}，PB={pb_med if pb_med is not None else 'N/A'}，行业整体：{mood}")
                                        tags_out = []
                                        for sym, row in core_df.iterrows():
                                            tg = generate_security_tags(float(row["收益率"]), float(row["波动率"]), float(row["最大回撤"]), float(row["夏普比率"]), row["PE"] if not pd.isna(row["PE"]) else None, row["PB"] if not pd.isna(row["PB"]) else None, baseline_ret, baseline_vol)
                                            tags_out.append({"symbol": sym, "标签": "、".join(tg)})
                                        tags_df = pd.DataFrame(tags_out).set_index("symbol")
                                        core_df = core_df.join(tags_df, how="left")
                                        st.dataframe(core_df)
                                        
                                        st.subheader("个股星级报告")
                                        def _to_0_100_series(s):
                                            s_num = pd.to_numeric(s, errors="coerce")
                                            clipped = s_num.clip(-3.0, 3.0)
                                            return (clipped + 3.0) / 6.0 * 100.0
                                        def _star(p):
                                            if p is None: return "N/A"
                                            if p >= 90: return "⭐⭐⭐⭐⭐"
                                            if p >= 80: return "⭐⭐⭐⭐"
                                            if p >= 70: return "⭐⭐⭐"
                                            if p >= 60: return "⭐⭐"
                                            return "⭐"
                                        def _bucket(p, kind):
                                            if p is None: return "数据不足"
                                            if kind == "profit":
                                                if p >= 80: return "赚钱能力很强"
                                                if p >= 60: return "赚钱能力不错"
                                                if p >= 40: return "赚钱一般"
                                                if p >= 20: return "赚钱偏弱"
                                                return "赚钱能力差"
                                            if kind == "solvency":
                                                if p >= 80: return "财务非常稳健"
                                                if p >= 60: return "财务健康"
                                                if p >= 40: return "尚可接受"
                                                if p >= 20: return "压力偏大"
                                                return "存在偿债风险"
                                            if kind == "growth":
                                                if p >= 80: return "高成长公司"
                                                if p >= 60: return "稳健增长"
                                                if p >= 40: return "增长一般"
                                                if p >= 20: return "增长乏力"
                                                return "基本不增长"
                                            if kind == "return":
                                                if p >= 80: return "回报非常友好"
                                                if p >= 60: return "回报较好"
                                                if p >= 40: return "回报一般"
                                                if p >= 20: return "回报偏低"
                                                return "回报不理想"
                                            return ""
                                        def _map_fin_profit(v):
                                            if v is None or pd.isna(v): return None
                                            x = float(v)
                                            if x >= 0:
                                                return 60.0 + min(x, 3.0) / 3.0 * 40.0
                                            return max(0.0, 60.0 + max(x, -3.0) / 3.0 * 60.0)
                                        def _map_fin_solvency(v):
                                            if v is None or pd.isna(v): return None
                                            x = float(v)
                                            if x >= 0:
                                                return 80.0 + min(x, 3.0) / 3.0 * 20.0
                                            if x >= -10.0:
                                                return 60.0 + (x + 10.0) / 10.0 * 20.0
                                            if x >= -30.0:
                                                return 30.0 + (x + 30.0) / 20.0 * 30.0
                                            return max(0.0, 0.0 + min(x, -60.0) / -30.0 * 30.0)
                                        def _map_fin_growth(v):
                                            if v is None or pd.isna(v): return None
                                            x = float(v)
                                            if x <= -1.0: return 0.0
                                            if x <= 0.0: return 0.0
                                            return min(100.0, x * 100.0)
                                        def _map_fin_return(v):
                                            return _map_fin_profit(v)
                                        def _percentile_score(series, val, reverse=False):
                                            s = pd.to_numeric(series, errors="coerce").dropna()
                                            if s.empty or val is None or pd.isna(val): return None
                                            import numpy as np
                                            arr = s.values
                                            rank = float((arr <= val).sum()) / float(arr.size) * 100.0 if not reverse else float((arr >= val).sum()) / float(arr.size) * 100.0
                                            return rank
                                        def _valuation_score(val, med):
                                            if val is None or pd.isna(val) or med is None: return None
                                            m = float(med)
                                            if m == 0.0: return None
                                            d = abs(float(val) - m) / abs(m)
                                            sc = 100.0 - min(100.0, d * 100.0)
                                            return max(0.0, sc)
                                        tag_map = {"↑中性": 70.0, "△高波动不稳定": 40.0}
                                        Pp_raw = res["score_profitability"]
                                        Ps_raw = res["score_solvency"]
                                        Pg_raw = res["score_growth"]
                                        Pr_raw = res["score_return"]
                                        Pp = Pp_raw.apply(_map_fin_profit)
                                        Ps = Ps_raw.apply(_map_fin_solvency)
                                        Pg = Pg_raw.apply(_map_fin_growth)
                                        Pr = Pr_raw.apply(_map_fin_return)
                                        Pc = pd.Series(dtype=float)
                                        for sym in res.index.tolist():
                                            p = Pp.loc[sym] if sym in Pp.index else None
                                            ssv = Ps.loc[sym] if sym in Ps.index else None
                                            g = Pg.loc[sym] if sym in Pg.index else None
                                            r = Pr.loc[sym] if sym in Pr.index else None
                                            ret_v = float(core_df.loc[sym, "收益率"]) if sym in core_df.index and not pd.isna(core_df.loc[sym, "收益率"]) else None
                                            shp_v = float(core_df.loc[sym, "夏普比率"]) if sym in core_df.index and not pd.isna(core_df.loc[sym, "夏普比率"]) else None
                                            mdd_v = float(core_df.loc[sym, "最大回撤"]) if sym in core_df.index and not pd.isna(core_df.loc[sym, "最大回撤"]) else None
                                            vol_v = float(core_df.loc[sym, "波动率"]) if sym in core_df.index and not pd.isna(core_df.loc[sym, "波动率"]) else None
                                            pe_v = core_df.loc[sym, "PE"] if sym in core_df.index else None
                                            pb_v = core_df.loc[sym, "PB"] if sym in core_df.index else None
                                            ret_sc = _percentile_score(core_df["收益率"], ret_v, reverse=False)
                                            shp_sc = _percentile_score(core_df["夏普比率"], shp_v, reverse=False)
                                            mdd_sc = _percentile_score(abs(core_df["最大回撤"]), abs(mdd_v), reverse=True)
                                            vol_sc = _percentile_score(core_df["波动率"], vol_v, reverse=True)
                                            pe_sc = _valuation_score(pe_v, pe_med) if pe_med is not None else None
                                            pb_sc = _valuation_score(pb_v, pb_med) if pb_med is not None else None
                                            fin_list = [p, ssv, g, r]
                                            fin_avg = float(pd.Series([x for x in fin_list if x is not None]).mean()) if any(x is not None for x in fin_list) else None
                                            mkt_list = [ret_sc, shp_sc, mdd_sc, vol_sc]
                                            mkt_avg = float(pd.Series([x for x in mkt_list if x is not None]).mean()) if any(x is not None for x in mkt_list) else None
                                            val_list = [pe_sc, pb_sc]
                                            val_avg = float(pd.Series([x for x in val_list if x is not None]).mean()) if any(x is not None for x in val_list) else None
                                            tag_raw = core_df.loc[sym, "标签"] if sym in core_df.index and "标签" in core_df.columns else None
                                            tag_score = None
                                            if tag_raw and isinstance(tag_raw, str):
                                                tags_arr = [t.strip() for t in tag_raw.split("、") if t.strip()]
                                                if tags_arr:
                                                    tag_vals = [tag_map.get(t, 50.0) for t in tags_arr]
                                                    tag_score = float(pd.Series(tag_vals).mean())
                                            parts = []
                                            if fin_avg is not None: parts.append(fin_avg * 0.5)
                                            if mkt_avg is not None: parts.append(mkt_avg * 0.3)
                                            if val_avg is not None: parts.append(val_avg * 0.1)
                                            if tag_score is not None: parts.append(tag_score * 0.1)
                                            comp = float(pd.Series(parts).sum()) if parts else None
                                            Pc.loc[sym] = comp if comp is not None else np.nan
                                            try:
                                                comp_raw_val = float(res["composite_score"].loc[sym])
                                                if pd.isna(comp_raw_val):
                                                # safeguard for NaN
                                                    comp_raw = None
                                                else:
                                                    comp_raw = comp_raw_val
                                            except Exception:
                                                comp_raw = None
                                            star = _star(comp)
                                            header_txt = f"{sym} | 综合星级：{star}"
                                            if comp_raw is not None:
                                                header_txt += f" | 综合分（标准化）：{comp_raw:.2f}"
                                            def _explain(kind, tag):
                                                if kind == "profit":
                                                    if tag == "赚钱能力很强": return "ROE/EPS与利润率显著领先，盈利结构稳健，具有持续性。"
                                                    if tag == "赚钱能力不错": return "盈利稳定，盈利质量良好，具备一定护城河与成本控制能力。"
                                                    if tag == "赚钱一般": return "处于行业中游，利润率随周期波动，需关注提效与产品结构优化。"
                                                    if tag == "赚钱偏弱": return "盈利水平偏低或不稳定，建议谨慎观察基本面改善信号。"
                                                    return "短期盈利承压或商业模式待验证，需降低预期并控制仓位。"
                                                if kind == "solvency":
                                                    if tag == "财务非常稳健": return "负债可控，现金流充足，流动/速动比率在安全区间，抗风险能力强。"
                                                    if tag == "财务健康": return "负债结构合理，偿债压力不大，资金周转正常。"
                                                    if tag == "尚可接受": return "偿债能力一般，需关注负债与现金流的变化趋势。"
                                                    if tag == "压力偏大": return "负债率偏高或偿付能力走弱，建议降风险暴露。"
                                                    return "财务风险较高，建议回避或等待财务结构显著改善。"
                                                if kind == "growth":
                                                    if tag == "高成长公司": return "收入与利润高速扩张，外延与内生增长均具备动力。"
                                                    if tag == "稳健增长": return "增速高于行业平均，增长质量较好。"
                                                    if tag == "增长一般": return "增长接近行业平均，需寻找新产品或渠道提升动能。"
                                                    if tag == "增长乏力": return "接近停滞，需通过降本增效或结构调整改善。"
                                                    return "成熟或衰退阶段，增长弹性弱，策略以估值与分红为主。"
                                                if kind == "return":
                                                    if tag == "回报非常友好": return "高ROE与分红水平，资本回报突出，长期配置友好。"
                                                    if tag == "回报较好": return "股东回报较优，具备中长期持有价值。"
                                                    if tag == "回报一般": return "回报更多来自股价波动，需结合估值与趋势择时。"
                                                    if tag == "回报偏低": return "资金占用效率较低，需等待基本面改善或估值切换。"
                                                    return "长期回报不理想，建议谨慎或回避。"
                                                return ""
                                            with st.expander(header_txt, expanded=False):
                                                bp = _bucket(p, 'profit')
                                                st.markdown(f"📈 盈利能力：{bp}")
                                                st.caption(_explain('profit', bp))
                                                bs_ = _bucket(ssv, 'solvency')
                                                st.markdown(f"🛡️ 财务结构：{bs_}")
                                                st.caption(_explain('solvency', bs_))
                                                bg_ = _bucket(g, 'growth')
                                                st.markdown(f"🌱 成长性：{bg_}")
                                                st.caption(_explain('growth', bg_))
                                                br_ = _bucket(r, 'return')
                                                st.markdown(f"💰 投资回报：{br_}")
                                                st.caption(_explain('return', br_))
                                                if comp is not None:
                                                    one_line = "优秀，各项表现均衡优异" if comp >= 90 else ("良好，多数指标表现突出" if comp >= 80 else ("中等，符合行业平均水平" if comp >= 70 else ("一般，存在明显短板" if comp >= 60 else "较差，多指标表现不佳")))
                                                    st.markdown(f"🧠 综合判断：{one_line}")
                                                    if comp >= 90:
                                                        st.caption("建议：核心持仓，分散控制行业与风格风险，关注估值与回撤约束。")
                                                    elif comp >= 80:
                                                        st.caption("建议：重点关注，结合估值分批配置，保持止损纪律。")
                                                    elif comp >= 70:
                                                        st.caption("建议：中性配置，择机参与，需与行业环境与估值共振。")
                                                    elif comp >= 60:
                                                        st.caption("建议：轻仓观察，存在短板，等待基本面或行业催化改善后再提高权重。")
                                                    else:
                                                        st.caption("建议：回避或低权重持有，风险较高，需明确改善信号后再评估。")
                                                try:
                                                    dj = []
                                                    if isinstance(bp, str): dj.append(f"盈利：{bp}")
                                                    if isinstance(bs_, str): dj.append(f"财务：{bs_}")
                                                    if isinstance(bg_, str): dj.append(f"成长：{bg_}")
                                                    if isinstance(br_, str): dj.append(f"回报：{br_}")
                                                    comp_txt = f"综合星级：{star}，综合得分：{comp:.1f}" if comp is not None else f"综合星级：{star}"
                                                    base_txt = "；".join(dj)
                                                    xj_text = f"{sym}（{selected_industry}）综合点评：{comp_txt}；{base_txt}。"
                                                    advisor_text(xj_text)
                                                except Exception:
                                                    pass
                                       
                                        st.subheader("排名与梯队")
                                        rank_profit = res["score_profitability"].sort_values(ascending=False)
                                        rank_risk = res["score_solvency"].sort_values(ascending=False)
                                        rank_comp = res["composite_score"].sort_values(ascending=False)
                                        def _tiers(series: pd.Series) -> pd.DataFrame:
                                            n = series.shape[0]
                                            q1 = int(max(1, round(n*0.33)))
                                            q2 = int(max(1, round(n*0.66)))
                                            labels = []
                                            for i, s in enumerate(series.index):
                                                if i < q1:
                                                    labels.append("第一梯队")
                                                elif i < q2:
                                                    labels.append("第二梯队")
                                                else:
                                                    labels.append("第三梯队")
                                            return pd.DataFrame({"symbol": series.index, "tier": labels}).set_index("symbol")
                                        tiers_df = _tiers(rank_comp)
                                        st.markdown(f"收益能力排名：{', '.join(rank_profit.index.tolist())}")
                                        st.markdown(f"风险控制能力排名：{', '.join(rank_risk.index.tolist())}")
                                        st.markdown(f"综合评分排名：{', '.join(rank_comp.index.tolist())}")
                                        t_groups = tiers_df.groupby("tier").apply(lambda x: ", ".join(x.index.tolist()))
                                        for tier, members in t_groups.items():
                                            st.markdown(f"{tier}：{members}")
                                        st.subheader("行业相关性与分散性")
                                        corr = compute_corr_matrix(pivot_close)
                                        fig_corr = plot_corr_heatmap(corr)
                                        st.plotly_chart(fig_corr, width="stretch")
                                        st.caption(f"平均相关性 {corr.mean().mean():.3f}，相关性越低分散效果越好。")
                                        rets = pivot_close.pct_change(fill_method=None).dropna()
                                        if not rets.empty:
                                            port_ret = rets.mean(axis=1)
                                            port_vol = float(port_ret.std() * np.sqrt(252))
                                            ind_vol_mean = float(rets.std().mean() * np.sqrt(252))
                                            drop_pct = float((ind_vol_mean - port_vol) / ind_vol_mean) if ind_vol_mean else 0.0
                                            avgc = float(corr.mean().mean())
                                            if avgc >= 0.7:
                                                tag = "高度联动型"
                                            elif avgc >= 0.3:
                                                tag = "弱联动型"
                                            elif avgc > 0.0:
                                                tag = "分散配置型"
                                            else:
                                                tag = "对冲型"
                                            if drop_pct >= 0.10:
                                                eff = "分散效果显著"
                                            elif drop_pct >= 0.05:
                                                eff = "分散效果一般"
                                            else:
                                                eff = "分散效果有限"
                                            st.markdown(f"小结论：行业整体为「{tag}」，等权组合波动率较个股平均降低 {drop_pct*100:.1f}%（{eff}）。")
                                            if avgc >= 0.5:
                                                st.markdown("提示：平均相关性偏高，建议引入低相关行业以提升组合稳定性。")
                                            st.caption("人话：如果老是一起涨、一起跌，那分散就不明显；走势越不一样，甚至反着走，才更能帮你降风险。")
                                        industry_summary = {
                                            "industry": selected_industry,
                                            "selected_stocks": selected_symbols,
                                            "weights": weights,
                                            "scores": res.to_dict(orient="index"),
                                            "baseline": {"avg_return": baseline_ret, "avg_volatility": baseline_vol, "pe_median": pe_med, "pb_median": pb_med, "mood": mood},
                                            "core_metrics": core_df.to_dict(orient="index")
                                        }
                                    else:
                                        industry_summary = {
                                            "industry": selected_industry,
                                            "selected_stocks": selected_symbols,
                                            "weights": weights,
                                            "scores": res.to_dict(orient="index")
                                        }
                                    # 大模型分析区域
                                    st.divider()
                                    st.subheader("🤖 小金的行业分析")
                                    
                                    # 生成行业分析报告
                                    
                                    
                                    # 使用大模型生成行业分析
                                    try:
                                        from src.conclusion import generate_industry_analysis
                                        industry_analysis = generate_industry_analysis(industry_summary)
                                        
                                        with st.expander("行业分析报告", expanded=True):
                                            st.markdown(industry_analysis)
                                        
                                        # 导出功能
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            buf = io.StringIO()
                                            res.to_csv(buf, index=True, encoding="utf-8-sig")
                                            st.download_button("导出评分CSV", buf.getvalue(), 
                                                             file_name=f"{selected_industry}_scores.csv", 
                                                             mime="text/csv")
                                        with col2:
                                            st.download_button("导出行业分析报告", industry_analysis, 
                                                             file_name=f"{selected_industry}_analysis.txt", 
                                                             mime="text/plain")
                                        
                                        # 生成个股报告
                                        st.subheader("个股详细报告")
                                        all_txts = []
                                        for s in res.index:
                                            m = compute_symbol_metrics(funda_dir / s.replace('.', '_'))
                                            txt = generate_text_report(s, m)
                                            all_txts.append(txt)
                                            with st.expander(f"{s} 详细报告"):
                                                st.markdown(txt)
                                        
                                        if all_txts:
                                            st.download_button("导出所有报告", "\n\n".join(all_txts), 
                                                             file_name=f"{selected_industry}_reports.txt", 
                                                             mime="text/plain")
                                        
                                    except Exception as e:
                                        st.warning(f"大模型分析生成失败: {str(e)}")
                                        # 生成基本报告
                                        st.info("生成基本分析报告...")
                                        all_txts = []
                                        for s in res.index:
                                            m = compute_symbol_metrics(funda_dir / s.replace('.', '_'))
                                            txt = generate_text_report(s, m)
                                            all_txts.append(txt)
                                            with st.expander(f"{s} 报告"):
                                                st.markdown(txt)
                                        
                                        if all_txts:
                                            st.download_button("导出报告文本", "\n\n".join(all_txts), 
                                                             file_name=f"{selected_industry}_reports.txt", 
                                                             mime="text/plain")
                                else:
                                    st.warning("行业财报数据不足，请先在侧边栏执行'批量采集财报(四表)'")
                else:
                    st.warning("该行业下没有证券数据")
            else:
                st.warning("请先在侧边栏上传行业映射文件")
        else:
            st.info("请先在侧边栏上传行业映射文件以获取行业列表")
    else:
        pass
