import os
from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.config import load_local_env
from src.mapping import process_stock_map, resolve_security
from src.data_io import ts_to_baostock, fetch_kline_baostock, export_financials_batch, validate_tushare_token
from src.analysis.timeseries import compute_indicators, compute_adx
from src.visualization import plot_candlestick_with_indicators
from src.conclusion import build_facts, generate_conclusions_with_llm, generate_conclusion_text, generate_module_advice_text, generate_module_followup_text
from src.report import summarize_fundamentals, render_industry_report, render_single_report
from src.fundamentals import compute_industry_scoring, compute_symbol_metrics, generate_text_report

ROOT = Path(__file__).parent
EXPORT_DIR = ROOT / "export"
DATA_DIR = ROOT / "data"
load_local_env()
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(ROOT / "web" / "static")), name="static")
STATE = {"stock_map": None}

@app.get("/", response_class=HTMLResponse)
def index():
    fp = ROOT / "web" / "index.html"
    return fp.read_text(encoding="utf-8")

@app.post("/upload-mapping")
async def upload_mapping(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception:
            df = pd.read_csv(pd.io.common.BytesIO(content), encoding="gbk")
        STATE["stock_map"] = process_stock_map(df)
        return {"ok": True, "rows": int(STATE["stock_map"].shape[0])}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.get("/resolve")
def resolve(query: str = Query(...)):
    df_map = STATE.get("stock_map")
    res = resolve_security(df_map, query)
    if not res:
        return JSONResponse({"ok": False, "error": "未识别"}, status_code=404)
    symbol, industry, name = res
    return {"ok": True, "symbol": symbol, "industry": industry, "name": name}

@app.get("/ohlcv")
def ohlcv(symbol: str, start: str, end: str, freq: str = "d"):
    bs_code = ts_to_baostock(symbol)
    df = fetch_kline_baostock(bs_code, start, end, freq)
    if df.empty:
        return JSONResponse({"ok": False, "error": "无数据"}, status_code=404)
    df["symbol"] = symbol
    return {"ok": True, "rows": len(df), "data": df.to_dict(orient="records")}

@app.get("/single/analysis")
def single_analysis(symbol: str, industry: str, start: str, end: str, freq: str = "d", ma_short: int = 20, ma_long: int = 60):
    bs_code = ts_to_baostock(symbol)
    df = fetch_kline_baostock(bs_code, start, end, freq)
    if df.empty:
        return JSONResponse({"ok": False, "error": "无数据"}, status_code=404)
    df["symbol"] = symbol
    df["industry"] = industry
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    ind = compute_indicators(df.set_index("date"), ma_short=ma_short, ma_long=ma_long)
    fig_k = plot_candlestick_with_indicators(df, date_col="date", indicators=ind, time_span="自定义", show_text=False)
    adx = compute_adx(df)
    facts = build_facts(df[df["symbol"] == symbol], ind, None, None, symbol, industry)
    conclusions = generate_conclusions_with_llm(facts, [])
    txt = generate_conclusion_text(facts, [])
    out_json = EXPORT_DIR / f"{symbol}_{industry}_conclusions.json"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(conclusions, f, ensure_ascii=False, indent=2)
    out_txt = EXPORT_DIR / f"{symbol}_{industry}_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    funda_dir_default = str((ROOT / "data" / "fundamentals").resolve())
    out_fp = render_single_report(symbol, industry, Path(funda_dir_default), EXPORT_DIR, [], conclusions, facts)
    advisor_k = generate_module_advice_text("K线与指标", {"time_range": "自定义", "stock_name": "", "stock_code": symbol, "industry": industry, "module_data_summary": ""}, facts)
    return {"ok": True, "fig_k": json.loads(fig_k.to_json()), "adx_last": float(adx.iloc[-1]) if not adx.empty else None, "advisor": advisor_k, "facts": facts, "report_path": str(out_fp)}

@app.post("/advisor/followup")
def advisor_followup(module: str = Form(...), symbol: str = Form(...), industry: str = Form(...), summary: str = Form(""), question: str = Form(...)):
    facts = {"symbol": symbol, "industry": industry}
    txt = generate_module_followup_text(module, {"time_range": "自定义", "stock_code": symbol, "industry": industry, "module_data_summary": summary}, facts, "", question)
    return {"ok": True, "reply": txt}

@app.post("/industry/score")
async def industry_score(industry: str = Form(...), symbols_json: str = Form(...), weights_json: str = Form(...)):
    df_map = STATE.get("stock_map")
    try:
        symbols = json.loads(symbols_json)
        weights = json.loads(weights_json)
    except Exception:
        symbols = []
        weights = {"profitability": 0.4, "solvency": 0.3, "growth": 0.2, "return": 0.1}
    funda_dir = Path(ROOT / "data" / "fundamentals")
    res = compute_industry_scoring(df_map if df_map is not None else pd.DataFrame(columns=["symbol","industry"]), industry, funda_dir, symbols, weights)
    if res is None or res.empty:
        return JSONResponse({"ok": False, "error": "无评分数据"}, status_code=404)
    top_reports = []
    for s in res.index[:3]:
        m = compute_symbol_metrics(funda_dir / s.replace('.', '_'))
        txt = generate_text_report(s, m)
        top_reports.append({"symbol": s, "text": txt})
    return {"ok": True, "rows": res.shape[0], "score": res.reset_index().to_dict(orient="records"), "reports": top_reports}

@app.post("/financials/batch")
async def financials_batch(industry: str = Form(...), start_date: str = Form(...), end_date: str = Form(...)):
    df_map = STATE.get("stock_map")
    ok = validate_tushare_token()
    if not ok:
        return JSONResponse({"ok": False, "error": "Token无效"}, status_code=400)
    if df_map is None or df_map.empty:
        return JSONResponse({"ok": False, "error": "缺少映射"}, status_code=400)
    try:
        export_financials_batch(df_map, industry, start_date.replace("-",""), end_date.replace("-",""), ROOT / "data" / "fundamentals")
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/industry/report")
def industry_report(industry: str):
    charts = []
    ind_dir = EXPORT_DIR / industry
    if ind_dir.exists():
        for fp in ind_dir.glob("*.png"):
            charts.append(str(fp))
    conclusions_paths = []
    for fp in EXPORT_DIR.glob(f"*_{industry}_conclusions.json"):
        conclusions_paths.append(str(fp))
    symbols = []
    df_map = STATE.get("stock_map")
    if df_map is not None:
        df_ind_map = df_map[df_map["industry"] == industry]
        symbols = df_ind_map["symbol"].dropna().astype(str).tolist() if "symbol" in df_ind_map.columns else []
    out_fp = render_industry_report(industry, symbols, ROOT / "data" / "fundamentals", EXPORT_DIR, charts, conclusions_paths)
    html = Path(out_fp).read_text(encoding="utf-8")
    return HTMLResponse(html)
