from pathlib import Path
import numpy as np
import pandas as pd

def _read_csv(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()

def _latest_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if "end_date" in df.columns and "ann_date" in df.columns:
        try:
            return df.sort_values(["end_date", "ann_date"]).iloc[-1]
        except Exception:
            pass
    return df.iloc[-1]

def _get(row: pd.Series, keys: list[str]) -> float | None:
    for k in keys:
        if row is not None and k in row.index:
            try:
                v = float(row[k])
                if np.isfinite(v):
                    return v
            except Exception:
                continue
    return None

def compute_symbol_metrics(sym_dir: Path) -> dict:
    fi = _read_csv(sym_dir / "fina_indicator.csv")
    bs = _read_csv(sym_dir / "balancesheet.csv")
    ic = _read_csv(sym_dir / "income.csv")
    cf = _read_csv(sym_dir / "cashflow.csv")
    rfi = _latest_row(fi)
    rbs = _latest_row(bs)
    ric = _latest_row(ic)
    rcf = _latest_row(cf)
    roe = _get(rfi, ["roe"])
    eps = _get(rfi, ["basic_eps", "eps"])
    gross = _get(rfi, ["grossprofit_margin"])
    debt_ratio = _get(rfi, ["debt_to_assets"])
    if debt_ratio is None:
        ta = _get(rbs, ["total_assets"])
        tl = _get(rbs, ["total_liab", "total_liabilities"])
        debt_ratio = (tl/ta) if (tl is not None and ta) else None
    net_profit_margin = None
    n_income = _get(ric, ["n_income", "net_profit"])
    total_rev = _get(ric, ["total_revenue", "oper_rev", "revenue"])
    if n_income is not None and total_rev:
        try:
            net_profit_margin = n_income / total_rev if total_rev != 0 else None
        except Exception:
            net_profit_margin = None
    cur_assets = _get(rbs, ["total_current_assets", "current_assets", "total_cur_assets"])
    cur_liab = _get(rbs, ["total_current_liab", "current_liabilities", "total_cur_liab"])
    inventories = _get(rbs, ["inventories", "inventories_curr"])
    current_ratio = (cur_assets/cur_liab) if (cur_assets and cur_liab) else None
    quick_ratio = ((cur_assets - inventories)/cur_liab) if (cur_assets and cur_liab and inventories is not None) else None
    cash_oper = _get(rcf, ["n_cashflow_act", "net_cashflow_operate"])
    cash_flow_ratio = (cash_oper/cur_liab) if (cash_oper is not None and cur_liab) else None
    # Growth
    def _yoy_from_df(df: pd.DataFrame, col_candidates: list[str]) -> float | None:
        if df is None or df.empty:
            return None
        cols = [c for c in col_candidates if c in df.columns]
        if not cols:
            return None
        s = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
        if s.shape[0] >= 2:
            try:
                return float(s.iloc[-1]/s.iloc[-2] - 1.0)
            except Exception:
                return None
        return None
    revenue_growth = _yoy_from_df(ic, ["total_revenue", "oper_rev", "revenue"])
    if revenue_growth is None and fi is not None and not fi.empty:
        revenue_growth = _yoy_from_df(fi, ["total_revenue_yoy", "oper_rev_yoy", "revenue_yoy"])
    net_profit_growth = _yoy_from_df(ic, ["n_income", "net_profit"])
    if net_profit_growth is None and fi is not None and not fi.empty:
        net_profit_growth = _yoy_from_df(fi, ["net_profit_yoy", "n_income_yoy"])
    sales_growth = revenue_growth
    # Return
    roi = _get(rfi, ["roe"])  # proxy
    dividend_yield = _get(rfi, ["dividend_yield"])
    return {
        "roe": roe,
        "eps": eps,
        "grossprofit_margin": gross,
        "net_profit_margin": net_profit_margin,
        "debt_to_assets": debt_ratio,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "cash_flow_ratio": cash_flow_ratio,
        "revenue_growth": revenue_growth,
        "net_profit_growth": net_profit_growth,
        "sales_growth": sales_growth,
        "roi": roi,
        "dividend_yield": dividend_yield,
    }

def compute_industry_scoring(stock_map_df: pd.DataFrame, industry: str, fundamentals_dir: Path, symbols: list[str] | None, weights: dict) -> pd.DataFrame:
    df_ind = stock_map_df[stock_map_df["industry"] == industry]
    syms = []
    if symbols:
        syms = symbols
    else:
        if "symbol" in df_ind.columns:
            syms = df_ind["symbol"].dropna().astype(str).tolist()
    rows = []
    for s in syms:
        m = compute_symbol_metrics(fundamentals_dir / s.replace('.', '_'))
        rows.append({"symbol": s, **m})
    F = pd.DataFrame(rows).set_index("symbol")
    if F.empty:
        return F
    def _score_cols(cols: list[str], sign: int=1, allow_negative_fallback: bool=False) -> pd.Series:
        vals = F[cols].apply(pd.to_numeric, errors="coerce")
        stds = vals.std(ddof=0)
        z = (vals - vals.mean())/stds.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        if z.isna().all().all():
            row_mean = vals.mean(axis=1)
            rng = row_mean.max() - row_mean.min()
            scaled = (row_mean - row_mean.min())/rng if rng and rng != 0 else pd.Series(0.0, index=vals.index)
            if allow_negative_fallback:
                centered = (scaled * 2.0) - 1.0
                return (centered * sign)
            else:
                return (scaled * sign)
        z = z.fillna(0.0)
        return (z.mean(axis=1) * sign)
    profitability = _score_cols(["roe", "eps", "grossprofit_margin", "net_profit_margin"], sign=1)
    solvency = _score_cols(["debt_to_assets", "current_ratio", "quick_ratio", "cash_flow_ratio"], sign=1)
    solvency = solvency - pd.to_numeric(F["debt_to_assets"], errors="coerce").fillna(0.0)  # penalize higher debt
    growth = _score_cols(["revenue_growth", "net_profit_growth", "sales_growth"], sign=1, allow_negative_fallback=True)
    return_score = _score_cols(["roi", "dividend_yield"], sign=1)
    Wp = float(weights.get("profitability", 0.4))
    Ws = float(weights.get("solvency", 0.3))
    Wg = float(weights.get("growth", 0.2))
    Wr = float(weights.get("return", 0.1))
    composite = Wp*profitability + Ws*solvency + Wg*growth + Wr*return_score
    out = F.copy()
    out["score_profitability"] = profitability
    out["score_solvency"] = solvency
    out["score_growth"] = growth
    out["score_return"] = return_score
    out["composite_score"] = composite
    out = out.sort_values("composite_score", ascending=False)
    return out

def generate_text_report(ts_code: str, metrics: dict) -> str:
    roe = metrics.get("roe"); debt = metrics.get("debt_to_assets"); npm = metrics.get("net_profit_margin"); cfp = metrics.get("cash_flow_ratio")
    tone = "中性偏谨慎"
    if roe and roe > 0.1 and (debt is None or debt < 0.6):
        tone = "稳健偏多"
    lines = []
    lines.append("一、核心结论（Executive Summary）")
    lines.append(f"在当前基本面条件下，{ts_code} 整体表现为{tone}，更适合中低风险偏好投资者。")
    lines.append("")
    lines.append("二、关键数据与因子信号概览（What I See）")
    lines.append(f"- 盈利能力: ROE 为 {roe:.2f}，表现稳定" if roe is not None else "- 盈利能力: ROE 数据缺失")
    lines.append(f"- 财务结构: 负债率 {debt:.2f}，短期偿债压力适中" if debt is not None else "- 财务结构: 负债率数据缺失")
    lines.append(f"- 盈利水平: 净利率 {npm:.2f}，盈利能力稳健" if npm is not None else "- 盈利水平: 净利率数据缺失")
    pos_cash = cfp is not None and cfp > 0
    lines.append(f"- 现金流: {'为正' if pos_cash else '为负' if cfp is not None else '数据缺失'}，需关注波动")
    lines.append("")
    lines.append("三、综合逻辑判断（How I Think）")
    lines.append("该公司盈利能力稳定，成长性适中，现金流和财务结构健康。整体偏防守型，短期不宜激进操作。")
    lines.append("")
    lines.append("四、主要风险提示（Risk First）")
    lines.append("- 盈利下行风险：若行业需求减弱，利润可能下滑")
    lines.append("- 估值风险：低估值可能源于长期成长性不足")
    lines.append("- 宏观政策风险：利率、政策变化可能放大波动")
    lines.append("")
    lines.append("五、策略建议（Actionable but Optional）")
    lines.append("- 稳健型投资者: 低仓位持有")
    lines.append("- 进取型投资者: 关注盈利改善信号再介入")
    lines.append("- 已持有者: 继续观察，不宜加仓")
    lines.append("")
    lines.append("六、适用期限与复盘触发条件（Time & Trigger）")
    lines.append("- 适用期限：中短期（1-3个月）")
    lines.append("- 复盘条件：新财报披露、行业景气度显著变化、股价异常波动")
    lines.append("")
    lines.append("七、模型说明与免责声明（Professional Touch）")
    lines.append("基于历史公开数据与量化因子生成，仅供研究与参考，不构成投资承诺。")
    return "\n".join(lines)
