import os
import json
import pandas as pd
from typing import List, Dict, Any
from tools.llm_conclusion import generate_conclusions, generate_summary_text
from tools.llm_conclusion import generate_module_advice_human
from tools.llm_conclusion import generate_module_followup

def build_facts(df_symbol: pd.DataFrame, indicators: pd.DataFrame, corr: pd.DataFrame, explained, symbol: str, industry: str) -> Dict[str, Any]:
    facts = {
        "symbol": symbol,
        "industry": industry,
        "last_close": float(df_symbol["close"].iloc[-1]),
        "ma_slope": float(indicators["SMA"].diff().iloc[-1]) if "SMA" in indicators.columns else None,
        "ema_slope": float(indicators["EMA"].diff().iloc[-1]) if "EMA" in indicators.columns else None,
    }
    
    # Add recent price history for K-line analysis
    try:
        recent = df_symbol.tail(5).copy()
        if "date" in recent.columns:
            recent["date"] = recent["date"].astype(str)
        facts["recent_ohlcv"] = recent.to_dict(orient="records")
        
        # Calculate recent high/low for support/resistance context
        last_20 = df_symbol.tail(20)
        facts["recent_20d_high"] = float(last_20["high"].max())
        facts["recent_20d_low"] = float(last_20["low"].min())
    except Exception:
        pass

    if corr is not None:
        facts["industry_corr_mean"] = float(corr.mean().mean())
    if explained is not None:
        facts["pca_first_var"] = float(pd.Series(explained).iloc[0])
    try:
        facts["rsi_last"] = float(indicators["RSI"].iloc[-1]) if "RSI" in indicators.columns else None
    except Exception:
        facts["rsi_last"] = None
    return facts

def generate_conclusions_with_llm(facts: Dict[str, Any], chart_paths: List[str]) -> List[Dict[str, Any]]:
    endpoint = os.environ.get("LLM_ENDPOINT")
    api_key = os.environ.get("LLM_API_KEY")
    if not endpoint:
        return _rule_based_conclusions(facts, chart_paths)
    try:
        return generate_conclusions(facts, chart_paths, endpoint, api_key)
    except Exception:
        return _rule_based_conclusions(facts, chart_paths)

def _mk_item(i, title, summary, method, confidence, facts, chart_paths, advice):
    return {
        "id": f"C{i}",
        "title": title,
        "summary": summary,
        "method": method,
        "confidence": confidence,
        "metrics": facts,
        "evidence_refs": [{"type": "chart", "path": p} for p in chart_paths],
        "advice": advice,
        "risk_notes": ["需结合更多维度数据进行确认"]
    }

def _rule_based_conclusions(facts: Dict[str, Any], chart_paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    i = 1
    ma = facts.get("ma_slope")
    ema = facts.get("ema_slope")
    rsi = facts.get("rsi_last")
    corrm = facts.get("industry_corr_mean")
    pca1 = facts.get("pca_first_var")
    trend_up = (ma is not None and ma > 0) or (ema is not None and ema > 0)
    trend_down = (ma is not None and ma < 0) and (ema is not None and ema < 0)
    if trend_up:
        out.append(_mk_item(i, "趋势向上", "均线斜率为正，价格处于上行趋势", "trend", 0.7, facts, chart_paths, "关注回调买入"))
    else:
        out.append(_mk_item(i, "趋势不强", "均线斜率不显著为正，需等待趋势确认", "trend", 0.5, facts, chart_paths, "观望"))
    i += 1
    if trend_down:
        out.append(_mk_item(i, "趋势向下", "均线斜率为负，价格处于下降趋势", "trend", 0.7, facts, chart_paths, "谨慎，避免追高"))
    else:
        out.append(_mk_item(i, "趋势未见明显下行", "均线斜率未显著为负", "trend", 0.5, facts, chart_paths, "保持观察"))
    i += 1
    if rsi is not None and rsi < 30:
        out.append(_mk_item(i, "动量偏弱但或有超卖", "RSI<30，存在技术性反弹可能", "momentum", 0.6, facts, chart_paths, "关注反弹信号"))
    elif rsi is not None and rsi > 70:
        out.append(_mk_item(i, "动量偏强或有超买", "RSI>70，短期或有回调压力", "momentum", 0.6, facts, chart_paths, "分批减仓"))
    else:
        out.append(_mk_item(i, "动量中性", "RSI位于中性区域", "momentum", 0.5, facts, chart_paths, "观望"))
    i += 1
    if corrm is not None:
        out.append(_mk_item(i, "行业联动性", f"行业相关性均值为{corrm:.2f}", "correlation", 0.5, facts, chart_paths, "注意板块共振风险"))
    else:
        out.append(_mk_item(i, "行业联动信息缺失", "未加载到行业相关性数据", "correlation", 0.4, facts, chart_paths, "以个股为主"))
    i += 1
    if pca1 is not None:
        out.append(_mk_item(i, "市场主因子驱动", f"PCA第一因子解释率约{pca1:.2f}", "pca", 0.5, facts, chart_paths, "结合因子暴露评估"))
    else:
        out.append(_mk_item(i, "主因子信息缺失", "未加载到PCA分解", "pca", 0.4, facts, chart_paths, "补充数据后再评估"))
    i += 1
    out.append(_mk_item(i, "风险控制建议", "结合波动与趋势，严格设置止损与仓位控制", "risk", 0.6, facts, chart_paths, "控制仓位与止损"))
    i += 1
    out.append(_mk_item(i, "分批参与策略", "趋势初步形成时可分批试探性参与", "strategy", 0.6, facts, chart_paths, "分批买入"))
    i += 1
    out.append(_mk_item(i, "回撤与耐心", "若回撤未破关键支撑，可耐心等待上行确认", "risk", 0.5, facts, chart_paths, "保持耐心"))
    i += 1
    out.append(_mk_item(i, "事件与基本面跟踪", "结合财报与行业事件，避免信息缺口", "fundamental", 0.6, facts, chart_paths, "持续跟踪"))
    i += 1
    out.append(_mk_item(i, "综合建议", "当前信号综合评估后给出操作提议", "summary", 0.6, facts, chart_paths, "结合风险偏好执行"))
    return out

def generate_conclusion_text(facts: Dict[str, Any], chart_paths: List[str]) -> str:
    endpoint = os.environ.get("LLM_ENDPOINT")
    api_key = os.environ.get("LLM_API_KEY")
    if not endpoint:
        return _rule_based_text(facts, chart_paths)
    try:
        return generate_summary_text(facts, chart_paths, endpoint, api_key)
    except Exception:
        return _rule_based_text(facts, chart_paths)

def generate_module_advice_text(module: str, inputs: Dict[str, Any], facts: Dict[str, Any]) -> str:
    endpoint = os.environ.get("LLM_ENDPOINT")
    api_key = os.environ.get("LLM_API_KEY")
    if not endpoint:
        return _rule_based_module_text(module, inputs, facts)
    try:
        return generate_module_advice_human(facts, module, inputs, endpoint, api_key)
    except Exception:
        return _rule_based_module_text(module, inputs, facts)

def generate_module_followup_text(module: str, inputs: Dict[str, Any], facts: Dict[str, Any], advisor_text: str, question: str) -> str:
    endpoint = os.environ.get("LLM_ENDPOINT")
    api_key = os.environ.get("LLM_API_KEY")
    if not endpoint:
        return _rule_based_module_followup(module, inputs, facts, advisor_text, question)
    try:
        return generate_module_followup(facts, module, inputs, advisor_text, question, endpoint, api_key)
    except Exception:
        return _rule_based_module_followup(module, inputs, facts, advisor_text, question)
def _rule_based_text(facts: Dict[str, Any], chart_paths: List[str]) -> str:
    lines = []
    lines.append(f"证券：{facts.get('symbol','')} 行业：{facts.get('industry','')}")
    lines.append(f"最新收盘：{facts.get('last_close','')}；SMA斜率：{facts.get('ma_slope','')}；EMA斜率：{facts.get('ema_slope','')}")
    if 'industry_corr_mean' in facts:
        lines.append(f"行业相关性均值：{facts.get('industry_corr_mean')}")
    if 'pca_first_var' in facts:
        lines.append(f"PCA第一因子解释率：{facts.get('pca_first_var')}")
    for i in range(1, 11):
        lines.append(f"结论{i}：基于当前技术与行业指标的综合判断，请结合图表与数据进行验证。")
    if chart_paths:
        lines.append("图表：")
        for p in chart_paths:
            lines.append(f"- {p}")
    lines.append("风险提示：仅基于局部事实，需结合更全面数据；市场存在样本外风险。")
    return "\n".join(lines)

def _rule_based_module_text(module: str, inputs: Dict[str, Any], facts: Dict[str, Any]) -> str:
    s = facts.get("symbol", "")
    ind = facts.get("industry", "")
    rng = inputs.get("time_range") or ""
    lines = []
    if module == "K线与指标":
        ma = facts.get("ma_slope"); ema = facts.get("ema_slope")
        dir_txt = "中性偏谨慎" if not ((ma and ma > 0) or (ema and ema > 0)) else "偏多但需谨慎"
        lines.append(f"结论：在{rng}区间，{s}（{ind}）走势{dir_txt}。")
        lines.append("建议：轻仓观察，涨幅伴随放量且站稳关键均线时再跟随；任何仓位都配止损。")
    elif module == "相关性分析":
        m = facts.get("industry_corr_mean"); tag = "联动性较高，分散效果有限" if (m and m >= 0.6) else "联动性中等或偏低"
        lines.append(f"结论：板块{tag}。")
        lines.append("建议：降低总仓位，避免同类标的集中持有，优先选择关联度低的补充配置。")
    elif module == "PCA分析":
        p1 = facts.get("pca_first_var"); tag = "少数因子主导，市场更一边倒" if (p1 and p1 >= 0.4) else "因子分散，结构较均衡"
        lines.append(f"结论：{tag}。")
        lines.append("建议：避免单一风格重仓，分散到不同风格与行业，降低系统性风险。")
    elif module == "波动性分析":
        lines.append("结论：波动性偏高，回撤风险增加。")
        lines.append("建议：缩短持仓周期，减小仓位；只在回撤受控且趋势改善时增加暴露。")
    elif module == "季节性分析":
        lines.append("结论：存在可识别的节奏与周期。")
        lines.append("建议：按节奏分批布局，避开随机波动增强阶段的激进操作。")
    elif module == "风险-收益聚类分析":
        lines.append("结论：不同标的风险特征差异显著。")
        lines.append("建议：稳健型优先低波动簇，进取型可关注高回报簇但必须配套风险控制。")
    elif module == "基本面因子暴露分析":
        lines.append("结论：基本面因子能部分解释收益，但并非确定性。")
        lines.append("建议：结合盈利质量、现金流与负债结构，多维验证后再做加仓决策。")
    elif module == "涨跌概率分析":
        lines.append("结论：概率倾向仅作参考，不等同确定性。")
        lines.append("建议：按概率优势优化仓位结构，但每笔交易都配止损与退出规则。")
    else:
        lines.append("结论：当前信号中性偏谨慎，更适合稳健型。")
    lines.append("建议（可执行但非承诺）：")
    lines.append("稳健型：仓位10%-30%，止损3%-5%；确认基本面改善或趋势共振后再提高暴露。")
    lines.append("进取型：仓位30%-50%，止损5%-8%；仅在量价配合且回撤受控时加仓。")
    lines.append("已持有者：以风险预算为先，突破关键位或基本面改善再考虑加仓。")
    return "\n".join(lines)

def _rule_based_module_followup(module: str, inputs: Dict[str, Any], facts: Dict[str, Any], advisor_text: str, question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "请明确你的问题，例如“是否适合加仓？”或“当前风险主要来自哪里？”"
    s = facts.get("symbol",""); ind = facts.get("industry","")
    rng = inputs.get("time_range","")
    last = facts.get("last_close")
    ma = facts.get("ma_slope"); ema = facts.get("ema_slope"); rsi = facts.get("rsi_last")
    corrm = facts.get("industry_corr_mean"); pca1 = facts.get("pca_first_var")
    hi20 = facts.get("recent_20d_high"); lo20 = facts.get("recent_20d_low")
    def _yn(b): return "是" if b else "否"
    def _fmt(x, pct=False):
        if x is None: return "N/A"
        try:
            return f"{x:.2%}" if pct else f"{x:.2f}"
        except Exception:
            return str(x)
    ql = q.lower()
    ans = []
    ans.append(f"{s}（{ind}）| 区间：{rng} | 问题：{q}")
    if "加仓" in q or "买入" in q or "增持" in q:
        trend_up = ((ma or 0) > 0) or ((ema or 0) > 0)
        ok = trend_up and (rsi is None or rsi <= 70)
        ans.append(f"结论：{_yn(ok)}，更偏向在趋势转正且未显著超买时分批加仓。")
        ans.append(f"依据：SMA斜率={_fmt(ma)}，EMA斜率={_fmt(ema)}，RSI={_fmt(rsi)}；支撑位≈{_fmt(lo20)}，压力位≈{_fmt(hi20)}，最新收盘={_fmt(last)}。")
        ans.append("建议：若收盘价站稳支撑并突破关键均线，少量试探；任意加仓均配3%-5%止损。")
    elif "减仓" in q or "卖出" in q or "止盈" in q:
        overbought = (rsi is not None and rsi >= 70)
        trend_down = ((ma or 0) < 0) and ((ema or 0) < 0)
        need_trim = overbought or trend_down
        ans.append(f"结论：{_yn(need_trim)}，在超买或趋势转负时优先减仓。")
        ans.append(f"依据：RSI={_fmt(rsi)}，SMA斜率={_fmt(ma)}，EMA斜率={_fmt(ema)}；压力位≈{_fmt(hi20)}。")
        ans.append("建议：分批减仓，跌破支撑位或关键均线时加速退出。")
    elif "风险" in q or "波动" in q or "下行" in q:
        ans.append("结论：风险主要来自趋势不稳与行业联动性。")
        ans.append(f"依据：SMA斜率={_fmt(ma)}，EMA斜率={_fmt(ema)}；行业相关性均值={_fmt(corrm)}；主因子解释率={_fmt(pca1)}。")
        ans.append("建议：提高止损纪律，降低同质化持仓，引入低相关板块分散。")
    elif "支撑" in q or "压力" in q or "关键位" in q:
        ans.append(f"结论：近期支撑≈{_fmt(lo20)}，压力≈{_fmt(hi20)}，最新收盘={_fmt(last)}。")
        ans.append("建议：靠近支撑观察反弹信号；突破压力并放量时再关注跟随。")
    elif "相关性" in q or "分散" in q or "联动" in q:
        ans.append(f"结论：行业平均相关性≈{_fmt(corrm)}。相关性高时分散效果弱。")
        ans.append("建议：补充低相关或负相关行业的配置以降低组合波动。")
    elif "pca" in q.lower() or "因子" in q or "解释率" in q:
        ans.append(f"结论：第一主成分解释率≈{_fmt(pca1)}。解释率高意味风格集中度提升。")
        ans.append("建议：避免单一风格重仓，拉平行业与风格暴露。")
    elif "概率" in q or "上涨" in q or "下跌" in q:
        ans.append("结论：概率倾向需结合样本与AUC评估，适合用于排序与权重微调。")
        ans.append("建议：仅在概率优势与基本面改善共振时提高暴露；始终配止损。")
    else:
        ans.append("结论：当前以趋势与位置为主，结合基本面与行业环境综合判断。")
        ans.append(f"依据：SMA斜率={_fmt(ma)}，EMA斜率={_fmt(ema)}，RSI={_fmt(rsi)}，支撑≈{_fmt(lo20)}，压力≈{_fmt(hi20)}，相关性均值≈{_fmt(corrm)}。")
        ans.append("建议：分批操作、控制仓位与止损，等待明确信号再调整。")
    return "\n".join(ans)
def generate_industry_analysis(summary) -> str:
    ind = str(summary.get("industry", ""))
    syms = summary.get("selected_stocks") or []
    scores = summary.get("scores") or {}
    baseline = summary.get("baseline") or {}
    core = summary.get("core_metrics") or {}

    items = []
    # 生成每只股票的综合评分和解读
    for s in syms:
        row = scores.get(s) or {}
        sc = float(row.get("composite_score", 0.0))
        core_row = core.get(s) or {}
        dd = core_row.get("最大回撤")
        shp = core_row.get("夏普比率")
        vol = core_row.get("波动率")
        # 默认解读
        comment = "成长潜力可关注"
        try:
            if shp is not None and dd is not None:
                if abs(dd) <= 0.15 and shp >= 1.0:
                    comment = "盈利稳定、回撤相对小"
                elif abs(dd) <= 0.20 and shp >= 0.8:
                    comment = "性价比较优，风险适中"
            elif vol is not None and baseline.get("avg_volatility") is not None:
                if vol <= float(baseline.get("avg_volatility")) and (shp or 0.0) >= 0.7:
                    comment = "波动相对可控，稳健偏好可关注"
        except Exception:
            pass
        items.append({"symbol": s, "score": sc, "drawdown": dd, "comment": comment})

    # 按综合评分排序
    items = sorted(items, key=lambda x: x["score"], reverse=True)

    # 缩放分数到 10 分制
    sc_vals = [it["score"] for it in items] if items else [0.0]
    sc_min, sc_max = min(sc_vals), max(sc_vals)

    def _scale10(x: float) -> float:
        return 7.0 if sc_max == sc_min else (x - sc_min) / (sc_max - sc_min) * 10.0

    # 行业阶段与波动描述
    phase = str(baseline.get("mood") or "震荡")
    avg_vol = float(baseline.get("avg_volatility") or 0.0)
    vol_desc = "幅度有限" if avg_vol < 0.20 else "中等波动" if avg_vol < 0.35 else "波动偏大"

    # 生成报告
    def add_section(title: str, content: List[str]) -> List[str]:
        return [title] + [""] + content + [""]

    lines = []
    lines.append(f"小金 · 行业投资结论（{ind}）")
    # 核心判断
    lines += add_section("一、核心判断", [
        f"当前 {ind} 行业短期走势 {phase}，涨跌幅 {vol_desc}，",
        "建议优先关注综合评分靠前的标的，采取分批参与策略，并以风险预算为先。"
    ])
    # 优选标的
    lines += add_section("二、优选标的（按综合评分排序）", [
        "证券代码\t阶段性表现\t综合评分\t小金一句话解读"
    ] + [
        f"{it['symbol']}\t{(it['drawdown']*100 if it['drawdown'] is not None else 'N/A'):.1f}%\t{_scale10(it['score']):.1f}/10\t{it['comment']}"
        for it in items
    ] + ["建议避免一次性重仓，可分批布局。"])

    # 模型依据
    lines += add_section("三、模型依据", [
        "本次筛选基于以下核心维度的综合评估：",
        "盈利质量：赚钱能力是否稳定",
        "偿债结构：财务抗风险能力",
        "成长动能：未来业绩增长潜力",
        "投资回报：股东回报水平",
        "综合评分越高，代表其在行业内盈利能力、稳定性与成长性更均衡。"
    ])

    # 风险提示
    risks = summary.get("risk_factors") or [
        "行业景气度波动可能导致股价短期回落",
        "盈利不及预期可能拖累投资收益",
        "流动性风险可能影响买卖操作",
        "政策变化可能带来行业调整"
    ]
    lines += add_section("四、主要风险提示", risks + ["行业波动或外部环境变化，均可能对短期表现造成影响。"])

    # 操作建议
    lines += add_section("五、操作建议（按风险偏好）", [
        "投资者类型\t建议仓位\t止损区间\t加仓条件",
        "稳健型 🛡\t10%-30%\t3%-5%\t盈利与现金流改善，估值合理时分批加仓",
        "进取型 🚀\t30%-50%\t5%-8%\t量价配合良好，基本面改善时分批加仓",
        "已持有 📊\t动态调整\t按风险预算\t关键位突破或基本面改善再加仓"
    ])

    # 复盘触发条件
    triggers = summary.get("review_triggers") or ["定期财报披露", "行业重大事件", "关键技术位突破或失守"]
    lines += add_section("六、复盘与再评估触发条件", ["建议在以下情况出现时，对持仓进行复盘："] + triggers)

    # 小金总结
    lines += add_section("七、小金一句话总结", [
        str(summary.get("summary_sentence") or "这不是拼短期博弈的行业，而是一个讲节奏、讲纪律、讲风险控制的配置方向。")
    ])

    return "\n".join(lines)
