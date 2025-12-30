import os
import json
from typing import List, Dict, Any
import requests
import math

def generate_conclusions(facts: Dict[str, Any], chart_paths: List[str], endpoint: str, api_key: str=None) -> List[Dict[str, Any]]:
    system = (
        "你是金融数据分析报告的专业写作者。仅基于给定事实与图表引用生成结论；"
        "每条结论包含title, summary, method, confidence, evidence_refs, metrics, scope, risk_notes；"
        "不得引入未提供的信息；如证据不足需明确说明。"
    )
    user = {
        "instruction": (
            "生成10条具有投资指导意义的分析结论。其中必须包含关于'K线形态与趋势'的详细分析，需涵盖以下要点：\n"
            "1. 当前趋势状态（上升 / 下降 / 震荡）\n"
            "2. 最近K线形态特征（实体大小、影线长度）\n"
            "3. 当前价格所处位置（支撑位 / 压力位 / 区间中部）\n"
            "4. 结合普通投资者视角，给出投资含义与风险提示\n\n"
            "整体要求：\n"
            "- 语言通俗易懂，不使用复杂公式\n"
            "- 不给绝对的确定性承诺（如'必涨'、'保证盈利'）\n"
            "- 覆盖不同分析方法，避免重复\n"
            "- 中文输出"
        ),
        "facts": facts,
        "chart_paths": chart_paths
    }
    payload = {
        "model": os.environ.get("LLM_MODEL", "deepseek-reasoner"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        "temperature": 0.2
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        conclusions = json.loads(content)
    except Exception:
        conclusions = []
    return conclusions

def generate_summary_text(facts: Dict[str, Any], chart_paths: List[str], endpoint: str, api_key: str=None) -> str:
    system = (
        "你是金融数据分析报告的专业写作者。根据提供的事实与图表引用，生成中文的十条投资分析结论与总述，"
        "要求结构清晰、可读性强，适合人阅读的纯文本格式；不要输出JSON或代码块。"
    )
    user = {
        "instruction": (
            "请基于事实生成十条有投资指导意义的结论，包含标题与简要解释，并在末尾给出总体建议与风险提示。\n"
            "特别注意：在分析K线与趋势时，请明确指出当前趋势状态、K线形态特征（实体/影线）、价格位置（支撑/压力），"
            "并给出通俗的投资含义，不使用公式，不给确定性承诺。"
        ),
        "facts": facts,
        "chart_paths": chart_paths
    }
    payload = {
        "model": os.environ.get("LLM_MODEL", "deepseek-reasoner"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        "temperature": 0.2
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content

def _module_instruction(module: str) -> str:
    """
    改造为小金风格人话输出，同时结合最终模板：
    每条结论直接生成完整报告结构：核心结论、关键数据、综合逻辑、风险提示、策略建议、适用期限、免责声明。
    """
    base_instruction = (
        "你是“小金”，理性稳健的智能证券分析顾问。"
        "目标：帮助投资者理解数据、趋势、风险和决策逻辑，输出完整中文投资分析报告。"
        "严格按照以下模板输出：\n"
        "一、核心结论（Executive Summary｜必须放第一）\n"
        "一句话结论 + 风险偏好前提\n"
        "二、关键数据与因子信号概览（What I See）\n"
        "用 bullet point，简明具体\n"
        "三、综合逻辑判断（How I Think）\n"
        "说明原因和逻辑闭环，不预测未来\n"
        "四、主要风险提示（Risk First）\n"
        "列出潜在风险，不可省略\n"
        "五、策略建议（Actionable but Optional）\n"
        "提供投资选项，不给买卖指令\n"
        "六、适用期限与复盘触发条件（Time & Trigger）\n"
        "说明何时需重新评估\n"
        "七、模型说明与免责声明（Professional Touch）\n"
        "说明数据来源与合规免责声明\n\n"
    )
    if module == "K线与指标":
        return base_instruction + (
            "重点：价格在趋势中的位置、K线实体与影线特征、均线支撑压力、短中长期走势一致性。"
            "用通俗语言解释趋势含义和风险，不给买卖点。"
        )
    if module == "相关性分析":
        return base_instruction + (
            "分析行业或资产间相关性。说明高相关、分散风险意义、同涨同跌风险，"
            "用投资者易懂的语言说明相关性含义。"
        )
    if module == "PCA分析":
        return base_instruction + (
            "分析PCA碎石图。说明主导因子、数据复杂度，对风险管理和组合稳定性的启示。"
        )
    if module == "波动性分析":
        return base_instruction + (
            "分析GARCH波动。判断波动水平、聚集性，并解释对持仓风险的含义。"
        )
    if module == "季节性分析":
        return base_instruction + (
            "分析STL分解与ACF/PACF。说明趋势、周期性、自相关性，解释统计规律对操作的参考意义。"
        )
    if module == "风险-收益聚类分析":
        return base_instruction + (
            "分析风险-收益聚类。说明不同聚类特征、当前资产类别，给出适合投资风格的解读。"
        )
    if module == "基本面因子暴露分析":
        return base_instruction + (
            "分析基本面因子暴露。解释主要因子、价值/成长/周期特征、潜在风险，用人话说明投资含义。"
        )
    if module == "涨跌概率分析":
        return base_instruction + (
            "分析涨跌概率。说明概率倾向、不确定性，强调概率高不等于确定性。"
        )
    return base_instruction + "请结合数据进行理性分析，生成完整投资分析报告。"


def _clip_text(s: str, limit: int = 800) -> str:
    if s is None:
        return ""
    ss = str(s)
    return ss[:limit] if len(ss) > limit else ss

def generate_module_advice_human(facts: Dict[str, Any], module: str, inputs: Dict[str, Any], endpoint: str, api_key: str=None) -> str:
    system = (
        "你是“小金”，一位资深、审慎、以风险为先的金融分析师。"
        "目标：面向投资者输出更权威、更准确、更可执行的建议。"
        "要求：基于给定事实与模块数据摘要，先给结论，再给依据与风险，再给可执行建议。"
        "建议需可操作但不做确定性承诺，明确仓位、止损与复盘触发条件，风格专业、克制。"
    )

    module_instruction = _module_instruction(module)
    summary = _clip_text(inputs.get("module_data_summary", ""), 1000)

    user_text = (
        f"模块：{module}\n"
        f"证券：{inputs.get('stock_name','')}（{inputs.get('stock_code','')}），行业：{inputs.get('industry','')}\n"
        f"区间：{inputs.get('time_range','')}\n"
        f"数据摘要：{summary}\n"
        f"{module_instruction}\n"
        "输出要求（中文）：\n"
        "1）核心结论：一句话给出偏多/偏空/中性判断，明确风险偏好适配。\n"
        "2）关键依据：列出3-8条核心指标或现象，具体、可验证。\n"
        "3）风险：盈利、估值、政策/流动性三类至少各给一条。\n"
        "4）可执行建议：\n"
        "   - 稳健型：建议仓位比例与止损区间（例如 10%-30%，止损 3%-5%），给触发条件。\n"
        "   - 进取型：建议仓位比例与止损区间（例如 30%-50%，止损 5%-8%），给触发条件。\n"
        "   - 已持有者：是否加减仓、如何复盘，给触发条件。\n"
        "5）复盘触发：明确何时需要重新评估（财报、行业事件、技术位被破等）。\n"
        "6）专业提示：说明建议基于当前数据与概率而非确定性，避免绝对化承诺。\n"
        "语言专业克制，先结论、后依据与风险、最后给可执行建议。"
    )

    payload = {
        "model": os.environ.get("LLM_MODEL", "deepseek-reasoner"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.2
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def generate_module_followup(facts: Dict[str, Any], module: str, inputs: Dict[str, Any], advisor_text: str, question: str, endpoint: str, api_key: str=None) -> str:
    system = (
        "你是一位资深金融分析师。请以常规金融研究视角，直接、专业回答用户的追问。"
        "基于上下文与数据，给出明确判断与关键依据，包含适度风险提示；不输出分节模板，不重复模块说明。中文输出。"
    )
    user_text = (
        f"模块：{module}\n"
        f"已给出的建议摘要：{_clip_text(advisor_text or '', 600)}\n"
        f"追问：{_clip_text(question or '', 300)}\n"
        f"上下文摘要：{_clip_text(inputs.get('module_data_summary',''), 800)}\n"
        f"请用专业金融分析师口吻，直接回答该追问，突出关键依据与风险提示，不给确定性承诺。"
    )
    payload = {
        "model": os.environ.get("LLM_FAST_MODEL", os.environ.get("LLM_MODEL", "deepseek-chat")),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.1
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=8)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content
