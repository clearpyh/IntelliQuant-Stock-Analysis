from typing import Dict
from pathlib import Path
from src.conclusion import generate_module_advice_text, generate_module_followup_text
from storage.repository import load_module_advisor, save_module_advisor, build_advisor_payload, is_stale
from pathlib import Path

def get_or_generate_advisor(session_state: Dict, module: str, inputs: Dict, facts: Dict) -> str:
    store = session_state.setdefault("module_store", {})
    state = store.setdefault(module, {"params": None, "advisor": None, "followups": {}})
    params = {"symbol": facts.get("symbol"), "industry": facts.get("industry"), "time_range": inputs.get("time_range")}
    root = Path(__file__).parent
    if state["params"] != params or not state["advisor"]:
        cached = load_module_advisor(root, params["symbol"], module)
        if cached and not is_stale(cached, params):
            state["advisor"] = cached.get("advisor_text")
            state["params"] = params
        else:
            advisor = generate_module_advice_text(module, inputs, facts)
            state["advisor"] = advisor
            state["params"] = params
            save_module_advisor(root, params["symbol"], module, build_advisor_payload(advisor, params))
    return state["advisor"]

def generate_followup_reply(session_state: Dict, module: str, inputs: Dict, facts: Dict, advisor_text: str, question: str) -> str:
    store = session_state.setdefault("module_store", {})
    state = store.setdefault(module, {"params": None, "advisor": None, "followups": {}})
    qkey = (question or "").strip()
    if not qkey:
        return "请明确你的问题，例如“是否适合加仓？”或“当前风险主要来自哪里？”"
    if qkey in state["followups"]:
        return state["followups"][qkey]
    reply = generate_module_followup_text(module, inputs, facts, advisor_text, qkey)
    state["followups"][qkey] = reply
    return reply
