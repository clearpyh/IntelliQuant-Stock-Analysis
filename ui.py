import streamlit as st
from typing import Dict
from advisor import get_or_generate_advisor, generate_followup_reply

def analysis_card(analysis: Dict):
    if not analysis or not analysis.get("valid", False):
        return
    s = analysis.get("sentiment", "neutral")
    icon = "⚪"
    color = "blue"
    if s == "positive":
        icon = "🟢"
        color = "green"
    elif s == "negative":
        icon = "🔴"
        color = "red"
    elif s == "warning":
        icon = "🟠"
        color = "orange"
    st.markdown(f"#### {icon} {analysis['title']}")
    with st.chat_message("assistant", avatar=icon):
        st.markdown(f"**核心结论:** {analysis['signal']}")
        st.markdown(f"**投资建议:** :dart: **{analysis['advice']}**")
        st.caption(f"详细分析: {analysis['assessment']}")
        st.caption(f"回答核心问题: *{analysis['question']}*")
    st.divider()

def advisor_text(txt: str):
    if not txt:
        return
    st.markdown("#### 🤖 小金想说的话")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(txt)
    st.divider()

def followup(module: str, inputs: Dict, facts: Dict):
    key_base = module.replace(" ", "_")
    q = st.text_input("继续追问", value="", key=f"ask_{key_base}")
    send = st.button("发送", key=f"ask_{key_base}_btn")
    if send:
        with st.spinner("正在生成追问回复..."):
            advisor = get_or_generate_advisor(st.session_state, module, inputs, facts)
            reply = generate_followup_reply(st.session_state, module, inputs, facts, advisor, (q or "").strip())
        advisor_text(reply)

def nav_pills(label: str, modules: list, default: str, key: str):
    sel = st.pills(label, modules, default=default, key=key)
    return sel or default
