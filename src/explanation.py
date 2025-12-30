import pandas as pd
import numpy as np
from typing import Dict, Any, List

def _get_status_icon(sentiment: str) -> str:
    if sentiment == "positive": return "🟢"
    if sentiment == "negative": return "🔴"
    if sentiment == "warning": return "🟠"
    return "⚪"

def explain_trend_adx(df: pd.DataFrame, adx_series: pd.Series, ma_short: int, ma_long: int) -> Dict[str, Any]:
    """
    Analyzes Trend Strength and Sustainability.
    Core Question: "Is the trend strong or weak? Is it sustainable?"
    """
    if df.empty or adx_series.empty:
        return {"valid": False, "msg": "数据不足"}
    
    last_close = df["close"].iloc[-1]
    last_adx = adx_series.iloc[-1]
    
    # Calculate MAs if not present
    sma = df["close"].rolling(ma_short).mean()
    lma = df["close"].rolling(ma_long).mean()
    
    last_sma = sma.iloc[-1]
    last_lma = lma.iloc[-1]
    
    # Trend Direction
    if last_close > last_sma > last_lma:
        direction = "Bullish (多头)"
        sentiment = "positive"
    elif last_close < last_sma < last_lma:
        direction = "Bearish (空头)"
        sentiment = "negative"
    else:
        direction = "Uncertain/Consolidation (震荡)"
        sentiment = "neutral"
        
    # Trend Strength (ADX)
    if last_adx < 20:
        strength = "Weak (弱势)"
        sustainability = "趋势不明显，市场处于震荡整理阶段"
        advice = "当前缺乏明确趋势，建议观望或采用区间交易策略（高抛低吸）。不宜追涨杀跌。"
    elif 20 <= last_adx < 40:
        strength = "Moderate (中等)"
        sustainability = "趋势正在形成或稳步运行"
        advice = f"当前处于{direction}趋势中，且具备一定强度。建议顺势而为。"
    elif 40 <= last_adx < 60:
        strength = "Strong (强劲)"
        sustainability = "趋势非常强劲，需关注延续性"
        advice = f"趋势强劲，主要持有仓位。但需警惕短期乖离过大后的回调风险。"
    else:
        strength = "Extreme (极强)"
        sustainability = "趋势极度强劲，可能面临衰竭或反转"
        advice = "市场情绪可能过热（或过恐），趋势随时可能反转。建议收紧止损，适当止盈，切勿盲目加仓。"

    return {
        "valid": True,
        "title": "趋势强度与方向分析",
        "signal": f"{direction} | ADX: {last_adx:.1f} ({strength})",
        "assessment": f"当前价格处于{direction}状态。ADX指标显示趋势强度为{strength}。{sustainability}。",
        "advice": advice,
        "sentiment": sentiment,
        "question": "当前趋势是强还是弱？是否具备持续性？"
    }

def explain_volatility_risk(vol_series: pd.Series, garch_forecast_variance=None) -> Dict[str, Any]:
    """
    Analyzes Volatility Risk.
    Core Question: "What is the risk level? Suitable for ordinary investors?"
    """
    if vol_series.empty:
        return {"valid": False, "msg": "数据不足"}
    
    current_vol = vol_series.iloc[-1]
    avg_vol = vol_series.mean()
    
    # Risk Level Assessment
    ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    
    if ratio < 0.8:
        level = "Low (低风险)"
        desc = "当前波动率低于历史平均水平，市场情绪相对平稳。"
        suitability = "适合大多数投资者参与，但需警惕波动率回归（变盘）。"
        sentiment = "positive"
    elif 0.8 <= ratio < 1.2:
        level = "Normal (正常)"
        desc = "当前波动率处于正常范围内。"
        suitability = "适合具备基本风险承受能力的投资者。"
        sentiment = "neutral"
    elif 1.2 <= ratio < 2.0:
        level = "High (高风险)"
        desc = "当前波动率显著高于平均水平，市场剧烈波动。"
        suitability = "风险较高，仅适合风险偏好较高的激进投资者。"
        sentiment = "warning"
    else:
        level = "Extreme (极高风险)"
        desc = "市场处于极端波动状态，恐慌或狂热情绪主导。"
        suitability = "极不适合普通投资者，建议空仓或极轻仓观望。"
        sentiment = "negative"

    garch_msg = ""
    if garch_forecast_variance is not None:
        garch_vol = np.sqrt(garch_forecast_variance)
        garch_msg = f"GARCH模型预测下一期波动率可能为 {garch_vol:.4f}。"

    return {
        "valid": True,
        "title": "风险水平评估 (基于波动率)",
        "signal": f"风险等级: {level}",
        "assessment": f"{desc} (当前/均值 = {ratio:.2f})。{garch_msg}",
        "advice": f"{suitability} 建议根据风险等级调整仓位：{level}环境下应{'加大' if ratio < 0.8 else '减少'}杠杆或持仓。",
        "sentiment": sentiment,
        "question": "当前风险水平如何？是否适合普通投资者参与？"
    }

def explain_correlation_risk(corr_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes Systemic Risk via Correlation.
    Core Question: "Risk/Return Match? (Diversification value)"
    """
    if corr_matrix is None or corr_matrix.empty:
        return {"valid": False, "msg": "数据不足"}
    
    # Use upper triangle only to avoid self-correlation and duplicates
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_corr = corr_matrix.where(mask).stack().mean()
    
    if np.isnan(avg_corr):
        avg_corr = 0.0

    if avg_corr > 0.7:
        status = "High Systemic Risk (高度联动)"
        desc = "行业内个股同涨同跌现象极强，选股很难获得超额收益。"
        advice = "分散投资在本行业内几乎无效。建议要么配置行业ETF，要么通过配置其他相关性低的行业来对冲风险。"
        sentiment = "negative"
    elif avg_corr > 0.4:
        status = "Moderate Correlation (中度联动)"
        desc = "行业具备一定的整体效应，但个股分化依然存在。"
        advice = "可以进行精选个股，寻找强于行业指数的龙头。适度分散持仓可以降低非系统性风险。"
        sentiment = "neutral"
    else:
        status = "Low Correlation (低联动/分化)"
        desc = "行业内个股走势独立性强，受宏观或行业贝塔影响小。"
        advice = "是“轻指数重个股”的最佳时机。深入挖掘基本面独特的个股有望获得显著的Alpha收益。"
        sentiment = "positive"

    return {
        "valid": True,
        "title": "行业系统性风险分析 (相关性)",
        "signal": f"联动性: {avg_corr:.2f} ({status})",
        "assessment": desc,
        "advice": advice,
        "sentiment": sentiment,
        "question": "收益与风险是否匹配？(是否可以通过分散投资降低风险)"
    }

def explain_pca_structure(explained_variance: pd.Series) -> Dict[str, Any]:
    """
    Analyzes Market Structure via PCA.
    """
    if explained_variance is None or len(explained_variance) == 0:
         return {"valid": False, "msg": "数据不足"}
    
    first_comp = explained_variance.iloc[0]
    
    if first_comp > 0.7:
        msg = "市场由单一主导因素（通常是宏观或大盘情绪）驱动，个股特性被掩盖。"
        advice = "宏观分析优于个股分析。重点关注大盘走势和宏观政策，而非个股基本面。"
    elif first_comp > 0.4:
        msg = "市场受主要因素影响，但仍有部分个股逻辑在演绎。"
        advice = "结合宏观与个股。既要看大势，也要选好股。"
    else:
        msg = "市场驱动力分散，个股行情独立，缺乏统一的主线。"
        advice = "精选个股的黄金时期。主要关注个股自身的催化剂。"
        
    return {
        "valid": True,
        "title": "市场驱动力结构 (PCA)",
        "signal": f"主成分解释率: {first_comp:.2%}",
        "assessment": msg,
        "advice": advice,
        "sentiment": "neutral",
        "question": "当前趋势是强还是弱？(市场结构维度)"
    }

def explain_seasonality(stl_res: Any) -> Dict[str, Any]:
    """
    Analyzes Seasonality.
    """
    if stl_res is None:
        return {"valid": False, "msg": "数据不足"}
        
    # Simple logic: Check variance of seasonal component vs residual
    seasonal_var = np.var(stl_res.seasonal)
    resid_var = np.var(stl_res.resid)
    
    if seasonal_var > resid_var * 1.5:
        status = "Significant Seasonality (显著季节性)"
        advice = "该标的具有较强的周期/季节规律。建议研究历史同期的走势（如月度/季度效应）来辅助择时。"
    else:
        status = "Weak Seasonality (弱季节性)"
        advice = "季节性规律不明显，更多关注趋势和随机冲击。"
        
    return {
        "valid": True,
        "title": "周期与季节性分析",
        "signal": status,
        "assessment": f"季节成分波动({'大于' if seasonal_var > resid_var else '小于'})随机噪音。",
        "advice": advice,
        "sentiment": "neutral",
        "question": "在历史统计意义上，未来上涨或下跌的概率如何？(周期维度)"
    }

def explain_prediction_probability(proba_series: pd.Series, auc_score: float) -> Dict[str, Any]:
    """
    Analyzes Prediction Probability.
    Core Question: "Future probability?"
    """
    if proba_series is None or proba_series.empty:
        return {"valid": False, "msg": "数据不足"}
        
    # Assuming proba is "Probability of Up" for the specific stock if passed, 
    # OR distribution of probabilities for the industry.
    # Here we analyze the distribution if it's a series of multiple stocks, 
    # or a single value if specific.
    
    mean_prob = proba_series.mean()
    
    reliability = "低"
    if auc_score and auc_score > 0.7: reliability = "高"
    elif auc_score and auc_score > 0.6: reliability = "中"
    
    if mean_prob > 0.6:
        direction = "Upward Bias (看涨)"
        sentiment = "positive"
        advice = "模型预测上涨概率较大。在控制风险的前提下，可考虑做多。"
    elif mean_prob < 0.4:
        direction = "Downward Bias (看跌)"
        sentiment = "negative"
        advice = "模型预测下跌概率较大。建议减仓或做空，避免逆势操作。"
    else:
        direction = "Neutral (中性)"
        sentiment = "neutral"
        advice = "模型预测涨跌概率接近五五开，方向不明。建议观望。"
        
    return {
        "valid": True,
        "title": "未来涨跌概率预测 (逻辑回归)",
        "signal": f"平均上涨概率: {mean_prob:.2%} (模型可靠性: {reliability})",
        "assessment": f"基于历史特征训练的模型显示，当前样本倾向于{direction}。",
        "advice": f"{advice} (注：历史统计意义上的概率，不代表绝对未来)",
        "sentiment": sentiment,
        "question": "在历史统计意义上，未来上涨或下跌的概率如何？"
    }

def explain_factor_regression(coefs: pd.Series, r2: float) -> Dict[str, Any]:
    """
    Explain Factor Exposure.
    """
    if coefs is None or coefs.empty:
        return {"valid": False, "msg": "数据不足"}
    
    top_factor = coefs.abs().idxmax()
    direction = "正向" if coefs[top_factor] > 0 else "负向"
    
    if r2 > 0.5:
        assess = f"该组合深受基本面因子驱动 (R2={r2:.2f})。主要受 {top_factor} 因子{direction}影响。"
        advice = f"关注 {top_factor} 的变化。如果预期该指标改善，则利好股价。"
    else:
        assess = f"基本面因子对收益率解释力较弱 (R2={r2:.2f})，可能受非基本面因素（如资金面、情绪）主导。"
        advice = "单纯依赖基本面因子选股可能失效，需结合技术面或资金面分析。"
        
    return {
        "valid": True,
        "title": "基本面驱动因素分析",
        "signal": f"主导因子: {top_factor} ({direction})",
        "assessment": assess,
        "advice": advice,
        "sentiment": "neutral",
        "question": "收益与风险是否匹配？(归因分析)"
    }

def explain_clustering(n_clusters: int, labels: pd.Series) -> Dict[str, Any]:
    """
    Explain Clustering results.
    """
    if labels is None or labels.empty:
        return {"valid": False, "msg": "数据不足"}
        
    counts = labels.value_counts()
    dominant_cluster = counts.idxmax()
    dominant_ratio = counts.max() / len(labels)
    
    if dominant_ratio > 0.8:
        msg = "大部分标的聚集在同一类别，市场分化程度低。"
        advice = "市场同质化严重，精选个股难度较大，建议顺势而为。"
    else:
        msg = f"市场明显分化为 {n_clusters} 个阵营，存在结构性机会。"
        advice = "市场存在分化，可寻找处于'高收益-低风险'聚类的个股。"
        
    return {
        "valid": True,
        "title": "市场风险收益结构 (聚类)",
        "signal": f"分化程度: {'低' if dominant_ratio > 0.8 else '高'}",
        "assessment": msg,
        "advice": advice,
        "sentiment": "neutral",
        "question": "收益与风险是否匹配？(市场结构维度)"
    }

def explain_factor_portrait(portrait: pd.DataFrame) -> Dict[str, Any]:
    if portrait is None or portrait.empty:
        return {"valid": False, "msg": "数据不足"}
    trend_pos = []
    trend_neg = []
    if "trend" in portrait.columns:
        try:
            tr = pd.to_numeric(portrait["trend"], errors="coerce")
            trend_pos = portrait.loc[tr > 0, "metric"].astype(str).tolist()
            trend_neg = portrait.loc[tr < 0, "metric"].astype(str).tolist()
        except Exception:
            trend_pos = []
            trend_neg = []
    if "percentile" in portrait.columns:
        try:
            avg_pct = float(pd.to_numeric(portrait["percentile"], errors="coerce").dropna().mean())
        except Exception:
            avg_pct = 0.5
        top = portrait.sort_values("percentile", ascending=False).iloc[0]["metric"]
        signal = f"行业分位均值: {avg_pct:.2f}；优势因子: {top}"
        assessment = f"分位偏{'高' if avg_pct>0.5 else '低'}；上行因子: {', '.join(trend_pos) if trend_pos else '无'}；下行因子: {', '.join(trend_neg) if trend_neg else '无'}。"
    else:
        top = portrait.sort_values("value", ascending=False).iloc[0]["metric"] if "value" in portrait.columns else "未知"
        try:
            avg_val = float(pd.to_numeric(portrait["value"], errors="coerce").dropna().mean()) if "value" in portrait.columns else None
        except Exception:
            avg_val = None
        signal = f"优势因子: {top}" if avg_val is None else f"平均指标值: {avg_val:.2f}；优势因子: {top}"
        assessment = f"上行因子: {', '.join(trend_pos) if trend_pos else '无'}；下行因子: {', '.join(trend_neg) if trend_neg else '无'}。"
    advice = "结合优势因子维持长期关注；劣势因子侧重改善与跟踪。"
    return {
        "valid": True,
        "title": "基本面画像分析（单证券）",
        "signal": signal,
        "assessment": assessment,
        "advice": advice,
        "sentiment": "neutral",
        "question": "在因子回归不可用时，如何理解该证券的长期风格与基本面特征？"
    }
