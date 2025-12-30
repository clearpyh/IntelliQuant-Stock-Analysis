import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

def generate_trend_conclusion(df_span: pd.DataFrame, indicators: pd.DataFrame):
    """
    根据短期价格走势和指标生成趋势结论文字
    """
    conclusion = []

    # 基础趋势判断（收盘价 vs SMA/EMA）
    if "SMA" in indicators.columns:
        sma_latest = indicators["SMA"].iloc[-1]
        close_latest = df_span["close"].iloc[-1]
        if close_latest > sma_latest:
            conclusion.append("价格高于SMA，短期多头趋势占优")
        elif close_latest < sma_latest:
            conclusion.append("价格低于SMA，短期空头趋势占优")
        else:
            conclusion.append("价格接近SMA，多空力量均衡")

    if "EMA" in indicators.columns:
        ema_latest = indicators["EMA"].iloc[-1]
        close_latest = df_span["close"].iloc[-1]
        if close_latest > ema_latest:
            conclusion.append("价格高于EMA，短期趋势偏多")
        elif close_latest < ema_latest:
            conclusion.append("价格低于EMA，短期趋势偏空")
        else:
            conclusion.append("价格接近EMA，短期趋势不明")

    # 简单波动分析
    if not df_span.empty:
        span_pct_change = (df_span["close"].iloc[-1] - df_span["close"].iloc[0]) / df_span["close"].iloc[0] * 100
        if abs(span_pct_change) < 1:
            conclusion.append("本时间段价格波动较小，市场偏震荡")
        elif span_pct_change > 0:
            conclusion.append(f"本时间段上涨约 {span_pct_change:.2f}% ，趋势向上")
        else:
            conclusion.append(f"本时间段下跌约 {abs(span_pct_change):.2f}% ，趋势向下")

    return "；".join(conclusion)


def plot_candlestick_with_indicators(df: pd.DataFrame, date_col: str, indicators: pd.DataFrame, time_span: str | None = None, key: str | None = None, show_text: bool = True):
    if time_span is None:
        time_span = st.selectbox("选择K线时间跨度", ["1周", "1个月", "3个月", "1年", "3年", "5年", "10年", "全部"], index=3, key=key)
        end_date = df[date_col].max()
        if time_span == "1周":
            start_date = end_date - pd.Timedelta(weeks=1)
        elif time_span == "1个月":
            start_date = end_date - pd.Timedelta(days=30)
        elif time_span == "3个月":
            start_date = end_date - pd.Timedelta(days=90)
        elif time_span == "1年":
            start_date = end_date - pd.Timedelta(days=365)
        elif time_span == "3年":
            start_date = end_date - pd.Timedelta(days=365*3)
        elif time_span == "5年":
            start_date = end_date - pd.Timedelta(days=365*5)
        elif time_span == "10年":
            start_date = end_date - pd.Timedelta(days=365*10)
        else:
            start_date = df[date_col].min()
        df_span = df[df[date_col] >= start_date].copy()
        indicators_span = indicators[indicators.index >= start_date].copy()
    else:
        df_span = df.copy()
        indicators_span = indicators.copy()

    if df_span.empty:
        st.warning("所选时间段无数据")
        return go.Figure()

    # 自动计算纵坐标
    min_price = df_span['close'].min()
    max_price = df_span['close'].max()
    # 简单的 padding
    y_range = [min_price * 0.98, max_price * 1.02]

    # 绘制 K 线图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_span[date_col],
        open=df_span["open"],
        high=df_span["high"],
        low=df_span["low"],
        close=df_span["close"],
        name="K线"
    ))

    if "SMA" in indicators_span.columns:
        fig.add_trace(go.Scatter(x=indicators_span.index, y=indicators_span["SMA"], name="SMA"))
    if "EMA" in indicators_span.columns:
        fig.add_trace(go.Scatter(x=indicators_span.index, y=indicators_span["EMA"], name="EMA"))

    sr_window = min(60, len(df_span))
    hi = float(df_span["high"].tail(sr_window).max()) if sr_window > 0 else None
    lo = float(df_span["low"].tail(sr_window).min()) if sr_window > 0 else None
    shapes = []
    if hi is not None:
        shapes.append(dict(type="line", xref="x", yref="y", x0=df_span[date_col].min(), x1=df_span[date_col].max(), y0=hi, y1=hi, line=dict(color="orange", width=1, dash="dash")))
    if lo is not None:
        shapes.append(dict(type="line", xref="x", yref="y", x0=df_span[date_col].min(), x1=df_span[date_col].max(), y0=lo, y1=lo, line=dict(color="purple", width=1, dash="dash")))
    fig.update_layout(
        title=f"K线图 - {time_span}",
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            range=y_range,
            autorange=False,
            fixedrange=False
        ),
        shapes=shapes,
        annotations=[dict(
            text="注：纵坐标根据选定时间段自动缩放以显示局部波动趋势",
            showarrow=False,
            xref="paper", yref="paper",
            x=0, y=-0.15,
            font=dict(size=10, color="gray")
        )],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 生成趋势结论
    if show_text:
        conclusion_text = generate_trend_conclusion(df_span, indicators_span)
        st.markdown(f"**趋势分析结论 ({time_span})：** {conclusion_text}")

    return fig

def plot_corr_heatmap(corr: pd.DataFrame):
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", origin="lower")
    return fig

def plot_pca_explained(explained: pd.Series):
    fig = px.bar(x=list(range(1, len(explained)+1)), y=explained.values, labels={"x": "Principal Component", "y": "Explained Variance Ratio"})
    return fig

def plot_stl_components(trend: pd.Series, seasonal: pd.Series, resid: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name="Trend"))
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal.values, name="Seasonal"))
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, name="Resid"))
    fig.update_layout(title="STL Decomposition")
    return fig

def plot_cluster_scatter(X: pd.DataFrame, labels: pd.Series):
    fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color=labels.astype(str), title="Risk-Return Clustering")
    return fig

def plot_acf_pacf(acf_vals: np.ndarray, pacf_vals: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"))
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF"))
    fig.update_layout(title="ACF & PACF")
    return fig

def plot_regression_coeffs(coefs: pd.Series):
    fig = px.bar(x=coefs.index.tolist(), y=coefs.values, title="Regression Coefficients")
    return fig

def plot_probability_hist(proba: pd.Series):
    fig = px.histogram(x=proba.values, nbins=20, title="Probability Distribution")
    return fig

def plot_factor_portrait(portrait: pd.DataFrame):
    if "percentile" in portrait.columns:
        fig = px.bar(portrait, x="metric", y="percentile", title="Factor Portrait (Percentile by Industry)", labels={"percentile":"Percentile"})
    else:
        fig = px.bar(portrait, x="metric", y="value", title="Factor Portrait (Value)", labels={"value":"Value"})
    return fig

def plot_returns_scatter(x: pd.Series, y: pd.Series, x_label: str, y_label: str):
    df = pd.DataFrame({x_label: pd.to_numeric(x, errors="coerce"), y_label: pd.to_numeric(y, errors="coerce")}).dropna()
    fig = px.scatter(df, x=x_label, y=y_label, trendline="ols", title="收益率散点图")
    return fig

def plot_rolling_corr(series: pd.Series, title: str):
    fig = px.line(x=series.index, y=series.values, labels={"x": "date", "y": title}, title=title)
    return fig
