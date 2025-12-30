import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

def compute_corr_matrix(pivot_close: pd.DataFrame) -> pd.DataFrame:
    return pivot_close.pct_change().dropna().corr()

def compute_pca(X: pd.DataFrame):
    model = PCA()
    model.fit(X)
    explained = pd.Series(model.explained_variance_ratio_)
    return model, explained

def compute_kmeans(X: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
    labels = km.fit_predict(X)
    return km, pd.Series(labels, index=X.index)

def compute_factor_regression(X: pd.DataFrame, y: pd.Series):
    Xf = X.fillna(X.median())
    yf = y.loc[Xf.index]
    model = LinearRegression()
    model.fit(Xf.values, yf.values)
    coef = pd.Series(model.coef_, index=Xf.columns)
    r2 = float(model.score(Xf.values, yf.values))
    return model, coef, r2

def compute_logistic_proba(X: pd.DataFrame, y_cls: pd.Series):
    Xf = X.fillna(X.median())
    yf = y_cls.loc[Xf.index].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xf.values, yf.values)
    proba = pd.Series(model.predict_proba(Xf.values)[:,1], index=Xf.index)
    try:
        auc = float(roc_auc_score(yf.values, proba.values))
    except Exception:
        auc = None
    return model, proba, auc

def generate_security_tags(ret: float, vol: float, mdd: float, sharpe: float, pe: float | None, pb: float | None, baseline_ret: float, baseline_vol: float) -> list[str]:
    tags = []
    r_rel = ret - baseline_ret
    v_rel = vol - baseline_vol
    if r_rel > 0.02 and v_rel > 0.02:
        tags.append("🚀 高收益高风险")
    if v_rel < -0.02 and sharpe >= 0.8:
        tags.append("🛡 稳健防守型")
    if v_rel > 0.05 and sharpe < 0.5:
        tags.append("⚠ 高波动不稳定")
    if sharpe >= 1.0 and (pe is not None and pe < 20 or pb is not None and pb < 3):
        tags.append("💎 性价比突出")
    if not tags:
        tags.append("ℹ 中性")
    return tags
