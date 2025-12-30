from analysis.kline.service import run_kline_analysis
from analysis.correlation.service import run_correlation_analysis
from analysis.pca.service import run_pca_analysis
from analysis.volatility.service import run_volatility_analysis
from analysis.seasonality.service import run_seasonality_analysis
from analysis.risk_cluster.service import run_risk_cluster_analysis
from analysis.factor_exposure.service import run_factor_exposure_analysis
from analysis.price_prob.service import run_price_prob_analysis

REGISTRY = {
    "K线与指标": run_kline_analysis,
    "相关性分析": run_correlation_analysis,
    "PCA分析": run_pca_analysis,
    "波动性分析": run_volatility_analysis,
    "季节性分析": run_seasonality_analysis,
    "风险-收益聚类分析": run_risk_cluster_analysis,
    "基本面因子暴露分析": run_factor_exposure_analysis,
    "涨跌概率分析": run_price_prob_analysis
}
