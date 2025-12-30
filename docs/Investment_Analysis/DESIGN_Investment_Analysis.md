# DESIGN_Investment_Analysis

## 1. System Architecture
- **Goal:** Embed analysis and investment advice directly alongside visualizations.
- **New Module:** `src/explanation.py`
  - **Responsibility:** Pure logic to translate raw metrics (ADX values, Volatility numbers, Correlation matrices) into human-readable "Investment Implications".
  - **Inputs:** Pandas DataFrames/Series, Numeric values.
  - **Outputs:** Structured Dictionaries containing:
    - `signal`: Short status (e.g., "Strong Uptrend", "High Risk").
    - `assessment`: Detailed reasoning (answering "Why?").
    - `advice`: Actionable investment advice (answering "What to do?").
    - `question_answered`: Which of the 4 core user questions this addresses.

## 2. Interface Contract (`src/explanation.py`)

### 2.1 Trend Analysis
- **Function:** `explain_trend_adx(price_df, adx_series, ma_short, ma_long)`
- **Core Question:** "Is the trend strong or weak? Is it sustainable?"
- **Logic:**
  - MA Alignment -> Direction.
  - ADX > 25 -> Strength.
  - ADX > 50 -> Overheated/Sustainable?

### 2.2 Risk Analysis
- **Function:** `explain_volatility_risk(vol_series, garch_sigma)`
- **Core Question:** "What is the risk level? Suitable for ordinary investors?"
- **Logic:**
  - Historic Volatility percentile -> Risk Level (Low/Medium/High).
  - GARCH Forecast -> Future Risk expectation.

### 2.3 Correlation/Systemic Risk
- **Function:** `explain_correlation_risk(corr_matrix, industry_name)`
- **Core Question:** "Risk/Return Match? (Diversification value)"
- **Logic:**
  - Avg Correlation > 0.7 -> High Systemic Risk (Hard to diversify).
  - Avg Correlation < 0.3 -> Low Systemic Risk (Good for picking alpha).

### 2.4 Probability/Prediction
- **Function:** `explain_prediction_probability(proba_series, auc_score)`
- **Core Question:** "Probability of future rise/fall?"
- **Logic:**
  - Probability > 60% -> High Confidence Upside.
  - AUC check -> Reliability of the model.

## 3. UI Integration (`app.py`)
- **Pattern:**
  ```python
  # 1. Compute
  metrics = compute_metrics(data)
  # 2. Visualize
  fig = plot_chart(metrics)
  st.plotly_chart(fig)
  # 3. Explain (New)
  insight = explain_metrics(metrics)
  render_insight_card(insight) # Helper to display standardized advice box
  ```

## 4. Module Dependencies
`app.py` -> `src/explanation.py` -> `src/analysis/*` (Indirectly via data passed)

## 5. Risk Management
- **Disclaimer:** All advice must be prefaced or footnoted with "Statistical based only, not financial advice."
- **Data Check:** If data is insufficient for a conclusion, explicitly state "Data Insufficient" rather than guessing.
