# ALIGNMENT_Investment_Analysis

## 1. Project Context Analysis
- **Current State:**
  - A Streamlit-based stock analysis tool (`app.py`).
  - Features: K-Line, ADX, Correlation Heatmap, PCA, Volatility (GARCH), Seasonality (STL), Clustering (KMeans), Factor Regression, Logistic Probability.
  - Visualization: Plotly charts.
  - Conclusion: Currently a separate section at the end, either rule-based or LLM-generated.
- **Architecture:**
  - `app.py`: UI and orchestration.
  - `src/visualization.py`: Chart generation.
  - `src/analysis/`: Analytical computations.
  - `src/conclusion.py`: Conclusion generation (bulk).

## 2. Requirement Understanding
- **Core Requirement:** Transformation from "Data Visualization" to "Actionable Investment Advice".
- **Design Principle:** "From Charts to Understandable Conclusions".
- **Specific Constraint:** *Every* chart/indicator must have a paired explanatory analysis.
- **Key Questions to Answer (per component):**
  1.  **Trend:** Strong/Weak? Sustainable?
  2.  **Risk:** Level? Suitable for ordinary investors?
  3.  **Risk/Return:** Matching?
  4.  **Probability:** Upside/Downside probability?
- **Failure Condition:** Displaying images without analysis is considered incomplete.

## 3. Ambiguity & Clarification
- **LLM vs Rule-Based:** The current system has an LLM option. The user wants "Understandable Conclusions". Should we rely primarily on LLM for the text generation (if configured) or robust rule-based logic?
  - *Decision:* We will implement a **Hybrid Approach**. We will build robust rule-based templates that answer the core questions directly from the metrics. If LLM is available, it can enhance the text, but the core "answers" (Strong/Weak, High/Low Risk) must be deterministically derivable from the data to ensure reliability even without an LLM.
- **Granularity:** Does "every chart" mean literally every single plot?
  - *Assumption:* Yes. We will wrap each major visualization section in `app.py` with a corresponding "Analysis & Implication" block.

## 4. Proposed Strategy
1.  **Refactor `app.py`:** Instead of a long list of charts followed by a conclusion, we will group them: `[Chart] -> [Analysis Card]`.
2.  **Enhance `src/conclusion.py` (or new `src/explanation.py`):** Add specific functions to generate insights for each chart type:
    - `explain_trend(df, adx)` -> Answers Trend Strength & Sustainability.
    - `explain_volatility(vol, garch)` -> Answers Risk Level.
    - `explain_correlation(corr)` -> Answers Systemic Risk.
    - `explain_probability(proba)` -> Answers Win Rate.
3.  **UI UX:** Use `st.info`, `st.success`, `st.warning` boxes immediately below charts to highlight the investment advice.
