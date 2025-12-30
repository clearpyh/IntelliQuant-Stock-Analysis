# CONSENSUS_Investment_Analysis

## 1. Requirement Confirmation
- We will transform the app to provide "Explainable Conclusions" for every chart.
- We will answer the 4 core questions: Trend, Risk, Risk/Return, Probability.

## 2. Technical Solution
- **New Module:** `src/explanation.py` containing rule-based logic for generating advice.
- **UI Update:** `app.py` will display an "Investment Insight" box after every chart.
- **Hybrid Approach:** Primary logic is deterministic rule-based (Python), optionally augmented by LLM if configured (but not required).

## 3. Acceptance Criteria
- [ ] Every chart in the app has a text description below it.
- [ ] The text description explicitly contains: "Current Trend/Status", "Risk Level", and "Investment Advice".
- [ ] Code runs without errors.
