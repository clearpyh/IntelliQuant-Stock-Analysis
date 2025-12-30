# TODO_Investment_Analysis

## Pending Items
- [ ] **LLM Narrative Enhancement:** Currently `src/explanation.py` uses templates. Connecting this to the `generate_conclusions_with_llm` to rewrite these templates dynamically could improve readability.
- [ ] **Threshold Tuning:** The values (e.g., ADX > 20, Correlation > 0.7) are standard defaults. They should be calibrated based on the specific market (A-Share) history.
- [ ] **Unit Tests:** Add unit tests for `src/explanation.py` to ensure logic correctness across edge cases (empty data, NaN values).

## Configuration
- No new environment variables required.
