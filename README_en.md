# IntelliQuant-Stock-Analysis

[![CI](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/clearpyh/IntelliQuant-Stock-Analysis.svg)](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/stargazers)

IntelliQuant-Stock-Analysis — an open-source, end-to-end quantitative research toolkit in Python:
Tushare data collection → data cleaning & indicators → strategy backtesting → visualization & reporting.

Repository: https://github.com/clearpyh/IntelliQuant-Stock-Analysis

---

Table of Contents
- Quick overview
- Key features
- Quickstart
- Examples & demos
- Configuration (Tushare) and compliance
- Usage example
- Project layout
- Tests & CI
- Contributing
- FAQ
- License
- Acknowledgements
- Contact & Roadmap

Quick overview

IntelliQuant-Stock-Analysis aims to provide a reproducible, modular workflow from raw market data to validated trading ideas. It integrates Tushare as the primary data source and includes tools for indicator/factor calculation, backtesting, and visualization.

Key features
- Tushare-based data fetching: market data, adjusted prices, and financials
- Data cleaning, alignment, and caching utilities
- Common technical indicators and factor calculation utilities (extensible)
- Lightweight strategy backtester (parameter sweep & batch backtests)
- Result visualization: equity curves, drawdowns, factor group performance
- Example scripts and Jupyter notebooks for quick onboarding

Quickstart

1. Clone:
   git clone https://github.com/clearpyh/IntelliQuant-Stock-Analysis.git
   cd IntelliQuant-Stock-Analysis

2. Create virtual environment and install:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .\.venv\Scripts\activate  # Windows PowerShell
   pip install -r requirements.txt

3. Set your Tushare token:
   export TUSHARE_TOKEN="your_token_here"  # macOS / Linux
   setx TUSHARE_TOKEN "your_token_here"    # Windows (restart terminal)

4. Run a quick example:
   python examples/run_sample_backtest.py

Examples & demos

- examples/ contains runnable scripts and notebooks demonstrating data fetching, backtests, and plotting.
- notebooks/ (if present) contains ready-to-run Jupyter examples; consider adding Binder/Colab links for one-click demos.

Configuration (Tushare) and compliance

- Get a Tushare token at https://tushare.pro and set it in the environment variable `TUSHARE_TOKEN`.
- This project uses Tushare data — please comply with Tushare's terms of use and any applicable licensing restrictions.
- Store secrets safely (environment variables or a .env file). Add `.env` to `.gitignore`.

Usage example (minimal)

```python
import os
from intelliquant.data import TushareFetcher
from intelliquant.backtest import Backtester
from intelliquant.strategies import MovingAverageCross

# initialize fetcher with environment token
fetcher = TushareFetcher(token=os.getenv("TUSHARE_TOKEN"))

# get daily data for a single symbol
df = fetcher.get_daily(ts_code="000001.SZ", start="2022-01-01", end="2022-12-31")

# define a simple strategy and run backtest
strategy = MovingAverageCross(short_window=20, long_window=50)
bt = Backtester(data=df, strategy=strategy, cash=1_000_000)
result = bt.run()

print(result.summary())
result.plot_equity_curve().savefig("equity_curve.png")
```

Project layout (recommended)
```
- examples/            # runnable examples & notebooks
- intelliquant/        # main package (or iqsa/ depending on your layout)
  - data.py            # Tushare fetcher and data utilities
  - backtest.py        # backtesting core
  - strategies.py      # example strategies
  - indicators.py      # indicator/factor calculation
- notebooks/           # Jupyter examples (optional)
- tests/               # unit tests
- docs/                # documentation site (optional)
- requirements.txt
- setup.py / pyproject.toml
- LICENSE
- README_en.md
- README_zh.md
```

Tests & CI

- Add GitHub Actions to run lint (ruff/flake8), formatting (black/isort) and unit tests (pytest).
- Local test commands:
  pip install -r requirements-dev.txt
  pytest -q

Contributing

Contributions are welcome! Please add a CONTRIBUTING.md in the repo that covers:
- How to open issues and PRs
- Code style (black, isort) and tests
- Branching & release process
- Any maintainers or areas of ownership

FAQ

Q: How to get adjusted prices?
A: Use Tushare's adj_factor or pro_bar functions; the project wraps common flows in the data module. See examples/data_prep.ipynb.

Q: Why do backtest results differ from live trading?
A: Common causes: data revisions, missing trading costs/slippage, order execution assumptions, and differences in corporate action handling. Model transaction costs and slippage explicitly in backtests.

License

This project is licensed under the MIT License. See LICENSE for details.

Acknowledgements

- Thanks to the Tushare team for providing market and fundamental data APIs. Please follow Tushare's terms of use when using data from this project.
- Thanks to all contributors and users who report issues, send PRs, or suggest improvements.

Contact & Roadmap

- Author: clearpyh
- Repo: https://github.com/clearpyh/IntelliQuant-Stock-Analysis

Planned improvements (examples):
- More built-in factors and factor-research tools
- Support for additional data sources (Wind, RiceQuant adapters)
- Performance improvements (vectorized/parallel backtesting)
- Publish PyPI package and Docker image
- Improve documentation site (mkdocs / sphinx)
