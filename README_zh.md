# IntelliQuant-Stock-Analysis

[![CI](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/clearpyh/IntelliQuant-Stock-Analysis.svg)](https://github.com/clearpyh/IntelliQuant-Stock-Analysis/stargazers)

IntelliQuant-Stock-Analysis — 基于 Python 的一站式量化研究工具：Tushare 数据采集 → 数据清洗与指标 → 策略回测 → 可视化与报告导出。

仓库：https://github.com/clearpyh/IntelliQuant-Stock-Analysis

---

目录
- 快速概览
- 关键功能
- 快速开始
- 示例与演示
- 配置（Tushare）与合规
- 使用示例
- 项目结构
- 测试与持续集成
- 贡献
- 常见问题
- 许可证
- 致谢
- 联系与计划

快速概览

IntelliQuant-Stock-Analysis 致力于提供从原始市场数据到验证交易想法的可复现、模块化流程。项目以 Tushare 为主要数据源，包含因子/指标计算、回测、可视化等工具。

关键功能
- 基于 Tushare 的数据采集（行情、复权、财务）
- 数据清洗、对齐与缓存工具
- 常用技术指标与因子计算（可扩展）
- 轻量回测框架（支持参数扫描与批量回测）
- 回测结果可视化（净值曲线、回撤、因子分组表现）
- 示例脚本与 Jupyter Notebook 便于快速上手

快速开始

1. 克隆仓库：
   git clone https://github.com/clearpyh/IntelliQuant-Stock-Analysis.git
   cd IntelliQuant-Stock-Analysis

2. 创建虚拟环境并安装依赖：
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows PowerShell
   pip install -r requirements.txt

3. 设置 Tushare token：
   export TUSHARE_TOKEN="your_token_here"   # macOS / Linux
   setx TUSHARE_TOKEN "your_token_here"     # Windows（重启终端）

4. 运行示例：
   python examples/run_sample_backtest.py

示例与演示

- examples/ 包含可运行脚本与 Notebook，演示数据抓取、回测与绘图流程。
- 建议添加可一键运行的 Binder/Colab 链接以便演示。

配置（Tushare）与合规

- 在 https://tushare.pro 注册并获取 Tushare Token。
- 使用本项目时请遵守 Tushare 的使用条款与数据许可，商业使用或大规模抓取请确保合规。
- 推荐将 token 存放在环境变量或 .env（并将 .env 加入 .gitignore）。

使用示例（最小）

```python
import os
from intelliquant.data import TushareFetcher
from intelliquant.backtest import Backtester
from intelliquant.strategies import MovingAverageCross

fetcher = TushareFetcher(token=os.getenv("TUSHARE_TOKEN"))
df = fetcher.get_daily(ts_code="000001.SZ", start="2022-01-01", end="2022-12-31")

strategy = MovingAverageCross(short_window=20, long_window=50)
bt = Backtester(data=df, strategy=strategy, cash=1_000_000)
result = bt.run()

print(result.summary())
result.plot_equity_curve().savefig("equity_curve.png")
```

项目结构（示例）

```
- examples/            # 示例脚本与 Notebook
- intelliquant/        # 主包（或 iqsa/，依据实际）
  - data.py            # Tushare 抓取与数据工具
  - backtest.py        # 回测核心
  - strategies.py      # 策略示例
  - indicators.py      # 指标/因子计算
- notebooks/           # Jupyter 示例（可选）
- tests/               # 单元测试
- docs/                # 文档站点（可选）
- requirements.txt
- setup.py / pyproject.toml
- LICENSE
- README_en.md
- README_zh.md
```

测试与 CI

- 建议添加 GitHub Actions：lint（ruff/flake8）、format（black/isort）、单元测试（pytest）。
- 本地运行测试：
  pip install -r requirements-dev.txt
  pytest -q

贡献

欢迎贡献！请在仓库中添加 CONTRIBUTING.md，包含：
- Issue 与 PR 流程
- 代码风格与测试要求
- 分支与发布流程

常见问题

Q: 如何获取复权数据？
A: 使用 Tushare 的 adj_factor 或 pro_bar，项目在 data 模块中封装了常用流程，参考 examples/data_prep.ipynb。

Q: 回测结果为何和实盘不同？
A: 常见原因包括数据修正、未建模交易成本与滑点、撮合与订单假设差异等。请在回测中显式建模交易成本与滑点。

许可证

本项目使用 MIT 许可证。详情见 LICENSE 文件。

致谢

- 感谢 Tushare 团队提供数据接口与文档支持。本项目使用 Tushare 作为主要数据源，请遵守其使用条款。
- 感谢所有提交 issue、PR 或建议的开源社区成员。

联系方式与计划

- 作者：clearpyh
- 仓库：https://github.com/clearpyh/IntelliQuant-Stock-Analysis

后续计划示例：
- 增加更多内置因子与因子研究工具
- 支持更多数据源（Wind、RiceQuant 适配层）
- 提升回测性能（向量化/并行）
- 发布 PyPI 包与 Docker 镜像
- 完善文档站点（mkdocs / sphinx）
