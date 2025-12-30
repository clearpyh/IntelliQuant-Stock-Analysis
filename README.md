# 智能证券分析系统

面向企业实践的端到端证券分析平台：集数据采集、量化分析、画像识别、自然语言解读与可视化报告于一体，支持模块化扩展与并行后台预计算，提供面向投资决策的高质量结论。
页面展示
<img width="2552" height="2711" alt="image" src="https://github.com/user-attachments/assets/2397d355-0534-40e5-8303-6755ff9fdf5d" />
<img width="2552" height="3775" alt="image" src="https://github.com/user-attachments/assets/4d2cbd5f-dfca-4546-8d8d-d10caeccc3c7" />


## 产品概览
- 场景覆盖：单证券分析、行业对比、批量采集与报告输出
- 模块化分析：K线与指标、相关性、PCA、波动性（GARCH）、季节性（STL/ACF/PACF）、聚类、基本面因子暴露/画像、涨跌概率（逻辑回归）
- 智能解读：统一人格的“大模型顾问（小金）”以通俗语言输出可读结论与风险提示
- 企业级体验：并行后台预计算、缓存与状态管理、降级策略与解释层解耦

## 关键能力
- 模块导航：侧边胶囊选择，模块独立渲染与解读
- 并行后台预计算：当前模块展示时，其余模块在后台异步计算并缓存，提高切换体验
- 缓存与状态管理：
  - 每模块维护 pending/ready/failed 状态
  - 行业矩阵（pivot_close）与文件读取结果缓存，避免重复构建
- 自动降级：
  - 基本面因子暴露分析在单证券样本不足时自动切换为“因子画像分析”，输出分位数与趋势解释
- 报告与输出：单证券与行业HTML报告、结论JSON与文本摘要TXT自动生成（在“小金卡片”提示）

## 架构与设计
- 前端：Streamlit（模块导航、图表渲染、交互控件、并行任务触发与状态读写）
- 分析层：
  - 时间序列：指标、ADX、波动率、GARCH、STL/ACF/PACF
  - 统计与机器学习：相关性、PCA、KMeans、线性/逻辑回归
- 可视化层：Plotly（K线、热力图、碎石图、聚类散点、因子画像）
- 解释层：规则化解释（explanation.py）与LLM文本（tools/llm_conclusion.py）
- 数据层：CSV（OHLCV与财报），支持批量采集与行业映射

## 模块说明
- K线与指标：多时间跨度K线（胶囊切换）、SMA/EMA、ADX强度与趋势结论
- 相关性分析：行业内多证券相关性热力图与分散化建议
- PCA分析：碎石图与解释方差，驱动结构分析
- 波动性分析：HV(20)与GARCH预测方差，风险等级与仓位建议
- 季节性分析：STL分解三图与ACF/PACF，自相关与周期性解释
- 风险-收益聚类分析：收益-波动聚类与分化程度建议
- 基本面因子暴露分析：
  - 正常回归：系数柱状图与R²、主导因子与建议
  - 自动降级为画像：行业分位柱状图、趋势方向、风格倾向（价值/成长）
- 涨跌概率分析：概率分布直方图与AUC，方向倾向与可靠性提示

## 并行与缓存
- 后台并行：首次分析后，除当前模块外其他模块在后台异步计算并写入缓存
- 行业矩阵构建：多文件并行读取（pyarrow优先）、单文件结果缓存、统一矩阵缓存
- 渲染策略：ready直接渲染；pending显示轻量加载；failed显示原因提示

## 快速开始
1. 安装依赖
   ```bash
   pip install streamlit plotly pandas numpy scikit-learn jinja2 baostock tushare requests arch statsmodels pandas-ta pyarrow
   ```
2. 配置环境（根目录 `.env.local`）
   ```properties
   LLM_PROVIDER=deepseek
   LLM_ENDPOINT=https://api.deepseek.com/v1/chat/completions
   LLM_MODEL=deepseek-reasoner
   LLM_API_KEY=your_key_here
   TUSHARE_TOKEN=your_tushare_token
   ```
3. 启动与访问
   ```bash
   streamlit run app.py
   ```
   - 本地地址：`http://localhost:8501`

## 使用指南
- 单证券分析
  - 侧边栏输入证券 → 点击“开始分析” → 在模块导航间切换观察各模块图表与“小金”解读
  - 基本面因子分析不足时自动切换为画像，不显示工程化错误
- 行业研究
  - 上传行业映射CSV或扫描本地 `data/ohlcv` → 构建行业矩阵 → 相关性/PCA/聚类/因子/概率模块依次就绪
- 报告与摘要
  - 在“K线与指标”模块完成后由“小金”卡片提示：自动生成结论JSON、文本摘要TXT与单证券HTML报告

## 目录结构
```
├─ app.py                      # 前端主程序（模块导航、渲染、并行与缓存）
├─ src/
│  ├─ analysis/                # 指标/统计/回归
│  ├─ visualization.py         # 图表绘制（含因子画像）
│  ├─ explanation.py           # 模块解释（含画像解释）
│  ├─ conclusion.py            # 结论与小金文本封装
│  ├─ mapping.py               # 行业映射加载与解析
│  ├─ report.py                # HTML 报告渲染
│  └─ data_io.py               # 数据采集与导出
├─ tools/llm_conclusion.py     # LLM调用（小金建议、追问、模板）
├─ templates/                  # 报告模板
├─ data/                       # ohlcv/ fundamentals/
└─ export/                     # PNG/ JSON/ TXT/ HTML
```

## 功能总览（不遗漏）
- 单只证券分析（K线与指标、ADX、趋势结论）
- 行业内多证券横向分析（行业基准、核心指标对比、自动标签、排名与梯队）
- 行业内相关性与分散性（热力图、分散性小结、等权组合波动对比）
- PCA主因子分析（解释方差、碎石图）
- 波动性分析（年化历史波动、GARCH下一期方差）
- 季节性分解（STL/ACF/PACF）
- 风险-收益聚类（收益/波动聚类、分化程度说明）
- 基本面因子暴露（因子回归）与自动降级为画像（分位数与趋势）
- 涨跌概率（逻辑回归、AUC与概率分布）
- 行业评分与综合分数（盈利/偿债/成长/回报四维加权）
- 小金行业投资结论（通用模板，按综合评分排序与一句话解读）
- 报告导出（评分CSV、行业文本报告、个股详细报告TXT、单证券/行业HTML）
- 批量财报采集（四表）与行业映射解析（CSV）
- 并行后台计算与模块缓存（提升交互性能）
- LLM不可用自动降级为规则化文本

## 文件与模块说明（逐文件）
- 根目录
  - app.py：Streamlit前端入口，模式切换（单只/行业），模块渲染、并行与缓存、导出与“小金”集成
  - analysis.py：批量后台计算协调（波动性、季节性、相关性、PCA、聚类、因子回归、概率）
  - advisor.py：模块建议文本缓存与生成封装
  - ui.py：统一的分析卡片、顾问文本与追问交互组件
  - data.py：行业映射读入、代码解析、行情数据读取（baostock接口封装）
  - server.py：可选的后端服务入口（用于非Streamlit场景）
  - requirements.txt：项目依赖
  - stock_industry.csv：默认行业映射示例
  - README.md：项目说明文档
  - .streamlit/config.toml：Streamlit运行配置
  - config.yaml：通用配置样例
  - 数据目录 data/ohlcv/*.csv：本地行情数据（按 ts_code 命名）
  - export/：图表PNG、结论JSON、文本摘要TXT、HTML报告输出
  - docs/Investment_Analysis/*：对齐/设计/任务/验收/最终报告（6A工作流文档）
  - web/index.html, web/static/main.js：静态页面样例（可用于嵌入/预览）
  - templates/report.html.j2, templates/single_report.html.j2：Jinja2报告模板
  - 脚本 scripts/：
    - run_demo.py：演示运行脚本
    - export_ohlcv.py：导出行情数据
    - render_reports.py：批量渲染报告
    - test_llm.py：LLM调用测试
    - classify_ohlcv_by_industry.py：按行业归类行情
    - test_cache.py：缓存相关测试
  - tests/test_kline.py：K线与指标模块的示例测试
  - @startuml.txt：UML草图入口
  - 智能证券分析系统需求规格说明书.md、数据分析报告模板.md、2025数据分析课程作业.md：项目与课程文档

- src/
  - __init__.py：包入口
  - cache.py：通用缓存封装（会话/模块缓存）
  - storage.py：通用存取辅助
  - scheduler.py：任务调度占位（可用于后台队列）
  - fundamentals.py：
    - compute_symbol_metrics：从四表提取基本面指标（ROE/EPS/毛利率/负债率/流动/速动/现金流比、增长、ROE代理、股息率）
    - compute_industry_scoring：行业评分（盈利/偿债/成长/回报）与综合加权排序
    - generate_text_report：个股详细报告（规则化文本）
  - conclusion.py：
    - build_facts：汇总事实数据（价格、均线斜率、相关性、PCA、RSI）
    - generate_industry_analysis：行业通用模板文本（核心判断、优选标的、依据、风险、策略、复盘、总结）
  - explanation.py：
    - explain_*：各模块解释卡片（趋势/波动/相关性/PCA/季节性/聚类/画像）
    - explain_factor_portrait：画像解释（分位均值、优势因子、趋势上下行与建议）
  - visualization.py：
    - plot_*：K线、相关性热力图、PCA解释、STL分解、ACF/PACF、聚类、回归系数、概率分布、因子画像
  - mapping.py：行业映射加载与标准化、查询解析
  - data_io.py：ts<->baostock代码转换、行情抓取、财报四表采集、token校验
  - report.py：单证券与行业报告渲染与写盘
  - analysis/
    - stats.py：相关性、PCA、KMeans、回归（线性/逻辑）、标签生成（高收益高风险/稳健/高波动/性价比）
    - timeseries.py：SMA/EMA/RSI、ADX、年化收益/波动、GARCH、STL、ACF/PACF、回撤、52周接近度、ROC
  - modules/（模块化计算/渲染封装示例，便于注册与扩展）
    - kline.py、correlation.py、pca.py、volatility.py、seasonality.py、clustering.py、factor_portrait.py、probability.py

- storage/
  - repository.py：模块结果的读写与陈旧判断（is_stale）、统一payload构建
  - local_store.py：本地存储适配（读写缓存/模块结果）
  - __init__.py：包入口

- registry/
  - dispatcher.py：前端请求到后台模块执行的分发器（run_selected_modules）
  - analysis_registry.py：分析模块注册表（名称到函数映射）
  - __init__.py：包入口

- schemas/
  - module_status.py：模块状态数据结构（pending/ready/failed）
  - analysis_result.py：分析结果数据结构（统一输出格式）

- core/
  - __init__.py：包入口
  - cache.py：底层缓存抽象
  - data_loader.py：数据加载加速与统一入口
  - returns.py：收益率相关计算
  - time_window.py：时间窗口工具
  - validator.py：输入/配置校验
  - logger.py：日志适配

- frontend_adapter/
  - formatter.py：前端显示格式化工具（指标/单位/提示）

- tools/
  - llm_conclusion.py：LLM调用封装（结论生成、模块建议、追问）

- web/
  - index.html、static/main.js：静态页面展示（可用于框架外预览）

## 运行方式与入口说明
- 交互式前端：`streamlit run app.py`
- 批处理脚本：scripts/ 下各工具（数据导出、报告渲染、演示）
- 报告模板：templates/ 下的 Jinja2 模板按 report.py 渲染

## 数据要求与采集
- 行业映射：CSV需包含 symbol, industry（可拓展 name）
- 行情数据：data/ohlcv 目录（自动发现），缺少时由 data_io 抓取
- 财报四表：fina_indicator.csv、balancesheet.csv、income.csv、cashflow.csv（缺少将影响基本面/行业评分）

## 扩展指南
- 新增分析模块：在 src/modules/ 添加模块实现与解释/可视化，更新 registry/analysis_registry.py 完成注册
- 新增报告：更新 templates/ 并在 src/report.py 新增渲染逻辑
- 调整“小金”模板：改写 src/conclusion.py 的 generate_industry_analysis

## 配置与安全
- 环境变量：LLM与Tushare密钥仅在 `.env.local` 存储，不提交版本库
- 数据合规：只使用公开数据与教学用途数据集；敏感信息不在日志或报告中输出
- 依赖版本：建议固定主依赖版本并在CI/CD中进行安全扫描

## 性能与体验
- 后台并行与缓存：减少切换等待、提升感知性能
- 渲染精简：K线只生成当前区间；运行态不写静态图（导出时生成）
- 计算窗口：建议行业矩阵默认近252交易日，以兼顾稳定与性能

## 质量与扩展
- 单元测试：建议为关键分析函数编写测试（指标计算、矩阵构建、回归/聚类）
- 扩展模块：统一按照分析/可视化/解释/前端四层规范进行扩展与注册
- 设计原则：结果优先、过程延后；并行计算、按需展示；解释层与计算层解耦

## 常见问题
- 未配置 Tushare：财报相关模块不可用，其余模块正常
- 模块无内容：首次需点击“开始分析”；行业模块需要行业内多证券CSV
- LLM不可用：自动回退到规则化文本，功能不受影响

## 许可与支持
- 许可：仅用于课程与教学场景，如需商用请先联系维护者
- 支持：提交 Issue 或联系维护人邮箱（企业内可接入工单系统）
