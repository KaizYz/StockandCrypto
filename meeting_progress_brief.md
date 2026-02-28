# Meeting Progress Brief / 项目阶段汇报

## 1) Project Title / 项目名称
- **Multi-Market Forecast Dashboard**  
- Markets: **Crypto + CN A-share + US Equity**

## Requirement Mapping (From Instructor Prompt) / 教师要求逐条映射
| Instructor Requirement | Covered in this file | Section |
|---|---|---|
| State your MVP | Yes | Section 2 |
| Discuss what you have done so far | Yes | Section 3 |
| Possible software applications | Yes | Section 9 |
| Datasets to be used (links acceptable) | Yes | Section 6 |
| Connection between datasets | Yes | Section 7 |
| Other important aspects / issues | Yes | Section 8 |
| What to show on laptop during meeting | Yes | Section 10 |
| Visual aid (ERD/Schema) if no DB | Yes (schema-style data flow provided) | Section 7 |

## 2) MVP Statement / MVP 定义
**EN**
- Build an end-to-end forecasting and decision-support MVP that ingests market data, generates directional and interval forecasts, maps them to execution plans, and visualizes everything in one dashboard.

**中文**
- 搭建一个端到端的预测与交易决策支持 MVP：从数据采集到特征/标签、模型训练与校准、预测输出、策略映射、回测与监控，并在统一仪表盘中展示。

## 3) What Has Been Done So Far / 当前已完成内容
**EN**
1. End-to-end pipeline implemented (`src/*`): ingestion, preprocessing, features, labels, split, HPO, train, calibrate, predict, backtest, drift, reporting.
2. Multi-market universe implemented:
   - Crypto: BTC/ETH/SOL + top100 ex-stablecoins
   - CN: SSE constituents + CSI300
   - US: Dow30 + Nasdaq100 + S&P500
3. Streamlit dashboard implemented (`dashboard/app.py`):
   - Crypto / CN A-share / US pages
   - Session forecast pages
   - Selection-Research-Tracking
   - Paper Trading-Execution
4. Governance artifacts are generated under `data/processed/` (go-live decision, holdout report, drift monitor, model status, HPO trials, etc.).

**中文**
1. 已实现完整流水线（`src/*`）：数据采集、预处理、特征、标签、切分、调参、训练、校准、预测、回测、漂移监控、报告输出。
2. 已实现多市场标的池：
   - 加密：BTC/ETH/SOL + 市值前100（剔除稳定币）
   - A股：上证成分 + 沪深300
   - 美股：道琼斯30 + 纳指100 + 标普500
3. 已实现仪表盘（`dashboard/app.py`）：
   - Crypto/A股/美股页面
   - 交易时段预测页面
   - Selection/Research/Tracking 页面
   - Paper Trading/Execution 页面
4. 已产出治理与审计文件（`data/processed/`）：go-live 决策、holdout 报告、drift 报告、model status、HPO 试验等。

## 4) Current Evaluation Snapshot / 当前评估快照
**EN**
- Latest holdout report (`data/processed/holdout_report.json`, generated at **2026-02-11 12:56:25 UTC**):
  - Hourly direction accuracy: about **0.507 ~ 0.523**
  - Daily direction accuracy: up to **0.670** at longer horizon
  - Tracked metrics include Brier score, ECE, and interval coverage.
- Release signoff (`release_signoff.md`, **2026-02-07 21:21:33 UTC**) is currently **NO-GO** due to drift and risk-adjusted stability thresholds.

**中文**
- 最新 holdout 报告（`data/processed/holdout_report.json`，时间 **2026-02-11 12:56:25 UTC**）：
  - 小时级方向准确率约 **0.507 ~ 0.523**
  - 日线方向准确率在长周期可到 **0.670**
  - 同时跟踪 Brier、ECE、区间覆盖率等指标。
- 发布签核（`release_signoff.md`，**2026-02-07 21:21:33 UTC**）当前为 **NO-GO**，主要原因是漂移告警与风险调整后稳定性未达阈值。

## 5) Software Stack / 软件与工程栈
- Python, Pandas, NumPy, scikit-learn
- LightGBM / XGBoost
- Streamlit + Plotly
- AkShare + Requests (data ingestion)
- Config-driven pipeline (`configs/config.yaml`)

## 6) Datasets and Sources / 数据集与数据源
> This is a dataset-driven pipeline (file-based), not a relational DB schema.
> 本项目是“数据集驱动 + 文件型数据流”，非传统关系型数据库。

**Core sources / 核心来源**
1. Binance API (klines/ticker)
   - `https://api.binance.com/api/v3/klines`
   - `https://api.binance.com/api/v3/ticker/price`
2. Yahoo Finance (quote/chart/summary)
   - `https://query1.finance.yahoo.com/v7/finance/quote`
   - `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`
   - `https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}`
3. Eastmoney quote API (CN realtime)
   - `https://push2.eastmoney.com/api/qt/stock/get`
4. CoinGecko (crypto market-cap and price)
   - `https://api.coingecko.com/api/v3/coins/markets`
   - `https://api.coingecko.com/api/v3/simple/price`
   - `https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart`
5. Constituents / 成分股来源
   - CN: AkShare index constituents
   - US: public index constituent tables (Dow/Nasdaq100/S&P500)

## 7) Data Connection (Schema-like View) / 数据连接关系（可视化口径）
**EN**
`Raw Market Data -> Preprocessing/Quality -> Features -> Labels -> Time Split/Folds -> Train/HPO/Calibrate -> Predictions -> Policy Signals -> Backtest/Drift/Governance -> Dashboard`

**中文**
`原始行情数据 -> 预处理/质量检查 -> 特征 -> 标签 -> 时序切分 -> 训练/调参/校准 -> 预测 -> 策略信号 -> 回测/漂移/治理 -> 仪表盘展示`

## 8) Key Risks and Issues / 当前主要问题
**EN**
1. Drift alerts remain high in current release.
2. Risk-adjusted stability (Sharpe consistency across folds) is not yet passing go-live thresholds.
3. Some live source latency/availability affects refresh speed and fallback mode behavior.

**中文**
1. 当前版本漂移告警偏高。
2. 风险调整后稳定性（尤其跨折 Sharpe 稳定性）未达到上线阈值。
3. 部分实时数据源存在延迟或可用性问题，影响页面刷新速度与 fallback 行为。

## 9) Next Steps / 下一步计划
**EN**
1. Continue retraining + calibration in batched runs with strict stop criteria.
2. Improve drift robustness and confidence gating.
3. Optimize dashboard loading performance and fallback transparency.
4. Strengthen execution explainability (reason codes, gate diagnostics, TP/SL hit probabilities).
5. Possible software applications:
   - Research and screening workstation for multi-market assets
   - Paper-trading decision console for strategy validation
   - Monitoring cockpit for model governance and drift alerts

**中文**
1. 按批次持续训练与校准，并按停训标准收敛。
2. 提升抗漂移能力与置信度门控质量。
3. 优化页面加载性能与降级路径可解释性。
4. 加强执行解释：阻断原因、规则诊断、TP/SL 触发概率等。
5. 可能的软件应用方向：
   - 多市场研究与筛选工作台
   - 纸上交易（Paper Trading）决策控制台
   - 模型治理与漂移监控驾驶舱

## 10) What I Will Show in Meeting / 会议现场展示清单
1. Live dashboard flow (market select -> forecast -> trade plan -> backtest -> tracking).
2. Pipeline commands and generated artifacts under `data/processed/`.
3. Current NO-GO evidence and concrete remediation plan.
4. Data source links and cross-market data connection logic.

---

## Quick 60-second version / 60秒口播版
**EN**
“My MVP is a multi-market forecasting and decision dashboard. I have completed the full ML pipeline and a Streamlit product layer for crypto, China A-shares, and US equities. The system outputs direction probabilities, quantile intervals, execution plans, and backtesting diagnostics. Current status is NO-GO due to drift and stability thresholds, and my next milestone is retraining/calibration plus drift-robust gating to move toward paper-trading readiness.”

**中文**
“我的 MVP 是一个多市场预测与交易决策仪表盘。我已经完成从数据到模型到前端展示的全链路，实现了加密、A股、美股三市场输出：方向概率、区间预测、执行计划和回测诊断。当前签核状态是 NO-GO，主要问题是漂移和稳定性阈值未通过。下一阶段重点是分批重训+校准+抗漂移门控，推进到稳定的 paper trading 准入状态。”
