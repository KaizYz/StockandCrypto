# 交易时间段预测：模拟K线（平均路径）功能方案【终稿】

> 目标：在“交易时间段预测（Crypto）”与“交易时间段预测（指数）”页面新增 **未来模拟K线（Mean Path）** 模块，把现有预测输出 `p_up / q10 / q50 / q90` 变成可解释、可复现、可回测的“未来价格路径 + 风险区间”，并为后续“TP/SL 触发概率、胜率/盈亏比评估”打基础。  
> 声明：模拟K线是基于当前概率分布的统计路径，不代表未来真实轨迹，仅用于风险评估与场景演练。

---

## 1. 目标（需求核心）

在以下两个页面新增“未来模拟K线”模块：

1) `交易时间段预测（Crypto）`  
2) `交易时间段预测（指数）`

功能目标：

1. 将每步预测的 `p_up / q10 / q50 / q90` 转换为“未来价格路径”可视化（蜡烛图 + 区间带）。
2. 使用多路径模拟（Monte Carlo）生成未来若干步的价格路径与 K 线。
3. 默认输出“平均路径（Mean Path）”，满足“取平均值”的展示需求，并同时输出风险带（q10~q90）。
4. 为 P1 升级项预留：**TP/SL 触发概率（先触发TP还是SL）**。

---

## 2. 页面效果（用户视角）

### 2.1 新模块名称
`模拟K线（未来路径：平均值） / Simulated K-line (Mean Path)`

### 2.2 推荐位置
放在“时段卡片 + 热力图”之后，“TopN榜单”之前。

### 2.3 模块组成
1. 参数区（可折叠）
2. 模拟K线图（主图：蜡烛 + 均值收盘线 + 区间带 q10~q90）
3. 关键摘要卡（终点价、预期涨跌幅、风险带、最大回撤/上冲）
4. 明细表（每一步的 ts + OHLC + close_q10/close_q90 + step_ret）

（可选）图2：终点价格分布（P_T histogram）

---

## 3. 参数设计（MVP）

### 3.1 参数项
- `n_steps`：默认 24（未来 24 个“可交易步”）
- `n_paths`：默认 300（可选 100/300/500）
- `seed`：默认 42（复现）
- `agg`：默认 `mean`（可选 `median`）
- `vol_scale`：默认 1.0（范围建议 0.5~2.0）
- `sigma_floor`：默认 0.0003（防止波动为0，约 0.03%）
- `sigma_cap`：默认 0.03（上限，约 3%/step，防爆）
- `wick_cap`：默认 0.015（影线最大 1.5%）

### 3.2 默认说明
- `n_steps=24` 对 Crypto 表示未来 24 小时；对指数表示未来 24 个“可交易小时步”（严格按交易日历过滤）。
- `seed` 固定时同输入可复现；修改 seed 应产生不同路径。

---

## 4. 数据输入（复用现有时段页）

页面已有 per-step 预测表 `hourly_df`（或 index_step_df），字段至少包含：

- `p_up`
- `q10_change_pct`
- `q50_change_pct`
- `q90_change_pct`
- `hour_bj`（或 market hour）
- `is_trading_hour`（指数页必须；Crypto 可认为全时段交易）
- `forecast_timestamp_utc`（建议加入，作为缓存key）

起点价格：
- `current_price`（页面已存在）
- 建议额外显示：
  - `price_source`
  - `price_timestamp_market`
  - `price_timestamp_utc`

---

## 5. 模拟方法（核心定义）

### 5.0 收益率形式（建议用 log-return）
为稳定性与避免负价，建议用 log-return 叠乘：

- 若预测输出为 simple return `r`（百分比变化），转换为：
  - `lr = log(1 + r)`
- 价格更新：
  - `P_t = P_{t-1} * exp(lr_t)`

> 若后端直接输出 log-return 的分位数，可跳过转换，但需统一口径。

---

### 5.1 单步分布构建：Split-Normal / Truncated Sampling（推荐）

目标：同时利用 `p_up`（方向概率）与 `q10/q50/q90`（幅度区间）。

对每步 t：

- `m_t = q50_t`（中位数）
- 左侧尺度（下跌侧）：
  - `sigma_L = max((m_t - q10_t) / 1.2816, sigma_floor)`
- 右侧尺度（上涨侧）：
  - `sigma_R = max((q90_t - m_t) / 1.2816, sigma_floor)`

并进行上限裁剪：
- `sigma_L = min(sigma_L * vol_scale, sigma_cap)`
- `sigma_R = min(sigma_R * vol_scale, sigma_cap)`

采样步骤：
1) 方向采样：
- `dir_t ~ Bernoulli(p_up_t)`  
  - `dir_t=1` 表示 up；`dir_t=0` 表示 down

2) 幅度采样（截断以确保方向一致）：
- 若 up：从 `Normal(mean=m_t, std=sigma_R)` 采样并截断为 `ret_t >= 0`
- 若 down：从 `Normal(mean=m_t, std=sigma_L)` 采样并截断为 `ret_t <= 0`

3) 得到单步 simple return：
- `ret_t`（单位：比例，如 0.002 表示 0.2%）

4) 转为 log-return：
- `lr_t = log(1 + ret_t)`

说明：
- split-normal 让左右波动不同，更贴近 q10~q90 的不对称。
- truncation 保证与 `p_up` 的方向逻辑一致。

---

### 5.2 价格路径生成（向量化）
每条路径 i：

- `P_0 = current_price`
- `P_t = P_{t-1} * exp(lr_t^i)`

实现推荐一次性采样矩阵：
- `LR` shape: `[n_paths, n_steps]`
- `P` shape: `[n_paths, n_steps+1]`

---

### 5.3 路径转 K 线（每一步一根）

对每条路径 i、每一步 t：
- `open_t^i = P_{t-1}^i`
- `close_t^i = P_t^i`

影线噪声（与波动相关，且有 cap）：
- `wick_amp_t = clip(abs(N(0, wick_scale * sigma_step_t)), 0, wick_cap)`
- `high_t^i = max(open_t^i, close_t^i) * (1 + wick_amp_t)`
- `low_t^i  = min(open_t^i, close_t^i) * (1 - wick_amp_t)`

建议：
- `wick_scale` 默认 0.6（可写死或参数化）
- `wick_cap` 防“离谱针”

---

### 5.4 多路径聚合：生成“合法”的平均K线（MVP 必须）

**不要直接 mean(OHLC)**（会产生不合法K线）。  
推荐聚合方式如下：

1) 聚合 close：
- `close_t = agg(close_t^i)`（默认 mean，可选 median）
- 风险带：
  - `close_q10_t = quantile(close_t^i, 0.10)`
  - `close_q90_t = quantile(close_t^i, 0.90)`

2) 聚合 open（保证连续性）：
- `open_0 = current_price`
- `open_t = close_{t-1}`（聚合后的上一根 close）

3) 聚合 high/low（用分位数更稳，避免均值不合法）：
- `high_t_raw = quantile(high_t^i, 0.70)`
- `low_t_raw  = quantile(low_t^i, 0.30)`

4) 合法性修正（强制满足高低点约束）：
- `high_t = max(high_t_raw, open_t, close_t)`
- `low_t  = min(low_t_raw,  open_t, close_t)`

输出：
- `open, high, low, close`（代表性“平均路径K线”）
- `close_q10, close_q90`（风险带）

---

## 6. 指数页交易时段约束（必须）

### 6.1 原则：指数页必须遵守交易日历（含 DST）
- **不能硬编码** “美盘=北京时间 00:00-07:59” 来当作指数交易时段。  
- 必须用交易日历生成可交易时间段，再转换展示时区。

推荐实现：
- 使用 `pandas_market_calendars`（或同类）获取：
  - 美股：NYSE/NASDAQ 交易日历（RTH）
  - A股：上交所/深交所日历（含午休）

### 6.2 规则
1) 非交易时段不参与模拟
2) 非交易时段不显示模拟K线
3) 时间轴只显示“可交易步”
4) 展示时区：
   - A股指数：北京时间（Asia/Shanghai）
   - 美股指数：美东时间（America/New_York），或在页面显式标注换算后的北京时间（两者择一但必须标注）

---

## 7. 输出字段（统一 DataFrame Schema）

返回 DataFrame（供图表 + 表格共用）：

- `step_idx`
- `ts_utc`
- `ts_market`（Crypto/沪深：bj；美股指数：et 或标注后的 bj）
- `open`
- `high`
- `low`
- `close`
- `close_q10`
- `close_q90`
- `mean_step_ret`（聚合后的单步收益，可用 `close/open - 1`）
- `cum_ret_from_start`（`close/current_price - 1`）
- `active_session`（亚盘/欧盘/美盘 或 RTH/非RTH）

（可选调试输出）
- `sigma_L, sigma_R, p_up`（每步参数，便于解释）

---

## 8. 图表规范（页面表现）

### 图1：模拟K线（主图）
- 蜡烛图：`open/high/low/close`
- 叠加线：`close`（mean/median path close）
- 区间带：`close_q10 ~ close_q90`
- 标注：起点价 current_price、终点 close_T、累计收益 cum_ret

### 图2（可选）：终点分布
- 终点价格 `P_T` 分布直方图
- 标注均值/中位数/10%/90%

---

## 9. 代码落点与接口（建议）

新增模块：
- `src/markets/simulated_kline.py`

建议函数：

1) `simulate_future_ohlc(...) -> pd.DataFrame`
- 输入：
  - `current_price`
  - per-step df（含 p_up/q10/q50/q90 + ts）
  - `n_steps, n_paths, seed, agg, vol_scale, sigma_floor, sigma_cap, wick_cap`
  - `calendar`（指数页需要）
- 输出：
  - 第 7 章 schema 的 df

2) `build_simulation_summary(df, current_price) -> Dict[str, float]`
- 输出摘要：
  - `end_price`
  - `end_ret_pct`
  - `max_up_from_start`
  - `max_dd_from_start`
  - `avg_band_width`（均值区间宽度）

（P1 预留）
3) `estimate_tp_sl_hit_prob(paths_prices, entry, tp, sl) -> Dict`
- 输出：
  - `p_hit_tp_first`
  - `p_hit_sl_first`
  - `p_no_hit`
  - `expected_steps_to_tp/sl`

页面接入：
- `dashboard/app.py`
  - `_render_crypto_session_page()`
  - `_render_index_session_page()`

---

## 10. 缓存与性能（必须）

### 10.1 Streamlit 缓存
- `@st.cache_data(ttl=300)` 缓存模拟结果
- cache key 必须包含：
  - `symbol/index`
  - `forecast_timestamp_utc`
  - `n_steps/n_paths/seed/vol_scale/agg`
  - `market_mode`（crypto/index）
  - `calendar_version`（指数页）

### 10.2 性能策略
- 用 numpy 向量化生成 `[n_paths, n_steps]` 的采样矩阵
- 避免 python for 循环逐路径逐步累乘

---

## 11. 验收标准（Done Definition）

完成后应满足：

1. Crypto 时段页可看到模拟K线（默认 24 steps / 300 paths / mean）。
2. 指数时段页可看到模拟K线，并严格遵守交易日历（含 DST / 午休）。
3. 修改 `seed` → 结果变化；固定 `seed` → 可复现。
4. 增大 `n_paths` → 路径更平滑（统计稳定性提升）。
5. 图中有 q10~q90 风险带，并展示终点与累计收益摘要。
6. 页面有风险提示文案（见第 12 章）。

---

## 12. 风险提示文案（直接可用）

中文：
> `模拟K线基于当前概率输出（p_up/q10/q50/q90）的统计路径，不代表未来真实价格轨迹，仅用于风险评估与场景演练。`

English:
> `Simulated candles are scenario paths derived from current probability outputs (p_up/q10/q50/q90), not exact future prices. Use for risk/scenario analysis only.`

---

## 13. 后续升级项（P1 / P2）

### P1（强烈推荐，直接提升“交易决策可用性”）
1. “均值/中位数”切换（mean/median）
2. 加入“新闻冲击开关”（event_risk=True 时放大波动：`vol_scale` 动态上调）
3. 与入场/止损/止盈联动，输出触发概率（先 TP 还是 SL）：
   - `p_hit_tp_first / p_hit_sl_first / expected_time`
4. 历史后验验证（walk-forward 测试段）：
   - coverage：真实 close 是否落在 `[close_q10, close_q90]`
   - calibration：mean path error 分布

### P2（增强：更像机构级场景分析）
1. 多模型路径混合（forecast + seasonality + event shock 加权）
2. 路径分簇（乐观/中性/悲观）
3. 对不同 regime（趋势/震荡/高波动）分别校准模拟参数

---
