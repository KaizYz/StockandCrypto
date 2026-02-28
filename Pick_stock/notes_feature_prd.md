# Notes Feature 产品需求（V1）

## 1. 功能目标
新增 `Notes Feature`，支持两种使用方式：
1. 交易日志：记录每笔交易为什么下单、回报比（RR）、是否符合策略、复盘结论。
2. 普通笔记：如果用户不想填交易字段，可直接当通用笔记使用。

## 2. 核心功能范围

### 2.1 交易日志模式（Trade Journal）
每条交易日志建议包含以下字段：
- `title`：标题
- `market`：市场（外汇/贵金属/股票/加密）
- `symbol`：交易标的
- `direction`：方向（Long/Short）
- `entry_price`、`stop_loss`、`take_profit`
- `risk_reward_ratio`：回报比（RR）
- `strategy_name`：策略名称
- `strategy_match`：是否符合策略（Yes/No）
- `reason`：下单原因
- `result`：结果（Win/Loss/BE/进行中）
- `review`：复盘总结
- `created_at`、`updated_at`

### 2.2 普通笔记模式（General Notes）
- 只需要 `title + content + tags` 即可保存
- 不强制填写交易相关字段
- 支持按时间、标签、关键词检索

### 2.3 导入与导出
- `Import`：支持从 `CSV / JSON / Markdown` 导入历史笔记或交易日志
- `Export`：支持导出为 `CSV / JSON / Markdown`
- 提供字段映射与导入错误提示（例如缺少必要字段、格式错误）

### 2.4 交易策略分享（Strategy Share）
- 用户可发布自己的交易策略或交易计划（Strategy/Plan）
- 可设置可见性：`Private / Link Only / Public`
- 他人可查看分享内容并进行讨论（后续可扩展点赞、收藏）

### 2.5 分板块群聊（Community Chat）
按市场分房间：
- 外汇（Forex）
- 贵金属（Metals）
- 股票（Stocks）
- 加密（Crypto）

每个房间支持基础能力：
- 发送文本消息
- 关联某条交易日志或策略分享
- 按时间倒序浏览

### 2.6 用户注册与登录（Auth）
- 支持用户 `注册 / 登录 / 退出登录`
- 注册字段建议：`username`、`email`、`password`
- 登录支持：`email + password`（后续可扩展用户名登录）
- 密码必须加密存储（例如 `Werkzeug Password Hash`），禁止明文存储
- 预留基础账户能力：找回密码、修改密码、账号状态（active/disabled）
- 未登录用户只能查看公开内容；创建笔记、策略分享、聊天发言需要登录

## 3. 技术方案（指定栈）
- 后端框架：`Flask`
- ORM：`SQLAlchemy`
- 数据库：`SQLite`（V1 本地轻量部署，后续可迁移 PostgreSQL）

推荐模块划分：
- `auth`：用户登录与权限
- `notes`：笔记与交易日志
- `strategy_share`：策略分享
- `chat`：分板块群聊
- `import_export`：导入导出

## 4. 数据模型建议（V1）
1. `users`
2. `notes`
3. `trade_journal_entries`（与 notes 一对一，可为空）
4. `strategy_posts`
5. `chat_rooms`
6. `chat_messages`

说明：
- 普通笔记只写入 `notes`
- 交易日志写入 `notes + trade_journal_entries`
- 通过 `note_type` 区分 `general` 与 `trade`

## 5. MVP 验收标准
1. 用户可完成注册、登录、退出登录；密码以哈希形式存储。
2. 未登录用户无法创建笔记、发布策略、发送聊天消息。
3. 用户可创建、编辑、删除普通笔记和交易日志。
4. 交易日志可完整记录：下单原因、RR、是否符合策略、复盘结果。
5. 支持至少一种导入格式和一种导出格式（建议 CSV）。
6. 用户可分享一条交易策略，并设置可见性。
7. 四个群聊房间可用（外汇/贵金属/股票/加密），支持基础消息收发。

## 6. 后续迭代方向（V2+）
1. 图表化复盘（胜率、RR 分布、策略命中率）
2. 聊天消息引用策略卡片与交易卡片
3. 分享内容的评论、点赞、收藏
4. 审计日志与敏感词过滤
