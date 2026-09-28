# 小析易 · Agent 基础框架

基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 搭建的最小可用 Agent 框架，
面向项目「小析易」——为教师提供更全面、更深入的学情分析。

## 能力对照

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 流式输出 | `agent_core.chat_streamed()` | `Runner.run_streamed()` 逐 token 返回 `stream_events()` |
| 思考链展示 | `agent_core._StreamRenderer.reasoning()` | 分流 `response.reasoning_summary_text.delta` / `response.reasoning_text.delta`，灰色流式打印 |
| 最终答案 | `agent_core._StreamRenderer.content()` | 分流 `response.output_text.delta`，逐 token 打印 |
| 历史会话 | `session_manager.SessionManager` | 与 `SQLiteSession` 共用同一 SQLite 库，元数据/状态机/级联删除/自动标题，重启恢复 |
| 工具调用展示 | `agent_core._render_raw_event()` | 按 `event.data.type` 分流：`output_item.added` 登记 → `function_call_arguments.done` 打印 🔧 名称+参数（去重）；`tool_output` 事件打印 📤 执行结果 |

## 快速开始

```bash
# 1. 配置 API Key（二选一）
export OPENAI_API_KEY=sk-...
# 或：cp .env.example .env 后填入 OPENAI_API_KEY

# 2. 启动交互式对话（使用已有 .venv 可跳过前两步）
source .venv/bin/activate
python main.py

# 可选参数
python main.py --session s1        # 指定会话 ID（历史按 ID 分别持久化）
python main.py --model o4-mini     # 切换模型
python main.py --no-reasoning      # 关闭思考链，加快响应
```

> **关于「思考链」**：只有推理模型（`gpt-5` / `gpt-5-mini` / `o4-mini` 等）会产生 reasoning 事件。
> 若使用 `gpt-4o` 等非推理模型，框架会正常流式输出答案与工具调用，但没有思考链内容。
> 默认模型可通过 `OPENAI_MODEL` 或 `--model` 覆盖。

> **关于第三方中转**：`.env` 支持两套变量名，`OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_MODEL`
> 或 `API_KEY`/`BASE_URL`/`MODEL_NAME`（后者自动映射为前者）。
> 端点需支持 Responses API（`/responses`）；使用非官方端点时框架会自动禁用 SDK tracing 上报。

## Jev 题型分类工具

`jev/question_type.py` 把「题干 -> 题型」这一封闭集判断交给 [Jev](https://typesafe.ai)（TypeSafe
System One 结构化决策模型）：输入题干文本 + 固定选项集，返回选项 + 置信度 + 完整概率分布。

- **固定 taxonomy**：`QUESTION_TYPES`（选择题/判断题/填空题/解答题/证明题/其他），
  内容与顺序都是代码常量，不允许运行时自由发挥（缓解 Jev 的选项顺序敏感性）
- **needs_review 判定**：置信度 < 0.70 或前两名概率差 margin < 0.15 时标记，
  提示 agent 复核后再写入结果
- **自动降级**：未配置 Key / 网络失败 / 上游 5xx 时返回 `source=error` 的结构化 JSON，
  agent 转为自行判断，主流程不中断（已内置 4 次退避重试）
- **在框架内的位置**：`build_agent()` 的 tools 列表，走与其它工具完全相同的
  🔧调用/📤返回 渲染链路

配置（.env，Command Code 网关优先）：

```bash
COMMAND_CODE_BASE_URL=https://api.commandcode.ai/provider/v1/systemone
COMMAND_CODE_API_KEY=sk-...
```

### 可靠性测试

```bash
.venv/bin/python tests/test_jev_question_type.py   # 本地假服务：17 项断言（无网络依赖）
.venv/bin/python tests/jev_question_type_real_check.py  # 真实 Jev：12 道模拟题评测
```

真实评测初测结果（12 题，含 3 个易混淆用例）：网络可达 12/12，首判准确 10/12，
**2 个误判全部被低置信机制捕获**（needs_review 自动降级，不污染结果）。
注意：Command Code 网络偶发重置，脚本内置 3 次外层重试，平均耗时约 5~7s/题。

## Jev 错因分析工具

`jev/error_cause.py` 把「一道错题错在哪」交给 Jev：一次 `system_one` 调用里
**并行组合两类问题**（Jev 不按输出计费，多问题几乎零边际成本）：

| 问题 | 原语 | 用途 |
|---|---|---|
| `primary` | Choice | 主错因（单标签，带概率分布/置信度/margin） |
| `multi::<错因>` × 8 | Noul | 每个错因是否存在的概率，按 `MULTI_CAUSE_THRESHOLD`(0.75) 汇总为 `detected_causes` 多因标签 |

输入 state 由固定模板拼装：**题目 / 参考答案 / 学生作答**三段。

- 固定错因 taxonomy（`ERROR_CAUSES`）：概念不清、计算失误、审题不清、方法错误、
  粗心失误、知识遗忘、表述不完整、其他
- `detected_causes` 的**多因标签**直接服务「学生易错标签可累加」的产品设计
- `needs_review` 阈值：主错因置信度 <0.70 或 margin <0.15 或多因为空
- 降级策略与题型工具一致：异常返回 `source=error`，agent 自行分析

### 可靠性测试

```bash
.venv/bin/python tests/test_jev_error_cause.py      # 本地假服务：18 项断言
.venv/bin/python tests/jev_error_cause_real_check.py  # 真实 Jev：12 道模拟错题评测
```

真实评测结果（12 题，8 类错因，含 4 个易混淆/多因用例）：
- 主错因判定 7/12；**多因标签覆盖 + needs_review 拦截后，无静默误判**
  （4 个主错因不一致的被 needs_review 捕获；1 个主错因不同但正确错因在多因标签中）
- needs_review 触发 9/12——偏保守，适合「标签累加」场景，
  后续可用更多标注数据微调阈值
- 平均耗时 7.6s/题（9 个问题一次调用）

## 目录结构

```
├── main.py            # CLI 入口：交互循环、会话命令编排
├── agent_core.py      # 核心框架：Agent 构建 / 流式运行 / 事件渲染
├── tools.py           # 本地工具（@function_tool）
├── jev/               # Jev 决策工具包
│   ├── client.py      #   Jev API 共享客户端（配置解析/重试，不含业务逻辑）
│   ├── question_type.py #  题型分类工具
│   └── error_cause.py #  错因分析工具（多因标签）
├── session_manager.py # 会话管理：元数据 / 状态机 / 隔离 / 搜索导出（GUI 直接消费）
├── tests/             # 测试（假服务单元测试 + 真实评测）
├── requirements.txt   # 依赖清单
├── .env.example       # 环境变量模板
└── conversation_history.db  # 运行时生成的会话历史（已 gitignore）
```

> **会话存储为什么是一个库**：`agent_sessions` / `agent_messages` 两表由 SDK 的
> `SQLiteSession` 读写（Runner 持久化）；`session_meta` 由 `SessionManager` 维护，
> 通过 `session_id` 外键与前者级联关联——元数据和历史消息永远不会漂移，
> 删会话时两张表一起清。

## 交互命令

- `exit` / `quit`：退出
- `new [标题]`：开启新会话（标题缺省时首条消息自动命名）
- `sessions [a|d]`：列出会话（默认活跃；a=已归档，d=回收站）
- `use <会话ID>`：切换到指定会话，历史自动带上
- `rename <新标题>`：重命名当前会话
- `history [n]`：查看当前会话最近 n 条对话（默认 10）
- `search <关键词>`：按标题/摘要搜索会话
- `archive` / `restore`：归档 / 恢复当前会话
- `delete`：移入回收站（`restore` 可找回）
- `export [路径]` / `import <路径>`：导出 / 导入会话 JSON
- `stats`：会话库统计

会话生命周期：`active -> archived -> active`、`active -> deleted -> active`、
`deleted -> purge（硬删，消息级联清除，不可恢复）`。
启动时不指定 `--session` 会自动恢复最近活跃的会话。

## 测试

```bash
.venv/bin/python tests/smoke_test.py                # 框架回归（含 SessionManager 记账断言）
.venv/bin/python tests/test_session_manager.py     # 会话管理：51 项（状态机/隔离/级联/迁移/并发）
.venv/bin/python tests/test_jev_question_type.py   # 题型工具：17 项
.venv/bin/python tests/test_jev_error_cause.py     # 错因工具：18 项
```

无需真实 API Key：
- **单元级**：用真实 openai 事件对象驱动 `_render_raw_event()`，验证按 `event.data.type`
  分流渲染、工具调用去重（`arguments.done` 与 `output_item.done` 重复到达只打印一次）、
  未知事件安全忽略；
- **端到端**：SDK 官方 `ScriptedModel` 走真实 `Runner.run_streamed()` 管线
  （reasoning 增量 / 工具真实执行 / 结果回传 / SQLiteSession 历史写入），
  共 18 项断言。

## 如何扩展

- **新增本地工具**：在 `tools.py` 写一个带类型标注和 docstring 的函数并加 `@function_tool`，
  然后加入 `build_agent()` 的 `tools` 列表；
- **新增 Jev 工具**：在 `jev/` 下新建模块，复用 `jev/client.py` 的 `make_client()`，
  遵循 `*_raw() / *_impl() / @function_tool` 三层结构与 `source/needs_review/hint`
  降级契约，`.env` 无需任何改动；
- **改人设/职责**：编辑 `agent_core.AGENT_INSTRUCTIONS`；
- **换持久化后端**：`session_manager.open_session()` 目前返回 `SQLiteSession`，
  可换成 SDK 提供的 `RedisSession`、`DaprSession` 等，接口不变；
- **多 Agent 协作**：给 `Agent(handoffs=[...])` 加子 Agent 后，在 `chat_streamed()` 的
  事件循环里加一个 `AgentUpdatedStreamEvent` 分支即可渲染切换提示；
- **接 Web 服务**：`chat_streamed()` 是 async 接口，可直接搬到 FastAPI 的 SSE/WebSocket 路由里；
  `SessionManager` 的所有读接口返回 `SessionInfo`（带 `to_dict()`），可直接序列化为
  GUI 的会话列表/详情/搜索接口；线程安全（RLock + WAL），GUI 多线程调用无额外处理。
