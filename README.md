# 小析易 · Agent 基础框架

基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 搭建的最小可用 Agent 框架，
面向项目「小析易」——为教师提供更全面、更深入的学情分析。

## 能力对照

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 流式输出 | `agent_core.chat_streamed()` | `Runner.run_streamed()` 逐 token 返回 `stream_events()` |
| 思考链展示 | `agent_core._StreamRenderer.reasoning()` | 分流 `response.reasoning_summary_text.delta` / `response.reasoning_text.delta`，灰色流式打印 |
| 最终答案 | `agent_core._StreamRenderer.content()` | 分流 `response.output_text.delta`，逐 token 打印 |
| 历史会话 | `agent_core.create_session()` | `SQLiteSession` 持久化到 `conversation_history.db`，重启恢复，按 `session_id` 区分 |
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

## 目录结构

```
├── main.py            # CLI 入口：交互循环、参数解析、API Key 检查
├── agent_core.py      # 核心框架：Agent 构建 / 会话 / 流式运行 / 事件渲染
├── tools.py           # 示例工具（@function_tool）
├── tests/smoke_test.py  # 冒烟测试（无需 API Key）
├── requirements.txt   # 依赖清单
├── .env.example       # 环境变量模板
└── conversation_history.db  # 运行时生成的会话历史（已 gitignore）
```

## 交互命令

- `exit` / `quit`：退出
- `new`：开启新会话（换一个会话 ID，互不影响）

## 测试

```bash
.venv/bin/python tests/smoke_test.py
```

无需真实 API Key：
- **单元级**：用真实 openai 事件对象驱动 `_render_raw_event()`，验证按 `event.data.type`
  分流渲染、工具调用去重（`arguments.done` 与 `output_item.done` 重复到达只打印一次）、
  未知事件安全忽略；
- **端到端**：SDK 官方 `ScriptedModel` 走真实 `Runner.run_streamed()` 管线
  （reasoning 增量 / 工具真实执行 / 结果回传 / SQLiteSession 历史写入），
  共 18 项断言。

## 如何扩展

- **新增工具**：在 `tools.py` 写一个带类型标注和 docstring 的函数并加 `@function_tool`，
  然后加入 `build_agent()` 的 `tools` 列表；
- **改人设/职责**：编辑 `agent_core.AGENT_INSTRUCTIONS`；
- **换持久化后端**：`create_session()` 目前用 `SQLiteSession`，
  可换成 SDK 提供的 `RedisSession`、`DaprSession` 等，接口不变；
- **多 Agent 协作**：给 `Agent(handoffs=[...])` 加子 Agent 后，在 `chat_streamed()` 的
  事件循环里加一个 `AgentUpdatedStreamEvent` 分支即可渲染切换提示；
- **接 Web 服务**：`chat_streamed()` 是 async 接口，可直接搬到 FastAPI 的 SSE/WebSocket 路由里。
