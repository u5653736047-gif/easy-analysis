# 小析易 · Agent 基础框架

基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 搭建的最小可用 Agent 框架，
面向项目「小析易」——为教师提供更全面、更深入的学情分析。

## 能力对照

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 流式输出 | `agent_core.chat_streamed()` | `Runner.run_streamed()` 逐 token 返回 `stream_events()` |
| 思考链展示 | `agent_core._StreamRenderer.reasoning_delta()` | 监听 `response.reasoning_summary_text.delta` 等事件，灰色流式打印 reasoning 摘要 |
| 最终答案 | `agent_core._StreamRenderer.text_delta()` | 监听 `response.output_text.delta`，逐 token 打印 |
| 历史会话 | `agent_core.create_session()` | `SQLiteSession` 持久化到 `conversation_history.db`，重启恢复，按 `session_id` 区分 |
| 工具调用展示 | `agent_core._dispatch_event()` | 监听 `tool_call_item` / `tool_call_output_item`，打印 `🔧 调用工具(参数)` 与 `📤 工具返回(结果)` |

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

## 目录结构

```
├── main.py            # CLI 入口：交互循环、参数解析、API Key 检查
├── agent_core.py      # 核心框架：Agent 构建 / 会话 / 流式运行 / 事件渲染
├── tools.py           # 示例工具（@function_tool）
├── requirements.txt   # 依赖清单
├── .env.example       # 环境变量模板
└── conversation_history.db  # 运行时生成的会话历史（已 gitignore）
```

## 交互命令

- `exit` / `quit`：退出
- `new`：开启新会话（换一个会话 ID，互不影响）

## 如何扩展

- **新增工具**：在 `tools.py` 写一个带类型标注和 docstring 的函数并加 `@function_tool`，
  然后加入 `build_agent()` 的 `tools` 列表；
- **改人设/职责**：编辑 `agent_core.AGENT_INSTRUCTIONS`；
- **换持久化后端**：`create_session()` 目前用 `SQLiteSession`，
  可换成 SDK 提供的 `RedisSession`、`DaprSession` 等，接口不变；
- **多 Agent 协作**：渲染器已预留 `AgentUpdatedStreamEvent` 分支，往 `Agent(handoffs=[...])` 加子 Agent 即可看到切换提示；
- **接 Web 服务**：`chat_streamed()` 是 async 生成式接口，可直接搬到 FastAPI 的 SSE/WebSocket 路由里。
