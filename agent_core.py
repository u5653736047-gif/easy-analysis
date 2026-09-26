"""「小析易」Agent 基础框架（基于 OpenAI Agents SDK）。

本模块封装四块能力，对应需求：
1. Agent 构建：人设 instructions + 模型 + 工具注入            -> build_agent()
2. 历史会话：SQLiteSession 持久化，重启进程后自动恢复        -> create_session()
3. 流式输出：Runner.run_streamed 逐 token 返回事件          -> chat_streamed()
4. 过程展示：思考链/reasoning、工具调用、最终答案全量渲染     -> _StreamRenderer

事件分发只依赖 SDK 的两种事件类型，后续扩展（handoff 多 Agent、guardrail）
只需在 chat_streamed 的事件循环里增加分支。
"""
from __future__ import annotations

import json
import os
from typing import Any

from agents import Agent, ModelSettings, Runner, SQLiteSession
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from rich.console import Console
from rich.markup import escape

from tools import (
    classroom_score_stats,
    get_current_time,
    summarize_error_types,
)

console = Console()

# --------------------------------------------------------------------------- #
# 默认配置（均可用环境变量 / CLI 参数覆盖）
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gpt-5-mini"          # 推理模型才能展示「思考链」
DEFAULT_DB = "conversation_history.db"  # 会话历史 SQLite 文件
AGENT_NAME = "小析易"

AGENT_INSTRUCTIONS = """
你是「小析易」的智能学情分析助手，服务对象是中小学教师。

你的职责：
1. 帮助教师分析班级考试情况：主要考察题型、各题型正确率、高出错率题型等；
2. 帮助教师归纳学生易错点，为每名学生建立并累加易错标签；
3. 涉及数据计算（班级成绩统计、错因占比等）时，优先调用工具，不要凭空估算。

回答要求：
- 使用简体中文，风格专业、简洁、有条理；
- 给出结论时说明依据（来自哪次考试、哪份数据）；
- 信息不足时，主动向教师追问，而不是臆测。
""".strip()


# --------------------------------------------------------------------------- #
# 1. Agent 构建
# --------------------------------------------------------------------------- #
def build_agent(model: str | None = None, *, enable_reasoning: bool = True) -> Agent:
    """构建 Agent 实例。

    Args:
        model: 模型名，缺省时依次取 --model 参数、OPENAI_MODEL、DEFAULT_MODEL。
               要看到「思考链」，请使用推理模型（gpt-5 / gpt-5-mini / o4-mini 等）。
        enable_reasoning: 是否开启模型的 reasoning（思考链）输出。
    """
    model_name = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    return Agent(
        name=AGENT_NAME,
        model=model_name,
        instructions=AGENT_INSTRUCTIONS,
        tools=[get_current_time, classroom_score_stats, summarize_error_types],
        # reasoning=None 时对非推理模型关闭思考；dict 会被 SDK 解析为 Reasoning 配置
        model_settings=ModelSettings(
            reasoning={"effort": "medium", "summary": "auto"} if enable_reasoning else None
        ),
    )


# --------------------------------------------------------------------------- #
# 2. 历史会话（SQLite 持久化）
# --------------------------------------------------------------------------- #
def create_session(session_id: str, db_path: str | None = None) -> SQLiteSession:
    """创建/恢复一个会话。

    同一个 session_id 前后两次运行会自动带上历史消息，实现多轮记忆；
    db_path 指向的 SQLite 文件跨进程保持，重启不丢失。
    """
    return SQLiteSession(session_id, db_path or DEFAULT_DB)


# --------------------------------------------------------------------------- #
# 3+4. 流式运行 + 事件渲染
# --------------------------------------------------------------------------- #
class _StreamRenderer:
    """把 run_streamed 的事件流渲染为控制台输出。

    渲染约定：
    - 🧠 思考链（推理模型的 reasoning 摘要，dim 灰色）先流式打印；
    - 💬 最终答案（response.output_text.delta）接着打印；
    - 🔧 工具调用 / 📤 工具返回 在对应 run item 事件到达时打印。
    """

    def __init__(self) -> None:
        self._reasoning_open = False
        self._text_open = False
        self.tool_names: list[str] = []
        # SDK 0.22.x 里 tool_output 事件不带工具名，用 call_id -> name 映射补全
        self._call_names: dict[str, str] = {}
        # Runner 开始时会发一次 AgentUpdatedStreamEvent，不算「切换」，跳过
        self._agent_announced = False

    # -- 思考链 --
    def reasoning_delta(self, delta: str) -> None:
        """流式打印一段推理摘要。"""
        if not self._reasoning_open:
            console.print("[dim]🧠 思考链：[/]", end="", soft_wrap=True)
            self._reasoning_open = True
        console.print(f"[dim]{escape(delta)}[/dim]", end="", soft_wrap=True)

    # -- 最终答案 --
    def text_delta(self, delta: str) -> None:
        """流式打印一段最终答案。"""
        if self._reasoning_open:
            self._close_reasoning()
        console.print(escape(delta), end="", soft_wrap=True)
        self._text_open = True

    # -- 工具调用 --
    def tool_called(self, name: str, arguments: str, call_id: str | None = None) -> None:
        self._close_reasoning()          # 工具调用前结束思考/文本区块
        self._close_text()
        if call_id:
            self._call_names[call_id] = name
        try:                             # 参数美化：单行/缩进均可读
            pretty = json.dumps(json.loads(arguments), ensure_ascii=False)
        except (ValueError, TypeError):
            pretty = arguments
        console.print(
            f"\n[cyan]🔧 调用工具[/] [bold cyan]{escape(name)}[/]"
            f"[dim]({escape(pretty)})[/dim]",
            soft_wrap=True,
        )
        self.tool_names.append(name)

    def tool_output(self, output: Any, call_id: str | None = None) -> None:
        name = self._call_names.get(call_id or "", "工具")
        console.print(
            f"[green]📤 工具返回[/] [bold]{escape(name)}[/]: "
            f"{escape(_stringify(output))}",
            soft_wrap=True,
        )

    # -- 生命周期 --
    def agent_switched(self, name: str) -> None:
        """多 Agent 切换提示（runner 起始的那个 agent 不算切换）。"""
        if self._agent_announced:
            console.print(f"[yellow]🤝 已切换 Agent → {escape(name)}[/]")
        self._agent_announced = True

    def finish(self) -> None:
        self._close_reasoning()
        self._close_text()

    def _close_reasoning(self) -> None:
        if self._reasoning_open:
            console.print("", soft_wrap=True)
            self._reasoning_open = False

    def _close_text(self) -> None:
        if self._text_open:
            console.print("", soft_wrap=True)
            self._text_open = False


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """兼容地从对象或 dict 上取属性。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify(value: Any) -> str:
    """工具返回值统一转成可读字符串。"""
    if isinstance(value, str):
        return value
    if value is None:
        return "(无返回)"
    return json.dumps(value, ensure_ascii=False, default=str)


async def chat_streamed(
    agent: Agent,
    user_input: str,
    session: SQLiteSession | None = None,
) -> str:
    """流式运行一轮对话，全过程打印到控制台。

    返回该轮 assistant 的最终文本；出错时打印提示并返回空串。
    传入了 session 时，Runner 会自动把本轮 messages 追加进会话历史。
    """
    renderer = _StreamRenderer()
    try:
        result = Runner.run_streamed(agent, input=user_input, session=session)
        async for event in result.stream_events():
            _dispatch_event(event, renderer)
        renderer.finish()
        final = str(result.final_output or "")
    except Exception as exc:  # noqa: BLE001 - 统一兜底，保证 CLI 不崩
        renderer.finish()
        console.print(f"[bold red]✗ 运行出错：[/]{escape(str(exc))}")
        console.print(
            "[yellow]提示：检查 OPENAI_API_KEY 是否有效；"
            "模型不可用时可用 --model gpt-4o 降级（注意：非推理模型没有思考链）。[/]"
        )
        return ""

    if renderer.tool_names:
        console.print(
            f"[dim]—— 本轮调用工具 {len(renderer.tool_names)} 次："
            f"{', '.join(renderer.tool_names)} ——[/]"
        )
    return final


def _dispatch_event(event: Any, renderer: _StreamRenderer) -> None:
    """把单个流事件分派给渲染器。"""
    # 原始 token 流：reasoning 摘要增量、答案文本增量
    if isinstance(event, RawResponsesStreamEvent):
        data = event.data
        etype = getattr(data, "type", "")
        delta = getattr(data, "delta", None)
        if not delta:
            return
        if "reasoning" in etype:
            renderer.reasoning_delta(delta)
        elif etype == "response.output_text.delta":
            renderer.text_delta(delta)
        return

    # 结构化运行项：工具调用开始、工具返回、消息完成等
    if isinstance(event, RunItemStreamEvent):
        item = event.item
        if event.name == "tool_called":
            raw = _attr(item, "raw_item", None)
            name = getattr(item, "tool_name", None) or _attr(raw, "name", None) or "?"
            renderer.tool_called(
                str(name),
                _attr(raw, "arguments", "") or "",
                call_id=_attr(raw, "call_id", None) or _attr(raw, "id", None),
            )
        elif event.name == "tool_output":
            renderer.tool_output(
                getattr(item, "output", None),
                call_id=_attr(_attr(item, "raw_item", None), "call_id", None),
            )
        return

    # 多 Agent 切换（handoff）
    if isinstance(event, AgentUpdatedStreamEvent):
        renderer.agent_switched(event.new_agent.name)
