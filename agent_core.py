"""「小析易」Agent 基础框架（基于 OpenAI Agents SDK）。

四大能力：
1. Agent 构建：人设 + 模型 + 工具                 -> build_agent()
2. 历史会话：SQLiteSession 持久化                  -> create_session()
3. 流式运行：Runner.run_streamed() 逐 token 出事件 -> chat_streamed()
4. 过程渲染：按事件的类型字段分流                  -> _StreamRenderer

渲染不依赖 isinstance，全部按事件自身的类型字段分流：

- event.data.type（Responses API 原始流事件，Runner 原样转发）
    - response.reasoning_summary_text.delta
      response.reasoning_text.delta                  -> 🧠 思考链
    - response.output_text.delta                     -> 💬 最终答案
    - response.output_item.added（function_call）     -> 登记待打印的工具调用
    - response.function_call_arguments.done           -> 🔧 打印工具调用与参数
    - response.output_item.done（function_call）      -> 参数事件的兜底
- event.name（SDK 运行项事件）
    - tool_output                                    -> 📤 工具执行结果
      （原始流里没有工具执行结果，只能从运行项事件拿）
"""
from __future__ import annotations

import json
import os
from typing import Any

from agents import Agent, ModelSettings, Runner, SQLiteSession
from rich.console import Console
from rich.markup import escape

from tools import (
    classroom_score_stats,
    get_current_time,
    summarize_error_types,
)
from jev_tool import classify_question_type
from error_cause_tool import analyze_error_cause

console = Console()

# --------------------------------------------------------------------------- #
# 默认配置（均可用环境变量 / CLI 参数覆盖）
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gpt-5-mini"           # 推理模型才能展示「思考链」
DEFAULT_DB = "conversation_history.db"  # 会话历史 SQLite 文件
AGENT_NAME = "小析易"

AGENT_INSTRUCTIONS = """
你是「小析易」的智能学情分析助手，服务对象是中小学教师。

你的职责：
1. 帮助教师分析班级考试情况：主要考察题型、各题型正确率、高出错率题型等；
2. 帮助教师归纳学生易错点，为每名学生建立并累加易错标签；
3. 涉及数据计算（班级成绩统计、错因占比等）时，优先调用工具，不要凭空估算。

题型判定规则：
- 需要确定某道题的题型时，必须调用 classify_question_type 工具（由 Jev 模型判定）；
- 工具返回 needs_review=true 时，先结合题干自行复核，再决定是否采信；
- 工具返回 source=error 时，自行判断题型并在结果中说明降级原因。

错因分析规则：
- 分析某道错题「错在哪」时，必须调用 analyze_error_cause 工具（由 Jev 模型判定），
  需要同时提供题目、参考答案、学生作答三项信息；
- 结果是多因的：detected_causes 列表可同时打多个错因标签，供学生易错标签累加使用；
- 同样遵循 needs_review 复核与 source=error 降级规则。

回答要求：
- 使用简体中文，风格专业、简洁、有条理；
- 给出结论时说明依据（来自哪次考试、哪份数据）；
- 信息不足时，主动向教师追问，而不是臆测。
""".strip()

def _is_reasoning_delta(etype: str) -> bool:
    """思考链增量事件的宽松匹配。

    覆盖 response.reasoning_summary_text.delta / response.reasoning_text.delta，
    也兼容部分中转实现的命名变体；不含 delta 的纯结构事件（part.added 等）不匹配。
    """
    return "reasoning" in etype and etype.endswith(".delta")


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
        tools=[get_current_time, classroom_score_stats, summarize_error_types, classify_question_type, analyze_error_cause],
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
# 4. 事件渲染器
# --------------------------------------------------------------------------- #
class _StreamRenderer:
    """把流事件按类型分流渲染到控制台。"""

    def __init__(self) -> None:
        self._reasoning_open = False
        self._text_open = False
        self._pending: dict[str, tuple[str, str | None]] = {}  # item_id -> (工具名, call_id)
        self._printed: set[str] = set()                        # 已打印过调用的 item_id
        self._names: dict[str, str] = {}                      # call_id -> 工具名
        self.tool_names: list[str] = []

    # -- reasoning：思考链 --
    def reasoning(self, delta: str) -> None:
        if not self._reasoning_open:
            console.print("[dim]🧠 思考链：[/]", end="", soft_wrap=True)
            self._reasoning_open = True
        console.print(f"[dim]{escape(delta)}[/dim]", end="", soft_wrap=True)

    # -- content：最终答案 --
    def content(self, delta: str) -> None:
        if self._reasoning_open:
            self._close_reasoning()
        console.print(escape(delta), end="", soft_wrap=True)
        self._text_open = True

    # -- function call：工具调用 --
    def call_started(self, item_id: str, name: str, call_id: str | None = None) -> None:
        """response.output_item.added：登记工具名 + call_id，等参数到齐再打印。"""
        self._pending[item_id] = (name, call_id)
        if call_id:
            self._names[call_id] = name

    def call_done(
        self,
        item_id: str,
        arguments: str,
        name: str | None = None,
        call_id: str | None = None,
    ) -> None:
        """response.function_call_arguments.done / output_item.done：打印工具调用。"""
        if item_id in self._printed:
            return  # 两个事件可能都到，只打印一次
        tool = self._pending.pop(item_id, None)
        tool_name = name or (tool[0] if tool else "工具")
        call_id = call_id or (tool[1] if tool else None)
        self._printed.add(item_id)
        self._close_reasoning()
        self._close_text()
        if call_id:
            self._names[call_id] = tool_name
        try:  # 参数美化：解析失败时回落原始字符串
            pretty = json.dumps(json.loads(arguments), ensure_ascii=False)
        except (ValueError, TypeError):
            pretty = arguments
        console.print(
            f"\n[cyan]🔧 调用工具[/] [bold cyan]{escape(tool_name)}[/]"
            f"[dim]({escape(str(pretty))})[/dim]",
            soft_wrap=True,
        )
        self.tool_names.append(tool_name)

    def call_result(self, call_id: str | None, output: Any) -> None:
        """tool_output 运行项事件：打印工具执行结果。"""
        name = self._names.get(call_id or "", "工具")
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
        console.print(
            f"[green]📤 工具返回[/] [bold]{escape(name)}[/]: {escape(text)}",
            soft_wrap=True,
        )

    # -- 收尾 --
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


# --------------------------------------------------------------------------- #
# 3. 流式运行
# --------------------------------------------------------------------------- #
async def chat_streamed(
    agent: Agent,
    user_input: str,
    session: SQLiteSession | None = None,
) -> str:
    """流式运行一轮对话，全过程打印到控制台。

    返回该轮 assistant 的最终文本；出错时打印提示并返回空串。
    传入了 session 时，Runner 会自动把本轮 messages 追加进会话历史。
    """
    r = _StreamRenderer()
    try:
        result = Runner.run_streamed(agent, input=user_input, session=session)
        async for event in result.stream_events():
            data = getattr(event, "data", None)  # 原始流事件才有 .data
            if data is not None:
                _render_raw_event(data, r)
            elif getattr(event, "name", None) == "tool_output":
                r.call_result(_call_id_of(event.item), getattr(event.item, "output", None))
        r.finish()
        final = str(result.final_output or "")
    except Exception as exc:  # noqa: BLE001 - 统一兜底，保证 CLI 不崩
        r.finish()
        console.print(f"[bold red]✗ 运行出错：[/]{escape(str(exc))}")
        console.print(
            "[yellow]提示：检查 OPENAI_API_KEY 是否有效；"
            "模型不可用时可用 --model gpt-4o 降级（注意：非推理模型没有思考链）。[/]"
        )
        return ""

    if r.tool_names:
        console.print(
            f"[dim]—— 本轮调用工具 {len(r.tool_names)} 次：{', '.join(r.tool_names)} ——[/]"
        )
    return final


def _render_raw_event(data: Any, r: _StreamRenderer) -> None:
    """按 Responses API 原始事件 type 分流。"""
    etype = getattr(data, "type", "")
    item = getattr(data, "item", None)

    if _is_reasoning_delta(etype):
        if delta := getattr(data, "delta", None):
            r.reasoning(delta)
    elif etype == "response.output_text.delta":
        if delta := getattr(data, "delta", None):
            r.content(delta)
    elif etype == "response.output_item.added" and getattr(item, "type", "") == "function_call":
        # 登记 name/call_id；参数在随后的 arguments.done / output_item.done 到齐
        r.call_started(item.id, item.name, item.call_id)
    elif etype == "response.function_call_arguments.done":
        r.call_done(data.item_id, getattr(data, "arguments", "") or "")
    elif etype == "response.output_item.done" and getattr(item, "type", "") == "function_call":
        # 兜底：某些链路没有独立的 arguments.done 事件
        r.call_done(item.id, getattr(item, "arguments", "") or "", item.name, item.call_id)


def _call_id_of(item: Any) -> str | None:
    """从工具输出项取 call_id（raw_item 可能是对象或 dict）。"""
    raw = getattr(item, "raw_item", None)
    return raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
