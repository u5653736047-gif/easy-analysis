"""冒烟测试：不需要真实 API Key，端到端验证 Agent 框架。

两层测试：
1. test_event_dispatch   —— 单元级：用真实 openai 事件对象驱动 _render_raw_event，
   验证按 event.data.type 分流渲染、工具调用去重、未知事件忽略；
2. test_end_to_end       —— 端到端：SDK 官方 ScriptedModel 走真实流式管线
   （reasoning 增量 / 工具真实执行 / SQLiteSession 历史写入），并捕获控制台
   输出验证「本轮只打印一次工具调用」等渲染行为。

运行：
    .venv/bin/python tests/smoke_test.py
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_core
from agent_core import _StreamRenderer, _render_raw_event, build_agent, chat_streamed
from session_manager import SessionManager
from openai.types.responses import (
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from agents import set_tracing_disabled
from agents.stream_events import RawResponsesStreamEvent
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response import Response
from rich.console import Console

from agents.testing import ScriptedModel, assistant_message, function_call

set_tracing_disabled(True)  # 测试不需要上报 trace，也避免无 Key 时的告警噪音

PASS, FAIL = "\033[32m✓\033[0m", "\033[31m✗\033[0m"


class _Recorder(Console):
    """记录所有 print 的 Console，用于断言渲染输出。"""

    def __init__(self) -> None:
        super().__init__(record=True, width=120, no_color=True)


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" | {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. 单元级：事件分发
# --------------------------------------------------------------------------- #
async def test_event_dispatch(results: list) -> None:
    print("\n===== 单元级：_render_raw_event 分流 ===========================")
    r = _StreamRenderer()
    recorder = _Recorder()
    original = agent_core.console
    agent_core.console = recorder
    try:
        # reasoning delta
        _render_raw_event(
            ResponseReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                item_id="rs-1", output_index=0, summary_index=0,
                delta="先统计成绩，再给结论", sequence_number=1,
            ),
            r,
        )
        # content delta
        _render_raw_event(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id="msg-1", output_index=0, content_index=0,
                delta="同学你好", logprobs=[], sequence_number=2,
            ),
            r,
        )
        # function call：added -> arguments.done -> output_item.done（重复到达）
        _render_raw_event(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added", output_index=1, sequence_number=3,
                item=ResponseFunctionToolCall(
                    id="fc-1", call_id="call-1", name="get_current_time",
                    arguments="", type="function_call",
                ),
            ),
            r,
        )
        _render_raw_event(
            ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                item_id="fc-1", output_index=1, arguments="{}", sequence_number=4,
            ),
            r,
        )
        _render_raw_event(
            ResponseOutputItemDoneEvent(
                type="response.output_item.done", output_index=1, sequence_number=5,
                item=ResponseFunctionToolCall(
                    id="fc-1", call_id="call-1", name="get_current_time",
                    arguments="{}", type="function_call",
                ),
            ),
            r,
        )
        # 未知事件：不应报错、不应产生输出
        _render_raw_event(
            ResponseCreatedEvent(
                type="response.created", response=Response(id="resp-1", object="response", created_at=0, model="gpt-5-mini", status="in_progress", output=[], parallel_tool_calls=False, tool_choice="auto", tools=[]), sequence_number=0,
            ),
            r,
        )
        r.call_result("call-1", "2026-01-01 12:00:00")
        r.finish()
    finally:
        agent_core.console = original

    text = recorder.export_text()
    _check(results, "reasoning delta 渲染为思考链", "🧠 思考链" in text and "先统计成绩" in text)
    _check(results, "content delta 渲染为答案", "同学你好" in text)
    _check(results, "工具调用只打印一次（去重）", text.count("🔧 调用工具") == 1, f"出现 {text.count('🔧 调用工具')} 次")
    _check(results, "工具调用带名称与参数", "get_current_time" in text and "({})" in text)
    _check(results, "工具返回按 call_id 关联到工具名", "📤 工具返回" in text and "get_current_time: 2026-01-01" in text)
    _check(results, "未知事件不产生输出", "response.created" not in text and "加工" not in text)
    _check(results, "tool_names 记录一次", r.tool_names == ["get_current_time"], str(r.tool_names))


# --------------------------------------------------------------------------- #
# 2. 端到端：ScriptedModel 走真实流式管线
# --------------------------------------------------------------------------- #
def _scripted_model() -> ScriptedModel:
    """编排三轮行为：思考+调工具 -> 给结论 -> 第二轮闲聊（不带工具）。"""
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Summary

    return ScriptedModel(
        [
            # 第一轮第一次调用：思考链 + 工具调用
            [
                ResponseReasoningItem(
                    id="rs-1",
                    type="reasoning",
                    summary=[Summary(text="老师给了 7 个学生的分数，先调工具统计成绩分布。", type="summary_text")],
                ),
                function_call("classroom_score_stats", {"scores": [88, 92, 59, 45, 76, 61, 95]}, call_id="call-1"),
            ],
            # 第一轮第二次调用（拿到工具结果后）：给结论
            [
                ResponseReasoningItem(
                    id="rs-2",
                    type="reasoning",
                    summary=[Summary(text="工具已返回平均分和及格率，可以汇总结论了。", type="summary_text")],
                ),
                assistant_message(
                    "本次考试平均分 73.7 分，及格率 71.4%。"
                    "低分段（45/59/61）集中在计算失误，建议针对计算类题型专项练习。"
                ),
            ],
            # 第二轮：不带工具，直接回答（验证历史会话仍在模型输入中）
            [assistant_message("好的，上一轮的结论我仍然记得。")],
        ]
    )


def _inject_model(agent, model: ScriptedModel):
    """把 ScriptedModel 注入 build_agent() 产出的 Agent。"""
    try:
        return dataclasses.replace(agent, model=model)
    except TypeError:
        agent.model = model
        return agent


async def test_end_to_end(results: list) -> None:
    print("\n===== 端到端：脚本模型 -> Runner.stream -> SQLiteSession ========")
    tmpdir = tempfile.mkdtemp(prefix="xiaoxiyi-smoke-")
    mgr = SessionManager(str(Path(tmpdir) / "smoke.db"))
    mgr.create("smoke")  # 元数据先行；Runner 写消息后由 record_activity 记账
    session = mgr.open_session("smoke")

    original = agent_core.console
    recorder = _Recorder()
    agent_core.console = recorder
    agent = _inject_model(build_agent(), _scripted_model())
    try:
        final1 = await chat_streamed(
            agent, "我上传了本次月考成绩：88, 92, 59, 45, 76, 61, 95，帮我分析一下", session
        )
        items_after_1 = len(await session.get_items())
        final2 = await chat_streamed(agent, "你还记得刚才的分析结论吗？", session)
        items_after_2 = len(await session.get_items())
    finally:
        agent_core.console = original
    text = recorder.export_text()

    _check(results, "第一轮返回了最终答案", bool(final1.strip()))
    _check(results, "第二轮返回了最终答案", bool(final2.strip()))
    _check(results, "思考链出现", text.count("🧠 思考链") >= 2, f"{text.count('🧠 思考链')} 段")
    _check(results, "本轮只打印一次工具调用（去重）", text.count("🔧 调用工具") == 1)
    _check(results, "工具返回已展示", text.count("📤 工具返回") == 1 and "平均分" in text)
    _check(results, "答案含工具统计结果", "73.7" in final1)

    model: ScriptedModel = agent.model
    _check(results, "模型共被调用 3 次（工具前后 + 第二次会话）", len(model.calls) == 3, str(len(model.calls)))
    tool_input = str(model.calls[1].input)
    _check(results, "工具输出已进入第二轮模型输入", "73.71" in tool_input and "平均分" in tool_input)
    second_input = str(model.calls[2].input)
    _check(results, "第二轮模型输入包含第一轮历史", "帮我分析一下" in second_input and "专项练习" in second_input)

    _check(results, "会话历史第一轮后已写入", items_after_1 >= 4, f"{items_after_1} 条")
    _check(results, "历史消息随轮次增长", items_after_2 >= items_after_1 + 2, f"{items_after_1} -> {items_after_2}")

    # SessionManager 与 Runner 落在同一个库：记账/自动标题/预览应与历史一致
    mgr.record_activity("smoke", "我上传了本次月考成绩")
    info = mgr.get("smoke")
    _check(results, "SessionManager 能读到 Runner 写入的消息数",
           info is not None and info.message_count == items_after_2,
           f"meta={info.message_count if info else None} vs history={items_after_2}")
    _check(results, "首条消息自动生成标题", info is not None and info.title.startswith("我上传了"))
    _check(results, "预览取自最新消息", info is not None and bool(info.preview))
    mgr.close()


async def main() -> int:
    results: list = []
    await test_event_dispatch(results)
    await test_end_to_end(results)
    print("\n===== 结果 =====================================================")
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{FAIL} 未通过 {len(failed)}/{len(results)}：{failed}")
        return 1
    print(f"{PASS} 全部通过（{len(results)} 项）：事件分流 / 去重 / 工具调用 / 历史会话 均正常")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
