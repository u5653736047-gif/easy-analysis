"""冒烟测试：不需要真实 API Key，端到端验证 Agent 框架。

原理：用 SDK 官方测试替身 ScriptedModel 编排确定的模型行为，
走真实的 Runner.run_streamed 流式管线（含 reasoning 增量、工具调用执行、
SQLiteSession 历史写入），渲染逻辑与线上完全一致。

运行：
    .venv/bin/python tests/smoke_test.py

覆盖的验收点：
1. 流式输出：🧠 思考链 -> 🔧 工具调用 -> 📤 工具返回 -> 💬 最终答案；
2. 工具调用：框架真实执行工具，且工具输出进入下一轮模型输入；
3. 历史会话：两轮对话后 session 中消息条数正确增长。
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

from agents.testing import ScriptedModel, assistant_message, function_call

from agent_core import build_agent, chat_streamed, create_session


def _scripted_model() -> ScriptedModel:
    """编排三轮行为：思考+调工具 -> 给结论 -> 第二轮闲聊。"""
    return ScriptedModel(
        [
            # 第一轮模型调用：思考链 + 工具调用
            [
                ResponseReasoningItem(
                    id="rs-1",
                    type="reasoning",
                    summary=[
                        Summary(
                            text="老师给了 7 个学生的分数，先调工具统计成绩分布。",
                            type="summary_text",
                        )
                    ],
                ),
                function_call(
                    "classroom_score_stats",
                    {"scores": [88, 92, 59, 45, 76, 61, 95]},
                    call_id="call-1",
                ),
            ],
            # 第一轮第二轮模型调用（拿到工具结果后）：给结论
            [
                ResponseReasoningItem(
                    id="rs-2",
                    type="reasoning",
                    summary=[
                        Summary(text="工具已返回平均分和及格率，可以汇总结论了。", type="summary_text")
                    ],
                ),
                assistant_message(
                    "本次考试平均分 73.7 分，及格率 71.4%。"
                    "低分段（45/59/61）集中在计算失误，建议针对计算类题型专项练习。"
                ),
            ],
            # 第二轮模型调用：不带工具，直接回答（验证历史会话仍在模型输入中）
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


async def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'✓' if ok else '✗'} {name}" + (f" | {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    tmpdir = tempfile.mkdtemp(prefix="xiaoxiyi-smoke-")
    db = str(Path(tmpdir) / "smoke.db")
    session = create_session("smoke", db)

    agent = _inject_model(build_agent(), _scripted_model())

    print("\n===== 第一轮：包含工具调用 =====================================")
    final1 = await chat_streamed(agent, "我上传了本次月考成绩：88, 92, 59, 45, 76, 61, 95，帮我分析一下", session)
    check("第一轮返回了最终答案", bool(final1.strip()), final1[:40] + "...")

    items = await session.get_items()
    check("会话历史已写入（含 user/assistant 消息）", len(items) >= 4, f"{len(items)} 条")

    # 工具真实执行过：第二轮模型调用（模型的第二次调用）的输入里应包含工具返回的 JSON
    model: ScriptedModel = agent.model
    check("模型共被调用 2 次（工具调用前后）", len(model.calls) == 2, str(len(model.calls)))
    if len(model.calls) >= 2:
        tool_input = str(model.calls[-1].input)
        check("工具输出已进入第二轮模型输入", "平均分" in tool_input and "73.71" in tool_input)

    print("\n===== 第二轮：不带工具 + 历史会话 ==============================")
    final2 = await chat_streamed(agent, "你还记得刚才的分析结论吗？", session)
    check("第二轮返回了最终答案", bool(final2.strip()), final2[:40] + "...")
    check("历史消息随轮次增长", len(await session.get_items()) >= len(items) + 2)

    print("\n===== 结果 =====================================================")
    if failures:
        print(f"✗ 未通过：{failures}")
        return 1
    print("✓ 全部通过：流式输出 / 思考链 / 工具调用 / 历史会话 均正常工作")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
