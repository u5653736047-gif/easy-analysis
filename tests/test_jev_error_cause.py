"""错因分析工具测试：本地假 Jev 服务 + 工具逻辑 + 框架内端到端。

运行：
    .venv/bin/python tests/test_jev_error_cause.py

与 tests/test_jev_question_type.py 同构：假服务返回编排好的响应，覆盖
- 正常路径（主错因 + 多因阈值过滤 + state 模板拼装 + 问题数量）；
- 低置信/margin 过小/多因为空 -> needs_review；
- 上游 5xx / 连接失败 -> source=error 降级；
- 框架内端到端：ScriptedModel function_call -> 真实工具 -> 🔧/📤 渲染。
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_core
from agent_core import AGENT_INSTRUCTIONS, AGENT_NAME, chat_streamed
from agents import Agent, set_tracing_disabled
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.tool import function_tool
from jev.error_cause import (
    ERROR_CAUSES,
    MULTI_CAUSE_THRESHOLD,
    analyze_error_cause_raw,
    build_state,
)
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary
from rich.console import Console

set_tracing_disabled(True)

PASS, FAIL = "\033[32m✓\033[0m", "\033[31m✗\033[0m"


class _Recorder(Console):
    def __init__(self) -> None:
        super().__init__(record=True, width=120, no_color=True)


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" | {detail}" if detail else ""))


def start_fake_jev(behavior) -> tuple[str, HTTPServer, list]:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server 约定
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append({"path": self.path, "body": body})
            status, payload = behavior(body)
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server, received


def _resp(primary: str, conf: float, nouls: dict[str, float]) -> dict:
    answers = {
        "primary": {
            "type": "choice", "choice": primary, "confidence": conf,
            "probabilities": nouls,
        }
    }
    for cause, p in nouls.items():
        answers[f"multi::{cause}"] = {"type": "noul", "noul": p}
    return {"model": "jev-latest",
            "usage": {"input_tokens": 200, "output_tokens": 0},
            "answers": answers}


def _impl_with_base(q: str, ref: str, ans: str, base_url: str) -> dict:
    """走真实实现体，但把请求打到假服务。"""
    from jev import error_cause as jev_ec

    orig = jev_ec.analyze_error_cause_raw
    jev_ec.analyze_error_cause_raw = (
        lambda question, reference, student, *, api_key=None, **kw: orig(
            question, reference, student, api_key=api_key or "test-key",
            base_url=base_url, timeout=15,
        )
    )
    try:
        return jev_ec.analyze_error_cause_impl(q, ref, ans)
    finally:
        jev_ec.analyze_error_cause_raw = orig


async def test_tool_logic(results: list) -> None:
    print("\n===== 假服务：错因工具逻辑 =====================================")

    nouls_full = {
        "概念不清": 0.10, "计算失误": 0.85, "审题不清": 0.05,
        "方法错误": 0.08, "粗心失误": 0.60, "知识遗忘": 0.03,
        "表述不完整": 0.12, "其他": 0.02,
    }

    # 场景 1：正常路径 + 校验 state 模板与问题结构
    url, server, received = start_fake_jev(lambda b: (200, _resp("计算失误", 0.85, nouls_full)))
    try:
        out = _impl_with_base("解方程 2(x+1)=x-1", "x=-3", "去括号得 2x+1=x-1，解得 x=0", url)
    finally:
        server.shutdown()

    _check(results, "正常：primary_cause 正确", out.get("primary_cause") == "计算失误")
    _check(results, "正常：needs_review=False", out.get("needs_review") is False)
    _check(results, "正常：detected_causes 按阈值过滤",
           [c["cause"] for c in out.get("detected_causes", [])] == ["计算失误"],
           str([c["cause"] for c in out.get("detected_causes", [])]))
    _check(results, "正常：margin 已计算", abs(out.get("margin", 0) - 0.25) < 1e-6, str(out.get("margin")))

    body = received[0]["body"]
    _check(results, "state 模板包含题目/参考答案/学生作答三段",
           all(k in body["state"] for k in ("题目：", "参考答案：", "学生作答：")))
    q = body["questions"]
    _check(results, "问题包含 primary Choice", q.get("primary", {}).get("type") == "choice")
    _check(results, "每个错因各有一个 Noul 问题",
           sum(1 for k in q if k.startswith("multi::")) == len(ERROR_CAUSES),
           f"{sum(1 for k in q if k.startswith('multi::'))} 个")
    _check(results, "Choice 的 taxonomy 顺序固定",
           list(q["primary"]["criteria"].keys()) == list(ERROR_CAUSES))

    # 场景 2：低置信 -> needs_review
    url, server, _ = start_fake_jev(lambda b: (200, _resp("计算失误", 0.45, nouls_full)))
    try:
        out = _impl_with_base("题", "答", "学生作答", url)
    finally:
        server.shutdown()
    _check(results, "低置信：needs_review=True", out.get("needs_review") is True and "hint" in out)

    # 场景 3：多因未过阈值（全部 Noul < 阈值）-> needs_review
    low_nouls = {k: (0.5 if k == "计算失误" else 0.1) for k in ERROR_CAUSES}
    url, server, _ = start_fake_jev(lambda b: (200, _resp("计算失误", 0.95, low_nouls)))
    try:
        out = _impl_with_base("题", "答", "学生作答", url)
    finally:
        server.shutdown()
    _check(results, f"Noul 均 <{MULTI_CAUSE_THRESHOLD}：needs_review=True（避免无据打标签）",
           out.get("needs_review") is True and out.get("detected_causes") == [])

    # 场景 4：margin 过小 -> needs_review
    close = {"概念不清": 0.46, "计算失误": 0.48, "审题不清": 0.02, "方法错误": 0.01,
             "粗心失误": 0.01, "知识遗忘": 0.01, "表述不完整": 0.01, "其他": 0.0}
    url, server, _ = start_fake_jev(lambda b: (200, _resp("计算失误", 0.60, close)))
    try:
        out = _impl_with_base("题", "答", "学生作答", url)
    finally:
        server.shutdown()
    _check(results, "margin 过小：needs_review=True",
           out.get("needs_review") is True and out.get("margin", 1) < 0.15,
           f"margin={out.get('margin')}")

    # 场景 5：上游 5xx -> 降级不抛异常
    url, server, _ = start_fake_jev(lambda b: (500, {"error": {"message": "boom"}}))
    try:
        out = _impl_with_base("题", "答", "学生作答", url)
    finally:
        server.shutdown()
    _check(results, "上游 5xx：source=error 且不抛异常",
           out.get("source") == "error" and "自行分析" in out.get("hint", ""))

    # 场景 6：连接失败 -> 降级
    dead_url, dead_server, _ = start_fake_jev(lambda b: (200, _resp("x", 1.0, {"x": 1.0})))
    dead_server.shutdown()
    dead_server.server_close()
    out = _impl_with_base("题", "答", "学生作答", dead_url)
    _check(results, "连接失败：source=error",
           out.get("source") == "error" and out.get("needs_review") is True)


async def test_end_to_end(results: list) -> None:
    print("\n===== 框架内端到端：agent -> 错因工具 -> 渲染 ====================")
    nouls_full = {
        "概念不清": 0.10, "计算失误": 0.85, "审题不清": 0.05,
        "方法错误": 0.08, "粗心失误": 0.60, "知识遗忘": 0.03,
        "表述不完整": 0.12, "其他": 0.02,
    }
    url, server, received = start_fake_jev(lambda b: (200, _resp("计算失误", 0.85, nouls_full)))
    try:
        # 直接复用生产工具：只把底层 *_raw 重定向到假服务
        from jev import error_cause as jev_ec

        orig_raw = jev_ec.analyze_error_cause_raw
        jev_ec.analyze_error_cause_raw = (
            lambda question, reference, student, *, api_key=None, **kw: orig_raw(
                question, reference, student,
                api_key=api_key or "test-key", base_url=url, timeout=15,
            )
        )
        try:
            model = ScriptedModel([
                [
                    ResponseReasoningItem(
                        id="rs-1", type="reasoning",
                        summary=[Summary(text="调错因分析工具。", type="summary_text")],
                    ),
                    function_call(
                        "analyze_error_cause",
                        {"question_text": "解方程 2(x+1)=x-1", "reference_answer": "x=-3",
                         "student_answer": "去括号得 2x+1=x-1，解得 x=0"},
                        call_id="call-err-1",
                    ),
                ],
                [assistant_message("该生主要错因为计算失误（置信度0.85，无需复核），同时建议关注粗心标签。")],
            ])
            agent = Agent(
                name=AGENT_NAME, model=model,
                instructions=AGENT_INSTRUCTIONS,
                tools=[jev_ec.analyze_error_cause],
            )

            recorder = _Recorder()
            original = agent_core.console
            agent_core.console = recorder
            try:
                final = await chat_streamed(
                    agent, "分析这道错题：题目「解方程 2(x+1)=x-1」，答案 x=-3，学生答 x=0"
                )
            finally:
                agent_core.console = original
            text = recorder.export_text()

            _check(results, "端到端：工具被真实调用", len(received) >= 1)
            _check(results, "端到端：🔧 工具调用已渲染",
                   "🔧 调用工具" in text and "analyze_error_cause" in text)
            _check(results, "端到端：📤 工具返回已渲染",
                   "📤 工具返回" in text and "计算失误" in text)
            _check(results, "端到端：工具结果回传给模型",
                   any("0.85" in str(c.input) for c in model.calls))
            _check(results, "端到端：最终答案非空", bool((final or "").strip()))
        finally:
            jev_ec.analyze_error_cause_raw = orig_raw
    finally:
        server.shutdown()


async def main() -> int:
    results: list = []
    await test_tool_logic(results)
    await test_end_to_end(results)
    print("\n===== 结果 =====================================================")
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{FAIL} 未通过 {len(failed)}/{len(results)}：{failed}")
        return 1
    print(f"{PASS} 全部通过（{len(results)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
