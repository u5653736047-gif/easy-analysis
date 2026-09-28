"""Jev 工具测试：本地假 Jev 服务 + 工具逻辑 + 框架内端到端。

运行：
    .venv/bin/python tests/test_jev_tool.py

两层测试：
1. 工具逻辑（本地假 Jev 服务，确定性）：高置信 / 低置信 / margin 过小 /
   上游 5xx / 连接失败 / 未配置 Key / taxonomy 固定顺序与 Bearer 鉴权头；
2. 框架内端到端：ScriptedModel 发起 function_call -> 真实执行 Jev 工具
   （打假服务）-> 工具结果回传模型 -> 渲染 🔧/📤。

注意：假服务验证的是「我们的集成与降级逻辑」的可靠性；
Jev 模型本身的真实准确率请用真实调用评测（见 tests/jev_real_check.py）。
"""
from __future__ import annotations

import asyncio
import json
import os
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
from jev_tool import MARGIN_THRESHOLD, QUESTION_TYPES, classify_question_type_raw
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


# --------------------------------------------------------------------------- #
# 假 Jev 服务：behavior(request_body, received) -> (http_status, response_dict)
# --------------------------------------------------------------------------- #
def start_fake_jev(behavior) -> tuple[str, HTTPServer, list]:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server 约定
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "body": body,
            })
            status, payload = behavior(body, received)
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:  # 静默
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server, received


def _answer(choice: str, confidence: float, probabilities: dict) -> dict:
    return {
        "model": "jev-latest",
        "usage": {"input_tokens": 128, "output_tokens": 0},
        "answers": {
            "题型": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
    }


def _impl_with_base(question_text: str, base_url: str) -> dict:
    """走 classify_question_type_impl 的真实后处理逻辑，但把请求打到假服务。"""
    raw = classify_question_type_raw
    try:
        import jev_tool

        jev_tool.classify_question_type_raw = (
            lambda text, *, api_key=None, **kw: raw(
                text, api_key=api_key or "test-key", base_url=base_url, timeout=15
            )
        )
        return jev_tool.classify_question_type_impl(question_text)
    finally:
        import jev_tool

        jev_tool.classify_question_type_raw = raw


# --------------------------------------------------------------------------- #
# 1. 工具逻辑
# --------------------------------------------------------------------------- #
async def test_tool_logic(results: list) -> None:
    print("\n===== 假服务：工具逻辑 =========================================")

    # --- 场景 1：高置信正常路径 + 校验请求形态与 taxonomy 顺序 ---
    url, server, received = start_fake_jev(lambda b, r: (200, _answer(
        "解答题", 0.93,
        {"选择题": 0.02, "判断题": 0.01, "填空题": 0.01,
         "解答题": 0.93, "证明题": 0.03, "其他": 0.0},
    )))
    try:
        out = _impl_with_base("已知函数 f(x)=x²-2x，求 f(x) 的最小值。", url)
    finally:
        server.shutdown()

    _check(results, "高置信：返回正确答案", out.get("type") == "解答题", str(out.get("type")))
    _check(results, "高置信：needs_review=False", out.get("needs_review") is False)
    _check(results, "高置信：source=jev", out.get("source") == "jev")
    _check(results, "高置信：margin 已计算", abs(out.get("margin", 0) - 0.90) < 1e-6, str(out.get("margin")))

    req = received[0]
    _check(results, "请求路径为 /v1/systemone", req["path"] == "/v1/systemone", req["path"])
    _check(results, "API Key 以 Bearer 头传递", req["authorization"].startswith("Bearer "),
           req["authorization"][:14] + "...")
    sent_options = list(req["body"]["questions"]["题型"]["criteria"].keys())
    _check(results, "taxonomy 顺序固定（与代码常量一致）",
           sent_options == list(QUESTION_TYPES), ",".join(sent_options))

    # --- 场景 2：低置信 -> needs_review ---
    url, server, _ = start_fake_jev(lambda b, r: (200, _answer(
        "解答题", 0.45,
        {"选择题": 0.30, "判断题": 0.05, "填空题": 0.08,
         "解答题": 0.45, "证明题": 0.10, "其他": 0.02},
    )))
    try:
        out = _impl_with_base("……一段难以判断的题干……", url)
    finally:
        server.shutdown()
    _check(results, "低置信：needs_review=True 且附复核提示",
           out.get("needs_review") is True and "hint" in out,
           f"confidence={out.get('confidence')}")

    # --- 场景 3：置信度尚可但 margin 过小（选项顺序敏感的典型信号）---
    url, server, _ = start_fake_jev(lambda b, r: (200, _answer(
        "解答题", 0.52,
        {"选择题": 0.41, "判断题": 0.02, "填空题": 0.01,
         "解答题": 0.52, "证明题": 0.03, "其他": 0.01},
    )))
    try:
        out = _impl_with_base("下列四个选项中，能证明该结论的是（ ）", url)
    finally:
        server.shutdown()
    _check(results, f"margin<{MARGIN_THRESHOLD}：同样标记 needs_review",
           out.get("needs_review") is True and out.get("margin", 1) < MARGIN_THRESHOLD,
           f"margin={out.get('margin')}")

    # --- 场景 4：上游 5xx -> 结构化错误，不抛异常 ---
    url, server, _ = start_fake_jev(
        lambda b, r: (500, {"error": {"message": "upstream exploded"}})
    )
    try:
        out = _impl_with_base("任意题干", url)
    finally:
        server.shutdown()
    _check(results, "上游 5xx：source=error 且不抛异常",
           out.get("source") == "error" and "自行判断" in out.get("hint", ""),
           (out.get("error") or "")[:40])

    # --- 场景 5：连接失败（指向已关闭端口）---
    dead_url, dead_server, _ = start_fake_jev(lambda b, r: (200, _answer("x", 1.0, {"x": 1.0})))
    dead_server.shutdown()
    dead_server.server_close()
    out = _impl_with_base("任意题干", dead_url)
    _check(results, "连接失败：source=error",
           out.get("source") == "error" and out.get("needs_review") is True,
           (out.get("error") or "")[:40])

    # --- 场景 6：未配置 Key -> 可读错误 ---
    saved = {k: os.environ.pop(k, None)
             for k in ("COMMAND_CODE_API_KEY", "TYPESAFE_API_KEY")}
    import jev_tool

    jev_tool._dotenv_loaded = True  # 阻止重复加载 .env
    try:
        try:
            classify_question_type_raw("任意题干")
            raised = False
        except RuntimeError as exc:
            raised = "API Key" in str(exc)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        jev_tool._dotenv_loaded = False
    _check(results, "未配置 Key：抛出可读错误", raised)


# --------------------------------------------------------------------------- #
# 2. 框架内端到端
# --------------------------------------------------------------------------- #
async def test_end_to_end(results: list) -> None:
    print("\n===== 框架内端到端：agent -> Jev 工具 -> 渲染 =====================")
    question = "已知函数 f(x)=x²-2x，求 f(x) 的最小值。"

    url, server, received = start_fake_jev(lambda b, r: (200, _answer(
        "解答题", 0.93,
        {"选择题": 0.02, "判断题": 0.01, "填空题": 0.01,
         "解答题": 0.93, "证明题": 0.03, "其他": 0.0},
    )))
    try:
        @function_tool
        def classify_question_type(question_text: str) -> str:
            """调用 Jev 判定一道题的题型（测试替身，打本地假服务）。"""
            import json as _json

            r = classify_question_type_raw(
                question_text, api_key="test-key", base_url=url, timeout=15
            )
            probs = sorted(r["probabilities"].values(), reverse=True)
            margin = round(probs[0] - (probs[1] if len(probs) > 1 else 0.0), 4)
            out = dict(r, margin=margin, source="jev")
            out["needs_review"] = (
                out["confidence"] < __import__("jev_tool").CONFIDENCE_THRESHOLD
                or margin < __import__("jev_tool").MARGIN_THRESHOLD
            )
            return _json.dumps(out, ensure_ascii=False)

        model = ScriptedModel([
            [
                ResponseReasoningItem(
                    id="rs-1", type="reasoning",
                    summary=[Summary(text="先调工具判定题型。", type="summary_text")],
                ),
                function_call(
                    "classify_question_type",
                    {"question_text": question},
                    call_id="call-jev-1",
                ),
            ],
            [assistant_message("这道题的题型判定为：解答题。")],
        ])
        agent = Agent(
            name=AGENT_NAME,
            model=model,
            instructions=AGENT_INSTRUCTIONS,
            tools=[classify_question_type],
        )

        recorder = _Recorder()
        original = agent_core.console
        agent_core.console = recorder
        try:
            final = await chat_streamed(agent, f"请判定这道题的题型：{question}")
        finally:
            agent_core.console = original
        text = recorder.export_text()

        _check(results, "端到端：工具被真实调用（假服务收到请求）", len(received) >= 1)
        _check(results, "端到端：🔧 工具调用已渲染",
               "🔧 调用工具" in text and "classify_question_type" in text)
        _check(results, "端到端：📤 工具返回已渲染",
               "📤 工具返回" in text and "解答题" in text)
        _check(results, "端到端：工具结果回传给模型",
               any("0.93" in str(c.input) for c in model.calls))
        _check(results, "端到端：最终答案非空", bool((final or "").strip()), (final or "")[:30])
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
