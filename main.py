"""「小析易」Agent CLI 入口。

用法：
    python main.py                          # 交互式对话（默认会话 main）
    python main.py --session s1             # 指定会话 ID，历史按 ID 分别持久化
    python main.py --model o4-mini          # 切换模型（推理模型才会展示思考链）
    python main.py --no-reasoning           # 关闭思考链，加快响应

环境变量：
    OPENAI_API_KEY   必填，OpenAI API Key
    OPENAI_MODEL     可选，默认 gpt-5-mini
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime

from rich.console import Console

from agent_core import DEFAULT_DB, build_agent, chat_streamed, create_session

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="xiaoxiyi",
        description="小析易 Agent 基础框架（OpenAI Agents SDK）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL"),
        help="OpenAI 模型名，默认 gpt-5-mini",
    )
    parser.add_argument(
        "--session",
        default="main",
        help="会话 ID，用于区分/恢复历史会话，默认 main",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"会话历史 SQLite 文件路径，默认 {DEFAULT_DB}",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="关闭模型思考链（reasoning）输出",
    )
    return parser.parse_args()


def _prepare_env() -> bool:
    """加载 .env，并把第三方常用变量名映射为 SDK 标准变量。

    支持两套变量名：
    - OpenAI 官方：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    - 中转/兼容层 ：API_KEY / BASE_URL / MODEL_NAME
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    for src, dst in (
        ("API_KEY", "OPENAI_API_KEY"),
        ("BASE_URL", "OPENAI_BASE_URL"),
        ("MODEL_NAME", "OPENAI_MODEL"),
    ):
        if os.getenv(src) and not os.getenv(dst):
            os.environ[dst] = os.environ[src]

    # 非 OpenAI 官方端点时禁用 SDK tracing（否则会尝试上报到 OpenAI ingest 失败并刷屏）
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if base_url and "api.openai.com" not in base_url:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)

    if os.getenv("OPENAI_API_KEY"):
        return True
    console.print("[bold red]✗ 未检测到 API Key[/]")
    console.print("  方式一：export OPENAI_API_KEY=sk-...")
    console.print("  方式二：在 .env 中配置 OPENAI_API_KEY=sk-...（或 API_KEY=sk-...）")
    return False


async def _run(args: argparse.Namespace) -> None:
    agent = build_agent(model=args.model, enable_reasoning=not args.no_reasoning)
    session = create_session(args.session, args.db)

    # 展示历史会话规模，验证「历史会话能力」已生效
    try:
        history = await session.get_items()
        history_count = len(history)
    except Exception:  # noqa: BLE001
        history_count = 0

    console.rule(
        f"[bold cyan]小析易 Agent[/]  "
        f"会话: [bold]{args.session}[/]  "
        f"模型: [bold]{agent.model}[/]  "
        f"历史消息: [bold]{history_count}[/] 条"
    )
    console.print("命令：[bold]exit[/] 退出 ｜ [bold]new[/] 开启新会话\n")

    current_session = session
    while True:
        try:
            user_input = await asyncio.to_thread(input, "你 > ")
        except EOFError:
            console.print("\n再见 👋")
            break
        text = user_input.strip()
        if not text:
            continue
        if text in {"exit", "quit"}:
            console.print("再见 👋")
            break
        if text == "new":
            new_id = f"session-{datetime.now():%Y%m%d-%H%M%S}"
            current_session = create_session(new_id, args.db)
            console.print(f"[green]✓ 已切换到新会话: {new_id}[/]\n")
            continue
        await chat_streamed(agent, text, session=current_session)
        console.print("")


def main() -> None:
    if not _prepare_env():
        sys.exit(1)
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
