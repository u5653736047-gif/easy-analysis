"""「小析易」Agent CLI 入口。

用法：
    python main.py                          # 交互式对话（自动恢复最近会话）
    python main.py --session s1             # 指定会话 ID，历史按 ID 分别持久化
    python main.py --model o4-mini          # 切换模型（推理模型才会展示思考链）
    python main.py --no-reasoning           # 关闭思考链，加快响应

环境变量：
    OPENAI_API_KEY   必填，OpenAI API Key
    OPENAI_MODEL     可选，默认 gpt-5-mini

会话管理（列表/搜索/归档/导出等）统一由 session_manager.SessionManager 负责，
本文件只做交互编排；输入 help 查看全部命令。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rich.console import Console

from agent_core import DEFAULT_DB, build_agent, chat_streamed
from session_manager import (
    SessionError,
    SessionManager,
    SessionNotFoundError,
    format_ts,
)

console = Console()

HELP_TEXT = """命令：
  help                 显示本帮助
  exit / quit          退出
  new [标题]           新建会话并切换（标题缺省时首条消息自动命名）
  sessions [a|d]       列出会话：默认活跃 ｜ a=已归档 ｜ d=回收站
  use <会话ID>         切换到指定会话（历史自动带上）
  rename <新标题>      重命名当前会话
  history [n]          查看当前会话最近 n 条消息（默认 10）
  search <关键词>      按标题/摘要搜索会话
  archive              归档当前会话（列表默认不再显示）
  restore              恢复当前会话（归档/回收站 -> 活跃）
  delete               当前会话移入回收站（restore 可找回）
  stats                查看会话库统计"""


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
        default=None,
        help="会话 ID；缺省时自动恢复最近活跃的会话，没有则新建 main",
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

        load_dotenv(Path(".env"))
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


# --------------------------------------------------------------------------- #
# 会话命令（每个命令 = 一个薄封装，逻辑全在 SessionManager）
# --------------------------------------------------------------------------- #
def _print_sessions(rows: list, *, header: str) -> None:
    console.print(f"[bold]{header}[/]（{len(rows)} 个）")
    if not rows:
        console.print("[dim]  （空）[/]")
        return
    for info in rows:
        badge = {"active": "●", "archived": "▣", "deleted": "✗"}.get(info.status, "?")
        console.print(
            f"  {badge} [bold cyan]{info.session_id}[/]  "
            f"[dim]{info.display_time()}[/]  "
            f"{info.title}  [dim]{info.message_count}条｜{info.preview or '（暂无消息）'}[/]"
        )


def _cmd_sessions(mgr: SessionManager, arg: str) -> None:
    mapping = {"": "active", "a": "archived", "d": "deleted"}
    status = mapping.get(arg.strip().lower())
    if status is None:
        console.print("[yellow]用法：sessions [a|d]（a=已归档，d=回收站）[/]")
        return
    _print_sessions(mgr.list(status=status, limit=30), header="会话列表")


def _cmd_use(mgr: SessionManager, current_id: str, arg: str) -> str | None:
    target = arg.strip()
    if not target:
        console.print("[yellow]用法：use <会话ID>[/]")
        return None
    info = mgr.get(target)
    if info is None:
        console.print(f"[yellow]会话不存在：{target}（用 sessions 查看可用会话）[/]")
        return None
    console.print(
        f"[green]✓ 已切换会话：{info.session_id}[/]  {info.title}"
        f"  [dim]（{info.message_count} 条历史，{info.display_time()}）[/]\n"
    )
    return target


def _cmd_history(mgr: SessionManager, session_id: str, arg: str) -> None:
    try:
        limit = int(arg.strip() or "10")
    except ValueError:
        console.print("[yellow]用法：history [n][/]")
        return
    items = mgr.get_history(session_id, limit=limit)
    rows = []
    for m in items:
        role = str(m.get("role", "?"))
        if not role or role == "?" or m.get("type") == "reasoning":
            continue  # reasoning/工具等中间条目不进对话历史视图
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        rows.append((role, str(content)[:200]))
    if not rows:
        console.print("[dim]（该会话暂无可展示的对话消息）[/]")
        return
    console.print(f"[bold]最近 {len(rows)} 条对话[/]")
    for role, content in rows:
        console.print(f"  [bold]{role}[/] > {content}")


def _cmd_search(mgr: SessionManager, arg: str) -> None:
    kw = arg.strip()
    if not kw:
        console.print("[yellow]用法：search <关键词>[/]")
        return
    _print_sessions(mgr.search(kw, limit=20), header=f"搜索「{kw}」")


def _cmd_export(mgr: SessionManager, session_id: str, arg: str) -> None:
    path = arg.strip() or f"session-{session_id}.json"
    data = mgr.export_session(session_id)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]✓ 已导出[/] {path}（{data['session']['message_count']} 条消息）")


def _cmd_import(mgr: SessionManager, arg: str) -> str | None:
    path = arg.strip()
    if not path:
        console.print("[yellow]用法：import <JSON路径>[/]")
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        info = mgr.import_session(data)
    except (OSError, ValueError, SessionError) as exc:
        console.print(f"[red]✗ 导入失败：{exc}[/]")
        return None
    console.print(f"[green]✓ 已导入会话：{info.session_id}[/]  {info.title}\n")
    return info.session_id


def _cmd_stats(mgr: SessionManager) -> None:
    s = mgr.stats()
    console.print(
        f"[bold]会话库[/] {s['db_path']}\n"
        f"  活跃 {s['sessions_active']} ｜ 归档 {s['sessions_archived']} ｜ "
        f"回收站 {s['sessions_deleted']} ｜ 合计 {s['sessions']}\n"
        f"  历史消息 {s['messages']} 条 ｜ 库大小 {s['db_size_bytes'] / 1024:.1f} KB"
    )


async def _handle_command(
    cmd: str, arg: str, mgr: SessionManager, current_id: str
) -> str | None:
    """处理斜杠/裸命令；返回新的当前会话 ID（未切换则原样返回）。"""
    if cmd in {"exit", "quit"}:
        raise SystemExit(0)
    if cmd == "help":
        console.print(HELP_TEXT)
    elif cmd == "new":
        info = mgr.create(title=arg.strip())
        console.print(f"[green]✓ 已创建新会话：{info.session_id}[/]  {info.title}\n")
        return info.session_id
    elif cmd == "sessions":
        _cmd_sessions(mgr, arg)
    elif cmd == "use":
        return _cmd_use(mgr, current_id, arg) or current_id
    elif cmd == "rename":
        try:
            info = mgr.rename(current_id, arg)
            console.print(f"[green]✓ 已重命名：{info.title}[/]\n")
        except SessionError as exc:
            console.print(f"[red]✗ {exc}[/]")
    elif cmd == "history":
        _cmd_history(mgr, current_id, arg)
    elif cmd == "search":
        _cmd_search(mgr, arg)
    elif cmd == "archive":
        info = mgr.archive(current_id)
        console.print(f"[green]✓ 已归档：{info.session_id}[/]（sessions a 可找回）\n")
    elif cmd == "restore":
        info = mgr.restore(current_id)
        console.print(f"[green]✓ 已恢复为活跃：{info.session_id}[/]\n")
    elif cmd == "delete":
        info = mgr.delete(current_id)
        console.print(f"[yellow]✓ 已移入回收站：{info.session_id}[/]（restore 可找回）\n")
    elif cmd == "export":
        _cmd_export(mgr, current_id, arg)
    elif cmd == "import":
        new_id = _cmd_import(mgr, arg)
        if new_id:
            return new_id
    elif cmd == "stats":
        _cmd_stats(mgr)
    else:
        console.print(f"[yellow]未知命令：{cmd}（输入 help 查看命令）[/]")
    return current_id


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
async def _run(args: argparse.Namespace) -> None:
    agent = build_agent(model=args.model, enable_reasoning=not args.no_reasoning)

    mgr = SessionManager(args.db)
    try:
        if args.session:
            info = mgr.ensure(args.session)
            if info.status != "active":
                info = mgr.restore(info.session_id)
        else:
            latest = mgr.latest_active()
            info = latest or mgr.ensure("main")

        console.rule(
            f"[bold cyan]小析易 Agent[/]  "
            f"会话: [bold]{info.session_id}[/]（{info.title}）  "
            f"模型: [bold]{agent.model}[/]  "
            f"历史消息: [bold]{info.message_count}[/] 条"
        )
        console.print("输入 [bold]help[/] 查看会话命令\n")

        current_id = info.session_id
        while True:
            try:
                user_input = await asyncio.to_thread(input, "你 > ")
            except EOFError:
                console.print("\n再见 👋")
                break
            text = user_input.strip()
            if not text:
                continue

            # 命令（兼容裸词与 /前缀两种写法）
            cmd, _, arg = text.lstrip("/").partition(" ")
            cmd, arg = cmd.strip().lower(), arg.strip()
            if cmd in COMMANDS:
                try:
                    switched = await _handle_command(cmd, arg, mgr, current_id)
                except SystemExit:
                    raise
                except SessionError as exc:
                    console.print(f"[red]✗ {exc}[/]")
                    continue
                if switched != current_id:
                    current_id = switched
                continue

            session = mgr.open_session(current_id)
            await chat_streamed(agent, text, session=session)
            current = mgr.record_activity(current_id, text)
            console.print(
                f"[dim]（会话 {current.session_id}｜{current.message_count} 条消息）[/]\n"
            )
    finally:
        mgr.close()


COMMANDS = {
    "help", "exit", "quit", "new", "sessions", "use", "rename",
    "history", "search", "archive", "restore", "delete", "export",
    "import", "stats",
}


def main() -> None:
    if not _prepare_env():
        sys.exit(1)
    args = parse_args()
    try:
        asyncio.run(_run(args))
    except (KeyboardInterrupt, SystemExit):
        console.print("\n再见 👋")


if __name__ == "__main__":
    main()
