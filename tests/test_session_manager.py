"""session_manager 专项测试。

全部用临时库文件 + 本地假数据，不碰网络、不调真实模型。
同时覆盖「与 SDK SQLiteSession 互通」：历史消息用 SDK 写入，
SessionManager 的计数/级联删除/记账必须与之一致。

运行：.venv/bin/python tests/test_session_manager.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    mark = PASS if ok else FAIL
    line = f"  {mark} {name}"
    if detail and not ok:
        line += f"  [dim]{detail}[/dim]"
    print(line)


def _tmp_db() -> str:
    return str(Path(tempfile.mkdtemp(prefix="xiaoxiyi-sess-")) / "history.db")


# --------------------------------------------------------------------------- #
def test_crud_and_listing(results: list) -> None:
    print("\n===== 创建 / 查询 / 列表 / 分页 =================================")
    from session_manager import (
        DEFAULT_TITLE,
        InvalidSessionIDError,
        SessionExistsError,
        SessionManager,
    )

    mgr = SessionManager(_tmp_db())
    try:
        a = mgr.create("s-alpha", title="第一次月考分析")
        b = mgr.create("s-beta", title="  期末复习  ")
        _check(results, "create 返回 active 状态", a.status == "active")
        _check(results, "create 自动裁剪标题空白", b.title == "期末复习")
        _check(results, "缺省标题为「未命名会话」",
               mgr.create("s-no-title").title == DEFAULT_TITLE)

        _check(results, "get 能取回", mgr.get("s-alpha").title == "第一次月考分析")
        _check(results, "get 不存在返回 None", mgr.get("s-none") is None)

        _check(results, "重复 create 抛 SessionExistsError",
               _raises(lambda: mgr.create("s-alpha"), SessionExistsError))
        _check(results, "空 session_id 抛 InvalidSessionIDError",
               _raises(lambda: mgr.create(""), InvalidSessionIDError))
        _check(results, "非法字符 session_id 被拒（注入/路径穿越）",
               _raises(lambda: mgr.create("bad/id;drop"), InvalidSessionIDError))

        rows = mgr.list(limit=10)
        _check(results, "list 默认按活跃时间倒序", [r.session_id for r in rows] ==
               ["s-no-title", "s-beta", "s-alpha"], str([r.session_id for r in rows]))
        page1 = mgr.list(limit=2, offset=0)
        page2 = mgr.list(limit=2, offset=2)
        _check(results, "分页 offset 生效", page1[0].session_id == "s-no-title"
               and page2[0].session_id == "s-alpha")

        c = mgr.rename("s-alpha", "  三次月考趋势  ")
        _check(results, "rename 裁剪并写入", c.title == "三次月考趋势")
        _check(results, "空 rename 被拒", _raises(lambda: mgr.rename("s-alpha", "  "), Exception))
        _check(results, "list order=created_at 可用", len(mgr.list(order="created_at")) == 3)
    finally:
        mgr.close()


def test_auto_title_and_preview(results: list) -> None:
    print("\n===== 自动标题 / 预览 / 记账 =====================================")
    from agents import SQLiteSession

    from session_manager import SessionManager

    db = _tmp_db()
    mgr = SessionManager(db)
    try:
        info = mgr.create("s-auto")
        _check(results, "新建时标题为默认值", info.title == "未命名会话")

        # 用 SDK 直接写两条消息（模拟 Runner 持久化）
        s = SQLiteSession("s-auto", db)
        import asyncio

        asyncio.run(s.add_items([
            {"role": "user", "content": "帮我分析初二(3)班本次期中考试的整体情况"},
            {"role": "assistant", "content": "已生成分析报告：均分73.7，需关注解答题。"},
        ]))

        updated = mgr.record_activity("s-auto", "帮我分析初二(3)班本次期中考试的整体情况")
        _check(results, "首条用户消息自动命名", updated.title.startswith("帮我分析初二"))
        _check(results, "标题长度受控", len(updated.title) <= 25)
        _check(results, "message_count 与 SDK 写入一致", updated.message_count == 2,
               str(updated.message_count))
        _check(results, "预览取自最新一条消息", "已生成分析报告" in updated.preview, updated.preview)

        # 改名后 record_activity 不再覆盖标题
        mgr.rename("s-auto", "固定标题")
        again = mgr.record_activity("s-auto", "新一轮提问")
        _check(results, "自定义标题不被自动命名覆盖", again.title == "固定标题")

        # get_history 与 SDK get_items 口径一致
        sdk_items = asyncio.run(s.get_items())
        mine = mgr.get_history("s-auto")
        _check(results, "get_history 与 SDK get_items 条数一致",
               len(sdk_items) == len(mine), f"{len(sdk_items)} vs {len(mine)}")
        _check(results, "get_history limit 生效", len(mgr.get_history("s-auto", limit=1)) == 1)
    finally:
        mgr.close()


def test_status_machine_and_isolation(results: list) -> None:
    print("\n===== 状态机 / 隔离 =============================================""")
    from session_manager import SessionManager

    db = _tmp_db()
    mgr = SessionManager(db)
    try:
        mgr.create("s-1", title="一")
        mgr.create("s-2", title="二")
        mgr.create("s-3", title="三")

        mgr.archive("s-2")
        mgr.delete("s-3")
        active = [r.session_id for r in mgr.list()]
        archived = [r.session_id for r in mgr.list(status="archived")]
        deleted = [r.session_id for r in mgr.list(status="deleted")]
        _check(results, "归档后从 active 列表消失", active == ["s-1"], str(active))
        _check(results, "archived 列表可见", archived == ["s-2"])
        _check(results, "软删后进 deleted 列表", deleted == ["s-3"])
        _check(results, " archived_at 已记录", mgr.get("s-2").archived_at is not None)
        _check(results, "deleted_at 已记录", mgr.get("s-3").deleted_at is not None)

        mgr.restore("s-3")
        _check(results, "restore 恢复为 active",
               [r.session_id for r in mgr.list()] == ["s-3", "s-1"])

        # 隔离性：不同 session_id 的历史互不串读
        import asyncio

        from agents import SQLiteSession

        asyncio.run(SQLiteSession("s-1", db).add_items(
            [{"role": "user", "content": "会话一的问题"}]))
        asyncio.run(SQLiteSession("s-x", db).add_items(
            [{"role": "user", "content": "会话X的问题"}]))
        _check(results, "会话间历史隔离（各读各的）",
               mgr.get_history("s-1")[-1]["content"] == "会话一的问题"
               and mgr.get_history("s-x")[-1]["content"] == "会话X的问题")
        _check(results, "未创建元数据的会话 get_history 也可读（SDK 自动建行）",
               len(mgr.get_history("s-x")) == 1)
    finally:
        mgr.close()


def test_cascade_and_trash(results: list) -> None:
    print("\n===== 级联删除 / 回收站 / 导出导入 ==============================""")
    import asyncio

    from agents import SQLiteSession

    from session_manager import SessionManager

    db = _tmp_db()
    mgr = SessionManager(db)
    try:
        mgr.create("s-del", title="待删除")
        asyncio.run(SQLiteSession("s-del", db).add_items(
            [{"role": "user", "content": "一些历史"}, {"role": "assistant", "content": "一些回答"}]))

        mgr.delete("s-del")
        _check(results, "软删后消息仍保留", mgr.get_history("s-del") and len(mgr.get_history("s-del")) == 2)

        # 导出 / 导入
        data = mgr.export_session("s-del")
        _check(results, "导出含元数据与消息", data["session"]["title"] == "待删除"
               and len(data["messages"]) == 2)
        imported = mgr.import_session(data, session_id="s-copy")
        _check(results, "导入生成新会话", imported.session_id == "s-copy"
               and imported.message_count == 2)
        _check(results, "导入的会话与源隔离",
               mgr.get_history("s-copy")[-1]["content"] == "一些回答")
        _check(results, "重复导入同 ID 被拒",
               _raises(lambda: mgr.import_session(data, session_id="s-copy"), Exception))

        # 硬删：meta + messages 一起级联消失
        mgr.purge("s-del")
        _check(results, "purge 后元数据消失", mgr.get("s-del") is None)
        _check(results, "purge 后历史消息级联删除", mgr.get_history("s-del") == [])
        raw = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM agent_messages WHERE session_id='s-del'").fetchone()[0]
        _check(results, "agent_messages 表内也无残留", raw == 0, str(raw))

        # 清空回收站
        n = mgr.empty_trash()
        _check(results, "empty_trash 清掉 deleted", n == 0 and mgr.list(status="deleted") == [])
        _check(results, "active 会话不受回收站清理影响",
               mgr.get("s-copy").status == "active")
    finally:
        mgr.close()


def test_search_stats_repair(results: list) -> None:
    print("\n===== 搜索 / 统计 / 修复 / 并发 ==================================""")
    import asyncio

    from agents import SQLiteSession

    from session_manager import SessionManager

    db = _tmp_db()
    mgr = SessionManager(db)
    try:
        for sid, title in (("s-a", "初二数学期中"), ("s-b", "初三英语期末"), ("s-c", "初二物理错题")):
            mgr.create(sid, title=title)
        asyncio.run(SQLiteSession("s-a", db).add_items(
            [{"role": "user", "content": "解答题得分率偏低"}]))
        mgr.record_activity("s-a", "解答题得分率偏低")

        _check(results, "search 命中标题（按活跃时间倒序）",
               [r.session_id for r in mgr.search("初二")] == ["s-a", "s-c"],
               str([r.session_id for r in mgr.search("初二")]))
        _check(results, "search 命中预览", [r.session_id for r in mgr.search("得分率")] == ["s-a"])
        _check(results, "search 无命中返回空", mgr.search("不存在的关键词xyz") == [])

        s = mgr.stats()
        _check(results, "stats 统计会话总数", s["sessions"] == 3, str(s))
        _check(results, "stats 统计消息数", s["messages"] == 1, str(s["messages"]))

        # 漂移修复：SDK 单独写过（有 agent_sessions 行无 meta 行）
        asyncio.run(SQLiteSession("s-orphan", db).add_items(
            [{"role": "user", "content": "孤儿会话"}]))
        fixed = mgr.repair()
        _check(results, "repair 补齐缺失的元数据行", fixed["meta_created"] == 1, str(fixed))
        info = mgr.get("s-orphan")
        _check(results, "repair 后孤儿会话可见（默认标题）",
               info is not None and info.title == "未命名会话" and info.message_count == 1)

        # 并发：多线程同时 touch 不同会话
        mgr.create("s-t1"); mgr.create("s-t2")
        errors: list[str] = []

        def worker(sid: str) -> None:
            try:
                for _ in range(20):
                    mgr.record_activity(sid, "并发写")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sid}: {exc}")

        threads = [threading.Thread(target=worker, args=(sid,))
                   for sid in ("s-t1", "s-t2", "s-t1", "s-t2")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        _check(results, "多线程并发写无异常（锁生效）", not errors, "; ".join(errors[:2]))
    finally:
        mgr.close()


def test_migration_and_compat(results: list) -> None:
    print("\n===== Schema 迁移 / 与旧版 create_session 数据兼容 ================""")
    from session_manager import SessionManager

    # 模拟「旧版」库：只有 SDK 两张表、有数据，没有 session_meta、user_version=0
    db = _tmp_db()
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE agent_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)
                ON DELETE CASCADE
        );
        INSERT INTO agent_sessions (session_id) VALUES ('legacy');
        INSERT INTO agent_messages (session_id, message_data)
        VALUES ('legacy', '{"role":"user","content":"旧版会话的第一条消息"}');
        """
    )
    conn.commit()
    conn.close()

    mgr = SessionManager(db)  # 打开即迁移
    try:
        info = mgr.get("legacy")
        _check(results, "旧库平滑升级：会话可见", info is not None)
        _check(results, "旧库消息计数正确", info is not None and info.message_count == 1,
               str(info and info.message_count))
        version = sqlite3.connect(db).execute("PRAGMA user_version").fetchone()[0]
        _check(results, "user_version 已推进", version == 1, str(version))

        # 反复打开（迁移幂等）
        mgr.close()
        mgr2 = SessionManager(db)
        try:
            _check(results, "迁移幂等（重复打开不报错）", mgr2.get("legacy") is not None)
        finally:
            mgr2.close()
    finally:
        mgr.close()


def _raises(fn, exc_type) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def main() -> int:
    results: list = []
    test_crud_and_listing(results)
    test_auto_title_and_preview(results)
    test_status_machine_and_isolation(results)
    test_cascade_and_trash(results)
    test_search_stats_repair(results)
    test_migration_and_compat(results)
    print("\n===== 结果 =====================================================")
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{FAIL} 未通过 {len(failed)}/{len(results)}：{failed}")
        return 1
    print(f"{PASS} 全部通过（{len(results)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
