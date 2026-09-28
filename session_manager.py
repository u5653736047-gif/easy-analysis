"""会话管理模块（Session Manager）。

与 agent_core 彻底解耦：本模块只负责「会话」这一件事的管理，不接触
Agent / Runner / 渲染。唯一的耦合点是**共用同一个 SQLite 库文件**：

- 历史消息表：agent_sessions / agent_messages —— 由 openai-agents 的
  SQLiteSession 读写（Runner 持久化就靠它）；
- 元数据表：session_meta —— 本模块维护（标题/状态/时间戳/预览等），
  session_id 与 agent_sessions 外键关联、级联删除，
  因此「元数据」和「历史消息」永远不会漂移。

为后期 GUI 提供的读接口全部返回 SessionInfo dataclass（带 to_dict），
支持列表分页/排序/搜索/统计/导出导入；写接口带并发锁与状态机校验。

会话生命周期（隔离的核心）：

    create ──> active ──archive──> archived ──restore──> active
                  │                                    
                  └──delete(软删)──> deleted ──restore──> active
                                        │
                                   purge(硬删，消息一并级联清除)

默认列表只看 active；GUI 上可按状态切 tab。
"""

from __future__ import annotations

import functools
import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from agents import SQLiteSession

DEFAULT_DB = "conversation_history.db"

# SDK SQLiteSession 的表名（默认值在此固化，本模块直接按名操作）
SDK_SESSIONS_TABLE = "agent_sessions"
SDK_MESSAGES_TABLE = "agent_messages"

SCHEMA_VERSION = 1

# 会话状态
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"
STATUS_DELETED = "deleted"
ALL_STATUSES = (STATUS_ACTIVE, STATUS_ARCHIVED, STATUS_DELETED)

# session_id 白名单：只允许安全字符（防注入/路径穿越，也便于做 URL 参数）
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

DEFAULT_TITLE = "未命名会话"
PREVIEW_MAX = 80
TITLE_MAX = 60


# --------------------------------------------------------------------------- #
# 异常（GUI 层可映射为 HTTP 状态码）
# --------------------------------------------------------------------------- #
class SessionError(Exception):
    """会话管理基础异常。"""


class SessionNotFoundError(SessionError):
    """会话不存在（或状态不符）。"""


class InvalidSessionIDError(SessionError):
    """session_id 不合法（为空或含非法字符）。"""


class SessionExistsError(SessionError):
    """session_id 已存在。"""


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SessionInfo:
    """会话元数据（GUI 列表/详情直接消费）。"""

    session_id: str
    title: str
    status: str
    created_at: str  # ISO 8601 UTC
    updated_at: str  # ISO 8601 UTC
    message_count: int = 0
    preview: str = ""
    archived_at: str | None = None
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def display_time(self) -> str:
        """updated_at 转本地时间，用于 CLI 展示。"""
        return format_ts(self.updated_at)


def format_ts(iso_utc: str | None) -> str:
    """ISO UTC 时间串 -> 本地 'MM-DD HH:MM'；解析失败原样返回。"""
    if not iso_utc:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_utc)
    except ValueError:
        return iso_utc
    return dt.astimezone().strftime("%m-%d %H:%M")


def _now() -> str:
    # 微秒精度：同一毫秒内连续创建/激活多个会话时排序依然可靠（GUI 列表会看到）
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _message_text(raw_json: str) -> tuple[str, str]:
    """从 agent_messages.message_data 提取 (role, 纯文本)。"""
    try:
        item = json.loads(raw_json)
    except (ValueError, TypeError):
        return "", ""
    role = str(item.get("role", ""))
    content = item.get("content", "")
    if isinstance(content, str):
        return role, content
    if isinstance(content, list):  # Responses API 的分段 content
        parts = [
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") in (None, "output_text", "text")
        ]
        return role, " ".join(t for t in parts if t)
    return role, ""


def _auto_title(first_user_text: str) -> str:
    return _clip(first_user_text, 24) or DEFAULT_TITLE


# --------------------------------------------------------------------------- #
# 管理器
# --------------------------------------------------------------------------- #
def _synchronized(fn: Callable) -> Callable:
    """公开方法统一加实例级 RLock 守卫。

    SQLite 连接是进程内共享的（GUI 会多线程调），即使串行化，
    两个线程同时 execute 同一连接也会触发 SQLITE_MISUSE；
    RLock 可重入，内部 _tx() 再拿锁不会死锁。
    """

    @functools.wraps(fn)
    def wrapper(self: SessionManager, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


class SessionManager:
    """SQLite 会话管理器（线程安全）。

    用法::

        mgr = SessionManager("conversation_history.db")
        info = mgr.create()                      # 自动生成 id
        session = mgr.open_session(info.session_id)   # 交给 Runner/CLI
        mgr.record_activity(info.session_id, "你好")   # 每轮对话后调用
        mgr.close()

    一个进程内可持有多个 SessionManager 实例（指向不同库）；
    对同一库的多实例/多线程访问由 WAL + busy_timeout + 实例级 RLock 保证安全。
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._migrate()

    # -- 连接与迁移 -------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：GUI 多线程调用；RLock 串行化所有操作
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")  # 级联删除的前提
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:
        """按 user_version 执行增量迁移（后期加表只加一个版本分支）。"""
        with self._lock, self._conn:
            self._ensure_sdk_schema()
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= SCHEMA_VERSION:
                return
            if version < 1:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_meta (
                        session_id    TEXT PRIMARY KEY,
                        title         TEXT NOT NULL DEFAULT '',
                        status        TEXT NOT NULL DEFAULT 'active',
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL,
                        archived_at   TEXT,
                        deleted_at    TEXT,
                        message_count INTEGER NOT NULL DEFAULT 0,
                        preview       TEXT NOT NULL DEFAULT '',
                        FOREIGN KEY (session_id)
                            REFERENCES agent_sessions(session_id) ON DELETE CASCADE
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_meta_status
                    ON session_meta (status, updated_at DESC)
                    """
                )
                # 双向回填：SDK 表里有而 meta 没有（旧库）——从 SDK 补 meta；
                # 反方向（meta 有而 SDK 表没有）——从 meta 补 SDK 表。
                self._conn.execute(
                    f"""
                    INSERT OR IGNORE INTO session_meta
                        (session_id, title, status, created_at, updated_at)
                    SELECT s.session_id, '', 'active', ?, ?
                    FROM {SDK_SESSIONS_TABLE} s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM session_meta m WHERE m.session_id = s.session_id
                    )
                    """,
                    (_now(), _now()),
                )
                self._conn.execute(
                    f"INSERT OR IGNORE INTO {SDK_SESSIONS_TABLE} (session_id) "
                    "SELECT session_id FROM session_meta"
                )
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _ensure_sdk_schema(self) -> None:
        """确保 SDK 的两张表存在（与 openai-agents 的 DDL 保持一致）。

        全新库打开时 SDK 还没初始化，这里先建；之后 SQLiteSession 的
        CREATE TABLE IF NOT EXISTS 自然成为空操作，两张表始终只有一份定义。
        """
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SDK_SESSIONS_TABLE} (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SDK_MESSAGES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES {SDK_SESSIONS_TABLE} (session_id)
                    ON DELETE CASCADE
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{SDK_MESSAGES_TABLE}_session_id
            ON {SDK_MESSAGES_TABLE} (session_id, id)
            """
        )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self._conn:  # 事务；异常自动 rollback
                yield self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 校验与内部查询 ---------------------------------------------------- #
    @staticmethod
    def _validate_id(session_id: str) -> str:
        if not session_id or not _SESSION_ID_RE.match(session_id):
            raise InvalidSessionIDError(
                f"非法 session_id：{session_id!r}（仅允许 1-64 位字母/数字/_/-）"
            )
        return session_id

    def _row(self, session_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM session_meta WHERE session_id = ?", (session_id,)
        ).fetchone()

    def _require(self, session_id: str) -> sqlite3.Row:
        row = self._row(session_id)
        if row is None:
            raise SessionNotFoundError(f"会话不存在：{session_id}")
        return row

    @staticmethod
    def _to_info(row: sqlite3.Row) -> SessionInfo:
        return SessionInfo(
            session_id=row["session_id"],
            title=row["title"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            preview=row["preview"],
            archived_at=row["archived_at"],
            deleted_at=row["deleted_at"],
        )

    def _live_message_count(self, session_id: str) -> int:
        return int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM {SDK_MESSAGES_TABLE} WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        )

    def _latest_text(self, session_id: str) -> tuple[str, str]:
        row = self._conn.execute(
            f"SELECT message_data FROM {SDK_MESSAGES_TABLE} "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return _message_text(row["message_data"]) if row else ("", "")

    # -- 创建 / 查询 ------------------------------------------------------- #
    @_synchronized
    def create(
        self, session_id: str | None = None, *, title: str = ""
    ) -> SessionInfo:
        """创建新会话；title 缺省首次对话时由首条用户消息自动生成。"""
        if session_id is None:
            session_id = f"s-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._validate_id(session_id)
        if self._row(session_id) is not None:
            raise SessionExistsError(f"会话已存在：{session_id}")

        now = _now()
        with self._tx() as conn:
            conn.execute(
                f"INSERT INTO {SDK_SESSIONS_TABLE} (session_id, created_at, updated_at) "
                "VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO session_meta "
                "(session_id, title, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, _clip(title, TITLE_MAX) or DEFAULT_TITLE,
                 STATUS_ACTIVE, now, now),
            )
        return self._to_info(self._require(session_id))

    @_synchronized
    def ensure(self, session_id: str, *, title: str = "") -> SessionInfo:
        """存在则返回（不论状态），不存在则创建。CLI `--session` 恢复用。"""
        self._validate_id(session_id)
        row = self._row(session_id)
        if row is not None:
            info = self._to_info(row)
            info.message_count = self._live_message_count(session_id)
            return info
        return self.create(session_id, title=title)

    @_synchronized
    def get(self, session_id: str) -> SessionInfo | None:
        """取会话元数据；message_count 实时重算（别的进程写入也准确）。"""
        self._validate_id(session_id)
        row = self._row(session_id)
        if row is None:
            return None
        info = self._to_info(row)
        info.message_count = self._live_message_count(session_id)
        return info

    @_synchronized
    def list(
        self,
        *,
        status: str = STATUS_ACTIVE,
        limit: int = 50,
        offset: int = 0,
        order: str = "updated_at",
    ) -> list[SessionInfo]:
        """分页列出会话（默认最近活跃的 active 会话，GUI 列表直接可用）。"""
        if status not in ALL_STATUSES:
            raise SessionError(f"未知状态：{status}")
        if order not in ("updated_at", "created_at"):
            raise SessionError(f"未知排序字段：{order}")
        rows = self._conn.execute(
            f"SELECT * FROM session_meta WHERE status = ? "
            f"ORDER BY {order} DESC, rowid DESC LIMIT ? OFFSET ?",
            (status, max(0, limit), max(0, offset)),
        ).fetchall()
        return [self._to_info(r) for r in rows]

    @_synchronized
    def search(
        self, keyword: str, *, status: str | None = None, limit: int = 20
    ) -> list[SessionInfo]:
        """按标题/预览模糊搜索（GUI 搜索框）。"""
        kw = f"%{keyword.strip()}%"
        sql = "SELECT * FROM session_meta WHERE (title LIKE ? OR preview LIKE ?)"
        params: list[Any] = [kw, kw]
        if status is not None:
            if status not in ALL_STATUSES:
                raise SessionError(f"未知状态：{status}")
            sql += " AND status = ?"
            params.append(status)
        rows = self._conn.execute(
            sql + " ORDER BY updated_at DESC, rowid DESC LIMIT ?", (*params, max(0, limit))
        ).fetchall()
        return [self._to_info(r) for r in rows]

    @_synchronized
    def latest_active(self) -> SessionInfo | None:
        rows = self.list(status=STATUS_ACTIVE, limit=1)
        return rows[0] if rows else None

    # -- 更新 -------------------------------------------------------------- #
    @_synchronized
    def rename(self, session_id: str, title: str) -> SessionInfo:
        self._validate_id(session_id)
        title = _clip(title, TITLE_MAX)
        if not title:
            raise SessionError("标题不能为空")
        with self._tx() as conn:
            self._require(session_id)
            conn.execute(
                "UPDATE session_meta SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, _now(), session_id),
            )
        return self._to_info(self._require(session_id))

    @_synchronized
    def touch(self, session_id: str) -> SessionInfo:
        """刷新 updated_at 并重算消息数/预览（对话结束后调用）。"""
        self._validate_id(session_id)
        with self._tx() as conn:
            self._require(session_id)
            count = self._live_message_count(session_id)
            _, text = self._latest_text(session_id)
            conn.execute(
                "UPDATE session_meta SET updated_at = ?, message_count = ?, preview = ? "
                "WHERE session_id = ?",
                (_now(), count, _clip(text, PREVIEW_MAX), session_id),
            )
        return self._to_info(self._require(session_id))

    @_synchronized
    def record_activity(self, session_id: str, user_input: str = "") -> SessionInfo:
        """一轮对话结束后的统一记账：touch + 首条消息自动命名。

        auto-title 只在标题还是默认值时触发，用户/程序改名过就不覆盖。
        """
        self._validate_id(session_id)
        info = self.touch(session_id)
        row = self._row(session_id)
        if row is None or not user_input.strip():
            return info
        if row["title"] in ("", DEFAULT_TITLE):
            return self.rename(session_id, _auto_title(user_input))
        return info

    # -- 状态机：归档 / 软删 / 恢复 / 硬删 --------------------------------- #
    @_synchronized
    def archive(self, session_id: str) -> SessionInfo:
        return self._set_status(session_id, STATUS_ARCHIVED)

    @_synchronized
    def restore(self, session_id: str) -> SessionInfo:
        """从 archived / deleted 恢复为 active。"""
        return self._set_status(session_id, STATUS_ACTIVE)

    @_synchronized
    def delete(self, session_id: str) -> SessionInfo:
        """软删除：进回收站，消息保留，可 restore。"""
        return self._set_status(session_id, STATUS_DELETED)

    def _set_status(self, session_id: str, status: str) -> SessionInfo:
        self._validate_id(session_id)
        if status not in ALL_STATUSES:
            raise SessionError(f"未知状态：{status}")
        now = _now()
        with self._tx() as conn:
            self._require(session_id)
            conn.execute(
                "UPDATE session_meta SET status = ?, updated_at = ?, "
                "archived_at = CASE WHEN ? = 'archived' THEN ? ELSE NULL END, "
                "deleted_at  = CASE WHEN ? = 'deleted'  THEN ? ELSE NULL END "
                "WHERE session_id = ?",
                (status, now, status, now, status, now, session_id),
            )
        return self._to_info(self._require(session_id))

    @_synchronized
    def purge(self, session_id: str) -> None:
        """硬删除：元数据与历史消息一起级联清除，不可恢复。"""
        self._validate_id(session_id)
        with self._tx() as conn:
            self._require(session_id)
            conn.execute(
                f"DELETE FROM {SDK_SESSIONS_TABLE} WHERE session_id = ?", (session_id,)
            )  # FK ON DELETE CASCADE 同时清掉 session_meta 与 agent_messages

    @_synchronized
    def empty_trash(self) -> int:
        """清空回收站（硬删所有 deleted 会话），返回清除条数。"""
        rows = self._conn.execute(
            "SELECT session_id FROM session_meta WHERE status = ?", (STATUS_DELETED,)
        ).fetchall()
        for row in rows:
            self.purge(row["session_id"])
        return len(rows)

    # -- 与 Runner / CLI 的衔接 -------------------------------------------- #
    def open_session(
        self, session_id: str, sessions_table: str = SDK_SESSIONS_TABLE,
        messages_table: str = SDK_MESSAGES_TABLE,
    ) -> SQLiteSession:
        """返回绑定好的 SQLiteSession，交给 Runner.run_streamed(session=...)。

        不校验存在性：SDK 会在首次写入时自动建行（幂等）。
        """
        self._validate_id(session_id)
        return SQLiteSession(session_id, self.db_path, sessions_table, messages_table)

    @_synchronized
    def get_history(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """同步读取历史消息（GUI 消息列表 / CLI /history 用）。

        与 Runner 写入口径一致：直接读 agent_messages。
        """
        self._validate_id(session_id)
        sql = (
            f"SELECT message_data FROM {SDK_MESSAGES_TABLE} WHERE session_id = ? "
            "ORDER BY id ASC"
        )
        rows = self._conn.execute(sql, (session_id,)).fetchall()
        items = [json.loads(r["message_data"]) for r in rows]
        return items[-limit:] if limit else items

    # -- 导出 / 导入 -------------------------------------------------------- #
    @_synchronized
    def export_session(self, session_id: str) -> dict[str, Any]:
        """导出为可序列化 dict（含元数据 + 全量历史消息）。"""
        info = self.get(session_id)
        if info is None:
            raise SessionNotFoundError(f"会话不存在：{session_id}")
        return {
            "version": SCHEMA_VERSION,
            "session": info.to_dict(),
            "messages": self.get_history(session_id),
        }

    @_synchronized
    def import_session(
        self, data: dict[str, Any], *, session_id: str | None = None
    ) -> SessionInfo:
        """从 export_session 的 dict 导入；session_id 缺省用数据里的。"""
        meta = data.get("session", {})
        target = session_id or meta.get("session_id", "")
        self._validate_id(target)
        if self._row(target) is not None:
            raise SessionExistsError(f"目标会话已存在：{target}")

        now = _now()
        with self._tx() as conn:
            conn.execute(
                f"INSERT INTO {SDK_SESSIONS_TABLE} (session_id) VALUES (?)", (target,)
            )
            conn.execute(
                "INSERT INTO session_meta "
                "(session_id, title, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    target,
                    _clip(str(meta.get("title", "")), TITLE_MAX) or DEFAULT_TITLE,
                    STATUS_ACTIVE,
                    meta.get("created_at", now),
                    now,
                ),
            )
            conn.executemany(
                f"INSERT INTO {SDK_MESSAGES_TABLE} (session_id, message_data) "
                "VALUES (?, ?)",
                [(target, json.dumps(m, ensure_ascii=False)) for m in data.get("messages", [])],
            )
        return self.touch(target)

    # -- 维护 --------------------------------------------------------------- #
    @_synchronized
    def stats(self) -> dict[str, Any]:
        """总览统计（GUI 仪表盘 / CLI /stats）。"""
        by_status = {
            s: self._conn.execute(
                "SELECT COUNT(*) FROM session_meta WHERE status = ?", (s,)
            ).fetchone()[0]
            for s in ALL_STATUSES
        }
        total_messages = self._conn.execute(
            f"SELECT COUNT(*) FROM {SDK_MESSAGES_TABLE}"
        ).fetchone()[0]
        size = (
            Path(self.db_path).stat().st_size
            if Path(self.db_path).exists()
            else 0
        )
        return {
            "db_path": self.db_path,
            "sessions": sum(by_status.values()),
            **{f"sessions_{s}": c for s, c in by_status.items()},
            "messages": total_messages,
            "db_size_bytes": size,
        }

    @_synchronized
    def repair(self) -> dict[str, int]:
        """修复漂移：补缺失的元数据行、清孤儿、重算全部计数。

        场景：库被 SDK 单独写过（有 agent_sessions 行无 meta 行）、
        或 meta 行残留但消息已被外部清掉。
        """
        fixed_meta = fixed_messages = removed_orphans = 0
        with self._tx() as conn:
            now = _now()
            cur = conn.execute(
                f"""
                INSERT INTO session_meta (session_id, title, status, created_at, updated_at)
                SELECT s.session_id, '', 'active', ?, ? FROM {SDK_SESSIONS_TABLE} s
                WHERE NOT EXISTS (
                    SELECT 1 FROM session_meta m WHERE m.session_id = s.session_id
                )
                """,
                (now, now),
            )
            fixed_meta = cur.rowcount or 0

            for row in conn.execute(
                "SELECT session_id FROM session_meta"
            ).fetchall():
                sid = row["session_id"]
                count = self._live_message_count(sid)
                _, text = self._latest_text(sid)
                conn.execute(
                    "UPDATE session_meta SET message_count = ?, preview = ?, "
                    "title = CASE WHEN title = '' THEN ? ELSE title END "
                    "WHERE session_id = ?",
                    (count, _clip(text, PREVIEW_MAX), DEFAULT_TITLE, sid),
                )
                fixed_messages += 1
            removed_orphans = conn.execute(
                f"""
                DELETE FROM session_meta WHERE session_id NOT IN (
                    SELECT session_id FROM {SDK_SESSIONS_TABLE}
                )
                """
            ).rowcount or 0
        return {
            "meta_created": fixed_meta,
            "recounted": fixed_messages,
            "orphans_removed": removed_orphans,
        }
