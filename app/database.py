from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    password_changed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_sessions (
    token_hash TEXT PRIMARY KEY,
    admin_id INTEGER NOT NULL REFERENCES admins(id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT
);

CREATE TABLE IF NOT EXISTS login_failures (
    key TEXT PRIMARY KEY,
    failures INTEGER NOT NULL DEFAULT 0,
    first_failure_at TEXT NOT NULL,
    locked_until TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    is_secret INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    updated_by TEXT
);

CREATE TABLE IF NOT EXISTS guilds (
    guild_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    system_prompt TEXT,
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_settings (
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL DEFAULT '',
    listen_enabled INTEGER NOT NULL DEFAULT 0,
    proactive_enabled INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id TEXT,
    username TEXT,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('explicit', 'fact', 'preference', 'summary')),
    content TEXT NOT NULL,
    source_message_id INTEGER,
    confidence REAL NOT NULL DEFAULT 1.0,
    importance REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'forgotten')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS channel_summaries (
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    through_message_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    familiarity REAL NOT NULL DEFAULT 0.0,
    trust REAL NOT NULL DEFAULT 0.0,
    warmth REAL NOT NULL DEFAULT 0.0,
    fatigue REAL NOT NULL DEFAULT 0.0,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    last_interaction_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE IF NOT EXISTS bot_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL UNIQUE,
    keywords_json TEXT NOT NULL DEFAULT '[]',
    weight REAL NOT NULL DEFAULT 0.0,
    source TEXT NOT NULL DEFAULT 'seed',
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    locked INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mood_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    valence REAL NOT NULL DEFAULT 0.0,
    energy REAL NOT NULL DEFAULT 0.6,
    social_budget REAL NOT NULL DEFAULT 0.7,
    label TEXT NOT NULL DEFAULT '平静',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proactive_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_context
ON messages(guild_id, channel_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_messages_expiry
ON messages(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_owner
ON memories(guild_id, user_id, status, importance DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_memories_expiry
ON memories(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_expiry
ON admin_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_proactive_channel_time
ON proactive_log(guild_id, channel_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_time
ON audit_log(created_at DESC);
"""


MIGRATION_2 = """
CREATE TABLE IF NOT EXISTS discord_admins (
    user_id TEXT PRIMARY KEY,
    note TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    style_json TEXT NOT NULL DEFAULT '{}',
    boundaries_json TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_terms (
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    term TEXT NOT NULL,
    PRIMARY KEY (memory_id, term)
);

CREATE TABLE IF NOT EXISTS open_loops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    public_safe INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed', 'expired')),
    followup_after TEXT,
    expires_at TEXT,
    followup_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback_events (
    message_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    origin_user_id TEXT,
    guild_id TEXT NOT NULL,
    emoji TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (message_id, user_id, emoji)
);

CREATE TABLE IF NOT EXISTS safety_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'custom',
    direction TEXT NOT NULL DEFAULT 'both' CHECK(direction IN ('input', 'output', 'both')),
    pattern TEXT NOT NULL,
    match_type TEXT NOT NULL DEFAULT 'contains' CHECK(match_type IN ('contains', 'word', 'regex')),
    action TEXT NOT NULL DEFAULT 'block' CHECK(action IN ('block', 'redact', 'log')),
    replacement TEXT NOT NULL DEFAULT '[已隐藏]',
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 100,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS safety_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER REFERENCES safety_rules(id) ON DELETE SET NULL,
    guild_id TEXT,
    channel_id TEXT,
    user_id TEXT,
    direction TEXT NOT NULL,
    category TEXT NOT NULL,
    action TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog_cache (
    provider TEXT NOT NULL,
    endpoint_hash TEXT NOT NULL,
    models_json TEXT NOT NULL DEFAULT '[]',
    fetched_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_error TEXT,
    PRIMARY KEY (provider, endpoint_hash)
);

CREATE TABLE IF NOT EXISTS usage_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    guild_id TEXT,
    user_id TEXT,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ok',
    error_code TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    guild_id TEXT,
    user_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    run_after TEXT NOT NULL,
    locked_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_sessions (
    id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source_start_id TEXT,
    source_end_id TEXT,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT,
    source_user_id TEXT,
    kind TEXT NOT NULL DEFAULT 'experience',
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    importance REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    locked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

ALTER TABLE messages ADD COLUMN discord_message_id TEXT;
ALTER TABLE messages ADD COLUMN reply_to_message_id TEXT;
ALTER TABLE memories ADD COLUMN normalized_content TEXT NOT NULL DEFAULT '';
ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE memories ADD COLUMN last_recalled_at TEXT;
ALTER TABLE memories ADD COLUMN last_confirmed_at TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_discord_id
ON messages(guild_id, channel_id, discord_message_id)
WHERE discord_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_terms_lookup
ON memory_terms(guild_id, user_id, term, memory_id);
CREATE INDEX IF NOT EXISTS idx_open_loops_due
ON open_loops(status, followup_after, expires_at);
CREATE INDEX IF NOT EXISTS idx_feedback_origin
ON feedback_events(origin_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_safety_rules_active
ON safety_rules(enabled, direction, priority, id);
CREATE INDEX IF NOT EXISTS idx_safety_events_time
ON safety_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_time
ON usage_metrics(created_at DESC, kind, provider, model);
CREATE INDEX IF NOT EXISTS idx_jobs_ready
ON background_jobs(status, run_after, id);
CREATE INDEX IF NOT EXISTS idx_summary_sessions_expiry
ON summary_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_bot_experiences_scope
ON bot_experiences(guild_id, importance DESC, updated_at DESC);
"""

MIGRATION_3 = """
CREATE TABLE IF NOT EXISTS guild_personas (
    guild_id TEXT PRIMARY KEY REFERENCES guilds(guild_id) ON DELETE CASCADE,
    system_prompt TEXT NOT NULL CHECK(length(trim(system_prompt)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO guild_personas(guild_id, system_prompt, created_at, updated_at)
SELECT guild_id, trim(system_prompt), updated_at, updated_at
FROM guilds
WHERE system_prompt IS NOT NULL AND length(trim(system_prompt)) > 0;

UPDATE guilds SET system_prompt = NULL WHERE system_prompt IS NOT NULL;

DELETE FROM channel_settings
WHERE listen_enabled = 0 AND proactive_enabled = 0;
"""


MIGRATIONS: tuple[tuple[int, str], ...] = ((2, MIGRATION_2), (3, MIGRATION_3))


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utcnow().isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()

    async def _open(self) -> aiosqlite.Connection:
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA busy_timeout = 5000")
        return self._connection

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._connection_lock:
            connection = await self._open()
            try:
                yield connection
            except BaseException:
                # asyncio.CancelledError inherits BaseException.  Message
                # generation is intentionally cancellable, so a cancellation
                # must never leave its SQLite transaction open for the next
                # request that reuses this persistent connection.
                try:
                    if connection.in_transaction:
                        await asyncio.shield(connection.rollback())
                finally:
                    raise

    async def close(self) -> None:
        async with self._connection_lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        await self._backup_before_upgrade()
        async with self.connect() as connection:
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA synchronous = NORMAL")
            await connection.executescript(SCHEMA)
            now = iso_now()
            await connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(1, ?)",
                (now,),
            )
            await connection.commit()
            applied_rows = await connection.execute_fetchall(
                "SELECT version FROM schema_migrations"
            )
            applied = {int(row[0]) for row in applied_rows}
            for version, script in MIGRATIONS:
                if version in applied:
                    continue
                escaped_now = now.replace("'", "''")
                await connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + script
                    + f"\nINSERT INTO schema_migrations(version, applied_at) VALUES({version}, '{escaped_now}');\n"
                    + f"PRAGMA user_version = {version};\nCOMMIT;"
                )
            await connection.execute(
                """INSERT OR IGNORE INTO mood_state
                   (id, valence, energy, social_budget, label, updated_at)
                   VALUES(1, 0.0, 0.6, 0.7, '平静', ?)""",
                (now,),
            )
            seeds = (
                ("Discord 社区设计", ["discord", "dc", "频道", "身份组", "机器人"], 0.82),
                ("编程与系统", ["python", "代码", "api", "部署", "数据库"], 0.75),
                ("古典音乐", ["柴可夫斯基", "序曲", "古典", "交响", "音乐"], 0.68),
                ("科幻", ["科幻", "太空", "宇宙", "farscape"], 0.55),
            )
            for topic, keywords, weight in seeds:
                await connection.execute(
                    """INSERT OR IGNORE INTO bot_preferences
                       (topic, keywords_json, weight, source, confidence,
                        evidence_count, locked, updated_at)
                       VALUES(?, ?, ?, 'seed', 0.8, 0, 0, ?)""",
                    (topic, json.dumps(keywords, ensure_ascii=False), weight, now),
                )
            await connection.execute("PRAGMA optimize")
            await connection.commit()

    async def _backup_before_upgrade(self) -> Path | None:
        """Create a consistent SQLite backup before a pending schema upgrade."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        target_version = max(version for version, _script in MIGRATIONS)

        def backup() -> Path | None:
            with sqlite3.connect(self.path) as source:
                table = source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
                ).fetchone()
                current_version = 0
                if table:
                    row = source.execute(
                        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                    ).fetchone()
                    current_version = int(row[0]) if row else 0
                if current_version >= target_version:
                    return None
                stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
                destination = self.path.with_name(
                    f"{self.path.name}.pre-v{target_version}-{stamp}.bak"
                )
                with sqlite3.connect(destination) as output:
                    source.backup(output)
                return destination

        return await asyncio.to_thread(backup)

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        async with self.connect() as connection:
            try:
                cursor = await connection.execute(sql, parameters)
                await connection.commit()
                if sql.lstrip().upper().startswith("INSERT") and cursor.lastrowid:
                    return int(cursor.lastrowid)
                return max(0, int(cursor.rowcount))
            except Exception:
                await connection.rollback()
                raise

    async def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> None:
        async with self.connect() as connection:
            try:
                await connection.executemany(sql, parameters)
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

    async def fetchone(self, sql: str, parameters: Sequence[Any] = ()) -> dict[str, Any] | None:
        async with self.connect() as connection:
            cursor = await connection.execute(sql, parameters)
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetchall(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self.connect() as connection:
            cursor = await connection.execute(sql, parameters)
            return [dict(row) for row in await cursor.fetchall()]

    async def scalar(self, sql: str, parameters: Sequence[Any] = ()) -> Any:
        row = await self.fetchone(sql, parameters)
        return next(iter(row.values())) if row else None

    async def cleanup_expired(self) -> dict[str, int]:
        now = iso_now()
        deleted: dict[str, int] = {}
        async with self.connect() as connection:
            for table, sql in (
                (
                    "messages",
                    "DELETE FROM messages WHERE expires_at IS NOT NULL AND expires_at <= ?",
                ),
                (
                    "memories",
                    "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                ),
                (
                    "open_loops",
                    "DELETE FROM open_loops WHERE expires_at IS NOT NULL AND expires_at <= ?",
                ),
                (
                    "summary_sessions",
                    "DELETE FROM summary_sessions WHERE expires_at <= ?",
                ),
                (
                    "bot_experiences",
                    "DELETE FROM bot_experiences WHERE expires_at IS NOT NULL AND expires_at <= ?",
                ),
                (
                    "model_catalog_cache",
                    "DELETE FROM model_catalog_cache WHERE expires_at <= ?",
                ),
                ("admin_sessions", "DELETE FROM admin_sessions WHERE expires_at <= ?"),
            ):
                cursor = await connection.execute(sql, (now,))
                deleted[table] = cursor.rowcount
            await connection.commit()
        return deleted

    async def health(self) -> bool:
        try:
            value = await self.scalar("SELECT 1 AS ok")
            return value == 1
        except (aiosqlite.Error, OSError):
            return False

    async def audit(
        self,
        actor: str,
        action: str,
        *,
        target: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> None:
        await self.execute(
            """INSERT INTO audit_log
               (actor, action, target, details_json, ip_address, created_at)
               VALUES(?, ?, ?, ?, ?, ?)""",
            (
                actor,
                action,
                target,
                json.dumps(details or {}, ensure_ascii=False),
                ip_address,
                iso_now(),
            ),
        )

    async def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        queries = {
            "guilds": "SELECT COUNT(*) AS n FROM guilds",
            "messages": "SELECT COUNT(*) AS n FROM messages",
            "memories": "SELECT COUNT(*) AS n FROM memories",
            "relationships": "SELECT COUNT(*) AS n FROM relationships",
            "bot_preferences": "SELECT COUNT(*) AS n FROM bot_preferences",
            "channel_summaries": "SELECT COUNT(*) AS n FROM channel_summaries",
            "user_profiles": "SELECT COUNT(*) AS n FROM user_profiles",
            "discord_admins": "SELECT COUNT(*) AS n FROM discord_admins WHERE enabled = 1",
            "open_loops": "SELECT COUNT(*) AS n FROM open_loops WHERE status = 'open'",
        }
        for name, sql in queries.items():
            counts[name] = int(await self.scalar(sql) or 0)
        return counts

    async def purge_user(self, user_id: str) -> None:
        """Delete all data owned by one Discord user across every privacy scope."""
        async with self.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                # Derive affected guilds from the user's messages before
                # deleting them — needed to scope flow-side-channel cleanup.
                cursor = await connection.execute(
                    "SELECT DISTINCT guild_id FROM messages WHERE user_id = ?",
                    (user_id,),
                )
                affected_guilds = [row["guild_id"] for row in await cursor.fetchall()]

                await connection.execute(
                    "DELETE FROM messages WHERE user_id = ?",
                    (user_id,),
                )
                await connection.execute(
                    "DELETE FROM memories WHERE user_id = ?",
                    (user_id,),
                )
                await connection.execute(
                    "DELETE FROM relationships WHERE user_id = ?",
                    (user_id,),
                )
                # A compressed channel summary has no per-speaker ownership
                # map.  Clearing the small cache globally is the only honest
                # way to guarantee that a forgotten user's words are not left
                # embedded in one of those summaries.
                await connection.execute("DELETE FROM channel_summaries")
                for table, column in (
                    ("user_profiles", "user_id"),
                    ("open_loops", "user_id"),
                    ("summary_sessions", "user_id"),
                    ("usage_metrics", "user_id"),
                    ("background_jobs", "user_id"),
                    ("bot_experiences", "source_user_id"),
                    ("safety_events", "user_id"),
                    ("discord_admins", "user_id"),
                ):
                    await connection.execute(f"DELETE FROM {table} WHERE {column} = ?", (user_id,))
                await connection.execute(
                    "DELETE FROM feedback_events WHERE user_id = ? OR origin_user_id = ?",
                    (user_id, user_id),
                )
                await connection.execute(
                    """DELETE FROM audit_log
                       WHERE actor = ? OR target = ? OR details_json LIKE ?""",
                    (f"discord:{user_id}", user_id, f"%{user_id}%"),
                )
                # Flow side-channels: proactive_log flow rows carry
                # multi-user-derived hook text with no per-user attribution;
                # conservative whole-guild flow-row deletion is privacy-correct.
                # safety_events from the flow path are written with user_id=""
                # (no single actor), so purge the unattributable rows per guild.
                for gid in affected_guilds:
                    await connection.execute(
                        "DELETE FROM proactive_log WHERE guild_id = ? AND reason LIKE 'flow:%'",
                        (gid,),
                    )
                    await connection.execute(
                        "DELETE FROM safety_events WHERE guild_id = ? AND (user_id IS NULL OR user_id = '')",
                        (gid,),
                    )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

    @staticmethod
    def expiry_after(days: int | float | None) -> str | None:
        if not days or days <= 0:
            return None
        return (utcnow() + timedelta(days=float(days))).isoformat()
