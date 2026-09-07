from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any

from app.database import Database, iso_now, utcnow

_MEMORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fact", re.compile(r"(?:我叫|我的名字是)\s*([^，。！？\n]{1,40})", re.I)),
    ("preference", re.compile(r"我(?:很|最|比较|特别)?喜欢\s*([^，。！？\n]{1,80})", re.I)),
    ("preference", re.compile(r"我(?:很|最|比较|特别)?讨厌\s*([^，。！？\n]{1,80})", re.I)),
    ("fact", re.compile(r"我的(?:工作|职业|专业|生日|家乡|宠物)是\s*([^，。！？\n]{1,80})", re.I)),
    (
        "preference",
        re.compile(r"I\s+(?:really\s+)?(?:like|love|hate|prefer)\s+([^.!?\n]{1,80})", re.I),
    ),
    ("fact", re.compile(r"my name is\s+([^,.!?\n]{1,40})", re.I)),
)

_SENSITIVE_MEMORY_PATTERN = re.compile(
    r"(?:密码|口令|token|api[ _-]?key|私钥|银行卡|信用卡|身份证|住址|家庭地址|"
    r"诊断|病史|政治身份|政治立场|党派)",
    re.I,
)
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_+#.-]{1,31}", re.I)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().lower().split())


def memory_terms(value: str, *, limit: int = 48) -> list[str]:
    """Create deterministic Chinese-friendly lookup terms without an embedding service."""
    normalized = normalize_text(value)
    terms: list[str] = []
    for word in _ASCII_WORD.findall(normalized):
        terms.append(word)
    for run in _CJK_RUN.findall(normalized):
        if len(run) <= 3:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
            terms.extend(run[index : index + 3] for index in range(len(run) - 2))
    return list(dict.fromkeys(term for term in terms if term))[:limit]


class MemoryService:
    def __init__(self, database: Database):
        self.database = database

    async def add(
        self,
        guild_id: str,
        user_id: str,
        content: str,
        *,
        kind: str = "explicit",
        confidence: float = 1.0,
        importance: float = 0.8,
        expires_at: str | None = None,
        source_message_id: int | None = None,
        reinforce_expires_at: str | None = None,
    ) -> int:
        content = " ".join(content.strip().split())[:500]
        if not content:
            raise ValueError("记忆内容不能为空")
        normalized = normalize_text(content)
        existing = await self.database.fetchone(
            """SELECT id FROM memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
                 AND lower(content) = lower(?)""",
            (guild_id, user_id, content),
        )
        if existing:
            if reinforce_expires_at is not None:
                # 再次印证：延长过期（候选期 → 完整保留期）；只延长不缩短，永不过期保持永不过期
                await self.database.execute(
                    """UPDATE memories SET confidence = MAX(confidence, ?),
                       importance = MAX(importance, ?), reinforcement_count = reinforcement_count + 1,
                       last_confirmed_at = ?, updated_at = ?,
                       expires_at = CASE WHEN expires_at IS NULL THEN NULL
                                         ELSE MAX(expires_at, ?) END
                       WHERE id = ?""",
                    (
                        confidence,
                        importance,
                        iso_now(),
                        iso_now(),
                        reinforce_expires_at,
                        existing["id"],
                    ),
                )
            else:
                await self.database.execute(
                    """UPDATE memories SET confidence = MAX(confidence, ?),
                       importance = MAX(importance, ?), reinforcement_count = reinforcement_count + 1,
                       last_confirmed_at = ?, updated_at = ? WHERE id = ?""",
                    (confidence, importance, iso_now(), iso_now(), existing["id"]),
                )
            await self._index_memory(int(existing["id"]), guild_id, user_id, content)
            return int(existing["id"])
        now = iso_now()
        memory_id = await self.database.execute(
            """INSERT INTO memories
               (guild_id, user_id, kind, content, source_message_id, confidence,
                importance, status, created_at, updated_at, expires_at,
                normalized_content, reinforcement_count, last_confirmed_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 1, ?)""",
            (
                guild_id,
                user_id,
                kind,
                content,
                source_message_id,
                confidence,
                importance,
                now,
                now,
                expires_at,
                normalized,
                now,
            ),
        )
        await self._index_memory(memory_id, guild_id, user_id, content)
        return memory_id

    async def _index_memory(
        self, memory_id: int, guild_id: str, user_id: str, content: str
    ) -> None:
        await self.database.execute("DELETE FROM memory_terms WHERE memory_id = ?", (memory_id,))
        terms = memory_terms(content)
        if terms:
            await self.database.executemany(
                """INSERT OR IGNORE INTO memory_terms(memory_id, guild_id, user_id, term)
                   VALUES(?, ?, ?, ?)""",
                [(memory_id, guild_id, user_id, term) for term in terms],
            )

    async def rebuild_legacy_index(self, *, batch_size: int = 500) -> int:
        """Backfill v1 memories once so upgrades retain relevant recall."""
        total = 0
        batch_size = max(1, min(5000, int(batch_size)))
        while True:
            rows = await self.database.fetchall(
                """SELECT id, guild_id, user_id, content FROM memories
                   WHERE normalized_content = '' ORDER BY id LIMIT ?""",
                (batch_size,),
            )
            if not rows:
                return total
            async with self.database.connect() as connection:
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    for row in rows:
                        memory_id = int(row["id"])
                        content = str(row["content"])
                        normalized = normalize_text(content) or "[empty]"
                        await connection.execute(
                            "UPDATE memories SET normalized_content = ? WHERE id = ?",
                            (normalized, memory_id),
                        )
                        terms = memory_terms(content)
                        if terms:
                            await connection.executemany(
                                """INSERT OR IGNORE INTO memory_terms
                                   (memory_id, guild_id, user_id, term)
                                   VALUES(?, ?, ?, ?)""",
                                [
                                    (memory_id, row["guild_id"], row["user_id"], term)
                                    for term in terms
                                ],
                            )
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise
            total += len(rows)

    async def touch_profile(self, user_id: str, display_name: str = "") -> None:
        now = iso_now()
        await self.database.execute(
            """INSERT INTO user_profiles
               (user_id, display_name, style_json, boundaries_json, first_seen_at,
                last_seen_at, interaction_count)
               VALUES(?, ?, '{}', '{}', ?, ?, 1)
               ON CONFLICT(user_id) DO UPDATE SET
                 display_name = CASE WHEN user_profiles.display_name = ''
                                          AND excluded.display_name != ''
                                     THEN excluded.display_name ELSE user_profiles.display_name END,
                 last_seen_at = excluded.last_seen_at,
                 interaction_count = user_profiles.interaction_count + 1""",
            (user_id, display_name[:120], now, now),
        )

    async def list_for_user(
        self, guild_id: str, user_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        return await self.database.fetchall(
            """SELECT id, kind, content, confidence, importance, created_at, expires_at
               FROM memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
                 AND (expires_at IS NULL OR expires_at > ?)
               ORDER BY importance DESC, id DESC LIMIT ?""",
            (guild_id, user_id, iso_now(), limit),
        )

    async def retrieve(
        self,
        guild_id: str,
        user_id: str,
        query: str,
        *,
        limit: int,
        kinds: tuple[str, ...] | None = None,
        min_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """检索相关记忆。

        Args:
            kinds: 若指定，只返回 kind 在此集合内的记忆。
            min_confidence: 若指定，只返回 confidence >= 该阈值的记忆。
        """
        params: list[Any] = [guild_id, user_id, iso_now()]
        kind_filter = ""
        confidence_filter = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_filter = f" AND kind IN ({placeholders})"
            params.extend(kinds)
        if min_confidence is not None:
            confidence_filter = " AND confidence >= ?"
            params.append(min_confidence)
        rows = await self.database.fetchall(
            f"""SELECT id, kind, content, confidence, importance, reinforcement_count,
                      created_at, updated_at, expires_at
               FROM memories WHERE guild_id = ? AND user_id = ? AND status = 'active'
                 AND (expires_at IS NULL OR expires_at > ?){kind_filter}{confidence_filter}
               ORDER BY importance DESC, updated_at DESC LIMIT 60""",
            params,
        )
        query_terms = set(memory_terms(query))
        if not rows:
            return []
        term_rows = await self.database.fetchall(
            """SELECT memory_id, term FROM memory_terms
               WHERE guild_id = ? AND user_id = ?""",
            (guild_id, user_id),
        )
        by_memory: dict[int, set[str]] = {}
        for term_row in term_rows:
            by_memory.setdefault(int(term_row["memory_id"]), set()).add(str(term_row["term"]))
        now = utcnow()
        targeted = kinds is not None or min_confidence is not None
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            stored_terms = by_memory.get(int(row["id"]), set())
            overlap = len(query_terms & stored_terms)
            relevance = overlap / max(1, min(len(query_terms), len(stored_terms)))
            updated = datetime.fromisoformat(row["updated_at"])
            age_days = max(0.0, (now - updated).total_seconds() / 86400)
            recency = math.pow(0.5, age_days / 90)
            reinforcement = min(1.0, math.log2(int(row["reinforcement_count"]) + 1) / 4)
            score = (
                relevance * 0.45
                + float(row["importance"]) * 0.20
                + float(row["confidence"]) * 0.15
                + recency * 0.10
                + reinforcement * 0.10
            )
            # 定向检索（指定了 kinds 或 min_confidence）时跳过相关性阈值，
            # 因为调用方已在 SQL 层面筛选了目标类型和置信度。
            if targeted or row["kind"] == "explicit" or relevance > 0 or score >= 0.42:
                row["retrieval_score"] = round(score, 4)
                ranked.append((score, row))
        selected = [
            row for _, row in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
        ]
        if selected:
            await self.database.executemany(
                "UPDATE memories SET last_recalled_at = ? WHERE id = ?",
                [(iso_now(), row["id"]) for row in selected],
            )
        return selected

    async def auto_extract(
        self,
        guild_id: str,
        user_id: str,
        content: str,
        *,
        confidence_threshold: float,
        expires_days: int,
        max_per_user: int,
        source_message_id: int | None = None,
        candidate_expiry_days: int = 0,
    ) -> list[int]:
        # Deliberately conservative: only explicit first-person claims and no sensitive facts.
        if _SENSITIVE_MEMORY_PATTERN.search(content):
            return []
        candidates: list[tuple[str, str, float]] = []
        for kind, pattern in _MEMORY_PATTERNS:
            match = pattern.search(content)
            if match:
                claim = match.group(0).strip(" ，。.!！?")
                candidates.append((kind, claim, 0.86))
        created: list[int] = []
        # 候选期：新记忆短期保留；再次出现时延长到完整保留期（捕获宽松，晋升严格）
        new_expiry = (
            Database.expiry_after(candidate_expiry_days)
            if candidate_expiry_days and candidate_expiry_days > 0
            else Database.expiry_after(expires_days)
        )
        reinforce_expiry = Database.expiry_after(expires_days)
        for kind, claim, confidence in candidates[:2]:
            if confidence < confidence_threshold:
                continue
            memory_id = await self.add(
                guild_id,
                user_id,
                claim,
                kind=kind,
                confidence=confidence,
                importance=0.55,
                expires_at=new_expiry,
                source_message_id=source_message_id,
                reinforce_expires_at=reinforce_expiry,
            )
            created.append(memory_id)
        await self._enforce_limit(guild_id, user_id, max_per_user)
        return created

    async def _enforce_limit(self, guild_id: str, user_id: str, limit: int) -> None:
        # Explicit memories are never selected by this automatic eviction query.
        rows = await self.database.fetchall(
            """SELECT id FROM memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
                 AND kind != 'explicit'
               ORDER BY importance DESC, updated_at DESC""",
            (guild_id, user_id),
        )
        overflow = rows[max(0, limit) :]
        if overflow:
            await self.database.executemany(
                "DELETE FROM memories WHERE id = ?",
                [(row["id"],) for row in overflow],
            )

    async def save_message(
        self,
        guild_id: str,
        channel_id: str,
        role: str,
        content: str,
        *,
        retention_days: int,
        user_id: str | None = None,
        username: str | None = None,
        discord_message_id: str | None = None,
        reply_to_message_id: str | None = None,
    ) -> int:
        async with self.database.connect() as connection:
            try:
                if discord_message_id is not None:
                    cursor = await connection.execute(
                        """SELECT id FROM messages
                           WHERE guild_id = ? AND channel_id = ? AND discord_message_id = ?""",
                        (guild_id, channel_id, discord_message_id),
                    )
                    existing = await cursor.fetchone()
                    if existing:
                        return int(existing["id"])
                cursor = await connection.execute(
                    """INSERT INTO messages
                       (guild_id, channel_id, user_id, username, role, content,
                        created_at, expires_at, discord_message_id, reply_to_message_id)
                       VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        guild_id,
                        channel_id,
                        user_id,
                        username,
                        role,
                        content[:8000],
                        iso_now(),
                        Database.expiry_after(retention_days),
                        discord_message_id,
                        reply_to_message_id,
                    ),
                )
                await connection.commit()
                return int(cursor.lastrowid)
            except Exception:
                await connection.rollback()
                raise

    async def channel_history(
        self, guild_id: str, channel_id: str, *, limit: int
    ) -> list[dict[str, str]]:
        rows = await self.database.fetchall(
            """SELECT role, content, user_id, username FROM messages
               WHERE guild_id = ? AND channel_id = ?
                 AND (expires_at IS NULL OR expires_at > ?)
               ORDER BY id DESC LIMIT ?""",
            (guild_id, channel_id, iso_now(), limit),
        )
        history: list[dict[str, str]] = []
        for row in reversed(rows):
            content = str(row["content"])
            if row["role"] == "user":
                speaker = str(row.get("username") or "频道成员")[:120]
                user_id = str(row.get("user_id") or "未知")
                content = f"[说话者：{speaker}；Discord ID：{user_id}]\n{content}"
            history.append({"role": row["role"], "content": content})
        return history

    async def channel_summary(self, guild_id: str, channel_id: str) -> dict[str, Any] | None:
        return await self.database.fetchone(
            """SELECT through_message_id, summary, updated_at FROM channel_summaries
               WHERE guild_id = ? AND channel_id = ?""",
            (guild_id, channel_id),
        )

    async def summary_batch(
        self,
        guild_id: str,
        channel_id: str,
        *,
        trigger: int,
        keep_recent: int,
    ) -> tuple[list[dict[str, Any]], int] | None:
        existing = await self.channel_summary(guild_id, channel_id)
        through_id = int(existing["through_message_id"]) if existing else 0
        count = int(
            await self.database.scalar(
                """SELECT COUNT(*) AS n FROM messages
                   WHERE guild_id = ? AND channel_id = ? AND id > ?""",
                (guild_id, channel_id, through_id),
            )
            or 0
        )
        if count < trigger:
            return None
        limit = max(1, count - keep_recent)
        rows = await self.database.fetchall(
            """SELECT id, role, content FROM messages
               WHERE guild_id = ? AND channel_id = ? AND id > ?
               ORDER BY id LIMIT ?""",
            (guild_id, channel_id, through_id, limit),
        )
        if not rows:
            return None
        return rows, int(rows[-1]["id"])

    async def store_channel_summary(
        self, guild_id: str, channel_id: str, through_message_id: int, summary: str
    ) -> None:
        await self.database.execute(
            """INSERT INTO channel_summaries
               (guild_id, channel_id, through_message_id, summary, updated_at)
               VALUES(?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, channel_id) DO UPDATE SET
                 through_message_id = excluded.through_message_id,
                 summary = excluded.summary,
                 updated_at = excluded.updated_at""",
            (guild_id, channel_id, through_message_id, summary[:8000], iso_now()),
        )

    async def clear_channel(self, guild_id: str, channel_id: str) -> int:
        rows = int(
            await self.database.scalar(
                "SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND channel_id = ?",
                (guild_id, channel_id),
            )
            or 0
        )
        await self.database.execute(
            "DELETE FROM messages WHERE guild_id = ? AND channel_id = ?",
            (guild_id, channel_id),
        )
        await self.database.execute(
            "DELETE FROM channel_summaries WHERE guild_id = ? AND channel_id = ?",
            (guild_id, channel_id),
        )
        return rows
