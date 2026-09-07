from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.cognition import MoodService, PreferenceService, RelationshipService, clamp
from app.database import Database, iso_now, utcnow
from app.gate import score_gate


class ChannelSettingsService:
    def __init__(self, database: Database):
        self.database = database

    async def get(self, guild_id: str, channel_id: str) -> dict[str, Any]:
        row = await self.database.fetchone(
            """SELECT guild_id, channel_id, channel_name, listen_enabled,
                      proactive_enabled, updated_at
               FROM channel_settings WHERE guild_id = ? AND channel_id = ?""",
            (guild_id, channel_id),
        )
        if row:
            row["listen_enabled"] = bool(row["listen_enabled"])
            row["proactive_enabled"] = bool(row["proactive_enabled"])
            return row
        return {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "channel_name": "",
            "listen_enabled": False,
            "proactive_enabled": False,
            "updated_at": None,
        }

    async def set(
        self,
        guild_id: str,
        channel_id: str,
        channel_name: str,
        *,
        listen_enabled: bool,
        proactive_enabled: bool,
    ) -> None:
        if proactive_enabled and not listen_enabled:
            raise ValueError("主动发言依赖频道上下文")
        if not listen_enabled and not proactive_enabled:
            await self.delete(guild_id, channel_id)
            return
        await self.database.execute(
            """INSERT INTO channel_settings
               (guild_id, channel_id, channel_name, listen_enabled,
                proactive_enabled, updated_at)
               VALUES(?, ?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, channel_id) DO UPDATE SET
                 channel_name = excluded.channel_name,
                 listen_enabled = excluded.listen_enabled,
                 proactive_enabled = excluded.proactive_enabled,
                 updated_at = excluded.updated_at""",
            (
                guild_id,
                channel_id,
                channel_name[:120],
                int(listen_enabled),
                int(proactive_enabled),
                iso_now(),
            ),
        )

    async def delete(self, guild_id: str, channel_id: str) -> bool:
        return bool(
            await self.database.execute(
                "DELETE FROM channel_settings WHERE guild_id = ? AND channel_id = ?",
                (guild_id, channel_id),
            )
        )

    async def list(self) -> list[dict[str, Any]]:
        rows = await self.database.fetchall(
            """SELECT c.*, g.name AS guild_name FROM channel_settings c
               LEFT JOIN guilds g ON g.guild_id = c.guild_id
               WHERE c.listen_enabled = 1
               ORDER BY g.name, c.channel_name"""
        )
        for row in rows:
            row["listen_enabled"] = bool(row["listen_enabled"])
            row["proactive_enabled"] = bool(row["proactive_enabled"])
        return rows


@dataclass(frozen=True)
class ProactiveDecision:
    should_speak: bool
    reason: str
    probability: float = 0.0
    score: int = 0


class ProactiveService:
    def __init__(
        self,
        database: Database,
        channels: ChannelSettingsService,
        relationships: RelationshipService,
        preferences: PreferenceService,
        mood: MoodService,
        *,
        random_value: Callable[[], float] = random.random,
    ):
        self.database = database
        self.channels = channels
        self.relationships = relationships
        self.preferences = preferences
        self.mood = mood
        self.random_value = random_value

    @staticmethod
    def _local_now(config: dict[str, Any], now: datetime | None = None) -> datetime:
        try:
            zone = ZoneInfo(str(config["timezone"]))
        except ZoneInfoNotFoundError:
            zone = UTC
        return (now or utcnow()).astimezone(zone)

    @staticmethod
    def _in_quiet_hours(now_local: datetime, start_value: str, end_value: str) -> bool:
        start = time.fromisoformat(start_value)
        end = time.fromisoformat(end_value)
        current = now_local.time().replace(tzinfo=None)
        if start == end:
            return False
        if start < end:
            return start <= current < end
        return current >= start or current < end

    async def _reserve_channel_slot(
        self,
        guild_id: str,
        channel_id: str,
        reason: str,
        *,
        now_utc: datetime,
        utc_start: str,
        cooldown_minutes: int,
        daily_limit: int,
        kind_cooldown_minutes: int | None = None,
    ) -> str | None:
        """Atomically check and consume a channel slot before model generation.

        When *kind_cooldown_minutes* is set, an additional kind-specific cooldown
        is enforced: the last ``reason LIKE 'flow:%'`` row must be older than
        *kind_cooldown_minutes*.  A flow row still restarts the generic proactive
        cooldown window.  Existing callers that do not pass this kwarg are
        unaffected.
        """

        async with self.database.connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """SELECT created_at FROM proactive_log
                       WHERE guild_id = ? AND channel_id = ?
                       ORDER BY id DESC LIMIT 1""",
                    (guild_id, channel_id),
                )
                last_row = await cursor.fetchone()
                if last_row is not None:
                    elapsed = (
                        now_utc - datetime.fromisoformat(str(last_row["created_at"]))
                    ).total_seconds() / 60
                    if elapsed < cooldown_minutes:
                        await connection.rollback()
                        return "频道冷却中"
                # Kind-specific cooldown: last flow:% row must be old enough.
                if kind_cooldown_minutes is not None and kind_cooldown_minutes > 0:
                    cursor = await connection.execute(
                        """SELECT created_at FROM proactive_log
                           WHERE guild_id = ? AND channel_id = ?
                             AND reason LIKE 'flow:%'
                           ORDER BY id DESC LIMIT 1""",
                        (guild_id, channel_id),
                    )
                    last_flow_row = await cursor.fetchone()
                    if last_flow_row is not None:
                        flow_elapsed = (
                            now_utc - datetime.fromisoformat(str(last_flow_row["created_at"]))
                        ).total_seconds() / 60
                        if flow_elapsed < kind_cooldown_minutes:
                            await connection.rollback()
                            return "心流冷却中"
                cursor = await connection.execute(
                    """SELECT COUNT(*) AS n FROM proactive_log
                       WHERE guild_id = ? AND channel_id = ? AND created_at >= ?""",
                    (guild_id, channel_id, utc_start),
                )
                count_row = await cursor.fetchone()
                if int(count_row["n"]) >= daily_limit:
                    await connection.rollback()
                    return "今日额度已用完"
                await connection.execute(
                    """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
                       VALUES(?, ?, ?, ?)""",
                    (guild_id, channel_id, reason, now_utc.isoformat()),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return None

    async def update_proactive_reason(
        self,
        guild_id: str,
        channel_id: str,
        old_reason_prefix: str,
        new_reason: str,
    ) -> None:
        """Update the reason of the most recent matching proactive_log row."""

        await self.database.execute(
            """UPDATE proactive_log SET reason = ?
               WHERE id = (
                   SELECT id FROM proactive_log
                   WHERE guild_id = ? AND channel_id = ?
                     AND reason LIKE ?
                   ORDER BY id DESC LIMIT 1
               )""",
            (new_reason, guild_id, channel_id, old_reason_prefix + "%"),
        )

    async def decide(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        content: str,
        config: dict[str, Any],
        *,
        now: datetime | None = None,
        pending_count: int = 0,
    ) -> ProactiveDecision:
        if not config["proactive_global_enabled"]:
            return ProactiveDecision(False, "全局主动发言已关闭")
        channel = await self.channels.get(guild_id, channel_id)
        if not channel["listen_enabled"] or not channel["proactive_enabled"]:
            return ProactiveDecision(False, "频道未启用")
        if len(content.strip()) < int(config["proactive_min_message_length"]):
            return ProactiveDecision(False, "消息太短")
        now_utc = now or utcnow()
        now_local = self._local_now(config, now_utc)
        local_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_start = local_start.astimezone(UTC).isoformat()
        if await self.soft_budget_reached(config, now=now_utc):
            return ProactiveDecision(False, "今日 Token 软预算已达上限")
        if self._in_quiet_hours(
            now_local,
            str(config["proactive_quiet_start"]),
            str(config["proactive_quiet_end"]),
        ):
            return ProactiveDecision(False, "安静时段")
        last = await self.database.scalar(
            """SELECT created_at FROM proactive_log
               WHERE guild_id = ? AND channel_id = ?
               ORDER BY id DESC LIMIT 1""",
            (guild_id, channel_id),
        )
        if last:
            elapsed = (now_utc - datetime.fromisoformat(last)).total_seconds() / 60
            if elapsed < int(config["proactive_cooldown_minutes"]):
                return ProactiveDecision(False, "频道冷却中")
        count = int(
            await self.database.scalar(
                """SELECT COUNT(*) AS n FROM proactive_log
                   WHERE guild_id = ? AND channel_id = ? AND created_at >= ?""",
                (guild_id, channel_id, utc_start),
            )
            or 0
        )
        if count >= int(config["proactive_daily_limit"]):
            return ProactiveDecision(False, "今日额度已用完")
        # Deciding whether to speak is observation, not feedback.  Learning here
        # would make mobo's preferences drift even when it ultimately stays quiet.
        # ── 闸门评分（积分 ≥ 阈值直接触发，否则落回概率路径）─────────
        bot_name = str(config.get("bot_name", "mobo"))
        # 近 5 分钟频道消息构成（原始消息保存关闭时无数据，惩罚自然为 0）
        window_start = (now_utc - timedelta(minutes=5)).isoformat()
        recent = await self.database.fetchone(
            """SELECT COUNT(*) AS total,
                      COALESCE(SUM(CASE WHEN role = 'bot' THEN 1 ELSE 0 END), 0) AS self_cnt
               FROM messages
               WHERE guild_id = ? AND channel_id = ? AND created_at >= ?""",
            (guild_id, channel_id, window_start),
        )
        gate_score, gate_detail = score_gate(
            content,
            bot_name,
            pending_count=pending_count,
            recent_self_messages=int(recent["self_cnt"]) if recent else 0,
            recent_total_messages=int(recent["total"]) if recent else 0,
        )
        gate_threshold = int(config.get("gate_threshold", 80))

        interest, topics = await self.preferences.interest_for(content, learn=False)
        if config["relationship_enabled"]:
            relationship = await self.relationships.get(
                guild_id, user_id, int(config["relationship_decay_days"])
            )
            familiarity = relationship.familiarity
            fatigue = relationship.fatigue
        else:
            familiarity = 0.0
            fatigue = 0.0
        social_budget = (
            float((await self.mood.current(config))["social_budget"])
            if config["mood_enabled"]
            else float(config["mood_baseline_social_budget"])
        )
        probability = float(config["proactive_base_probability"])
        probability *= max(0.1, 0.7 + interest)
        probability *= 0.75 + familiarity * 0.5
        probability *= 0.5 + social_budget
        probability *= 1.0 - fatigue * 0.7
        probability = clamp(probability, 0.0, 0.5)

        # 闸门路径：分数达标则跳过概率判定
        if gate_score >= gate_threshold:
            reason = f"闸门({gate_detail})"
            denied = await self._reserve_channel_slot(
                guild_id,
                channel_id,
                reason,
                now_utc=now_utc,
                utc_start=utc_start,
                cooldown_minutes=int(config["proactive_cooldown_minutes"]),
                daily_limit=int(config["proactive_daily_limit"]),
            )
            if denied is not None:
                return ProactiveDecision(False, denied, probability, gate_score)
            return ProactiveDecision(True, reason, probability, gate_score)

        if self.random_value() >= probability:
            return ProactiveDecision(False, "本次保持安静", probability, gate_score)
        reason = "偏好话题：" + "、".join(topics) if topics else "自然参与"
        denied = await self._reserve_channel_slot(
            guild_id,
            channel_id,
            reason,
            now_utc=now_utc,
            utc_start=utc_start,
            cooldown_minutes=int(config["proactive_cooldown_minutes"]),
            daily_limit=int(config["proactive_daily_limit"]),
        )
        if denied is not None:
            return ProactiveDecision(False, denied, probability, gate_score)
        return ProactiveDecision(True, reason, probability, gate_score)

    async def soft_budget_reached(
        self, config: dict[str, Any], *, now: datetime | None = None
    ) -> bool:
        """Return whether background/proactive work should pause for the local day."""

        soft_budget = int(config.get("daily_soft_token_budget", 0) or 0)
        if not soft_budget:
            return False
        now_utc = now or utcnow()
        now_local = self._local_now(config, now_utc)
        utc_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
        used_tokens = int(
            await self.database.scalar(
                """SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS n
                   FROM usage_metrics WHERE created_at >= ?""",
                (utc_start.isoformat(),),
            )
            or 0
        )
        return used_tokens >= soft_budget
