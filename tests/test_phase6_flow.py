"""Phase B — 心流：冷场主动开话题测试。

覆盖 B2 资格判定、B3 共享预算（含 flow 冷却 / 通用冷却 / 槽位消耗）、
B4 话题生成（hook 验证 / JSON 鲁棒 / 不可信包裹 / usage 记账 /
recent-topics 注入 / max_tokens 封顶 / 80 字上限）、
B5 发送（无持久化 / 即时拒绝）、B1 循环弹性。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.discord_bot import (
    _FLOW_CANDIDATE_FETCH,
    _FLOW_CONTEXT_MESSAGES,
    _FLOW_IDLE_MINUTES,
    _FLOW_MAX_TOKENS,
    _FLOW_MIN_USER_MESSAGES,
    _FLOW_WINDOW_HOURS,
    MoboBot,
)
from app.llm import ModelResult

# ── 常量 ────────────────────────────────────────────────────────────────────

GUILD_ID = "333333333333333"
CHANNEL_ID = "444444444444444"
USER_ID = "111111111111111"


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _model_result(text: str) -> ModelResult:
    return ModelResult(text, 50, 20, 100.0, "fake", "fake-utility")


def _valid_llm_result(hook: str, text: str) -> ModelResult:
    return _model_result(json.dumps({"hook": hook, "text": text}, ensure_ascii=False))


async def _base_config(state) -> None:
    await state.runtime.update(
        {
            "proactive_global_enabled": True,
            "flow_enabled": True,
            "flow_probability": 1.0,
            "save_raw_messages": True,
            "raw_history_days": 30,
            "timezone": "UTC",
            "proactive_quiet_start": "03:00",
            "proactive_quiet_end": "04:00",
            "proactive_cooldown_minutes": 45,
            "proactive_daily_limit": 6,
            "mood_enabled": False,
            "mood_baseline_social_budget": 0.7,
            "humanization_enabled": False,
        },
        actor="test",
    )


async def _enable_channel(state, guild_id=GUILD_ID, channel_id=CHANNEL_ID) -> None:
    await state.channels.set(
        guild_id, channel_id, "general", listen_enabled=True, proactive_enabled=True
    )


async def _seed_user_messages(
    state, count: int, *, newest_age_minutes: float = 30.0,
    guild_id: str = GUILD_ID, channel_id: str = CHANNEL_ID,
) -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    for i in range(count):
        age = newest_age_minutes + (count - 1 - i) * 5
        created = now - timedelta(minutes=age)
        await state.memories.save_message(
            guild_id,
            channel_id,
            "user",
            f"消息 {i}: 最近在聊编程和游戏",
            retention_days=30,
            user_id=f"100000000000{i:04d}",
            username=f"用户{i}",
        )
        # Backdate the created_at to simulate realistic timestamps
        await state.database.execute(
            "UPDATE messages SET created_at = ? WHERE content = ?",
            (_iso(created), f"消息 {i}: 最近在聊编程和游戏"),
        )


async def _seed_assistant_messages(state, count: int) -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    for i in range(count):
        created = now - timedelta(minutes=30 + i * 5)
        await state.memories.save_message(
            GUILD_ID,
            CHANNEL_ID,
            "assistant",
            f"bot 回复 {i}",
            retention_days=30,
            username="mobo",
        )
        await state.database.execute(
            "UPDATE messages SET created_at = ? WHERE content = ?",
            (_iso(created), f"bot 回复 {i}"),
        )


def _fake_channel(channel_id: int = 444444444444444) -> MagicMock:
    channel = MagicMock()
    channel.id = channel_id
    sent_messages = []

    async def fake_send(content, **kwargs):
        msg = SimpleNamespace(
            id=9000000 + len(sent_messages),
            content=content,
            channel=channel,
        )
        sent_messages.append(msg)
        return msg

    channel.send = AsyncMock(side_effect=fake_send)
    channel._sent = sent_messages
    return channel


def _make_bot(state) -> MoboBot:
    bot = MoboBot(state)
    bot._connection.user = SimpleNamespace(id=999999999999999, __str__=lambda s: "mobo")
    return bot


# ═══════════════════════════════════════════════════════════════════════════════
#  B2 — 资格判定
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowEligibility:
    @pytest.mark.asyncio
    async def test_global_off_returns_early(self, state):
        await _base_config(state)
        await state.runtime.update({"proactive_global_enabled": False}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_flow_enabled_off_returns_early(self, state):
        await _base_config(state)
        await state.runtime.update({"flow_enabled": False}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_quiet_hours_skip(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        # Set quiet hours to cover our mocked time (12:30 UTC)
        await state.runtime.update(
            {"proactive_quiet_start": "11:00", "proactive_quiet_end": "13:00"},
            actor="test",
        )

        mock_now = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        with patch("app.discord_bot.utcnow", return_value=mock_now):
            await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_raw_messages_off_skips_with_log(self, state, caplog):
        await _base_config(state)
        await state.runtime.update({"save_raw_messages": False}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        with caplog.at_level(logging.INFO, logger="mobo.discord"):
            await bot._flow_tick()

        assert any("save_raw_messages" in r.message for r in caplog.records)
        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_few_messages_skip(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 5)  # Below _FLOW_MIN_USER_MESSAGES
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(
            GUILD_ID, CHANNEL_ID, config, datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        )
        assert not eligible

    @pytest.mark.asyncio
    async def test_not_idle_long_enough_skip(self, state):
        await _base_config(state)
        await _enable_channel(state)
        # Seed messages but newest is only 5 min ago (< _FLOW_IDLE_MINUTES)
        await _seed_user_messages(state, 12, newest_age_minutes=5.0)
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(
            GUILD_ID, CHANNEL_ID, config, datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        )
        assert not eligible

    @pytest.mark.asyncio
    async def test_exactly_20_min_idle_skip(self, state):
        """Fix 5: exactly 20 minutes idle → REJECTED (strictly > required)."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # Seed messages with newest exactly 20 minutes ago
        await _seed_user_messages(state, 12, newest_age_minutes=20.0)
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(GUILD_ID, CHANNEL_ID, config, now)
        assert not eligible

    @pytest.mark.asyncio
    async def test_20_min_plus_epsilon_eligible(self, state):
        """Fix 5: 20 min + epsilon → eligible (strictly > 20)."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # Seed messages with newest 20.5 minutes ago (> 20, should pass)
        await _seed_user_messages(state, 12, newest_age_minutes=20.5)
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(GUILD_ID, CHANNEL_ID, config, now)
        assert eligible

    @pytest.mark.asyncio
    async def test_mood_below_budget_skip(self, state):
        await _base_config(state)
        await state.runtime.update({"mood_enabled": True}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)
        config = await state.runtime.all()

        # Mock mood to return low social_budget
        bot.state.mood.current = AsyncMock(
            return_value={"social_budget": 0.1, "valence": 0.0, "energy": 0.5}
        )

        eligible = await bot._flow_channel_eligible(
            GUILD_ID, CHANNEL_ID, config, datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        )
        assert not eligible

    @pytest.mark.asyncio
    async def test_soft_budget_reached_skip(self, state):
        await _base_config(state)
        await state.runtime.update({"daily_soft_token_budget": 100}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        mock_now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # Seed usage to exceed budget (within same local day)
        await state.usage.record(
            "chat",
            input_tokens=80,
            output_tokens=30,
            created_at=mock_now - timedelta(hours=1),
        )

        with patch("app.discord_bot.utcnow", return_value=mock_now):
            await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_eligible_channel_passes(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(
            GUILD_ID, CHANNEL_ID, config, datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        )
        assert eligible

    @pytest.mark.asyncio
    async def test_expired_history_skip(self, state):
        """Fix 12: messages with expires_at in the past are excluded."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # Seed 12 user messages but set expires_at to the past
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            expired = (now - timedelta(minutes=1)).isoformat()
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}",
                 f"消息 {i}: 聊天", _iso(created), expired),
            )
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(GUILD_ID, CHANNEL_ID, config, now)
        assert not eligible

    @pytest.mark.asyncio
    async def test_out_of_window_history_skip(self, state):
        """Fix 12: messages older than _FLOW_WINDOW_HOURS are excluded."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # Seed messages all older than _FLOW_WINDOW_HOURS
        for i in range(12):
            created = now - timedelta(hours=_FLOW_WINDOW_HOURS + 1 + i)
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}",
                 f"消息 {i}: 聊天", _iso(created)),
            )
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(GUILD_ID, CHANNEL_ID, config, now)
        assert not eligible

    @pytest.mark.asyncio
    async def test_disabled_listen_proactive_skip(self, state):
        """Fix 12: channels without listen+proactive are skipped at tick level."""
        await _base_config(state)
        await _seed_user_messages(state, 12)
        # Set channel with listen=False, proactive=False
        await state.channels.set(
            GUILD_ID, CHANNEL_ID, "general", listen_enabled=False, proactive_enabled=False
        )
        bot = _make_bot(state)

        await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  B2 — role 过滤
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoleFiltering:
    @pytest.mark.asyncio
    async def test_assistant_messages_do_not_count(self, state):
        await _base_config(state)
        await _enable_channel(state)
        # Seed 5 user messages + 10 assistant messages
        await _seed_user_messages(state, 5, newest_age_minutes=30.0)
        await _seed_assistant_messages(state, 10)
        bot = _make_bot(state)
        config = await state.runtime.all()

        eligible = await bot._flow_channel_eligible(
            GUILD_ID, CHANNEL_ID, config, datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        )
        assert not eligible  # Only 5 user messages < _FLOW_MIN_USER_MESSAGES


# ═══════════════════════════════════════════════════════════════════════════════
#  B3 — 共享预算（flow 冷却 / 通用冷却 / 槽位消耗）
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowReservation:
    @pytest.mark.asyncio
    async def test_flow_counts_toward_proactive_daily_limit(self, state):
        """B3 显式契约: flow 计入同一个 proactive_daily_limit 池。"""
        await _base_config(state)
        await state.runtime.update({"proactive_daily_limit": 1}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Reserve one flow slot (should succeed)
        denied = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:测试",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=1, daily_limit=1, flow=True,
        )
        assert denied is None

        # Now try a normal proactive slot (past generic cooldown, but daily limit hit)
        denied2 = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "自然参与",
            now_utc=now + timedelta(minutes=2), utc_start=utc_start,
            cooldown_minutes=1, daily_limit=1,
        )
        assert denied2 == "今日额度已用完"

    @pytest.mark.asyncio
    async def test_flow_refused_when_daily_limit_consumed(self, state):
        """B3: 当日限已用完，flow 同样被拒。"""
        await _base_config(state)
        await state.runtime.update({"proactive_daily_limit": 1}, actor="test")
        await _enable_channel(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Consume the daily limit with a normal proactive entry
        await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "自然参与",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=1, daily_limit=1,
        )

        # Flow should be denied by daily limit (past generic cooldown)
        denied = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:测试",
            now_utc=now + timedelta(minutes=2), utc_start=utc_start,
            cooldown_minutes=1, daily_limit=1, flow=True,
        )
        assert denied == "今日额度已用完"

    @pytest.mark.asyncio
    async def test_flow_cooldown_120_minutes(self, state):
        """B3a: 第二条 flow 在 120 分钟内被 LIKE 'flow:%' 检查拒绝。"""
        await _base_config(state)
        await _enable_channel(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # First flow succeeds
        denied1 = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题A",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=5, daily_limit=10, flow=True,
        )
        assert denied1 is None

        # Second flow 60 minutes later → blocked by flow cooldown (120min)
        denied2 = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题B",
            now_utc=now + timedelta(minutes=60), utc_start=utc_start,
            cooldown_minutes=5, daily_limit=10, flow=True,
        )
        assert denied2 == "心流冷却中"

        # Third flow 121 minutes later → passes flow cooldown
        denied3 = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题C",
            now_utc=now + timedelta(minutes=121), utc_start=utc_start,
            cooldown_minutes=5, daily_limit=10, flow=True,
        )
        assert denied3 is None

    @pytest.mark.asyncio
    async def test_flow_passes_generic_cooldown(self, state):
        """B3b: flow 自身也必须通过通用冷却检查。"""
        await _base_config(state)
        await _enable_channel(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # First proactive entry
        await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "自然参与",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=45, daily_limit=10,
        )

        # Flow 10 minutes later → blocked by generic cooldown
        denied = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题",
            now_utc=now + timedelta(minutes=10), utc_start=utc_start,
            cooldown_minutes=45, daily_limit=10, flow=True,
        )
        assert denied == "频道冷却中"

    @pytest.mark.asyncio
    async def test_flow_restarts_generic_cooldown(self, state):
        """B3a: flow 行同样重启通用 proactive_cooldown_minutes 窗口。"""
        await _base_config(state)
        await _enable_channel(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Flow succeeds
        await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=45, daily_limit=10, flow=True,
        )

        # Normal proactive 10 minutes later → blocked by generic cooldown
        denied = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "自然参与",
            now_utc=now + timedelta(minutes=10), utc_start=utc_start,
            cooldown_minutes=45, daily_limit=10,
        )
        assert denied == "频道冷却中"

    @pytest.mark.asyncio
    async def test_slot_consumed_even_on_discard(self, state):
        """B3c: 槽位在生成前保留——输出被丢弃仍消耗槽位。"""
        await _base_config(state)
        await _enable_channel(state)

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Reserve slot (simulating before-generation reservation)
        denied = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:待定",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=1, daily_limit=10, flow=True,
        )
        assert denied is None

        # Slot is consumed — another flow within 120min should be refused
        # Must be past generic cooldown but within flow cooldown
        denied2 = await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:话题B",
            now_utc=now + timedelta(minutes=2), utc_start=utc_start,
            cooldown_minutes=1, daily_limit=10, flow=True,
        )
        assert denied2 == "心流冷却中"


# ═══════════════════════════════════════════════════════════════════════════════
#  B4 — 话题生成（hook 验证 / JSON 鲁棒 / max_tokens 封顶 / 80 字上限）
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowTopicGeneration:
    @pytest.mark.asyncio
    async def test_fabricated_hook_discards_no_send(self, state):
        """伪造 hook（不在上下文中）→ 不发送，槽位已消耗。"""
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)

        # LLM returns a hook that is NOT in the context
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("这是一个完全不存在的伪造hook", "大家好，今天天气不错")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="大家好，今天天气不错")
        )

        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Reserve slot first (as _flow_tick does)
        await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:待定",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=5, daily_limit=10, flow=True,
        )

        # Call _generate_flow_topic
        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is None  # Discarded

    @pytest.mark.asyncio
    async def test_valid_hook_sends(self, state):
        """有效 hook（上下文中的逐字子串）→ 成功返回。"""
        await _base_config(state)
        await _enable_channel(state)
        # Seed messages with known content
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 今天在讨论Python编程技巧"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)

        # LLM returns a hook that IS in the context
        hook = "今天在讨论Python编程技巧"
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result(hook, "Python 确实是很有趣的语言，大家最近在学什么？")
        )

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is not None
        assert result[0] == hook
        assert "Python" in result[1]

    @pytest.mark.asyncio
    async def test_malformed_json_discards(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        state.llm.complete = AsyncMock(return_value=_model_result("这不是JSON"))

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_keys_discards(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Missing "text" key
        state.llm.complete = AsyncMock(
            return_value=_model_result(json.dumps({"hook": "some hook"}))
        )

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_discards(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        state.llm.complete = AsyncMock(
            return_value=_model_result(json.dumps({"hook": "some hook", "text": ""}))
        )

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is None

    @pytest.mark.asyncio
    async def test_code_fenced_json_parsed(self, state):
        """LLM 返回带 code fence 的 JSON 也能解析。"""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 聊聊最近看的电影"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )
        bot = _make_bot(state)

        hook = "聊聊最近看的电影"
        fenced = f"```json\n{json.dumps({'hook': hook, 'text': '有人最近看了好电影吗？'})}\n```"
        state.llm.complete = AsyncMock(return_value=_model_result(fenced))

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is not None
        assert result[0] == hook

    @pytest.mark.asyncio
    async def test_text_over_80_chars_discards(self, state):
        """Fix 4: 81-char text → discarded."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论编程"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )
        bot = _make_bot(state)

        text_81 = "x" * 81
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论编程", text_81)
        )

        result = await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)
        assert result is None

    @pytest.mark.asyncio
    async def test_max_tokens_capped(self, state):
        """Fix 3: llm.complete receives config with llm_max_tokens capped at _FLOW_MAX_TOKENS."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论天气"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )
        bot = _make_bot(state)

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论天气", "明天天气怎么样？")
        )

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        state.llm.complete.assert_called_once()
        config_arg = state.llm.complete.call_args[0][0]
        assert config_arg["llm_max_tokens"] == _FLOW_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_max_tokens_capped_even_when_config_lower(self, state):
        """Fix 3: if config already lower than _FLOW_MAX_TOKENS, use the config value."""
        await _base_config(state)
        await state.runtime.update({"llm_max_tokens": 100}, actor="test")
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论天气"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )
        bot = _make_bot(state)

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论天气", "明天天气怎么样？")
        )

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        state.llm.complete.assert_called_once()
        config_arg = state.llm.complete.call_args[0][0]
        assert config_arg["llm_max_tokens"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
#  B4 — recent-topics 注入（anti-repetition, fix 2）
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecentTopicsInjection:
    @pytest.mark.asyncio
    async def test_recent_flow_topics_in_prompt(self, state):
        """Fix 2: recent flow topics from proactive_log are injected with untrusted wrapper."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论技术"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Seed recent flow topics in proactive_log
        for idx, topic in enumerate(["Python编程", "电影推荐", "游戏攻略"]):
            await state.database.execute(
                """INSERT INTO proactive_log (guild_id, channel_id, reason, created_at)
                   VALUES (?, ?, ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"flow:话题：{topic}",
                 (now - timedelta(hours=idx)).isoformat()),
            )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论技术", "技术话题不错")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Untrusted-data wrapper applied to recent flow topics
        assert "不可信的较近心流话题参考" in prompt
        assert "你最近在该频道开过的话题（不要重复）" in prompt
        assert "Python编程" in prompt

    @pytest.mark.asyncio
    async def test_no_recent_flows_no_injection(self, state):
        """Fix 2: no recent flows → no injection label in prompt."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论技术"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论技术", "技术话题不错")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "你最近在该频道开过的话题（不要重复）" not in prompt

    @pytest.mark.asyncio
    async def test_recent_topics_from_another_channel_not_in_prompt(self, state):
        """F1: recent flow topics from ANOTHER channel in the same guild must NOT appear."""
        await _base_config(state)
        await _enable_channel(state)
        other_channel_id = "999999999999999"
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages for the target channel
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论技术"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Seed flow topics in ANOTHER channel (same guild)
        for idx, topic in enumerate(["电影推荐", "游戏攻略"]):
            await state.database.execute(
                """INSERT INTO proactive_log (guild_id, channel_id, reason, created_at)
                   VALUES (?, ?, ?, ?)""",
                (GUILD_ID, other_channel_id, f"flow:话题：{topic}",
                 (now - timedelta(hours=idx)).isoformat()),
            )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论技术", "技术话题不错")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Topics from another channel must NOT leak into this channel's prompt
        assert "电影推荐" not in prompt
        assert "游戏攻略" not in prompt


# ═══════════════════════════════════════════════════════════════════════════════
#  B4 — usage 记账
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowUsage:
    @pytest.mark.asyncio
    async def test_usage_recorded_after_generation(self, state):
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12, newest_age_minutes=30.0)
        bot = _make_bot(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("聊编程", "大家最近在用什么框架？")
        )

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        usage_rows = await state.database.fetchall(
            "SELECT * FROM usage_metrics WHERE kind = 'flow_topic'"
        )
        assert len(usage_rows) == 1
        assert int(usage_rows[0]["input_tokens"]) == 50
        assert int(usage_rows[0]["output_tokens"]) == 20


# ═══════════════════════════════════════════════════════════════════════════════
#  B4 — 不可信包裹 + 偏好话题
# ═══════════════════════════════════════════════════════════════════════════════


class TestUntrustedWrapping:
    @pytest.mark.asyncio
    async def test_prompt_contains_untrusted_markers(self, state):
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论技术"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论技术", "技术话题不错")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "不可信" in prompt

    @pytest.mark.asyncio
    async def test_per_source_untrusted_wrapping(self, state):
        """Fix 12: each context source has its own untrusted wrapper in the prompt."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages for history context
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论技术"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Seed positive preferences
        await state.preferences.upsert("编程", ["python", "代码"], 0.8, locked=False)

        # Seed channel summary
        await state.database.execute(
            """INSERT INTO channel_summaries
               (guild_id, channel_id, through_message_id, summary, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (GUILD_ID, CHANNEL_ID, 0, "之前聊过天气和编程", now.isoformat()),
        )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论技术", "技术话题不错")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Summary has its own wrapper
        assert "不可信的较早频道对话摘要" in prompt
        # History has its own wrapper
        assert "不可信的较近频道消息" in prompt
        # Preferences have their own wrapper
        assert "不可信的偏好话题参考" in prompt

    @pytest.mark.asyncio
    async def test_only_positive_prefs_in_prompt(self, state):
        """Fix 12: only positive-preference topics appear in the prompt; negative/avoid never do."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论各种话题"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Seed positive AND negative preferences
        await state.preferences.upsert("编程", ["python"], 0.8, locked=False)
        await state.preferences.upsert("政治", ["选举"], -0.5, locked=False)

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论各种话题", "不错的开场白")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Positive preference appears
        assert "编程" in prompt
        # Negative preference does NOT appear
        assert "政治" not in prompt


# ═══════════════════════════════════════════════════════════════════════════════
#  B5 — 无持久化（fix 2: flow output 不再写入 messages 表）
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowPersistence:
    @pytest.mark.asyncio
    async def test_no_message_persistence_on_flow(self, state):
        """Fix 2 (SEC-2): flow 发送后消息不写入 messages 表。"""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Seed user messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论编程语言"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        hook = "讨论编程语言"
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result(hook, "大家最近在用什么编程语言？")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="大家最近在用什么编程语言？")
        )
        bot.get_channel = MagicMock(return_value=channel)

        # Count assistant messages before
        before_count = int(
            await state.database.scalar(
                "SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND guild_id = ?",
                (GUILD_ID,),
            )
        )

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # Count assistant messages after — should be unchanged
        after_count = int(
            await state.database.scalar(
                "SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND guild_id = ?",
                (GUILD_ID,),
            )
        )
        assert after_count == before_count

        # Proactive log should have a flow entry
        flow_rows = await state.database.fetchall(
            "SELECT * FROM proactive_log WHERE reason LIKE 'flow:%'"
        )
        assert len(flow_rows) == 1

    @pytest.mark.asyncio
    async def test_no_remember_bot_message_on_flow(self, state):
        """Fix 2: _remember_bot_message is NOT called on the flow path."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed user messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论音乐"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论音乐", "大家喜欢什么音乐？")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="大家喜欢什么音乐？")
        )
        bot.get_channel = MagicMock(return_value=channel)

        original_remember = bot._remember_bot_message
        remember_calls: list[tuple] = []
        bot._remember_bot_message = lambda *args: remember_calls.append(args)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # _remember_bot_message should NOT have been called
        assert len(remember_calls) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  B5 — 安全顺序 + 即时拒绝（tick 级别, fix 8/11）
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowSafetyOrdering:
    @pytest.mark.asyncio
    async def test_safety_called_on_flow_output_tick_level(self, state):
        """Fix 11: tick 级别——flow 输出经过 safety.check_output。
        注：80 字上限保证 flow 输出恒为单碎片，safety 恰好调用一次。"""
        await _base_config(state)
        await state.runtime.update({"humanization_enabled": False}, actor="test")
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论天气和风景"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        hook = "讨论天气和风景"
        text_val = "今天天气真好，适合出去走走！"
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result(hook, text_val)
        )
        safety_calls: list[str] = []

        async def track_safety(text, **kwargs):
            safety_calls.append(text)
            return SimpleNamespace(allowed=True, text=text)

        state.safety.check_output = AsyncMock(side_effect=track_safety)
        bot.get_channel = MagicMock(return_value=channel)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # Safety should have been called exactly once (80-char cap → single fragment)
        assert len(safety_calls) == 1
        assert safety_calls[0] == text_val

    @pytest.mark.asyncio
    async def test_refusal_returns_immediately_no_send(self, state):
        """Fix 8: safety refusal → immediate return, no message sent, slot consumed."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论敏感话题"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论敏感话题", "这是敏感内容")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=False, text="这是敏感内容")
        )
        bot.get_channel = MagicMock(return_value=channel)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # No messages should have been sent
        assert len(channel._sent) == 0

        # Slot should be consumed (proactive_log entry exists with placeholder reason)
        flow_rows = await state.database.fetchall(
            "SELECT * FROM proactive_log WHERE reason LIKE 'flow:%'"
        )
        assert len(flow_rows) == 1
        # Reason should still be the placeholder (not updated since we returned early)
        assert flow_rows[0]["reason"] == "flow:待定"


# ═══════════════════════════════════════════════════════════════════════════════
#  B1 — 循环弹性
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowLoopResilience:
    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self, state, caplog):
        """tick 内异常被捕获并记录，不向上抛出。"""
        await _base_config(state)
        bot = _make_bot(state)

        # Make _flow_tick raise
        with patch.object(bot, "_flow_tick", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.ERROR, logger="mobo.discord"):
                # flow() calls _flow_tick inside try/except
                await bot.flow()

        assert any("flow tick failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_flow_error_handler_invocation(self, state, caplog):
        """Fix 10: flow_error 可以被直接调用且不崩溃，loop 对象保持可用。"""
        bot = _make_bot(state)
        exc = RuntimeError("test loop error")

        with caplog.at_level(logging.ERROR, logger="mobo.discord"):
            await bot.flow_error(exc)

        assert any("flow loop error" in r.message for r in caplog.records)
        # Loop error handler remains registered
        assert bot.flow._error is not None

    @pytest.mark.asyncio
    async def test_flow_error_handler_signature(self, state):
        """Fix 10: flow_error 签名正确——bound method 不含 self, 只有 exc。"""
        import inspect

        bot = _make_bot(state)
        sig = inspect.signature(bot.flow_error)
        params = list(sig.parameters.keys())
        # bound method: self is implicit, only exc remains
        assert len(params) == 1
        assert params[0] == "exc"

    @pytest.mark.asyncio
    async def test_closing_flag_skips_tick(self, state):
        """_closing 标志跳过 tick。"""
        await _base_config(state)
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)
        bot._closing = True

        await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  B2 — 概率节流
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowProbability:
    @pytest.mark.asyncio
    async def test_probability_zero_never_triggers(self, state):
        await _base_config(state)
        await state.runtime.update({"flow_probability": 0.0}, actor="test")
        await _enable_channel(state)
        await _seed_user_messages(state, 12)
        bot = _make_bot(state)

        with patch.object(bot, "_flow_channel_eligible", new_callable=AsyncMock, return_value=True):
            await bot._flow_tick()

        count = int(
            await state.database.scalar(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_probability_one_always_triggers_when_eligible(self, state):
        await _base_config(state)
        await state.runtime.update({"flow_probability": 1.0}, actor="test")
        await _enable_channel(state)
        mock_now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed eligible messages
        for i in range(12):
            created = mock_now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论游戏"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        hook = "讨论游戏"
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result(hook, "最近有什么好玩的游戏推荐？")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="最近有什么好玩的游戏推荐？")
        )
        bot.get_channel = MagicMock(return_value=channel)

        with patch("app.discord_bot.utcnow", return_value=mock_now):
            await bot._flow_tick()

        # Verify a flow log was created
        flow_rows = await state.database.fetchall(
            "SELECT * FROM proactive_log WHERE reason LIKE 'flow:%'"
        )
        assert len(flow_rows) == 1
        assert "讨论游戏" in flow_rows[0]["reason"]


# ═══════════════════════════════════════════════════════════════════════════════
#  B6 — 记账（reason 更新 / mood.observe 不调用）
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowBookkeeping:
    @pytest.mark.asyncio
    async def test_reason_updated_after_generation(self, state):
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        utc_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat()

        # Reserve with placeholder
        await state.proactive._reserve_channel_slot(
            GUILD_ID, CHANNEL_ID, "flow:待定",
            now_utc=now, utc_start=utc_start,
            cooldown_minutes=5, daily_limit=10, flow=True,
        )

        # Update reason
        await state.proactive.update_proactive_reason(
            GUILD_ID, CHANNEL_ID, "flow:待定", "flow:话题：今天讨论Python编程"
        )

        row = await state.database.fetchone(
            "SELECT reason FROM proactive_log WHERE guild_id = ? AND channel_id = ?",
            (GUILD_ID, CHANNEL_ID),
        )
        assert row is not None
        assert row["reason"] == "flow:话题：今天讨论Python编程"

    @pytest.mark.asyncio
    async def test_no_mood_observe_on_flow(self, state):
        """B6: flow 路径不调用 mood.observe。"""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论音乐"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论音乐", "大家喜欢听什么类型的音乐？")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="大家喜欢听什么类型的音乐？")
        )
        bot.get_channel = MagicMock(return_value=channel)
        state.mood.observe = AsyncMock()

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # mood.observe should NOT have been called
        state.mood.observe.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
#  B7 — 配置键
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowConfig:
    @pytest.mark.asyncio
    async def test_flow_enabled_default_false(self, state):
        config = await state.runtime.all()
        assert config["flow_enabled"] is False

    @pytest.mark.asyncio
    async def test_flow_probability_default(self, state):
        config = await state.runtime.all()
        assert float(config["flow_probability"]) == pytest.approx(0.15)


# ═══════════════════════════════════════════════════════════════════════════════
#  Coverage additions (fix 12)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowCoverage:
    @pytest.mark.asyncio
    async def test_multiple_eligible_channels_one_send(self, state):
        """Fix 12: 多个合格频道 → 每 tick 只发一次。"""
        await _base_config(state)
        ch1_id = "444444444444444"
        ch2_id = "555555555555555"
        await _enable_channel(state, channel_id=ch1_id)
        await _enable_channel(state, channel_id=ch2_id)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed eligible messages for both channels
        for ch_id in [ch1_id, ch2_id]:
            for i in range(12):
                created = now - timedelta(minutes=30 + i * 5)
                content = f"用户{i}: 讨论话题"
                await state.database.execute(
                    """INSERT INTO messages
                       (guild_id, channel_id, user_id, username, role, content, created_at)
                       VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                    (GUILD_ID, ch_id, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
                )

        bot = _make_bot(state)
        ch1 = _fake_channel(int(ch1_id))
        ch2 = _fake_channel(int(ch2_id))

        def get_channel(cid):
            return {int(ch1_id): ch1, int(ch2_id): ch2}.get(cid)

        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("讨论话题", "大家在聊什么？")
        )
        state.safety.check_output = AsyncMock(
            return_value=SimpleNamespace(allowed=True, text="大家在聊什么？")
        )
        bot.get_channel = MagicMock(side_effect=get_channel)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # Exactly ONE channel received a send
        total_sent = len(ch1._sent) + len(ch2._sent)
        assert total_sent >= 1
        # Only one channel should have messages
        assert (len(ch1._sent) >= 1) != (len(ch2._sent) >= 1)

    @pytest.mark.asyncio
    async def test_cross_guild_followup_not_in_context(self, state):
        """Fix 1: open_loops removed from flow context entirely (no source_channel column)."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论天气"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Insert open_loops directly (they should NOT appear in flow context)
        await state.database.execute(
            """INSERT INTO open_loops
               (guild_id, user_id, topic, public_safe, status, followup_after,
                expires_at, followup_count, created_at, updated_at)
               VALUES(?, ?, ?, 1, 'open', ?, ?, 0, ?, ?)""",
            (GUILD_ID, USER_ID, "上次聊到的电影推荐",
             (now - timedelta(minutes=5)).isoformat(),
             (now + timedelta(days=7)).isoformat(),
             now.isoformat(), now.isoformat()),
        )

        bot = _make_bot(state)
        captured_prompts: list[str] = []

        async def capture_complete(config, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return _valid_llm_result("讨论天气", "大家觉得明天天气怎么样？")

        state.llm.complete = AsyncMock(side_effect=capture_complete)

        await bot._generate_flow_topic(GUILD_ID, CHANNEL_ID, await state.runtime.all(), now)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Open loops topics should NOT appear in the prompt (removed entirely)
        assert "电影推荐" not in prompt

    @pytest.mark.asyncio
    async def test_discard_consumes_slot_tick_level(self, state):
        """Fix 11: tick 级别——输出被丢弃仍消耗槽位，后续 flow 被冷却拒绝。"""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed messages
        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论话题"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        bot = _make_bot(state)
        channel = _fake_channel()

        # LLM returns fabricated hook → discarded
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result("不存在的hook", "大家好")
        )
        bot.get_channel = MagicMock(return_value=channel)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # Slot consumed: proactive_log has "flow:待定" entry
        flow_rows = await state.database.fetchall(
            "SELECT * FROM proactive_log WHERE reason LIKE 'flow:%'"
        )
        assert len(flow_rows) == 1

        # No messages sent
        assert len(channel._sent) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  F2 — /忘记我 purge reaches flow side-channels
# ═══════════════════════════════════════════════════════════════════════════════


class TestPurgeFlowSideChannels:
    @pytest.mark.asyncio
    async def test_purge_deletes_flow_proactive_log_and_unattributed_safety(self, state):
        """F2: purge_user removes flow proactive_log rows and unattributed safety_events
        for the affected guild, while retaining attributed events of other users."""
        user_id = "111111111111111"
        other_user_id = "222222222222222"
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed user messages (needed for guild derivation)
        for i in range(3):
            created = now - timedelta(minutes=30 + i * 5)
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, user_id, "用户",
                 f"消息 {i}: 编程话题", _iso(created)),
            )

        # Seed a flow proactive_log row in the same guild
        await state.database.execute(
            """INSERT INTO proactive_log (guild_id, channel_id, reason, created_at)
               VALUES (?, ?, ?, ?)""",
            (GUILD_ID, CHANNEL_ID, "flow:话题：Python编程",
             (now - timedelta(hours=1)).isoformat()),
        )

        # Seed an unattributed safety_event (flow path: user_id="")
        await state.database.execute(
            """INSERT INTO safety_events
               (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
               VALUES(?, ?, ?, 'output', 'test', 'log', 'hash-flow', ?)""",
            (GUILD_ID, CHANNEL_ID, "", _iso(now)),
        )

        # Seed an attributed safety_event for ANOTHER user
        await state.database.execute(
            """INSERT INTO safety_events
               (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
               VALUES(?, ?, ?, 'input', 'test', 'block', 'hash-other', ?)""",
            (GUILD_ID, CHANNEL_ID, other_user_id, _iso(now)),
        )

        await state.database.purge_user(user_id)

        # User's messages gone
        assert (
            await state.database.scalar(
                "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
            )
            == 0
        )

        # Flow proactive_log row gone
        assert (
            await state.database.scalar(
                "SELECT COUNT(*) FROM proactive_log WHERE reason LIKE 'flow:%'"
            )
            == 0
        )

        # Unattributed safety_event gone
        assert (
            await state.database.scalar(
                "SELECT COUNT(*) FROM safety_events WHERE user_id = ''"
            )
            == 0
        )

        # Attributed safety_event for other user RETAINED
        assert (
            await state.database.scalar(
                "SELECT COUNT(*) FROM safety_events WHERE user_id = ?",
                (other_user_id,),
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_purge_flow_rows_only_in_affected_guild(self, state):
        """F2: flow proactive_log in a DIFFERENT guild is not touched by purge."""
        user_id = "111111111111111"
        other_guild = "999999999999999"
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Seed user messages in GUILD_ID (for guild derivation)
        await state.database.execute(
            """INSERT INTO messages
               (guild_id, channel_id, user_id, username, role, content, created_at)
               VALUES(?, ?, ?, ?, 'user', ?, ?)""",
            (GUILD_ID, CHANNEL_ID, user_id, "用户", "消息", _iso(now)),
        )

        # Flow proactive_log in ANOTHER guild
        await state.database.execute(
            """INSERT INTO proactive_log (guild_id, channel_id, reason, created_at)
               VALUES (?, ?, ?, ?)""",
            (other_guild, CHANNEL_ID, "flow:话题：不该被删",
             (now - timedelta(hours=1)).isoformat()),
        )

        await state.database.purge_user(user_id)

        # Flow row in the OTHER guild is retained
        assert (
            await state.database.scalar(
                "SELECT COUNT(*) FROM proactive_log WHERE guild_id = ? AND reason LIKE 'flow:%'",
                (other_guild,),
            )
            == 1
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  F3 — flow safety-check attribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlowSafetyAttribution:
    @pytest.mark.asyncio
    async def test_flow_safety_events_carry_empty_user_id(self, state):
        """F3: the flow path's safety_events rows carry user_id="" so that
        the F2 purge condition (user_id IS NULL OR user_id = '') matches."""
        await _base_config(state)
        await _enable_channel(state)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        for i in range(12):
            created = now - timedelta(minutes=30 + i * 5)
            content = f"用户{i}: 讨论天气和风景"
            await state.database.execute(
                """INSERT INTO messages
                   (guild_id, channel_id, user_id, username, role, content, created_at)
                   VALUES(?, ?, ?, ?, 'user', ?, ?)""",
                (GUILD_ID, CHANNEL_ID, f"100000000000{i:04d}", f"用户{i}", content, _iso(created)),
            )

        # Insert a safety rule that will trigger on output (log action = non-blocking)
        await state.database.execute(
            """INSERT INTO safety_rules
               (name, category, direction, pattern, match_type, action, enabled, priority, created_at, updated_at)
               VALUES ('test-log-rule', 'test', 'output', '风景', 'contains', 'log', 1, 100, ?, ?)""",
            (_iso(now), _iso(now)),
        )

        bot = _make_bot(state)
        channel = _fake_channel()

        hook = "讨论天气和风景"
        text_val = "今天天气真好，适合出去看风景！"
        state.llm.complete = AsyncMock(
            return_value=_valid_llm_result(hook, text_val)
        )
        bot.get_channel = MagicMock(return_value=channel)

        with patch("app.discord_bot.utcnow", return_value=now):
            await bot._flow_tick()

        # Safety event should have user_id="" (the flow path convention)
        safety_rows = await state.database.fetchall(
            "SELECT * FROM safety_events WHERE guild_id = ?", (GUILD_ID,)
        )
        assert len(safety_rows) >= 1
        for row in safety_rows:
            assert row["user_id"] == ""
