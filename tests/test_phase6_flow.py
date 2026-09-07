"""Phase 6B 心流测试：冷场判定、由头拒绝、共享预算池、同事务冷却、循环异常存活。"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from app.database import utcnow
from app.llm import ModelResult
from tests.test_discord_pipeline_v4 import FakeChannel, _ready_bot


def _model_result(text: str) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=10,
        output_tokens=5,
        latency_ms=1,
        provider="test",
        model="test-model",
    )


async def _insert_message(
    state,
    guild_id: str,
    channel_id: str,
    content: str,
    *,
    minutes_ago: float = 30,
    role: str = "user",
    user_id: str = "111111111111111",
    expired: bool = False,
) -> None:
    created = (utcnow() - timedelta(minutes=minutes_ago)).isoformat()
    expires_at = (utcnow() - timedelta(minutes=1)).isoformat() if expired else None
    await state.database.execute(
        """INSERT INTO messages
           (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
           VALUES(?, ?, ?, '小明', ?, ?, ?, ?)""",
        (guild_id, channel_id, user_id, role, content, created, expires_at),
    )


async def _seed_idle_channel(state, guild_id: str = "g1", channel_id: str = "444444") -> None:
    """12 条用户消息散布在近 3 小时内，最后一条距今 30 分钟：活跃过、已冷场。"""
    for index in range(12):
        minutes_ago = 30 + (11 - index) * 10
        await _insert_message(state, guild_id, channel_id, f"消息{index}", minutes_ago=minutes_ago)


async def _enable_flow(state) -> None:
    await state.runtime.update(
        {
            "flow_enabled": True,
            "proactive_global_enabled": True,
            "flow_probability": 1.0,
            # start==end 表示永不安静，让测试不受运行时刻影响
            "proactive_quiet_start": "00:00",
            "proactive_quiet_end": "00:00",
            "save_raw_messages": True,
        },
        actor="test",
    )
    await state.channels.set("g1", "444444", "general", listen_enabled=True, proactive_enabled=True)


def _stub_llm(state, text: str, *, captured: list | None = None) -> AsyncMock:
    async def complete(config, messages, **kwargs):
        if captured is not None:
            captured.append(messages)
        return _model_result(text)

    return AsyncMock(side_effect=complete)


class TestFlowEligibility:
    """B2：冷场判定（活跃窗口 + 冷场时长），无全表扫描。"""

    @pytest.mark.asyncio
    async def test_eligible_channel_detected(self, state):
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        prepared = await bot._flow_prepare_channel("g1", "444444")
        assert prepared is not None
        last_at, guild_id, channel_id, rows = prepared
        assert guild_id == "g1" and channel_id == "444444"
        assert len(rows) == 12

    @pytest.mark.asyncio
    async def test_not_idle_yet_rejected(self, state):
        for index in range(12):
            minutes_ago = 5 + (11 - index) * 10
            await _insert_message(state, "g1", "444444", f"消息{index}", minutes_ago=minutes_ago)
        bot, _bot_user, _channel = await _ready_bot(state)
        assert await bot._flow_prepare_channel("g1", "444444") is None

    @pytest.mark.asyncio
    async def test_silence_longer_than_window_rejected(self, state):
        for index in range(12):
            minutes_ago = 430 + (11 - index) * 10
            await _insert_message(state, "g1", "444444", f"消息{index}", minutes_ago=minutes_ago)
        bot, _bot_user, _channel = await _ready_bot(state)
        assert await bot._flow_prepare_channel("g1", "444444") is None

    @pytest.mark.asyncio
    async def test_too_few_messages_rejected(self, state):
        for index in range(5):
            minutes_ago = 30 + (4 - index) * 10
            await _insert_message(state, "g1", "444444", f"消息{index}", minutes_ago=minutes_ago)
        bot, _bot_user, _channel = await _ready_bot(state)
        assert await bot._flow_prepare_channel("g1", "444444") is None

    @pytest.mark.asyncio
    async def test_expired_messages_ignored(self, state):
        for index in range(12):
            minutes_ago = 30 + (11 - index) * 10
            await _insert_message(
                state, "g1", "444444", f"消息{index}", minutes_ago=minutes_ago, expired=True
            )
        bot, _bot_user, _channel = await _ready_bot(state)
        assert await bot._flow_prepare_channel("g1", "444444") is None


class TestFlowTickGates:
    """B1/B2：开关、安静时段、频道授权的全局门。"""

    @pytest.mark.asyncio
    async def test_flow_disabled_no_generation(self, state):
        await _seed_idle_channel(state)
        await state.channels.set(
            "g1", "444444", "general", listen_enabled=True, proactive_enabled=True
        )
        bot, _bot_user, _channel = await _ready_bot(state)
        stub = _stub_llm(state, "开场白")
        bot.state.llm.complete = stub
        await bot._flow_tick()
        stub.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_channel_without_proactive_skipped(self, state):
        await _seed_idle_channel(state)
        await state.channels.set(
            "g1", "444444", "general", listen_enabled=True, proactive_enabled=False
        )
        await state.runtime.update(
            {"flow_enabled": True, "proactive_global_enabled": True, "flow_probability": 1.0},
            actor="test",
        )
        bot, _bot_user, _channel = await _ready_bot(state)
        stub = _stub_llm(state, "开场白")
        bot.state.llm.complete = stub
        await bot._flow_tick()
        stub.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quiet_hours_block_tick(self, state, monkeypatch):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        from app.behavior import ProactiveService

        monkeypatch.setattr(
            ProactiveService, "_in_quiet_hours", staticmethod(lambda *args, **kwargs: True)
        )
        stub = _stub_llm(state, "开场白")
        bot.state.llm.complete = stub
        await bot._flow_tick()
        stub.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_loop_survives_exception(self, state, monkeypatch):
        bot, _bot_user, _channel = await _ready_bot(state)

        async def boom():
            raise RuntimeError("tick exploded")

        monkeypatch.setattr(bot, "_flow_tick_inner", boom)
        await bot._flow_tick()  # 不应抛出


class TestFlowTopicGeneration:
    """B4：由头契约、上下文不可信包裹、public_safe 过滤。"""

    @pytest.mark.asyncio
    async def test_missing_hook_discarded_without_slot(self, state):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        fake_channel = FakeChannel(444444)
        monkey_target = lambda channel_id: fake_channel  # noqa: E731
        bot.get_channel = monkey_target
        bot.state.llm.complete = _stub_llm(state, "NOHOOK")
        await bot._flow_tick()
        assert fake_channel.sent == []
        count = await state.database.scalar("SELECT COUNT(*) AS n FROM proactive_log")
        assert int(count) == 0

    @pytest.mark.asyncio
    async def test_context_untrusted_wrapping_and_public_safe_followups(self, state, monkeypatch):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        bot.get_channel = lambda channel_id: FakeChannel(444444)
        now_iso = utcnow().isoformat()
        await state.database.execute(
            """INSERT INTO open_loops
               (guild_id, user_id, topic, public_safe, status, followup_after,
                expires_at, followup_count, created_at, updated_at)
               VALUES('g1', '111111111111111', '周末爬山的事', 1, 'open', ?, NULL, 0, ?, ?)""",
            ((utcnow() - timedelta(hours=1)).isoformat(), now_iso, now_iso),
        )
        await state.database.execute(
            """INSERT INTO open_loops
               (guild_id, user_id, topic, public_safe, status, followup_after,
                expires_at, followup_count, created_at, updated_at)
               VALUES('g1', '111111111111111', '他的心理咨询预约', 0, 'open', ?, NULL, 0, ?, ?)""",
            ((utcnow() - timedelta(hours=1)).isoformat(), now_iso, now_iso),
        )
        captured: list = []
        bot.state.llm.complete = _stub_llm(state, "刚才你们说到摩天轮，我也想去", captured=captured)
        await bot._flow_tick()
        user_content = captured[0][-1]["content"]
        assert "不可信" in user_content
        assert "周末爬山的事" in user_content
        assert "心理咨询" not in user_content

    @pytest.mark.asyncio
    async def test_overlong_output_discarded(self, state):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        fake_channel = FakeChannel(444444)
        bot.get_channel = lambda channel_id: fake_channel
        bot.state.llm.complete = _stub_llm(state, "刚才你们说到" + "很长的话题" * 15)
        await bot._flow_tick()
        assert fake_channel.sent == []


class TestFlowBudgetAndSend:
    """B3/B5/B6：共享预算池、同事务冷却、发送与记账。"""

    @pytest.mark.asyncio
    async def test_happy_path_sends_and_records(self, state):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        bot, _bot_user, _channel = await _ready_bot(state)
        fake_channel = FakeChannel(444444)
        bot.get_channel = lambda channel_id: fake_channel
        bot.state.llm.complete = _stub_llm(state, "刚才你们说到摩天轮，周末一起去？")
        await bot._flow_tick()
        assert len(fake_channel.sent) == 1
        row = await state.database.fetchone(
            "SELECT reason FROM proactive_log ORDER BY id DESC LIMIT 1"
        )
        assert str(row["reason"]).startswith("flow:话题：")
        # 开场白照回复路径落库为 assistant 消息，后续回复上下文不丢失
        saved = await state.database.fetchone(
            """SELECT role, content FROM messages
               WHERE guild_id = 'g1' AND role = 'assistant' ORDER BY id DESC LIMIT 1"""
        )
        assert saved is not None and "摩天轮" in saved["content"]
        usage = await state.database.fetchone(
            "SELECT kind FROM usage_metrics WHERE kind = 'flow' ORDER BY id DESC LIMIT 1"
        )
        assert usage is not None
        mood = await state.database.fetchone("SELECT social_budget FROM mood_state WHERE id = 1")
        assert float(mood["social_budget"]) < 0.7

    @pytest.mark.asyncio
    async def test_flow_cooldown_in_transaction(self, state):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        await state.database.execute(
            """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
               VALUES('g1', '444444', 'flow:话题：早前的', ?)""",
            ((utcnow() - timedelta(minutes=60)).isoformat(),),
        )
        bot, _bot_user, _channel = await _ready_bot(state)
        fake_channel = FakeChannel(444444)
        bot.get_channel = lambda channel_id: fake_channel
        bot.state.llm.complete = _stub_llm(state, "刚才你们说到摩天轮，周末一起去？")
        await bot._flow_tick()
        assert fake_channel.sent == []
        count = await state.database.scalar("SELECT COUNT(*) AS n FROM proactive_log")
        assert int(count) == 1

    @pytest.mark.asyncio
    async def test_shared_daily_limit_with_replies(self, state):
        await _enable_flow(state)
        await _seed_idle_channel(state)
        await state.runtime.update({"proactive_daily_limit": 2}, actor="test")
        await state.database.execute(
            """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
               VALUES('g1', '444444', '自然参与', ?)""",
            ((utcnow() - timedelta(hours=2)).isoformat(),),
        )
        bot, _bot_user, _channel = await _ready_bot(state)
        fake_channel = FakeChannel(444444)
        bot.get_channel = lambda channel_id: fake_channel
        bot.state.llm.complete = _stub_llm(state, "刚才你们说到摩天轮，周末一起去？")
        await bot._flow_tick()
        # 回复已占 1 个名额，心流用掉第 2 个：合计不超日限
        assert len(fake_channel.sent) == 1
        count = await state.database.scalar("SELECT COUNT(*) AS n FROM proactive_log")
        assert int(count) == 2
        # 第三个主动行为（无论回复还是心流）都必须被拒
        # 冷却传 0，绕过频道冷却；utc_start 覆盖此前两行，专门验证共享日限池
        denied = await state.proactive._reserve_channel_slot(
            "g1",
            "444444",
            "flow:话题：再试一次",
            now_utc=utcnow(),
            utc_start=(utcnow() - timedelta(hours=12)).isoformat(),
            cooldown_minutes=0,
            daily_limit=2,
        )
        assert denied == "今日额度已用完"
