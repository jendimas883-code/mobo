"""Phase 6A 偏好可见化测试：行为化关系描述、公开路径熟悉度、负权重语义、文风映射。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.cognition import Relationship
from app.database import utcnow
from app.discord_bot import GenerationPayload
from tests.test_discord_pipeline_v4 import FakeMessage, FakeUser, _ready_bot


class TestRelationshipDescription:
    """A1：行为倾向映射，禁用情感断言词。"""

    def test_engaged_tendency(self):
        rel = Relationship(0.8, 0.5, 0.6, 0.1, 20)
        assert rel.description == "和这位用户聊天很顺，你会更愿意接他的话头"

    def test_drained_by_fatigue(self):
        rel = Relationship(0.8, 0.5, 0.6, 0.6, 40)
        assert rel.description == "和他的互动让你有点累，倾向简短回应、少接他的梗"

    def test_drained_by_net_hostile_warmth(self):
        rel = Relationship(0.5, 0.2, 0.05, 0.0, 10)
        assert "简短回应" in rel.description

    def test_new_user_neutral(self):
        rel = Relationship(0.0, 0.0, 0.1, 0.0, 0)
        assert rel.description == "保持平常的中性互动距离"

    def test_no_emotion_assertion_words(self):
        rel = Relationship(0.9, 0.9, 0.9, 0.0, 99)
        for word in ("喜欢", "讨厌", "烦"):
            assert word not in rel.description


class TestFamiliarityOnlyObserve:
    """A2 服务面：公开路径只累积 familiarity，敌意词启发式不参与。"""

    @pytest.mark.asyncio
    async def test_hostile_public_content_does_not_touch_warmth(self, state):
        first = await state.relationships.observe(
            "g1", "u1", "普通发言", learning_rate=0.1, decay_days=60, familiarity_only=True
        )
        second = await state.relationships.observe(
            "g1", "u1", "闭嘴，滚", learning_rate=0.1, decay_days=60, familiarity_only=True
        )
        assert second.familiarity == pytest.approx(first.familiarity + 0.1)
        assert second.warmth == first.warmth == pytest.approx(0.1)
        assert second.trust == first.trust == pytest.approx(0.0)
        assert second.fatigue == first.fatigue == pytest.approx(0.0)
        assert second.interaction_count == first.interaction_count + 1

    @pytest.mark.asyncio
    async def test_direct_observe_still_moves_warmth(self, state):
        """回归护栏：direct 路径语义不变，敌意内容仍压 warmth。"""
        await state.relationships.observe("g1", "u2", "闭嘴，滚", learning_rate=0.1, decay_days=60)
        rel = await state.relationships.get("g1", "u2", 60)
        assert rel.warmth == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_decay_interaction_on_public_observe(self, state):
        """过期行先衰减再累加；公开观测同时刷新衰减时钟（有意锁定：互动确实发生）。"""
        await state.relationships.observe("g1", "u3", "初次认识", learning_rate=0.5, decay_days=60)
        stale = (utcnow() - timedelta(days=60)).isoformat()
        await state.database.execute(
            "UPDATE relationships SET last_interaction_at = ? WHERE guild_id = ? AND user_id = ?",
            (stale, "g1", "u3"),
        )
        decayed = await state.relationships.get("g1", "u3", 60)
        assert decayed.familiarity == pytest.approx(0.25, abs=0.01)

        after = await state.relationships.observe(
            "g1", "u3", "公开回复", learning_rate=0.1, decay_days=60, familiarity_only=True
        )
        assert after.familiarity == pytest.approx(decayed.familiarity + 0.1, abs=1e-6)
        assert after.warmth == pytest.approx(decayed.warmth, abs=1e-9)
        assert after.trust == pytest.approx(decayed.trust, abs=1e-9)
        # 衰减时钟已被公开互动刷新：重新读取时 warmth 不再继续衰减
        refreshed = await state.relationships.get("g1", "u3", 60)
        assert refreshed.warmth == pytest.approx(decayed.warmth, abs=1e-6)


class TestPublicPathLearning:
    """A2 调用方：公开回复以减半速率、direct 以全速率积累熟悉度。"""

    @staticmethod
    def _payload(message, config, *, direct: bool) -> GenerationPayload:
        # 私信场景的 guild_id 与主流程一致，使用 dm:<user_id> 占位
        guild_id = f"dm:{message.author.id}" if direct else "333333333333333"
        return GenerationPayload(
            message=message,
            guild_id=guild_id,
            channel_id="444444444444444",
            context_channel_id="444444444444444",
            user_id="555555555555555",
            text="随便聊聊",
            content="随便聊聊",
            config=config,
            direct=direct,
            listened=True,
            proactive_reason=None,
            source_message_db_id=None,
            generation_version=1,
        )

    @pytest.mark.asyncio
    async def test_public_reply_uses_half_rate(self, state):
        await state.runtime.update({"relationship_learning_rate": 0.2}, actor="test")
        bot, _bot_user, channel = await _ready_bot(state)
        message = FakeMessage(700, FakeUser(555555555555555), channel, "随便聊聊")
        await bot._learn_after_success(
            self._payload(message, await state.runtime.all(), direct=False)
        )
        rel = await state.relationships.get("333333333333333", "555555555555555", 60)
        assert rel.familiarity == pytest.approx(0.1)
        assert rel.warmth == pytest.approx(0.1)
        assert rel.interaction_count == 1

    @pytest.mark.asyncio
    async def test_direct_reply_uses_full_rate(self, state):
        await state.runtime.update({"relationship_learning_rate": 0.2}, actor="test")
        bot, _bot_user, channel = await _ready_bot(state)
        message = FakeMessage(701, FakeUser(555555555555555), channel, "随便聊聊", guild_id=None)
        await bot._learn_after_success(
            self._payload(message, await state.runtime.all(), direct=True)
        )
        rel = await state.relationships.get("dm:555555555555555", "555555555555555", 60)
        assert rel.familiarity == pytest.approx(0.2)


class TestNegativeWeightSemantics:
    """A3：负权重聚合、生效与学习三面契约。"""

    @pytest.mark.asyncio
    async def test_negative_dominates_positive_aggregation(self, state):
        await state.preferences.upsert("游戏", ["游戏"], 0.8, locked=True)
        await state.preferences.upsert("争议话题", ["吵架"], -0.6, locked=True)
        score, topics = await state.preferences.interest_for("来聊游戏顺便吵架")
        assert score == pytest.approx(-0.6)
        assert topics == []

    @pytest.mark.asyncio
    async def test_positive_max_without_negatives(self, state):
        # 关键词避开内置种子偏好（如"音乐"），防止意外命中
        await state.preferences.upsert("陶艺", ["陶艺"], 0.4, locked=True)
        await state.preferences.upsert("钓鱼", ["钓鱼"], 0.9, locked=True)
        score, topics = await state.preferences.interest_for("聊聊陶艺和钓鱼")
        assert score == pytest.approx(0.9)
        assert set(topics) == {"陶艺", "钓鱼"}

    @pytest.mark.asyncio
    async def test_no_match_returns_zero(self, state):
        await state.preferences.upsert("游戏", ["游戏"], 0.4, locked=True)
        score, topics = await state.preferences.interest_for("今天天气不错")
        assert score == 0.0
        assert topics == []

    @pytest.mark.asyncio
    async def test_learning_skips_negative_weights(self, state):
        await state.preferences.upsert("争议话题", ["吵架"], -0.5, locked=False)
        await state.preferences.interest_for("又吵起来了", learn=True)
        row = await state.database.fetchone(
            "SELECT * FROM bot_preferences WHERE topic = ?", ("争议话题",)
        )
        assert float(row["weight"]) == pytest.approx(-0.5)
        assert int(row["evidence_count"]) == 0

    @pytest.mark.asyncio
    async def test_learning_still_grows_positive(self, state):
        await state.preferences.upsert("游戏", ["游戏"], 0.4, locked=False)
        await state.preferences.interest_for("聊聊游戏", learn=True)
        row = await state.database.fetchone(
            "SELECT * FROM bot_preferences WHERE topic = ?", ("游戏",)
        )
        assert float(row["weight"]) == pytest.approx(0.405)

    @pytest.mark.asyncio
    async def test_below_weight_query_returns_only_negatives(self, state):
        await state.preferences.upsert("游戏", ["游戏"], 0.8, locked=True)
        await state.preferences.upsert("争议话题", ["吵架"], -0.6, locked=True)
        rows = await state.preferences.list(5, below_weight=0.0)
        assert [row["topic"] for row in rows] == ["争议话题"]


class TestPreferencePrompt:
    """A4：偏好与回避双行注入。"""

    @pytest.mark.asyncio
    async def test_preferred_and_avoid_lines_injected(self, state):
        # 权重给到 1.0，保证排进 weight DESC 前 5（内置种子默认 0.68 等）
        await state.preferences.upsert("陶艺", ["陶艺"], 1.0, locked=True)
        await state.preferences.upsert("争议话题", ["吵架"], -0.6, locked=True)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        system = context[0]["content"]
        assert "你目前较偏好的话题：陶艺(1.00)" in system
        assert "想避开的话题（少接相关话头、降低参与意愿）：争议话题" in system

    @pytest.mark.asyncio
    async def test_no_avoid_line_when_only_positive(self, state):
        await state.preferences.upsert("陶艺", ["陶艺"], 0.8, locked=True)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "想避开的话题" not in context[0]["content"]


class TestMoodStyleHint:
    """A5：心情→文风软指令映射。"""

    @pytest.mark.asyncio
    async def test_low_valence_converges_style(self, state):
        await state.mood.set(-0.8, 0.3, 0.5)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "语气收敛、少用表情" in context[0]["content"]

    @pytest.mark.asyncio
    async def test_high_energy_more_lively(self, state):
        await state.mood.set(0.2, 0.8, 0.5)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "更活泼" in context[0]["content"]

    @pytest.mark.asyncio
    async def test_good_mood_relaxed(self, state):
        await state.mood.set(0.7, 0.3, 0.5)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "语气可以放松" in context[0]["content"]

    @pytest.mark.asyncio
    async def test_neutral_mood_no_style_line(self, state):
        await state.mood.set(0.0, 0.3, 0.5)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "心情对文风的影响" not in context[0]["content"]

    @pytest.mark.asyncio
    async def test_mood_disabled_no_style_line_and_no_query(self, state, monkeypatch):
        await state.runtime.update({"mood_enabled": False}, actor="test")

        def unexpected(*args, **kwargs):
            raise AssertionError("disabled mood was queried")

        monkeypatch.setattr(state.mood, "current", unexpected)
        context = await state.context.build("guild-a", "channel-a", "user-1", "你好")
        assert "心情对文风的影响" not in context[0]["content"]
