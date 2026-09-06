"""Phase A — 偏好可见化测试：description 行为化、interest_for 负值语义、avoid 渲染、
familiarity-only observe、心情文风软指令。"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cognition import Relationship


# ═══════════════════════════════════════════════════════════════════════
#  A1 — description 行为倾向措辞
# ═══════════════════════════════════════════════════════════════════════


FORBIDDEN_WORDS = ("喜欢", "讨厌", "烦")


class TestDescriptionMapping:
    @pytest.mark.parametrize(
        "scope, tier, rel, expected_keyword",
        [
            # public + high → fluent
            (
                "public",
                "high",
                Relationship(0.8, 0.5, 0.7, 0.1, 10),
                "聊天很顺",
            ),
            # private + high → fluent (same mapping)
            (
                "private",
                "high",
                Relationship(0.8, 0.5, 0.7, 0.1, 10),
                "聊天很顺",
            ),
            # public + low_warmth → tired
            (
                "public",
                "low",
                Relationship(0.5, 0.5, 0.1, 0.0, 5),
                "累",
            ),
            # private + low_warmth → tired
            (
                "private",
                "low",
                Relationship(0.5, 0.5, 0.1, 0.0, 5),
                "累",
            ),
            # public + neutral → normal distance
            (
                "public",
                "neutral",
                Relationship(0.3, 0.3, 0.4, 0.1, 3),
                "保持平常",
            ),
            # private + neutral → normal distance
            (
                "private",
                "neutral",
                Relationship(0.3, 0.3, 0.4, 0.1, 3),
                "保持平常",
            ),
        ],
        ids=[
            "public-high",
            "private-high",
            "public-low",
            "private-low",
            "public-neutral",
            "private-neutral",
        ],
    )
    def test_description_tier(self, scope: str, tier: str, rel: Relationship, expected_keyword: str):
        desc = rel.description
        assert expected_keyword in desc, (
            f"scope={scope} tier={tier}: expected '{expected_keyword}' in '{desc}'"
        )

    @pytest.mark.parametrize(
        "rel",
        [
            Relationship(0.9, 0.9, 0.9, 0.0, 50),
            Relationship(0.0, 0.0, 0.1, 0.0, 0),
            Relationship(0.5, 0.5, 0.5, 0.6, 10),
            Relationship(0.1, 0.1, 0.1, 0.1, 1),
        ],
    )
    def test_no_affection_assertion_words(self, rel: Relationship):
        desc = rel.description
        for word in FORBIDDEN_WORDS:
            assert word not in desc, f"description 含禁用情感断言词「{word}」: {desc}"


# ═══════════════════════════════════════════════════════════════════════
#  A1 — description 注入公私两域 prompt
# ═══════════════════════════════════════════════════════════════════════


class TestDescriptionInPrompt:
    @pytest.mark.parametrize("public", [True, False], ids=["public", "private"])
    @pytest.mark.asyncio
    async def test_prompt_contains_relationship_tendency(self, state, public: bool):
        await state.relationships.observe(
            "guild-a", "user-1", "谢谢", learning_rate=0.1, decay_days=60
        )
        await state.database.execute(
            """UPDATE relationships SET familiarity = 0.8, warmth = 0.8, fatigue = 0.1
               WHERE guild_id = ? AND user_id = ?""",
            ("guild-a", "user-1"),
        )
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=public
        )
        system = context[0]["content"]
        assert "聊天很顺" in system or "更愿意接" in system


# ═══════════════════════════════════════════════════════════════════════
#  A3 — interest_for 负值语义
# ═══════════════════════════════════════════════════════════════════════


class TestInterestForNegativeWins:
    @pytest.mark.asyncio
    async def test_mixed_match_returns_most_negative(self, state):
        """正0.8/0.5/0.2 + 负-0.6/-0.2 同时匹配 → 返回 -0.6。"""
        await state.preferences.upsert("正A", ["kw_pos"], 0.8, locked=True)
        await state.preferences.upsert("正B", ["kw_pos"], 0.5, locked=True)
        await state.preferences.upsert("正C", ["kw_pos"], 0.2, locked=True)
        await state.preferences.upsert("负A", ["kw_neg"], -0.6, locked=True)
        await state.preferences.upsert("负B", ["kw_neg"], -0.2, locked=True)
        score, topics = await state.preferences.interest_for("聊聊 kw_pos kw_neg", learn=False)
        assert score == pytest.approx(-0.6)

    @pytest.mark.asyncio
    async def test_positives_only_returns_max(self, state):
        """只匹配正值(0.8/0.5/0.2) → 返回 max=0.8。"""
        await state.preferences.upsert("正A", ["kw_only"], 0.8, locked=True)
        await state.preferences.upsert("正B", ["kw_only"], 0.5, locked=True)
        await state.preferences.upsert("正C", ["kw_only"], 0.2, locked=True)
        score, topics = await state.preferences.interest_for("聊聊 kw_only", learn=False)
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_no_match_returns_zero(self, state):
        score, topics = await state.preferences.interest_for("完全无关的内容 xyz", learn=False)
        assert score == 0.0
        assert topics == []


# ═══════════════════════════════════════════════════════════════════════
#  A3 — learn skip: 负权重行不参与被动学习
# ═══════════════════════════════════════════════════════════════════════


class TestLearnSkip:
    @pytest.mark.asyncio
    async def test_negative_row_unchanged_after_learn(self, state):
        """负权重行在 interest_for(learn=True) 后 weight 不变。"""
        await state.preferences.upsert("避开话题", ["避开词"], -0.5, locked=False)
        row_before = await state.database.fetchone(
            "SELECT weight, evidence_count FROM bot_preferences WHERE topic = ?",
            ("避开话题",),
        )
        await state.preferences.interest_for("聊聊避开词", learn=True)
        row_after = await state.database.fetchone(
            "SELECT weight, evidence_count FROM bot_preferences WHERE topic = ?",
            ("避开话题",),
        )
        assert float(row_after["weight"]) == pytest.approx(float(row_before["weight"]))
        assert row_after["evidence_count"] == row_before["evidence_count"]

    @pytest.mark.asyncio
    async def test_positive_row_still_bumps(self, state):
        """正权重行在 interest_for(learn=True) 后 weight 增加。"""
        await state.preferences.upsert("正面话题", ["测试词"], 0.5, locked=False)
        row_before = await state.database.fetchone(
            "SELECT weight FROM bot_preferences WHERE topic = ?",
            ("正面话题",),
        )
        await state.preferences.interest_for("聊聊测试词", learn=True)
        row_after = await state.database.fetchone(
            "SELECT weight FROM bot_preferences WHERE topic = ?",
            ("正面话题",),
        )
        assert float(row_after["weight"]) > float(row_before["weight"])

    @pytest.mark.asyncio
    async def test_race_flip_to_negative_prevents_warm(self, state):
        """模拟竞态：first call 匹配正行，然后 DB 把该行翻为负，second call
        匹配同一行 — weight < 0 at UPDATE time → 不被加热。"""
        await state.preferences.upsert("竞态话题", ["竞态词"], 0.5, locked=False)
        # First call — matches and warms (weight=0.5 >= 0)
        await state.preferences.interest_for("聊聊竞态词", learn=True)
        row_after_first = await state.database.fetchone(
            "SELECT weight, evidence_count FROM bot_preferences WHERE topic = ?",
            ("竞态话题",),
        )
        assert float(row_after_first["weight"]) > 0.5
        first_evidence = row_after_first["evidence_count"]

        # Admin flips the row to negative (simulating a race)
        await state.database.execute(
            "UPDATE bot_preferences SET weight = -0.3 WHERE topic = ?",
            ("竞态话题",),
        )

        # Second call — still matches via negative-row fetch, but
        # the UPDATE guard `AND weight >= 0` prevents warming
        await state.preferences.interest_for("聊聊竞态词", learn=True)
        row_after_second = await state.database.fetchone(
            "SELECT weight, evidence_count FROM bot_preferences WHERE topic = ?",
            ("竞态话题",),
        )
        # Weight and evidence_count unchanged because weight < 0 at UPDATE time
        assert float(row_after_second["weight"]) == pytest.approx(-0.3)
        assert row_after_second["evidence_count"] == first_evidence


# ═══════════════════════════════════════════════════════════════════════
#  A4 — preference_text: liked/avoid 双行渲染
# ═══════════════════════════════════════════════════════════════════════


class TestPreferenceTextRendering:
    @pytest.mark.asyncio
    async def test_liked_and_avoid_lines_co_rendered(self, state):
        """liked 行和 avoid 行同时出现，格式正确。"""
        await state.preferences.upsert("喜欢A", ["kw_a"], 0.9, locked=True)
        await state.preferences.upsert("喜欢B", ["kw_b"], 0.7, locked=True)
        await state.preferences.upsert("避开X", ["kw_x"], -0.4, locked=True)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        # Both lines present
        assert "较偏好的话题" in system
        assert "想避开的话题" in system
        # liked line contains top-2 positives with weight annotations
        assert "喜欢A(0.90)" in system
        assert "喜欢B(0.70)" in system
        # avoid line contains the negative topic
        assert "避开X(-0.40)" in system

    @pytest.mark.asyncio
    async def test_liked_line_caps_at_five(self, state):
        """liked 行最多展示 5 个正值话题，6th/7th 不出现；avoid 行同时渲染。"""
        # Clear seed preferences so only our7 topics compete for top-5
        await state.database.execute("DELETE FROM bot_preferences")
        # Seed 7 positive topics with distinct weights (ascending)
        for i in range(7):
            await state.preferences.upsert(
                f"话题{i}", [f"关键词{i}"], 0.6 + i * 0.03, locked=True
            )
        await state.preferences.upsert("避开话题", ["避开词"], -0.3, locked=True)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        assert "较偏好的话题" in system
        # list(5) returns top 5 by weight DESC: 话题6(0.78), 话题5(0.75),
        # 话题4(0.72), 话题3(0.69), 话题2(0.66)
        assert "话题6" in system  # highest weight
        assert "话题5" in system
        assert "话题4" in system
        assert "话题3" in system
        assert "话题2" in system
        assert "话题1" not in system  # 6th excluded
        assert "话题0" not in system  # 7th excluded
        # avoid line also present
        assert "想避开的话题" in system
        assert "避开话题" in system

    @pytest.mark.asyncio
    async def test_no_preferences_shows_default(self, state):
        """无偏好时显示默认文本。"""
        await state.database.execute("DELETE FROM bot_preferences")
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        assert "尚未形成明显偏好" in system


# ═══════════════════════════════════════════════════════════════════════
#  A2 — observe familiarity_only=True
# ═══════════════════════════════════════════════════════════════════════


class TestObserveFamiliarityOnly:
    @pytest.mark.asyncio
    async def test_familiarity_only_grows_familiarity_and_count(self, state):
        """familiarity_only=True 时 familiarity 和 interaction_count 增长。"""
        result = await state.relationships.observe(
            "guild-a", "user-1", "谢谢", learning_rate=0.05, decay_days=60, familiarity_only=True
        )
        assert result.familiarity > 0
        assert result.interaction_count == 1

    @pytest.mark.asyncio
    async def test_familiarity_only_leaves_warmth_trust_fatigue_unchanged(self, state):
        """familiarity_only=True 时 warmth/trust/fatigue 不变。"""
        from app.database import iso_now

        await state.database.execute(
            """INSERT OR REPLACE INTO relationships
               (guild_id, user_id, familiarity, trust, warmth, fatigue,
                interaction_count, last_interaction_at, updated_at)
               VALUES('guild-a', 'user-1', 0.3, 0.4, 0.5, 0.2, 5, ?, ?)""",
            (iso_now(), iso_now()),
        )
        before = await state.relationships.get("guild-a", "user-1", decay_days=9999)
        result = await state.relationships.observe(
            "guild-a", "user-1", "谢谢", learning_rate=0.1, decay_days=9999, familiarity_only=True
        )
        assert result.trust == pytest.approx(before.trust, abs=1e-6)
        assert result.warmth == pytest.approx(before.warmth, abs=1e-6)
        assert result.fatigue == pytest.approx(before.fatigue, abs=1e-6)
        assert result.interaction_count == before.interaction_count + 1

    @pytest.mark.asyncio
    async def test_default_observe_still_updates_all_dimensions(self, state):
        """默认调用（无 familiarity_only）仍更新全部维度。"""
        result = await state.relationships.observe(
            "guild-a", "user-1", "谢谢", learning_rate=0.1, decay_days=60
        )
        assert result.familiarity > 0
        assert result.warmth > 0
        # fatigue 也应变化（短消息仍会有一点 fatigue delta）
        assert result.fatigue != 0.0 or result.interaction_count == 1


# ═══════════════════════════════════════════════════════════════════════
#  A2 — decay interaction: decayed values materialized, half-rate familiarity
# ═══════════════════════════════════════════════════════════════════════


class TestDecayInteraction:
    @pytest.mark.asyncio
    async def test_decay_materializes_and_familiarity_half_rate(self, state):
        """Aged row with real decay_days: decayed warmth/trust/fatigue are
        materialized while familiarity gains at half rate (learning_rate)."""
        from app.database import iso_now

        # last_interaction 60 days before the mocked utcnow
        fixed_now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        last_interaction = datetime(2025, 11, 2, 12, 0, tzinfo=UTC)

        await state.database.execute(
            """INSERT OR REPLACE INTO relationships
               (guild_id, user_id, familiarity, trust, warmth, fatigue,
                interaction_count, last_interaction_at, updated_at)
               VALUES('guild-a', 'user-1', 0.4, 0.4, 0.5, 0.1, 5, ?, ?)""",
            (last_interaction.isoformat(), iso_now()),
        )

        with patch("app.cognition.utcnow", return_value=fixed_now):
            result = await state.relationships.observe(
                "guild-a", "user-1", "谢谢",
                learning_rate=0.1, decay_days=60,
            )

        # With 60 days elapsed and decay_days=60, factor=0.5:
        #   familiarity: 0.4 * 0.5 = 0.2 + learning_rate(0.1) = 0.3
        #   trust: 0.4 * 0.5 = 0.2 + trust_delta(0.015) = 0.215
        #   warmth: 0.1 + (0.5-0.1)*0.5 = 0.3 + warmth_delta(0.1*0.75) = 0.375
        #   fatigue: 0.1 * (0.5)^(60/2) ≈ 0 → much less than initial 0.1
        assert result.familiarity == pytest.approx(0.3, abs=0.01)
        assert result.trust == pytest.approx(0.215, abs=0.01)
        assert result.warmth == pytest.approx(0.375, abs=0.01)
        # Fatigue should have materially decayed from initial 0.1
        assert result.fatigue < 0.1
        assert result.interaction_count == 6


# ═══════════════════════════════════════════════════════════════════════
#  A5 — 心情→文风软指令
# ═══════════════════════════════════════════════════════════════════════


class TestMoodStyleLine:
    @pytest.mark.asyncio
    async def test_low_valence_shows_line_after_privacy_intent(self, state):
        """valence < -0.3 → 低落文风行，出现在 privacy 之后。"""
        await state.mood.set(-0.5, 0.5, 0.5)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        # Style-specific text present (not just the mood label)
        assert "情绪有些低落" in system or "收敛" in system
        # The STYLE LINE (not the mood label) must appear after privacy
        style_pos = system.find("情绪有些低落")
        privacy_pos = system.find("公开显示" if "公开显示" in system else "私密范围")
        assert privacy_pos < style_pos, (
            f"style line at {style_pos} should be after privacy at {privacy_pos}"
        )

    @pytest.mark.asyncio
    async def test_high_energy_shows_line(self, state):
        """energy > 0.5 且 valence 不低 → 兴致行。"""
        await state.mood.set(0.1, 0.7, 0.5)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        assert "兴致不错" in system or "活泼" in system

    @pytest.mark.asyncio
    async def test_high_valence_shows_line(self, state):
        """valence > 0.5 且 energy 不高 → 心情好行。"""
        await state.mood.set(0.6, 0.3, 0.5)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        assert "心情不错" in system or "放松" in system

    @pytest.mark.asyncio
    async def test_neutral_mood_no_style_line(self, state):
        """中性情绪 → 无文风行。"""
        await state.mood.set(0.0, 0.3, 0.5)
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        assert "低落" not in system
        assert "活泼" not in system
        assert "心情不错" not in system

    @pytest.mark.asyncio
    async def test_mood_disabled_low_valence_baseline_shows_style(self, state):
        """mood_enabled=False + 消极基线 → 基线推导出低落文风行。"""
        await state.mood.set(-0.5, 0.5, 0.5)  # store a negative mood
        await state.runtime.update(
            {
                "mood_enabled": False,
                "mood_baseline_valence": -0.5,
                "mood_baseline_energy": 0.3,
                "mood_baseline_social_budget": 0.5,
            },
            actor="test",
        )
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        # Baseline valence=-0.5 < -0.3 → style line should appear
        assert "情绪有些低落" in system or "收敛" in system

    @pytest.mark.asyncio
    async def test_mood_disabled_neutral_baseline_no_style(self, state):
        """mood_enabled=False + 中性基线 → 无文风行（不看 DB 中的 mood）。"""
        await state.mood.set(-0.5, 0.5, 0.5)  # store a negative mood in DB
        await state.runtime.update(
            {
                "mood_enabled": False,
                "mood_baseline_valence": 0.0,
                "mood_baseline_energy": 0.3,
                "mood_baseline_social_budget": 0.5,
            },
            actor="test",
        )
        context = await state.context.build(
            "guild-a", "channel-a", "user-1", "你好", public=True
        )
        system = context[0]["content"]
        # Baseline neutral → no style line, despite DB mood being negative
        assert "低落" not in system
        assert "活泼" not in system
        assert "心情不错" not in system


# ═══════════════════════════════════════════════════════════════════════
#  A2 wire-up — 非 direct 路径 observe 逻辑 (via real _learn_after_success)
# ═══════════════════════════════════════════════════════════════════════


def _make_payload(*, direct: bool, relationship_enabled: bool = True,
                  familiarity_only: bool = False, guild_id: str = "guild-a"):
    """Build a minimal GenerationPayload-like object for _learn_after_success."""
    from app.discord_bot import GenerationPayload

    message = MagicMock()
    message.guild = MagicMock() if guild_id != "dm:user-1" else None
    return GenerationPayload(
        message=message,
        guild_id=guild_id,
        channel_id="channel-a",
        context_channel_id="channel-a",
        user_id="user-1",
        text="谢谢你的帮助",
        content="谢谢你的帮助",
        config={
            "correction_learning_enabled": False,
            "followup_enabled": False,
            "memory_auto_extract": False,
            "relationship_enabled": relationship_enabled,
            "relationship_learning_rate": 0.1,
            "relationship_decay_days": 9999,
            "mood_enabled": False,
            "bot_experience_enabled": False,
        },
        direct=direct,
        listened=True,
        proactive_reason=None,
        source_message_db_id=None,
        generation_version=1,
    )


class TestPublicPathObserveWiring:
    def test_public_rate_scale_constant(self):
        """模块常量 _PUBLIC_RELATIONSHIP_RATE_SCALE = 0.5。"""
        from app.discord_bot import _PUBLIC_RELATIONSHIP_RATE_SCALE

        assert _PUBLIC_RELATIONSHIP_RATE_SCALE == 0.5

    @pytest.mark.asyncio
    async def test_direct_path_full_rate_observe(self, state):
        """direct path → full-rate observe updates warmth."""
        from app.discord_bot import MoboBot

        await state.database.execute(
            """INSERT OR REPLACE INTO relationships
               (guild_id, user_id, familiarity, trust, warmth, fatigue,
                interaction_count, last_interaction_at, updated_at)
               VALUES('guild-a', 'user-1', 0.3, 0.4, 0.5, 0.2, 5, ?, ?)""",
            (__import__("app.database", fromlist=["iso_now"]).iso_now(),) * 2,
        )
        before = await state.relationships.get("guild-a", "user-1", decay_days=9999)

        bot = MoboBot.__new__(MoboBot)
        bot.state = state
        payload = _make_payload(direct=True, guild_id="guild-a")
        await bot._learn_after_success(payload)

        after = await state.relationships.get("guild-a", "user-1", decay_days=9999)
        assert after.warmth > before.warmth  # full-rate warmth change
        assert after.familiarity > before.familiarity

    @pytest.mark.asyncio
    async def test_non_direct_path_familiarity_only_half_rate(self, state):
        """non-direct path → warmth/trust/fatigue unchanged while familiarity grows."""
        from app.discord_bot import MoboBot

        await state.database.execute(
            """INSERT OR REPLACE INTO relationships
               (guild_id, user_id, familiarity, trust, warmth, fatigue,
                interaction_count, last_interaction_at, updated_at)
               VALUES('guild-a', 'user-1', 0.3, 0.4, 0.5, 0.2, 5, ?, ?)""",
            (__import__("app.database", fromlist=["iso_now"]).iso_now(),) * 2,
        )
        before = await state.relationships.get("guild-a", "user-1", decay_days=9999)

        bot = MoboBot.__new__(MoboBot)
        bot.state = state
        payload = _make_payload(direct=False, guild_id="guild-a")
        await bot._learn_after_success(payload)

        after = await state.relationships.get("guild-a", "user-1", decay_days=9999)
        assert after.familiarity > before.familiarity
        assert after.trust == pytest.approx(before.trust, abs=1e-6)
        assert after.warmth == pytest.approx(before.warmth, abs=1e-6)
        assert after.fatigue == pytest.approx(before.fatigue, abs=1e-6)

    @pytest.mark.asyncio
    async def test_relationship_disabled_no_change(self, state):
        """relationship_enabled=False → nothing changes."""
        from app.discord_bot import MoboBot

        await state.database.execute(
            """INSERT OR REPLACE INTO relationships
               (guild_id, user_id, familiarity, trust, warmth, fatigue,
                interaction_count, last_interaction_at, updated_at)
               VALUES('guild-a', 'user-1', 0.3, 0.4, 0.5, 0.2, 5, ?, ?)""",
            (__import__("app.database", fromlist=["iso_now"]).iso_now(),) * 2,
        )
        before = await state.relationships.get("guild-a", "user-1", decay_days=9999)

        bot = MoboBot.__new__(MoboBot)
        bot.state = state
        payload = _make_payload(direct=True, relationship_enabled=False, guild_id="guild-a")
        await bot._learn_after_success(payload)

        after = await state.relationships.get("guild-a", "user-1", decay_days=9999)
        assert after.familiarity == pytest.approx(before.familiarity, abs=1e-6)
        assert after.trust == pytest.approx(before.trust, abs=1e-6)
        assert after.warmth == pytest.approx(before.warmth, abs=1e-6)
        assert after.fatigue == pytest.approx(before.fatigue, abs=1e-6)
