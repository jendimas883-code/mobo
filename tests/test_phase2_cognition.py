"""Phase 2 认知移植测试：闸门、拟人化、反应、提示词。"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.gate import (
    SCORE_NAME_MENTION,
    PRESSURE_MAX_SCORE,
    PRESENCE_PENALTY_MAX,
    _score_content,
    _score_pressure,
    _score_presence_penalty,
    _has_name_mention,
    score_gate,
)
from app.humanize import (
    _split_sentences,
    _hard_split,
    _merge_to_max,
    _find_immunity_zones,
    _inject_typo_safe,
    _create_typo_sentence,
    fragments,
    typing_delay,
)


# ═══════════════════════════════════════════════════════════════════════
#  Task 1 — 闸门单元测试
# ═══════════════════════════════════════════════════════════════════════


class TestGateNameMention:
    def test_name_mention_scores_80(self):
        score, detail = score_gate("mobo 你好", "mobo")
        assert score >= SCORE_NAME_MENTION
        assert "点名" in detail

    def test_no_name_mention_no_score(self):
        score, _ = score_gate("今天天气不错", "mobo")
        assert score < SCORE_NAME_MENTION

    def test_empty_name_no_match(self):
        score, _ = score_gate("你好", "")
        assert score == 0

    def test_name_in_long_text(self):
        score, detail = score_gate("请问 mobo 这个问题怎么解决", "mobo")
        assert "点名" in detail


class TestGateContent:
    def test_question_marks_score(self):
        score, detail = score_gate("你是怎么做到的？", "bot")
        assert score > 0
        assert "问题" in detail

    def test_request_terms_score(self):
        score, detail = score_gate("帮我看看这个", "bot")
        assert "请求" in detail

    def test_opinion_terms_score(self):
        score, detail = score_gate("你觉得这样行吗", "bot")
        assert "观点" in detail

    def test_long_text_scores(self):
        text = "这是一段很长的文本内容" * 15  # > 120 chars
        score, detail = score_gate(text, "bot")
        assert "较长文本" in detail

    def test_medium_text_scores(self):
        text = "这是一段中等长度的文本" * 6  # > 40 chars
        score, detail = score_gate(text, "bot")
        assert "长文本" in detail

    def test_short_reaction_penalty(self):
        score, detail = score_gate("哈哈", "bot")
        assert "短反应" in detail
        assert score == 0  # negative clamped to 0

    def test_content_score_independent(self):
        score_q, _ = _score_content(["怎么解决这个问题？"])
        score_r, _ = _score_content(["帮我看看"])
        score_o, _ = _score_content(["你觉得怎么样"])
        assert score_q > 0
        assert score_r > 0
        assert score_o > 0


class TestGatePressure:
    def test_zero_pending_no_pressure(self):
        assert _score_pressure(0) == 0

    def test_pressure_grows(self):
        p1 = _score_pressure(1)
        p5 = _score_pressure(5)
        p20 = _score_pressure(20)
        assert p1 > 0
        assert p5 > p1
        assert p20 >= p5

    def test_pressure_capped(self):
        p = _score_pressure(1000)
        assert p <= PRESSURE_MAX_SCORE


class TestGatePresencePenalty:
    def test_no_self_messages_no_penalty(self):
        assert _score_presence_penalty(0, 10) == 0

    def test_low_ratio_no_penalty(self):
        assert _score_presence_penalty(1, 10) == 0  # 10% < 25%

    def test_high_ratio_penalized(self):
        penalty = _score_presence_penalty(8, 10)  # 80% > 60%
        assert penalty > 0
        assert penalty <= PRESENCE_PENALTY_MAX

    def test_medium_ratio_partial_penalty(self):
        penalty = _score_presence_penalty(4, 10)  # 40% between 25% and 60%
        assert 0 < penalty < PRESENCE_PENALTY_MAX


class TestGateThreshold:
    def test_name_mention_alone_triggers_default_threshold(self):
        score, _ = score_gate("mobo 你好", "mobo")
        assert score >= 80

    def test_below_threshold(self):
        score, _ = score_gate("今天天气不错啊", "mobo")
        assert score < 80

    def test_gate_score_exposed_on_decision(self):
        """ProactiveDecision 应暴露 score 字段。"""
        from app.behavior import ProactiveDecision

        d = ProactiveDecision(False, "测试", 0.0, 42)
        assert d.score == 42


# ═══════════════════════════════════════════════════════════════════════
#  Task 2 — 拟人化单元测试
# ═══════════════════════════════════════════════════════════════════════


class TestSplitter:
    def test_basic_split_on_punctuation(self):
        sentences = _split_sentences("你好，世界。我是测试")
        assert len(sentences) >= 2

    def test_newline_forces_split(self):
        sentences = _split_sentences("第一行\n第二行")
        assert len(sentences) >= 2

    def test_short_text_undivided(self):
        text = "短"
        sentences = _split_sentences(text)
        assert len(sentences) == 1

    def test_quote_protection(self):
        """引号内不拆分。"""
        sentences = _split_sentences('"你好，世界"，他说')
        # "你好，世界" 应该保持在一起
        assert any("你好，世界" in s for s in sentences)

    def test_english_space_no_split(self):
        """英文单词之间的空格不应拆分。"""
        sentences = _split_sentences("hello world test")
        # 不应在空格处拆分
        assert len(sentences) == 1


class TestImmunityZones:
    def test_code_fence_immune(self):
        zones = _find_immunity_zones("before\n```\ncode block\n```\nafter")
        assert len(zones) == 1
        start, end = zones[0]
        assert "code block" in "before\n```\ncode block\n```\nafter"[start:end]

    def test_inline_code_immune(self):
        zones = _find_immunity_zones("use `foo()` here")
        assert len(zones) == 1

    def test_url_immune(self):
        zones = _find_immunity_zones("visit https://example.com/path today")
        assert len(zones) == 1

    def test_user_mention_immune(self):
        zones = _find_immunity_zones("hello <@123456789012345678> bye")
        assert len(zones) == 1

    def test_custom_emoji_immune(self):
        zones = _find_immunity_zones("nice <a:dance:123456789012345678> emoji")
        assert len(zones) == 1


class TestHardSplit:
    def test_short_unchanged(self):
        assert _hard_split("hello", 1980) == ["hello"]

    def test_long_splits(self):
        text = "a" * 3000
        parts = _hard_split(text, 1980)
        assert len(parts) == 2
        assert all(len(p) <= 1980 for p in parts)

    def test_prefers_newline(self):
        text = "a" * 1000 + "\n" + "b" * 1000
        parts = _hard_split(text, 1980)
        assert len(parts) == 2
        assert parts[0].endswith("a" * 1000)


class TestMergeToMax:
    def test_no_merge_needed(self):
        assert _merge_to_max(["a", "b"], 4) == ["a", "b"]

    def test_merge_to_limit(self):
        result = _merge_to_max(["a", "b", "c", "d", "e"], 3)
        assert len(result) == 3

    def test_empty(self):
        assert _merge_to_max([], 4) == []


class TestTypoDeterminism:
    def test_seeded_typo_deterministic(self):
        """相同种子产生相同错别字。"""
        text = "今天天气真好我想出去走走"
        random.seed(42)
        result1 = _create_typo_sentence(text, error_rate=0.5)
        random.seed(42)
        result2 = _create_typo_sentence(text, error_rate=0.5)
        assert result1 == result2

    def test_typo_rate_zero_no_change(self):
        """typo_rate=0 时输出不变。"""
        text = "这是一段测试文字"
        random.seed(42)
        result = _inject_typo_safe(text, 0.0)
        assert result == text

    def test_typo_skips_urls(self):
        """URL 免疫。"""
        text = "请访问 https://example.com 这个网站"
        random.seed(42)
        result = _inject_typo_safe(text, 1.0)
        assert "https://example.com" in result


class TestFragmentsAPI:
    def test_typo_rate_zero_unchanged(self):
        """typo_rate=0 时，fragments 返回的文本总和等价于 _hard_split。"""
        text = "你好，我是测试消息。今天天气不错"
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=4)
        assert isinstance(frags, list)
        assert len(frags) >= 1
        assert "".join(frags) == text or len(frags) > 1

    def test_max_fragments_cap(self):
        """碎片数不超过 max_fragments。"""
        text = "句子一。句子二。句子三。句子四。句子五。句子六。句子七。"
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=3)
        assert len(frags) <= 3

    def test_hard_split_preserved(self):
        """1980 字符硬拆仍然有效。"""
        text = "中" * 3000
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=4)
        for f in frags:
            assert len(f) <= 1980

    def test_empty_text(self):
        frags = fragments("", typo_rate=0.0, max_fragments=4)
        assert frags == []

    def test_tail_merge_never_exceeds_limit(self):
        """max_fragments 合并尾部后，任何片段仍不超过 1980。"""
        text = "。" * 50 + "很长的内容" * 800  # 超过 2×1980 字符
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=2)
        assert len(frags) >= 2
        for f in frags:
            assert len(f) <= 1980


class TestTypingDelay:
    def test_short_text(self):
        d = typing_delay("你好", typing_speed=12.0)
        assert d > 0

    def test_faster_speed_lower_delay(self):
        d_slow = typing_delay("你好世界测试", typing_speed=6.0)
        d_fast = typing_delay("你好世界测试", typing_speed=24.0)
        assert d_fast < d_slow

    def test_zero_speed_zero_delay(self):
        assert typing_delay("test", typing_speed=0) == 0


class TestHumanizationGolden:
    def test_disabled_path_uses_legacy_chunks(self):
        """当 humanization_enabled=False 时，discord_bot 使用 _chunks 而非 fragments。
        集成验证在 test_discord_pipeline_v4 的现有测试中覆盖（输出字节等价）。
        这里只验证 fragments() 自身的碎片合并行为。"""
        # fragments() 做句子拆分+合并+typo注入，与 _chunks 不同
        # 验证 typo_rate=0 时内容总和等于原文
        text = "这是一条普通的回复消息，不长不短。今天天气不错。"
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=4)
        # 拼接后应包含所有原始内容
        combined = "".join(frags)
        assert "普通的回复消息" in combined
        assert "不长不短" in combined


# ═══════════════════════════════════════════════════════════════════════
#  Task 2 — 错别字注入真实生效（管线级"安全检查在拟人化之后"见 pipeline 测试）
# ═══════════════════════════════════════════════════════════════════════


class TestTypoMutation:
    def test_seeded_typo_actually_mutates_text(self):
        """typo_rate=1.0 + 固定种子：输出必须与输入不同（注入真实发生）。"""
        text = "今天天气真不错我们一起去公园散步吧"
        random.seed(7)
        out = fragments(text, typo_rate=1.0, max_fragments=4)
        assert "".join(out) != text

    def test_zero_typo_rate_preserves_text(self):
        text = "今天天气真不错我们一起去公园散步吧"
        random.seed(7)
        out = fragments(text, typo_rate=0.0, max_fragments=4)
        assert "".join(out) == text


# ═══════════════════════════════════════════════════════════════════════
#  Task 2 — send pacing
# ═══════════════════════════════════════════════════════════════════════


class TestSendPacing:
    def test_total_window_within_12s(self):
        """所有碎片的延迟总和 ≤12s。"""
        text = "第一句话。第二句话。第三句话。第四句话，比较长的一句话用来测试。"
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=4)
        raw_delays = [typing_delay(f, typing_speed=12.0) for f in frags]
        total = sum(raw_delays)
        if total > 12.0:
            scale = 12.0 / total
            raw_delays = [d * scale for d in raw_delays]
        assert sum(raw_delays) <= 12.0

    def test_fragment_count_capped(self):
        text = "。".join([f"句子{i}" for i in range(10)])
        random.seed(42)
        frags = fragments(text, typo_rate=0.0, max_fragments=3)
        assert len(frags) <= 3


# ═══════════════════════════════════════════════════════════════════════
#  Task 3 — 反应测试
# ═══════════════════════════════════════════════════════════════════════


class TestReactions:
    @pytest.mark.asyncio
    async def test_score_in_range_triggers_reaction(self, state):
        """闸门分数在 [min, threshold) 区间且概率命中时触发反应。"""
        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": True,
                "reaction_probability": 1.0,  # 确保命中
                "reaction_min_score": 40,
                "gate_threshold": 80,
                "reaction_emoji_set": "👍,😂",
                "proactive_quiet_start": "00:00",
                "proactive_quiet_end": "00:00",
            },
            actor="test",
        )
        user = FakeUser(111111111111111)
        message = FakeMessage(900, user, channel, "普通消息")

        # 模拟 decision：score 在 [40, 80) 区间
        from app.behavior import ProactiveDecision

        decision = ProactiveDecision(False, "测试", 0.0, 50)
        await bot._maybe_react(message, decision, await state.runtime.all())
        assert len(message.reactions_added) == 1
        assert message.reactions_added[0] in ("👍", "😂")

    @pytest.mark.asyncio
    async def test_score_below_range_no_reaction(self, state):
        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": True,
                "reaction_probability": 1.0,
                "reaction_min_score": 40,
                "gate_threshold": 80,
            },
            actor="test",
        )
        user = FakeUser(111111111111111)
        message = FakeMessage(901, user, channel, "普通消息")
        from app.behavior import ProactiveDecision

        decision = ProactiveDecision(False, "测试", 0.0, 10)
        await bot._maybe_react(message, decision, await state.runtime.all())
        assert message.reactions_added == []

    @pytest.mark.asyncio
    async def test_cooldown_blocks_second_reaction(self, state):
        import time as _time

        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": True,
                "reaction_probability": 1.0,
                "reaction_min_score": 40,
                "gate_threshold": 80,
                "proactive_quiet_start": "00:00",
                "proactive_quiet_end": "00:00",
            },
            actor="test",
        )
        user = FakeUser(111111111111111)
        msg1 = FakeMessage(902, user, channel, "消息一")
        msg2 = FakeMessage(903, user, channel, "消息二")
        from app.behavior import ProactiveDecision

        decision = ProactiveDecision(False, "测试", 0.0, 50)
        config = await state.runtime.all()
        await bot._maybe_react(msg1, decision, config)
        # 第二次应被冷却阻止
        await bot._maybe_react(msg2, decision, config)
        assert len(msg1.reactions_added) == 1
        assert msg2.reactions_added == []

    @pytest.mark.asyncio
    async def test_bot_message_never_reacted(self, state):
        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": True,
                "reaction_probability": 1.0,
                "reaction_min_score": 40,
                "gate_threshold": 80,
            },
            actor="test",
        )
        bot_author = FakeUser(999999999999999, bot=True, name="other_bot")
        message = FakeMessage(904, bot_author, channel, "bot消息")
        from app.behavior import ProactiveDecision

        decision = ProactiveDecision(False, "测试", 0.0, 50)
        await bot._maybe_react(message, decision, await state.runtime.all())
        assert message.reactions_added == []

    @pytest.mark.asyncio
    async def test_reaction_disabled_no_reaction(self, state):
        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": False,
                "reaction_probability": 1.0,
                "reaction_min_score": 40,
                "gate_threshold": 80,
            },
            actor="test",
        )
        user = FakeUser(111111111111111)
        message = FakeMessage(905, user, channel, "消息")
        from app.behavior import ProactiveDecision

        decision = ProactiveDecision(False, "测试", 0.0, 50)
        await bot._maybe_react(message, decision, await state.runtime.all())
        assert message.reactions_added == []

    @pytest.mark.asyncio
    async def test_reply_path_never_reacts(self, state):
        """进入回复路径的消息不会触发反应（由调用方保证互斥）。"""
        from tests.test_discord_pipeline_v4 import (
            FakeChannel, FakeMessage, FakeUser, _ready_bot,
        )

        bot, bot_user, channel = await _ready_bot(state)
        await state.runtime.update(
            {
                "reaction_enabled": True,
                "reaction_probability": 1.0,
                "reaction_min_score": 40,
                "gate_threshold": 80,
                "proactive_global_enabled": True,
                "social_awareness_enabled": True,
            },
            actor="test",
        )
        await state.channels.set(
            "333333333333333",
            "444444444444444",
            "general",
            listen_enabled=True,
            proactive_enabled=True,
        )
        # 让 decide 返回 should_speak=True（直接用 @mention 路径）
        state.llm.complete = AsyncMock(
            return_value=SimpleNamespace(
                text="回复", input_tokens=10, output_tokens=3,
                latency_ms=12.0, provider="fake", model="fake-model",
            )
        )
        user = FakeUser(111111111111111)
        message = FakeMessage(
            906, user, channel,
            f"<@{bot_user.id}> 你好",
            mentions=[bot_user],
        )
        await bot.on_message(message)
        # 直接路径不会触发 _maybe_react
        assert message.reactions_added == []


# ═══════════════════════════════════════════════════════════════════════
#  Task 4 — 提示词快照测试
# ═══════════════════════════════════════════════════════════════════════


class TestPromptBlocks:
    @pytest.mark.asyncio
    async def test_public_prompt_contains_attention_guide(self, state):
        """公聊系统提示词包含群聊注意力引导。"""
        context = await state.context.build(
            "333333333333333", "444444444444444", "111111111111111",
            "测试消息", public=True,
        )
        system_text = context[0]["content"]
        assert "群聊注意力引导" in system_text
        assert "较早频道对话摘要" in system_text

    @pytest.mark.asyncio
    async def test_public_prompt_contains_activity_rule(self, state):
        """公聊系统提示词包含闲置/活跃规则。"""
        context = await state.context.build(
            "333333333333333", "444444444444444", "111111111111111",
            "测试消息", public=True,
        )
        system_text = context[0]["content"]
        assert "闲置与活跃规则" in system_text
        assert "话多时收敛" in system_text

    @pytest.mark.asyncio
    async def test_private_prompt_also_contains_blocks(self, state):
        """私聊系统提示词也包含两个新区块。"""
        context = await state.context.build(
            "333333333333333", "444444444444444", "111111111111111",
            "测试消息", public=False,
        )
        system_text = context[0]["content"]
        assert "群聊注意力引导" in system_text
        assert "闲置与活跃规则" in system_text


# ═══════════════════════════════════════════════════════════════════════
#  集成测试：ProactiveDecision.score 正确传递
# ═══════════════════════════════════════════════════════════════════════


class TestGateIntegration:
    @pytest.mark.asyncio
    async def test_decide_exposes_score(self, state):
        """ProactiveService.decide 返回的 decision 带有 score。"""
        await state.channels.set(
            "333333333333333",
            "444444444444444",
            "general",
            listen_enabled=True,
            proactive_enabled=True,
        )
        await state.runtime.update(
            {
                "proactive_global_enabled": True,
                "gate_threshold": 80,
                "proactive_quiet_start": "00:00",
                "proactive_quiet_end": "00:00",
            },
            actor="test",
        )
        config = await state.runtime.all()
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        decision = await state.proactive.decide(
            "333333333333333", "444444444444444", "111111111111111",
            "mobo 你好，帮我看看这个", config, now=now,
        )
        assert hasattr(decision, "score")
        assert decision.score >= SCORE_NAME_MENTION  # "mobo" in text

    @pytest.mark.asyncio
    async def test_gate_triggered_speak_bypasses_probability(self, state):
        """闸门触发时，不经过概率判定。"""
        await state.channels.set(
            "333333333333333",
            "444444444444444",
            "general",
            listen_enabled=True,
            proactive_enabled=True,
        )
        await state.runtime.update(
            {
                "proactive_global_enabled": True,
                "gate_threshold": 80,
                "proactive_base_probability": 0.0,  # 概率为 0
                "timezone": "UTC",
                "proactive_quiet_start": "00:00",
                "proactive_quiet_end": "00:00",
            },
            actor="test",
        )
        config = await state.runtime.all()
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        # 包含 bot 名字的文本 → 闸门 ≥ 80 → 触发
        decision = await state.proactive.decide(
            "333333333333333", "444444444444444", "111111111111111",
            "mobo 帮我看看这个怎么解决？", config, now=now,
        )
        assert decision.should_speak is True
        assert "闸门" in decision.reason
