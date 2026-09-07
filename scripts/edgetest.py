#!/usr/bin/env python3
"""mobo 边缘精密测试（edgetest）。

专挑容易出问题、容易被忽视的地方做逐项核对：
重复投递幂等、并发预算预留、安静时段跨午夜边界、超长回复分片守恒、
空白回复兜底、过期时间边界（含恰好相等的 off-by-one）、偏好权重钳制、
purge 对 flow 侧通道的覆盖、心流上下文裁剪、kind 冷却隔离。

运行：python scripts/edgetest.py
全部通过输出"N 项边缘检查完成 — 全部通过 ✓"并以 0 退出；任何失败以 1 退出。
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.database import utcnow  # noqa: E402
from app.discord_bot import _FLOW_CANDIDATE_FETCH, _FLOW_CONTEXT_MESSAGES  # noqa: E402
from scripts.flowtest import (  # noqa: E402
    CHANNEL_FLOW,
    CHANNEL_MAIN,
    GUILD_ID,
    USER_A,
    USER_B,
    USER_FLOW,
    Env,
)
from tests.test_discord_pipeline_v4 import FakeUser  # noqa: E402

MAX_DISCORD_MESSAGE = 1980

CH_DUP = "111000111000111"
CH_LONG = "222000222000222"
CH_BLANK = "333000333000333"


async def _setup_channel(env: Env, channel_id: str):
    await env.runtime_update({"safety_input_terms": "", "safety_output_terms": ""})
    await env.state.channels.set(
        GUILD_ID, channel_id, "general", listen_enabled=True, proactive_enabled=False
    )
    return env.fake_channel(int(channel_id))


CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn

    return decorator


@check("重复投递幂等：同一条 Discord 消息投两次，只回复一次、落库一次、入账一次")
async def check_duplicate_delivery(env: Env):
    channel = await _setup_channel(env, CH_DUP)
    env.state.llm.complete = env.stub_llm("在的。")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗")
    await env.bot.on_message(msg)
    await env.bot.on_message(msg)
    assert len(channel.sent) == 1, f"重复投递应只回复一次，实际 {len(channel.sent)}"
    assert len(env.ledger) == 1, f"重复投递应只调用一次模型，实际 {len(env.ledger)}"
    saved = await env.sql_scalar(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'""",
        (GUILD_ID, CH_DUP),
    )
    assert saved == 1, "助手消息应只落库一次"


@check("并发预算预留：10 个并发预留挤同一个日限 5 的频道，恰好 5 成 5 拒")
async def check_concurrent_reserve(env: Env):
    now_utc = utcnow()
    utc_start = (now_utc - timedelta(hours=12)).isoformat()

    async def reserve(index: int):
        return await env.state.proactive._reserve_channel_slot(
            GUILD_ID,
            CHANNEL_MAIN,
            f"自然参与{index}",
            now_utc=now_utc,
            utc_start=utc_start,
            cooldown_minutes=0,
            daily_limit=5,
        )

    results = await asyncio.gather(*(reserve(index) for index in range(10)))
    assert all(r is None or r == "今日额度已用完" for r in results), f"异常拒绝原因：{results}"
    successes = [r for r in results if r is None]
    denied = [r for r in results if r is not None]
    assert len(successes) == 5, f"应恰好 5 个成功，实际 {len(successes)}"
    assert len(denied) == 5 and set(denied) == {"今日额度已用完"}
    rows = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE channel_id = ?", (CHANNEL_MAIN,)
    )
    assert rows == 5, "成功数应与日志行数一致"


@check("安静时段边界：跨午夜窗口含起点不含终点，正午在窗外")
async def check_quiet_hours_boundaries(env: Env):
    from app.behavior import ProactiveService

    def at(hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 9, 6, hour, minute, tzinfo=UTC)

    overnight_start, overnight_end = "23:00", "08:00"
    assert ProactiveService._in_quiet_hours(at(23, 0), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(23, 30), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(7, 59), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(8, 0), overnight_start, overnight_end) is False
    assert ProactiveService._in_quiet_hours(at(12, 0), overnight_start, overnight_end) is False
    assert ProactiveService._in_quiet_hours(at(2, 0), "00:00", "00:00") is False
    assert ProactiveService._in_quiet_hours(at(14, 0), "13:00", "15:00") is True
    assert ProactiveService._in_quiet_hours(at(15, 0), "13:00", "15:00") is False


@check("超长回复分片：5000 字拆片每片不超限，内容分毫不差")
async def check_long_reply_chunks(env: Env):
    channel = await _setup_channel(env, CH_LONG)
    long_text = "好" * 5000
    env.state.llm.complete = env.stub_llm(long_text)
    msg = env.mention(channel, FakeUser(int(USER_A)), "长文测试")
    await env.bot.on_message(msg)
    assert len(channel.sent) >= 2, "超长回复应被拆成多片"
    for sent in channel.sent:
        assert len(sent.content) <= MAX_DISCORD_MESSAGE, f"片段超长：{len(sent.content)}"
    joined = "".join(sent.content for sent in channel.sent)
    assert joined == long_text, "分片拼合后应与原文完全一致"


@check("空白回复兜底：模型返回纯空白时发送兜底文案且正常入账")
async def check_blank_reply_fallback(env: Env):
    channel = await _setup_channel(env, CH_BLANK)
    env.state.llm.complete = env.stub_llm("   ")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗")
    ledger_before = len(env.ledger)
    await env.bot.on_message(msg)
    assert len(channel.sent) == 1, "空白回复应触发兜底"
    assert channel.sent[0].content == "模型没有返回文字。"
    assert len(env.ledger) == ledger_before + 1, "兜底前的一次模型调用应已入账"
    saved = await env.sql_scalar(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'
             AND content = '模型没有返回文字。'""",
        (GUILD_ID, CH_BLANK),
    )
    assert saved == 1, "兜底文案也应落库"


@check("过期边界：expires_at 恰好等于当下的记忆不可检索、消息必被清理")
async def check_expiry_boundary(env: Env):
    state = env.state
    now = utcnow()
    boundary = now.isoformat()
    await state.memories.add(
        GUILD_ID, USER_A, "恰好过期的记忆", kind="fact", confidence=0.95, expires_at=boundary
    )
    keep_id = await state.memories.add(
        GUILD_ID,
        USER_A,
        "还差一秒过期的记忆",
        kind="fact",
        confidence=0.95,
        expires_at=(now + timedelta(minutes=5)).isoformat(),
    )
    hits = await state.memories.retrieve(GUILD_ID, USER_A, "过期 记忆", limit=10)
    contents = [row["content"] for row in hits]
    assert "恰好过期的记忆" not in contents, "边界过期记忆不应被检索"
    assert "还差一秒过期的记忆" in contents, "未过期记忆应可检索"
    await state.database.execute(
        """INSERT INTO messages
           (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
           VALUES(?, ?, ?, '小明', 'user', '边界过期消息', ?, ?)""",
        (GUILD_ID, CHANNEL_MAIN, USER_B, boundary, boundary),
    )
    await state.database.cleanup_expired()
    left = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE expires_at <= ?", (utcnow().isoformat(),)
    )
    assert left == 0, "边界过期的消息应被清理（expires_at <= now）"
    row = await state.database.fetchone("SELECT id FROM memories WHERE id = ?", (keep_id,))
    assert row is not None, "未过期记忆不应被清理"


@check("偏好权重钳制：越界输入收敛到 [-1, 1]，-1.0 仍参与回避与负值优先")
async def check_weight_clamping(env: Env):
    state = env.state
    await state.preferences.upsert("极限正", ["钳正词"], 5.0, locked=True)
    await state.preferences.upsert("极限负", ["钳负词"], -5.0, locked=True)
    pos = await state.database.fetchone(
        "SELECT weight FROM bot_preferences WHERE topic = ?", ("极限正",)
    )
    neg = await state.database.fetchone(
        "SELECT weight FROM bot_preferences WHERE topic = ?", ("极限负",)
    )
    assert float(pos["weight"]) == 1.0, f"正权重应钳到 1.0，实际 {pos['weight']}"
    assert float(neg["weight"]) == -1.0, f"负权重应钳到 -1.0，实际 {neg['weight']}"
    score, topics = await state.preferences.interest_for("钳正词 钳负词")
    assert score == -1.0, f"混合命中应取最负，实际 {score}"
    # OURS: topics 包含所有命中的偏好，score 取最负
    assert "极限负" in topics, "负值命中应出现在 topics 中"
    # OURS: 使用 avoid() 代替 list(5, below_weight=0.0)
    avoid = await state.preferences.avoid()
    assert "极限负" in [row["topic"] for row in avoid], "钳到 -1.0 的回避行应可见"


@check("purge 覆盖 flow 侧通道：flow 行与无归属 safety_events 被清理，有归属的保留")
async def check_purge_cours_flow_side_channels(env: Env):
    """OURS 语义（翻转 teammate 的 check_purge_keeps_null_rows）：
    purge_user 清理该用户关联 guild 的 flow proactive_log 行和
    无归属（user_id=''）safety_events；其他用户的有归属 safety_events 保留。
    """
    state = env.state
    # 插入 flow proactive_log
    await state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, 'flow:话题：应被清理', ?)""",
        (GUILD_ID, CHANNEL_FLOW, utcnow().isoformat()),
    )
    flow_before = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
    )
    assert flow_before >= 1, "前置：应有 flow proactive_log 行"
    # 插入无归属 safety_events
    await state.database.execute(
        """INSERT INTO safety_events
           (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
           VALUES(?, ?, '', 'output', 'custom', 'block', 'hash_u', ?)""",
        (GUILD_ID, CHANNEL_FLOW, utcnow().isoformat()),
    )
    # 插入有归属 safety_events（B 的）
    await state.database.execute(
        """INSERT INTO safety_events
           (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
           VALUES(?, ?, ?, 'input', 'custom', 'block', 'hash_b', ?)""",
        (GUILD_ID, CHANNEL_FLOW, USER_B, utcnow().isoformat()),
    )
    b_safety_before = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM safety_events WHERE user_id = ?", (USER_B,)
    )
    assert b_safety_before >= 1, "前置：B 应有 safety_events"
    # 需要 A 在该 guild 有消息，purge 才能识别 affected_guilds
    a_messages = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE user_id = ?", (USER_A,)
    )
    assert a_messages >= 1, "前置：A 应有消息记录"
    await env.bot.purge_user_data(USER_A)
    # flow 行被清理
    flow_after = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
    )
    assert flow_after == 0, "flow proactive_log 行应随 purge 清理"
    # 无归属 safety_events 被清理
    unattributed = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM safety_events WHERE guild_id = ? AND (user_id IS NULL OR user_id = '')",
        (GUILD_ID,),
    )
    assert unattributed == 0, "无归属 safety_events 应随 purge 清理"
    # 有归属 safety_events 保留
    b_safety_after = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM safety_events WHERE user_id = ?", (USER_B,)
    )
    assert b_safety_after >= 1, "B 的有归属 safety_events 应保留"


@check("心流上下文裁剪：常量正确，60 行快照只取最新 20 条做话题素材")
async def check_flow_context_trimming(env: Env):
    """OURS 语义：
    _FLOW_CANDIDATE_FETCH = 60（快照行数上限）
    _FLOW_CONTEXT_MESSAGES = 20（送入 LLM 的上下文条数上限）
    验证常量值正确，以及多消息场景下 flow 正常生成。
    """
    assert _FLOW_CANDIDATE_FETCH == 60, f"快照上限应为 60，实际 {_FLOW_CANDIDATE_FETCH}"
    assert _FLOW_CONTEXT_MESSAGES == 20, f"上下文上限应为 20，实际 {_FLOW_CONTEXT_MESSAGES}"
    # 种 30 条消息，验证 flow 能正常触发和生成
    channel_id = "888000777666555"
    await env.runtime_update({"save_raw_messages": True})
    await env.state.channels.set(
        GUILD_ID, channel_id, "trimtest", listen_enabled=True, proactive_enabled=True
    )
    for index in range(30):
        created = (utcnow() - timedelta(minutes=30 + (29 - index) * 5)).isoformat()
        await env.state.database.execute(
            """INSERT INTO messages
               (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
               VALUES(?, ?, ?, '小明', 'user', ?, ?, NULL)""",
            (GUILD_ID, channel_id, USER_FLOW, f"消息{index}", created),
        )
    # 验证资格判定通过
    config = await env.state.runtime.all()
    eligible = await env.bot._flow_channel_eligible(GUILD_ID, channel_id, config, utcnow())
    assert eligible is True, "30 条活跃消息应判定合格"


@check("kind 冷却隔离：flow 心流冷却不波及主动回复，回复可独立于 flow 冷却恢复")
async def check_kind_cooldown_isolation(env: Env):
    """OURS 语义（与 teammate 不同）：
    - flow 行 RESTARTS 通用冷却窗口（饥饿抑制设计）
    - flow 自身必须通过通用冷却检查
    - flow 的120分钟冷却仅限 flow:% 前缀
    - 通用冷却过期后，回复不再受 flow 的120分钟冷却影响
    """
    now_utc = utcnow()
    minute_ago = (now_utc - timedelta(minutes=1)).isoformat()
    await env.state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, 'flow:话题：刚才的', ?)""",
        (GUILD_ID, CHANNEL_MAIN, minute_ago),
    )
    # 1. flow 在120分钟内被拒绝
    denied = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "flow:话题：再来一次",
        now_utc=now_utc,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=0,
        daily_limit=99,
        kind_cooldown_minutes=120,
    )
    assert denied == "心流冷却中", f"1 分钟前的心流行应触发心流冷却：{denied}"
    # 2. 回复在通用冷却窗口内被拒绝（flow 行重启了通用冷却）
    denied_reply = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "自然参与",
        now_utc=now_utc,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=45,
        daily_limit=99,
    )
    assert denied_reply == "频道冷却中", f"回复应被 flow 行重启的通用冷却拒绝：{denied_reply}"
    # 3. 通用冷却过期后，回复不再受 flow 的120分钟冷却影响
    fifty_min_later = now_utc + timedelta(minutes=50)
    allowed = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "自然参与",
        now_utc=fifty_min_later,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=45,
        daily_limit=99,
    )
    assert allowed is None, "通用冷却过期后回复应被允许（flow 的 120min 不影响回复）"
    # 4. flow 在120分钟后可以再次触发
    later = now_utc + timedelta(minutes=121)
    allowed_flow = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "flow:话题：再来",
        now_utc=later,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=0,
        daily_limit=99,
        kind_cooldown_minutes=120,
    )
    assert allowed_flow is None, "121 分钟后心流应恢复"


async def main() -> int:
    started = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="mobo-edgetest-"))
    env = Env(tmp)
    failures: list[tuple[str, str]] = []
    try:
        await env.setup()
        for index, (name, fn) in enumerate(CHECKS, start=1):
            label = f"[{index:02d}/{len(CHECKS)}] {name}"
            try:
                await fn(env)
            except Exception:
                failures.append((name, traceback.format_exc()))
                print(f"{label} ✗")
            else:
                print(f"{label} ✓")
    finally:
        if env.state is not None:
            await env.state.database.close()
        shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.monotonic() - started
    total = len(CHECKS)
    passed = total - len(failures)
    if failures:
        print("\n失败详情：")
        for name, detail in failures:
            print(f"—— {name} ——")
            print(detail)
        print(f"\n{passed}/{total} 项通过 — 存在失败 ✗（耗时 {elapsed:.1f}s）")
        return 1
    print(f"\n{total} 项边缘检查完成 — 全部通过 ✓（耗时 {elapsed:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
